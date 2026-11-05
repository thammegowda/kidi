"""Run controlled warm-only E2B comparisons, optionally limited to Kidi CUDA BF16 versus vLLM."""

import argparse
from datetime import UTC, datetime
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import median
import subprocess
import sys

from huggingface_hub import snapshot_download


ROOT = Path(__file__).resolve().parents[2]
CHECKPOINTS = {
    "bf16": ("google/gemma-4-E2B-it", "3e22461f65e89153144f8adb70e3b8c2cc9845a7"),
    "qat": ("google/gemma-4-E2B-it-qat-mobile-transformers", "dd693ff40353f057ca5f07e945ad867f4afbf2ec"),
}
METRICS = ("prefill_through_first_token_tps", "decode_interval_tps", "request_seconds")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def summarize(records, warmups, repeats, output_tokens):
    result = {}
    for workload in ("short", "long"):
        all_rows = [row for row in records if row.get("workload") == workload]
        rows = [row for row in all_rows if not row["warmup"]]
        if len(all_rows) != warmups + repeats or len(rows) != repeats:
            raise RuntimeError(f"{workload}: incomplete warm-up/measurement count")
        expected_input = rows[0]["input_token_ids"]
        for row in all_rows:
            if row["output_tokens"] != output_tokens or len(row["token_ids"]) != output_tokens:
                raise RuntimeError(f"{workload}: incorrect output length")
            if row["input_token_ids"] != expected_input or row["prompt_tokens"] != len(expected_input):
                raise RuntimeError(f"{workload}: prompt tokens changed")
            for key in METRICS:
                if not math.isfinite(row[key]) or row[key] <= 0:
                    raise RuntimeError(f"{workload}: invalid or missing {key}")
        if any(row["token_ids"] != rows[0]["token_ids"] for row in rows):
            raise RuntimeError(f"{workload}: generated tokens changed across warm measurements")
        text = rows[0]["text"].lower()
        if "binary" not in text or "search" not in text or len(set(rows[0]["token_ids"])) < 12:
            raise RuntimeError(f"{workload}: output failed the task/repetition sanity check")
        result[workload] = {key: median(row[key] for row in rows) for key in METRICS}
        result[workload].update({"warm_repeats": len(rows), "deterministic_tokens": True,
                                 "prompt_tokens": len(expected_input), "output_tokens": output_tokens,
                                 "first_token_seconds": median(row.get("submit_first_token_seconds", row.get("first_token_seconds")) for row in rows)})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--chunk", type=int, default=512)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--vllm-kv-mib", type=int, default=512)
    parser.add_argument("--bf16-only", action="store_true")
    parser.add_argument("--short-copies", type=int, default=1)
    parser.add_argument("--long-copies", type=int, default=16)
    parser.add_argument("--decode-ahead", type=int, default=16)
    arguments = parser.parse_args()
    if not 1 <= arguments.repeats <= 20 or not 1 <= arguments.warmups <= 20:
        parser.error("expected 1-20 repeats and warmups")
    if not 1 <= arguments.threads <= 8 or arguments.tokens < 2 or arguments.context <= arguments.tokens or arguments.chunk < 1:
        parser.error("invalid threads, context, chunk, or output count")
    if not 1 <= arguments.short_copies <= 4096 or not 1 <= arguments.long_copies <= 4096:
        parser.error("prompt copies must be between 1 and 4096")
    if not 1 <= arguments.decode_ahead <= 64:
        parser.error("decode-ahead must be between 1 and 64")
    arguments.output = arguments.output.resolve()
    if arguments.output.exists():
        parser.error("output already exists; select a new report path")
    work = ROOT / ".cache/gemma-comparison" / arguments.output.stem
    work.mkdir(parents=True, exist_ok=True)
    protocol = {"context_tokens": arguments.context, "prefill_chunk_tokens": arguments.chunk,
                "output_tokens": arguments.tokens, "warmups": arguments.warmups,
                "measured_repeats": arguments.repeats, "threads": arguments.threads,
                "packed_prefill": True, "concurrency": 1, "prefix_caching": False,
                "short_prompt_copies": arguments.short_copies, "long_prompt_copies": arguments.long_copies,
                "decode_ahead_tokens": arguments.decode_ahead,
                "timing": "submit after tokenization to first token; first-to-last output token interval",
                "precision_note": "same official E2B BF16 checkpoint" if arguments.bf16_only else
                    "same E2B architecture; QAT and official BF16 are deliberately separate checkpoint encodings"}
    report = {"version": 1, "status": "running", "started": datetime.now(UTC).isoformat(),
              "protocol": protocol, "checkpoints": {}, "runs": {},
              "hardware": subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv"], text=True)}
    report["source"] = {"native_binary_sha256": digest(ROOT / "build-release/kidi_android_baseline"),
                        "driver_sha256": digest(__file__),
                        "vllm_driver_sha256": digest(ROOT / "benchmarks/cuda/vllm_reference.py"),
                        "kidi_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                        "ynnpack_commit": subprocess.check_output(["git", "-C", "third_party/ynnpack-dev", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}
    arguments.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        arguments.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    save()
    try:
        checkpoints = {"bf16": CHECKPOINTS["bf16"]} if arguments.bf16_only else CHECKPOINTS
        for encoding, (model, revision) in checkpoints.items():
            print(f"Preparing pinned {encoding} checkpoint: {model}", flush=True)
            directory = Path(snapshot_download(model, revision=revision, allow_patterns=[
                "config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
                "generation_config.json", "special_tokens_map.json"]))
            config = json.loads((directory / "config.json").read_text())
            text = config.get("text_config", config)
            architecture = {key: text[key] for key in ("hidden_size", "num_hidden_layers", "intermediate_size",
                "num_attention_heads", "num_key_value_heads", "head_dim", "global_head_dim", "num_kv_shared_layers",
                "vocab_size", "vocab_size_per_layer_input", "layer_types")}
            report["checkpoints"][encoding] = {"model": model, "revision": revision, "directory": str(directory),
                "architecture": architecture, "config_sha256": digest(directory / "config.json"),
                "weights_sha256": digest(directory / "model.safetensors"), "tokenizer_sha256": digest(directory / "tokenizer.json")}
            save()
        if not arguments.bf16_only and report["checkpoints"]["bf16"]["architecture"] != report["checkpoints"]["qat"]["architecture"]:
            raise RuntimeError("BF16/QAT checkpoint architectures differ")
        cells = (("kidi_cpu_qat", "qat", "cpu"), ("kidi_cuda_qat", "qat", "cuda:0"),
                 ("kidi_cuda_bf16", "bf16", "cuda:0"), ("vllm_cuda_bf16", "bf16", "cuda:0"))
        if arguments.bf16_only:
            cells = tuple(cell for cell in cells if cell[1] == "bf16")
        prompt_reference = {}
        for label, encoding, device in cells:
            print(f"Starting {label}", flush=True)
            environment = dict(os.environ, KIDI_ACCELERATOR=device, CUDA_VISIBLE_DEVICES="0", OMP_NUM_THREADS=str(arguments.threads))
            for key in ("KIDI_PROFILE_OPS", "KIDI_PROFILE_OPS_SKIP", "KIDI_REPLAY", "KIDI_REPLAY_CAPACITY", "KIDI_INT8_KV_CACHE", "KIDI_CUDA_PROFILE"):
                environment.pop(key, None)
            directory = report["checkpoints"][encoding]["directory"]
            native_settings = {key: protocol[key] for key in ("context_tokens", "prefill_chunk_tokens", "output_tokens", "warmups", "packed_prefill", "short_prompt_copies", "long_prompt_copies", "decode_ahead_tokens")}
            if label.startswith("vllm"):
                raw_report = work / "vllm.json"
                command = [sys.executable, str(ROOT / "benchmarks/cuda/vllm_reference.py"), "--model", CHECKPOINTS[encoding][0],
                    "--revision", CHECKPOINTS[encoding][1], "--output", str(raw_report), "--context", str(arguments.context),
                    "--prefill-chunk", str(arguments.chunk), "--tokens", str(arguments.tokens), "--warmups", str(arguments.warmups),
                    "--repeats", str(arguments.repeats), "--kv-cache-mib", str(arguments.vllm_kv_mib),
                    "--short-copies", str(arguments.short_copies), "--long-copies", str(arguments.long_copies)]
            else:
                command = [str(ROOT / "build-release/kidi_android_baseline"), "gemma", directory, str(arguments.threads),
                           str(arguments.repeats), json.dumps(native_settings)]
            stdout, stderr = work / f"{label}.stdout", work / f"{label}.stderr"
            run = {"status": "running", "encoding": encoding, "device": device, "command": command,
                   "stdout": str(stdout), "stderr": str(stderr)}
            report["runs"][label] = run
            save()
            with stdout.open("w") as output, stderr.open("w") as errors:
                process = subprocess.run(command, stdout=output, stderr=errors, cwd=ROOT, env=environment)
            if process.returncode:
                raise RuntimeError(f"{label} failed; see {stderr}\n" + stderr.read_text()[-3000:])
            if label.startswith("vllm"):
                raw = json.loads(raw_report.read_text())
                if raw["status"] != "completed":
                    raise RuntimeError(f"{label}: {raw.get('error', 'incomplete vLLM result')}")
                records = raw["records"]
                run["settings"] = raw["settings"]
                run["versions"] = raw["versions"]
                run["initialization_seconds"] = raw["initialization_seconds"]
            else:
                all_records = [json.loads(line) for line in stdout.read_text().splitlines() if line.startswith("{")]
                records = [row for row in all_records if row.get("stage") == "generation"]
                run["settings"] = native_settings
                run["load"] = next(row for row in all_records if row.get("stage") == "load")
                captures = []
                for line in stderr.read_text().splitlines():
                    if line.startswith("kidi_cuda_graph|"):
                        capture = dict(part.split("=", 1) for part in line.split("|")[1:])
                        capture["steady_ns"] = int(capture["steady_ns"])
                        captures.append(capture)
                for row in records:
                    if row["warmup"]:
                        continue
                    if row["preparation_ms"] != 0:
                        raise RuntimeError(f"{label}: operator preparation occurred during measurement")
                    start = row["submitted_steady_ns"]
                    if any(start <= capture["steady_ns"] <= start + row["request_seconds"] * 1e9 for capture in captures):
                        raise RuntimeError(f"{label}: CUDA graph capture occurred during measurement")
                run["cuda_graph"] = {"captures": captures, "no_measured_captures": True}
                if device.startswith("cuda") and arguments.decode_ahead > 1:
                    for row in records:
                        if row["decode_submitted_tokens"] != arguments.tokens - 1 or not 1 <= row["decode_queue_peak"] <= arguments.decode_ahead:
                            raise RuntimeError(f"{label}: invalid asynchronous decode queue or submission count")
                    run["decode_pipeline"] = {"window": arguments.decode_ahead,
                        "peak": max(row["decode_queue_peak"] for row in records), "no_extra_tokens_submitted": True}
            run["summary"] = summarize(records, arguments.warmups, arguments.repeats, arguments.tokens)
            for workload in ("short", "long"):
                inputs = next(row["input_token_ids"] for row in records if row["workload"] == workload)
                if workload in prompt_reference and inputs != prompt_reference[workload]:
                    raise RuntimeError(f"{label}/{workload}: input token IDs differ across runtimes")
                prompt_reference[workload] = inputs
            run.update({"status": "completed", "records": records})
            save()
            print(json.dumps({"profile": label, "summary": run["summary"]}), flush=True)
        report.update({"status": "completed", "completed": datetime.now(UTC).isoformat(), "exact_input_tokens_across_profiles": True})
        save()
    except Exception as error:
        report.update({"status": "failed", "completed": datetime.now(UTC).isoformat(), "error": str(error)})
        save()
        raise
    print("\nWARM MEDIANS: prefill-through-first-token / decode interval (tokens/s)")
    for label, run in report["runs"].items():
        print(label, " | ".join(f"{workload}: {run['summary'][workload][METRICS[0]]:.2f} / {run['summary'][workload][METRICS[1]]:.2f}" for workload in ("short", "long")))


if __name__ == "__main__":
    main()