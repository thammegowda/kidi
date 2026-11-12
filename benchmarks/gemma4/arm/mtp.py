#!/usr/bin/env python3
"""Compare CPU Gemma MTP off/on with identical greedy outputs and matched settings."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
from statistics import median
import subprocess


ROOT = Path(__file__).resolve().parents[3]
PROMPT = "Explain how binary search works in at least 300 words, including an example and its time complexity."


def measure(arguments, drafts, reference):
    settings = {
        "prompt": arguments.prompt,
        "workload": "short",
        "output_tokens": arguments.output_tokens,
        "context_tokens": arguments.context_tokens,
        "prefill_chunk_tokens": arguments.context_tokens,
        "packed_prefill": True,
        "warmups": arguments.warmups,
        "dense_lm_head": True,
        "kv_cache_precision": arguments.kv_cache_precision,
    }
    if drafts:
        settings.update(assistant_model=str(arguments.assistant), draft_tokens=drafts)
    command = [str(arguments.runner), "gemma", str(arguments.model), str(arguments.threads),
               str(arguments.repeats), json.dumps(settings)]
    process = subprocess.run(command, cwd=ROOT, env={**os.environ, "KIDI_ACCELERATOR": "cpu"},
                             capture_output=True, text=True)
    if process.returncode:
        raise RuntimeError(f"MTP draft length {drafts} failed:\n{process.stderr}\n{process.stdout}")
    records = [json.loads(line) for line in process.stdout.splitlines() if line.strip()]
    generations = [row for row in records if row["stage"] == "generation"]
    runs = [row for row in generations if not row["warmup"]]
    if len(runs) != arguments.repeats or len(generations) != arguments.repeats + arguments.warmups:
        raise ValueError(f"Expected {arguments.repeats} measurements, found {len(runs)}")
    for row in generations:
        if (row["output_tokens"] != arguments.output_tokens or row["observed_output_tokens"] != arguments.output_tokens
                or row["threads"] != arguments.threads or row["speculative"] != bool(drafts)):
            raise ValueError("Measured settings or output counts differ from the requested comparison")
        if drafts and (not row["draft_tokens"] or not row["verification_steps"]):
            raise ValueError("MTP was requested but no drafting or verification was measured")
        if reference is None:
            reference = {"input": row["input_token_ids"], "tokens": row["token_ids"], "text": row["text"]}
        if row["input_token_ids"] != reference["input"]:
            raise ValueError("MTP modes received different prompt token IDs")
        if row["token_ids"] != reference["tokens"]:
            mismatch = next((index for index, pair in enumerate(zip(row["token_ids"], reference["tokens"]))
                             if pair[0] != pair[1]), min(len(reference["tokens"]), len(row["token_ids"])))
            raise ValueError(f"MTP draft length {drafts} changes greedy output at token {mismatch}")
        if row["text"] != reference["text"]:
            raise ValueError("MTP modes decoded different text from identical token IDs")
    proposed = sum(row["draft_tokens"] for row in runs)
    accepted = sum(row["accepted_draft_tokens"] for row in runs)
    summary = {
        "draft_length": drafts,
        "prompt_tokens": runs[0]["prompt_tokens"],
        "output_tokens": arguments.output_tokens,
        "decode_tps": median(row["decode_interval_tps"] for row in runs),
        "decode_tps_min": min(row["decode_interval_tps"] for row in runs),
        "decode_tps_max": max(row["decode_interval_tps"] for row in runs),
        "prefill_tps": median(row["prefill_through_first_token_tps"] for row in runs),
        "ttft_ms": median(row["first_token_seconds"] * 1000 for row in runs),
        "peak_rss_gib": max(row["peak_rss_kib"] for row in runs) / 1048576,
        "draft_acceptance": accepted / proposed if proposed else None,
        "verification_steps": median(row["verification_steps"] for row in runs),
        "draft_ms": median(row["draft_ms"] for row in runs),
        "verify_ms": median(row["verify_ms"] for row in runs),
        "greedy_output_matches_baseline": True,
    }
    return {"settings": settings, "summary": summary, "records": records}, reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", type=Path, default=ROOT / "build-mtp/kidi_android_baseline")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--assistant", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--output-tokens", type=int, default=64)
    parser.add_argument("--context-tokens", type=int, default=1024)
    parser.add_argument("--draft-tokens", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--kv-cache-precision", choices=["auto", "fp32", "int8"], default="int8")
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--output", type=Path, default=ROOT / ".cache/benchmarks/mtp-e2b-int8.json")
    arguments = parser.parse_args()
    if (arguments.threads < 1 or arguments.repeats < 1 or arguments.warmups < 1
            or arguments.output_tokens < 3 or arguments.context_tokens <= arguments.output_tokens
            or not arguments.draft_tokens or any(value < 1 or value > 16 for value in arguments.draft_tokens)):
        parser.error("Counts must be positive, the context must fit the output, and draft lengths must be 1 to 16")
    for name in ("runner", "model", "assistant", "output"):
        setattr(arguments, name, getattr(arguments, name).expanduser().absolute())
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu": (subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
                if platform.system() == "Darwin" else platform.processor()),
        "threads": arguments.threads,
        "warmups": arguments.warmups,
        "repeats": arguments.repeats,
        "target_model": str(arguments.model),
        "assistant_model": str(arguments.assistant),
        "assistant_compute_precision": "fp32",
        "kv_cache_precision": arguments.kv_cache_precision,
        "runner_sha256": hashlib.sha256(arguments.runner.read_bytes()).hexdigest(),
        "runs": [],
    }
    target_config = arguments.model / "config.json"
    if target_config.is_file():
        document = target_config.read_bytes()
        report["target_config_sha256"] = hashlib.sha256(document).hexdigest()
        report["target_weight_bits"] = json.loads(document).get("quantization_config", {}).get("num_bits")
    reference = None
    for drafts in [0, *dict.fromkeys(arguments.draft_tokens)]:
        result, reference = measure(arguments, drafts, reference)
        speed = result["summary"]["decode_tps"]
        baseline = report["runs"][0]["summary"]["decode_tps"] if report["runs"] else speed
        result["summary"]["speedup"] = speed / baseline
        report["runs"].append(result)
        label = f"MTP on, draft={drafts}" if drafts else "MTP off"
        print(f"{label}: {speed:.2f} tokens/s, {speed / baseline:.3f}x baseline", flush=True)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Identical greedy outputs verified. Results: {arguments.output}")


if __name__ == "__main__":
    main()
