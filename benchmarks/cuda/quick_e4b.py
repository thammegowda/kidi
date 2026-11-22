"""Quick GPU 7 E4B Q8 speed comparison and translation-quality smoke test."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
from statistics import median
import subprocess
import time


ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / ".cache/benchmarks"
REFERENCE = BENCH / "e4b-q8-gpu7"
MODEL = BENCH / "x64-gemma4-e4b/int8-ptq-from-4ecc2ba"
QUALITY_PROTOCOL = BENCH / "x64-gemma4-int8/eval_protocol.json"
VLLM_PYTHON = ROOT / ".cache/vllm-bench-venv/bin/python"
WORK = BENCH / "quick-e4b"
BASELINE = WORK / "baseline.json"
REFERENCES = WORK / "references.json"


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def fingerprint(value):
    content = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(content.encode()).hexdigest()


def protocol_fingerprint(protocol):
    return fingerprint({key: value for key, value in protocol.items() if key not in ("warmups", "repeats")})


def records_from(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.startswith("{")]


def summarize_rows(records, case, protocol, engine):
    if engine == "kidi":
        rows = [row for row in records if row.get("stage") == "generation"]
    else:
        rows = [row for row in records if row["workload"] == case["name"]]
    warmups, repeats = protocol["warmups"], protocol["repeats"]
    if len(rows) != warmups + repeats:
        raise RuntimeError(f"{engine}/{case['name']}: expected {warmups + repeats} runs, got {len(rows)}")
    for iteration, row in enumerate(rows):
        if row["iteration"] != iteration or row["warmup"] != (iteration < warmups):
            raise RuntimeError(f"{engine}/{case['name']}: invalid warmup sequence")
        if row["input_token_ids"] != case["input_token_ids"] or row["prompt_tokens"] != len(case["input_token_ids"]):
            raise RuntimeError(f"{engine}/{case['name']}: prompt token mismatch")
        if row["output_tokens"] != protocol["output_tokens"] or len(row["token_ids"]) != protocol["output_tokens"]:
            raise RuntimeError(f"{engine}/{case['name']}: output token count mismatch")
        if row.get("reused_prompt_tokens", 0):
            raise RuntimeError(f"{engine}/{case['name']}: unexpected prompt cache reuse")
        if engine == "llamacpp" and row["device"] != "CUDA0":
            raise RuntimeError(f"{engine}/{case['name']}: expected logical CUDA device 0")
        first, decode = row["first_token_seconds"], row["decode_seconds"]
        if not math.isfinite(first) or first <= 0 or not math.isfinite(decode) or decode <= 0:
            raise RuntimeError(f"{engine}/{case['name']}: invalid timing")
        if not math.isclose(row["prefill_through_first_token_tps"], len(case["input_token_ids"]) / first, rel_tol=1e-8):
            raise RuntimeError(f"{engine}/{case['name']}: invalid prefill throughput")
        if not math.isclose(row["decode_interval_tps"], (protocol["output_tokens"] - 1) / decode, rel_tol=1e-8):
            raise RuntimeError(f"{engine}/{case['name']}: invalid decode throughput")
    measured = rows[warmups:]
    result = {
        "prefill_tps": median(row["prefill_through_first_token_tps"] for row in measured),
        "ttft_ms": median(row["first_token_seconds"] * 1000 for row in measured),
        "decode_tps": median(row["decode_interval_tps"] for row in measured),
        "deterministic": None if repeats == 1 else all(row["token_ids"] == measured[0]["token_ids"] for row in measured),
        "token_ids": measured[0]["token_ids"],
    }
    if engine != "llamacpp":
        result["text"] = measured[0]["text"]
        if not result["text"]:
            raise RuntimeError(f"{engine}/{case['name']}: empty output text")
    return result


def execute(command, label, environment):
    output = WORK / f"{label}.jsonl"
    log = WORK / f"{label}.log"
    with output.open("w") as stdout, log.open("w") as stderr:
        try:
            subprocess.run(command, cwd=ROOT, env=environment, stdout=stdout, stderr=stderr, check=True, timeout=3600)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
            raise RuntimeError(f"{label} failed; see {log}") from error
    return records_from(output)


def check_kidi_device(records):
    runtime = next((row for row in records if row.get("stage") == "runtime"), None)
    loaded = next((row for row in records if row.get("stage") == "load"), None)
    if not runtime or runtime.get("device") != "cuda:0" or not loaded or loaded.get("execution") != "cuda":
        raise RuntimeError("Kidi did not execute on logical CUDA device 0")


def run_case(case, protocol, args, environment):
    settings = {
        "context_tokens": protocol["context"], "prefill_chunk_tokens": protocol["chunk"],
        "output_tokens": protocol["output_tokens"], "warmups": args.warmups,
        "packed_prefill": True, "dense_lm_head": True, "kv_cache_precision": args.cache,
        "workload": "short", "short_prompt_copies": 1, "decode_ahead_tokens": 16,
        "prompt": case["prompt"].removeprefix("<bos><|turn>user\n").removesuffix("<turn|>\n<|turn>model\n"),
    }
    command = [str(ROOT / args.binary), "gemma", str(MODEL), str(protocol["threads"]),
               str(args.repeats), json.dumps(settings)]
    rows = execute(command, f"{args.label}-{case['name']}", environment)
    check_kidi_device(rows)
    return summarize_rows(rows, case, {**protocol, "warmups": args.warmups, "repeats": args.repeats}, "kidi")


def clean_translation(texts):
    spec = importlib.util.spec_from_file_location("gemma4_arm_benchmark", ROOT / "benchmarks/gemma4/arm/run.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load the existing translation text cleaner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return [module.clean_translation_text(text) for text in texts]


def load_quality_fixture():
    fixture = json.loads(QUALITY_PROTOCOL.read_text())
    items = fixture["items"]
    if not items or any(len(fixture[key]) != items for key in ("prompts", "references", "input_token_ids_list")):
        raise RuntimeError("Invalid WMT23 translation fixture")
    return fixture


def score_quality(fixture, token_ids, outputs, truncated):
    from sacrebleu.metrics import CHRF

    items = fixture["items"]
    if any(len(values) != items for values in (token_ids, outputs, truncated)):
        raise RuntimeError("WMT23 evaluation returned an unexpected number of outputs")
    hypotheses = clean_translation(outputs)
    if any(not text for text in hypotheses):
        raise RuntimeError("WMT23 evaluation produced an empty translation")
    metric = CHRF()
    score = metric.corpus_score(hypotheses, [fixture["references"]]).score
    return {
        "task": fixture["task"], "languages": fixture["langs"], "items": items,
        "fixture_fingerprint": fingerprint(fixture), "chrf_vs_human": score,
        "chrf_signature": str(metric.get_signature()), "truncated": sum(truncated),
        "token_ids": token_ids, "hypotheses": hypotheses,
    }


def run_quality(protocol, fixture, args, environment):
    items = fixture["items"]
    settings = {
        "context_tokens": fixture["context_tokens"],
        "prefill_chunk_tokens": min(protocol["chunk"], fixture["context_tokens"]),
        "output_tokens": fixture["max_output_tokens"], "warmups": args.warmups,
        "packed_prefill": True, "dense_lm_head": True, "kv_cache_precision": args.cache,
        "workload": "eval", "eval_protocol": str(QUALITY_PROTOCOL), "decode_ahead_tokens": 16,
    }
    command = [str(ROOT / args.binary), "gemma", str(MODEL), str(protocol["threads"]), "1", json.dumps(settings)]
    rows = execute(command, f"{args.label}-wmt23", environment)
    check_kidi_device(rows)
    summaries = [row for row in rows if row.get("stage") == "eval"]
    evaluations = [row for row in rows if row.get("stage") == "eval_item"]
    if len(summaries) != 1 or len(evaluations) != items:
        raise RuntimeError("WMT23 evaluation returned an unexpected number of records")
    summary = summaries[0]
    if summary["items"] != items or len(summary["outputs"]) != items or len(summary["token_ids"]) != items:
        raise RuntimeError("WMT23 evaluation returned an unexpected number of outputs")
    for index, row in enumerate(evaluations):
        if "input_token_ids" not in row or "truncated" not in row:
            raise RuntimeError("Kidi quality runner is out of date; rebuild kidi_android_baseline")
        if (row["id"] != index or row["input_token_ids"] != fixture["input_token_ids_list"][index]
                or row["prompt_tokens"] != len(fixture["input_token_ids_list"][index])
                or row["token_ids"] != summary["token_ids"][index]):
            raise RuntimeError(f"WMT23 item {index}: prompt or output mismatch")
    return score_quality(fixture, summary["token_ids"], summary["outputs"],
                         [row["truncated"] for row in evaluations])


def make_reference_quality_protocol(protocol, fixture):
    return {
        "context": fixture["context_tokens"], "chunk": min(protocol["chunk"], fixture["context_tokens"]),
        "threads": protocol["threads"], "output_tokens": fixture["max_output_tokens"],
        "warmups": 0, "repeats": 1,
        "workloads": [
            {"name": f"wmt{index}", "prompt": prompt, "input_token_ids": ids}
            for index, (prompt, ids) in enumerate(zip(fixture["prompts"], fixture["input_token_ids_list"]))
        ],
    }


def reference_quality_tokens(records, protocol, engine):
    answers, truncated = [], []
    for case in protocol["workloads"]:
        tokens = summarize_rows(records, case, protocol, engine)["token_ids"]
        end = next((index for index, token in enumerate(tokens) if token in (1, 106)), len(tokens))
        answers.append(tokens[:end])
        truncated.append(end == len(tokens))
    return answers, truncated


def stopped_vllm_quality_tokens(records, protocol):
    rows = [row for row in records if row.get("stage") == "eval_item"]
    if len(rows) != len(protocol["workloads"]):
        raise RuntimeError("vLLM quality returned an unexpected number of records")
    answers, truncated = [], []
    for case, row in zip(protocol["workloads"], rows):
        tokens = row["token_ids"]
        if (row["workload"] != case["name"] or row["input_token_ids"] != case["input_token_ids"]
                or len(tokens) > protocol["output_tokens"] or any(token in (1, 106) for token in tokens)
                or not isinstance(row["truncated"], bool)):
            raise RuntimeError(f"vLLM/{case['name']}: invalid quality output")
        if (row["truncated"] and (row["finish_reason"] != "length" or len(tokens) != protocol["output_tokens"])
                or not row["truncated"] and (row["finish_reason"] != "stop" or row["stop_reason"] not in (1, 106))):
            raise RuntimeError(f"vLLM/{case['name']}: invalid quality stopping reason")
        answers.append(tokens)
        truncated.append(row["truncated"])
    return answers, truncated


def decode_reference_tokens(token_ids, environment):
    program = (
        "import json,sys\n"
        "from tokenizers import Tokenizer\n"
        "tokenizer = Tokenizer.from_file(sys.argv[1])\n"
        "print(json.dumps([tokenizer.decode(ids, skip_special_tokens=True) for ids in json.load(sys.stdin)]))\n"
    )
    try:
        process = subprocess.run(
            [str(VLLM_PYTHON), "-c", program, str(REFERENCE / "source/tokenizer.json")],
            input=json.dumps(token_ids), text=True, capture_output=True, env=environment, check=True, timeout=60,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise RuntimeError(f"Reference token decoding failed: {error.stderr}") from error
    return json.loads(process.stdout)


def summarize_reference_quality(fixture, protocol, vllm_records, llama_records, environment):
    quality_protocol = make_reference_quality_protocol(protocol, fixture)
    vllm_ids, vllm_truncated = stopped_vllm_quality_tokens(vllm_records, quality_protocol)
    llama_ids, llama_truncated = reference_quality_tokens(llama_records, quality_protocol, "llamacpp")
    decoded = decode_reference_tokens(vllm_ids + llama_ids, environment)
    count = fixture["items"]
    return {
        "vllm": score_quality(fixture, vllm_ids, decoded[:count], vllm_truncated),
        "llamacpp": score_quality(fixture, llama_ids, decoded[count:], llama_truncated),
    }


def refresh_quality(protocol, fixture, environment):
    quality_protocol = make_reference_quality_protocol(protocol, fixture)
    protocol_path = WORK / "reference-quality-protocol.json"
    save(protocol_path, quality_protocol)
    print("Refreshing vLLM WMT23 quality on GPU 7...", flush=True)
    vllm = execute([str(VLLM_PYTHON), str(ROOT / "benchmarks/cuda/vllm_e4b_quality.py"), str(protocol_path),
                    str(REFERENCE / "vllm-w8a8")], "reference-quality-vllm", environment)
    runtime = next((row for row in vllm if row.get("stage") == "runtime"), None)
    if (not runtime or runtime["settings"]["dtype"] != "bfloat16"
            or runtime["settings"]["quantization"] != "compressed-tensors"
            or runtime["settings"]["enable_prefix_caching"]):
        raise RuntimeError("Unexpected vLLM quality precision or prefix-caching configuration")
    print("Refreshing llama.cpp WMT23 quality on GPU 7...", flush=True)
    llama = execute([str(REFERENCE / "llama_reference"), str(REFERENCE / "gemma-4-E4B-it-Q8_0.gguf"),
                     str(protocol_path)], "reference-quality-llamacpp", environment)
    quality = summarize_reference_quality(fixture, protocol, vllm, llama, environment)
    quality["vllm"].update({"vllm_version": runtime["vllm_version"], "stopping": "explicit_stop_tokens"})
    quality["llamacpp"]["stopping"] = "trim_after_forced_generation"
    return quality


def validate_reference_quality(quality, fixture):
    fixture_hash = fingerprint(fixture)
    for engine, stopping in (("vllm", "explicit_stop_tokens"), ("llamacpp", "trim_after_forced_generation")):
        row = quality.get(engine)
        if not row or row.get("fixture_fingerprint") != fixture_hash or row.get("stopping") != stopping:
            raise RuntimeError(f"{engine} quality reference is incompatible; run with --refresh-quality")


def refresh_references(protocol, args, environment):
    reference_protocol = {**protocol, "warmups": args.warmups, "repeats": args.repeats}
    protocol_path = WORK / "reference-protocol.json"
    vllm_report_path = WORK / "reference-vllm-report.json"
    save(protocol_path, reference_protocol)
    print("Refreshing vLLM reference on GPU 7...", flush=True)
    execute([str(VLLM_PYTHON), str(REFERENCE / "vllm_reference.py"),
             str(protocol_path), str(vllm_report_path)], "reference-vllm", environment)
    vllm = json.loads(vllm_report_path.read_text())
    if (vllm["settings"]["dtype"] != "bfloat16" or vllm["settings"]["quantization"] != "compressed-tensors"
            or vllm["settings"]["enable_prefix_caching"]):
        raise RuntimeError("Unexpected vLLM precision or prefix-caching configuration")
    print("Refreshing llama.cpp reference on GPU 7...", flush=True)
    llama = execute([str(REFERENCE / "llama_reference"), str(REFERENCE / "gemma-4-E4B-it-Q8_0.gguf"),
                     str(protocol_path)], "reference-llamacpp", environment)
    result = {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "protocol_fingerprint": protocol_fingerprint(protocol),
        "manifest_sha256": hashlib.sha256((REFERENCE / "model-manifest.json").read_bytes()).hexdigest(),
        "warmups": args.warmups, "repeats": args.repeats, "vllm_versions": vllm["versions"],
        "cases": {
            case["name"]: {
                "vllm": summarize_rows(vllm["records"], case, reference_protocol, "vllm"),
                "llamacpp": summarize_rows(llama, case, reference_protocol, "llamacpp"),
            }
            for case in protocol["workloads"]
        },
    }
    return result


def first_divergence(left, right):
    return next((index for index, (a, b) in enumerate(zip(left, right)) if a != b), min(len(left), len(right)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="current")
    parser.add_argument("--binary", default="build-release-cuda/kidi_android_baseline")
    parser.add_argument("--cache", choices=["fp32", "bf16", "auto"], default="bf16")
    parser.add_argument("--prompts", default="p128,p512,p2048")
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    refresh = parser.add_mutually_exclusive_group()
    refresh.add_argument("--refresh-references", action="store_true",
                         help="remeasure vLLM and llama.cpp speed and quality")
    refresh.add_argument("--refresh-quality", action="store_true",
                         help="remeasure vLLM and llama.cpp quality, retaining cached speed")
    parser.add_argument("--save-baseline", action="store_true", help="replace the saved Kidi speed and quality baseline")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.label):
        parser.error("label must contain only letters, numbers, '.', '_' or '-'")
    if args.label in {"references", "reference-protocol", "reference-vllm-report",
                      "reference-quality-protocol", "reference-quality-vllm-report"} or (
            args.label == "baseline" and not args.save_baseline):
        parser.error("label conflicts with a saved reference or baseline")
    if not 1 <= args.warmups <= 20 or not 1 <= args.repeats <= 20:
        parser.error("warmups and repeats must be between 1 and 20")
    if args.save_baseline and args.repeats < 2:
        parser.error("saving a baseline requires at least two measured runs")

    WORK.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((REFERENCE / "protocol.json").read_text())
    fixture = load_quality_fixture()
    wanted = {name.strip() for name in args.prompts.split(",") if name.strip()}
    available = {case["name"] for case in protocol["workloads"]}
    if not wanted or not wanted <= available:
        parser.error(f"prompts must be selected from {', '.join(sorted(available))}")
    if args.save_baseline and wanted != available:
        parser.error("saving a baseline requires all prompts")

    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="7", KIDI_ACCELERATOR="cuda:0",
                       OMP_NUM_THREADS=str(protocol["threads"]), HF_HUB_OFFLINE="1")
    for key in ("KIDI_CUDA_PROFILE", "KIDI_PROFILE_OPS", "KIDI_PROFILE_OPS_SKIP", "KIDI_REPLAY",
                "KIDI_REPLAY_CAPACITY", "KIDI_CUDA_GRAPH_LOG", "CUDA_LAUNCH_BLOCKING"):
        environment.pop(key, None)
    started = time.monotonic()
    if args.refresh_references:
        references = refresh_references(protocol, args, environment)
    elif REFERENCES.exists():
        references = json.loads(REFERENCES.read_text())
    else:
        raise RuntimeError(f"No vLLM/llama.cpp references at {REFERENCES}; run with --refresh-references")
    manifest_hash = hashlib.sha256((REFERENCE / "model-manifest.json").read_bytes()).hexdigest()
    if (references["protocol_fingerprint"] != protocol_fingerprint(protocol)
            or references["manifest_sha256"] != manifest_hash):
        raise RuntimeError("Reference protocol or model manifest changed; run with --refresh-references")
    if not wanted <= references["cases"].keys():
        raise RuntimeError("Reference measurements are missing requested prompts; run with --refresh-references")
    if args.refresh_references or args.refresh_quality:
        references["quality"] = refresh_quality(protocol, fixture, environment)
        references["quality_recorded_at"] = datetime.now(timezone.utc).isoformat()
        save(REFERENCES, references)
    if "quality" not in references or "quality_recorded_at" not in references:
        raise RuntimeError("Missing vLLM/llama.cpp quality results; run with --refresh-quality")
    validate_reference_quality(references["quality"], fixture)

    baseline = json.loads(BASELINE.read_text()) if BASELINE.exists() else {}
    metadata = {"cache": args.cache, "protocol_fingerprint": protocol_fingerprint(protocol),
                "manifest_sha256": manifest_hash}
    if baseline and "metadata" in baseline and baseline["metadata"] != metadata:
        print("Saved Kidi baseline uses different settings; output comparison is unavailable.")
        baseline = {}
    elif baseline and "metadata" not in baseline:
        print("Saved Kidi baseline predates settings metadata; output comparison assumes the original BF16 protocol.")
    baseline_cases = baseline.get("cases", baseline)

    results = {}
    for case in protocol["workloads"]:
        if case["name"] in wanted:
            results[case["name"]] = run_case(case, protocol, args, environment)
    quality = run_quality(protocol, fixture, args, environment)
    for engine in ("vllm", "llamacpp"):
        if references["quality"][engine]["chrf_signature"] != quality["chrf_signature"]:
            raise RuntimeError("chrF settings or version changed; run with --refresh-quality")

    print("\nSpeed: prefill = prompt tokens / TTFT; decode = 127 token intervals.")
    print(f"Kidi: live, {args.warmups} warmup(s), {args.repeats} measured; "
          f"vLLM/llama.cpp: recorded {references['recorded_at']}, "
          f"{references['warmups']} warmup(s), {references['repeats']} measured.")
    print(f"{'prompt':>6}  {'engine':<10} {'prefill tok/s':>14} {'TTFT ms':>9} "
          f"{'decode tok/s':>13} {'chrF/ref':>9} {'det':>4}")
    for name, result in results.items():
        for label, row, quality_row in (("Kidi", result, quality),
                                        ("vLLM", references["cases"][name]["vllm"], references["quality"]["vllm"]),
                                        ("llama.cpp", references["cases"][name]["llamacpp"],
                                         references["quality"]["llamacpp"])):
            deterministic = "n/a" if row["deterministic"] is None else "yes" if row["deterministic"] else "NO"
            print(f"{name[1:]:>6}  {label:<10} {row['prefill_tps']:>14.1f} {row['ttft_ms']:>9.2f} "
                  f"{row['decode_tps']:>13.1f} {quality_row['chrf_vs_human']:>9.2f} {deterministic:>4}")

    from sacrebleu.metrics import CHRF

    print("\nKidi output regression vs saved Kidi baseline (not an accuracy score):")
    for name, result in results.items():
        pinned = baseline_cases.get(name)
        if not pinned:
            print(f"  {name}: no baseline")
            continue
        prefix = first_divergence(result["token_ids"], pinned["token_ids"])
        exact = result["token_ids"] == pinned["token_ids"]
        chrf = CHRF().sentence_score(result["text"], [pinned["text"]]).score
        print(f"  {name}: {'MATCH' if exact else 'CHANGED'}, common token prefix "
              f"{prefix}/{len(result['token_ids'])}, text chrF {chrf:.1f}")
    print(f"\nchrF/ref: same {quality['items']} WMT23 {','.join(quality['languages'])}->en human references "
          f"for every engine; scores repeat across speed-prompt rows.")
    print(f"vLLM/llama.cpp quality recorded {references['quality_recorded_at']}.")
    for label, quality_row in (("Kidi", quality), ("vLLM", references["quality"]["vllm"]),
                               ("llama.cpp", references["quality"]["llamacpp"])):
        if quality_row["truncated"]:
            print(f"  WARNING: {label} hit the output limit on {quality_row['truncated']} translation(s).")
    pinned_quality = baseline.get("quality")
    if pinned_quality and pinned_quality["fixture_fingerprint"] == quality["fixture_fingerprint"]:
        if len(pinned_quality["token_ids"]) != quality["items"]:
            raise RuntimeError("Saved WMT23 baseline has an unexpected number of outputs")
        matches = sum(a == b for a, b in zip(quality["token_ids"], pinned_quality["token_ids"]))
        print(f"  Exact Kidi output match vs baseline: {matches}/{quality['items']} sentences.")
    else:
        print("  No matching saved Kidi translation baseline; use --save-baseline to record one.")
    print(f"Elapsed {time.monotonic() - started:.0f}s; raw output in {WORK}")

    report = {"metadata": metadata, "cases": results, "quality": quality,
              "references": references, "warmups": args.warmups, "repeats": args.repeats}
    if args.save_baseline:
        if any(result["deterministic"] is not True for result in results.values()):
            raise RuntimeError("Refusing to save a nondeterministic Kidi baseline")
    if args.label != "baseline":
        save(WORK / f"{args.label}.json", report)
    if args.save_baseline:
        save(BASELINE, {"metadata": metadata, "cases": results, "quality": quality})
        print(f"Saved Kidi baseline to {BASELINE}")


if __name__ == "__main__":
    main()
