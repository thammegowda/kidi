"""Run the official unquantized Gemma checkpoint as a separate BF16 vLLM anchor."""

import argparse
from datetime import UTC, datetime
import importlib.metadata
import json
from pathlib import Path
from statistics import median
import time
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="google/gemma-4-E4B-it")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--memory-utilization", type=float, default=0.88)
    parser.add_argument("--kv-cache-mib", type=int, default=0)
    parser.add_argument("--prefill-chunk", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--short-copies", type=int, default=1)
    parser.add_argument("--long-copies", type=int, default=16)
    parser.add_argument("--eager", action="store_true")
    arguments = parser.parse_args()
    if not 1 <= arguments.repeats <= 20 or not 1 <= arguments.warmups <= 20 or arguments.tokens < 2:
        parser.error("expected 1-20 repeats/warmups and at least two output tokens")
    if not 1 <= arguments.short_copies <= 4096 or not 1 <= arguments.long_copies <= 4096:
        parser.error("prompt copies must be between 1 and 4096")
    import torch
    from vllm import LLM, SamplingParams

    settings = {
        "model": arguments.model,
        "revision": arguments.revision,
        "dtype": "bfloat16",
        "quantization": None,
        "tensor_parallel_size": 1,
        "max_model_len": arguments.context,
        "max_num_batched_tokens": arguments.prefill_chunk or arguments.context,
        "max_num_seqs": 1,
        "gpu_memory_utilization": arguments.memory_utilization,
        "language_model_only": True,
        "enable_prefix_caching": False,
        "cpu_offload_gb": 0,
        "enforce_eager": arguments.eager,
        "seed": 0,
        "disable_log_stats": False,
    }
    if arguments.kv_cache_mib:
        settings["kv_cache_memory_bytes"] = arguments.kv_cache_mib * 1024**2
    report = {
        "version": 2,
        "started": datetime.now(UTC).isoformat(),
        "status": "initializing",
        "settings": settings,
        "versions": {name: importlib.metadata.version(name) for name in ("vllm", "torch", "transformers")},
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": torch.cuda.get_device_capability(0),
        "memory_free_total_bytes": torch.cuda.mem_get_info(0),
        "comparison_note": "Official BF16 weights/activations; compare directly with Kidi's same-checkpoint BF16 path, not native mobile QAT.",
        "records": [],
        "protocol": {"warmups": arguments.warmups, "measured_repeats": arguments.repeats,
                 "short_prompt_copies": arguments.short_copies, "long_prompt_copies": arguments.long_copies,
                 "output_tokens": arguments.tokens, "concurrency": 1, "prefix_caching": False,
                 "timing": "submit after tokenization to first token; first to last token for decode"},
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        arguments.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    save()
    try:
        started = time.perf_counter()
        llm = LLM(**settings)
        report["initialization_seconds"] = time.perf_counter() - started
        sampling = SamplingParams(temperature=0, max_tokens=arguments.tokens, ignore_eos=True, seed=0)
        for workload, copies in (("short", arguments.short_copies), ("long", arguments.long_copies)):
            text = "Binary search repeatedly halves a sorted search range. It is useful for lookup and boundary finding. " * copies
            text += "Explain the main idea and give practical examples."
            prompt = "<bos><|turn>user\n" + text + "<turn|>\n<|turn>model\n"
            prompt_ids = llm.get_tokenizer().encode(prompt, add_special_tokens=False)
            if len(prompt_ids) + arguments.tokens > arguments.context:
                raise RuntimeError("vLLM benchmark prompt and output exceed context")
            for iteration in range(arguments.repeats + arguments.warmups):
                started = time.perf_counter()
                result = llm.generate([{"prompt_token_ids": prompt_ids}], sampling, use_tqdm=False)[0]
                elapsed = time.perf_counter() - started
                output = result.outputs[0]
                if len(output.token_ids) != arguments.tokens:
                    raise RuntimeError("vLLM did not generate the requested output count")
                record = {
                    "workload": workload, "iteration": iteration, "warmup": iteration < arguments.warmups,
                    "input_token_ids": prompt_ids,
                    "wall_seconds": elapsed, "prompt_tokens": len(result.prompt_token_ids),
                    "output_tokens": len(output.token_ids), "token_ids": list(output.token_ids), "text": output.text,
                }
                metrics = getattr(result, "metrics", None)
                if metrics is not None:
                    first = getattr(metrics, "first_token_ts", getattr(metrics, "first_token_time", None))
                    finished = getattr(metrics, "last_token_ts", getattr(metrics, "finished_time", None))
                    scheduled = getattr(metrics, "scheduled_ts", getattr(metrics, "first_scheduled_time", None))
                    if first is not None and finished is not None and scheduled is not None and first > scheduled and finished > first:
                        record.update({"first_token_seconds": first - scheduled,
                                       "prefill_tps": len(result.prompt_token_ids) / (first - scheduled),
                                       "decode_tps": (len(output.token_ids) - 1) / (finished - first)})
                        if first <= started:
                            raise RuntimeError("vLLM timing clock is incompatible with driver submission clock")
                        record.update({"submit_first_token_seconds": first - started,
                                       "decode_seconds": finished - first,
                                       "prefill_through_first_token_tps": len(prompt_ids) / (first - started),
                                       "decode_interval_tps": (len(output.token_ids) - 1) / (finished - first),
                                       "request_seconds": finished - started})
                if "decode_interval_tps" not in record:
                    raise RuntimeError("vLLM did not expose first/last-token timing; comparison cannot be completed")
                report["records"].append(record)
                print(json.dumps(record), flush=True)
                save()
        report["summary"] = {}
        for workload in ("short", "long"):
            rows = [row for row in report["records"] if row["workload"] == workload and not row["warmup"]]
            report["summary"][workload] = {
                key: median(row[key] for row in rows)
                for key in ("wall_seconds", "first_token_seconds", "prefill_tps", "decode_tps",
                            "submit_first_token_seconds", "decode_seconds", "prefill_through_first_token_tps",
                            "decode_interval_tps", "request_seconds") if key in rows[0]
            }
        report["status"] = "completed"
    except Exception as error:
        report.update({"status": "failed", "error_type": type(error).__name__, "error": str(error)})
        traceback.print_exc()
    report["completed"] = datetime.now(UTC).isoformat()
    save()
    print(json.dumps({"status": report["status"], "output": str(arguments.output),
                      "summary": report.get("summary"), "error": report.get("error")}), flush=True)
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())