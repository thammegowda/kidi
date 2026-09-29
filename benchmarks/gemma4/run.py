#!/usr/bin/env python3
"""Sequential batch-one Gemma benchmarks with explicit runtime and precision labels."""

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

from tokenizers import Tokenizer


def round_metrics(value):
    if isinstance(value, float):
        return round(value, 5)
    if isinstance(value, dict):
        return {key: round_metrics(item) for key, item in value.items()}
    if isinstance(value, list):
        return [round_metrics(item) for item in value]
    return value


def memory_snapshot():
    if sys.platform != "darwin":
        return None
    report = subprocess.check_output(["vm_stat"], text=True)
    counters = {}
    for line in report.splitlines():
        name, _, value = line.partition(":")
        if name in ("Swapins", "Swapouts"):
            counters[name.lower()] = int(value.strip().rstrip("."))
    return {"timestamp_ns": time.time_ns(), **counters,
            "page_bytes": int(subprocess.check_output(["sysctl", "-n", "hw.pagesize"], text=True)),
            "swap_usage": subprocess.check_output(["sysctl", "vm.swapusage"], text=True).strip()}


def make_prompt(tokenizer, count):
    prefix = "Write a detailed explanation in at least four hundred words of this passage:\n"
    passage = "Sunlight contains many colors. Air scatters blue light more strongly than red light. "
    wrap = lambda text: "<bos><|turn>user\n" + text + "<turn|>\n<|turn>model\n"
    encode = lambda text: tokenizer.encode(text, add_special_tokens=False).ids
    while len(encode(wrap(prefix + passage))) <= count:
        prefix += passage
    remaining = count - len(encode(wrap(prefix)))
    if remaining < 0:
        raise ValueError("Prefill length is too small for the benchmark prompt")
    prompt = wrap(prefix + " a" * remaining)
    tokens = encode(prompt)
    if len(tokens) != count:
        raise ValueError(f"Prompt round-trip has {len(tokens)} tokens, expected {count}")
    return prompt, tokens


def run_kidi(args, prompt, tokenizer):
    command = [str(args.binary), "generate", "--model", str(args.model),
               "--backend", "ynnpack" if args.backend == "cpu" else "mps", "--threads", str(args.threads),
               "--context-size", str(args.context), "--max-new-tokens", str(args.decode + 1),
               "--prefill-chunk-size", str(args.chunk), "--ignore-eos", "--profile",
               "--max-active", "1", "--queue-size", "1", "--cache-tokens", str(args.context)]
    if args.weight_bits:
        command += ["--weight-bits", str(args.weight_bits), "--group-size", str(args.group_size)]
    if args.packed_prefill:
        command += ["--packed-prefill"]
    if args.full_attention_cache:
        command += ["--full-attention-cache"]
    content = prompt.removeprefix("<bos><|turn>user\n").removesuffix("<turn|>\n<|turn>model\n")
    request = json.dumps({"messages": [{"role": "user", "content": content}]}) + "\n"
    started = time.perf_counter()
    process = subprocess.run(command, input=request * (args.warmups + args.runs), text=True, capture_output=True)
    if process.returncode:
        raise RuntimeError(f"Kidi failed ({process.returncode}): {process.stderr}")
    wall = time.perf_counter() - started
    summary = next(line for line in process.stderr.splitlines() if line.startswith("kidi_serving|"))
    fields = dict(part.split("=", 1) for part in summary.split("|")[1:])
    if bool(int(fields["packed_prefill"])) != args.packed_prefill:
        raise ValueError("Executed prefill policy does not match requested policy")
    all_records = [json.loads(line) for line in process.stdout.splitlines()]
    if len(all_records) != args.warmups + args.runs:
        raise ValueError("Missing JSONL responses")
    records = all_records[args.warmups:]
    for index, record in enumerate(all_records, start=-args.warmups):
        record.update(run=index, native_qat=int(fields["native_qat"]), packed_prefill=args.packed_prefill,
                      load_ns=int(fields["load_ns"]), text=record["message"]["content"])
        if record["prompt_tokens"] != args.prefill or record["decode_tokens"] != args.decode:
            raise ValueError(f"Kidi token count mismatch: {record}")
        record["decode_tokens_per_second"] = args.decode * 1e9 / record["decode_ns"] if args.decode else 0
        record["prefill_tokens_per_second"] = args.prefill * 1e9 / record["prefill_ns"]
    if len(records) != args.runs:
        raise ValueError(f"Expected {args.runs} measurements, got {len(records)}")
    precision = "original BF16"
    model_config = json.loads((args.model / "config.json").read_text())
    native_qat = model_config.get("quantization_config", {}).get("quant_method") == "gemma"
    if any(bool(record.get("native_qat", 0)) != native_qat for record in records):
        raise ValueError("Executed checkpoint policy does not match model configuration")
    if native_qat:
        precision = "native mobile QAT mixed Q2/Q4/Q8; trained activation/cache scales"
    if args.weight_bits == 4:
        precision = f"MLP Q4 group {args.group_size} / other projections per-channel Q8 PTQ"
    elif args.weight_bits == 8:
        precision = "per-channel Q8 PTQ"
    with args.binary.open("rb") as binary:
        binary_sha256 = hashlib.file_digest(binary, "sha256").hexdigest()
        return {"records": records, "warmup_records": all_records[:args.warmups],
            "command": command, "process_wall_seconds": wall,
            "binary_sha256": binary_sha256,
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
            "stderr": process.stderr, "weight_precision": precision,
            "execution": "ordered JSONL, serial admission, checkpoint chat template",
            "prefill_precision": ("native QAT packed" if args.backend == "cpu" or args.packed_prefill else "native QAT cached FP16 matrices")
            if native_qat else "packed" if args.weight_bits and args.packed_prefill else "original floating (multi-row)"}


def run_llama(args):
    phases = {}
    commands = []
    logs = []
    started = time.perf_counter()
    for phase in ("prefill", "decode"):
        if phase == "decode" and not args.decode:
            continue
        command = [str(args.llama_binary), "-m", str(args.gguf), "-o", "json",
                   "-r", str(args.warmups + args.runs), "--no-warmup",
                   "-t", str(args.threads), "-b", str(args.chunk), "-ub", str(args.chunk),
                   "-ngl", "0" if args.backend == "cpu" else "99", "-fa", "on",
                   "-p", str(args.prefill if phase == "prefill" else 0),
                   "-n", str(args.decode if phase == "decode" else 0),
                   "-d", str(args.prefill if phase == "decode" else 0)]
        if args.backend == "cpu":
            command += ["-dev", "none", "-nkvo", "1", "-nopo", "1"]
        process = subprocess.run(command, text=True, capture_output=True, timeout=600)
        if process.returncode:
            raise RuntimeError(f"llama-bench failed ({process.returncode}): {process.stderr}")
        results = json.loads(process.stdout)
        if len(results) != 1:
            raise ValueError(f"Expected one llama-bench {phase} result, got {len(results)}")
        result = results[0]
        expected = (args.prefill, 0, 0) if phase == "prefill" else (0, args.decode, args.prefill)
        actual = tuple(result[key] for key in ("n_prompt", "n_gen", "n_depth"))
        if actual != expected or len(result["samples_ns"]) != args.warmups + args.runs:
            raise ValueError(f"llama-bench token/repetition count mismatch: {result}")
        phases[phase] = result
        commands.append(command)
        logs.append(process.stderr)
    records = []
    for index in range(args.warmups + args.runs):
        prefill_ns = phases["prefill"]["samples_ns"][index]
        decode_ns = phases["decode"]["samples_ns"][index] if args.decode else 0
        records.append({"run": index - args.warmups, "prefill_ns": prefill_ns, "decode_ns": decode_ns,
                        "prefill_tokens_per_second": args.prefill * 1e9 / prefill_ns,
                        "decode_tokens_per_second": args.decode * 1e9 / decode_ns if args.decode else 0})
    with args.llama_binary.open("rb") as binary:
        binary_sha256 = hashlib.file_digest(binary, "sha256").hexdigest()
    return {"records": records[args.warmups:], "warmup_records": records[:args.warmups],
            "commands": commands, "process_wall_seconds": time.perf_counter() - started,
            "binary_sha256": binary_sha256, "gguf": str(args.gguf), "gguf_bytes": args.gguf.stat().st_size,
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
            "stderr": logs, "raw_phases": phases, "weight_precision": phases["prefill"]["model_type"],
            "execution": "llama-bench synthetic tokens; separate prefill and context-conditioned decode; no sampling",
            "warmup_policy": "discard initial full phase repetitions; built-in short warmup disabled",
            "context_policy": "llama-bench allocates from prompt + generation + depth, rounded by runtime"}


def run_litert(args, prompt, tokens):
    import litert_lm

    backend = litert_lm.Backend.CPU(thread_count=args.threads) if args.backend == "cpu" else litert_lm.Backend.GPU()
    started = time.perf_counter()
    engine = litert_lm.Engine(str(args.reference_model), backend=backend, max_num_tokens=args.context,
                             cache_dir=str(args.cache), enable_benchmark=True,
                             enable_speculative_decoding=args.runtime == "litert-mtp",
                             use_ringbuffers_local_attention=True if args.backend == "gpu" else None)
    load = time.perf_counter() - started
    records = []
    with engine:
        native_prompt = prompt.removeprefix("<bos>")
        actual = [engine.bos_token_id, *engine.tokenize(native_prompt)]
        if actual != tokens:
            raise ValueError(f"LiteRT tokenizer differs from original: {actual[:20]} != {tokens[:20]}")
        for iteration in range(-args.warmups, args.runs):
            request_started = time.perf_counter_ns()
            with engine.create_session(apply_prompt_template=False,
                                       sampler_config=litert_lm.SamplerConfig(top_k=1, temperature=0.0, seed=0),
                                       max_output_tokens=args.decode + 1) as session:
                started = time.perf_counter()
                session.run_prefill([native_prompt])
                prefill_wall = time.perf_counter() - started
                started = time.perf_counter()
                response = session.run_decode()
                decode_wall = time.perf_counter() - started
                first_output_ns = time.perf_counter_ns() - request_started if not args.decode else None
                info = session.get_benchmark_info()
                if iteration < 0:
                    continue
                if info.last_prefill_token_count != args.prefill:
                    raise ValueError(f"LiteRT prefill count mismatch: {info}")
                records.append({"run": iteration, **asdict(info), "load_seconds": load,
                                "ttft_ns": first_output_ns,
                                "prefill_wall_seconds": prefill_wall, "decode_wall_seconds": decode_wall,
                                "text": response.texts, "decode_tokens_per_second": info.last_decode_tokens_per_second,
                                "prefill_tokens_per_second": info.last_prefill_tokens_per_second})
    return {"records": records, "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "weight_precision": "released mixed 2/4/8-bit bundle", "activation_precision": "runtime default",
            "ring_buffers": args.backend == "gpu", "gpu_steps_per_sync": "runtime default"}


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", choices=["kidi", "llama", "litert", "litert-mtp"], required=True)
    parser.add_argument("--backend", choices=["cpu", "gpu"], required=True)
    parser.add_argument("--model", type=Path, default=root.parent / "models/gemma-4-E2B-it")
    parser.add_argument("--reference-model", type=Path,
                        default=root.parent / "models/gemma-4-E2B-it-litert-lm/gemma-4-E2B-it.litertlm")
    parser.add_argument("--binary", type=Path, default=root / "build-release/kidi")
    parser.add_argument("--llama-binary", type=Path,
                        default=root / ".cache/llama.cpp-bench/build/bin/llama-bench")
    parser.add_argument("--gguf", type=Path, help="GGUF weights for --runtime llama")
    parser.add_argument("--cache", type=Path, default=root / ".cache/gemma-litert")
    parser.add_argument("--prefill", type=int, default=128)
    parser.add_argument("--decode", type=int, default=64, help="recurrent tokens after the first; 0 measures first-output latency")
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--chunk", type=int, default=128)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--weight-bits", type=int, choices=[0, 4, 8], default=0)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--packed-prefill", action="store_true")
    parser.add_argument("--full-attention-cache", action="store_true")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.prefill, args.context, args.chunk, args.threads, args.runs) <= 0 or min(args.warmups, args.decode) < 0:
        parser.error("counts must be positive; warmups and recurrent decode tokens must be non-negative")
    if args.runtime == "llama" and (args.gguf is None or not args.gguf.is_file()):
        parser.error("--runtime llama requires an existing --gguf checkpoint")
    if args.runtime == "llama" and (args.weight_bits or args.packed_prefill or args.full_attention_cache):
        parser.error("Kidi precision/cache overrides do not apply to llama-bench")
    args.cache.mkdir(parents=True, exist_ok=True)
    tokenizer = None
    prompt, tokens = None, None
    if args.runtime != "llama":
        tokenizer = Tokenizer.from_file(str(args.model / "tokenizer.json"))
        prompt, tokens = make_prompt(tokenizer, args.prefill)
    before = memory_snapshot()
    if args.runtime == "llama":
        result = run_llama(args)
    else:
        result = run_kidi(args, prompt, tokenizer) if args.runtime == "kidi" else run_litert(args, prompt, tokens)
    result["environment"] = {"platform": platform.platform(), "before": before, "after": memory_snapshot(),
                             "veclib_maximum_threads": os.environ.get("VECLIB_MAXIMUM_THREADS"),
                             "power": subprocess.check_output(["pmset", "-g", "batt"], text=True).strip()
                             if sys.platform == "darwin" else None}
    result.update({"runtime": args.runtime, "backend": args.backend, "batch_size": 1, "threads": args.threads,
                   "prefill_tokens": args.prefill, "decode_target": args.decode,
                   "context": None if args.runtime == "llama" else args.context,
                   "warmups": args.warmups, "prompt": prompt, "prompt_tokens": tokens,
                   "prompt_used": args.runtime != "llama",
                   "prefill_chunk_size": args.chunk, "packed_prefill": args.packed_prefill,
                   "model_config_sha256": hashlib.sha256((args.model / "config.json").read_bytes()).hexdigest()
                   if args.runtime != "llama" else None,
                   "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest() if prompt is not None else None})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(round_metrics(result), indent=2) + "\n")
    for record in result["records"]:
        print(f"{args.runtime}/{args.backend}: prefill {record['prefill_tokens_per_second']:.2f}, "
              f"decode {record['decode_tokens_per_second']:.2f} tokens/s")


if __name__ == "__main__":
    main()