#!/usr/bin/env python3
"""Sequential batch-one Gemma benchmarks with explicit runtime and precision labels."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import time

from tokenizers import Tokenizer


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
               "--prefill-chunk-size", str(args.chunk), "--raw-prompt", "--ignore-eos", "--profile",
               "--warmups", str(args.warmups), "--runs", str(args.runs), "--prompt", prompt]
    if args.weight_bits:
        command += ["--weight-bits", str(args.weight_bits), "--group-size", str(args.group_size)]
    if args.packed_prefill:
        command += ["--packed-prefill"]
    if args.full_attention_cache:
        command += ["--full-attention-cache"]
    started = time.perf_counter()
    process = subprocess.run(command, text=True, capture_output=True)
    if process.returncode:
        raise RuntimeError(f"Kidi failed ({process.returncode}): {process.stderr}")
    wall = time.perf_counter() - started
    records = []
    for line in process.stderr.splitlines():
        if not line.startswith("kidi_generation|"):
            continue
        fields = dict(part.split("=", 1) for part in line.split("|")[1:])
        record = {key: int(value) if value.isdigit() else value for key, value in fields.items()}
        if bool(record.get("packed_prefill", 0)) != args.packed_prefill:
            raise ValueError("Executed prefill policy does not match requested policy")
        ids = [int(value) for value in fields["token_ids"].split(",") if value]
        record["token_ids"] = ids
        record["text"] = tokenizer.decode(ids, skip_special_tokens=False)
        if record["prompt_tokens"] != args.prefill or record["decode_tokens"] != args.decode:
            raise ValueError(f"Kidi token count mismatch: {record}")
        record["decode_tokens_per_second"] = args.decode * 1e9 / record["decode_ns"] if args.decode else 0
        record["prefill_tokens_per_second"] = args.prefill * 1e9 / record["prefill_ns"]
        records.append(record)
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
    return {"records": records, "process_wall_seconds": wall,
            "binary_sha256": binary_sha256,
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
            "stderr": process.stderr, "weight_precision": precision,
            "prefill_precision": ("native QAT packed" if args.backend == "cpu" or args.packed_prefill else "native QAT cached FP16 matrices")
            if native_qat else "packed" if args.weight_bits and args.packed_prefill else "original floating (multi-row)"}


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
    parser.add_argument("--runtime", choices=["kidi", "litert", "litert-mtp"], required=True)
    parser.add_argument("--backend", choices=["cpu", "gpu"], required=True)
    parser.add_argument("--model", type=Path, default=root.parent / "models/gemma-4-E2B-it")
    parser.add_argument("--reference-model", type=Path,
                        default=root.parent / "models/gemma-4-E2B-it-litert-lm/gemma-4-E2B-it.litertlm")
    parser.add_argument("--binary", type=Path, default=root / "build-release/kidi")
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
    args.cache.mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer.from_file(str(args.model / "tokenizer.json"))
    prompt, tokens = make_prompt(tokenizer, args.prefill)
    result = run_kidi(args, prompt, tokenizer) if args.runtime == "kidi" else run_litert(args, prompt, tokens)
    result.update({"runtime": args.runtime, "backend": args.backend, "batch_size": 1, "threads": args.threads,
                   "prefill_tokens": args.prefill, "decode_target": args.decode, "context": args.context,
                   "warmups": args.warmups, "prompt": prompt, "prompt_tokens": tokens,
                   "prefill_chunk_size": args.chunk, "packed_prefill": args.packed_prefill,
                   "model_config_sha256": hashlib.sha256((args.model / "config.json").read_bytes()).hexdigest(),
                   "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for record in result["records"]:
        print(f"{args.runtime}/{args.backend}: prefill {record['prefill_tokens_per_second']:.2f}, "
              f"decode {record['decode_tokens_per_second']:.2f} tokens/s")


if __name__ == "__main__":
    main()