"""Run Google LiteRT-LM under the Gemma 4 E2B CPU benchmark protocol."""

import json
from pathlib import Path
import resource
import sys
import time


def peak_rss_kib():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss // 1024 if sys.platform == "darwin" else rss


def emit(record):
    record["peak_rss_kib"] = peak_rss_kib()
    print(json.dumps(record), flush=True)


def main(argv=None):
    arguments = sys.argv[1:] if argv is None else argv
    if len(arguments) != 2:
        raise SystemExit("usage: litert_bench.py MODEL.litertlm PROTOCOL.json")
    model_path = Path(arguments[0]).expanduser().resolve()
    protocol = json.loads(Path(arguments[1]).expanduser().resolve().read_text())
    mode = protocol.get("mode", "speed")
    warmups = protocol.get("warmups", 1)
    context = protocol["context_tokens"]
    threads = protocol["threads"]
    cache_dir = model_path.parent / ".litert-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    import litert_lm

    emit({
        "stage": "runtime",
        "device": "cpu:0",
        "backend": "LiteRT-LM+XNNPACK",
        "threads": threads,
    })
    loading = time.perf_counter()
    engine = litert_lm.Engine(
        str(model_path),
        backend=litert_lm.Backend.CPU(thread_count=threads),
        max_num_tokens=context,
        cache_dir=str(cache_dir),
        enable_benchmark=True,
    )
    emit({"stage": "load", "ms": (time.perf_counter() - loading) * 1000})

    if mode == "eval":
        prompts = protocol["prompts"]
        max_outputs = int(protocol["max_output_tokens"])
        with engine:
            for _ in range(warmups):
                with engine.create_session(
                    apply_prompt_template=False,
                    sampler_config=litert_lm.SamplerConfig(top_k=1, temperature=0.0, seed=0),
                    max_output_tokens=min(8, max_outputs),
                ) as session:
                    session.run_prefill([prompts[0].removeprefix("<bos>")])
                    session.run_decode()
            all_token_ids = []
            outputs = []
            total_prompt_tokens = 0
            total_output_tokens = 0
            eval_started = time.perf_counter()
            for raw_prompt in prompts:
                native_prompt = raw_prompt.removeprefix("<bos>")
                with engine.create_session(
                    apply_prompt_template=False,
                    sampler_config=litert_lm.SamplerConfig(top_k=1, temperature=0.0, seed=0),
                    max_output_tokens=max_outputs,
                ) as session:
                    session.run_prefill([native_prompt])
                    response = session.run_decode()
                    info = session.get_benchmark_info()
                    text = response.texts[0]
                    gen = list(engine.tokenize(text))
                    total_prompt_tokens += int(info.last_prefill_token_count)
                    total_output_tokens += len(gen)
                    all_token_ids.append(gen)
                    outputs.append(text)
            emit({
                "stage": "eval",
                "items": len(prompts),
                "eval_seconds": time.perf_counter() - eval_started,
                "prompt_tokens": total_prompt_tokens,
                "output_tokens": total_output_tokens,
                "outputs": outputs,
                "token_ids": all_token_ids,
            })
        return 0

    prompt = protocol["prompt"]
    expected_ids = protocol["input_token_ids"]
    outputs = protocol["output_tokens"]
    repeats = protocol["repeats"]

    native_prompt = "<|turn>user\n" + prompt + "<turn|>\n<|turn>model\n"
    with engine:
        actual_ids = [engine.bos_token_id, *engine.tokenize(native_prompt)]
        if actual_ids != expected_ids:
            raise ValueError(f"LiteRT-LM prompt token IDs differ: {actual_ids[:20]} != {expected_ids[:20]}")
        for iteration in range(warmups + repeats):
            with engine.create_session(
                apply_prompt_template=False,
                sampler_config=litert_lm.SamplerConfig(top_k=1, temperature=0.0, seed=0),
                max_output_tokens=outputs,
            ) as session:
                submitted = time.perf_counter()
                session.run_prefill([native_prompt])
                response = session.run_decode()
                finished = time.perf_counter()
                info = session.get_benchmark_info()
                generated = list(engine.tokenize(response.texts[0]))
                if info.last_prefill_token_count != len(expected_ids) or len(generated) != outputs:
                    raise ValueError(f"LiteRT-LM token count mismatch: prefill={info.last_prefill_token_count}, "
                                     f"generated={len(generated)}")
                first_token_seconds = float(info.time_to_first_token_in_second)
                decode_seconds = (finished - submitted) - first_token_seconds
                if first_token_seconds <= 0 or decode_seconds <= 0:
                    raise ValueError("invalid LiteRT-LM timing")
                emit({
                    "stage": "generation",
                    "iteration": iteration,
                    "warmup": iteration < warmups,
                    "input_token_ids": expected_ids,
                    "token_ids": generated,
                    "prompt_tokens": len(expected_ids),
                    "output_tokens": len(generated),
                    "decode_tokens": len(generated) - 1,
                    "reused_prompt_tokens": 0,
                    "first_token_seconds": first_token_seconds,
                    "decode_seconds": decode_seconds,
                    "prefill_through_first_token_tps": len(expected_ids) / first_token_seconds,
                    "decode_interval_tps": (outputs - 1) / decode_seconds,
                })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
