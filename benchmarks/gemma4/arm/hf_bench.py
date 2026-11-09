"""Run Hugging Face Transformers BF16 reference under the Gemma 4 E2B CPU benchmark protocol."""

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
        raise SystemExit("usage: hf_bench.py MODEL_DIR PROTOCOL.json")
    model_path = Path(arguments[0]).expanduser().resolve()
    protocol = json.loads(Path(arguments[1]).expanduser().resolve().read_text())
    mode = protocol.get("mode", "speed")
    warmups = int(protocol.get("warmups", 1))
    threads = int(protocol["threads"])

    root = Path(__file__).resolve().parents[3]
    venv_site = root / ".cache/gemma-venv/lib/python3.12/site-packages"
    if venv_site.exists() and str(venv_site) not in sys.path:
        sys.path.insert(0, str(venv_site))

    import torch
    import transformers

    torch.set_num_threads(threads)
    transformers.utils.logging.set_verbosity_error()

    emit({
        "stage": "runtime",
        "device": "cpu:0",
        "backend": "HF-Transformers+PyTorch-BF16",
        "threads": threads,
    })
    loading = time.perf_counter()
    model = transformers.Gemma4ForConditionalGeneration.from_pretrained(
        str(model_path),
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).eval()
    emit({"stage": "load", "ms": (time.perf_counter() - loading) * 1000})

    @torch.inference_mode()
    def step_decode(prompt_ids, max_tokens, stop_on_eos):
        input_ids = torch.tensor([prompt_ids], dtype=torch.long)
        submitted = time.perf_counter()
        out = model(
            input_ids=input_ids,
            use_cache=True,
            logits_to_keep=1,
        )
        past_kv = out.past_key_values
        logits = out.logits[0, -1].float()
        if not torch.isfinite(logits).all():
            raise RuntimeError("non-finite HF Transformers logits")
        token = int(torch.argmax(logits).item())
        first = time.perf_counter()
        generated = []
        if stop_on_eos and token in (1, 106):
            return generated, first - submitted, 0.0
        generated.append(token)
        for _ in range(1, max_tokens):
            next_input = torch.tensor([[token]], dtype=torch.long)
            out = model(
                input_ids=next_input,
                past_key_values=past_kv,
                use_cache=True,
                logits_to_keep=1,
            )
            past_kv = out.past_key_values
            logits = out.logits[0, -1].float()
            if not torch.isfinite(logits).all():
                raise RuntimeError("non-finite HF Transformers logits")
            token = int(torch.argmax(logits).item())
            if stop_on_eos and token in (1, 106):
                break
            generated.append(token)
        last = time.perf_counter()
        return generated, first - submitted, last - first

    if mode == "eval":
        inputs_list = protocol["input_token_ids_list"]
        max_outputs = int(protocol["max_output_tokens"])
        for _ in range(warmups):
            step_decode(inputs_list[0], min(8, max_outputs), True)
        all_token_ids = []
        total_prompt_tokens = 0
        total_output_tokens = 0
        eval_started = time.perf_counter()
        for prompt_ids in inputs_list:
            total_prompt_tokens += len(prompt_ids)
            gen, _, _ = step_decode(prompt_ids, max_outputs, True)
            total_output_tokens += len(gen)
            all_token_ids.append(gen)
        emit({
            "stage": "eval",
            "items": len(inputs_list),
            "eval_seconds": time.perf_counter() - eval_started,
            "prompt_tokens": total_prompt_tokens,
            "output_tokens": total_output_tokens,
            "token_ids": all_token_ids,
        })
        return 0

    prompt = protocol["input_token_ids"]
    outputs = int(protocol["output_tokens"])
    repeats = int(protocol["repeats"])
    for iteration in range(warmups + repeats):
        generated, prefill, decode = step_decode(prompt, outputs, False)
        emit({
            "stage": "generation",
            "iteration": iteration,
            "warmup": iteration < warmups,
            "input_token_ids": prompt,
            "token_ids": generated,
            "prompt_tokens": len(prompt),
            "output_tokens": len(generated),
            "decode_tokens": len(generated) - 1,
            "reused_prompt_tokens": 0,
            "first_token_seconds": prefill,
            "decode_seconds": decode,
            "prefill_through_first_token_tps": len(prompt) / prefill,
            "decode_interval_tps": (outputs - 1) / decode,
        })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
