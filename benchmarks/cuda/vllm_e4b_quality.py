"""Generate the pinned E4B translation fixture with vLLM on physical GPU 7."""

import importlib.metadata
import json
import os
from pathlib import Path
import sys


os.environ["CUDA_VISIBLE_DEVICES"] = "7"


def main():
    from vllm import LLM, SamplingParams

    if len(sys.argv) != 3:
        raise ValueError("usage: vllm_e4b_quality.py PROTOCOL.json MODEL")
    protocol = json.loads(Path(sys.argv[1]).read_text())
    settings = {
        "model": sys.argv[2], "dtype": "bfloat16", "quantization": "compressed-tensors",
        "tensor_parallel_size": 1, "max_model_len": protocol["context"],
        "max_num_batched_tokens": protocol["chunk"], "max_num_seqs": 1,
        "kv_cache_memory_bytes": 1024**3, "language_model_only": True,
        "enable_prefix_caching": False, "cpu_offload_gb": 0, "seed": 0,
        "disable_log_stats": False,
    }
    llm = LLM(**settings)
    sampling = SamplingParams(temperature=0, max_tokens=protocol["output_tokens"], ignore_eos=True,
                              stop_token_ids=[1, 106], seed=0)
    print(json.dumps({"stage": "runtime", "settings": settings,
                      "vllm_version": importlib.metadata.version("vllm")}), flush=True)
    for case in protocol["workloads"]:
        ids = case["input_token_ids"]
        if llm.get_tokenizer().encode(case["prompt"], add_special_tokens=False) != ids:
            raise RuntimeError(f"{case['name']}: vLLM tokenizer disagrees with the fixture")
        result = llm.generate([{"prompt_token_ids": ids}], sampling, use_tqdm=False)[0]
        if list(result.prompt_token_ids) != ids:
            raise RuntimeError(f"{case['name']}: vLLM received different prompt tokens")
        output = result.outputs[0]
        tokens = list(output.token_ids)
        if output.finish_reason == "stop" and output.stop_reason not in (1, 106):
            raise RuntimeError(f"{case['name']}: unexpected stop reason {output.stop_reason!r}")
        if output.finish_reason not in ("stop", "length"):
            raise RuntimeError(f"{case['name']}: unexpected finish reason {output.finish_reason!r}")
        if tokens and tokens[-1] in (1, 106):
            tokens.pop()
        if any(token in (1, 106) for token in tokens):
            raise RuntimeError(f"{case['name']}: output continues after a stop token")
        print(json.dumps({
            "stage": "eval_item", "workload": case["name"], "input_token_ids": ids,
            "token_ids": tokens, "finish_reason": output.finish_reason, "stop_reason": output.stop_reason,
            "truncated": output.finish_reason == "length",
        }), flush=True)


if __name__ == "__main__":
    main()
