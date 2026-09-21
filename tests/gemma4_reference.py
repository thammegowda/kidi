#!/usr/bin/env python3
"""Regenerate the small independent Gemma 4 FP32 numerical fixture."""

import argparse
import json
import re
from pathlib import Path

from safetensors.torch import save_file
from safetensors import safe_open
import torch
from transformers import Gemma4ForCausalLM, Gemma4TextConfig
import yaml


def generate(destination: Path, qat: bool = False):
    torch.manual_seed(173)
    torch.set_num_threads(1)
    config = Gemma4TextConfig(
        hidden_size=8, intermediate_size=12, num_hidden_layers=4, num_kv_shared_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=4, global_head_dim=8,
        vocab_size=16, vocab_size_per_layer_input=16, hidden_size_per_layer_input=2,
        max_position_embeddings=16, sliding_window=3, use_double_wide_mlp=True, final_logit_softcapping=30.0,
        layer_types=["sliding_attention", "full_attention", "sliding_attention", "full_attention"],
        rope_parameters={
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
            "full_attention": {"rope_type": "proportional", "rope_theta": 1000000.0, "partial_rotary_factor": 0.25},
        },
        attn_implementation="eager",
    )
    model = Gemma4ForCausalLM(config).float().eval()
    if qat:
        from transformers.integrations.gemma_quant import QuantizedEmbedding, QuantizedLinear, apply_srq
        from transformers.cache_utils import DynamicCache

        for name, module in list(model.named_modules()):
            if isinstance(module, torch.nn.Linear) and "per_layer_model_projection" not in name:
                bits = 2 if name == "lm_head" or ".layers.2.mlp." in name or ".layers.3.mlp." in name else 8 if "per_layer_" in name else 4
                replacement = QuantizedLinear(module.in_features, module.out_features, num_bits=bits)
                with torch.no_grad():
                    if bits == 8:
                        replacement.weight.copy_(torch.randint(-127, 128, replacement.weight.shape, dtype=torch.int8))
                    else:
                        replacement.weight.copy_(torch.randint(0, 256, replacement.weight.shape, dtype=torch.uint8))
                    replacement.weight_scale.fill_(0.01 if bits < 8 else 0.001)
                    replacement.input_activation_scale.fill_(0.012731)
                    replacement.output_activation_scale.fill_(0.009173)
                model.set_submodule(name, replacement)
        for name, width, bits, groups, scale in (
            ("model.embed_tokens", 8, 2, 1, 8 ** 0.5),
            ("model.embed_tokens_per_layer", 8, 4, 4, 2 ** 0.5),
        ):
            replacement = QuantizedEmbedding(16, width, torch.float32, scale, bits)
            replacement.embedding_scale = torch.nn.Parameter(torch.full((16, groups), 0.03))
            with torch.no_grad():
                replacement.embedding_quantized.copy_(torch.randint(0, 256, replacement.embedding_quantized.shape, dtype=torch.uint8))
            model.set_submodule(name, replacement)
        for layer in model.model.layers[:2]:
            layer.self_attn.register_buffer("k_cache_scale", torch.tensor(0.01))
            layer.self_attn.register_buffer("v_cache_scale", torch.tensor(0.01))
        cache = DynamicCache(config=config)
        original_update = cache.update

        def rounded_cache(key, value, layer_idx, *args, **kwargs):
            layer = model.model.layers[layer_idx].self_attn
            return original_update(apply_srq(key, layer.k_cache_scale), apply_srq(value, layer.v_cache_scale), layer_idx, *args, **kwargs)

        cache.update = rounded_cache
    with torch.no_grad():
        for index, layer in enumerate(model.model.layers):
            layer.layer_scalar.fill_(0.7 + index * 0.1)
        tokens = torch.tensor([[2, 5, 7, 3, 8]])
        logits = model(tokens, use_cache=qat, **({"past_key_values": cache} if qat else {})).logits.contiguous()
    weights = {
        "model.language_model." + name.removeprefix("model."): value.contiguous()
        for name, value in model.state_dict().items() if name.startswith("model.")
    }
    if qat:
        weights.update({name: value.contiguous() for name, value in model.state_dict().items() if name.startswith("lm_head.")})
    destination.mkdir(parents=True, exist_ok=True)
    save_file(weights, destination / "model.safetensors")
    save_file({"tokens": tokens.to(torch.int32), "logits": logits}, destination / "reference.safetensors")
    document = config.to_dict()
    document["global_head_dim"] = 8
    document["num_global_key_value_heads"] = None
    document["type"] = "gemma4_text"
    if qat:
        document["tie_word_embeddings"] = False
        document["quantization_config"] = {
            "quant_method": "gemma", "num_bits": 4, "quantize_embeddings": True,
            "modules_to_not_convert": ["per_layer_model_projection"],
            "module_quant_configs": {
                "^lm_head$": {"num_bits": 2},
                "language_model\\.embed_tokens$": {"num_bits": 2},
                "language_model\\.embed_tokens_per_layer$": {"num_bits": 4},
                "layers\\.[23]\\.mlp\\.": {"num_bits": 2},
                "per_layer_input_gate$": {"num_bits": 8},
                "per_layer_projection$": {"num_bits": 8},
            },
        }
    (destination / "model.yaml").write_text(yaml.safe_dump({"model": document}, sort_keys=False))
    print(f"Saved {len(weights)} parameters and {logits.numel()} reference logits in {destination}")


def checkpoint_reference(directory: Path, destination: Path, prompt_tokens: Path | None = None):
    from transformers.integrations.gemma_quant import QuantizedEmbedding, QuantizedLinear, apply_srq
    from transformers.cache_utils import DynamicCache

    torch.set_num_threads(4)
    document = json.loads((directory / "config.json").read_text())
    config = Gemma4TextConfig(**document["text_config"], attn_implementation="eager")
    policy = document["quantization_config"]

    def bits_for(name):
        if any(excluded in name for excluded in policy["modules_to_not_convert"]):
            return 0
        for pattern, value in policy["module_quant_configs"].items():
            if re.search(pattern, name):
                return value["num_bits"]
        return policy["num_bits"]

    with torch.device("meta"):
        model = Gemma4ForCausalLM(config)
        for name, module in list(model.named_modules()):
            source_name = "model.language_model." + name.removeprefix("model.") if name.startswith("model.") else name
            bits = bits_for(source_name)
            if bits and isinstance(module, torch.nn.Linear):
                model.set_submodule(name, QuantizedLinear(module.in_features, module.out_features, num_bits=bits))
            elif bits and isinstance(module, torch.nn.Embedding):
                replacement = QuantizedEmbedding(module.num_embeddings, module.embedding_dim, torch.float32,
                    getattr(module, "scalar_embed_scale", 1.0), bits)
                if name.endswith("embed_tokens_per_layer"):
                    replacement.embedding_scale = torch.nn.Parameter(torch.empty(module.num_embeddings, config.num_hidden_layers))
                model.set_submodule(name, replacement)
        for layer in model.model.layers:
            if not layer.self_attn.is_kv_shared_layer:
                layer.self_attn.register_buffer("k_cache_scale", torch.tensor(0.0))
                layer.self_attn.register_buffer("v_cache_scale", torch.tensor(0.0))
    state = {}
    declared = set(model.state_dict())
    with safe_open(directory / "model.safetensors", framework="pt") as source:
        for key in source.keys():
            name = key.replace("model.language_model.", "model.", 1)
            if name in declared:
                value = source.get_tensor(key)
                state[name] = value.float() if value.is_floating_point() else value
    model.load_state_dict(state, strict=True, assign=True)
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding
    model.model.rotary_emb = Gemma4TextRotaryEmbedding(config)
    model.eval()
    cache = DynamicCache(config=config)
    original_update = cache.update

    def rounded_cache(key, value, layer_idx, *args, **kwargs):
        layer = model.model.layers[layer_idx].self_attn
        return original_update(apply_srq(key, layer.k_cache_scale), apply_srq(value, layer.v_cache_scale), layer_idx, *args, **kwargs)

    cache.update = rounded_cache
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
    text = "<bos><|turn>user\nExplain why sunlight appears blue in the sky, then give a short Python function that adds two numbers.<turn|>\n<|turn>model\n"
    token_ids = json.loads(prompt_tokens.read_text())["prompt_tokens"] if prompt_tokens else tokenizer.encode(text, add_special_tokens=False).ids
    tokens = torch.tensor([token_ids])
    with torch.no_grad():
        logits = model(tokens, use_cache=True, past_key_values=cache).logits.contiguous()
    destination.mkdir(parents=True, exist_ok=True)
    save_file({"tokens": tokens.to(torch.int32), "logits": logits}, destination / "reference.safetensors")
    print(f"Saved full QAT reference for {tokens.numel()} positions in {destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--qat", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--prompt-tokens", type=Path, help="benchmark JSON containing independent reference input IDs")
    args = parser.parse_args()
    if args.checkpoint:
        checkpoint_reference(args.checkpoint, args.destination, args.prompt_tokens)
    else:
        generate(args.destination, args.qat)