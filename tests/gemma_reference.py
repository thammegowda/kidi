#!/usr/bin/env python3
"""Regenerate the small independent Gemma 4 FP32 numerical fixture."""

import argparse
from pathlib import Path

from safetensors.torch import save_file
import torch
from transformers import Gemma4ForCausalLM, Gemma4TextConfig
import yaml


def generate(destination: Path):
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
    with torch.no_grad():
        for index, layer in enumerate(model.model.layers):
            layer.layer_scalar.fill_(0.7 + index * 0.1)
        tokens = torch.tensor([[2, 5, 7, 3, 8]])
        logits = model(tokens, use_cache=False).logits.contiguous()
    weights = {
        "model.language_model." + name.removeprefix("model."): value.contiguous()
        for name, value in model.state_dict().items() if name.startswith("model.")
    }
    destination.mkdir(parents=True, exist_ok=True)
    save_file(weights, destination / "model.safetensors")
    save_file({"tokens": tokens.to(torch.int32), "logits": logits}, destination / "reference.safetensors")
    document = config.to_dict()
    document["global_head_dim"] = 8
    document["num_global_key_value_heads"] = None
    document["type"] = "gemma4_text"
    (destination / "model.yaml").write_text(yaml.safe_dump({"model": document}, sort_keys=False))
    print(f"Saved {len(weights)} parameters and {logits.numel()} reference logits in {destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    generate(parser.parse_args().destination)