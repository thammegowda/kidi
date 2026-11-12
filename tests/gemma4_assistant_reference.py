#!/usr/bin/env python3
"""Generate independent FP32 Gemma MTP assistant fixtures with Transformers."""

import argparse
import json
from pathlib import Path

import torch
import yaml
from safetensors.torch import load_file, save_file
from transformers import (
    Gemma4AssistantConfig,
    Gemma4AssistantForCausalLM,
    Gemma4ForCausalLM,
    Gemma4TextConfig,
)


def generate(target_directory: Path, destination: Path):
    torch.set_num_threads(1)
    document = yaml.safe_load((target_directory / "model.yaml").read_text())["model"]
    target_config = Gemma4TextConfig(**document, attn_implementation="eager")
    target = Gemma4ForCausalLM(target_config).float().eval()
    state = {
        "model." + name.removeprefix("model.language_model."): value
        for name, value in load_file(target_directory / "model.safetensors").items()
    }
    state["lm_head.weight"] = state["model.embed_tokens.weight"]
    target.load_state_dict(state)
    raw_hidden = []
    hook = target.model.norm.register_forward_pre_hook(
        lambda module, args: raw_hidden.append(args[0].detach().clone())
    )
    tokens = torch.tensor([[2, 5, 7, 3, 8]])
    with torch.no_grad():
        target_output = target(
            tokens, use_cache=False, output_hidden_states=True, return_shared_kv_states=True
        )
    hook.remove()
    torch.testing.assert_close(target_output.hidden_states[-1], target.model.norm(raw_hidden[0]))

    text = Gemma4TextConfig(
        hidden_size=4,
        intermediate_size=12,
        num_hidden_layers=2,
        num_kv_shared_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        global_head_dim=8,
        vocab_size=16,
        vocab_size_per_layer_input=0,
        hidden_size_per_layer_input=0,
        max_position_embeddings=16,
        sliding_window=3,
        use_double_wide_mlp=False,
        final_logit_softcapping=None,
        layer_types=["sliding_attention", "full_attention"],
        rope_parameters=document["rope_parameters"],
        attn_implementation="eager",
    )
    config = Gemma4AssistantConfig(
        text_config=text,
        backbone_hidden_size=8,
        use_ordered_embeddings=True,
        num_centroids=4,
        centroid_intermediate_top_k=2,
        attn_implementation="eager",
    )
    torch.manual_seed(319)
    assistant = Gemma4AssistantForCausalLM(config).float().eval()
    with torch.no_grad():
        assistant.masked_embedding.token_ordering.copy_(torch.randperm(16))
        for index, layer in enumerate(assistant.model.layers):
            layer.layer_scalar.fill_(0.7 + index * 0.1)
        for name, parameter in assistant.named_parameters():
            if "norm" in name and parameter.ndim == 1:
                parameter.copy_(torch.linspace(0.8, 1.2, parameter.numel()))
    reference = {
        "tokens": tokens.to(torch.int32),
        "target_hidden": target_output.hidden_states[-1].contiguous(),
    }
    first_token = target_output.logits[:, -1:].argmax(dim=-1)
    reference["first_token"] = first_token.to(torch.int32)
    for ordered in (True, False):
        assistant.config.use_ordered_embeddings = ordered
        hidden = target_output.hidden_states[-1][:, -1:]
        token = first_token
        predictions, activations = [], []
        with torch.no_grad():
            for _ in range(3):
                output = assistant(
                    inputs_embeds=torch.cat((target.model.embed_tokens(token), hidden), dim=-1),
                    position_ids=torch.tensor([[tokens.shape[1]]]),
                    shared_kv_states=target_output.shared_kv_states,
                    use_cache=False,
                )
                token = output.logits.argmax(dim=-1)
                hidden = output.last_hidden_state
                predictions.append(token)
                activations.append(hidden)
        suffix = "" if ordered else "_dense"
        reference["draft_tokens" + suffix] = torch.cat(predictions, dim=1).to(torch.int32)
        reference["draft_hidden" + suffix] = torch.cat(activations, dim=1).contiguous()
    assistant.config.use_ordered_embeddings = True
    destination.mkdir(parents=True, exist_ok=True)
    save_file(
        {name: value.detach().clone().contiguous() for name, value in assistant.state_dict().items()
         if name != "lm_head.weight"},
        destination / "model.safetensors",
    )
    save_file(reference, destination / "reference.safetensors")
    document = config.to_dict()
    document["text_config"]["global_head_dim"] = 8
    document["text_config"]["num_global_key_value_heads"] = None
    (destination / "config.json").write_text(json.dumps(document, indent=2) + "\n")
    print(f"Saved independent Gemma MTP fixtures in {destination}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    data = Path(__file__).resolve().parent / "data"
    parser.add_argument("--target", type=Path, default=data / "gemma4")
    parser.add_argument("--output", type=Path, default=data / "gemma4-assistant")
    arguments = parser.parse_args()
    generate(arguments.target, arguments.output)


if __name__ == "__main__":
    main()
