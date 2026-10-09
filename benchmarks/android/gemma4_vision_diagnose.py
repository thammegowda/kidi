#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "safetensors==0.8.0",
#   "torch==2.14.0",
# ]
# ///
"""Diagnose Gemma 4 vision drift from matching official and Kidi tensor dumps."""

import argparse
import json
from pathlib import Path

from safetensors import safe_open
import torch
import torch.nn.functional as F


PROJECTIONS = {
    "q_proj": ("self_attn.q_proj", "input_norm"),
    "k_proj": ("self_attn.k_proj", "input_norm"),
    "v_proj": ("self_attn.v_proj", "input_norm"),
    "o_proj": ("self_attn.o_proj", "attended"),
    "gate_proj": ("mlp.gate_proj", "pre_ffn_norm"),
    "up_proj": ("mlp.up_proj", "pre_ffn_norm"),
    "down_proj": ("mlp.down_proj", "mlp_hidden"),
}


def metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    if actual.shape != expected.shape:
        raise ValueError(f"shape mismatch: {tuple(actual.shape)} != {tuple(expected.shape)}")
    actual = actual.double()
    expected = expected.double()
    delta = actual - expected
    error = delta.square().sum().item()
    energy = expected.square().sum().item()
    norm = actual.square().sum().item()
    return {
        "relative_rmse": (error / max(energy, 1e-30)) ** 0.5,
        "cosine": (actual * expected).sum().item() / max((energy * norm) ** 0.5, 1e-30),
        "max_abs": delta.abs().max().item(),
        "mismatches": torch.count_nonzero(actual != expected).item(),
        "elements": actual.numel(),
    }


def apply_srq(value: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    scale = scale.to(value.dtype)
    if scale.item() == 0:
        return value
    return torch.clamp(torch.round(value / scale), -128, 127) * scale


def quantization_margin(raw: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    if scale.item() == 0:
        return torch.full_like(raw, float("inf"))
    normalized = raw / scale.to(raw.dtype)
    return ((normalized - torch.floor(normalized)) - 0.5).abs()


def projection_key(layer: int, module: str, suffix: str) -> str:
    return f"model.vision_tower.encoder.layers.{layer}.{module}.linear.{suffix}"


def load_tensor(path: Path, name: str) -> torch.Tensor:
    with safe_open(path, framework="pt") as source:
        return source.get_tensor(name)


def analyze_projection(
    official_path: Path,
    kidi_path: Path,
    checkpoint_path: Path,
    layer: int,
    stage: str,
) -> list[dict]:
    module, input_stage = PROJECTIONS[stage]
    prefix = f"layer_{layer:02d}_"
    expected = load_tensor(official_path, prefix + stage).float()
    actual = load_tensor(kidi_path, prefix + stage).float()
    official_input = load_tensor(official_path, prefix + input_stage).float()
    kidi_input = load_tensor(kidi_path, prefix + input_stage).float()

    with safe_open(checkpoint_path, framework="pt") as checkpoint:
        weight = checkpoint.get_tensor(projection_key(layer, module, "weight")).float()
        weight_scale = checkpoint.get_tensor(projection_key(layer, module, "weight_scale")).float()
        input_scale = checkpoint.get_tensor(projection_key(layer, module, "input_activation_scale")).float()
        output_scale = checkpoint.get_tensor(projection_key(layer, module, "output_activation_scale")).float()
    weight = weight * weight_scale

    official_input_srq = apply_srq(official_input, input_scale)
    kidi_input_srq = apply_srq(kidi_input, input_scale)
    official_raw = F.linear(official_input_srq, weight)
    local_raw = F.linear(kidi_input_srq, weight)
    official_recomputed = apply_srq(official_raw, output_scale)
    official_on_kidi_input = apply_srq(local_raw, output_scale)
    if not torch.equal(official_recomputed, expected):
        mismatch = torch.count_nonzero(official_recomputed != expected).item()
        maximum = (official_recomputed - expected).abs().max().item()
        raise RuntimeError(
            f"official {layer}:{stage} recomputation changed: mismatches={mismatch} max={maximum}"
        )

    record = {
        "kind": "projection",
        "layer": layer,
        "stage": stage,
        "input_scale": input_scale.item(),
        "output_scale": output_scale.item(),
        "total": metrics(actual, expected),
        "local": metrics(actual, official_on_kidi_input),
        "propagated": metrics(official_on_kidi_input, expected),
    }
    local_mismatch = actual != official_on_kidi_input
    total_mismatch = actual != expected
    local_margins = quantization_margin(local_raw, output_scale)
    total_margins = quantization_margin(official_raw, output_scale)
    for label, mask, margins in (
        ("local", local_mismatch, local_margins),
        ("total", total_mismatch, total_margins),
    ):
        selected = margins[mask]
        record[f"{label}_margin"] = (
            {
                "minimum_quanta": selected.min().item(),
                "median_quanta": selected.median().item(),
                "p95_quanta": torch.quantile(selected, 0.95).item(),
            }
            if selected.numel()
            else None
        )

    return [record]


def analyze(
    official_path: Path,
    kidi_path: Path,
    checkpoint_path: Path | None,
    projections: list[str],
) -> list[dict]:
    records = []
    with safe_open(official_path, framework="pt") as official, safe_open(kidi_path, framework="pt") as kidi:
        common = sorted(set(official.keys()) & set(kidi.keys()))
        for name in common:
            expected = official.get_tensor(name)
            actual = kidi.get_tensor(name)
            records.append({"kind": "tensor", "name": name, **metrics(actual, expected)})
    if checkpoint_path is not None:
        for value in projections:
            layer_text, stage = value.split(":", 1)
            if stage not in PROJECTIONS:
                raise ValueError(f"unknown projection stage: {stage}")
            records.extend(
                analyze_projection(
                    official_path,
                    kidi_path,
                    checkpoint_path,
                    int(layer_text),
                    stage,
                )
            )
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official", type=Path, required=True)
    parser.add_argument("--kidi", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--projection",
        action="append",
        default=[],
        help="projection to analyze as LAYER:STAGE, for example 0:q_proj",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.projection and args.checkpoint is None:
        parser.error("--projection requires --checkpoint")
    for projection in args.projection:
        parts = projection.split(":", 1)
        if len(parts) != 2 or parts[1] not in PROJECTIONS:
            parser.error(f"invalid projection {projection!r}; expected LAYER:STAGE")
        try:
            layer = int(parts[0])
        except ValueError:
            parser.error(f"invalid projection layer in {projection!r}")
        if not 0 <= layer < 16:
            parser.error(f"projection layer must be from 0 to 15: {projection!r}")
    checkpoint = args.checkpoint
    if checkpoint is not None and checkpoint.is_dir():
        checkpoint = checkpoint / "model.safetensors"
    records = analyze(
        args.official,
        args.kidi,
        checkpoint,
        args.projection,
    )
    for record in records:
        print(json.dumps(record, sort_keys=True))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
