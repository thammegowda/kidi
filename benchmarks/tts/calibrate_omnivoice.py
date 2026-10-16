#!/usr/bin/env python3
"""Collect OmniVoice Qwen activation ranges for calibrated A8 NPU islands."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import re


@dataclass
class Statistics:
    count: int = 0
    maximum: float = 0.0
    sum_squares: float = 0.0
    samples: list[float] | None = None

    def update(self, tensor) -> None:
        values = tensor.detach().float()
        self.count += values.numel()
        self.maximum = max(self.maximum, values.abs().max().item())
        self.sum_squares += values.square().sum().item()
        absolute = values.abs().flatten()
        stride = max(1, (absolute.numel() + 2047) // 2048)
        sampled = absolute[::stride][:2048].cpu().tolist()
        if self.samples is None:
            self.samples = sampled
        else:
            self.samples.extend(sampled)
            if len(self.samples) > 65536:
                self.samples = self.samples[::2]

    def record(self) -> dict:
        samples = sorted(self.samples or [])

        def percentile(fraction: float) -> float:
            if not samples:
                return 0.0
            return samples[min(len(samples) - 1, round(fraction * (len(samples) - 1)))]

        p99 = percentile(0.99)
        p999 = percentile(0.999)
        p9999 = percentile(0.9999)
        return {
            "count": self.count,
            "max_abs": self.maximum,
            "rms": (self.sum_squares / self.count) ** 0.5 if self.count else 0.0,
            "sample_count": len(samples),
            "p99_abs": p99,
            "p99_9_abs": p999,
            "p99_99_abs": p9999,
            "symmetric_int8_scale": self.maximum / 127.0 if self.maximum else 1.0,
            "symmetric_int8_scale_p99": p99 / 127.0 if p99 else 1.0,
            "symmetric_int8_scale_p99_9": p999 / 127.0 if p999 else 1.0,
            "symmetric_int8_scale_p99_99": p9999 / 127.0 if p9999 else 1.0,
        }


def bucket(step: int, steps: int) -> str:
    third = max(1, (steps + 2) // 3)
    if step < third:
        return "early"
    if step < 2 * third:
        return "middle"
    return "late"


def projection(name: str) -> tuple[int, str] | None:
    match = re.search(r"(?:^|\.)layers\.(\d+)\.(?:self_attn|mlp)\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$", name)
    if not match:
        return None
    return int(match[1]), match[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="Hugging Face model ID or local OmniVoice directory")
    parser.add_argument("output", type=Path)
    parser.add_argument("--text", action="append", required=True, help="text or @UTF8_FILE; repeatable")
    parser.add_argument("--language", default="English")
    parser.add_argument("--instruction")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--guidance-scale", type=float, default=2.0)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.steps <= 0:
        raise ValueError("--steps must be positive")

    import torch
    from omnivoice import OmniVoice

    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]
    model = OmniVoice.from_pretrained(args.model, device_map=args.device, dtype=dtype)
    statistics: dict[tuple[int, str, str], Statistics] = defaultdict(Statistics)
    calls: dict[str, int] = defaultdict(int)
    handles = []

    def observe(source_name: str, layer: int, edge: str, value) -> None:
        call = calls[source_name]
        calls[source_name] += 1
        statistics[layer, edge, bucket(call % args.steps, args.steps)].update(value)

    for name, module in model.named_modules():
        found = projection(name)
        if not found:
            continue
        layer, kind = found
        if kind in {"q_proj", "k_proj", "v_proj"}:
            input_edge, output_edge = "attention_norm", "qkv"
        elif kind == "o_proj":
            input_edge, output_edge = "attention_output", "attention_residual"
        elif kind in {"gate_proj", "up_proj"}:
            input_edge, output_edge = "ffn_norm", "gate_up"
        else:
            input_edge, output_edge = "ffn_hidden", "ffn_residual"

        observe_input = kind not in {"k_proj", "v_proj", "up_proj"}

        def post_hook(_, __, output, source_name=name, layer=layer, edge=output_edge):
            observe(source_name + ":output", layer, edge, output)

        if observe_input:
            def pre_hook(_, inputs, source_name=name, layer=layer, edge=input_edge):
                observe(source_name + ":input", layer, edge, inputs[0])

            handles.append(module.register_forward_pre_hook(pre_hook))
        handles.append(module.register_forward_hook(post_hook))

    if not handles:
        raise RuntimeError("no Qwen projection modules were found")

    texts = []
    for specification in args.text:
        if specification.startswith("@"):
            texts.append(Path(specification[1:]).read_text(encoding="utf-8"))
        else:
            texts.append(specification)

    try:
        for index, text in enumerate(texts):
            torch.manual_seed(args.seed + index)
            model.generate(
                text=text,
                language=args.language,
                instruct=args.instruction,
                num_step=args.steps,
                guidance_scale=args.guidance_scale,
                postprocess_output=False,
            )
    finally:
        for handle in handles:
            handle.remove()

    layers: dict[str, dict[str, dict[str, dict]]] = {}
    for (layer, edge, step_bucket), values in sorted(statistics.items()):
        layers.setdefault(str(layer), {}).setdefault(edge, {})[step_bucket] = values.record()
    document = {
        "format_version": 1,
        "model": args.model,
        "device": args.device,
        "dtype": args.dtype,
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "language": args.language,
        "instruction": args.instruction,
        "samples": len(texts),
        "layers": layers,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
