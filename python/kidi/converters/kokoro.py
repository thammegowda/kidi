#!/usr/bin/env python3
"""Convert hexgrad/Kokoro-82M to a self-contained Kidi INT8 TTS package."""

import argparse
import importlib.resources
import json
import os
import shutil
import tempfile
import math
from pathlib import Path

import torch
import yaml
from safetensors.torch import save_file

from ._licenses import write_license_bundle


TRANSPOSED_CONVOLUTIONS = {
    "predictor.module.F0.1.pool.weight",
    "predictor.module.N.1.pool.weight",
    "decoder.module.decode.3.pool.weight",
    "decoder.module.generator.ups.0.weight",
    "decoder.module.generator.ups.1.weight",
}

DEPTHWISE_TRANSPOSED = {
    "predictor.module.F0.1.pool.weight",
    "predictor.module.N.1.pool.weight",
    "decoder.module.decode.3.pool.weight",
}
GROUP_SIZE = 8


def _scale_name(name: str) -> str:
    return name.removesuffix(".weight") + ".scale" if name.endswith(".weight") else name + "_scale"


def _row_quantize(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    value = value.detach().to(torch.float32, copy=True).contiguous()
    rows = value.reshape(value.shape[0], -1)
    columns = rows.shape[1]
    groups = math.ceil(columns / GROUP_SIZE)
    padded = torch.nn.functional.pad(rows, (0, groups * GROUP_SIZE - columns))
    blocks = padded.reshape(rows.shape[0], groups, GROUP_SIZE)
    scale = blocks.abs().amax(dim=2) / 127.0
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    quantized = torch.round(blocks / scale.unsqueeze(-1)).clamp(-127, 127).to(torch.int8)
    quantized = quantized.reshape(rows.shape[0], groups * GROUP_SIZE)[:, :columns]
    return quantized.reshape(value.shape).contiguous(), scale.contiguous()


def _vector_quantize(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    value = value.detach().to(torch.float32, copy=True).reshape(-1).contiguous()
    scale = value.abs() / 127.0
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    quantized = torch.round(value / scale).clamp(-127, 127).to(torch.int8)
    return quantized.contiguous(), scale.to(torch.float32).contiguous()


def _flatten_checkpoint(path: Path) -> dict[str, torch.Tensor]:
    nested = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(nested, dict) or not all(isinstance(value, dict) for value in nested.values()):
        raise ValueError("Expected Kokoro checkpoint groups containing state dictionaries")
    return {
        f"{group}.{name}": tensor
        for group, state in nested.items()
        for name, tensor in state.items()
        if isinstance(tensor, torch.Tensor)
    }


def _fuse_weight_norm(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    pairs: dict[str, dict[str, torch.Tensor]] = {}
    for name, value in state.items():
        if name.endswith(".weight_g"):
            pairs.setdefault(name.removesuffix(".weight_g"), {})["g"] = value
        elif name.endswith(".weight_v"):
            pairs.setdefault(name.removesuffix(".weight_v"), {})["v"] = value
        else:
            result[name] = value
    for name, pair in pairs.items():
        if set(pair) != {"g", "v"}:
            raise ValueError(f"Incomplete Kokoro weight-normalization pair: {name}")
        value = pair["v"].to(torch.float32)
        dimensions = tuple(range(1, value.ndim))
        norm = value.norm(p=2, dim=dimensions, keepdim=True).clamp_min(1e-12)
        result[name + ".weight"] = pair["g"].to(torch.float32) * value / norm
    return result


def _encode_parameter(name: str, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    value = value.detach().to(torch.float32, copy=True).contiguous()
    if name in TRANSPOSED_CONVOLUTIONS:
        if value.ndim != 3:
            raise ValueError(f"Expected ConvTranspose1d weight for {name}, got {tuple(value.shape)}")
        if name in DEPTHWISE_TRANSPOSED:
            if value.shape[1] != 1:
                raise ValueError(f"Expected depthwise ConvTranspose1d weight for {name}")
            value = value[:, 0, :]
        else:
            value = value.permute(1, 2, 0).reshape(value.shape[1] * value.shape[2], value.shape[0])
    elif name.endswith(".weight") and value.ndim == 3:
        value = value.reshape(value.shape[0], -1)
    matrix = value.ndim >= 2 and (
        name.endswith(".weight")
        or ".weight_ih_" in name
        or ".weight_hh_" in name
    )
    if matrix:
        return _row_quantize(value)
    return _vector_quantize(value)


def _pronunciation(value: object) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        default = value.get("DEFAULT")
        if isinstance(default, str):
            return default
        return next((item for item in value.values() if isinstance(item, str)), None)
    return None


def _write_lexicon(destination: Path, british: bool) -> int:
    prefix = "gb" if british else "us"
    data_root = importlib.resources.files("misaki.data")
    entries: dict[str, str] = {}
    for tier in ("silver", "gold"):
        document = json.loads((data_root / f"{prefix}_{tier}.json").read_text(encoding="utf-8"))
        for word, value in document.items():
            pronunciation = _pronunciation(value)
            if pronunciation:
                entries[word.lower()] = pronunciation
    with destination.open("w", encoding="utf-8", newline="\n") as output:
        for word, pronunciation in sorted(entries.items()):
            if any(character in word for character in "\t\r\n") or any(
                character in pronunciation for character in "\t\r\n"
            ):
                continue
            output.write(f"{word}\t{pronunciation}\n")
    return len(entries)


def _model_document(config: dict, voices: list[str], license_files: list[str] | None = None) -> dict:
    return {
        "format_version": 1,
        "task": "tts",
        "license_files": license_files or [],
        "weights_file": "model.safetensors",
        "lexicon_file": "lexicon-us.tsv",
        "model": {
            "type": "kokoro",
            "config": config,
            "sample_rate": 24000,
            "voices": voices,
            "weight_group_size": GROUP_SIZE,
        },
        "synthesis": {
            "default_voice": voices[0],
            "voice_attributes": {
                "name": voices,
                "speed": "positive number; 1.0 is native rate",
            },
        },
        "quantization": {
            "format": "signed-int8",
            "matrix_scheme": "symmetric-groupwise",
            "group_size": GROUP_SIZE,
            "vector_scheme": "signed-value-with-per-element-scale",
            "scope": "all retained learned parameters and voice styles",
        },
    }


def convert(source: Path, destination: Path, voices: list[str], *, force: bool = False) -> Path:
    source = source.expanduser().resolve()
    destination = destination.expanduser().resolve()
    if not voices:
        raise ValueError("At least one Kokoro voice is required")
    required = [source / "config.json", source / "kokoro-v1_0.pth"]
    required.extend(source / "voices" / f"{voice}.pt" for voice in voices)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ValueError("Missing Kokoro files: " + ", ".join(missing))
    if destination.exists() and any(destination.iterdir()) and not force:
        raise ValueError(f"Destination is not empty: {destination}; pass --force to replace generated files")
    destination.mkdir(parents=True, exist_ok=True)

    state = _fuse_weight_norm(_flatten_checkpoint(source / "kokoro-v1_0.pth"))
    state = {name: value for name, value in state.items() if not name.startswith("bert.module.pooler.")}
    for voice in voices:
        value = torch.load(source / "voices" / f"{voice}.pt", map_location="cpu", weights_only=True)
        if not isinstance(value, torch.Tensor) or value.ndim != 3 or value.shape[1:] != (1, 256):
            raise ValueError(f"Unexpected Kokoro voice shape for {voice}: {getattr(value, 'shape', None)}")
        state[f"voices.{voice}.weight"] = value[:, 0, :]

    tensors: dict[str, torch.Tensor] = {}
    for name, value in state.items():
        encoded, scale = _encode_parameter(name, value)
        tensors[name] = encoded
        tensors[_scale_name(name)] = scale
    for name, value in tensors.items():
        scale = name.endswith(".scale") or name.endswith("_scale")
        expected = torch.float32 if scale else torch.int8
        if value.dtype != expected:
            raise ValueError(f"Unexpected converted dtype for {name}: {value.dtype}")

    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination, prefix=".model.", suffix=".safetensors", delete=False
        ) as output:
            temporary = Path(output.name)
        save_file(tensors, temporary)
        os.replace(temporary, destination / "model.safetensors")
        (destination / "model.safetensors").chmod(0o644)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    notice = """Kokoro converted model package

Kokoro-82M model and voice weights are licensed under Apache License 2.0.
The bundled pronunciation lexicon is derived from Misaki, licensed under Apache License 2.0.

Sources:
- https://huggingface.co/hexgrad/Kokoro-82M
- https://github.com/hexgrad/kokoro
- https://github.com/hexgrad/misaki
"""
    license_files = write_license_bundle(
        destination,
        {
            "KOKORO-APACHE-2.0.txt": "KOKORO-APACHE-2.0.txt",
            "MISAKI-APACHE-2.0.txt": "MISAKI-APACHE-2.0.txt",
        },
        notice,
    )
    document = _model_document(config, voices, license_files)
    (destination / "model.yaml").write_text(
        yaml.safe_dump(document, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    lexicon_entries = _write_lexicon(destination / "lexicon-us.tsv", british=False)
    for name in ("README.md", "VOICES.md"):
        if (source / name).is_file():
            shutil.copy2(source / name, destination / name)
    metadata = {
        "source": str(source),
        "voices": voices,
        "learned_tensors": sum(
            not (name.endswith(".scale") or name.endswith("_scale")) for name in tensors
        ),
        "scale_tensors": sum(name.endswith(".scale") or name.endswith("_scale") for name in tensors),
        "lexicon_entries": lexicon_entries,
    }
    (destination / "conversion.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="downloaded hexgrad/Kokoro-82M snapshot")
    parser.add_argument("destination", type=Path, help="output Kidi package directory")
    parser.add_argument(
        "--voices",
        default="af_heart",
        help="comma-separated voice packs to embed (default: af_heart)",
    )
    parser.add_argument("--force", action="store_true", help="replace generated files")
    args = parser.parse_args()
    voices = [voice.strip() for voice in args.voices.split(",") if voice.strip()]
    print(convert(args.source, args.destination, voices, force=args.force))


if __name__ == "__main__":
    main()
