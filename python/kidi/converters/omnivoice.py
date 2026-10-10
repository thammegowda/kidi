#!/usr/bin/env python3
"""Convert an OmniVoice Hugging Face snapshot to Kidi's decoder-only INT8 package."""

import argparse
import json
import math
import os
import shutil
import tempfile
from pathlib import Path

import torch
import yaml
from safetensors import safe_open
from safetensors.torch import save_file

from ._licenses import write_license_bundle


def _row_quantize(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    value = value.detach().to(torch.float32, copy=True).contiguous()
    rows = value.reshape(value.shape[0], -1)
    scale = rows.abs().amax(dim=1, keepdim=True) / 127.0
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    quantized = torch.round(rows / scale).clamp(-127, 127).to(torch.int8)
    return quantized.reshape(value.shape).contiguous(), scale.contiguous()


def _vector_quantize(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    value = value.detach().to(torch.float32, copy=True).reshape(-1).contiguous()
    maximum = value.abs().amax()
    scale = maximum / 127.0 if maximum > 0 else torch.tensor(1.0)
    quantized = torch.round(value / scale).clamp(-127, 127).to(torch.int8)
    return quantized.contiguous(), scale.reshape(1).to(torch.float32).contiguous()


def _weight_scale_name(name: str) -> str:
    if not name.endswith(".weight"):
        raise ValueError(f"Expected a weight tensor name, got {name}")
    return name.removesuffix(".weight") + ".scale"


def _store_weight(output: dict[str, torch.Tensor], name: str, value: torch.Tensor) -> None:
    if value.ndim < 2:
        quantized, scale = _vector_quantize(value)
    else:
        quantized, scale = _row_quantize(value)
    output[name] = quantized
    output[_weight_scale_name(name)] = scale


def _store_vector(
    output: dict[str, torch.Tensor],
    name: str,
    scale_name: str,
    value: torch.Tensor,
) -> None:
    quantized, scale = _vector_quantize(value)
    output[name] = quantized
    output[scale_name] = scale


def _convert_generator(source: Path, output: dict[str, torch.Tensor]) -> None:
    with safe_open(source / "model.safetensors", framework="pt", device="cpu") as checkpoint:
        consumed: set[str] = {"codebook_layer_offsets"}
        for name in checkpoint.keys():
            if name.endswith(".self_attn.q_proj.weight"):
                prefix = name.removesuffix("q_proj.weight")
                sources = [prefix + projection for projection in ("q_proj.weight", "k_proj.weight", "v_proj.weight")]
                _store_weight(
                    output,
                    prefix + "qkv_proj.weight",
                    torch.cat([checkpoint.get_tensor(source) for source in sources], dim=0),
                )
                consumed.update(sources)
            elif name.endswith(".mlp.gate_proj.weight"):
                prefix = name.removesuffix("gate_proj.weight")
                sources = [prefix + projection for projection in ("gate_proj.weight", "up_proj.weight")]
                _store_weight(
                    output,
                    prefix + "gate_up_proj.weight",
                    torch.cat([checkpoint.get_tensor(source) for source in sources], dim=0),
                )
                consumed.update(sources)
        for name in checkpoint.keys():
            if name in consumed:
                continue
            value = checkpoint.get_tensor(name)
            if not name.endswith(".weight"):
                raise ValueError(f"Unexpected OmniVoice generator tensor: {name}")
            _store_weight(output, name, value)


def _convert_audio_decoder(source: Path, output: dict[str, torch.Tensor], codebooks: int) -> None:
    path = source / "audio_tokenizer" / "model.safetensors"
    with safe_open(path, framework="pt", device="cpu") as checkpoint:
        for index in range(codebooks):
            upstream = f"quantizer.quantizers.{index}"
            destination = f"audio_tokenizer.quantizers.{index}"
            _store_weight(
                output,
                f"{destination}.codebook.weight",
                checkpoint.get_tensor(f"{upstream}.codebook.embed"),
            )
            _store_weight(
                output,
                f"{destination}.project_out.weight",
                checkpoint.get_tensor(f"{upstream}.project_out.weight"),
            )
            _store_vector(
                output,
                f"{destination}.project_out.bias",
                f"{destination}.project_out.bias_scale",
                checkpoint.get_tensor(f"{upstream}.project_out.bias"),
            )

        for module in ("fc2",):
            destination = f"audio_tokenizer.{module}"
            _store_weight(output, f"{destination}.weight", checkpoint.get_tensor(f"{module}.weight"))
            _store_vector(
                output,
                f"{destination}.bias",
                f"{destination}.bias_scale",
                checkpoint.get_tensor(f"{module}.bias"),
            )

        for name in checkpoint.keys():
            if not name.startswith("acoustic_decoder."):
                continue
            destination = f"audio_tokenizer.{name}"
            value = checkpoint.get_tensor(name)
            if name.endswith(".alpha"):
                _store_vector(output, destination, destination + "_scale", value)
            elif name.endswith(".bias"):
                _store_vector(output, destination, destination.removesuffix(".bias") + ".bias_scale", value)
            elif name.endswith(".weight"):
                if ".conv_t1.weight" in name:
                    if value.ndim != 3:
                        raise ValueError(f"Unexpected ConvTranspose1d weight shape: {name} {tuple(value.shape)}")
                    value = value.permute(1, 2, 0).reshape(value.shape[1] * value.shape[2], value.shape[0])
                else:
                    if value.ndim != 3:
                        raise ValueError(f"Unexpected Conv1d weight shape: {name} {tuple(value.shape)}")
                    value = value.reshape(value.shape[0], -1)
                _store_weight(output, destination, value)
            else:
                raise ValueError(f"Unexpected acoustic decoder tensor: {name}")


def _model_document(source: Path, license_files: list[str] | None = None) -> dict:
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    codec = json.loads((source / "audio_tokenizer" / "config.json").read_text(encoding="utf-8"))
    acoustic = codec["acoustic_model_config"]
    latent_width = acoustic["hidden_size"] + codec["semantic_model_config"]["hidden_size"]
    ratios = acoustic["upsampling_ratios"]
    sample_rate = codec["sample_rate"]
    frame_rate = sample_rate // math.prod(ratios)
    return {
        "format_version": 1,
        "task": "tts",
        "license_files": license_files or [],
        "weights_file": "model.safetensors",
        "tokenizer_file": "tokenizer.json",
        "template_file": "synthesis_template.jinja",
        "model": {
            "type": "omnivoice",
            "audio_vocab_size": config["audio_vocab_size"],
            "audio_mask_id": config["audio_mask_id"],
            "num_audio_codebook": config["num_audio_codebook"],
            "llm_config": config["llm_config"],
            "codec_config": {
                "codebook_size": codec["codebook_size"],
                "codebook_dim": codec["codebook_dim"],
                "hidden_size": latent_width,
                "acoustic_hidden_size": acoustic["hidden_size"],
                "decoder_hidden_size": acoustic["decoder_hidden_size"],
                "upsampling_ratios": ratios,
                "sample_rate": sample_rate,
                "frame_rate": frame_rate,
            },
        },
        "synthesis": {
            "steps": 32,
            "guidance_scale": 2.0,
            "time_shift": 0.1,
            "layer_penalty": 5.0,
            "position_temperature": 5.0,
            "auto_duration": {
                "reference_text": "Nice to meet you.",
                "reference_tokens": 25,
                "low_threshold": 50,
                "boost_strength": 3,
            },
            "voice_attributes": {
                "gender": ["male", "female"],
                "age": ["child", "teenager", "young adult", "middle-aged", "elderly"],
                "pitch": ["very low", "low", "moderate", "high", "very high"],
                "style": ["whisper"],
                "accent": [
                    "american",
                    "british",
                    "australian",
                    "chinese",
                    "canadian",
                    "indian",
                    "korean",
                    "portuguese",
                    "russian",
                    "japanese",
                ],
                "dialect": [
                    "河南话",
                    "陕西话",
                    "四川话",
                    "贵州话",
                    "云南话",
                    "桂林话",
                    "济南话",
                    "石家庄话",
                    "甘肃话",
                    "宁夏话",
                    "青岛话",
                    "东北话",
                ],
            },
        },
        "quantization": {
            "format": "signed-int8",
            "matrix_scheme": "symmetric-per-output-channel",
            "vector_scheme": "symmetric-per-tensor",
            "scope": "all retained learned parameters",
            "omitted": "voice-cloning encoder and unused training state",
        },
    }


def convert(source: Path, destination: Path, *, force: bool = False) -> Path:
    source = source.expanduser().resolve()
    destination = destination.expanduser().resolve()
    required = (
        source / "config.json",
        source / "model.safetensors",
        source / "tokenizer.json",
        source / "audio_tokenizer" / "config.json",
        source / "audio_tokenizer" / "model.safetensors",
        source / "audio_tokenizer" / "LICENSE",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ValueError("Missing OmniVoice files: " + ", ".join(missing))
    if destination.exists() and any(destination.iterdir()) and not force:
        raise ValueError(f"Destination is not empty: {destination}; pass --force to replace generated files")
    destination.mkdir(parents=True, exist_ok=True)

    notice = """OmniVoice converted model package

OmniVoice pretrained model weights are redistributed under CC BY-NC 4.0.

Built with Higgs Materials licensed from Boson AI USA, Inc., Copyright Boson AI USA, Inc., All Rights Reserved and Meta Llama 3 licensed under the Meta Llama 3 Community License, Copyright Meta Platforms, Inc., All Rights Reserved.

Meta Llama 3 is licensed under the Meta Llama 3 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved.
Boson Higgs Audio 2 is licensed under the Boson Community License, Copyright © Boson AI USA, Inc. All Rights Reserved.

Sources:
- https://huggingface.co/k2-fsa/OmniVoice
- https://huggingface.co/eustlb/higgs-audio-v2-tokenizer
"""
    license_files = write_license_bundle(
        destination,
        {
            "OMNIVOICE-CC-BY-NC-4.0.txt": "CC-BY-NC-4.0.txt",
            "META-LLAMA-3-LICENSE.txt": "META-LLAMA-3-LICENSE.txt",
            "META-LLAMA-3-USE-POLICY.md": "META-LLAMA-3-USE-POLICY.md",
        },
        notice,
        {"HIGGS-AUDIO-2-LICENSE.txt": source / "audio_tokenizer" / "LICENSE"},
    )
    document = _model_document(source, license_files)
    tensors: dict[str, torch.Tensor] = {}
    _convert_generator(source, tensors)
    _convert_audio_decoder(source, tensors, document["model"]["num_audio_codebook"])
    for name, value in tensors.items():
        if name.endswith(".scale") or name.endswith("_scale"):
            if value.dtype != torch.float32:
                raise ValueError(f"Quantization scale is not FP32: {name}")
        elif value.dtype != torch.int8:
            raise ValueError(f"Retained parameter is not INT8: {name}")

    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination, prefix=".model.", suffix=".safetensors", delete=False) as file:
            temporary = Path(file.name)
        save_file(tensors, temporary)
        os.replace(temporary, destination / "model.safetensors")
        (destination / "model.safetensors").chmod(0o644)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

    (destination / "model.yaml").write_text(
        yaml.safe_dump(document, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    (destination / "synthesis_template.jinja").write_text(
        "<|lang_start|>{{- language -}}<|lang_end|>"
        "<|instruct_start|>{{- instruction -}}<|instruct_end|>"
        "<|text_start|>{{- text -}}<|text_end|>",
        encoding="utf-8",
    )
    for name in ("tokenizer.json", "tokenizer_config.json", "README.md"):
        path = source / name
        if path.is_file():
            shutil.copy2(path, destination / name)
    metadata = {
        "source": str(source),
        "parameter_tensors": sum(
            not (name.endswith(".scale") or name.endswith("_scale")) for name in tensors
        ),
        "scale_tensors": sum(name.endswith(".scale") or name.endswith("_scale") for name in tensors),
        "stored_parameter_bytes": sum(
            value.numel() * value.element_size()
            for name, value in tensors.items()
            if not (name.endswith(".scale") or name.endswith("_scale"))
        ),
    }
    (destination / "conversion.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="downloaded k2-fsa/OmniVoice snapshot")
    parser.add_argument("destination", type=Path, help="output Kidi package directory")
    parser.add_argument("--force", action="store_true", help="replace generated files in a nonempty destination")
    args = parser.parse_args()
    print(convert(args.source, args.destination, force=args.force))


if __name__ == "__main__":
    main()
