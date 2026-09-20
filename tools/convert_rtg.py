#!/usr/bin/env python3
"""Convert an exported RTG Transformer NMT model into a kidi package."""

from __future__ import annotations

import argparse
import gzip
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Mapping, Tuple

import torch
from ruamel.yaml import YAML
from safetensors.torch import save_file


TensorShape = Tuple[int, ...]
POSITIONAL_BUFFERS = {"src_embed.1.pe", "tgt_embed.1.pe"}
TIED_OUTPUT_WEIGHT = "generator.proj.weight"
TIED_TARGET_WEIGHT = "tgt_embed.0.lut.weight"
EMBEDDING_WEIGHTS = {"src_embed.0.lut.weight", TIED_TARGET_WEIGHT}
WEIGHT_PRECISIONS = ("fp32", "bf16", "int8")
QUANTIZATION_SCALE_SUFFIX = ".scale"


class ConversionError(ValueError):
    """Raised when an RTG export is incompatible with kidi's MVP."""


def expected_tensors(model_args: Mapping[str, object]) -> Dict[str, TensorShape]:
    source_vocab = int(model_args["src_vocab"])
    target_vocab = int(model_args["tgt_vocab"])
    encoder_layers = int(model_args["enc_layers"])
    decoder_layers = int(model_args["dec_layers"])
    hidden_size = int(model_args["hid_size"])
    feed_forward_size = int(model_args["ff_size"])

    result: Dict[str, TensorShape] = {
        "src_embed.0.lut.weight": (source_vocab, hidden_size),
        "src_embed.1.pe": (1, 5000, hidden_size),
        "tgt_embed.0.lut.weight": (target_vocab, hidden_size),
        "tgt_embed.1.pe": (1, 5000, hidden_size),
        TIED_OUTPUT_WEIGHT: (target_vocab, hidden_size),
        "generator.proj.bias": (target_vocab,),
        "encoder.norm.weight": (hidden_size,),
        "encoder.norm.bias": (hidden_size,),
        "decoder.norm.weight": (hidden_size,),
        "decoder.norm.bias": (hidden_size,),
    }

    for layer in range(encoder_layers):
        prefix = f"encoder.layers.{layer}"
        for projection in range(4):
            result[f"{prefix}.self_attn.linears.{projection}.weight"] = (hidden_size, hidden_size)
            result[f"{prefix}.self_attn.linears.{projection}.bias"] = (hidden_size,)
        result[f"{prefix}.feed_forward.w_1.weight"] = (feed_forward_size, hidden_size)
        result[f"{prefix}.feed_forward.w_1.bias"] = (feed_forward_size,)
        result[f"{prefix}.feed_forward.w_2.weight"] = (hidden_size, feed_forward_size)
        result[f"{prefix}.feed_forward.w_2.bias"] = (hidden_size,)
        for sublayer in range(2):
            result[f"{prefix}.sublayer.{sublayer}.norm.weight"] = (hidden_size,)
            result[f"{prefix}.sublayer.{sublayer}.norm.bias"] = (hidden_size,)

    for layer in range(decoder_layers):
        prefix = f"decoder.layers.{layer}"
        for attention in ("self_attn", "src_attn"):
            for projection in range(4):
                result[f"{prefix}.{attention}.linears.{projection}.weight"] = (hidden_size, hidden_size)
                result[f"{prefix}.{attention}.linears.{projection}.bias"] = (hidden_size,)
        result[f"{prefix}.feed_forward.w_1.weight"] = (feed_forward_size, hidden_size)
        result[f"{prefix}.feed_forward.w_1.bias"] = (feed_forward_size,)
        result[f"{prefix}.feed_forward.w_2.weight"] = (hidden_size, feed_forward_size)
        result[f"{prefix}.feed_forward.w_2.bias"] = (hidden_size,)
        for sublayer in range(3):
            result[f"{prefix}.sublayer.{sublayer}.norm.weight"] = (hidden_size,)
            result[f"{prefix}.sublayer.{sublayer}.norm.bias"] = (hidden_size,)
    return result


def validate_model_args(model_args: Mapping[str, object]) -> None:
    required = {
        "src_vocab",
        "tgt_vocab",
        "enc_layers",
        "dec_layers",
        "hid_size",
        "ff_size",
        "n_heads",
        "attn_bias",
        "activation",
        "tied_emb",
    }
    missing = sorted(required - set(model_args))
    if missing:
        raise ConversionError(f"model_args is missing: {', '.join(missing)}")
    if not bool(model_args["attn_bias"]):
        raise ConversionError("kidi's RTG MVP requires attention biases")
    if model_args["activation"] != "gelu":
        raise ConversionError("kidi's RTG MVP supports only GELU")
    if model_args["tied_emb"] != "one-way":
        raise ConversionError("kidi's RTG MVP requires one-way target embedding tying")
    if int(model_args["hid_size"]) % int(model_args["n_heads"]) != 0:
        raise ConversionError("hidden size must be divisible by the number of attention heads")
    if int(model_args.get("self_attn_rel_pos", 0)) != 0:
        raise ConversionError("relative-position attention is not supported")


def validate_state(
    state: Mapping[str, torch.Tensor], model_args: Mapping[str, object]
) -> Dict[str, torch.Tensor]:
    expected = expected_tensors(model_args)
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing tensors: {', '.join(missing)}")
        if unexpected:
            details.append(f"unexpected tensors: {', '.join(unexpected)}")
        raise ConversionError("; ".join(details))

    for name, expected_shape in expected.items():
        tensor = state[name]
        if not isinstance(tensor, torch.Tensor):
            raise ConversionError(f"{name} is not a tensor")
        if tensor.dtype != torch.float32:
            raise ConversionError(f"{name} has dtype {tensor.dtype}; expected float32")
        if tuple(tensor.shape) != expected_shape:
            raise ConversionError(f"{name} has shape {tuple(tensor.shape)}; expected {expected_shape}")
        if not tensor.is_contiguous():
            raise ConversionError(f"{name} is not contiguous")

    target_weight = state[TIED_TARGET_WEIGHT]
    output_weight = state[TIED_OUTPUT_WEIGHT]
    if not torch.equal(target_weight, output_weight):
        raise ConversionError("target embedding and generator projection weights differ")

    exported = {
        name: tensor
        for name, tensor in state.items()
        if name not in POSITIONAL_BUFFERS and name != TIED_OUTPUT_WEIGHT
    }
    storages: Dict[int, str] = {}
    for name, tensor in exported.items():
        storage_pointer = tensor.storage().data_ptr()
        if storage_pointer in storages:
            raise ConversionError(f"unexpected shared storage: {storages[storage_pointer]} and {name}")
        storages[storage_pointer] = name
    return exported


def canonicalize_attention_state(
    state: Mapping[str, torch.Tensor], model_args: Mapping[str, object]
) -> Dict[str, torch.Tensor]:
    result = dict(state)

    def fuse(projections: Tuple[str, ...], destination: str) -> None:
        result[destination + ".weight"] = torch.cat(
            [result.pop(projection + ".weight") for projection in projections], dim=0
        ).contiguous()
        result[destination + ".bias"] = torch.cat(
            [result.pop(projection + ".bias") for projection in projections], dim=0
        ).contiguous()

    def rename(source: str, destination: str) -> None:
        result[destination + ".weight"] = result.pop(source + ".weight")
        result[destination + ".bias"] = result.pop(source + ".bias")

    for layer in range(int(model_args["enc_layers"])):
        prefix = f"encoder.layers.{layer}.self_attn"
        fuse(tuple(f"{prefix}.linears.{projection}" for projection in range(3)), f"{prefix}.qkv")
        rename(f"{prefix}.linears.3", f"{prefix}.out")

    for layer in range(int(model_args["dec_layers"])):
        self_prefix = f"decoder.layers.{layer}.self_attn"
        fuse(tuple(f"{self_prefix}.linears.{projection}" for projection in range(3)), f"{self_prefix}.qkv")
        rename(f"{self_prefix}.linears.3", f"{self_prefix}.out")

        source_prefix = f"decoder.layers.{layer}.src_attn"
        rename(f"{source_prefix}.linears.0", f"{source_prefix}.q")
        fuse((f"{source_prefix}.linears.1", f"{source_prefix}.linears.2"), f"{source_prefix}.kv")
        rename(f"{source_prefix}.linears.3", f"{source_prefix}.out")
    return result


def quantize_per_channel(tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    maximum = tensor.abs().max(dim=1, keepdim=True).values
    scale = maximum / 127.0
    scale = torch.where(maximum == 0, torch.ones_like(scale), scale)
    quantized = torch.round(tensor / scale).clamp(-127, 127).to(torch.int8)
    return quantized, scale.to(torch.float32)


def encode_state(state: Mapping[str, torch.Tensor], precision: str) -> Dict[str, torch.Tensor]:
    result: Dict[str, torch.Tensor] = {}
    target_encoded: torch.Tensor | None = None
    target_scale: torch.Tensor | None = None
    for name, tensor in state.items():
        if tensor.ndim != 2:
            result[name] = tensor
            continue

        if precision == "fp32":
            encoded = tensor
        elif precision == "bf16":
            encoded = tensor.to(torch.bfloat16)
        elif precision == "int8":
            encoded, scale = quantize_per_channel(tensor)
            result[name + QUANTIZATION_SCALE_SUFFIX] = scale.contiguous()
            if name == TIED_TARGET_WEIGHT:
                target_scale = scale
        else:
            raise ConversionError(f"unsupported precision: {precision}")

        result[name] = encoded.contiguous() if name in EMBEDDING_WEIGHTS else encoded.t().contiguous()
        if name == TIED_TARGET_WEIGHT:
            target_encoded = encoded

    if target_encoded is None:
        raise ConversionError("target embedding weight was not encoded")
    if precision == "int8":
        result[TIED_OUTPUT_WEIGHT] = target_encoded.t().contiguous()
        if target_scale is not None:
            result[TIED_OUTPUT_WEIGHT + QUANTIZATION_SCALE_SUFFIX] = target_scale.clone().contiguous()
    return result


def find_checkpoint(model_directory: Path) -> Path:
    checkpoints = sorted((model_directory / "models").glob("*.pkl"))
    if len(checkpoints) != 1:
        raise ConversionError(
            f"expected exactly one exported checkpoint under {model_directory / 'models'}, found {len(checkpoints)}"
        )
    return checkpoints[0]


def find_tokenizer_converter(explicit: Path | None) -> Path:
    candidates = []
    if explicit is not None:
        candidates.append(explicit)
    repository = Path(__file__).resolve().parents[1]
    candidates.extend(
        [
            repository / "third_party/tokenizerspp/tools/nlcodec_to_tokenizer_json.py",
            repository.parent / "tokenizerspp/tools/nlcodec_to_tokenizer_json.py",
        ]
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise ConversionError("NLCodec tokenizer converter was not found; pass --tokenizer-converter")


def convert_tokenizer(converter: Path, source: Path, destination: Path) -> None:
    command = [sys.executable, str(converter), str(source), str(destination)]
    result = subprocess.run(command, text=True, capture_output=True)
    if result.returncode != 0:
        raise ConversionError(f"tokenizer conversion failed for {source}: {result.stderr.strip()}")


def gzip_deterministic(path: Path) -> Path:
    compressed = path.with_suffix(path.suffix + ".gz")
    with path.open("rb") as source, compressed.open("wb") as raw_output:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_output, mtime=0) as output:
            shutil.copyfileobj(source, output)
    path.unlink()
    return compressed


def write_manifest(
    path: Path,
    model_args: Mapping[str, object],
    source_tokenizer: str,
    target_tokenizer: str,
    checkpoint: Mapping[str, object],
) -> None:
    document = {
        "format_version": 1,
        "weights_file": "model.safetensors",
        "tokenizers": {"source": source_tokenizer, "target": target_tokenizer},
        "io": {"input_format": "moses_tokenized", "output_format": "moses_tokenized"},
        "model": {
            "type": "rtg_transformer_nmt",
            "encoder_layers": int(model_args["enc_layers"]),
            "decoder_layers": int(model_args["dec_layers"]),
            "hidden_size": int(model_args["hid_size"]),
            "feed_forward_size": int(model_args["ff_size"]),
            "attention_heads": int(model_args["n_heads"]),
            "source_vocabulary_size": int(model_args["src_vocab"]),
            "target_vocabulary_size": int(model_args["tgt_vocab"]),
            "activation": str(model_args["activation"]),
            "attention_bias": True,
            "tied_embeddings": "one-way",
            "layer_norm_epsilon": 1.0e-5,
            "position_encoding": "sinusoidal",
            "maximum_position": 5000,
            "source_tokens": 160,
        },
        "decode": {
            "beam_size": 4,
            "maximum_extra_tokens": 50,
            "length_penalty": 0.6,
        },
        "provenance": {
            "rtg_model_type": str(checkpoint["model_type"]),
            "training_step": int(checkpoint["step"]),
            "averaged_checkpoints": int(checkpoint.get("num_checkpts", 1)),
        },
    }
    yaml = YAML()
    yaml.default_flow_style = False
    with path.open("w", encoding="utf-8") as output:
        yaml.dump(document, output)


def convert(args: argparse.Namespace) -> Path:
    source = args.model_directory.resolve()
    if not source.is_dir():
        raise ConversionError(f"RTG model directory does not exist: {source}")
    for relative in ("conf.yml", "_EXPORTED", "data/nlcodec.src.model", "data/nlcodec.tgt.model"):
        if not (source / relative).is_file():
            raise ConversionError(f"RTG export is missing {relative}")

    destination = args.output.resolve()
    if destination.exists():
        raise ConversionError(f"output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))

    try:
        yaml = YAML(typ="safe")
        config = yaml.load(source / "conf.yml")
        model_args = dict(config["model_args"])
        validate_model_args(model_args)

        checkpoint_path = find_checkpoint(source)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if checkpoint.get("model_type") != "tfmnmt":
            raise ConversionError(f"unsupported RTG model type: {checkpoint.get('model_type')!r}")
        checkpoint_args = checkpoint.get("model_args")
        if checkpoint_args and dict(checkpoint_args) != model_args:
            raise ConversionError("checkpoint model_args differs from conf.yml")
        state = checkpoint.get("model_state")
        if not isinstance(state, Mapping):
            raise ConversionError("checkpoint has no model_state mapping")

        canonical = canonicalize_attention_state(validate_state(state, model_args), model_args)
        exported = encode_state(canonical, args.precision)
        save_file(exported, staging / "model.safetensors", metadata={"format": "kidi-rtg-v1"})

        converter = find_tokenizer_converter(args.tokenizer_converter)
        source_tokenizer = staging / "tokenizer.src.json"
        target_tokenizer = staging / "tokenizer.tgt.json"
        convert_tokenizer(converter, source / "data/nlcodec.src.model", source_tokenizer)
        convert_tokenizer(converter, source / "data/nlcodec.tgt.model", target_tokenizer)
        if args.gzip_tokenizers:
            source_tokenizer = gzip_deterministic(source_tokenizer)
            target_tokenizer = gzip_deterministic(target_tokenizer)

        write_manifest(
            staging / "model.yaml",
            model_args,
            source_tokenizer.name,
            target_tokenizer.name,
            checkpoint,
        )
        os.replace(staging, destination)
        return destination
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_directory", type=Path, help="extracted RTG export directory")
    parser.add_argument("output", type=Path, help="new kidi model package directory")
    parser.add_argument("--tokenizer-converter", type=Path, help="path to nlcodec_to_tokenizer_json.py")
    parser.add_argument("--gzip-tokenizers", action="store_true", help="write deterministic .json.gz tokenizers")
    parser.add_argument(
        "--precision",
        choices=WEIGHT_PRECISIONS,
        default="fp32",
        help="matrix and embedding weight encoding (default: fp32)",
    )
    return parser.parse_args()


def main() -> int:
    try:
        destination = convert(parse_args())
    except (ConversionError, OSError, subprocess.SubprocessError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(f"converted RTG model to {destination}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())