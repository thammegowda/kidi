#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy==2.5.3",
#   "pillow==12.3.0",
#   "safetensors==0.8.0",
#   "torch==2.14.0",
#   "torchvision==0.29.1",
#   "transformers==5.17.0",
# ]
# ///
"""Generate deterministic images and official Gemma 4 vision reference tensors."""

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from safetensors import safe_open
from safetensors.torch import save_file
import torch
import transformers
from torch_intercept import TensorCapture
from transformers import Gemma4TextConfig, Gemma4VisionConfig
from transformers.integrations.gemma_quant import QuantizedLinear
from transformers.models.gemma4 import Gemma4ImageProcessor
from transformers.models.gemma4.modeling_gemma4 import (
    Gemma4MultimodalEmbedder,
    Gemma4VisionModel,
    Gemma4VisionRotaryEmbedding,
    apply_multidimensional_rope,
)


MODEL_ID = "google/gemma-4-E2B-it-qat-mobile-transformers"
REVISION = "dd693ff40353f057ca5f07e945ad867f4afbf2ec"
IMAGE_SPECS = (
    ("square-grid", 1600, 1600),
    ("landscape-shapes", 1536, 864),
    ("portrait-shapes", 864, 1536),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_image(name: str, width: int, height: int) -> Image.Image:
    x = np.linspace(0, 1, width, dtype=np.float32)[None, :]
    y = np.linspace(0, 1, height, dtype=np.float32)[:, None]
    if name == "square-grid":
        cells = ((np.arange(height)[:, None] // 80 + np.arange(width)[None, :] // 80) & 1).astype(np.float32)
        rgb = np.stack(
            (
                np.broadcast_to(0.15 + 0.7 * x, (height, width)),
                np.broadcast_to(0.15 + 0.7 * y, (height, width)),
                0.2 + 0.45 * cells,
            ),
            axis=-1,
        )
    elif name == "landscape-shapes":
        rgb = np.stack(
            (
                np.broadcast_to(0.1 + 0.75 * x, (height, width)),
                np.broadcast_to(0.25 + 0.5 * (1 - y), (height, width)),
                np.broadcast_to(0.2 + 0.55 * x * y, (height, width)),
            ),
            axis=-1,
        )
    else:
        radius = np.sqrt((np.broadcast_to(x, (height, width)) - 0.5) ** 2 +
                         (np.broadcast_to(y, (height, width)) - 0.5) ** 2)
        stripes = ((np.arange(height)[:, None] // 64) & 1).astype(np.float32)
        rgb = np.stack(
            (
                np.clip(1 - 1.5 * radius, 0, 1),
                np.broadcast_to(0.2 + 0.6 * stripes, (height, width)),
                np.broadcast_to(0.15 + 0.75 * y, (height, width)),
            ),
            axis=-1,
        )
    image = Image.fromarray(np.clip(np.rint(rgb * 255), 0, 255).astype(np.uint8), "RGB")
    draw = ImageDraw.Draw(image)
    if name == "landscape-shapes":
        draw.rectangle((width // 12, height // 5, width // 3, 4 * height // 5), fill=(245, 45, 45))
        draw.ellipse((7 * width // 12, height // 6, 11 * width // 12, 5 * height // 6), fill=(35, 85, 235))
    elif name == "portrait-shapes":
        draw.ellipse((width // 8, height // 12, 7 * width // 8, 5 * height // 12), fill=(35, 210, 85))
        draw.polygon(
            ((width // 2, height // 2), (width // 8, 11 * height // 12), (7 * width // 8, 11 * height // 12)),
            fill=(235, 175, 30),
        )
    return image


def resolve_checkpoint(checkpoint: Path | None) -> Path:
    if checkpoint is not None:
        return checkpoint
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            MODEL_ID,
            revision=REVISION,
            allow_patterns=["config.json", "model.safetensors", "preprocessor_config.json", "processor_config.json"],
        )
    )


def load_models(checkpoint: Path):
    document = json.loads((checkpoint / "config.json").read_text())
    # Kidi implements the official eager equations. Inputs are stripped of processor padding below, so every query has
    # valid keys and the eager reference does not encounter all-masked rows.
    vision_config = Gemma4VisionConfig(**document["vision_config"], attn_implementation="eager")
    text_config = Gemma4TextConfig(**document["text_config"])
    with torch.device("meta"):
        tower = Gemma4VisionModel(vision_config)
        for name, module in list(tower.named_modules()):
            if isinstance(module, torch.nn.Linear) and not name.startswith("patch_embedder"):
                tower.set_submodule(name, QuantizedLinear(module.in_features, module.out_features, num_bits=8))
        projector = Gemma4MultimodalEmbedder(vision_config, text_config)

    tower_names = set(tower.state_dict())
    projector_names = set(projector.state_dict())
    tower_state = {}
    projector_state = {}
    with safe_open(checkpoint / "model.safetensors", framework="pt") as source:
        for key in source.keys():
            if key.startswith("model.vision_tower."):
                name = key.removeprefix("model.vision_tower.")
                if name in tower_names:
                    tower_state[name] = source.get_tensor(key)
            elif key.startswith("model.embed_vision."):
                name = key.removeprefix("model.embed_vision.")
                if name in projector_names:
                    projector_state[name] = source.get_tensor(key)
    tower.load_state_dict(tower_state, strict=True, assign=True)
    projector.load_state_dict(projector_state, strict=True, assign=True)
    # Non-persistent rotary buffers are absent from the checkpoint and remain on the meta device after assign-loading.
    tower.encoder.rotary_emb = Gemma4VisionRotaryEmbedding(vision_config)
    return tower.float().eval(), projector.float().eval(), document


def capture_vision(tower, projector, patches, position_ids, stage_layers: int = 3):
    stages = {}
    with TensorCapture(stages) as capture:
        for index, layer in enumerate(tower.encoder.layers):
            capture.output(layer, f"block_{index:02d}")
        for index, layer in enumerate(tower.encoder.layers[:stage_layers]):
            prefix = f"layer_{index:02d}_"
            capture.input(layer, prefix + "input")
            capture.output(layer.input_layernorm, prefix + "input_norm")
            capture.output(layer.self_attn.q_proj, prefix + "q_proj")
            capture.output(layer.self_attn.q_norm, prefix + "q_norm")
            capture.output(layer.self_attn.k_proj, prefix + "k_proj")
            capture.output(layer.self_attn.k_norm, prefix + "k_norm")
            capture.output(layer.self_attn.v_proj, prefix + "v_proj")
            capture.output(layer.self_attn.v_norm, prefix + "v_norm")
            capture.input(layer.self_attn.o_proj, prefix + "attended")
            capture.output(layer.self_attn.o_proj, prefix + "o_proj")
            capture.output(layer.post_attention_layernorm, prefix + "post_attention_norm")
            capture.input(layer.pre_feedforward_layernorm, prefix + "attention_residual")
            capture.output(layer.pre_feedforward_layernorm, prefix + "pre_ffn_norm")
            capture.output(layer.mlp.gate_proj, prefix + "gate_proj")
            capture.output(layer.mlp.up_proj, prefix + "up_proj")
            capture.input(layer.mlp.down_proj, prefix + "mlp_hidden")
            capture.output(layer.mlp.down_proj, prefix + "down_proj")
            capture.output(layer.mlp, prefix + "mlp")
            capture.output(layer.post_feedforward_layernorm, prefix + "post_ffn_norm")
        with torch.no_grad():
            padding_positions = torch.zeros(position_ids.shape[:-1], dtype=torch.bool)
            inputs_embeds = tower.patch_embedder(patches, position_ids, padding_positions)
            cos, sin = tower.encoder.rotary_emb(inputs_embeds, position_ids)
            encoded = tower.encoder(
                inputs_embeds=inputs_embeds,
                attention_mask=None,
                pixel_position_ids=position_ids,
            ).last_hidden_state
            pooled, pooler_mask = tower.pooler(
                hidden_states=encoded,
                pixel_position_ids=position_ids,
                padding_positions=padding_positions,
                output_length=patches.shape[1] // (tower.config.pooling_kernel_size**2),
            )
            pooled = pooled[pooler_mask]
            if tower.config.standardize:
                pooled = (pooled - tower.std_bias.float()) * tower.std_scale.float()
            pooled = pooled.to(inputs_embeds.dtype).float()
            pooled = pooled.reshape(1, -1, pooled.shape[-1]).contiguous()
            visual_embeddings = projector(pooled).float().contiguous()

    block_names = [f"block_{index:02d}" for index in range(tower.config.num_hidden_layers)]
    if any(name not in stages for name in block_names):
        raise RuntimeError("vision capture missed one or more encoder blocks")

    quarter = cos.shape[-1] // 4
    stages["layer_00_rope_cos"] = torch.stack(
        (cos[:, :, :quarter], cos[:, :, 2 * quarter : 3 * quarter])
    ).squeeze(1).unsqueeze(2).float().contiguous()
    stages["layer_00_rope_sin"] = torch.stack(
        (sin[:, :, :quarter], sin[:, :, 2 * quarter : 3 * quarter])
    ).squeeze(1).unsqueeze(2).float().contiguous()
    for index in range(min(stage_layers, tower.config.num_hidden_layers)):
        prefix = f"layer_{index:02d}_"
        query = apply_multidimensional_rope(
            stages[prefix + "q_norm"], cos, sin, position_ids
        ).contiguous()
        key = apply_multidimensional_rope(
            stages[prefix + "k_norm"], cos, sin, position_ids
        ).contiguous()
        stages[prefix + "q_rope"] = query
        stages[prefix + "k_rope"] = key

    return {
        "patches": patches,
        "position_ids": position_ids,
        **stages,
        "encoder_hidden": stages[block_names[-1]].clone(),
        "pooled_hidden": pooled,
        "visual_embeddings": visual_embeddings,
    }


def tensors_sha256(tensors) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(tensors.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def generate_references(
    checkpoint: Path,
    output: Path,
    threads: int,
    verify_determinism: bool,
    determinism_threads: tuple[int, ...],
    determinism_repeats: int,
    image_names: tuple[str, ...],
    stage_layers: int,
) -> None:
    torch.manual_seed(173)
    torch.set_num_threads(threads)
    images_dir = output / "images"
    references_dir = output / "references"
    images_dir.mkdir(parents=True, exist_ok=True)
    references_dir.mkdir(parents=True, exist_ok=True)
    processor = Gemma4ImageProcessor.from_pretrained(checkpoint, local_files_only=True)
    tower, projector, config = load_models(checkpoint)
    manifest = {
        "model_id": MODEL_ID,
        "revision": REVISION,
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "torch_config": torch.__config__.show(),
        "platform": platform.platform(),
        "threads": threads,
        "attention_implementation": "eager",
        "tower_input": "official_processor_valid_patches_without_padding",
        "reference_format_version": 2,
        "stage_layers": stage_layers,
        "config_sha256": sha256(checkpoint / "config.json"),
        "model_sha256": sha256(checkpoint / "model.safetensors"),
        "images": [],
    }

    for name, width, height in IMAGE_SPECS:
        if image_names and name not in image_names:
            continue
        image_path = images_dir / f"{name}.png"
        make_image(name, width, height).save(image_path, format="PNG", compress_level=6)
        image = Image.open(image_path).convert("RGB")
        inputs = processor.preprocess(image, return_tensors="pt")
        padded_patches = inputs["pixel_values"].float()
        padded_positions = inputs["image_position_ids"].to(torch.int64)
        valid = ~(padded_positions == -1).all(dim=-1)
        patches = padded_patches[valid].reshape(1, -1, padded_patches.shape[-1]).contiguous()
        position_ids = padded_positions[valid].reshape(1, -1, 2).contiguous()
        tensors = capture_vision(tower, projector, patches, position_ids, stage_layers)
        determinism_sha256 = None
        if verify_determinism:
            determinism_sha256 = tensors_sha256(tensors)
            for candidate_threads in determinism_threads:
                torch.set_num_threads(candidate_threads)
                for repeat in range(determinism_repeats):
                    candidate = capture_vision(tower, projector, patches, position_ids, stage_layers)
                    for tensor_name, expected in tensors.items():
                        if not torch.equal(candidate[tensor_name], expected):
                            delta = (candidate[tensor_name].float() - expected.float()).abs()
                            raise RuntimeError(
                                f"official output changed for {name}/{tensor_name} at "
                                f"threads={candidate_threads} repeat={repeat}: "
                                f"mismatches={(candidate[tensor_name] != expected).sum().item()} "
                                f"max={delta.max().item()}"
                            )
                    if tensors_sha256(candidate) != determinism_sha256:
                        raise RuntimeError("official tensor hash changed despite elementwise equality")
            torch.set_num_threads(threads)
        reference_path = references_dir / f"{name}.safetensors"
        save_file(tensors, reference_path)
        manifest["images"].append(
            {
                "name": name,
                "file": str(image_path.relative_to(output)),
                "reference": str(reference_path.relative_to(output)),
                "source_width": width,
                "source_height": height,
                "patches": patches.shape[1],
                "patch_width": int(round(patches.shape[1] ** 0.5)) if width == height else None,
                "visual_tokens": tensors["visual_embeddings"].shape[1],
                "image_sha256": sha256(image_path),
                "reference_sha256": sha256(reference_path),
                "tensor_sha256": tensors_sha256(tensors),
                "determinism_sha256": determinism_sha256,
            }
        )
        print(
            f"{name}: patches={tuple(patches.shape)} pooled={tuple(tensors['pooled_hidden'].shape)} "
            f"embeddings={tuple(tensors['visual_embeddings'].shape)}"
        )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/android/.cache/vision-suite"))
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--verify-determinism", action="store_true")
    parser.add_argument("--determinism-threads", default="1,2,4")
    parser.add_argument("--determinism-repeats", type=int, default=2)
    parser.add_argument("--image", action="append", choices=[spec[0] for spec in IMAGE_SPECS])
    parser.add_argument("--stage-layers", type=int, default=3)
    args = parser.parse_args()
    determinism_threads = tuple(dict.fromkeys(int(value) for value in args.determinism_threads.split(",")))
    if args.threads <= 0 or args.determinism_repeats <= 0 or not 0 <= args.stage_layers <= 16 or not determinism_threads or any(
        value <= 0 for value in determinism_threads
    ):
        parser.error("thread counts/repeats must be positive and stage layers must be from 0 to 16")
    checkpoint = resolve_checkpoint(args.checkpoint)
    generate_references(
        checkpoint,
        args.output,
        args.threads,
        args.verify_determinism,
        determinism_threads,
        args.determinism_repeats,
        tuple(dict.fromkeys(args.image or ())),
        args.stage_layers,
    )


if __name__ == "__main__":
    main()
