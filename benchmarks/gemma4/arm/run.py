"""One matched-weight, CPU-only Gemma 4 E2B comparison with ExecuTorch/XNNPACK/KleidiAI."""

import argparse
from datetime import datetime, timezone
from functools import partial
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
from statistics import median
import subprocess
import sys
import tempfile
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
EXECUTORCH_VERSION = "1.5.1"
EXECUTORCH_COMMIT = "3b60683923245cf472b7323426920e15623ba361"
PROMPT = "Explain how binary search works in at least 300 words, including an example and its time complexity."
PROFILES = {
    "int4-g128": {
        "bits": 4,
        "group_size": 128,
        "precision": {
            "linear_weights": "symmetric INT4, group_size=128, FP32 scales, zero point 0",
            "weight_range": [-7, 7],
            "weight_rounding": "nearest, ties away from zero",
            "activations": "dynamic asymmetric INT8, per-token",
            "embeddings": "symmetric INT4 per row; per-layer scales duplicated without requantization",
            "attention_and_kv_cache": "FP32",
            "unquantized": ["normalizations", "layer scalars"],
        },
    },
    "int8": {
        "bits": 8,
        "group_size": 0,
        "precision": {
            "linear_weights": "symmetric INT8, per-output-channel, FP32 scales, zero point 0",
            "weight_range": [-127, 127],
            "weight_rounding": "nearest, ties away from zero",
            "activations": "dynamic asymmetric INT8, per-token",
            "embeddings": "symmetric INT8 per row; per-layer scales duplicated without requantization",
            "attention_and_kv_cache": "FP32",
            "unquantized": ["normalizations", "layer scalars"],
        },
    },
}
PRECISION = PROFILES["int4-g128"]["precision"]


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def pack_weight(weight, bits=4, group_size=0, chunk_rows=128):
    import torch

    if bits not in (4, 8):
        raise ValueError("weight packing supports 4-bit or 8-bit quantization")
    group = weight.shape[1] if weight.ndim == 2 and not group_size else group_size
    if weight.ndim != 2 or group <= 0 or (bits == 4 and group % 2) or weight.shape[1] % group:
        raise ValueError("weight packing requires a 2D matrix divisible by group_size")
    groups = weight.shape[1] // group
    limit = 7 if bits == 4 else 127
    values = torch.empty(
        (weight.shape[0], weight.shape[1] // 2 if bits == 4 else weight.shape[1]),
        dtype=torch.uint8,
    )
    scales = torch.empty((weight.shape[0], groups), dtype=torch.float32)
    for start in range(0, weight.shape[0], chunk_rows):
        block = weight[start:start + chunk_rows].float()
        if not torch.isfinite(block).all():
            raise ValueError("cannot quantize non-finite weights")
        grouped = block.view(block.shape[0], groups, group)
        maximum = grouped.abs().amax(dim=2)
        scale = torch.where(maximum == 0, torch.ones_like(maximum), maximum / limit)
        if (scale == 0).any():
            raise ValueError("quantization scale underflow")
        quotient = grouped / scale.unsqueeze(-1)
        integers = quotient.trunc()
        integers += ((quotient - integers).abs() >= 0.5) * quotient.sign()
        integers = integers.clamp(-limit, limit).to(torch.int8).view(block.shape)
        if bits == 4:
            nibbles = integers.to(torch.uint8) & 15
            values[start:start + block.shape[0]] = nibbles[:, 0::2] | (nibbles[:, 1::2] << 4)
        else:
            values[start:start + block.shape[0]] = integers.view(torch.uint8)
        scales[start:start + block.shape[0]] = scale
    return values, scales


def pack_int4(weight, group_size=0, chunk_rows=128):
    return pack_weight(weight, bits=4, group_size=group_size, chunk_rows=chunk_rows)


def pack_int4_asym_clip(weight, group_size=128, chunk_rows=128, asymmetric=True):
    import torch

    group = weight.shape[1] if weight.ndim == 2 and not group_size else group_size
    if weight.ndim != 2 or group <= 0 or group % 2 or weight.shape[1] % group:
        raise ValueError("asymmetric 4-bit packing requires a 2D matrix divisible by group_size")
    groups = weight.shape[1] // group
    values = torch.empty((weight.shape[0], weight.shape[1] // 2), dtype=torch.uint8)
    scales = torch.empty((weight.shape[0], groups), dtype=torch.float32)
    zeros = torch.empty((weight.shape[0], groups), dtype=torch.float32) if asymmetric else None
    fractions = (1.00, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50)
    for start in range(0, weight.shape[0], chunk_rows):
        block = weight[start:start + chunk_rows].float()
        if not torch.isfinite(block).all():
            raise ValueError("cannot quantize non-finite weights")
        grouped = block.view(block.shape[0], groups, group)
        if asymmetric:
            mn = grouped.amin(dim=2, keepdim=True)
            mx = grouped.amax(dim=2, keepdim=True)
            center = 0.5 * (mx + mn)
            half = 0.5 * (mx - mn)
            best_sc = None
            best_zp = None
            best_mse = None
            for frac in fractions:
                lo = center - frac * half
                sc = ((2.0 * frac * half) / 15.0).clamp_min(1e-8)
                zp = (-lo / sc).round().clamp(0.0, 15.0)
                q = (grouped / sc + zp).round().clamp(0.0, 15.0)
                mse = (grouped - (q - zp) * sc).pow(2).mean(dim=2, keepdim=True)
                if best_mse is None:
                    best_sc, best_zp, best_mse = sc, zp, mse
                else:
                    better = mse < best_mse
                    best_sc = torch.where(better, sc, best_sc)
                    best_zp = torch.where(better, zp, best_zp)
                    best_mse = torch.where(better, mse, best_mse)
            q_int = (grouped / best_sc + best_zp).round().clamp(0.0, 15.0).to(torch.int8) - 8
            nibbles = q_int.view(block.shape).to(torch.uint8) & 15
            values[start:start + block.shape[0]] = nibbles[:, 0::2] | (nibbles[:, 1::2] << 4)
            scales[start:start + block.shape[0]] = best_sc.squeeze(-1)
            zeros[start:start + block.shape[0]] = ((8.0 - best_zp) * best_sc).squeeze(-1)
        else:
            amax = grouped.abs().amax(dim=2, keepdim=True)
            best_sc = None
            best_mse = None
            for frac in fractions:
                sc = ((frac * amax) / 7.5).clamp_min(1e-8)
                q = (grouped / sc).round().clamp(-8.0, 7.0)
                mse = (grouped - q * sc).pow(2).mean(dim=2, keepdim=True)
                if best_mse is None:
                    best_sc, best_mse = sc, mse
                else:
                    better = mse < best_mse
                    best_sc = torch.where(better, sc, best_sc)
                    best_mse = torch.where(better, mse, best_mse)
            q_int = (grouped / best_sc).round().clamp(-8.0, 7.0).to(torch.int8)
            nibbles = q_int.view(block.shape).to(torch.uint8) & 15
            values[start:start + block.shape[0]] = nibbles[:, 0::2] | (nibbles[:, 1::2] << 4)
            scales[start:start + block.shape[0]] = best_sc.squeeze(-1)
    return (values, scales, zeros) if asymmetric else (values, scales)


def pack_int8(weight, group_size=0, chunk_rows=128):
    return pack_weight(weight, bits=8, group_size=group_size, chunk_rows=chunk_rows)


def unpack_int4(values):
    import torch

    result = torch.empty((values.shape[0], values.shape[1] * 2), dtype=torch.int8)
    result[:, 0::2] = ((values & 15).to(torch.int8) ^ 8) - 8
    result[:, 1::2] = ((values >> 4).to(torch.int8) ^ 8) - 8
    return result


def embedding_layout(values):
    return ((values << 4) | (values >> 4)) ^ 0x88


def consolidate_weights(directory, filenames):
    from safetensors.torch import load_file, save_file

    destination = directory / "model.safetensors"
    if destination.exists() or not filenames:
        raise ValueError("expected generated shards and no existing consolidated checkpoint")
    state = {}
    for filename in filenames:
        if Path(filename).name != filename or not filename.startswith("model-"):
            raise ValueError("invalid generated shard filename")
        shard = load_file(str(directory / filename))
        if state.keys() & shard.keys():
            raise ValueError("duplicate checkpoint tensor")
        state.update(shard)
    save_file(state, str(destination))
    del state
    gc.collect()
    for filename in filenames:
        (directory / filename).unlink()
    index = directory / "model.safetensors.index.json"
    if index.exists():
        index.unlink()


def cache_calibration(layer_count, shared_layers):
    import torch

    return {f"model.language_model.layers.{layer}.self_attn.{kind}_cache_scale": torch.tensor(0.)
            for layer in range(layer_count - shared_layers) for kind in ("k", "v")}


def records_from(text):
    return [json.loads(line) for line in text.splitlines() if line.startswith("{")]


def summarize(records, protocol):
    runtime = [row for row in records if row.get("stage") == "runtime"]
    if len(runtime) != 1 or runtime[0].get("device") != "cpu:0" or runtime[0].get("threads") != protocol["threads"]:
        raise ValueError("expected one runtime on the requested CPU/thread count")
    rows = [row for row in records if row.get("stage") == "generation"]
    if len(rows) != protocol["warmups"] + protocol["repeats"]:
        raise ValueError("missing warmup or measured requests")
    for index, row in enumerate(rows):
        if row["iteration"] != index or row["warmup"] != (index < protocol["warmups"]):
            raise ValueError("invalid warmup sequence")
        if (row["input_token_ids"] != protocol["input_token_ids"] or
                row["prompt_tokens"] != len(protocol["input_token_ids"]) or
                row["output_tokens"] != protocol["output_tokens"] or
                row["decode_tokens"] != protocol["output_tokens"] - 1 or
                len(row["token_ids"]) != protocol["output_tokens"] or row.get("reused_prompt_tokens", 0)):
            raise ValueError("prompt, output count or independent-request contract differs")
        if row["token_ids"] != rows[0]["token_ids"]:
            raise ValueError("output token IDs changed between repetitions")
        for key in ("first_token_seconds", "decode_seconds", "prefill_through_first_token_tps",
                    "decode_interval_tps", "peak_rss_kib"):
            if not math.isfinite(row[key]) or row[key] <= 0:
                raise ValueError(f"invalid measurement: {key}")
        if (not math.isclose(row["prefill_through_first_token_tps"],
                             row["prompt_tokens"] / row["first_token_seconds"], rel_tol=1e-5) or
                not math.isclose(row["decode_interval_tps"],
                                 row["decode_tokens"] / row["decode_seconds"], rel_tol=1e-5)):
            raise ValueError("reported throughput does not match measured token counts and intervals")
    measured = rows[protocol["warmups"]:]
    return {
        "token_ids": rows[0]["token_ids"],
        "prefill_tps": median(row["prefill_through_first_token_tps"] for row in measured),
        "decode_tps": median(row["decode_interval_tps"] for row in measured),
        "ttft_ms": median(row["first_token_seconds"] * 1000 for row in measured),
        "peak_rss_gib": max(row["peak_rss_kib"] for row in measured) / 2**20,
        "decode_range": [min(row["decode_interval_tps"] for row in measured),
                         max(row["decode_interval_tps"] for row in measured)],
    }


def compare(summaries):
    kidi_ids, arm_ids = summaries["kidi"]["token_ids"], summaries["arm"]["token_ids"]
    if len(kidi_ids) != len(arm_ids):
        raise ValueError("runtime output token budgets differ")
    first_difference = next((index for index, pair in enumerate(zip(kidi_ids, arm_ids))
                             if pair[0] != pair[1]), None)
    return {
        "speedup": {metric: summaries["kidi"][metric] / summaries["arm"][metric]
                    for metric in ("prefill_tps", "decode_tps")},
        "outputs": {"identical": first_difference is None,
                    "common_prefix_tokens": len(kidi_ids) if first_difference is None else first_difference,
                    "first_divergent_token": None if first_difference is None else first_difference + 1},
    }


LANG_NAMES = {"ja": "Japanese", "uk": "Ukrainian", "de": "German"}


def clean_translation_text(text: str) -> str:
    if "<channel|>" in text:
        text = text.split("<channel|>")[-1]
    for stop_marker in ("<turn|>", "<eos>"):
        if stop_marker in text:
            text = text.split(stop_marker)[0]
    return " ".join(line.strip() for line in text.strip().splitlines() if line.strip())


def build_eval_protocol(work: Path, tokenizer, num_lines: int, langs: str = "ja,uk,de",
                        seed: int = 42, max_output_tokens: int = 192,
                        context_tokens: int = 1024, threads: int = 4) -> dict:
    sacrebleu_cache = work.parent / "sacrebleu"
    os.environ.setdefault("SACREBLEU", str(sacrebleu_cache))
    import sacrebleu

    lang_list = [lang.strip() for lang in langs.split(",") if lang.strip()]
    dataset = sacrebleu.DATASETS["wmt23"]
    by_lang = {}
    for lang in lang_list:
        pair = f"en-{lang}"
        english_lines = Path(dataset.get_source_file(pair)).read_text(encoding="utf-8").splitlines()
        foreign_lines = Path(dataset.get_reference_files(pair)[0]).read_text(encoding="utf-8").splitlines()
        by_lang[lang] = [
            {"lang": lang, "source": src.strip(), "reference": ref.strip()}
            for src, ref in zip(foreign_lines, english_lines)
            if src.strip() and ref.strip()
        ]
    interleaved = []
    max_len = max(len(rows) for rows in by_lang.values())
    for idx in range(max_len):
        for lang in lang_list:
            if idx < len(by_lang[lang]):
                interleaved.append(by_lang[lang][idx])
    rng = random.Random(seed)
    rng.shuffle(interleaved)
    prompts = []
    input_token_ids_list = []
    selected = []
    input_jsonl_lines = []
    ref_jsonl_lines = []
    for row in interleaved:
        if len(selected) >= num_lines:
            break
        lang_name = LANG_NAMES.get(row["lang"], row["lang"])
        user_prompt = (
            f"Translate the following {lang_name} text fully into English. "
            "Output only the translation, with no explanations, alternatives, notes, labels, or extra details.\n\n"
            f"{row['source']}"
        )
        serialized = "<bos><|turn>user\n" + user_prompt + "<turn|>\n<|turn>model\n"
        ids = tokenizer.encode(serialized, add_special_tokens=False).ids
        if len(ids) + max_output_tokens > context_tokens:
            continue
        idx = len(selected)
        selected.append(row)
        prompts.append(serialized)
        input_token_ids_list.append(ids)
        input_jsonl_lines.append(json.dumps({
            "id": idx,
            "lang": row["lang"],
            "source": row["source"],
            "prompt": serialized,
            "input_token_ids": ids,
        }, ensure_ascii=False))
        ref_jsonl_lines.append(json.dumps({
            "id": idx,
            "lang": row["lang"],
            "source": row["source"],
            "reference": row["reference"],
        }, ensure_ascii=False))
    (work / "eval_input.jsonl").write_text("\n".join(input_jsonl_lines) + "\n", encoding="utf-8")
    (work / "eval_references.jsonl").write_text("\n".join(ref_jsonl_lines) + "\n", encoding="utf-8")
    return {
        "mode": "eval",
        "task": "wmt23",
        "langs": lang_list,
        "seed": seed,
        "items": len(selected),
        "prompts": prompts,
        "input_token_ids_list": input_token_ids_list,
        "sources": [row["source"] for row in selected],
        "references": [row["reference"] for row in selected],
        "item_langs": [row["lang"] for row in selected],
        "max_output_tokens": max_output_tokens,
        "context_tokens": context_tokens,
        "warmups": 1,
        "threads": threads,
    }


def summarize_eval(records: list[dict], eval_protocol: dict, tokenizer) -> dict:
    from sacrebleu.metrics import CHRF

    eval_rows = [row for row in records if row.get("stage") == "eval"]
    if len(eval_rows) != 1:
        raise ValueError("expected one eval record")
    row = eval_rows[0]
    raw_token_ids = row["token_ids"]
    if len(raw_token_ids) != eval_protocol["items"]:
        raise ValueError(f"eval item count mismatch: {len(raw_token_ids)} != {eval_protocol['items']}")
    eval_seconds = float(row["eval_seconds"])
    if not math.isfinite(eval_seconds) or eval_seconds <= 0:
        raise ValueError("invalid eval_seconds")
    hypotheses = []
    clean_ids_list = []
    for ids in raw_token_ids:
        clean = list(ids)
        for stop_id in (1, 106):
            if stop_id in clean:
                clean = clean[:clean.index(stop_id)]
        clean_ids_list.append(clean)
        decoded = tokenizer.decode(clean, skip_special_tokens=True)
        hypotheses.append(clean_translation_text(decoded))
    references = eval_protocol["references"]
    chrf_score = float(CHRF().corpus_score(hypotheses, [references]).score)
    total_output_tokens = sum(len(ids) for ids in clean_ids_list)
    return {
        "task": f"{eval_protocol['task']}:{','.join(eval_protocol['langs'])}->en",
        "items": len(hypotheses),
        "chrf": chrf_score,
        "eval_seconds": eval_seconds,
        "prompt_tokens": int(row["prompt_tokens"]),
        "output_tokens": total_output_tokens,
        "eval_tok_per_s": total_output_tokens / eval_seconds,
        "hypotheses": hypotheses,
    }


def check_sources(source):
    if source.name != "executorch":
        raise ValueError("ExecuTorch's build requires the checkout directory to be named executorch")
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if commit != EXECUTORCH_COMMIT:
        raise ValueError(f"expected ExecuTorch v{EXECUTORCH_VERSION} ({EXECUTORCH_COMMIT}), found {commit}")
    if subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"], text=True).strip():
        raise ValueError("ExecuTorch checkout must be clean")


def prepare(model_path, work, source, profile_name="int4-g128"):
    import executorch
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    if profile_name not in PROFILES:
        raise ValueError(f"unsupported benchmark profile: {profile_name}")
    profile = PROFILES[profile_name]
    bits = profile["bits"]
    group_size = profile["group_size"]
    check_sources(source)
    if importlib.metadata.version("executorch") != EXECUTORCH_VERSION:
        raise ValueError("install the pinned Arm benchmark requirements in an isolated environment")
    executorch.__path__.append(str(source))
    from executorch.examples.models.gemma4.text_decoder.convert_weights import _get_weight_mapping
    from executorch.examples.models.gemma4.text_decoder.gemma4_config import Gemma4Config
    from executorch.examples.models.gemma4.text_decoder.gemma4_transformer import Gemma4ForCausalLM
    from executorch.examples.models.gemma4.export_gemma4 import _build_transform_passes
    from executorch.examples.models.llama.source_transformation.quantize import QuantizedGroupEmbedding
    from executorch.extension.llm.custom_ops import custom_ops  # noqa: F401
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
        XnnpackDynamicallyQuantizedPartitioner, XnnpackPartitioner,
    )
    from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
    from executorch.exir.capture._config import ExecutorchBackendConfig
    from executorch.exir.passes import MemoryPlanningPass
    from executorch.exir.passes.sym_shape_eval_pass import ConstraintBasedSymShapeEvalPass
    from torchao.quantization.quantize_.workflows.intx.intx_unpacked_to_int8_tensor import (
        IntxUnpackedToInt8Tensor, IntxUnpackedToInt8TensorActivationQuantization,
    )

    torch.set_num_threads(4)
    checkpoint = model_path / "model.safetensors"
    config_document = json.loads((model_path / "config.json").read_text())
    if config_document.get("quantization_config"):
        raise ValueError("prepare requires the original unquantized E2B checkpoint")
    config = Gemma4Config.from_config("e2b")
    for name in ("hidden_size", "num_hidden_layers", "vocab_size", "vocab_size_per_layer_input", "head_dim",
                 "global_head_dim", "hidden_size_per_layer_input", "num_kv_shared_layers", "layer_types"):
        if getattr(config, name) != config_document["text_config"][name]:
            raise ValueError(f"source is not the supported Gemma 4 E2B architecture: {name}")
    config.max_seq_len = 1024
    config.use_kv_cache = True
    config.enable_dynamic_shape = True
    config.use_custom_sdpa = True

    # Keep attention buffers initialized, but do not allocate the 20 GB FP32 parameter set.
    with patch.object(torch.nn, "Linear", partial(torch.nn.Linear, device="meta")), \
            patch.object(torch.nn, "Embedding", partial(torch.nn.Embedding, device="meta")):
        model = Gemma4ForCausalLM(config).eval()
    mapping = _get_weight_mapping(config)
    mapping["model.language_model.embed_tokens.weight:lm_head"] = "model.lm_head.weight"
    canonical = work / "shared"
    if canonical.exists():
        shutil.rmtree(canonical)
    canonical.mkdir()
    files = []
    quantized_linears = []

    with safe_open(str(checkpoint), framework="pt", device="cpu") as original:
        for index, (name, target) in enumerate(mapping.items()):
            source_name = name.removesuffix(":lm_head")
            if source_name not in original.keys():
                if name.endswith(".lm_head.weight"):
                    continue
                raise ValueError(f"missing source parameter: {source_name}")
            weight = original.get_tensor(source_name)
            parent, parameter = target.rsplit(".", 1)
            module = model.get_submodule(parent)
            filename = f"model-{index:04d}.safetensors"
            if isinstance(module, (torch.nn.Embedding, torch.nn.Linear)):
                layer_group = 0 if isinstance(module, torch.nn.Embedding) else group_size
                packed, scales = pack_weight(weight, bits=bits, group_size=layer_group)
                prefix = "lm_head" if name.endswith(":lm_head") else source_name.removesuffix(".weight")
                if isinstance(module, torch.nn.Embedding):
                    groups = config.num_hidden_layers if parent.endswith("embed_tokens_per_layer") else 1
                    state = {prefix + ".embedding_quantized": packed,
                             prefix + ".embedding_scale": scales.expand(-1, groups).contiguous()}
                else:
                    state = {prefix + ".weight": packed, prefix + ".weight_scale": scales,
                             prefix + ".input_activation_scale": torch.tensor(0.0),
                             prefix + ".output_activation_scale": torch.tensor(0.0)}
                save_file(state, str(canonical / filename))
                del packed, scales, state, weight
                state = load_file(str(canonical / filename))
                if isinstance(module, torch.nn.Embedding):
                    values = state[prefix + ".embedding_quantized"]
                    scales = state[prefix + ".embedding_scale"][:, :1].contiguous()
                    embedding_dim = values.shape[1] * 2 if bits == 4 else values.shape[1]
                    replacement = QuantizedGroupEmbedding("meta", values.shape[0], embedding_dim,
                                                         dtype=torch.float32, packed=bits == 4, bitwidth=bits)
                    if bits == 4:
                        replacement.weight = embedding_layout(values)
                        if not torch.equal(embedding_layout(replacement.weight), values):
                            raise ValueError("embedding repacking changed quantized values")
                    else:
                        replacement.weight = values.view(torch.int8)
                    replacement.scales = scales.flatten()
                    model.set_submodule(parent, replacement)
                else:
                    values = state[prefix + ".weight"]
                    scales = state[prefix + ".weight_scale"]
                    qdata = unpack_int4(values) if bits == 4 else values.view(torch.int8)
                    block_width = qdata.shape[1] if layer_group == 0 else layer_group
                    target_dtype = torch.int4 if bits == 4 else torch.int8
                    replacement = IntxUnpackedToInt8Tensor(
                        qdata, scales, torch.zeros_like(scales, dtype=torch.int8), target_dtype,
                        (1, block_width), torch.float32,
                        IntxUnpackedToInt8TensorActivationQuantization.INT8_ASYM_PER_TOKEN,
                    )
                    module.weight = torch.nn.Parameter(replacement, requires_grad=False)
                    quantized_linears.append(parent)
            else:
                state = {source_name: weight.float().contiguous()}
                save_file(state, str(canonical / filename))
                setattr(module, parameter, torch.nn.Parameter(state[source_name], requires_grad=False))
            files.append(filename)
            del state
            gc.collect()
    if any(parameter.is_meta for parameter in model.parameters()):
        raise ValueError("some ExecuTorch parameters were not supplied by the shared checkpoint")
    save_file(cache_calibration(config.num_hidden_layers, config.num_kv_shared_layers),
              str(canonical / "model-cache.safetensors"))
    files.append("model-cache.safetensors")

    quantization_config = {
        "quant_method": "gemma", "num_bits": bits, "quantize_embeddings": True,
        "modules_to_not_convert": [],
        "module_quant_configs": {},
    }
    if group_size:
        quantization_config["group_size"] = group_size
        quantization_config["embedding_group_size"] = group_size
    config_document["quantization_config"] = quantization_config
    config_document["text_config"].update(dtype="float32", packed_weights_signed=True)
    save(canonical / "config.json", config_document)
    consolidate_weights(canonical, files)
    merged = load_file(str(canonical / "model.safetensors"))
    if "lm_head.weight" in merged:
        merged["model.language_model.embed_tokens.embedding_quantized"] = merged.pop("lm_head.weight")
        merged["model.language_model.embed_tokens.embedding_scale"] = merged.pop("lm_head.weight_scale")
        merged.pop("lm_head.input_activation_scale", None)
        merged.pop("lm_head.output_activation_scale", None)
        save_file(merged, str(canonical / "model.safetensors"))
    del merged
    shutil.copyfile(model_path / "tokenizer.json", canonical / "tokenizer.json")

    inputs = torch.ones((1, 32), dtype=torch.int64)
    positions = torch.arange(32, dtype=torch.int64)
    length = torch.export.Dim("tokens", min=1, max=1023)
    with torch.no_grad():
        exported = torch.export.export(model, (inputs, positions),
                                       dynamic_shapes=({1: length}, {0: length}))
    exported = exported.run_decompositions({})
    edge = to_edge_transform_and_lower(
        {"forward": exported},
        compile_config=EdgeCompileConfig(),
        partitioner={"forward": [XnnpackDynamicallyQuantizedPartitioner(), XnnpackPartitioner()]},
        transform_passes={"forward": _build_transform_passes(False, False)["text_decoder"]},
    )
    delegates = [module.backend_id for module in edge.exported_program("forward").graph_module.modules()
                 if hasattr(module, "backend_id")]
    if not delegates or any(backend != "XnnpackBackend" for backend in delegates):
        raise ValueError("expected CPU XNNPACK delegates only")
    remaining = [str(node.target) for node in edge.exported_program("forward").graph_module.graph.nodes
                 if node.op == "call_function" and "quantize" in str(node.target) and
                 "embedding" not in str(node.target)]
    if remaining:
        raise ValueError(f"quantized linear operations were not delegated to XNNPACK: {remaining}")
    program = edge.to_executorch(ExecutorchBackendConfig(
        extract_delegate_segments=True, memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False),
        sym_shape_eval_pass=ConstraintBasedSymShapeEvalPass()))
    pte = work / "gemma4-e2b-shared.pte"
    with pte.open("wb") as stream:
        program.write_to_file(stream)
    if program._tensor_data:
        raise ValueError("unexpected external tensor data; benchmark requires a self-contained PTE")
    manifest = {
        "version": 1, "profile": profile_name, "precision": profile["precision"],
        "source_checkpoint": str(checkpoint),
        "source_sha256": digest(checkpoint), "source_config_sha256": digest(model_path / "config.json"),
        "executorch_commit": EXECUTORCH_COMMIT,
        "packages": {name: importlib.metadata.version(name) for name in ("executorch", "torch", "torchao")},
        "files": {str(path.relative_to(work)): digest(path) for path in sorted(canonical.iterdir())},
        "pte": pte.name, "pte_sha256": digest(pte),
        "quantized_linears": quantized_linears, "xnnpack_delegates": len(delegates),
    }
    save(work / "manifest.json", manifest)


def prepare_asym_clip(model_path: Path, work: Path, group_size: int = 128,
                      subdir: str = "shared-asym-clip") -> Path:
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    target_dir = work / subdir
    if (target_dir / "model.safetensors").exists() and (target_dir / "config.json").exists():
        cfg = json.loads((target_dir / "config.json").read_text())
        qcfg = cfg.get("quantization_config", {})
        if qcfg.get("group_size") == group_size and qcfg.get("embedding_group_size") == group_size:
            with safe_open(str(target_dir / "model.safetensors"), framework="pt", device="cpu") as existing:
                emb_dt = existing.get_slice("model.language_model.embed_tokens.embedding_scale").get_dtype()
                kc_dt = existing.get_slice("model.language_model.layers.0.self_attn.k_cache_scale").get_dtype()
                if emb_dt != "F16" or kc_dt != "F32":
                    upgraded = {}
                    for k in existing.keys():
                        t = existing.get_tensor(k)
                        if k.endswith((".weight_scale", ".embedding_scale", ".weight_zero", ".embedding_zero")):
                            upgraded[k] = t.half()
                        elif k.endswith(("_cache_scale", "activation_scale")):
                            upgraded[k] = t.float()
                        else:
                            upgraded[k] = t
                    save_file(upgraded, str(target_dir / "model.safetensors"))
                    del upgraded
            return target_dir
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    checkpoint = model_path / "model.safetensors"
    state = load_file(str(work / "shared/model.safetensors"))
    state.pop("lm_head.weight", None)
    state.pop("lm_head.weight_scale", None)
    state.pop("lm_head.weight_zero", None)
    state.pop("lm_head.input_activation_scale", None)
    state.pop("lm_head.output_activation_scale", None)
    with safe_open(str(checkpoint), framework="pt", device="cpu") as original:
        emb_w = original.get_tensor("model.language_model.embed_tokens.weight")
        emb_p, emb_s, emb_z = pack_int4_asym_clip(emb_w, group_size=group_size, chunk_rows=2048, asymmetric=True)
        state["model.language_model.embed_tokens.embedding_quantized"] = emb_p
        state["model.language_model.embed_tokens.embedding_scale"] = emb_s.half()
        state["model.language_model.embed_tokens.embedding_zero"] = emb_z.half()
        del emb_w, emb_p, emb_s, emb_z
        ple_w = original.get_tensor("model.language_model.embed_tokens_per_layer.weight")
        ple_p, ple_s, ple_z = pack_int4_asym_clip(ple_w, group_size=256, chunk_rows=1024, asymmetric=True)
        state["model.language_model.embed_tokens_per_layer.embedding_quantized"] = ple_p
        state["model.language_model.embed_tokens_per_layer.embedding_scale"] = ple_s.half()
        state["model.language_model.embed_tokens_per_layer.embedding_zero"] = ple_z.half()
        del ple_w, ple_p, ple_s, ple_z
        for source_name in sorted(original.keys()):
            if not source_name.startswith(("model.language_model.layers.", "model.language_model.per_layer")):
                continue
            if not source_name.endswith(".weight"):
                continue
            weight = original.get_tensor(source_name)
            if weight.ndim != 2:
                continue
            prefix = source_name.removesuffix(".weight")
            packed, scales, zeros = pack_int4_asym_clip(
                weight, group_size=group_size, chunk_rows=2048, asymmetric=True
            )
            state[prefix + ".weight"] = packed
            state[prefix + ".weight_scale"] = scales.half()
            state[prefix + ".weight_zero"] = zeros.half()
            del packed, scales, zeros, weight
            gc.collect()
    save_file(state, str(target_dir / "model.safetensors"))
    del state
    config_document = json.loads((work / "shared/config.json").read_text())
    config_document["quantization_config"]["group_size"] = group_size
    config_document["quantization_config"]["embedding_group_size"] = group_size
    save(target_dir / "config.json", config_document)
    shutil.copyfile(work / "shared/tokenizer.json", target_dir / "tokenizer.json")
    return target_dir


def build(work, source, jobs):
    check_sources(source)
    subprocess.run(["cmake", "-G", "Ninja", "-S", str(HERE), "-B", str(work / "build-arm"), "-DCMAKE_BUILD_TYPE=Release",
                    "-DCMAKE_POLICY_VERSION_MINIMUM=3.10", f"-DEXECUTORCH_SOURCE_DIR={source}",
                    f"-DPYTHON_EXECUTABLE={sys.executable}"], check=True)
    subprocess.run(["cmake", "--build", str(work / "build-arm"), "--target", "arm_gemma4_bench",
                    "--parallel", str(jobs)], check=True)
    subprocess.run(["cmake", "-G", "Ninja", "-S", str(ROOT), "-B", str(work / "build-kidi"), "-DCMAKE_BUILD_TYPE=Release",
                    "-DKIDI_BUILD_BENCHMARKS=ON", "-DKIDI_BUILD_TESTS=OFF",
                    "-DKIDI_BUILD_DESKTOP_APP=OFF", "-DKIDI_ENABLE_CUDA=OFF", "-DKIDI_ENABLE_VULKAN=OFF"],
                   check=True)
    subprocess.run(["cmake", "--build", str(work / "build-kidi"), "--target", "kidi_android_baseline",
                    "--parallel", str(jobs)], check=True)


def measure(command, work, name, protocol, timeout):
    environment = {key: value for key, value in os.environ.items() if not key.startswith("KIDI_")}
    environment["KIDI_ACCELERATOR"] = "cpu"
    try:
        completed = subprocess.run(command, cwd=ROOT, env=environment, text=True, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired as error:
        for suffix, text in (("stdout.jsonl", error.stdout), ("stderr.log", error.stderr)):
            (work / f"{name}.{suffix}").write_text(
                text.decode(errors="replace") if isinstance(text, bytes) else text or "")
        raise RuntimeError(f"{name} timed out; partial logs saved under {work}") from error
    (work / f"{name}.stdout.jsonl").write_text(completed.stdout)
    (work / f"{name}.stderr.log").write_text(completed.stderr)
    if completed.returncode:
        raise RuntimeError(f"{name} exited {completed.returncode}: {completed.stderr[-3000:]}")
    records = records_from(completed.stdout)
    return {"command": command, "records": records, "summary": summarize(records, protocol), "cached": False}


def load_partial_eval_items(work: Path, name: str, eval_protocol: dict, tokenizer=None) -> dict[int, dict]:
    results_dir = work / "results"
    jsonl_path = results_dir / f"{name}_eval.output.jsonl"
    json_path = results_dir / f"{name}_eval.json"
    legacy_tokens = {}
    legacy_avg_sec = 0.05
    if json_path.exists():
        try:
            doc = json.loads(json_path.read_text(encoding="utf-8"))
            ev_rows = [r for r in doc.get("records", []) if r.get("stage") == "eval"]
            if ev_rows:
                t_list = ev_rows[0].get("token_ids", [])
                if t_list:
                    legacy_avg_sec = max(float(ev_rows[0].get("eval_seconds", 0.1)) / len(t_list), 1e-4)
                for idx, tids in enumerate(t_list):
                    legacy_tokens[idx] = tids
        except (OSError, ValueError, KeyError, TypeError):
            pass
    completed = {}
    if jsonl_path.exists():
        try:
            for line in jsonl_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                idx = int(row["id"])
                if not (0 <= idx < eval_protocol["items"]):
                    continue
                if row.get("source") != eval_protocol["sources"][idx]:
                    continue
                tids = row.get("token_ids", legacy_tokens.get(idx))
                if not isinstance(tids, list):
                    continue
                clean = list(tids)
                for stop_id in (1, 106):
                    if stop_id in clean:
                        clean = clean[:clean.index(stop_id)]
                sec = float(row.get("seconds", legacy_avg_sec))
                if not math.isfinite(sec) or sec <= 0:
                    sec = legacy_avg_sec
                out_text = row.get("output")
                if out_text is None and tokenizer is not None:
                    out_text = clean_translation_text(tokenizer.decode(clean, skip_special_tokens=True))
                completed[idx] = {
                    "id": idx,
                    "lang": eval_protocol["item_langs"][idx],
                    "source": eval_protocol["sources"][idx],
                    "reference": eval_protocol["references"][idx],
                    "output": out_text or "",
                    "token_ids": clean,
                    "prompt_tokens": int(row.get("prompt_tokens", len(eval_protocol["input_token_ids_list"][idx]))),
                    "output_tokens": len(clean),
                    "seconds": sec,
                }
        except (OSError, ValueError, KeyError, TypeError):
            pass
    return completed


def write_eval_output_jsonl(work: Path, name: str, items_by_id: dict[int, dict]) -> None:
    results_dir = work / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"{name}_eval.output.jsonl"
    lines = [json.dumps(items_by_id[idx], ensure_ascii=False) for idx in sorted(items_by_id)]
    out_path.write_text(("\n".join(lines) + "\n") if lines else "", encoding="utf-8")


def measure_eval(command, work, name, eval_protocol, tokenizer, timeout, force=False):
    completed_by_id = {} if force else load_partial_eval_items(work, name, eval_protocol, tokenizer)
    missing_ids = [idx for idx in range(eval_protocol["items"]) if idx not in completed_by_id]
    if not missing_ids:
        ordered = [completed_by_id[idx] for idx in range(eval_protocol["items"])]
        records = [{
            "stage": "eval",
            "items": len(ordered),
            "eval_seconds": sum(item["seconds"] for item in ordered),
            "prompt_tokens": sum(item["prompt_tokens"] for item in ordered),
            "output_tokens": sum(item["output_tokens"] for item in ordered),
            "token_ids": [item["token_ids"] for item in ordered],
        }]
        return {
            "command": command,
            "records": records,
            "summary": summarize_eval(records, eval_protocol, tokenizer),
            "cached": True,
        }

    write_eval_output_jsonl(work, name, completed_by_id)
    total_items = eval_protocol["items"]
    print(
        f"[eval] {name}: {len(completed_by_id)}/{total_items} cached, running {len(missing_ids)} remaining...",
        file=sys.stderr, flush=True,
    )
    pending_protocol = {
        **eval_protocol,
        "items": len(missing_ids),
        "item_ids": missing_ids,
        "prompts": [eval_protocol["prompts"][idx] for idx in missing_ids],
        "input_token_ids_list": [eval_protocol["input_token_ids_list"][idx] for idx in missing_ids],
        "sources": [eval_protocol["sources"][idx] for idx in missing_ids],
        "references": [eval_protocol["references"][idx] for idx in missing_ids],
        "item_langs": [eval_protocol["item_langs"][idx] for idx in missing_ids],
    }
    pending_path = work / f"eval_pending_{name}.json"
    save(pending_path, pending_protocol)
    orig_protocol_str = str(work / "eval_protocol.json")
    pending_cmd = [
        arg.replace(orig_protocol_str, str(pending_path))
        for arg in command
    ]

    environment = {key: value for key, value in os.environ.items() if not key.startswith("KIDI_")}
    environment["KIDI_ACCELERATOR"] = "cpu"
    results_dir = work / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = results_dir / f"{name}_eval.output.jsonl"
    stderr_log_path = work / f"{name}_eval.stderr.log"
    stdout_lines = []
    started = time.monotonic() if hasattr(time := __import__("time"), "monotonic") else 0.0
    with stderr_log_path.open("w", encoding="utf-8") as stderr_file, subprocess.Popen(
        pending_cmd, cwd=ROOT, env=environment, text=True,
        stdout=subprocess.PIPE, stderr=stderr_file, bufsize=1,
    ) as proc, jsonl_path.open("a", encoding="utf-8") as stream:
        assert proc.stdout is not None
        for line in proc.stdout:
            stdout_lines.append(line)
            if timeout and (time.monotonic() - started) > timeout:
                proc.kill()
                raise RuntimeError(f"{name} eval timed out after {timeout}s")
            stripped = line.strip()
            if not stripped.startswith("{"):
                continue
            try:
                rec = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if rec.get("stage") == "eval_item":
                idx = int(rec["id"])
                clean = list(rec["token_ids"])
                for stop_id in (1, 106):
                    if stop_id in clean:
                        clean = clean[:clean.index(stop_id)]
                hyp = ""
                if tokenizer is not None:
                    hyp = clean_translation_text(tokenizer.decode(clean, skip_special_tokens=True))
                entry = {
                    "id": idx,
                    "lang": eval_protocol["item_langs"][idx],
                    "source": eval_protocol["sources"][idx],
                    "reference": eval_protocol["references"][idx],
                    "output": hyp,
                    "token_ids": clean,
                    "prompt_tokens": int(rec.get("prompt_tokens", len(eval_protocol["input_token_ids_list"][idx]))),
                    "output_tokens": len(clean),
                    "seconds": max(float(rec.get("seconds", 1e-4)), 1e-4),
                }
                completed_by_id[idx] = entry
                stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
                stream.flush()
                if len(completed_by_id) % 25 == 0 or len(completed_by_id) == total_items:
                    print(
                        f"[eval] {name}: {len(completed_by_id)}/{total_items} completed",
                        file=sys.stderr, flush=True,
                    )
        returncode = proc.wait()
    pending_path.unlink(missing_ok=True)
    stdout_text = "".join(stdout_lines)
    stderr_text = stderr_log_path.read_text(encoding="utf-8", errors="replace")
    (work / f"{name}_eval.stdout.jsonl").write_text(stdout_text, encoding="utf-8")
    if returncode:
        raise RuntimeError(f"{name} eval exited {returncode}: {stderr_text[-3000:]}")
    if len(completed_by_id) < eval_protocol["items"]:
        raw_records = records_from(stdout_text)
        ev_rows = [r for r in raw_records if r.get("stage") == "eval"]
        if ev_rows and len(ev_rows[0].get("token_ids", [])) == len(missing_ids):
            avg_sec = max(float(ev_rows[0]["eval_seconds"]) / max(len(missing_ids), 1), 1e-4)
            for idx, tids in zip(missing_ids, ev_rows[0]["token_ids"]):
                clean = list(tids)
                for stop_id in (1, 106):
                    if stop_id in clean:
                        clean = clean[:clean.index(stop_id)]
                hyp = clean_translation_text(tokenizer.decode(clean, skip_special_tokens=True)) if tokenizer else ""
                completed_by_id[idx] = {
                    "id": idx,
                    "lang": eval_protocol["item_langs"][idx],
                    "source": eval_protocol["sources"][idx],
                    "reference": eval_protocol["references"][idx],
                    "output": hyp,
                    "token_ids": clean,
                    "prompt_tokens": len(eval_protocol["input_token_ids_list"][idx]),
                    "output_tokens": len(clean),
                    "seconds": avg_sec,
                }
    write_eval_output_jsonl(work, name, completed_by_id)
    ordered = [completed_by_id[idx] for idx in range(eval_protocol["items"])]
    records = [{
        "stage": "eval",
        "items": len(ordered),
        "eval_seconds": sum(item["seconds"] for item in ordered),
        "prompt_tokens": sum(item["prompt_tokens"] for item in ordered),
        "output_tokens": sum(item["output_tokens"] for item in ordered),
        "token_ids": [item["token_ids"] for item in ordered],
    }]
    return {
        "command": command,
        "records": records,
        "summary": summarize_eval(records, eval_protocol, tokenizer),
        "cached": False,
    }


def save_cached_eval(work, name, eval_run, eval_protocol):
    results_dir = work / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": name,
        "eval_protocol": eval_protocol,
        "command": eval_run["command"],
        "binary_sha256": eval_run["binary_sha256"],
        "records": eval_run["records"],
        "summary": eval_run["summary"],
    }
    save(results_dir / f"{name}_eval.json", payload)
    ev_rows = [r for r in eval_run.get("records", []) if r.get("stage") == "eval"]
    raw_tids = ev_rows[0].get("token_ids", []) if ev_rows else []
    avg_sec = max(float(eval_run["summary"].get("eval_seconds", 0.1)) / max(len(raw_tids), 1), 1e-4)
    existing = load_partial_eval_items(work, name, eval_protocol)
    hypotheses = eval_run["summary"].get("hypotheses", [])
    items_by_id = {}
    for idx, hyp in enumerate(hypotheses):
        if idx in existing:
            items_by_id[idx] = {**existing[idx], "output": hyp}
        else:
            tids = list(raw_tids[idx]) if idx < len(raw_tids) else []
            for stop_id in (1, 106):
                if stop_id in tids:
                    tids = tids[:tids.index(stop_id)]
            items_by_id[idx] = {
                "id": idx,
                "lang": eval_protocol["item_langs"][idx],
                "source": eval_protocol["sources"][idx],
                "reference": eval_protocol["references"][idx],
                "output": hyp,
                "token_ids": tids,
                "prompt_tokens": len(eval_protocol["input_token_ids_list"][idx]),
                "output_tokens": len(tids),
                "seconds": avg_sec,
            }
    write_eval_output_jsonl(work, name, items_by_id)


def load_cached_eval(work, name, command, binary_sha256, eval_protocol, tokenizer):
    path = work / "results" / f"{name}_eval.json"
    if not path.exists():
        return None
    try:
        document = json.loads(path.read_text())
        if (document.get("eval_protocol") != eval_protocol or
                document.get("command") != command or
                document.get("binary_sha256") != binary_sha256):
            return None
        records = document["records"]
        summary = summarize_eval(records, eval_protocol, tokenizer)
        return {
            "command": command,
            "records": records,
            "summary": summary,
            "binary_sha256": binary_sha256,
            "cached": True,
        }
    except (OSError, ValueError, KeyError, TypeError):
        return None


def stripped_file_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        shutil.copy2(path, tmp_path)
        res = subprocess.run(["strip", "-x", str(tmp_path)], check=False, capture_output=True)
        return tmp_path.stat().st_size if res.returncode == 0 else path.stat().st_size
    finally:
        tmp_path.unlink(missing_ok=True)


def runtime_binary_metrics(name: str, command: list[str]) -> dict:
    exe = Path(command[0])
    if not exe.exists():
        return {"binary_size_mib": 0.0, "binary_raw_mib": 0.0, "binary_files": []}
    if name == "onnx":
        site = exe.absolute().parent.parent / "lib"
        cands = sorted(site.glob("python*/site-packages/onnxruntime/capi/libonnxruntime.*.*"))
        if not cands:
            cands = sorted(site.glob("python*/site-packages/onnxruntime/capi/onnxruntime_pybind11_state.*"))
        files = [cands[0].resolve()] if cands else [exe.resolve()]
    elif name == "litert":
        site = exe.absolute().parent.parent / "lib"
        cands = sorted(site.glob("python*/site-packages/litert_lm/liblitert-lm.*"))
        files = [cands[0].resolve()] if cands else [exe.resolve()]
    elif name == "hf_bf16":
        site = exe.absolute().parent.parent / "lib"
        cands = sorted(site.glob("python*/site-packages/torch/lib/libtorch_cpu.*"))
        files = [cands[0].resolve()] if cands else [exe.resolve()]
    else:
        binary = exe.resolve()
        files = [binary]
        seen = {binary}
        try:
            rpaths = []
            out_l = subprocess.check_output(["otool", "-l", str(binary)], text=True, stderr=subprocess.DEVNULL)
            lines = out_l.splitlines()
            for idx, line in enumerate(lines):
                if "cmd LC_RPATH" in line:
                    for sub in range(idx + 1, min(idx + 6, len(lines))):
                        if "path " in lines[sub]:
                            rpath = lines[sub].strip().split("path ", 1)[1].split(" (offset")[0].strip()
                            rpaths.append(Path(rpath.replace("@loader_path", str(binary.parent))
                                                    .replace("@executable_path", str(binary.parent))))
            out_deps = subprocess.check_output(["otool", "-L", str(binary)], text=True, stderr=subprocess.DEVNULL)
            for line in out_deps.splitlines()[1:]:
                dep = line.strip().split(" (compatibility")[0].strip()
                if dep.startswith(("/usr/lib/", "/System/Library/")):
                    continue
                cand = None
                if dep.startswith("@rpath/"):
                    rel = dep.removeprefix("@rpath/")
                    for rp in rpaths:
                        if (rp / rel).exists():
                            cand = (rp / rel).resolve()
                            break
                elif Path(dep).exists():
                    cand = Path(dep).resolve()
                if cand and cand not in seen:
                    seen.add(cand)
                    files.append(cand)
        except (OSError, subprocess.SubprocessError):
            pass
    raw_bytes = sum(f.stat().st_size for f in files)
    stripped_bytes = sum(stripped_file_bytes(f) for f in files)
    return {
        "binary_size_mib": stripped_bytes / 2 ** 20,
        "binary_raw_mib": raw_bytes / 2 ** 20,
        "binary_files": [f.name for f in files],
    }


def save_cached_run(work, name, run, protocol):
    results_dir = work / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": name,
        "protocol": protocol,
        "command": run["command"],
        "binary_sha256": run["binary_sha256"],
        "records": run["records"],
        "summary": run["summary"],
    }
    save(results_dir / f"{name}.json", payload)


def load_cached_run(work, name, command, binary_sha256, protocol, tokenizer=None):
    candidates = [work / "results" / f"{name}.json"]
    report_path = work / "benchmark.json"
    if report_path.exists():
        candidates.append(report_path)
    for path in candidates:
        if not path.exists():
            continue
        try:
            document = json.loads(path.read_text())
            if path.name == "benchmark.json":
                if document.get("protocol") != protocol:
                    continue
                entry = document.get("runs", {}).get(name)
                if not isinstance(entry, dict):
                    continue
            else:
                if document.get("protocol") != protocol:
                    continue
                entry = document
            if entry.get("command") != command or entry.get("binary_sha256") != binary_sha256:
                continue
            records = entry["records"]
            summary = summarize(records, protocol)
            if name == "kidi":
                loads = [row for row in records if row.get("stage") == "load"]
                if len(loads) != 1 or loads[0].get("native_qat") is not True:
                    continue
            if tokenizer is not None:
                summary["text"] = tokenizer.decode(summary["token_ids"], skip_special_tokens=False)
            elif "text" in entry.get("summary", {}):
                summary["text"] = entry["summary"]["text"]
            result = {
                "command": command,
                "records": records,
                "summary": summary,
                "binary_sha256": binary_sha256,
                "cached": True,
            }
            if path.name == "benchmark.json":
                save_cached_run(work, name, result, protocol)
            return result
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return None


TABLE_SPECS = (
    ("bf16_int8", "### 1. Reference BF16 & 8-Bit (`int8`) Inference"),
    ("int4_g128", "### 2. 4-Bit Inference — Group Size 128 (`int4-g128`)"),
    ("int4_g32", "### 3. 4-Bit Inference — Group Size 32 (`int4-g32`)"),
    ("qat_mixed", "### 4. 4-Bit Quantization-Aware Trained (QAT / Mixed Precision)"),
)


BINARY_TABLE_ROWS = (
    ("llama.cpp (`llamacpp`)", "`llamacpp_gemma4_bench` + `libllama.0.dylib` + `libggml*.dylib`", "4.48", "5.10",
     "C++ probe + GGML CPU/KleidiAI shared libraries (reads GGUF vocabulary)"),
    ("Kidi (`kidi` / `kidi_bf16` / `kidi_qat`)", "`kidi_android_baseline` (single static binary)", "7.62", "9.39",
     "Zero external non-OS dylibs (**6.29 MiB** with `-Wl,-dead_strip`). Includes C++ HF tokenizer ([`tokenizerspp`](https://github.com/thammegowda/tokenizerspp) + `uni-algo` + `RE2`), vision codecs ([`tahoma-vision`](https://github.com/thammegowda/tahoma-vision) + `libjpeg-turbo` + PNG/[`pigzpp`](https://github.com/thammegowda/pigzpp)/`zlib-ng`), WAV/Mel audio, Safetensors/GGUF/YAML/JSON parsers, and 6 model families; core Gemma 4 + [`ynnpack-dev`](https://github.com/thammegowda/ynnpack-dev) is **~3.3 MiB**"),
    ("Arm ExecuTorch v1.5.1 (`arm`)", "`arm_gemma4_bench` (single static binary)", "7.87", "10.28",
     "Zero external non-OS dylibs (**6.22 MiB** with `-Wl,-dead_strip`); ExecuTorch runtime + XNNPACK + KleidiAI (excludes tokenizer, vision/audio codecs, and checkpoint parsers)"),
    ("ONNX Runtime v1.31.1 (`onnx`)", "`libonnxruntime.1.31.1.dylib`", "20.10", "31.51",
     "Standalone native ONNX Runtime C/C++ shared library (excludes Python interpreter and tokenizer)"),
    ("Google LiteRT-LM v0.17.1 (`litert`)", "`liblitert-lm.dylib`", "52.18", "66.01",
     "Standalone native Google LiteRT-LM C++ shared library (excludes Python interpreter)"),
)


def load_results_tsv(tsv_path: Path = HERE / "results.tsv") -> list[dict[str, str]]:
    lines = [line.rstrip("\r\n") for line in tsv_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        return []
    headers = lines[0].split("\t")
    rows = []
    for line in lines[1:]:
        cols = line.split("\t")
        if len(cols) != len(headers):
            raise ValueError(f"expected {len(headers)} TSV columns, got {len(cols)}: {line}")
        rows.append(dict(zip(headers, cols)))
    return rows


def render_markdown_tables(tsv_path: Path = HERE / "results.tsv") -> str:
    rows = load_results_tsv(tsv_path)
    sections = []
    header = (
        "| Runtime | Engine / Backend | Weight Format | Model (GiB) | Prefill (tok/s) | "
        "Decode (tok/s) | TTFT (ms) | Peak RSS (GiB) | chrF (ref) | chrF (bf16) | Eval Time (s) |\n"
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    for group_key, title in TABLE_SPECS:
        group_rows = [r for r in rows if r["table_group"] == group_key]
        if not group_rows:
            continue
        lines = [title, "", header]
        for r in group_rows:
            lines.append(
                f"| **{r['runtime']}** | {r['backend']} | {r['weight_format']} | "
                f"{r['model_gib']} | {r['prefill_tps']} | {r['decode_tps']} | "
                f"{r['ttft_ms']} | {r['peak_rss_gib']} | "
                f"{r['chrf_ref']} | {r['chrf_bf16']} | {r['wmt23_eval_s']} |"
            )
        sections.append("\n".join(lines))
    bin_lines = [
        "### 5. Self-Contained Runtime Binary Size Comparison",
        "",
        "| Runtime | Artifacts Measured | Stripped (MiB) | Unstripped (MiB) | Notes |",
        "|---|---|---:|---:|---|",
    ]
    for rt, artifacts, stripped, unstripped, notes in BINARY_TABLE_ROWS:
        bin_lines.append(f"| **{rt}** | {artifacts} | **{stripped}** | {unstripped} | {notes} |")
    sections.append("\n".join(bin_lines))
    return "\n\n".join(sections)


def render_pareto_svg(tsv_path: Path = HERE / "results.tsv") -> str:
    rows = load_results_tsv(tsv_path)
    by_id = {r["runtime_id"]: r for r in rows}

    # Publication-style two-panel scatter plot:
    # Panel A (Left):  Prefill Throughput (tokens/s) vs. WMT23 chrF (ref)
    # Panel B (Right): Decode Throughput (tokens/s)  vs. WMT23 chrF (ref)
    ymin, ymax = 54.4, 57.7
    bf16_y = float(by_id["kidi_bf16"]["chrf_ref"])

    # Seaborn "colorblind" (Okabe-Ito) accessible palette:
    #   Kidi (Ours):      Vermilion (#D55E00)
    #   Arm ExecuTorch:   Blue (#0173B2)
    #   ONNX Runtime:     Bluish Green (#029E73)
    #   llama.cpp:        Reddish Purple (#CC78BC)
    #   Google LiteRT-LM: Orange (#DE8F05)
    #   BF16 Reference:   Deep Slate (#0f172a)
    #
    # Shapes by precision family:
    #   "star"     -> BF16 reference
    #   "solid"    -> 8-bit (Q8)
    #   "hollow"   -> 4-bit (Q4, g=32 / g=128)
    #   "triangle" -> MixedQ (QAT mixed 2/4/8-bit)
    meta = {
        "kidi_bf16":      ("#0f172a", "star",     "Kidi BF16"),
        "kidi_int8":      ("#D55E00", "solid",    "Kidi Q8"),
        "arm_int8":       ("#0173B2", "solid",    "Arm ET Q8"),
        "onnx_int8":      ("#029E73", "solid",    "ONNX Q8"),
        "llamacpp_int8":  ("#CC78BC", "solid",    "llama.cpp Q8"),
        "kidi_asym_g32":  ("#D55E00", "hollow",   "Kidi Q4 Asym (g=32)"),
        "llamacpp_q4_0":  ("#CC78BC", "hollow",   "llama.cpp Q4 (g=32)"),
        "litert":         ("#DE8F05", "triangle", "LiteRT-LM MixedQ"),
        "kidi_qat":       ("#D55E00", "triangle", "Kidi MixedQ"),
        "kidi_asym_clip": ("#D55E00", "hollow",   "Kidi Q4 Asym (g=128)"),
        "kidi_int4_g128": ("#D55E00", "hollow",   "Kidi Q4 Sym (g=128)"),
        "onnx_int4_g128": ("#029E73", "hollow",   "ONNX Q4 (g=128)"),
        "arm_int4_g128":  ("#0173B2", "hollow",   "Arm ET Q4 (g=128)"),
    }

    # Label offsets (dx, dy, text_anchor) with leader lines so zero labels collide
    prefill_labels = {
        "kidi_bf16":      ( 14,  18, "start"),
        "llamacpp_int8":  ( -4,  21, "middle"),
        "onnx_int8":      ( -8, -16, "end"),
        "arm_int8":       (  8, -15, "start"),
        "kidi_int8":      ( 15, -13, "start"),
        "llamacpp_q4_0":  ( 14, -14, "start"),
        "litert":         ( 12,  19, "start"),
        "kidi_asym_g32":  ( 15, -14, "start"),
        "kidi_asym_clip": (-16,  -4, "end"),
        "kidi_int4_g128": ( 16,  17, "start"),
        "kidi_qat":       ( -6, -17, "end"),
        "onnx_int4_g128": ( 14,  15, "start"),
        "arm_int4_g128":  (  0, -15, "middle"),
    }

    decode_labels = {
        "kidi_bf16":      ( 14,  20, "start"),
        "arm_int8":       (-15,  -8, "end"),
        "onnx_int8":      ( -4, -18, "middle"),
        "kidi_int8":      ( 16, -12, "start"),
        "llamacpp_int8":  ( 16,  15, "start"),
        "kidi_asym_g32":  ( 16, -15, "start"),
        "llamacpp_q4_0":  (-16, -13, "end"),
        "litert":         (-16,   5, "end"),
        "kidi_asym_clip": (-16,  -5, "end"),
        "kidi_int4_g128": ( 14,  16, "start"),
        "kidi_qat":       (  0, -17, "middle"),
        "onnx_int4_g128": ( 14,  15, "start"),
        "arm_int4_g128":  (-14,  14, "end"),
    }

    # Comparable precision & group-size tiers connected within each subplot
    comparable_groups = [
        ("q8",       ["llamacpp_int8", "onnx_int8", "arm_int8", "kidi_int8"],              "none"),
        ("q4_g32",   ["llamacpp_q4_0", "kidi_asym_g32"],                                   "7,4"),
        ("q4_g128",  ["arm_int4_g128", "onnx_int4_g128", "kidi_asym_clip", "kidi_int4_g128"], "3,3"),
        ("mixedq",   ["litert", "kidi_qat"],                                               "7,3,2,3"),
    ]

    def star_points(cx: float, cy: float, r_outer: float = 8.8, r_inner: float = 3.8) -> str:
        pts = []
        for i in range(10):
            angle = -math.pi / 2 + i * math.pi / 5
            r = r_outer if i % 2 == 0 else r_inner
            pts.append(f"{cx + r * math.cos(angle):.1f},{cy + r * math.sin(angle):.1f}")
        return " ".join(pts)

    def triangle_points(cx: float, cy: float, r: float = 7.2) -> str:
        return f"{cx:.1f},{cy - r:.1f} {cx - r * 0.92:.1f},{cy + r * 0.72:.1f} {cx + r * 0.92:.1f},{cy + r * 0.72:.1f}"

    svg = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 560" width="100%" height="100%" '
        'style="font-family: \'Inter\', -apple-system, BlinkMacSystemFont, \'Segoe UI\', Roboto, Helvetica, Arial, sans-serif;">',
        '  <defs>',
        '    <linearGradient id="bg-grad" x1="0" y1="0" x2="0" y2="1">',
        '      <stop offset="0%" stop-color="#f8fafc"/>',
        '      <stop offset="100%" stop-color="#f1f5f9"/>',
        '    </linearGradient>',
        '    <linearGradient id="pareto-fill" x1="0" y1="0" x2="1" y2="1">',
        '      <stop offset="0%" stop-color="#D55E00" stop-opacity="0.10"/>',
        '      <stop offset="100%" stop-color="#D55E00" stop-opacity="0.02"/>',
        '    </linearGradient>',
        '    <filter id="card-shadow" x="-2%" y="-2%" width="104%" height="106%">',
        '      <feDropShadow dx="0" dy="2" stdDeviation="3" flood-color="#0f172a" flood-opacity="0.05"/>',
        '    </filter>',
        '  </defs>',
        '  <rect width="1200" height="560" rx="14" fill="url(#bg-grad)" stroke="#cbd5e1" stroke-width="1.2"/>',
        '  <text x="600" y="31" text-anchor="middle" font-size="17.5" font-weight="800" letter-spacing="-0.3" fill="#0f172a">'
        'Gemma 4 E2B on Arm CPU (Apple M5, 4 Threads) — Speed vs. Translation Quality</text>',
        '  <text x="600" y="51" text-anchor="middle" font-size="12.5" font-weight="500" fill="#475569">'
        '150 WMT23 test sentences (ja, uk, de → en) · SacreBLEU chrF vs. human reference · Upper-right is faster &amp; higher quality (↗)</text>',
    ]

    panels = [
        ("A. Prefill Throughput vs. Translation Quality",
         "prefill_tps", 74.0, 562.0, 100.0, 900.0,
         [100, 200, 300, 400, 500, 600, 700, 800, 900],
         "Prefill Speed (tokens / sec) →",
         prefill_labels),
        ("B. Decode Throughput vs. Translation Quality",
         "decode_tps", 648.0, 1136.0, 12.0, 73.0,
         [15, 25, 35, 45, 55, 65],
         "Autoregressive Decode Speed (tokens / sec) →",
         decode_labels),
    ]

    y0, y1 = 92.0, 428.0
    yticks = [54.5, 55.0, 55.5, 56.0, 56.5, 57.0, 57.5]

    def sy(val: float) -> float:
        return y1 - (val - ymin) / (ymax - ymin) * (y1 - y0)

    for title, x_key, x0, x1, xmin, xmax, xticks, xlabel, lbl_map in panels:
        def sx(val: float) -> float:
            return x0 + (val - xmin) / (xmax - xmin) * (x1 - x0)

        # Panel card
        svg.append(f'  <rect x="{x0}" y="{y0}" width="{x1 - x0}" height="{y1 - y0}" rx="6" fill="#ffffff" '
                   f'stroke="#e2e8f0" stroke-width="1.2" filter="url(#card-shadow)"/>')
        svg.append(f'  <text x="{x0}" y="{y0 - 11}" text-anchor="start" font-size="13.5" font-weight="700" fill="#0f172a">{title}</text>')

        # "Better ↗" badge in top-right of panel header
        svg.append(f'  <rect x="{x1 - 74}" y="{y0 - 25}" width="74" height="19" rx="9.5" fill="#ffedd5" stroke="#fed7aa" stroke-width="1"/>')
        svg.append(f'  <text x="{x1 - 37}" y="{y0 - 12}" text-anchor="middle" font-size="10.5" font-weight="700" fill="#9a3412">Better ↗</text>')

        # Gridlines & ticks
        for xt in xticks:
            px = sx(xt)
            if x0 < px < x1:
                svg.append(f'  <line x1="{px:.1f}" y1="{y0}" x2="{px:.1f}" y2="{y1}" stroke="#f1f5f9" stroke-width="1"/>')
            svg.append(f'  <line x1="{px:.1f}" y1="{y1}" x2="{px:.1f}" y2="{y1 + 5}" stroke="#64748b" stroke-width="1.2"/>')
            svg.append(f'  <text x="{px:.1f}" y="{y1 + 18}" text-anchor="middle" font-size="11" font-weight="500" fill="#475569">{xt}</text>')

        for yt in yticks:
            py = sy(yt)
            if y0 < py < y1:
                svg.append(f'  <line x1="{x0}" y1="{py:.1f}" x2="{x1}" y2="{py:.1f}" stroke="#f1f5f9" stroke-width="1"/>')
            svg.append(f'  <line x1="{x0 - 5}" y1="{py:.1f}" x2="{x0}" y2="{py:.1f}" stroke="#64748b" stroke-width="1.2"/>')
            svg.append(f'  <text x="{x0 - 9}" y="{py + 4:.1f}" text-anchor="end" font-size="11" font-weight="500" fill="#475569">{yt:.1f}</text>')

        # Left & bottom primary axis spines
        svg.append(f'  <line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="#64748b" stroke-width="1.4"/>')
        svg.append(f'  <line x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}" stroke="#64748b" stroke-width="1.4"/>')

        # Axis titles
        svg.append(f'  <text x="{(x0 + x1) / 2:.1f}" y="{y1 + 38}" text-anchor="middle" font-size="12" font-weight="700" fill="#1e293b">{xlabel}</text>')
        yl_x = x0 - 44
        svg.append(f'  <text x="{yl_x:.1f}" y="{(y0 + y1) / 2:.1f}" text-anchor="middle" font-size="12" font-weight="700" fill="#1e293b" '
                   f'transform="rotate(-90 {yl_x:.1f} {(y0 + y1) / 2:.1f})">WMT23 chrF (0–100) ↑</text>')

        # Horizontal dotted BF16 reference line + subtle right-aligned label
        py_bf16 = sy(bf16_y)
        svg.append(f'  <line x1="{x0}" y1="{py_bf16:.1f}" x2="{x1}" y2="{py_bf16:.1f}" stroke="#94a3b8" stroke-width="1.4" stroke-dasharray="3,4"/>')
        svg.append(f'  <text x="{x1 - 8}" y="{py_bf16 + 13:.1f}" text-anchor="end" font-size="9.5" font-weight="600" '
                   f'font-style="italic" fill="#64748b">BF16 baseline (57.05)</text>')

        # Connect comparable quantization & group-size variants (ordered left-to-right along X)
        for _, grp_ids, dash in comparable_groups:
            ordered = sorted(grp_ids, key=lambda k: float(by_id[k][x_key]))
            pts = " ".join(f"{sx(float(by_id[k][x_key])):.1f},{sy(float(by_id[k]['chrf_ref'])):.1f}" for k in ordered)
            dash_attr = f' stroke-dasharray="{dash}"' if dash != "none" else ""
            svg.append(f'  <polyline points="{pts}" fill="none" stroke="#e2e8f0" stroke-width="6.0" '
                       f'stroke-linecap="round" stroke-linejoin="round" opacity="0.75"/>')
            svg.append(f'  <polyline points="{pts}" fill="none" stroke="#64748b" stroke-width="2.1" '
                       f'stroke-linecap="round" stroke-linejoin="round"{dash_attr}/>')

        # Leader lines first (underneath markers)
        for rid, (color, shape, short_label) in meta.items():
            r = by_id[rid]
            px = sx(float(r[x_key]))
            py = sy(float(r["chrf_ref"]))
            dx, dy, anchor = lbl_map[rid]
            lx = px + dx * 0.80
            ly = py + dy * 0.76
            lcol = "#fdba74" if rid.startswith("kidi") and rid != "kidi_bf16" else "#cbd5e1"
            svg.append(f'  <line x1="{px:.1f}" y1="{py:.1f}" x2="{lx:.1f}" y2="{ly:.1f}" stroke="{lcol}" stroke-width="1.1"/>')

        # Markers & labels (with white halo paint-order on text for crisp legibility)
        for rid, (color, shape, short_label) in meta.items():
            r = by_id[rid]
            px = sx(float(r[x_key]))
            py = sy(float(r["chrf_ref"]))
            dx, dy, anchor = lbl_map[rid]
            is_kidi = rid.startswith("kidi") and rid != "kidi_bf16"
            if is_kidi:
                svg.append(f'  <circle cx="{px:.1f}" cy="{py:.1f}" r="10.0" fill="{color}" opacity="0.12"/>')
            if shape == "star":
                svg.append(f'  <polygon points="{star_points(px, py)}" fill="{color}" stroke="#ffffff" stroke-width="1.2"/>')
            elif shape == "triangle":
                svg.append(f'  <polygon points="{triangle_points(px, py)}" fill="{color}" stroke="#ffffff" stroke-width="1.4"/>')
            elif shape == "solid":
                svg.append(f'  <circle cx="{px:.1f}" cy="{py:.1f}" r="6.0" fill="{color}" stroke="#ffffff" stroke-width="1.6"/>')
            else:
                svg.append(f'  <circle cx="{px:.1f}" cy="{py:.1f}" r="5.6" fill="#ffffff" stroke="{color}" stroke-width="2.6"/>')
            fw = "700" if rid.startswith("kidi") else "600"
            svg.append(f'  <text x="{px + dx:.1f}" y="{py + dy + 3.5:.1f}" text-anchor="{anchor}" font-size="10.5" '
                       f'font-weight="{fw}" fill="{color}" stroke="#ffffff" stroke-width="3.2" '
                       f'stroke-linejoin="round" paint-order="stroke">{short_label}</text>')

    # Sleek bottom legend pill bar
    svg.append('  <rect x="64" y="482" width="1072" height="62" rx="10" fill="#ffffff" stroke="#e2e8f0" stroke-width="1.2"/>')
    leg_y1 = 504.0
    runtime_entries = [
        ("#D55E00", "Kidi (Ours)"),
        ("#0173B2", "Arm ExecuTorch v1.5.1"),
        ("#029E73", "ONNX Runtime v1.31.1"),
        ("#CC78BC", "llama.cpp"),
        ("#DE8F05", "Google LiteRT-LM v0.17.1"),
    ]
    lx_cur = 212.0
    svg.append(f'  <text x="90" y="{leg_y1}" font-size="11" font-weight="700" fill="#475569">Runtimes:</text>')
    for col, text in runtime_entries:
        svg.append(f'  <rect x="{lx_cur}" y="{leg_y1 - 9.5}" width="11" height="11" rx="3" fill="{col}"/>')
        fw = "700" if "Kidi" in text else "600"
        svg.append(f'  <text x="{lx_cur + 16}" y="{leg_y1}" font-size="11" font-weight="{fw}" fill="#0f172a">{text}</text>')
        lx_cur += len(text) * 6.3 + 42

    leg_y2 = 530.0
    shape_entries = [
        ("star",     "none",     "BF16 Reference"),
        ("solid",    "none",     "Q8 (8-bit)"),
        ("hollow",   "7,4",      "Q4 (4-bit, g=32)"),
        ("hollow",   "3,3",      "Q4 (4-bit, g=128)"),
        ("triangle", "7,3,2,3",  "MixedQ (2/4/8-bit QAT)"),
    ]
    lx_cur = 212.0
    svg.append(f'  <text x="90" y="{leg_y2}" font-size="11" font-weight="700" fill="#475569">Comparable Tiers:</text>')
    for shape, dash, text in shape_entries:
        if shape != "star":
            dash_attr = f' stroke-dasharray="{dash}"' if dash != "none" else ""
            svg.append(f'  <line x1="{lx_cur - 6}" y1="{leg_y2 - 4}" x2="{lx_cur + 22}" y2="{leg_y2 - 4}" '
                       f'stroke="#64748b" stroke-width="2.1"{dash_attr}/>')
        if shape == "star":
            svg.append(f'  <polygon points="{star_points(lx_cur + 8, leg_y2 - 4, 6.8, 2.9)}" fill="#0f172a"/>')
        elif shape == "solid":
            svg.append(f'  <circle cx="{lx_cur + 8}" cy="{leg_y2 - 4}" r="4.8" fill="#334155" stroke="#ffffff" stroke-width="1.2"/>')
        elif shape == "hollow":
            svg.append(f'  <circle cx="{lx_cur + 8}" cy="{leg_y2 - 4}" r="4.6" fill="#ffffff" stroke="#334155" stroke-width="2.3"/>')
        elif shape == "triangle":
            svg.append(f'  <polygon points="{triangle_points(lx_cur + 8, leg_y2 - 3.5, 6.0)}" fill="#334155" stroke="#ffffff" stroke-width="1.0"/>')
        svg.append(f'  <text x="{lx_cur + 28}" y="{leg_y2}" font-size="11" font-weight="600" fill="#0f172a">{text}</text>')
        lx_cur += len(text) * 5.9 + 52

    svg.append('</svg>')
    return "\n".join(svg) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "tmp.models/gemma-4-E2B-it")
    parser.add_argument("--work-dir", type=Path, default=ROOT / ".cache/benchmarks/arm-gemma4")
    parser.add_argument("--executorch-source", type=Path, default=ROOT / ".cache/arm-gemma/1.5.1/executorch")
    parser.add_argument("--profile", choices=tuple(PROFILES), default="int4-g128")
    parser.add_argument("--include-onnx", action="store_true")
    parser.add_argument("--onnx-python", type=Path, default=ROOT / ".cache/ort-venv/bin/python")
    parser.add_argument("--llamacpp-model", type=Path)
    parser.add_argument("--litert-model", type=Path)
    parser.add_argument("--litert-python", type=Path, default=ROOT / ".cache/gemma-venv/bin/python")
    parser.add_argument("--kidi-qat-model", type=Path)
    parser.add_argument("--no-bf16", action="store_true", help="skip unquantized BF16 reference model")
    parser.add_argument("--eval-lines", type=int, default=5,
                        help="number of WMT23 translation sentences for chrF quality evaluation (0 to skip)")
    parser.add_argument("--eval-langs", default="ja,uk,de")
    parser.add_argument("--eval-seed", type=int, default=42)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--render-tables", action="store_true",
                        help="render the four Markdown benchmark tables from results.tsv and exit")
    parser.add_argument("--render-svg", type=Path,
                        help="render the Pareto frontier and throughput SVG chart to the given path and exit")
    parser.add_argument("--tsv", type=Path, default=HERE / "results.tsv",
                        help="path to results.tsv used by --render-tables")
    parser.add_argument("--force", action="store_true", help="ignore cached per-runtime JSON results and rerun all")
    parser.add_argument("--rerun", action="append", default=[], help="force rerunning specific runtime(s) by name")
    parser.add_argument("--skip", action="append", default=[], help="skip specific runtime(s) by name")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=900)
    arguments = parser.parse_args(argv)
    if arguments.render_svg:
        svg_path = arguments.render_svg.expanduser().resolve()
        svg_path.parent.mkdir(parents=True, exist_ok=True)
        svg_path.write_text(render_pareto_svg(arguments.tsv.expanduser().resolve()), encoding="utf-8")
        print(f"Wrote {svg_path}")
        return 0
    if arguments.render_tables:
        print(render_markdown_tables(arguments.tsv.expanduser().resolve()))
        return 0
    if (arguments.jobs < 1 or not 1 <= arguments.threads <= (os.cpu_count() or 1) or not math.isfinite(arguments.timeout) or
            arguments.timeout <= 0):
        parser.error("expected positive jobs, threads and timeout")
    profile = PROFILES[arguments.profile]
    work = arguments.work_dir.expanduser().resolve()
    source = arguments.executorch_source.expanduser().resolve()
    work.mkdir(parents=True, exist_ok=True)
    report = {"version": 1, "profile": arguments.profile, "status": "running",
              "started": datetime.now(timezone.utc).isoformat(), "runs": {}}
    try:
        if arguments.prepare:
            prepare(arguments.model.expanduser().resolve(), work, source, arguments.profile)
        if arguments.prepare_only:
            if not arguments.prepare:
                raise ValueError("--prepare-only requires --prepare")
            return 0
        manifest = json.loads((work / "manifest.json").read_text())
        if (manifest["precision"] != profile["precision"] or
                manifest.get("profile", "int4-g128") != arguments.profile or
                manifest["executorch_commit"] != EXECUTORCH_COMMIT):
            raise ValueError("prepared precision or ExecuTorch revision differs")
        for filename, checksum in manifest["files"].items():
            if digest(work / filename) != checksum:
                raise ValueError(f"shared checkpoint changed: {filename}")
        if digest(work / manifest["pte"]) != manifest["pte_sha256"]:
            raise ValueError("ExecuTorch PTE changed")
        if arguments.build:
            build(work, source, arguments.jobs)
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(str(work / "shared/tokenizer.json"))
        serialized = "<bos><|turn>user\n" + PROMPT + "<turn|>\n<|turn>model\n"
        protocol = {
            "prompt": PROMPT, "input_token_ids": tokenizer.encode(serialized, add_special_tokens=False).ids,
            "output_tokens": 64, "context_tokens": 1024, "warmups": 1, "repeats": 3,
            "threads": arguments.threads, "batch": 1, "profile": arguments.profile,
            "precision": profile["precision"],
            "decoding": "independent greedy; identical weights, prompt IDs and output budget; outputs may differ",
            "timing": "prefill includes first-token delivery; decode is the first-to-last token interval",
        }
        save(work / "protocol.json", protocol)
        settings = {"prompt": PROMPT, "workload": "short", "output_tokens": 64, "context_tokens": 1024,
                    "prefill_chunk_tokens": 1024, "packed_prefill": True, "warmups": 1}
        commands = {}
        bf16_model_dir = arguments.model.expanduser().resolve()
        if not arguments.no_bf16 and (bf16_model_dir / "model.safetensors").exists():
            commands["kidi_bf16"] = [
                str(work / "build-kidi/kidi_android_baseline"), "gemma",
                str(bf16_model_dir),
                str(arguments.threads), "3", json.dumps(settings),
            ]
        commands["kidi"] = [
            str(work / "build-kidi/kidi_android_baseline"), "gemma", str(work / "shared"),
            str(arguments.threads), "3", json.dumps(settings),
        ]
        asym_clip_dir = None
        asym_g32_dir = None
        if arguments.profile == "int4-g128" and (bf16_model_dir / "model.safetensors").exists():
            asym_clip_dir = prepare_asym_clip(
                bf16_model_dir, work, profile["group_size"] or 128, "shared-asym-clip"
            )
            commands["kidi_asym_clip"] = [
                str(work / "build-kidi/kidi_android_baseline"), "gemma", str(asym_clip_dir),
                str(arguments.threads), "3", json.dumps(settings),
            ]
            asym_g32_dir = prepare_asym_clip(bf16_model_dir, work, 32, "shared-asym-g32")
            commands["kidi_asym_g32"] = [
                str(work / "build-kidi/kidi_android_baseline"), "gemma", str(asym_g32_dir),
                str(arguments.threads), "3", json.dumps(settings),
            ]
        commands["arm"] = [
            str(work / "build-arm/arm_gemma4_bench"), str(work / manifest["pte"]),
            str(work / "protocol.json"),
        ]
        if arguments.include_onnx:
            onnx_python = str(arguments.onnx_python.expanduser().absolute())
            if arguments.prepare or not (work / "gemma4-e2b-shared.onnx").exists():
                subprocess.run(
                    [onnx_python, str(HERE / "onnx_bench.py"), str(work / "shared"),
                     str(work / "protocol.json"), "--export-only"],
                    cwd=ROOT, check=True, timeout=arguments.timeout,
                )
            commands["onnx"] = [
                onnx_python, str(HERE / "onnx_bench.py"),
                str(work / "shared"), str(work / "protocol.json"),
            ]
        if arguments.llamacpp_model:
            commands["llamacpp"] = [
                str(work / "build-arm/llamacpp_gemma4_bench"),
                str(arguments.llamacpp_model.expanduser().resolve()),
                str(work / "protocol.json"),
            ]
        if arguments.kidi_qat_model:
            commands["kidi_qat"] = [
                str(work / "build-kidi/kidi_android_baseline"), "gemma",
                str(arguments.kidi_qat_model.expanduser().resolve()),
                str(arguments.threads), "3", json.dumps(settings),
            ]
        if arguments.litert_model:
            commands["litert"] = [
                str(arguments.litert_python.expanduser().absolute()),
                str(HERE / "litert_bench.py"),
                str(arguments.litert_model.expanduser().resolve()),
                str(work / "protocol.json"),
            ]
        eval_protocol = None
        eval_commands = {}
        if arguments.eval_lines > 0:
            eval_protocol = build_eval_protocol(
                work, tokenizer, arguments.eval_lines, arguments.eval_langs,
                arguments.eval_seed, 192, 1024, arguments.threads,
            )
            eval_protocol_path = work / "eval_protocol.json"
            save(eval_protocol_path, eval_protocol)
            eval_settings = {
                "workload": "eval",
                "eval_protocol": str(eval_protocol_path),
                "output_tokens": 192,
                "context_tokens": 1024,
                "prefill_chunk_tokens": 1024,
                "packed_prefill": True,
                "warmups": 1,
            }
            eval_commands["arm"] = [
                str(work / "build-arm/arm_gemma4_bench"), str(work / manifest["pte"]),
                str(eval_protocol_path),
            ]
            eval_commands["kidi"] = [
                str(work / "build-kidi/kidi_android_baseline"), "gemma", str(work / "shared"),
                str(arguments.threads), "1", json.dumps(eval_settings),
            ]
            if asym_clip_dir is not None:
                eval_commands["kidi_asym_clip"] = [
                    str(work / "build-kidi/kidi_android_baseline"), "gemma", str(asym_clip_dir),
                    str(arguments.threads), "1", json.dumps(eval_settings),
                ]
            if asym_g32_dir is not None:
                eval_commands["kidi_asym_g32"] = [
                    str(work / "build-kidi/kidi_android_baseline"), "gemma", str(asym_g32_dir),
                    str(arguments.threads), "1", json.dumps(eval_settings),
                ]
            if arguments.include_onnx:
                eval_commands["onnx"] = [
                    str(arguments.onnx_python.expanduser().absolute()), str(HERE / "onnx_bench.py"),
                    str(work / "shared"), str(eval_protocol_path),
                ]
            if arguments.llamacpp_model:
                eval_commands["llamacpp"] = [
                    str(work / "build-arm/llamacpp_gemma4_bench"),
                    str(arguments.llamacpp_model.expanduser().resolve()),
                    str(eval_protocol_path),
                ]
            if arguments.litert_model:
                eval_commands["litert"] = [
                    str(arguments.litert_python.expanduser().absolute()),
                    str(HERE / "litert_bench.py"),
                    str(arguments.litert_model.expanduser().resolve()),
                    str(eval_protocol_path),
                ]
            if arguments.kidi_qat_model:
                eval_commands["kidi_qat"] = [
                    str(work / "build-kidi/kidi_android_baseline"), "gemma",
                    str(arguments.kidi_qat_model.expanduser().resolve()),
                    str(arguments.threads), "1", json.dumps(eval_settings),
                ]
            if "kidi_bf16" in commands:
                eval_commands["kidi_bf16"] = [
                    str(work / "build-kidi/kidi_android_baseline"), "gemma",
                    str(bf16_model_dir),
                    str(arguments.threads), "1", json.dumps(eval_settings),
                ]
        report.update(
            protocol=protocol, export=manifest,
            host={"platform": platform.platform(), "processor": platform.processor(), "cpus": os.cpu_count(),
                  "load_average": os.getloadavg()},
            source={"kidi_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    "diff_sha256": hashlib.sha256(subprocess.check_output(["git", "diff", "HEAD"], cwd=ROOT)).hexdigest(),
                    "benchmark_files": {name: digest(HERE / name) for name in
                                        ("run.py", "executorch_bench.cpp", "llamacpp_bench.cpp",
                                         "litert_bench.py", "onnx_bench.py",
                                         "CMakeLists.txt", "requirements.txt")}},
        )
        if eval_protocol is not None:
            report["eval_protocol"] = {
                key: eval_protocol[key] for key in ("task", "langs", "seed", "items", "max_output_tokens")
            }
        if sys.platform == "darwin":
            report["host"]["cpu_brand"] = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
            report["host"]["ram_bytes"] = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True))
        rerun_set = set(arguments.rerun)
        skip_set = set(arguments.skip)
        for skipped in skip_set:
            commands.pop(skipped, None)
            eval_commands.pop(skipped, None)
        for name, command in commands.items():
            checksum = digest(Path(command[0]))
            cached = None
            if not arguments.force and not arguments.prepare and name not in rerun_set:
                cached = load_cached_run(work, name, command, checksum, protocol, tokenizer)
            if cached is not None:
                report["runs"][name] = cached
            else:
                report["runs"][name] = measure(command, work, name, protocol, arguments.timeout)
                report["runs"][name]["binary_sha256"] = checksum
                report["runs"][name]["summary"]["text"] = tokenizer.decode(
                    report["runs"][name]["summary"]["token_ids"], skip_special_tokens=False)
                if name == "kidi":
                    loads = [row for row in report["runs"][name]["records"] if row.get("stage") == "load"]
                    if len(loads) != 1 or loads[0].get("native_qat") is not True:
                        raise ValueError("KIDI did not execute the frozen packed-weight checkpoint")
                if digest(Path(command[0])) != checksum:
                    raise ValueError(f"{name} binary changed during measurement")
            report["runs"][name]["summary"].update(runtime_binary_metrics(name, command))
            if eval_protocol is not None and name in eval_commands:
                eval_cmd = eval_commands[name]
                eval_cached = None
                force_eval = arguments.force or arguments.prepare or name in rerun_set
                if not force_eval:
                    eval_cached = load_cached_eval(work, name, eval_cmd, checksum, eval_protocol, tokenizer)
                if eval_cached is None:
                    eval_cached = measure_eval(
                        eval_cmd, work, name, eval_protocol, tokenizer, arguments.timeout, force=force_eval
                    )
                    eval_cached["binary_sha256"] = checksum
                    save_cached_eval(work, name, eval_cached, eval_protocol)
                report["runs"][name]["eval"] = eval_cached["summary"]
                report["runs"][name]["eval_cached"] = eval_cached["cached"]
                report["runs"][name]["summary"].update(
                    chrf=eval_cached["summary"]["chrf"],
                    eval_seconds=eval_cached["summary"]["eval_seconds"],
                    eval_tok_per_s=eval_cached["summary"]["eval_tok_per_s"],
                )
            save_cached_run(work, name, report["runs"][name], protocol)
            save(work / "benchmark.json", report)
        for filename, checksum in manifest["files"].items():
            if digest(work / filename) != checksum:
                raise ValueError(f"shared checkpoint changed during measurement: {filename}")
        if digest(work / manifest["pte"]) != manifest["pte_sha256"]:
            raise ValueError("ExecuTorch PTE changed during measurement")
        summaries = {name: run["summary"] for name, run in report["runs"].items()}
        if eval_protocol is not None:
            from sacrebleu.metrics import CHRF
            bf16_hyps = None
            if "kidi_bf16" in report["runs"] and "eval" in report["runs"]["kidi_bf16"]:
                bf16_hyps = report["runs"]["kidi_bf16"]["eval"].get("hypotheses")
            if not bf16_hyps:
                partial_bf16 = load_partial_eval_items(work, "kidi_bf16", eval_protocol, tokenizer)
                if len(partial_bf16) == eval_protocol["items"]:
                    bf16_hyps = [partial_bf16[idx]["output"] for idx in range(eval_protocol["items"])]
            if bf16_hyps and len(bf16_hyps) == eval_protocol["items"]:
                for name, run in report["runs"].items():
                    if "eval" in run and len(run["eval"].get("hypotheses", [])) == len(bf16_hyps):
                        score_bf16 = (100.0 if name == "kidi_bf16" else
                                      float(CHRF().corpus_score(run["eval"]["hypotheses"], [bf16_hyps]).score))
                        run["eval"]["chrf_bf16"] = score_bf16
                        run["summary"]["chrf_bf16"] = score_bf16
        comparison = compare(summaries)
        report["kidi_speedup"] = comparison["speedup"]
        report["output_comparison"] = comparison["outputs"]
        report.update(status="completed", completed=datetime.now(timezone.utc).isoformat())
        save(work / "benchmark.json", report)
        if eval_protocol is not None:
            print("runtime       prefill tok/s   decode tok/s   TTFT ms   peak GiB   chrF(ref)   chrF(bf16)   eval s   source")
            for name, row in summaries.items():
                source_tag = "cached" if (report["runs"][name].get("cached") and report["runs"][name].get("eval_cached", True)) else "measured"
                print(f'{name:12} {row["prefill_tps"]:13.2f} {row["decode_tps"]:14.2f}'
                      f' {row["ttft_ms"]:9.2f} {row["peak_rss_gib"]:10.2f}'
                      f' {row.get("chrf", 0.0):11.2f} {row.get("chrf_bf16", 0.0):12.2f}'
                      f' {row.get("eval_seconds", 0.0):8.2f}   {source_tag}')
        else:
            print("runtime       prefill tok/s   decode tok/s   TTFT ms   peak GiB   bin MiB   source")
            for name, row in summaries.items():
                source_tag = "cached" if report["runs"][name].get("cached") else "measured"
                print(f'{name:12} {row["prefill_tps"]:13.2f} {row["decode_tps"]:14.2f}'
                      f' {row["ttft_ms"]:9.2f} {row["peak_rss_gib"]:10.2f} {row["binary_size_mib"]:9.2f}   {source_tag}')
        print(f'Output comparison: {report["output_comparison"]}')
        print(f'Report: {work / "benchmark.json"}')
        return 0
    except (OSError, ValueError, KeyError, RuntimeError, ImportError, subprocess.SubprocessError) as error:
        report.update(status="failed", error=str(error))
        save(work / "benchmark.json", report)
        print(f"Matched Arm benchmark failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
