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
import shutil
from statistics import median
import subprocess
import sys
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
    config_document["quantization_config"] = quantization_config
    config_document["text_config"].update(dtype="float32", packed_weights_signed=True)
    save(canonical / "config.json", config_document)
    consolidate_weights(canonical, files)
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
    return {"command": command, "records": records, "summary": summarize(records, protocol)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "tmp.models/gemma-4-E2B-it")
    parser.add_argument("--work-dir", type=Path, default=ROOT / ".cache/benchmarks/arm-gemma4")
    parser.add_argument("--executorch-source", type=Path, default=ROOT / ".cache/arm-gemma/1.5.1/executorch")
    parser.add_argument("--profile", choices=tuple(PROFILES), default="int4-g128")
    parser.add_argument("--include-kidi-base", action="store_true")
    parser.add_argument("--llamacpp-model", type=Path)
    parser.add_argument("--litert-model", type=Path)
    parser.add_argument("--litert-python", type=Path, default=ROOT / ".cache/gemma-venv/bin/python")
    parser.add_argument("--kidi-qat-model", type=Path)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=900)
    arguments = parser.parse_args(argv)
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
        commands = {
            "arm": [str(work / "build-arm/arm_gemma4_bench"), str(work / manifest["pte"]),
                    str(work / "protocol.json")],
            "kidi": [str(work / "build-kidi/kidi_android_baseline"), "gemma", str(work / "shared"),
                     str(arguments.threads), "3", json.dumps(settings)],
        }
        if arguments.include_kidi_base:
            base_settings = {**settings, "weight_bits": profile["bits"],
                             "group_size": profile["group_size"] or 128}
            commands["kidi_base"] = [
                str(work / "build-kidi/kidi_android_baseline"), "gemma",
                str(arguments.model.expanduser().resolve()),
                str(arguments.threads), "3", json.dumps(base_settings),
            ]
        if arguments.llamacpp_model:
            commands["llamacpp"] = [
                str(work / "build-arm/llamacpp_gemma4_bench"),
                str(arguments.llamacpp_model.expanduser().resolve()),
                str(work / "protocol.json"),
            ]
        if arguments.litert_model:
            commands["litert"] = [
                str(arguments.litert_python.expanduser().absolute()),
                str(HERE / "litert_bench.py"),
                str(arguments.litert_model.expanduser().resolve()),
                str(work / "protocol.json"),
            ]
        if arguments.kidi_qat_model:
            commands["kidi_qat"] = [
                str(work / "build-kidi/kidi_android_baseline"), "gemma",
                str(arguments.kidi_qat_model.expanduser().resolve()),
                str(arguments.threads), "3", json.dumps(settings),
            ]
        report.update(
            protocol=protocol, export=manifest,
            host={"platform": platform.platform(), "processor": platform.processor(), "cpus": os.cpu_count(),
                  "load_average": os.getloadavg()},
            source={"kidi_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    "diff_sha256": hashlib.sha256(subprocess.check_output(["git", "diff", "HEAD"], cwd=ROOT)).hexdigest(),
                    "benchmark_files": {name: digest(HERE / name) for name in
                                        ("run.py", "executorch_bench.cpp", "llamacpp_bench.cpp",
                                         "litert_bench.py", "CMakeLists.txt", "requirements.txt")}},
        )
        if sys.platform == "darwin":
            report["host"]["cpu_brand"] = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
            report["host"]["ram_bytes"] = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True))
        for name, command in commands.items():
            checksum = digest(Path(command[0]))
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
            save(work / "benchmark.json", report)
        for filename, checksum in manifest["files"].items():
            if digest(work / filename) != checksum:
                raise ValueError(f"shared checkpoint changed during measurement: {filename}")
        if digest(work / manifest["pte"]) != manifest["pte_sha256"]:
            raise ValueError("ExecuTorch PTE changed during measurement")
        summaries = {name: run["summary"] for name, run in report["runs"].items()}
        comparison = compare(summaries)
        report["kidi_speedup"] = comparison["speedup"]
        report["output_comparison"] = comparison["outputs"]
        report.update(status="completed", completed=datetime.now(timezone.utc).isoformat())
        save(work / "benchmark.json", report)
        print("runtime       prefill tok/s   decode tok/s   TTFT ms   peak GiB")
        for name, row in summaries.items():
            print(f'{name:12} {row["prefill_tps"]:13.2f} {row["decode_tps"]:14.2f}'
                  f' {row["ttft_ms"]:9.2f} {row["peak_rss_gib"]:10.2f}')
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
