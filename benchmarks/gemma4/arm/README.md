# Minimal matched Arm CPU benchmark

Compare KIDI with **ExecuTorch v1.5.1 + XNNPACK + KleidiAI**, a current stable
release of the CPU stack used by
[Arm's LLM-Runner](https://github.com/Arm-Examples/LLM-Runner).
Use the same physical CPU, one loaded model, batch one, one fixed short prompt,
64 greedy output tokens (ignoring EOS), a 1,024-token context, four threads,
one excluded warmup, and three measured independent requests.

The driver supports two matched quantization profiles exported from the original
Gemma 4 E2B Safetensors checkpoint and supplies those exact integer values and
FP32 scales to both runtimes:

1. **`--profile int4-g128` (default, group-128 INT4 / `8da4w+emb4`):** Linear
   weights use symmetric INT4 (`[-7, 7]`, ties away from zero) with
   `group_size=128` and FP32 scales; activations use dynamic asymmetric
   per-token INT8; embeddings use per-row INT4; attention and KV caches use
   FP32. This matches the group-128 INT4 scheme used by Arm's official Gemma 4
   E2B ExecuTorch reference model (`Arm/gemma-4-e2b-text-executorch`).
2. **`--profile int8` (per-channel INT8 / `8da8w+emb8`):** Linear weights use
   symmetric per-output-channel INT8 (`[-127, 127]`, ties away from zero) with
   FP32 scales; activations use dynamic asymmetric per-token INT8; embeddings
   use per-row INT8; attention and KV caches use FP32.

Normalizations and layer scalars remain FP32. Activation and cache calibration
scales are zero: dynamic activations are enabled, and cache rounding is
disabled. Passing `--include-kidi-base` also benchmarks KIDI quantizing
directly from the unquantized BF16 checkpoint at load/first-use time.
Optional flags also benchmark:

- `--include-onnx`: ONNX Runtime CPU (`onnx_bench.py`, using `MatMulNBits` with
  `block_size=128` for `int4-g128` and `DynamicQuantizeLinear` +
  `MatMulIntegerToFloat` for `int8` over the shared quantized weights, backed
  by MLAS + Arm KleidiAI).
- `--llamacpp-model PATH`: `llama.cpp` + GGML with `GGML_CPU_KLEIDIAI=ON`
  (`llamacpp_bench.cpp` on `Q4_0` or `Q8_0` GGUF).
- `--litert-model PATH` & `--kidi-qat-model PATH`: Google's `litert-lm`
  (`litert_bench.py` on `gemma-4-E2B-it.litertlm`) paired with KIDI on Google's
  mixed 2/4/8-bit QAT checkpoint (`gemma-4-E2B-it-qat-mobile-transformers`).
- Per-runtime results are stored under `<work-dir>/results/<runtime>.json` and
  reused automatically when the protocol, command, and binary SHA-256 match.
  Pass `--rerun <runtime>` to rerun specific runtimes or `--force` for a clean
  rerun of all requested runtimes.

## Setup and run

From the repository root, create an isolated export/build environment:

```bash
uv venv --python 3.12 .cache/arm-gemma/venv
uv pip install --python .cache/arm-gemma/venv/bin/python \
  -r benchmarks/gemma4/arm/requirements.txt
mkdir -p .cache/arm-gemma/1.5.1
git clone --depth 1 --branch v1.5.1 \
  https://github.com/pytorch/executorch.git .cache/arm-gemma/1.5.1/executorch
git -C .cache/arm-gemma/1.5.1/executorch submodule update \
  --init --recursive --depth 1 -- \
  backends/xnnpack/third-party/FP16 backends/xnnpack/third-party/FXdiv \
  backends/xnnpack/third-party/XNNPACK backends/xnnpack/third-party/cpuinfo \
  backends/xnnpack/third-party/pthreadpool kernels/optimized/third-party/eigen \
  third-party/flatbuffers third-party/flatcc third-party/json third-party/gflags \
  third-party/ao extension/llm/tokenizers third-party/pocketfft
.cache/arm-gemma/venv/bin/python benchmarks/gemma4/arm/run.py \
  --model tmp.models/gemma-4-E2B-it --prepare --build
```

Reuse an existing clean checkout instead of cloning over it. Supply an original
unquantized local Gemma 4 E2B directory containing `model.safetensors`,
`config.json`, and `tokenizer.json`. Model acquisition is intentionally separate.
Native builds require Ninja; the ExecuTorch checkout's final directory must
be named `executorch`.
Export is CPU-only and packs large tables in chunks; it does not instantiate
the full 20 GB FP32 parameter set. Prepared weights/builds/logs/results stay
under `.cache/benchmarks/arm-gemma4/`, outside Git.

To repeat without exporting or building:

```bash
.cache/arm-gemma/venv/bin/python benchmarks/gemma4/arm/run.py
```

Use a new `--work-dir` to make another export; existing shared checkpoints are
not overwritten. `--executorch-source` selects the pinned source checkout.
The native ExecuTorch probe sets and verifies its singleton threadpool's size
before loading the model or running jobs, using the same API as Arm's LLM-Runner.

## Validation and metrics

Both runtimes receive identical input IDs. The driver checks output counts,
CPU/thread placement, finite timings, warmup exclusion, repetition determinism,
and zero prompt-cache reuse. Each runtime must produce identical token IDs
across its repetitions. The comparison uses **independent greedy generation**:
the two runtimes share weights, prompt IDs, and a 64-token budget, but generated
text may differ. Reports retain both texts/token sequences and the first
divergent token. Decode trajectories differ after divergence; this is not a
teacher-forced comparison or a broad quality evaluation.

Prefill throughput includes submission through first-token delivery; decode
throughput is the remaining 63 tokens divided by the first-to-last token
interval. TTFT excludes model loading and tokenization. Peak process RSS includes
loading/warmup and is not incremental memory. Reports contain raw records,
commands, package/source revisions, model/binary hashes, and quantization.
GPU backends are disabled in the ExecuTorch build and KIDI is forced onto CPU.
Do not benchmark while exporting/building or under unrelated heavy host load.
