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

## Benchmark results

Measured on an **Apple M5** CPU (`arm64`, 16 GB unified memory, macOS) using:

- **Model:** `google/gemma-4-E2B-it` (35 layers, 1,536 hidden dim, 15 KV-donor
  layers + 20 KV-sharing layers, 262,144 vocab + per-layer embeddings)
- **Workload:** 4 CPU threads, batch 1, 1,024-token context, 31-token prompt
  (`"Explain how binary search works in at least 300 words, including an example and its time complexity."`),
  64 greedy output tokens (31 prefill tokens + first token for prefill/TTFT;
  63 decode steps for decode throughput), 1 excluded warmup + 3 measured
  independent runs (median reported, with min–max decode range).

### 1. Matched `int4-g128` profile (`--profile int4-g128`)

`kidi`, `arm`, and `onnx` execute over the **exact same** exported `int4-g128`
weights and FP32 scales (`shared/model.safetensors`); `kidi_base` quantizes the
original BF16 checkpoint on the fly into `int4-g128`; `llamacpp` runs a pure
`Q4_0` (`g=32`) GGUF with `GGML_CPU_KLEIDIAI=ON`.

| Runtime | Engine / Backend | Weight Format | Prefill (tok/s) | Decode (tok/s) | Decode Range (tok/s) | TTFT (ms) | Peak RSS (GiB) |
|---|---|---|---:|---:|---:|---:|---:|
| **KIDI (`kidi`)** | YNNPACK (`arm64` SME/DOTPROD) | Symmetric INT4 (`g=128`) | **531.24** | **50.67** | 50.65–50.70 | **58.35** | 1.68 |
| **KIDI (`kidi_base`)** | YNNPACK (runtime-packed from BF16) | Symmetric INT4 (`g=128`) | **526.43** | **49.95** | 49.94–50.03 | **58.89** | 1.80 |
| **ONNX Runtime v1.31.1 (`onnx`)** | MLAS + Arm KleidiAI (`MatMulNBits`) | Symmetric INT4 (`g=128`) | 396.27 | 50.65 | 50.31–50.91 | 78.23 | **1.45** |
| **llama.cpp (`llamacpp`)** | GGML CPU + Arm KleidiAI | `Q4_0` (`g=32`) | 145.93 | 47.82 | 47.30–47.89 | 212.42 | 3.88 |
| **Arm ExecuTorch v1.5.1 (`arm`)** | XNNPACK + Arm KleidiAI (`8da4w+emb4`) | Symmetric INT4 (`g=128`) | 234.11 | 43.51 | 43.44–43.64 | 132.42 | 1.84 |

- **vs. Arm ExecuTorch v1.5.1 (`XNNPACK + KleidiAI`):** KIDI achieves **2.27x
  faster prefill** (`531.24` vs. `234.11 tok/s`, reducing TTFT from `132.42 ms`
  to `58.35 ms`) and **1.16x faster decode** (`50.67` vs. `43.51 tok/s`) with
  lower peak RSS (`1.68 GiB` vs. `1.84 GiB`).

### 2. Matched `int8` per-channel profile (`--profile int8`)

`kidi`, `arm`, and `onnx` execute over the **exact same** per-channel symmetric
INT8 weights and FP32 scales; `kidi_base` quantizes the BF16 checkpoint on the
fly into per-channel INT8; `llamacpp` runs a pure `Q8_0` (`g=32`) GGUF with
`GGML_CPU_KLEIDIAI=ON`.

| Runtime | Engine / Backend | Weight Format | Prefill (tok/s) | Decode (tok/s) | Decode Range (tok/s) | TTFT (ms) | Peak RSS (GiB) |
|---|---|---|---:|---:|---:|---:|---:|
| **KIDI (`kidi_base`)** | YNNPACK (runtime-packed from BF16) | Symmetric INT8 per-channel | **586.03** | **33.46** | 29.14–33.48 | **52.90** | 2.96 |
| **KIDI (`kidi`)** | YNNPACK (`arm64` SME/DOTPROD) | Symmetric INT8 per-channel | **578.64** | **33.34** | 32.29–33.40 | **53.57** | 3.16 |
| **llama.cpp (`llamacpp`)** | GGML CPU + Arm KleidiAI | `Q8_0` (`g=32`) | 184.33 | 33.30 | 33.25–33.39 | 168.17 | 6.58 |
| **ONNX Runtime v1.31.1 (`onnx`)** | MLAS + Arm KleidiAI (`MatMulIntegerToFloat`) | Symmetric INT8 per-channel | 309.40 | 31.29 | 31.18–31.39 | 100.19 | **2.25** |
| **Arm ExecuTorch v1.5.1 (`arm`)** | XNNPACK + Arm KleidiAI (`8da8w+emb8`) | Symmetric INT8 per-channel | 436.82 | 28.42 | 28.42–28.66 | 70.97 | 2.44 |

- **vs. Arm ExecuTorch v1.5.1 (`XNNPACK + KleidiAI`):** KIDI achieves **1.32x
  faster prefill** (`578.64` vs. `436.82 tok/s`) and **1.17x faster decode**
  (`33.34` vs. `28.42 tok/s`), sharing the first **30 generated tokens** before
  floating-point tie-breaking diverges at token 31.

### 3. Production mixed-precision / QAT reference models

Google distributes official mixed-precision mobile checkpoints for Gemma 4 E2B
that combine 4-bit transformer projections with 2-bit vocabulary/embedding
tables:

| Runtime | Checkpoint | Weight Format | Prefill (tok/s) | Decode (tok/s) | Decode Range (tok/s) | TTFT (ms) | Peak RSS (GiB) |
|---|---|---|---:|---:|---:|---:|---:|
| **KIDI (`kidi_qat`)** | `gemma-4-E2B-it-qat-mobile-transformers` | Static INT8 act + INT4 FFN/Attn + INT2 LM head + INT4 PLE | **863.39** | **70.49** | 70.15–70.55 | **35.91** | 2.06 |
| **Google LiteRT-LM v0.17.1 (`litert`)** | `gemma-4-E2B-it.litertlm` (`XNNPACK`) | `.litertlm` mixed INT4/INT2 | 159.74 | 47.54 | 47.49–47.77 | 194.07 | **1.31** |

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
