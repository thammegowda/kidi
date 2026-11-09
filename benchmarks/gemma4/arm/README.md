# Gemma 4 E2B on Arm CPU: Kidi vs. Arm ExecuTorch, ONNX Runtime, llama.cpp & LiteRT-LM

This benchmark evaluates **Kidi** (a lightweight C++23 inference runtime built on
[`ynnpack-dev`](https://github.com/thammegowda/ynnpack-dev) and custom ARM64 NEON / SME2 kernels) against leading open-source Arm
CPU inference engines on **Google's Gemma 4 E2B (`google/gemma-4-E2B-it`)**:

- **Arm ExecuTorch v1.5.1** (`XNNPACK + Arm KleidiAI`, the stack powering
  [Arm's LLM-Runner](https://github.com/Arm-Examples/LLM-Runner))
- **ONNX Runtime v1.31.1** (`MLAS + Arm KleidiAI`)
- **`llama.cpp`** (`GGML CPU + Arm KleidiAI`)
- **Google LiteRT-LM v0.17.1** (`XNNPACK`)

All benchmark numbers are stored in [`results.tsv`](./results.tsv) and rendered
into the focused comparison tables below via `python benchmarks/gemma4/arm/run.py --render-tables`.

---

## Experimental Setup

- **Hardware:** Apple M5 (`arm64`, 16 GB unified memory, macOS), **4 CPU threads**,
  **batch size 1**, CPU-only execution.
- **Model Architecture:** `google/gemma-4-E2B-it` (35 transformer layers, 1,536
  hidden size, 15 KV-donor layers + 20 KV-sharing layers, 262,144 tied token
  embeddings + per-layer token embeddings).
- **Speed Protocol:** 1,024-token context window, fixed 31-token prompt, 64
  greedy output tokens (ignoring EOS), 1 excluded warmup + 3 measured
  repetitions (median reported).
  - **Prefill (tok/s) & TTFT (ms):** Measured from request submission through
    first-token delivery.
  - **Decode (tok/s):** Measured over the remaining 63 autoregressive decode
    steps.
- **Translation Quality Protocol (`chrF (ref)`, `chrF (bf16)` & `Eval Time`):**
  150 deterministic WMT23 test sentences (`ja->en`, `uk->en`, `de->en`, 50 per
  language, seed `42`) pre-formatted with the Gemma 4 chat template in
  `eval_input.jsonl` and decoded greedily up to `<turn|>` / `<eos>`. Two
  SacreBLEU corpus **chrF** (`sacrebleu==2.6.0`) metrics are reported:
  - **`chrF (ref)`:** Scored against WMT23 human reference translations
    (`eval_references.jsonl`).
  - **`chrF (bf16)`:** Scored against the unquantized **BF16 model outputs**
    (`kidi_bf16`) to measure direct quantization fidelity.

---

## Benchmark Results

<p align="center">
  <img src="../../../docs/gemma4-arm-pareto.svg" alt="Gemma 4 E2B on Arm CPU: Speed vs. Translation Quality Pareto Frontier" width="100%">
</p>

### 1. Reference BF16 & 8-Bit (`int8`) Inference

| Runtime | Engine / Backend | Weight Format | Model (GiB) | Prefill (tok/s) | Decode (tok/s) | TTFT (ms) | Peak RSS (GiB) | chrF (ref) | chrF (bf16) | Eval Time (s) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Kidi BF16 (kidi_bf16)** | YNNPACK (arm64 BF16) | BF16 (Reference) | 8.50 | 255.48 | 16.80 | 121.34 | 5.31 | 57.05 | 100.00 | 277.45 |
| **Kidi (kidi)** | YNNPACK (arm64 SME/DOTPROD) | INT8 per-channel (tied emb) | 4.37 | 583.12 | 32.92 | 53.16 | 3.19 | 57.16 | 94.34 | 146.32 |
| **Arm ExecuTorch v1.5.1 (arm)** | XNNPACK + KleidiAI (8da8w+emb8) | INT8 per-channel (tied emb) | 4.70 | 436.82 | 28.42 | 70.97 | 2.44 | 57.25 | 94.80 | 184.43 |
| **ONNX Runtime v1.31.1 (onnx)** | MLAS + KleidiAI (MatMulIntegerToFloat) | INT8 per-channel | 4.69 | 309.40 | 31.29 | 100.19 | 2.25 | 57.21 | 94.05 | 170.99 |
| **llama.cpp (llamacpp)** | GGML CPU + KleidiAI | Q8_0 (g=32, tied emb) | 4.61 | 184.33 | 33.30 | 168.17 | 6.58 | 56.95 | 94.48 | 177.41 |

**Takeaways (BF16 & 8-Bit):**
- **Near-lossless 8-bit fidelity:** Across all 150 WMT23 sentences, 8-bit
  quantization achieves **`94.05–94.80` `chrF (bf16)`** agreement with the
  unquantized BF16 reference (`57.05` `chrF (ref)`) while cutting model size in
  half (`8.50 GiB` $\rightarrow$ `4.37 GiB`).
- **Prefill vs. decode roofline:** Single-token 8-bit decode streams `2.48 GB`
  of active weights per token and saturates 4-core DRAM read bandwidth (`~81
  GB/s`), clustering all engines around `28–33 tok/s`. During compute-bound
  **prefill**, Kidi's SME2 kernels reach **583.12 tok/s** — **1.33x faster than
  Arm ExecuTorch**, **1.88x faster than ONNX Runtime**, and **3.16x faster than
  `llama.cpp`**, finishing the 150-sentence evaluation in **146.32 s** (vs.
  `170.99–184.43 s`).

### 2. 4-Bit Inference — Group Size 128 (`int4-g128`)

| Runtime | Engine / Backend | Weight Format | Model (GiB) | Prefill (tok/s) | Decode (tok/s) | TTFT (ms) | Peak RSS (GiB) | chrF (ref) | chrF (bf16) | Eval Time (s) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Kidi Symmetric (kidi)** | YNNPACK + NEON sdot (BlockwiseInt4Linear) | Symmetric INT4 (g=128, tied emb) | 2.27 | 544.13 | 58.23 | 56.97 | 1.44 | 55.45 | 81.50 | 103.57 |
| **Kidi Asymmetric+Clip (kidi_asym_clip)** | YNNPACK + NEON sdot (BlockwiseInt4Linear) | Asymmetric INT4 + MSE Clip (g=128, tied emb) | 2.27 | 537.91 | 56.37 | 57.63 | 1.50 | 55.63 | 84.65 | 101.54 |
| **Arm ExecuTorch v1.5.1 (arm)** | XNNPACK + KleidiAI (8da4w+emb4) | Symmetric INT4 (g=128, tied emb) | 2.39 | 234.11 | 43.51 | 132.42 | 1.84 | 54.87 | 80.23 | 160.32 |
| **ONNX Runtime v1.31.1 (onnx)** | MLAS + KleidiAI (MatMulNBits) | Symmetric INT4 (g=128, tied emb) | 2.41 | 396.27 | 50.65 | 78.23 | 1.45 | 54.96 | 80.49 | 103.67 |

**Takeaways (`int4-g128`):**
- **Matched `g=128` symmetric weights:** Kidi (`kidi`) reaches **544.13 tok/s
  prefill** (**2.32x faster** than Arm ExecuTorch, **1.37x faster** than ONNX
  Runtime) and **58.23 tok/s decode** (**1.34x faster** than Arm ExecuTorch,
  **1.15x faster** than ONNX Runtime) at the lowest memory footprint (**1.44
  GiB** Peak RSS) and smallest model size (**2.27 GiB**).
- **Recovering 4-bit quality with Asymmetric + MSE Outlier Clipping:** Standard
  symmetric `g=128` drops to `80.23–81.50` `chrF (bf16)` (`54.87–55.45` `chrF
  (ref)`). Applying blockwise asymmetric quantization (`0..15`) with an
  11-step MSE outlier clipping search
  ([blog post](https://gowda.ai/posts/2026/08/llm-fast-inference/),
  [WMT 2026 paper](https://aclanthology.org/2026.wmt-main.136/)) and FP16
  scale/zero metadata (`kidi_asym_clip`) keeps the exact same **2.27 GiB** model
  size while jumping `chrF (bf16)` by **`+4.42` points** over Arm ExecuTorch
  (`84.65` vs. `80.23`, and `55.63` vs. `54.87` `chrF (ref)`) and completing
  WMT23-150 in **101.54 s**.

### 3. 4-Bit Inference — Group Size 32 (`int4-g32`)

| Runtime | Engine / Backend | Weight Format | Model (GiB) | Prefill (tok/s) | Decode (tok/s) | TTFT (ms) | Peak RSS (GiB) | chrF (ref) | chrF (bf16) | Eval Time (s) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Kidi Asymmetric+Clip (kidi_asym_g32)** | YNNPACK + NEON sdot (BlockwiseInt4Linear) | Asymmetric INT4 + MSE Clip (g=32, tied emb) | 2.47 | 432.00 | 48.52 | 71.76 | 1.69 | 56.27 | 87.79 | 129.60 |
| **llama.cpp (llamacpp)** | GGML CPU + KleidiAI | Q4_0 (g=32 + BF16 PLE proj, tied emb) | 2.47 | 145.93 | 47.82 | 212.42 | 3.88 | 56.21 | 86.61 | 126.74 |

**Takeaways (`int4-g32`):**
- **Apples-to-apples at `g=32` and `2.47 GiB`:** `llama.cpp` `Q4_0` uses a
  32-weight block size (`2.47 GiB`). Matching that `g=32` granularity and
  **2.47 GiB** footprint in **`kidi_asym_g32`** achieves **56.27 `chrF (ref)`**
  and **87.79 `chrF (bf16)`** — the **highest quality of any 4-bit model** on
  both metrics — while running **2.96x faster on prefill** (`432.00` vs. `145.93
  tok/s`), faster on decode (`48.52` vs. `47.82 tok/s`), and using **2.30x less
  RAM** (`1.69 GiB` vs. `3.88 GiB` Peak RSS).

### 4. 4-Bit Quantization-Aware Trained (QAT / Mixed Precision)

| Runtime | Engine / Backend | Weight Format | Model (GiB) | Prefill (tok/s) | Decode (tok/s) | TTFT (ms) | Peak RSS (GiB) | chrF (ref) | chrF (bf16) | Eval Time (s) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Kidi QAT (kidi_qat)** | YNNPACK (arm64 SME/DOTPROD) | Static INT8 act + INT4 FFN/Attn + INT2 LM head | 2.29 | 844.83 | 68.56 | 36.69 | 2.36 | 55.71 | 84.76 | 76.04 |
| **Google LiteRT-LM v0.17.1 (litert)** | LiteRT-LM + XNNPACK | .litertlm mixed INT4/INT2 | 2.41 | 159.74 | 47.54 | 194.07 | 1.31 | 56.01 | 85.92 | 115.68 |

**Takeaways (QAT / Mixed Precision):**
- On Google's official QAT checkpoint (`gemma-4-E2B-it-qat-mobile-transformers`,
  which pairs static per-tensor INT8 activation scales with 4-bit transformer
  blocks and a 2-bit LM head), Kidi (`kidi_qat`) reaches **844.83 tok/s
  prefill** (**5.29x faster** than LiteRT-LM) and **68.56 tok/s decode**
  (**1.44x faster** than LiteRT-LM), completing WMT23-150 in just **76.04 s**.

### 5. Self-Contained Runtime Binary Size Comparison

| Runtime | Artifacts Measured | Stripped (MiB) | Unstripped (MiB) | Notes |
|---|---|---:|---:|---|
| **llama.cpp (`llamacpp`)** | `llamacpp_gemma4_bench` + `libllama.0.dylib` + `libggml*.dylib` | **4.48** | 5.10 | C++ probe + GGML CPU/KleidiAI shared libraries (reads GGUF vocabulary) |
| **Kidi (`kidi` / `kidi_bf16` / `kidi_qat`)** | `kidi_android_baseline` (single static binary) | **7.62** | 9.39 | Zero external non-OS dylibs (**6.29 MiB** with `-Wl,-dead_strip`). Includes C++ HF tokenizer ([`tokenizerspp`](https://github.com/thammegowda/tokenizerspp) + `uni-algo` + `RE2`), vision codecs ([`tahoma-vision`](https://github.com/thammegowda/tahoma-vision) + `libjpeg-turbo` + PNG/[`pigzpp`](https://github.com/thammegowda/pigzpp)/`zlib-ng`), WAV/Mel audio, Safetensors/GGUF/YAML/JSON parsers, and 6 model families; core Gemma 4 + [`ynnpack-dev`](https://github.com/thammegowda/ynnpack-dev) is **~3.3 MiB** |
| **Arm ExecuTorch v1.5.1 (`arm`)** | `arm_gemma4_bench` (single static binary) | **7.87** | 10.28 | Zero external non-OS dylibs (**6.22 MiB** with `-Wl,-dead_strip`); ExecuTorch runtime + XNNPACK + KleidiAI (excludes tokenizer, vision/audio codecs, and checkpoint parsers) |
| **ONNX Runtime v1.31.1 (`onnx`)** | `libonnxruntime.1.31.1.dylib` | **20.10** | 31.51 | Standalone native ONNX Runtime C/C++ shared library (excludes Python interpreter and tokenizer) |
| **Google LiteRT-LM v0.17.1 (`litert`)** | `liblitert-lm.dylib` | **52.18** | 66.01 | Standalone native Google LiteRT-LM C++ shared library (excludes Python interpreter) |

---

## Key Engineering Highlights in Kidi

1. **Zero-Copy Fused ARM64 NEON `sdot` Blockwise INT4 Kernel (`BlockwiseInt4Linear`):**
   - De-interleaves quantized INT8 activations into even/odd lanes (`vuzp1q_s8`
     / `vuzp2q_s8`) once per input row ($O(M \cdot K)$) so 4-bit weights are
     streamed **directly from `mmap`-ed `safetensors` pages** without a
     duplicate repacked weight buffer (`saving > 1 GiB` of RSS).
   - Accumulates `dot * (scale * act_scale) + act_bsum * zero` across blocks in
     registers using packed `FP16` metadata (`vld1q_f16`), eliminating
     `[num_blocks, M, N]` intermediate buffers.
   - Pins 4-output-row weight tiles (`3–24 KiB`) in the `128 KiB` L1D cache
     across multi-row prefill tiles (`MR = 8, 4, 3, 2, 1`).
2. **Tied Quantized Embeddings (`embed_tokens` $\leftrightarrow$ `lm_head`):**
   - Shares a single 262,144-row quantized embedding table between input token
     lookup and output `lm_head` projection across INT8, INT4-g128, and
     INT4-g32, saving `192–384 MiB` on disk and in RAM.
3. **Batteries-Included Native Binary Footprint (`7.62 MiB`):**
   - `kidi_android_baseline` (`7.62 MiB` stripped; `6.29 MiB` with
     `-Wl,-dead_strip`; `~3.3 MiB` for core Gemma 4 +
     [`ynnpack-dev`](https://github.com/thammegowda/ynnpack-dev)) is smaller than
     the barebones ExecuTorch runner (`7.87 MiB`) despite statically bundling a
     C++ Hugging Face tokenizer
     ([`tokenizerspp`](https://github.com/thammegowda/tokenizerspp), `uni-algo`,
     `RE2`), SIMD image codecs
     ([`tahoma-vision`](https://github.com/thammegowda/tahoma-vision),
     `libjpeg-turbo`, PNG/[`pigzpp`](https://github.com/thammegowda/pigzpp)/`zlib-ng`),
     WAV/Mel audio frontends, Safetensors/GGUF/YAML/JSON parsers, and 6
     multimodal model families (plus an embedded C++ HTTP/WebSocket server and
     EPUB reader via [`liteapp`](https://github.com/thammegowda/liteapp) in
     `kidi-app`).

---

## Reproducing the Benchmark

### 1. Environment & Build Setup

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

### 2. Running the Benchmark & WMT23 Evaluation

```bash
# Run 4-bit suite (Kidi sym/asym-g128/asym-g32, Arm ExecuTorch, ONNX, llama.cpp, LiteRT-LM, Kidi QAT)
.cache/gemma-venv/bin/python benchmarks/gemma4/arm/run.py \
  --profile int4-g128 \
  --include-onnx \
  --llamacpp-model .cache/benchmarks/arm-gemma4/gemma4-e2b-q4_0.gguf \
  --litert-model tmp.models/gemma-4-E2B-it-litert-lm/gemma-4-E2B-it.litertlm \
  --kidi-qat-model tmp.models/gemma-4-E2B-it-qat-mobile-transformers \
  --eval-lines 150

# Regenerate the four Markdown tables from benchmarks/gemma4/arm/results.tsv
.cache/gemma-venv/bin/python benchmarks/gemma4/arm/run.py --render-tables
```

Per-runtime results and per-sentence WMT23 outputs are cached in
`<work-dir>/results/<runtime>.json` and
`<work-dir>/results/<runtime>_eval.output.jsonl` so interrupted runs resume
automatically. Use `--rerun <runtime>` to force-rerun a specific runtime or
`--skip <runtime>` to omit one.
