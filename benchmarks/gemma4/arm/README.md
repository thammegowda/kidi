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
disabled. Optional flags also benchmark:

- `--include-onnx`: ONNX Runtime CPU (`onnx_bench.py`, using `MatMulNBits` with
  `block_size=128` for `int4-g128` and `DynamicQuantizeLinear` +
  `MatMulIntegerToFloat` for `int8` over the shared quantized weights, backed
  by MLAS + Arm KleidiAI).
- `--llamacpp-model PATH`: `llama.cpp` + GGML with `GGML_CPU_KLEIDIAI=ON`
  (`llamacpp_bench.cpp` on `Q4_0` or `Q8_0` GGUF).
- `--litert-model PATH` & `--kidi-qat-model PATH`: Google's `litert-lm`
  (`litert_bench.py` on `gemma-4-E2B-it.litertlm`) paired with KIDI on Google's
  mixed 2/4/8-bit QAT checkpoint (`gemma-4-E2B-it-qat-mobile-transformers`).
- `--eval-lines N` (default `5`): Prepares pre-templated WMT23 (`ja,uk,de` $\rightarrow$ `en`, `--eval-seed 42`) translation requests in `<work-dir>/eval_input.jsonl` and aligned human references in `<work-dir>/eval_references.jsonl`, feeds the identical JSONL prompts to every runtime, writes `<work-dir>/results/<runtime>_eval.output.jsonl`, and computes SacreBLEU **chrF** (`sacrebleu==2.6.0`) along with total evaluation inference time (`eval s`).
- Per-runtime results are stored under `<work-dir>/results/<runtime>.json` (and `<runtime>_eval.json`) and
  reused automatically when the protocol, command, and binary SHA-256 match.
  Pass `--rerun <runtime>` to rerun specific runtimes or `--force` for a clean
  rerun of all requested runtimes.

## Benchmark results

Measured on an **Apple M5** CPU (`arm64`, 16 GB unified memory, macOS) using:

- **Model:** `google/gemma-4-E2B-it` (35 layers, 1,536 hidden dim, 15 KV-donor
  layers + 20 KV-sharing layers, 262,144 vocab + per-layer embeddings)
- **Speed workload:** 4 CPU threads, batch 1, 1,024-token context, 31-token prompt
  (`"Explain how binary search works in at least 300 words, including an example and its time complexity."`),
  64 greedy output tokens (31 prefill tokens + first token for prefill/TTFT;
  63 decode steps for decode throughput), 1 excluded warmup + 3 measured
  independent runs (median reported, with min–max decode range).
- **Translation quality workload (`--eval-lines 5`):** 5 deterministic WMT23
  (`ja,uk,de` $\rightarrow$ `en`, seed `42`) sentences pre-formatted with the
  Gemma 4 chat template in `eval_input.jsonl`, greedy decoding until `<turn|>` /
  `<eos>`, scored with SacreBLEU corpus **chrF** (`chrF2`) against
  `eval_references.jsonl`.

### Unified Comparison by Precision Group (`BF16` $\rightarrow$ `8-bit` $\rightarrow$ `4-bit`)

| Precision Group | Runtime | Engine / Backend | Weight Format | Prefill (tok/s) | Decode (tok/s) | Decode Range (tok/s) | TTFT (ms) | Peak RSS (GiB) | Binary Size (MiB) | WMT23 chrF | Eval (s) |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **BF16 (16-bit)** | **HF Transformers (`hf_bf16`)** | Transformers v5.17.0 + PyTorch v2.14.0 | Unquantized BF16 (Reference) | 30.18 | 14.13 | 14.04–14.37 | 1027.09 | 4.56 | 310.11 | **63.56** | 15.61 |
| **BF16 (16-bit)** | **KIDI (`kidi_bf16`)** | YNNPACK (`arm64` BF16) | Unquantized BF16 | **243.47** | **16.40** | 16.37–16.68 | **127.33** | 4.57 | **7.54** | 62.88 | **5.58** |
| **───** | **───** | **───** | **───** | **───** | **───** | **───** | **───** | **───** | **───** | **───** | **───** |
| **8-bit** | **KIDI (`kidi`)** | YNNPACK (`arm64` SME/DOTPROD) | Symmetric INT8 per-channel | **583.12** | 32.92 | 31.33–33.34 | **53.16** | 3.19 | 7.54 | **63.57** | **2.88** |
| **8-bit** | **Arm ExecuTorch v1.5.1 (`arm`)** | XNNPACK + Arm KleidiAI (`8da8w+emb8`) | Symmetric INT8 per-channel | 436.82 | 28.42 | 28.42–28.66 | 70.97 | 2.44 | 7.87 | 59.11 | 3.51 |
| **8-bit** | **ONNX Runtime v1.31.1 (`onnx`)** | MLAS + Arm KleidiAI (`MatMulIntegerToFloat`) | Symmetric INT8 per-channel | 309.40 | 31.29 | 31.18–31.39 | 100.19 | **2.25** | 20.10 | 62.20 | 3.41 |
| **8-bit** | **llama.cpp (`llamacpp`)** | GGML CPU + Arm KleidiAI | `Q8_0` (`g=32`) | 184.33 | **33.30** | 33.25–33.39 | 168.17 | 6.58 | **4.48** | 62.67 | 3.35 |
| **───** | **───** | **───** | **───** | **───** | **───** | **───** | **───** | **───** | **───** | **───** | **───** |
| **4-bit** | **KIDI (`kidi`)** | YNNPACK (`arm64` SME/DOTPROD) | Symmetric INT4 (`g=128`) | **532.26** | 50.00 | 48.92–50.65 | **58.24** | 1.93 | 7.57 | 60.37 | 2.27 |
| **4-bit** | **KIDI (`kidi_asym_clip`)** | YNNPACK (`arm64` SME/DOTPROD) | Asymmetric INT4 (`g=128`) + MSE Clip | 464.44 | 46.97 | 40.56–46.99 | 66.75 | 1.97 | 7.57 | **65.37** | 2.60 |
| **4-bit** | **Arm ExecuTorch v1.5.1 (`arm`)** | XNNPACK + Arm KleidiAI (`8da4w+emb4`) | Symmetric INT4 (`g=128`) | 234.11 | 43.51 | 43.44–43.64 | 132.42 | 1.84 | 7.87 | 59.63 | 2.91 |
| **4-bit** | **ONNX Runtime v1.31.1 (`onnx`)** | MLAS + Arm KleidiAI (`MatMulNBits`) | Symmetric INT4 (`g=128`) | 396.27 | **50.65** | 50.31–50.91 | 78.23 | **1.45** | 20.10 | 59.55 | **2.18** |
| **4-bit** | **llama.cpp (`llamacpp`)** | GGML CPU + Arm KleidiAI | `Q4_0` (`g=32`) | 145.93 | 47.82 | 47.30–47.89 | 212.42 | 3.88 | **4.48** | 58.60 | 2.74 |
| **4-bit (QAT / Mixed)** | **KIDI (`kidi_qat`)** | YNNPACK (`arm64` SME/DOTPROD) | Static INT8 act + INT4 FFN/Attn + INT2 LM head | **847.81** | **67.68** | 67.24–68.00 | **36.56** | 2.36 | **7.57** | 51.09 | **1.57** |
| **4-bit (QAT / Mixed)** | **Google LiteRT-LM v0.17.1 (`litert`)** | LiteRT-LM + XNNPACK | `.litertlm` mixed INT4/INT2 | 159.74 | 47.54 | 47.49–47.77 | 194.07 | **1.31** | 52.18 | **51.97** | 2.72 |

- **BF16 Reference vs. KIDI BF16:** KIDI (`kidi_bf16`) achieves **8.07x faster
  prefill** (`243.47` vs. `30.18 tok/s`), **1.16x faster decode** (`16.40` vs.
  `14.13 tok/s`), and **2.80x faster WMT23 evaluation** (`5.58 s` vs. `15.61 s`)
  with a **41x smaller runtime binary** (`7.57 MiB` vs. `310.11 MiB`) and
  comparable translation quality (`62.88` vs. `63.56` chrF).
- **8-bit (`int8`) vs. Arm ExecuTorch v1.5.1:** On identical `int8` weights and
  scales, KIDI achieves **1.33x faster prefill** (`583.12` vs. `436.82 tok/s`),
  **1.16x faster decode** (`32.92` vs. `28.42 tok/s`), and **63.57 chrF**
  (matching the unquantized `hf_bf16` reference at `63.56` chrF), while sharing
  the first **58 out of 64 generated tokens** with Arm ExecuTorch before
  floating-point tie-breaking diverges at token 59.
- **4-bit (`int4-g128` Symmetric vs. Asymmetric + MSE Outlier Clipping):**
  - On matched **symmetric `int4-g128`** weights (`amax / 7`, `[-7, 7]`), KIDI
    (`kidi`) achieves **2.27x faster prefill** (`532.26` vs. `234.11 tok/s`) and
    **1.15x faster decode** (`50.00` vs. `43.51 tok/s`) than Arm ExecuTorch
    (`60.37` vs. `59.63` chrF), though all symmetric 4-bit runtimes show a
    ~3–5 chrF drop relative to BF16 (`63.56`).
  - Applying **blockwise asymmetric 4-bit quantization (`0..15` full 16-level
    lattice with per-block zero point) + 11-step data-free MSE outlier clipping
    search (`f in {1.00, 0.95, ..., 0.50}`)** ([gowda.ai](https://gowda.ai/posts/2026/08/llm-fast-inference/))
    in **`kidi_asym_clip`** recovers the full BF16 translation quality (**65.37
    chrF** vs. `63.56` on `hf_bf16`) while still outperforming Arm ExecuTorch in
    both prefill (**464.44 vs. 234.11 tok/s**, **1.98x**) and decode (**46.97 vs.
    43.51 tok/s**, **1.08x**) at `1.97 GiB` peak RSS.

### Self-contained runtime binary size breakdown

The **Binary Size (MiB)** column reports the stripped (`strip -x`) footprint of
the native executable plus any non-OS shared libraries it links against:

| Runtime | Artifacts Measured | Stripped (MiB) | Unstripped (MiB) | Notes |
|---|---|---:|---:|---|
| **`llamacpp`** | `llamacpp_gemma4_bench` + `libllama.0.dylib` + `libggml*.dylib` | **4.46** | 5.10 | C++ probe + GGML CPU/KleidiAI shared libraries (no SentencePiece/HF JSON tokenizer needed when reading GGUF vocabulary) |
| **`kidi` / `kidi_qat`** | `kidi_android_baseline` (single static binary) | **7.54** | 9.28 | Zero external non-OS dylibs; **6.29 MiB** with `-Wl,-dead_strip`. Includes built-in C++ Hugging Face tokenizer (`tokenizerpp` + `uni-algo` + `re2`, ~1.35 MiB) and multi-model CLI (Whisper, Kokoro, OmniVoice, Qwen3, Gemma 4 vision/audio); core Gemma 4 + YNNPACK runtime is **~3.3 MiB** |
| **`arm`** | `arm_gemma4_bench` (single static binary) | **7.87** | 10.28 | Zero external non-OS dylibs; **6.22 MiB** with `-Wl,-dead_strip`. Links ExecuTorch runtime + XNNPACK + KleidiAI + threadpool (excludes tokenizer; reads pre-tokenized prompt IDs) |
| **`onnx`** | `libonnxruntime.1.31.1.dylib` | **20.10** | 31.51 | Standalone native ONNX Runtime C/C++ shared library (excludes Python interpreter and `onnxruntime_pybind11_state.so`) |
| **`litert`** | `liblitert-lm.dylib` | **52.18** | 66.01 | Standalone native Google LiteRT-LM C++ shared library (excludes Python interpreter) |
| **`hf_bf16`** | `libtorch_cpu.dylib` | **310.11** | 370.52 | PyTorch CPU native shared library used by Hugging Face `transformers` (excludes Python interpreter and other torch dylibs) |

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
