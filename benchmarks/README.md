# Benchmarks

Keep runnable harnesses, their inputs, and the canonical reports listed below.
Write new measurements, profiler output, and temporary exports under `.cache/`,
not this source tree. Retired experiments and journals belong in Git history.

## Native Build

```bash
cmake --preset release -DKIDI_BUILD_BENCHMARKS=ON
cmake --build --preset release --target kidi_android_baseline kidi_tts_bench \
  kidi_gemma4_quality kidi_whisper_quality --parallel 8
```

`android/baseline.cpp` is also the native Linux/macOS Gemma and Whisper harness;
its directory name does not require running it on Android. Model loading and
warm-up must be excluded from steady-state measurements. Label checkpoint,
precision, input/output counts, context, threads, and timing boundaries explicitly.

## CUDA

CUDA is optional and requires CUDA Toolkit 13 and cuBLAS. CPU-only builds do not
require either. Select architectures appropriate to the deployment GPU:

```bash
cmake --preset release -DKIDI_ENABLE_CUDA=ON -DKIDI_BUILD_BENCHMARKS=ON \
  -DKIDI_BUILD_TESTS=ON -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build --preset release --target kidi_android_baseline kidi_tts_bench \
  kidi_cuda_test ynnpack_cuda_test --parallel 8
```

`cuda` aliases `cuda:0`; `cuda:N` respects `CUDA_VISIBLE_DEVICES`. Unsupported
operators fail rather than silently falling back to CPU. Checkpoint dimensions,
dtypes, and quantization metadata select Gemma E2B/E4B and QAT/BF16 execution.
Native QAT preserves trained packed weights and calibration; BF16 uses BF16
storage and FP32 accumulation. The default CUDA caches are FP32 for QAT and BF16
for ordinary BF16 checkpoints. Runtime-owned GPU allocation counters exclude
driver/context and cuBLAS-private memory.

### Matched Gemma Comparison

Install the reference runtime in an isolated Python environment with vLLM and
Hugging Face Hub. The driver downloads pinned checkpoints using existing Hugging
Face credentials. From the repository root:

```bash
.cache/vllm-venv/bin/python benchmarks/cuda/compare_gemma.py --bf16-only \
  --context 4096 --chunk 512 --tokens 128 --long-copies 128 --decode-ahead 16 \
  --warmups 2 --repeats 3 --threads 8 --vllm-kv-mib 512 \
  --output .cache/benchmarks/bf16-comparison.json
```

Omit `--bf16-only` for CPU QAT, CUDA QAT, CUDA BF16, and compiled vLLM BF16.
The driver checks identical prompt IDs, output counts, repeat determinism, finite
timings, and excluded preparation/capture. Reports include commands, revisions,
hashes, raw-log paths, and queue counters. Existing reports are not overwritten.

Prefill throughput includes submission through first-token delivery after
tokenization. Decode uses the remaining output tokens divided by the first-to-last
token interval. These are not pure kernel timings. QAT and BF16 are deliberately
different encodings; compare the same BF16 checkpoint for runtime conclusions.

### Canonical Evidence

| Report | Purpose |
| --- | --- |
| [Final BF16 comparison](cuda/2026-10-25-bf16-pipelined-long-tgpc01.json) | Same-checkpoint long-sequence Kidi/vLLM performance |
| [Decode validation](cuda/2026-10-25-async-decode-validation-tgpc01.json) | CUDA API synchronization audit and QAT token preservation |
| [Four-way reference](cuda/2026-10-25-controlled-e2b-four-way-tgpc01.json) | Earlier controlled CPU/QAT/BF16 comparison and validation reference |

Final E2B BF16 measurements on RTX A4000: 2,322 input tokens, 128 outputs,
4,096-token context, 512-token prefill budget, one request, no prefix cache or
CPU offload. Medians of three requests after two excluded warm-ups:

| Runtime | Prefill Through First Token | Decode |
| --- | ---: | ---: |
| Kidi CUDA BF16 | 11,285.85 tokens/s | 64.41 tokens/s |
| vLLM CUDA BF16, compiled/graphs | 8,062.70 tokens/s | 66.65 tokens/s |

Kidi wins long prefill by 1.40x; decode remains 3.4% slower. The four-way report
predates these optimizations and is not a current performance claim. These are
fixture-level quality checks, not broad accuracy or exhaustive tuning results.

The CUDA API audit records 127 warm long-context graph launches with no
synchronization or synchronous-copy API calls after decode begins. Cold capture,
prefill, and cleanup may synchronize. Kernel counters and Compute Sanitizer are
not verified on this WSL host; second-physical-GPU execution is also unverified.

### Decode and Profiling

Kidi owns graph capture/replay and generation scheduling; YNNPACK owns CUDA
operators and prepared resources. Device-token feedback uses asynchronous pinned
readbacks for host streaming/EOS checks, not blocking host-token feedback.
`ServingOptions::decode_ahead_tokens` selects policy default 16 on CUDA, synchronous
mode at one, or a bounded 2-64-token window. EOS stops submission; queued post-EOS
results are suppressed, work drains safely, and retained KV position is restored.
Multi-active/deferred serving and external host-gathered tables retain their
existing scheduling. Ordinary `generate()` uses the pipeline without prefix caching.

Diagnostic overrides: `KIDI_PROFILE_OPS=graph` profiles replay phases; `=events`
profiles captured operators; `KIDI_CUDA_PROFILE=1` brackets measured native
requests for `nsys --capture-range=cudaProfilerApi`. The comparison driver clears
these overrides. Store raw traces and logs under `.cache/`.

### Multi-Model Runner

[cuda/run.py](cuda/run.py) runs Gemma QAT, Whisper Small W8A8, and OmniVoice Q8
against [cpu/reference.json](cpu/reference.json), the pinned 2026-10-24 CPU input
and output fixture. Its machine-local checkpoint/audio paths must exist; pass
`--cpu-reference` for another prepared reference. NumPy is required for waveform
checks and SciPy for the optional ASR proxy.

```bash
.cache/test-venv/bin/python benchmarks/cuda/run.py --device cuda:0 --asr-quality \
  --output .cache/benchmarks/native-cuda.json
```

Use `--device cpu` for a new native CPU measurement. Raw subprocess output and
audio exports are kept in `.cache/cpu-baseline/` or `.cache/cuda-baseline/`.

## Gemma 4

[gemma4/run.py](gemma4/run.py) compares Kidi CPU/Metal with llama.cpp or LiteRT-LM.
Install [gemma4/requirements.txt](gemma4/requirements.txt); reference runtimes and
models are separate prerequisites. Do not compare the released mixed-QAT LiteRT
bundle with BF16 Kidi as if they were equal precision.

```bash
python benchmarks/gemma4/run.py --runtime kidi --backend cpu --model /path/to/model \
  --prefill 128 --decode 64 --runs 3 --warmups 2 \
  --output .cache/benchmarks/gemma-cpu.json
build-release/kidi_gemma4_quality /path/to/model cpu 0 128 generation-batch
```

The quality probe also exercises serving admission, streaming, cancellation,
backpressure, and cache reservations. Use runner `--help` for precision,
reference-runtime, context, and prefill options.

## Metal

On macOS, `KIDI_BUILD_BENCHMARKS=ON` builds `kidi_metal_lowering_bench`,
`kidi_metal_packed_bench`, and `kidi_metal_int8_bench`. They check numerical
results and measure projection, packed-kernel, or quantized execution rather than
establishing whole-model performance from isolated kernels.

```bash
cmake --build --preset release --target kidi_metal_lowering_bench \
  kidi_metal_packed_bench kidi_metal_int8_bench --parallel 8
build-release/kidi_metal_lowering_bench 1 100 10 8 argmax bf16
build-release/kidi_metal_packed_bench
build-release/kidi_metal_int8_bench 1
```

[metal/profile_rtg.py](metal/profile_rtg.py) profiles warmed RTG decoding with
optional macOS GPU counters.

## Other Harnesses

| Area | Entrypoints |
| --- | --- |
| Android CPU/Vulkan/QNN and vision | [Android guide](android/README.md) |
| RTG translation and PyTorch reference | [RTG guide](rtg/README.md) |
| TTS timing and calibration | [tts/synthesis.cpp](tts/synthesis.cpp), [tts/calibrate_omnivoice.py](tts/calibrate_omnivoice.py) |
| Whisper quality | [whisper/quality.cpp](whisper/quality.cpp) |

## Browser

[browser/runtime_bench.mjs](browser/runtime_bench.mjs),
[browser/asr_probe.mjs](browser/asr_probe.mjs), and
[browser/heap_probe.mjs](browser/heap_probe.mjs) exercise runtime latency,
ASR, and heap behavior. Build/setup instructions are in the
[web development guide](../web/README-dev.md).