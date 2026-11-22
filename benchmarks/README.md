# Benchmarks

This directory contains reproducible benchmark harnesses. Public summaries stay
small; raw logs, exported models, profiler traces, and interrupted-run state
belong under `.cache/`.

## Current Comparisons

| Benchmark | What it compares |
|---|---|
| [Gemma 4 E2B on Arm CPU](gemma4/arm/README.md) | Kidi vs. Arm ExecuTorch, ONNX Runtime, and `llama.cpp` |
| [CUDA](cuda/) | Kidi CUDA vs. compiled vLLM and CPU references; [H100 E4B Q8 comparison with vLLM and llama.cpp](cuda/2026-11-20-e4b-q8-h100-gpu7.md) |
| [Android](android/README.md) | CPU, Vulkan, QNN, and mobile integration |
| [Browser](browser/) | WebAssembly/WebGPU latency, ASR, and heap behavior |

## Specialized Harnesses

| Area | Purpose |
|---|---|
| [CPU](cpu/) | Stable native reference fixtures |
| [Gemma 4](gemma4/) | Whole-model runtime and quality probes |
| [Metal](metal/) | Operator lowering and packed-kernel measurements |
| [RTG](rtg/) | Translation runtime and PyTorch reference |
| [TTS](tts/) | Synthesis timing and calibration |
| [Whisper](whisper/) | Speech-recognition quality checks |

## Benchmark Policy

- Compare equivalent checkpoints and precisions.
- Report prompt prefill and autoregressive decode separately.
- Exclude model loading and warm-up from steady-state measurements.
- Record model, precision, context, token counts, threads, and timing boundaries.
- Use deterministic inputs and decoding for quality comparisons.
- Keep only current canonical summaries in Git; retain experimentation history in
  Git history rather than public tables.

## Quick E4B CUDA iteration

With the locally prepared E4B Q8 models and reference runners described in the
[H100 comparison](cuda/2026-11-20-e4b-q8-h100-gpu7.md), a Release CUDA
`kidi_android_baseline`, the cached WMT23 fixture at
`.cache/benchmarks/x64-gemma4-int8/eval_protocol.json`, and SacreBLEU installed:

```bash
# Remeasure reference-engine speed and quality and pin a Kidi baseline.
python3 benchmarks/cuda/quick_e4b.py --refresh-references --save-baseline

# After changing Kidi, rerun Kidi and compare with the saved engine references.
python3 benchmarks/cuda/quick_e4b.py --label trial

# Refresh only reference-engine quality, keeping the cached speed measurements.
python3 benchmarks/cuda/quick_e4b.py --refresh-quality
```

All GPU runs use physical GPU 7. The normal iteration runs only Kidi; the
displayed vLLM and llama.cpp timings are cached and dated. Refresh them when
their binaries, hardware, or runtime settings change. Prefill throughput is
prompt tokens divided by TTFT, and decode counts the 127 intervals between 128
generated tokens. The engines use different Q8 quantization schemes.

The table's chrF column compares all three engines on the same five WMT23
translations against human English references. This separate quality score is
repeated for each speed-prompt row; five sentences are not a reliable full
quality evaluation. All engines use the same prompt IDs and a 192-token limit.
vLLM stops at `<eos>` or `<turn|>` like Kidi. The llama.cpp reference runner
generates through EOS; the scorer keeps tokens before the first of those stops.

The separate Kidi output regression check compares tokens and chrF with the
saved Kidi output; it does not establish accuracy. `--save-baseline` explicitly
replaces the saved Kidi baseline.

Gemma CUDA warm-up preallocates one full-context KV cache. Eager and serving
generation reuse matching-capacity text-only states after device work and
pending readbacks complete; this is allocation reuse, not prompt-prefix reuse.
The pool respects serving limits, appears in `Generator::memory().cache_bytes`,
and is cleared by `release_workspaces()`. Multimodal and speculative states are
not pooled.

## Native Build

```bash
cmake --preset release -DKIDI_BUILD_BENCHMARKS=ON
cmake --build --preset release --target \
  kidi_android_baseline kidi_tts_bench kidi_gemma4_quality \
  kidi_whisper_quality --parallel 8
```

`android/baseline.cpp` is also the native Linux/macOS Gemma and Whisper harness;
the directory name does not require running it on Android.

Each benchmark directory documents its own optional runtimes, models, and
dependencies. Use `.cache/benchmarks/` for generated results.
