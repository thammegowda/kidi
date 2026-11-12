# Benchmarks

This directory contains reproducible benchmark harnesses. Public summaries stay
small; raw logs, exported models, profiler traces, and interrupted-run state
belong under `.cache/`.

## Current Comparisons

| Benchmark | What it compares |
|---|---|
| [Gemma 4 E2B on Arm CPU](gemma4/arm/README.md) | Kidi vs. Arm ExecuTorch, ONNX Runtime, and `llama.cpp` |
| [CUDA](cuda/) | Kidi CUDA vs. compiled vLLM and CPU references |
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
