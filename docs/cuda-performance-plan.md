# CUDA Performance Plan

Current state of Kidi's native CUDA backend: supported paths, headline measurements, release gates and known
limitations. Earlier milestone plans and experiment logs remain in Git history.

## Scope

- Kernels, prepared storage, dispatch and CUDA graph capture/replay live in [`kidi-ops`](../kidi-ops/README.md); step
  replay (`graph::Session`) and generation scheduling live in Kidi core. Models stay ordinary eager C++
  ([architecture](../ARCHITECTURE.md)), with no second model implementation, whole-model compiler or LibTorch.
- CUDA builds need Linux and the CUDA 13+ toolkit. The `release-cuda` preset links the CUDA runtime statically, uses no
  cuBLAS and embeds `sm_80` and `sm_90` code; see the [CUDA build notes](../README-dev.md#cuda).
- Select a GPU with `--backend cuda:N`. There is no silent CPU operator fallback; unsupported alignments and layouts use
  slower CUDA reference kernels.
- Precision is chosen with `--precision {checkpoint,bf16,q8,q4,vq3d-hybrid}`. On CUDA, `q8` is the W8A16 path, and
  `checkpoint` runs native packed INT8 checkpoints on the W8A8 path.

## Supported Paths

- **BF16 projections:** Tensor Core GEMM with tuned `K=64` tiles and Split-K over a workspace shared across layers,
  plus vector kernels for decode. Strict FP32 uses a tiled CUDA-core GEMM without TF32.
- **W8A16 (`q8`):** INT8 weights with BF16 activations, norms and KV cache. Tensor Core `pipelined_w8a16_gemm` with
  whole-group Split-K, 128-bit `w8a16_vector` decode GEMV, and load-time packing without duplicate BF16 projection VRAM.
- **W8A8 (native INT8):** INT8 weights with dynamic or calibrated INT8 activations and FP32 norms.
  `pipelined_w8a8_gemm` (`mma.m16n8k32`) keeps the affine correction, scales, bias and output rounding in a fused
  epilogue; decode uses the DP4A `quantized_vector_fast` kernel. Single-group packed INT2/INT4 weights are expanded to
  INT8 for GEMM. Building with a CUTLASS checkout adds a Hopper (`sm_90a`) TMA/WGMMA W8A8 GEMM.
- **Block-scaled projections:** E2M1 (NVFP4), E4M3 (MXFP8) and symmetric INT4 weights with configurable groups, used by
  routed Inkling 2 experts and BF16 symmetric INT4 projections.
- **Attention:** FlashAttention-2 for eligible BF16-cache prefill, a tiled Tensor Core kernel for short queries and
  arbitrary masks, and a tiled FP32 kernel for FP32 caches, all for `head_dim` 64/128/256/512 with GQA query-head
  packing and no score buffer. Decode attention splits keys into partitions (`decode_partitions`, `decode_merge`) for
  head widths 256 and 512.
- **Fusion:** W8A8 prefill folds the preceding RMS norm or GELU gate into activation quantization (bit-identical to the
  unfused sequence); per-head RMS norm + rotary is one kernel, and one scatter writes K and V while converting to BF16.
- **Sampling and audio:** greedy token selection, OmniVoice diffusion steps and unfold/fold convolution run on-device.
- **CUDA graphs:** fixed-shape decode steps are captured and replayed from a bounded cache. Single-row projections that
  share an input (such as `q_proj` and `kv_proj`) launch together, KV pointers are bound directly, transfers use pooled
  pinned buffers, and device-token feedback is bounded with asynchronous host EOS checks. The
  [validation record](../benchmarks/cuda/2026-10-25-async-decode-validation-tgpc01.json) shows 127 warm long-context
  graph launches with no synchronization or synchronous-copy API calls after decode starts.

## Headline Measurements

All runs are batch 1 with warm requests; model loading and warm-up are excluded.

**Gemma 4 E2B BF16 vs. vLLM, RTX A4000 (`sm_86`, 16 GiB), 2026-10-25:** matching checkpoint and input IDs,
2,322-token prompts and 128 outputs ([record](../benchmarks/cuda/2026-10-25-bf16-pipelined-long-tgpc01.json)).

| Engine | Long prefill (tokens/s) | Decode (tokens/s) |
|---|---:|---:|
| Kidi | 11,285.85 | 64.41 |
| vLLM | 8,062.70 | 66.65 |

Long prefill wins by 1.40x; a decode-speed win is not established.

**Gemma 4 E4B, H100 80GB (physical GPU 7), published 2026-11-21:** `chunk=128`, 2 warmups, 3 repeats at 162 tokens.
The shared cuBLAS row is the original Kidi build, kept as a reference. The compact builds match reference generated
token IDs.

| Configuration | 162-tok prefill (tok/s) | 162-tok decode (tok/s) | 1,098-tok prefill (tok/s) | Peak GPU mem (GiB) |
|---|---:|---:|---:|---:|
| Original shared cuBLAS BF16 | 8,493 | 105.6 | 15,226 | 15.24 |
| Compact BF16 | 9,281 | 134.1 | 14,273 | 15.26 |
| Compact W8A16 | 8,503 | 159.3 | 12,040 | 7.95 |
| Compact W8A8 | 7,735 | 139.3 | 9,974 | 8.58 |

**Gemma 4 E4B Q8 vs. vLLM and llama.cpp, H100 (physical GPU 7):** the
[comparison record](../benchmarks/cuda/2026-11-20-e4b-q8-h100-gpu7.md) times 128-, 512- and 2,048-token prompts (Kidi
measured 2026-11-22, competitor timings cached from 2026-11-21; the engines use different Q8 schemes). Kidi has the
lowest time to first token at 128 and 512 tokens; vLLM leads at 2,048 tokens, and vLLM and llama.cpp both decode faster.

## Release Gates

Keep two scoreboards: interactive batch 1, and serving throughput under stated latency and VRAM budgets. Compare exact
checkpoints and policies; different checkpoints, beam widths, steps, durations or quantization policies are not
backend-only speedups. A state-of-the-art claim needs measurements on the claimed hardware.

1. **Recover:** beat the 8-thread [CPU reference](../benchmarks/cpu/reference.json) for Gemma 4 E4B mobile QAT,
   Whisper Small W8A8 and OmniVoice Q8 W8A8 without quality regressions. The last A4000 CUDA fixture run, made before
   native replay, beat the CPU for Whisper on an 11-second clip (0.988 s vs. 1.938 s), OmniVoice for 4.2 seconds of
   output at 16 steps (2.529 s vs. 9.713 s) and Gemma short prefill (199.81 vs. 127.15 tokens/s). Gemma short decode
   trailed (4.81 vs. 14.56 tokens/s) and needs re-measurement.
2. **Compete:** meet or beat the strongest matched external baseline, reporting uncertainty, p95 latency, memory and
   first-output behavior. Not yet met: no decode win over vLLM is established.
3. **Sustain:** hold those results across long contexts, varied inputs, concurrent requests and repeated runs, with
   exact versions, hashes and settings published.

## Precision Policy

Native QAT and affine W8A8 behavior are the defaults: Kidi never silently swaps in FP16 weights or changes calibration.
Other precision changes (FP16 attention, TF32, new KV formats, changed RNG or sampling) are named, separately
benchmarked policies that need explicit approval before becoming defaults; a reference-compatible path aids diagnosis.

## Known Limitations

- Broader shapes and multi-request scheduling remain open; asynchronous decode covers single requests.
- Real second-GPU runs and multi-device isolation are not validated; NCCL/tensor parallelism is a separate project.
- Expanding packed INT2/INT4 weights to INT8 raises prepared-weight VRAM; grouped packed weights use the slower native
  kernels.
- CUDA has no calibrated INT8 KV cache for Gemma 4. AUTO selects BF16 caches; FP32-activation paths can keep FP32 with
  `--kv-cache-precision fp32`.
- Preparation and weight rebinding can allocate, so warm up with stable weights before graph capture.
- On the A4000 WSL host, Nsight Systems records CUDA API activity but no kernels, and hardware counters are
  unavailable; use a GPU host with profiler access, such as the H100, for kernel traces, counters and Compute Sanitizer.
- Correctness evidence is token-ID matching and small fixtures; broad Gemma, Whisper and TTS quality suites remain open.

## Benchmarks and Profiling

- [`quick_e4b.py`](../benchmarks/cuda/quick_e4b.py) reruns Kidi on the H100 E4B Q8 protocol beside cached vLLM and
  llama.cpp results ([usage](../benchmarks/README.md#quick-e4b-cuda-iteration));
  [`compare_gemma.py`](../benchmarks/cuda/compare_gemma.py) runs controlled warm-only E2B comparisons with vLLM.
- `kidi_ops_cuda_projection_benchmark` reports kernel-level projection timing and reference error.
- The developer benchmark tool `kidi_android_baseline` accepts `--profile-ops host|sync|graph|events`,
  `--log-graph-captures` and `--cuda-profiler`, and selects the device with `--device` (e.g. `--device cuda:0`).
- Detailed records: [E2B BF16 comparison](../benchmarks/cuda/2026-10-25-bf16-pipelined-long-tgpc01.json),
  [asynchronous decode validation](../benchmarks/cuda/2026-10-25-async-decode-validation-tgpc01.json),
  [H100 E4B Q8 comparison](../benchmarks/cuda/2026-11-20-e4b-q8-h100-gpu7.md)
  ([JSON](../benchmarks/cuda/2026-11-20-e4b-q8-h100-gpu7.json)), [CPU reference](../benchmarks/cpu/reference.json) and
  the [benchmark guide](../benchmarks/README.md).
