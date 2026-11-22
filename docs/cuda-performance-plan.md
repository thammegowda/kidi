# CUDA Performance Plan

Status: quantized/BF16 operators, native Kidi CUDA replay, and single-request asynchronous decode implemented; remaining tuning and concurrency milestones remain proposed.
Date: 2026-10-25.

## Objective

Make native Kidi CUDA inference competitive with the best validated implementation
for its model, precision, hardware, and workload. "State of the art" is a measured
outcome, not a promise attached to a particular kernel library or graph API.

Initial scope is Gemma 4 E4B mobile QAT, Whisper Small W8A8, and OmniVoice Q8 W8A8.
Start with batch-1 interactive latency on the RTX A4000 (`sm_86`, 16 GiB), then
measure throughput and concurrent workloads separately. Preserve CPU-only builds,
native quantized checkpoints, explicit `cuda:N` selection, and the prohibition on
silent CPU operator fallback.

CUDA kernels, preparation, and backend-specific capture/replay live in `kidi-ops`;
shared graph orchestration and generation scheduling belong to Kidi core.
Reuse Kidi's eager models, prepared operators, and fixed-shape captured-step
interfaces. Do not add a second model implementation, whole-model compiler,
LibTorch dependency, or exported-model conversion requirement.

## Starting Point

### BF16 and Asynchronous Decode Update

The [controlled E2B BF16 comparison](../benchmarks/cuda/2026-10-25-bf16-pipelined-long-tgpc01.json)
now measures 2,322-token prompts and 128 outputs on the A4000 with matching
checkpoint/input IDs and excluded startup/warm-up. Kidi long prefill is 11,285.85
tokens/s versus vLLM's 8,062.70; decode is 64.41 versus 66.65 tokens/s. Long
prefill wins by 1.40x, but a decode-speed win is not established.

Implemented mechanisms include BF16 prefill attention GEMMs, partitioned decode
attention, vector projections, normalization/gating fusion, local history crops,
native captured replay, direct KV pointer bindings, pooled pinned transfers, and
bounded device-token feedback with asynchronous host EOS checks. The
[validation artifact](../benchmarks/cuda/2026-10-25-async-decode-validation-tgpc01.json)
records 127 warm long-context graph launches with no synchronization/synchronous
copy API calls after decode starts, deterministic EOS/lifetime tests, and exact
historical QAT token preservation. Hardware counters and kernel-level trace
verification remain unavailable on this WSL host. Broader shapes, multi-request
scheduling, and a robust decode advantage remain future work.

### Implementation Update

The first tranche now includes an independent reference/optimized projection
microbenchmark, aligned integer cuBLAS GEMM with exact affine correction,
warp-coalesced DP4A decode for native 2/4/8-bit weights, and shared immutable
weight preparation across shape-specialized programs. Reference fallbacks remain
for unsupported alignments and layouts. Native calibration and output rounding
are preserved.

On the A4000, the recorded Whisper final pass improves from 4.134 s to 0.988 s,
OmniVoice from 39.333 s to 2.529 s, and Gemma short prefill from 15.71 to 199.81
tokens/s. Gemma short decode improves from 1.60 to 4.81 tokens/s but remains
slower than CPU. Low-bit expansion increases peak Gemma tracked VRAM to 9.03 GiB;
sharing avoids the preliminary duplicate-expansion peak of about 11.04 GiB.
Current harnesses and retained evidence are in the
[benchmark guide](../benchmarks/README.md#cuda).

At that initial milestone this was partial P0/P1/P2 progress. Stream support,
native capture, and BF16 attention were subsequently implemented as described
above; fused GEMM epilogues, broader tuning, and sustained competitive benchmarks
remain open. Nsight Systems recorded
CUDA API activity but no kernel activity on this WSL host, so actual Tensor Core
kernel selection remains unverified by a trace or hardware counters.

### Initial Baseline

References:

- [CPU reference fixture](../benchmarks/cpu/reference.json).
- [CUDA results and validation limits](../benchmarks/README.md#cuda).
- [Eager architecture](../ARCHITECTURE.md).

| Workload | Reference CPU | Initial CUDA |
| --- | ---: | ---: |
| Gemma short prefill | 127.15 tokens/s | 15.71 tokens/s |
| Gemma short decode | 14.56 tokens/s | 1.60 tokens/s |
| Whisper, 11-second clip | 1.938 s | 4.134 s |
| OmniVoice, 4.2-second output, 16 steps | 9.713 s | 39.333 s |

Initial static code inspection identified these performance hypotheses, not yet a ranked
Nsight profile:

1. `quantized_matrix` assigns one output to a thread and loops serially over K.
   It lacks tiled data reuse, coalesced decode layouts, and Tensor Core execution.
2. Attention materializes scores, launches a separate softmax, and reads scores
   again for the value product instead of streaming through an online softmax.
3. Transfers and device-wide synchronization remain on request/token paths;
   kernel launches and cuBLAS use the default stream.
4. CUDA does not implement the backend's captured-step replay capability.
5. Diffusion ranking and noise indexing contain quadratic scans.

Recorded stage times reinforce the projection-first priority: Whisper CUDA
encoding is approximately 3.11 s of 4.13 s total; OmniVoice generation is
approximately 38.79 s while codec decoding is approximately 0.40 s. The codec is
already faster than its CPU stage in this fixture. Do not optimize the smallest
stage first merely because it is easy to fuse.

## Tahoma Reference Findings

Findings below refer to inspected Tahoma source, not a claim that its H100
measurements reproduce on the A4000.

| Tahoma Component | Useful Mechanism | Kidi Adaptation or Caveat |
| --- | --- | --- |
| Graph-step state | Device-resident, pointer-stable step state; explicit reset; shape identity | Reuse the existing Kidi step contract and add missing device-side control, not another model hierarchy. |
| CUDA-graph decode driver | Warm-up before capture, side stream, thread-local capture mode, replay workspace cache, RNG registration | Native CUDA equivalents; capture mutations must not corrupt live KV, positions, done flags, or RNG state. |
| Continuous LM decode | Reusable pinned staging, decode bursts, precaptured batch buckets | Bounded per-lane caches; no lazy capture stalls in measured request execution. |
| IR evaluator | Capturability checks; invalidation on parameter pointer, device, dtype, and math-policy changes; capture-cost heuristic | Adopt the invariants, not Tahoma's IR. Copying large inputs merely to capture one fused kernel can cost more than replay saves. |
| Q/K normalization and RoPE kernel | Warp/block reductions; fused Q/K normalization, RoPE, and V handling | Support actual Gemma head geometry, shared KV, explicit attention scale, and calibration boundaries. |
| Linear dispatch | Prepared layouts, prequantized activations, shape-specific kernel dispatch | Preserve native checkpoint layout semantics and avoid quantizing the same normalized activation separately for Q/K/V. |
| INT8 GEMM kernels | Quantize once; integer Tensor Core GEMM; fused row/channel-scale epilogue | The inspected implementation uses symmetric quantization, BF16 inputs/output, and Hopper SM90. Kidi needs affine INT8 correction and FP32-output support on SM86. |
| W4A8 GEMM kernel | Persistent packing and M/N/K-dependent schedules | Its A8 is FP8 E4M3, not Kidi's INT8. It is not a native-QAT drop-in. |
| Paged decode attention | Coalesced KV loads, online softmax, GQA, live sequence lengths, sliding windows, typed KV storage | Separate decode and prefill paths; preserve calibrated KV semantics and Gemma's explicit scaling. |
| Paged variable-length attention | Packed variable-length prefill; direct current-KV use; avoid padding copies | Adopt after single-request attention is efficient; library dtype/head-dimension support must be checked. |
| Fusion and JIT | Narrow elementwise/reduction fusion, memoized launch plans, NVRTC disk cache, current-stream launch | Start with curated prepared fusions. Extract only useful dependency-light components after license review; keep compilation outside requests/capture. |
| CUDA-graph benchmark script and MoE optimization journal | Graph on/off comparisons, quality gates, launch/sync counts, memory headroom, capture cost | Reproduce the methodology, not MoE/H100/BF16 speedup numbers. |

Inspect source rather than treating journal/README claims as current guarantees.
Verify Tahoma and vendored-kernel licenses and preserve required notices before
extracting any implementation. LibTorch/ATen wrappers are not dependencies to
bring into Kidi.

## Runtime Architecture

Use three ownership levels:

1. **Device context:** selected CUDA device/context, immutable resident weights,
   prepared weight layouts, device capabilities, and memory-budget accounting.
2. **Execution lane:** explicit compute/copy streams and events, stream-bound
   library handles, pinned staging, mutable workspaces, and graph instances.
   Concurrent requests must not race on a prepared operator's scratch buffers.
3. **Prepared step:** fixed-capacity internal slots, logical tensor views,
   kernel choices, mutable device metadata, graph key, and output ownership.

Model/layer code remains ordinary eager C++. `ops::Context` owns dispatch validation
and delegates step reuse to Kidi's `graph::Session`. Kidi also owns backend graph
orchestration. The `kidi-ops` CUDA backend owns operators, kernels, dispatch,
and prepared storage. Backend `graph::Compiler`
implementations produce `graph::Executable` objects through the common Kidi contract.

All caches are bounded by memory and specialization count. Weight and graph keys
include the relevant device/context, weight generation, layouts, shapes, dtypes,
quantization parameters, and arithmetic policy. A compiled binary can be reused
where compatible, but CUDA module handles and graph executables are context-bound.
Weight replacement, device moves, and math-policy changes invalidate captures.

## Ordered Milestones

### P0: Performance and Quality Harness

- Add NVTX regions and stream-ordered CUDA-event timing for load, prepare,
  prefill/encode, decode/diffusion, codec, transfers, and first output.
- Capture Nsight Systems timelines and, where permitted, Nsight Compute counters:
  kernel/launch counts, synchronization, H2D/D2H, allocation calls, achieved memory
  bandwidth, Tensor Core utilization, occupancy, and register spills.
- Extract real M/N/K and group-size distributions from all three models. Build
  microbenchmarks from those distributions, including batch-1 decode and prompt
  tails, not just large square GEMMs.
- Measure pageable/pinned transfer cost, stream launch overhead, and graph replay
  overhead independently. Establish measured bandwidth/compute ceilings.
- Retain the current fixed-protocol regression benchmark and add a separate
  best-tuned benchmark with explicit memory/latency constraints. Report cold load,
  first-use preparation/JIT/capture, warmed p50/p95/p99, throughput, first-token or
  first-PCM latency, total process VRAM, and runtime-owned memory separately.
- Use longer/more varied workloads and enough repeated requests for uncertainty
  estimates; the current three-repeat fixtures are smoke baselines, not SOTA proof.

**Gate:** reproducible profiles identify the dominant time and resource costs.
Resolve WSL profiling/sanitizer limits or use a suitable GPU validation host;
unavailable checks remain explicitly unverified.

### P1: Stream-Safe, Allocation-Stable Execution

- Thread an explicit lane/stream through every kernel, copy, and library call;
  create cuBLAS/cuBLASLt/cuDNN handles and select algorithms before live execution.
- Use device guards at execution boundaries, not redundant device changes for
  every primitive. Keep device/context isolation and current-device restoration.
- Preallocate dtype-native activation and scratch slots using existing step
  liveness. Reuse immutable parameter residency and prepacked layouts across
  shape variants without unsafe sharing of mutable workspaces.
- Replace device-wide synchronization with stream/event dependencies. Use
  reusable pinned staging and asynchronous transfers only at genuine boundaries.
- Keep tokens, positions, masks/length metadata, sampling/done state, and feedback
  resident where possible. Return selected IDs or requested audio chunks rather
  than intermediate logits. Preserve streaming and cancellation behavior.

**Gate:** after warm-up, supported fixed-shape steps have no allocator churn,
weight uploads, or intermediate D2H transfers. There is no device-wide barrier
per primitive, and concurrent lanes cannot corrupt each other or retained outputs.

### P2: Native Quantized Projection Engine

The compact CUDA build can opt into an ahead-of-time CUTLASS INT8 GEMM. The
SM80-family path handles aligned transposed INT8 weights or single-group packed
INT2/INT4/INT8 weights (K divisible by 16, N divisible by 4, and 16-byte aligned
weights). INT2/INT4 weights expand to signed INT8 for GEMM, increasing GPU memory
use for prepared weights. Other layouts, including grouped packed weights, fall
back to the native kernels. The GEMM preserves the existing affine correction,
scales, bias, and output rounding. Eligible decode shapes still use the packed
vector kernels.

GELU and GEGLU run on the CUDA stream as separate kernels, not GEMM epilogues.
The CUDA runtime test captures a warmed immutable INT8 or packed INT4 projection
followed by GEGLU and replays it with changing activations without Kidi workspace
allocation during capture. Initial preparation and weight rebinding can still
allocate, so warm up with stable weights before capture.

On H100, a warm Gemma 4 E2B QAT run with 32-token prefill chunks went from
256 to 1,876 tok/s on a 36-token prompt and from 248 to 3,528 tok/s on a 162-token
prompt. Decode stayed near 91 and 78 tok/s respectively, and generated token IDs
matched plain compact. The CLI grew from 12.10 to 12.21 MiB with no CUDA toolkit
shared library dependencies. The `kidi_ops_cuda_projection_benchmark` target
reports kernel-level timing and reference error in both compact builds.

W8A16 (INT8 weights with 16-bit activations), direct packed W4A8, and FP8
Tensor Core paths require separate kernels. The CLI's existing `w8a8` alias refers
to E4M3 activations, not the affine INT8 activation math used here.

- Replace the serial kernel with separate families for small-M decode and
  larger-M prefill/encoding/diffusion. Decode needs bandwidth-efficient packed
  GEMV/split-K scheduling; larger M needs tiled, pipelined integer Tensor Core GEMM.
- Use cuBLASLt as a library baseline where its layouts/scaling contract fits;
  use a proven CUTLASS SM80-family implementation for fused/custom epilogues.
  Add SM90+ implementations only behind capability dispatch and hardware tests.
- For affine W8A8 preserve `dot(qA,qW) - zA * sum(qW)`, row scales, channel/group
  scales, bias, accumulator width, and rounding/clamping order. Prepare weight
  sums/layouts once. Symmetric `/127` quantization is not an invisible replacement.
- For native W2/W4 QAT benchmark direct packed decode against one-time expansion
  to signed INT8 for Tensor Core prefill. Expansion must preserve integer values
  exactly, account for extra VRAM, and never happen per token.
- Use aligned, coalesced weight layouts, vector loads, shared-memory staging and
  Ampere `cp.async` where beneficial. Tune tails and small batches explicitly.
- Fuse supported bias, scaling, activation, residual and calibrated output
  quantization epilogues. Reuse prequantized activations across compatible Q/K/V
  or gate/up projections without crossing required quantization boundaries.
- Cache dispatch/autotuning results by hardware, toolkit/library versions,
  dtype/layout, quantization policy, and shape. Tune offline or during bounded
  preparation, never unpredictably on an interactive token.

**Gate:** shape-distribution-weighted kernel results beat the initial kernels
and approach the relevant tuned-library baseline. Report end-to-end impact;
GEMM microbenchmark speedup alone does not complete this milestone.

#### Dense Prefill on H100

Gemma 4 E4B text projections contain about 4.03 billion weights, or 8.06 GB of
BF16 and 16.11 GB of FP32 weights per full sweep, excluding embeddings and the
final vocabulary projection. GEMM does `2*M*N*K` FLOPs. When weight reads
dominate, a 32-token tile gets at most about 32 FLOPs per weight byte at BF16
and 16 at FP32. A 162-token request takes six sweeps at chunk 32 versus two at
chunk 128. Larger chunks help reuse weights, but the tile grid and remainder
shape still determine how much of the GPU is active.

Nsight Systems traced a warmed chunk-32 request with one decode step, excluding
model load. Before tuning, GPU activity occupied 74 of 86 ms at BF16 and 238 of
253 ms at FP32, with only 1-2 ms of GPU copy time. GEMM accounted for 46 and
194 ms respectively, and generic normalization for 15-16 ms. CUDA API wait
time overlaps GPU work and is not an additional latency term. Nsight Compute
sampled one projection per precision: BF16 reached 44% of peak DRAM throughput
and 15% SM throughput; FP32 reached 20% DRAM and 67% L1/TEX throughput. Some
projection shapes launched fewer blocks than the H100's 132 SMs.

Compact CUDA now splits long contractions on underfilled grids and reduces
FP32 partials before output rounding. Workspace is prepared up front and
capped at 2 MiB per projection. Wider BF16 requests use a taller tile where
there are enough output blocks. The strict FP32 path uses 16-byte asynchronous
copies, bank-swizzled shared memory and vector reads, not TF32. Typed RMS keeps
inputs in registers and preserves BF16 rounding at fused boundaries. Fused
attention no longer allocates the fallback score buffer. Scalar async staging,
a generic reduction tweak, and skipping masked K/V reads did not improve the
tested end-to-end workload and were not retained.

Matched unprofiled runs used physical GPU 7, a 162-token prompt, 16 output
tokens, context 512, two warmups and three measured runs. W8A8 is the locally
quantized E4B checkpoint, not E2B QAT. Each row uses the same checkpoint and
input tokens across builds. Rates are prefill tokens per second:

| Precision | Chunk | Previous compact | Current compact | Original shared cuBLAS |
|---|---:|---:|---:|---:|
| BF16 | 32 | 2,254 | 3,825 | 3,957 |
| FP32 | 32 | 712 | 2,103 | 1,778 |
| W8A8 | 32 | 2,580 | 2,928 | 2,207 |
| BF16 | 128 | 4,840 | 6,762 | 8,493 |
| FP32 | 128 | 1,450 | 2,529 | 2,884 |

The compact build matched its previous generated token IDs for both tested
prompts. The BF16 long prompt at chunk 128 already differed from shared cuBLAS
at generated token 8. The CLI grew from 12.67 to 13.05 MiB without adding any
CUDA toolkit shared-library dependencies. On this request at chunk 128, Kidi's
tracked peak GPU memory rose by 0.34 GiB at BF16 and 0.52 GiB at FP32.

Final chunk-32 traces show BF16 GPU activity at 43 ms and FP32 at 87 ms. There
are still about 11-15 ms of gaps between GPU activities, 97 CUDA allocations,
and only one graph launch in those sampled requests. Larger-chunk projection
efficiency is the main remaining compute gap; prefill capture and allocation
reuse are separate opportunities as kernels get faster. Shared cuBLAS is a
reference, not a throughput ceiling.

### P3: Attention and KV

#### Exact 128-Token Prefill Profile

Gemma 4 E4B on physical GPU 7 (H100), batch 1, exactly 128 input tokens with no
prefix reuse, one 128-token chunk, context capacity 512. The BF16, strict FP32,
and locally quantized W8A8 runs used identical input IDs. Nsight Systems
captured one request after two warmups. The window runs from the first GPU
activity through the first token's 4-byte D2H copy; the next decode step is
excluded. Shares are percentages of that elapsed window, not of CUDA API
time. Idle gaps do not identify their host-side cause.

| Operation | Compact BF16 | Compact FP32 | Compact W8A8 | Original shared BF16 |
|---|---:|---:|---:|---:|
| Transformer projection GEMM and reductions | 54.2% | 79.9% | 30.6% | 30.8% |
| Attention (QK, softmax, PV) | 13.9% | 4.4% | 11.5% | 6.5% |
| Normalization, rotary, residual | 8.4% | 2.7% | 7.0% | 24.5% |
| W8A8 activation quantization and epilogue | - | - | 14.3% | - |
| Activations and other pointwise kernels | 3.7% | 1.6% | 4.3% | 5.0% |
| Vocabulary projection and token selection | 2.4% | 1.4% | 1.0% | 3.4% |
| Embedding | 0.1% | <0.1% | 0.1% | 0.1% |
| GPU copies and clears | 1.5% | 1.5% | 4.0% | 2.2% |
| Gaps between GPU activities | 15.7% | 8.6% | 27.3% | 27.7% |
| **Trace window** | **20.09 ms** | **63.33 ms** | **23.85 ms** | **14.21 ms** |

Attention took 2.80 ms in compact BF16 versus 0.92 ms in the original shared
build, counting its QK and PV GEMMs as attention rather than projections.
Matching that reference would recover 1.88 ms of the 5.88 ms BF16 gap at this
shape. Compact BF16 spends 10.89 ms in transformer projections versus 4.37 ms
in the shared reference. Attention is worth optimizing, but eliminating it
entirely would give at most 1.16x at this BF16 shape.

The compact kernel already avoids a global score buffer, but it assigns a CTA
to each query/head and does not explicitly reuse K/V across query tiles. One
sampled BF16 launch (Q=128, KV=128, head width 256) took 67 us and issued about
143 MB of global-load requests at L1, 53 MB of read sectors at L2, and only
0.88 MB of DRAM reads. Those measurements are from a single Nsight Compute
sample with caches and clocks left unchanged. With Q still 128, the median
observed width-256 attention launch increased from 64 us at KV=128 to 303 us at
KV=640 in a separate 1,024-token prompt. That trace did not record every
expected attention launch, so it is used only for per-launch scaling here.

The next attention experiment should tile multiple query rows, reuse K/V on
chip, and use online softmax and Tensor Core QK/PV for BF16. Benchmark the
standalone kernel against a tuned attention implementation before claiming
state-of-the-art performance. Validate Gemma's grouped-query mapping, scale,
causal and sliding masks, live KV offsets, and head widths 256 and 512. Keep
strict FP32 separate; make any probability rounding change explicit. Ship only
the measured AOT variants and track CLI growth and peak memory alongside
end-to-end prefill at short and long KV lengths.

- Use tiled streaming/online-softmax attention without materializing the full
  score tensor. Specialize prefill versus decode, GQA, local windows, head widths,
  and live lengths. Preserve Gemma's supplied scale instead of assuming `1/sqrt(D)`.
- Evaluate FlashAttention/FlashInfer or compatible CUTLASS attention kernels;
  accept only paths that support the actual dtype, geometry, window, and masking
  contract. Keep a native streaming FP32 reference-compatible path where needed.
- Add calibrated INT8 KV support with GPU-side decode of scale/zero metadata;

#### Compact BF16, W8A16, and FlashAttention-2 Update (H100 GPU 7)

Compact CUDA (`release-cuda` with build-time CUTLASS, `15.69 MiB` static
CLI with zero CUDA toolkit shared-library dependencies) now accelerates BF16
projections, native W8A16 projections (`INT8` weights with `BF16` activations,
norms, and KV cache), native W8A8 projections (`INT8` weights + dynamic/calibrated
`INT8` activations with `FP32` norms/attention), and BF16/FP32 FlashAttention-2
prefill across `sm_80` and `sm_90`:

- **Tuned `K=64` BF16 GEMM + shared Split-K workspace:** `64x128x64` (`Stages=3`)
  for wide `gate_up_proj`, `64x64x64`/`32x64x64` Split-K (`0` register spills)
  for `rows > 16`, and a shared `ProjectionCache::workspace` across layers.
- **Native CUDA W8A16 (`pipelined_w8a16_gemm` + `w8a16_vector` + CUDA Graph
  sibling fusion):** zero-spill `64x128`/`64x64` Tensor Core W8A16 GEMM with
  bank-conflict-free `W8WeightStride=80` shared memory, `prmt+lop3+hsub2` fast
  `int8 -> bfloat162` conversion, whole-group Split-K, 128-bit `w8a16_vector`
  decode GEMV, horizontal `q_proj + kv_proj` CUDA Graph fusion (`M=1`), and
  direct load-time BF16-to-W8A16 packing (`--precision w8a16`) without duplicate
  BF16 projection VRAM.
- **Native CUDA W8A8 (`pipelined_w8a8_gemm` + `quantized_vector_fast` +
  `decode_partitions<float>` + `flash_attention_f32`):** zero-spill `64x64x64`/
  `16x64x64` `mma.m16n8k32` INT8 GEMM with fused affine epilogue and shared Split-K
  workspace (eliminating `projection_epilogue` and per-layer `accumulators`
  buffers), vectorized `float4` warp-shuffle `quantize_rows`, 128-bit `uint4`
  `__dp4a` `quantized_vector_fast` with `ColsPerWarp=4` for wide `N`, horizontal
  `q_proj + kv_proj` CUDA Graph fusion, `gelu_gate` on CUDA, GQA-grouped
  `decode_partitions<float>` FlashDecoding, and tiled `Br=8, Bc=16` FP32
  FlashAttention-2 prefill (`0` spills).
- **Tiled Tensor Core FlashAttention-2 (`flash_attention.{h,cu}`):**
  warp-split-`D` (`Br=16, Bc=32`, 4 warps) using `cp.async`, `ldmatrix`, and
  `mma.sync.m16n8k16` for `head_dim ∈ {64, 128, 256, 512}` (`0` spills on both
  `sm_80` and `sm_90`), GQA query-head packing, 16-byte XOR-swizzled shared
  memory, early causal/sliding mask tile skipping, and zero score-buffer
  allocation.

Matched Gemma 4 E4B results on physical GPU 7 (H100 80GB, `chunk=128`, 2 warmups,
3 repeats at 162 tokens; Nsight Systems exact-128 window excluding model load):

| Configuration | 162-tok Prefill (tok/s) | 162-tok Decode (tok/s) | 1,098-tok Prefill (tok/s) | Exact-128 Proj GEMM (ms) | Exact-128 Attention (ms) | Exact-128 Window (ms) | Peak GPU Mem (GiB) | CLI Size (MiB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Original shared cuBLAS BF16 | 8,493 | 105.6 | 15,226 | 4.37 | 0.92 | 14.21 | 15.24 | 9.85 *(dyn)* |
| Previous compact BF16 | 6,762 | 127.1 | — | 10.89 | 2.80 | 20.09 | 15.58 | 13.05 |
| **Current compact BF16** | **9,281** | **134.1** | **14,273** | **5.51** | **0.78** | **11.93** | **15.26** | **15.69** |
| **Current compact W8A16** | **8,503** | **159.3** | **12,040** | **7.99** | **0.78** | **13.89** | **7.95** | **15.69** |
| **Current compact W8A8** | **7,735** | **139.3** | **9,974** | **6.59** *(+1.01 quant)* | **1.71** | **14.64** | **8.58** | **15.69** |

At `chunk=1024` on a 1,098-token prompt (`65` attention launches across all
layers), compact BF16 FlashAttention-2 takes `7.11 ms` (`17.1%` of the `41.69 ms`
chunk window). Compact BF16, W8A16, and W8A8 match reference generated token IDs
and pass `VerifyPortableElf.cmake` (`kidi_ops_cuda_test` verified both with and
without `KIDI_CUTLASS_ROOT`).
  avoid repeatedly scanning padding or expanding whole caches to floating point.
- Introduce paged/heterogeneous KV and prefix reuse when batching/concurrency
  justify them. Account for Gemma producer/shared-cache relationships rather
  than allocating independent pages for every logical layer.

**Gate:** attention/VRAM scaling is measured across short and long contexts,
mask/window boundaries, shared KV, cache reuse, and concurrent sequences.

### P4: Targeted Fusion

Prioritize measured launch and memory traffic:

- residual + RMSNorm/LayerNorm + supported activation quantization;
- QKV projection epilogues + Q/K normalization + RoPE + calibrated KV writes;
- gate/up + GeGLU/SwiGLU + next-input quantization;
- mask/scale/softmax within streaming attention;
- GPU argmax, EOS/done update, position advance, and next-token feedback;
- Snake + unfold/quantize + projection; transposed-convolution folding and bias
  where exact semantics permit it;
- diffusion candidate scoring plus CUB-style scan/select/top-k, replacing
  quadratic rank/noise-index scans while preserving tie and RNG conventions.

Retain named mathematical/quantization boundaries. Avoid a single giant kernel
that spills registers or destroys occupancy. Start with prepared templates or
small curated recipes; consider Tahoma-style NVRTC codegen only when measured
shape/fusion coverage justifies compile latency, disk-cache, and maintenance cost.

**Gate:** fewer launches/intermediate bytes and better phase latency, not just
fewer source-level operators. Preparation and compile costs are reported.

### P5: Native CUDA Graph Capture and Replay

- Integrate with existing captured steps. Do not merely turn on `supports_replay`:
  host-returning token selection and host-generated per-step metadata must first
  have explicit device-resident bindings and readback boundaries.
- Warm libraries, algorithms, modules/JIT code, and workspaces before capturing
  on a dedicated lane with the appropriate capture mode and stream dependencies.
  No allocation, compilation, handle creation, or synchronous host read belongs
  inside the captured computation.
- Persist input/output addresses, KV tables, token IDs, positions, masks/lengths,
  quantization metadata, and done flags. Changing values are device inputs, not
  accidentally baked scalar kernel arguments.
- Warm/capture against scratch or resettable state; restore KV, sequence positions,
  token buffers and RNG after capture. Preserve graph-safe seeded randomness.
- Bucket by batch, prefill length, context/KV geometry, TTS target length, and
  other structural choices. Prewarm the supported hot buckets and bound cache
  growth. Do not capture a new graph in a measured request unexpectedly.
- Start with one decode or diffusion step and stable encoder/codec regions.
  Add bounded decode bursts or multi-step capture only after cancellation,
  EOS, progress updates and first-output latency remain correct.
- Preserve output lifetime: a later replay must not overwrite a tensor/audio
  chunk still owned by a caller. Separate mutable state per execution lane.

**Gate:** graph on/off outputs agree under the same arithmetic policy; repeated
requests rebind correctly; cold capture and warm replay are measured separately;
traces show replay reduces a real launch/control bottleneck. Skip capture when
extra staging costs more than the launches it removes.

### P6: Model-Level Optimization and Serving

- **Gemma:** fast packed decode, efficient prompt GEMM/attention, device-resident
  greedy loop, shared/local KV behavior, then chunked prefill, prefix reuse and
  continuous batching. Measure TTFT and inter-token p95, not throughput alone.
- **Whisper:** prioritize encoder projections/attention, cache cross-attention
  K/V, capture encoder and decoder buckets, and retain fitted-audio behavior.
  Report padded-30-second and fitted-audio protocols separately.
- **OmniVoice:** prioritize transformer/diffusion projections before the codec;
  capture fixed-shape diffusion work, then optimize GPU selection and streaming
  codec chunks. Keep text, steps, pace, seed/RNG policy, voice and duration fixed
  for comparisons. Use cuDNN only where its quantization/layout math is equivalent;
  otherwise retain exact unfold + integer GEMM + fold semantics.
- Add independent request lanes, batching and admission budgets with separate
  immutable weights and mutable state. Validate one-GPU-per-replica multi-device
  execution first; NCCL/tensor parallelism is a separate, evidence-driven project.

**Gate:** real model latency/throughput wins, bounded memory, stable p95/p99,
preserved streaming/cancellation, and multi-request/device isolation.

## Precision, Hardware and Quality Policy

Default: preserve native QAT and affine W8A8 behavior; never replace the checkpoint
with FP16 weights or change calibration silently. Ampere SM86 has no Hopper TMA,
WGMMA, or native FP8 Tensor Core path. Tahoma's SM90/FP8 kernels need hardware
dispatch or replacement, not optimistic compilation flags.

An optional fast-math policy may evaluate BF16/FP16 attention, TF32, alternative
KV formats, or changed RNG/sampling. Each is a named, separately benchmarked
quality/performance policy requiring explicit approval before becoming default.
Keep a reference-compatible path so performance changes can be diagnosed.

Extend correctness beyond the present smoke fixtures:

- adversarial quantization ranges/rounding, group boundaries, packed tails,
  zero-length/broadcast views, masking, explicit attention scale, and KV offsets;
- per-layer/teacher-forced logit comparisons and a held-out multilingual prompt
  suite for Gemma, not keyword checks alone;
- Whisper WER/CER across varied languages, durations, noise, and silence;
- TTS intelligibility, timing, voice/speaker consistency and listening samples
  across short/long/multilingual utterances; one zero-WER proxy is insufficient;
- graph request rebinding, parameter replacement, output retention, EOS,
  cancellation, concurrency, unavailable devices, and real second-GPU runs;
- Compute Sanitizer/race checks on an enabled WSL or native Linux GPU host.

## Competitive Benchmarks and Release Gates

Maintain two scoreboards: interactive batch-1 and serving throughput under stated
latency/VRAM budgets. Compare compatible exact checkpoints and policies first.
Evaluate TensorRT-LLM, vLLM/SGLang, faster-whisper/CTranslate2, and the upstream
OmniVoice implementation only after confirming relevant model/precision support.
Do not present different checkpoints, beam widths, steps, durations or quantization
policies as backend-only speedups.

1. **Recover:** beat the recorded 8-thread CPU baseline for all three target
   workloads without quality regressions. This is a milestone, not SOTA.
2. **Compete:** meet or beat the strongest supported matched external baseline,
   with uncertainty, p95 latency, memory and first-output behavior reported.
3. **Sustain:** keep those results over long contexts, varied inputs, concurrent
   requests and repeated runs; publish exact versions, hashes and settings.

No universal tokens/s target or multiplicative speedup is promised before P0.
Use measured weight/KV bytes, achieved bandwidth and integer compute ceilings to
set credible per-hardware targets. A global SOTA claim needs measurements on the
claimed hardware, including newer architectures and a real multi-GPU host.

## First Implementation Tranche

Implement P0, the stream/resource foundation from P1, and the W8A8 GEMM baseline
from P2 first. Benchmark representative Whisper/OmniVoice projections plus Gemma
decode and prefill separately. Select the first production kernel/layout from
that evidence, then extend native packed QAT and attention. Begin graph-enabling
interfaces early, but prioritize replay once traces show launch overhead matters.

Before implementation, confirm these proposed defaults: interactive latency
first, SM86 first, native quantization by default, optional changed-precision paths
kept separate, and newer-GPU/multi-GPU validation when hardware is available.