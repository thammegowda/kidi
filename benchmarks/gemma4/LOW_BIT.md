# Packed Execution Study

The [continuation](#continuation-residual-fusion-and-local-history) below records
the next optimization pass, including a long-context numerical caveat.

## Outcome

Kidi now supports in-memory packed Q4/Q8 execution without changing the original
Safetensors checkpoint. On the local M5, the faster mixed setting improved
128-token-prompt decoding from **13.76 to 36.34 tokens/s on CPU** and **25.53 to
56.74 tokens/s on GPU**. These are observed 2.64x and 2.22x improvements over the
saved pre-change binary, not statistically established universal speedups.

**LiteRT-LM is still faster.** Its refreshed non-speculative rates were 46.11 CPU
and 102.30 GPU tokens/s on this workload. The remaining gap is about 1.27x on CPU
and 1.80x on GPU. Kidi's post-training quantization is not equivalent to LiteRT's
released mobile quantization scheme, and their phase counters differ slightly.

## What Was Measured

Apple M5, 10 CPU cores, 16 GiB unified memory, macOS 26.6.2; batch size one,
four CPU threads, 2048-token context. Each case ran sequentially with two warmups
and three measurements in one persistent process. Prompts and effective input
token IDs match the earlier [baseline methodology](README.md). Kidi generates
65 tokens, timing 64 recurrent decode calls; LiteRT reports 65 decode tokens.
Kidi's decoder timing excludes host token selection. No inference, builds, or
tests were run concurrently with timed cases. Each table entry is a median from
one process per case; there are no independent-process confidence intervals.

The before binary is `.cache/kidi-gemma-before-lowbit`, SHA256
`4aced26fcbe4ac0417fae0504c8ad912de0ec576c79239024a0f15ba231d9b58`.
The measured final binary is SHA256
`3f38cef2c792938e8104e778f1e18944b5bf7c83fd340fb814494afbbb767490`,
based on commit `2c3801f` plus this working-tree change. Model pins and
LiteRT-LM 0.17.1 are unchanged from the baseline report.

Raw measurements and quality data: [lowbit-results.json](lowbit-results.json).
All measured final Kidi warm runs reported zero additional preparation.
Load-time packing is excluded from warm throughput; smoke runs took roughly
5.6-6.2 seconds to load and pack, versus under two seconds without packing.

## Profile and Changes

The initial decode-shaped CPU profile attributed about 4.18 seconds to linear
operators and 0.78 seconds to attention over the sampled run, making projections
roughly 83% of recorded decode operator execution time. GPU synchronized profiles
showed many expensive small dispatches, but inserted fences distort normal GPU
execution and are not kernel timestamp measurements. A non-fencing Q4 profile
then showed GPU host encoding dominated by RMSNorm, attention, rotary, and
elementwise operations rather than the packed projection encoder.

Implemented changes:

- `ops::pack_weight` creates signed packed bytes plus FP32 scales directly from
  FP32/BF16 tensors. Packing is symmetric and groupwise; original data is read-only.
- `ops::Context` prepares and shares packed weights across input shapes, retaining
  owners to prevent pointer-identity reuse. Incompatible aliases are not matched.
- CPU uses native YNNPACK INT4/INT8 dot kernels with dynamic INT8 activations.
  Odd Q4 output widths are padded privately and sliced back to the logical shape.
- Metal uses vectorized packed GEMV and a tiled GEMM implementation. Tiles are
  decoded into bounded threadgroup FP16 storage with FP32 accumulation; decode
  GEMV uses FP32 activations. It never expands the full packed matrix per call.
  Function constants specialize bit widths and group sizes.
- Attention reads only active prefix buckets of 128 tokens, preserving the
  original cache and local/global masking semantics. It does not yet use a
  circular local-attention cache.
- Supported FP32 RMSNorm, rotary, tanh-GELU, tanh, and simple elementwise operations
  use direct Metal kernels. Other shapes and types retain the MPSGraph path.
  Explicit saturation handles large tanh/GELU inputs without NaNs.
- Default low-bit execution uses original floating weights for multi-row prefill,
  packed weights for single-row decoding. `--packed-prefill` also uses packed GEMM;
  it is currently slower than the native floating prefill path on this machine.

Experiments not retained as defaults:

- Uniform Q4 on every projection produced unacceptable drift in the initial
  fixed-token probe (about 69% CPU / 79% GPU next-token agreement). Final Q4 mode
  instead uses Q4 MLP matrices and per-output-channel Q8 elsewhere.
- Single-thread small packed CPU projections did not improve the measured
  workload and were removed.
- The first scalar packed Metal kernel was slower than BF16; vectorized word loads
  replaced it before final measurement.

## Performance

Prefill and decode rates are tokens/s. `Q4 g128` and `Q4 g32` mean Q4 MLPs with
128- or 32-value groups plus per-channel Q8 attention, PLE, and vocabulary head.
Embedding lookup tables remain original BF16. Native floating prefill is enabled.

| Runtime | Input | CPU Prefill | CPU Decode | GPU Prefill | GPU Decode |
|---|---:|---:|---:|---:|---:|
| Before, BF16 | 128 | 230.59 | 13.76 | 1001.14 | 25.53 |
| Kidi Q4 g128 | 128 | 282.41 | 36.34 | 1140.80 | 56.74 |
| Kidi Q4 g32 | 128 | 307.17 | 29.10 | 1148.99 | 52.13 |
| Kidi Q8 | 128 | 249.48 | 27.57 | 1313.58 | 41.57 |
| LiteRT-LM | 128 | 668.20 | 46.11 | 2312.66 | 102.30 |
| Before, BF16 | 1024 | 230.18 | 13.55 | 1042.23 | 25.37 |
| Kidi Q4 g128 | 1024 | 273.67 | 28.64 | 1220.24 | 50.95 |
| Kidi Q4 g32 | 1024 | 257.10 | 23.25 | 1239.93 | 47.79 |
| LiteRT-LM | 1024 | 538.58 | 45.98 | 2625.26 | 98.93 |

This change does not implement MTP. Earlier LiteRT-LM MTP measurements remain in
the baseline report and are not mixed into this ordinary-decoding comparison.

### Memory

Packing adds resident packed buffers; original model parameters are retained for
registered state semantics and native prefill. It is **not** a low-bit-only memory
footprint. Process peak RSS in the final mixed runs was roughly 9.8-10.9 GB, versus
8.1-9.6 GB for the pre-change BF16 process. RSS includes resident mapped pages and
does not uniformly account for GPU allocations, so it is not comparable to a
complete device-memory budget or a precise LiteRT memory ratio. Native CPU
preparation can also create private packed representations. Reducing retained
original/device copies remains work to do.

## Quality

`kidi_gemma4_quality` compares the same 156 teacher-forced next-token distributions
over six short passages using **incremental decoding**. It reports top-1 agreement,
negative log-likelihood change, and mean KL divergence against the same backend's
original BF16 path. This is a small regression probe, not a representative quality
benchmark or proof of equivalence to the original model.

| Mode | CPU Agreement | GPU Agreement | CPU NLL Delta | GPU NLL Delta | CPU KL | GPU KL |
|---|---:|---:|---:|---:|---:|---:|
| Q4 g128 | 87.18% | 87.82% | +0.1155 | +0.1093 | 0.2601 | 0.2578 |
| Q4 g32 | 91.03% | 92.31% | -0.1595 | -0.1259 | 0.2273 | 0.1670 |
| Q8 | 96.15% | 98.08% | +0.0054 | +0.0009 | 0.0371 | 0.0045 |

Lower NLL on this tiny sample does not establish a quality improvement. Q4 is an
explicit speed/quality tradeoff, not a transparent optimization. Q8 is the more
conservative measured low-bit option; BF16 remains the default. CPU activation
quantization and GPU FP32 GEMV also mean low-bit backends need not produce identical
token sequences. Broader held-out perplexity and task evaluations are required
before production quality claims.

## Validation

All 16 CTest targets pass. Existing CPU and Metal RTG regression outputs retain
chrF2 100 over all 50 sentences. Focused tests cover signed packing, zero groups,
group boundaries, odd output widths, vector and tiled tails, dynamic rebinding,
quantization reloads, large GELU inputs, and the 128-token attention-bucket
boundary. Independent FP32 Transformers Gemma logits still match for full-prefix,
incremental, and chunked execution. Real Q4 Paris smoke outputs match the original
CPU/GPU answer, but this is not a substitute for the quality measurements above.

## Reproduce

Use the original checkpoint and benchmark environment from [README.md](README.md).

```sh
build-release/kidi generate --model ../models/gemma-4-E2B-it \
  --backend mps --weight-bits 4 --group-size 32 \
  --prompt 'Explain why the sky is blue.' --profile

.cache/gemma-venv/bin/python benchmarks/gemma4/run.py \
  --runtime kidi --backend gpu --weight-bits 4 --group-size 32 \
  --prefill 128 --decode 64 --warmups 2 --runs 3 \
  --output .cache/gemma-bench/q4-gpu.json

cmake --preset release -DKIDI_BUILD_BENCHMARKS=ON
cmake --build build-release --target kidi_gemma4_quality
build-release/kidi_gemma4_quality ../models/gemma-4-E2B-it gpu 4 32
```

Use `--weight-bits 8` for per-channel Q8, or omit the flag for original precision.
For profiling, use `KIDI_PROFILE_OPS=host` to preserve GPU enqueue behavior or
`KIDI_PROFILE_OPS=sync` for intrusive per-operation synchronization. Neither
reports GPU hardware timestamps. Keep profiling separate from throughput runs.

## Remaining Priorities

1. Better low-bit calibration or a compatible quantization-aware checkpoint,
   with broader quality gates. Do not call naive PTQ equivalent to LiteRT's model.
2. Reduce retained weight copies and implement an efficient packed prefill path,
   so low-bit execution does not require a floating prefill copy.
3. Improve local-attention cache layout and fused attention; longer prefixes
   still cost more even when most local-attention positions are masked.
4. Fuse normalization/residual/activation stages where profiles justify it,
   and measure operator-private scheduling without introducing a model graph.
5. Reassess MTP separately after ordinary decoding and quality are established.

## Continuation: Residual Fusion and Local History

The next pass kept the quantization policy unchanged and investigated residual
dispatch cost, attention, and packed GEMV. Two changes were retained:

- Post-RMSNorm, residual addition, and optional output scaling are one eager
  operator on both backends. Gemma eliminates 105 separate residual-add calls and
  35 scale calls per token. The model equations and stored weights are unchanged.
- CPU local-attention layers crop old masked history within the prepared attention
  operator. The range is rounded to 128-token buckets, retains enough history for
  the first query of each chunk, and leaves shared K/V storage intact. GPU retains
  its previous range because cropping did not show a useful speed gain there.

Measured decode rates, batch one, mixed Q4 g128 / per-channel Q8, four CPU threads:

| Case | Before | Retained Changes | Observed Change |
|---|---:|---:|---:|
| CPU, 128-token prompt | 36.26 | 36.76 | +1.4% |
| GPU, 128-token prompt | 56.77 | 57.86 | +1.9% |
| CPU, 1024-token prompt | 29.62 | 30.19 | +1.9% |
| CPU, 4096-token prompt | 16.49 | 19.24 | +16.7% |

The short-prompt effects are small and may include run-to-run noise. The 4096-token
case uses an 8192-token cache; the other cases use 2048. Each measurement is a
median of three repeats after two warmups in a single process, with sequential
before/after runs. These are not independent-process confidence intervals. Warm
operator preparation was zero. LiteRT-LM was not remeasured in this continuation;
its faster measurements above still define the remaining gap, not proof of parity.

### Numerical Caveat

The three generated token sequences matched exactly before/after for the measured
128- and 1024-token prompts on both devices. The 4096-token CPU case differed
deterministically starting at generated token 21. Cropping changes floating-point
reduction geometry, and the changed shapes can affect the quantized computation
downstream. The full-prefix/chunked FP32 tests still pass; this is not a claim of
bitwise equivalence for the full quantized model.

A same-token long-context probe compares cropped and uncropped execution using
the same Q4/Q8 weights and 64 forced positions after the identical 4096-token
prompt. Top-1 agreement is 62/64 (96.875%), mean KL is 0.02933, and mean NLL delta
is +0.01353 nats. These numbers characterize this one sequence, not general model
quality. The earlier 156-position quantization probe is unchanged on both devices.

Use `--full-attention-cache` to disable CPU history cropping. It restored the exact
saved baseline token sequence on the divergent long-prompt case. Programmatic
callers can set `Gemma4State::crop_local_attention = false`. The original full-history
path remains available rather than hiding the numerical tradeoff.

```sh
build-release/kidi generate --model ../models/gemma-4-E2B-it \
  --backend ynnpack --weight-bits 4 --full-attention-cache --prompt 'Explain light scattering.'

build-release/kidi_gemma4_quality ../models/gemma-4-E2B-it cpu 4 128 \
  .cache/gemma-bench/before-cpu-p4096.json
```

The optional last quality-probe argument is a benchmark JSON containing
`prompt_tokens` and `records[0].token_ids`. Without it the original short
quantization-quality probe runs. Window mode compares full/cropped history at the
same precision; its legacy `quantized_nll` field means the cropped run's NLL.

### Rejected Experiments

A backend-private split attention kernel passed the numerical fixtures and
preserved sampled generated tokens, but reduced GPU decode to about 53.2 tokens/s
for the short prompt and 47.1 for the long prompt. It was removed. A four-output
rows-per-SIMD packed GEMV variant also failed to improve throughput (55.9 tokens/s)
and was removed. Hoisting the per-channel Q8 scale after reduction showed no
reliable gain (56.7 tokens/s) and was removed. These trials do not justify retaining
extra implementation paths or changing the numerical policy.

Raw evidence is in [continuation-results.json](continuation-results.json), including
rejected trials and the single cold compatibility run, which must not be treated
as a warm speed result. The `window-gpu-p1024` entry is the rejected GPU cropping
trial, not the final GPU path. The before binary is `.cache/kidi-before-attention`,
SHA256 `3f38cef2c792938e8104e778f1e18944b5bf7c83fd340fb814494afbbb767490`.

Validation: all 16 CTest targets and strict CPU/Metal RTG regression gates pass.
New tests cover fused residual scaling, grouped-attention masks and tails,
singleton-mask broadcasting, cropped/full history, and chunk boundaries past the
local-window bucket. No checkpoint files were rewritten or requantized on disk.