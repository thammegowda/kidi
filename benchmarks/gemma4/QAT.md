# Native Mobile QAT

The newer [prefill follow-up](#prefill-follow-up) below preserves this checkpoint
and reports independent paired measurements. The original results remain intact.

## Result

Kidi now loads Google's trained mixed-Q2/Q4/Q8 Gemma 4 E2B checkpoint directly
from Safetensors. It does not requantize these trained weights or rewrite the
checkpoint. This is a separate model from the original BF16 checkpoint and the
earlier Kidi post-training quantization experiments.

**The full LiteRT-LM performance target is not met.** Kidi exceeds LiteRT-LM's
measured short-prompt CPU decode rate, but remains behind on long-prompt CPU
decode, GPU decode, and prefill on both devices.

Fresh median rates, tokens/s, batch size one:

| Runtime | Input Tokens | CPU Prefill | CPU Decode | GPU Prefill | GPU Decode |
|---|---:|---:|---:|---:|---:|
| Kidi native QAT | 128 | 512.38 | **55.00** | 1237.63 | 76.20 |
| LiteRT-LM | 128 | 740.19 | 48.65 | 2194.18 | **99.40** |
| Kidi native QAT | 1024 | 459.83 | 43.34 | 1129.07 | 67.34 |
| LiteRT-LM | 1024 | **591.72** | **48.29** | **2627.28** | **101.24** |

The short-prompt CPU decode improvement over LiteRT is about 13%; this does not
establish general runtime parity. Native QAT also improves Kidi GPU decode from
the previous mixed-PTQ rate near 58 tokens/s to about 76, but that comparison
changes the checkpoint and compute policy and is not a kernel-only speedup.

Raw timing, outputs, memory and reference-quality measurements are in
[qat-results.json](qat-results.json). Earlier results remain separate in
[LOW_BIT.md](LOW_BIT.md).

## Why the Checkpoint Matters

Inspection of the released E2B LiteRT bundle found:

- 45 Q4 MLP matrices in the first 15 layers.
- 60 Q2 MLP matrices in the later 20 layers.
- Q4 attention projections, Q8 per-layer projections, and a Q2 vocabulary head.
- Trained activation and cache calibration, not just smaller weight storage.

The quantized projection payload in that bundle is approximately 777 MB. Kidi's
earlier Q4-MLP/Q8-other policy moved substantially more weight data per token.
Naively changing that BF16 model to Q2 would not preserve the trained model.

The native checkpoint is
`google/gemma-4-E2B-it-qat-mobile-transformers` at
`dd693ff40353f057ca5f07e945ad867f4afbf2ec`. After converting only offset-packed
byte encoding to signed packed encoding, **all 276 compared text projection
matrices and their per-channel scales match** the LiteRT bundle, including the
vocabulary head. The unquantized per-layer model projection is excluded from this
comparison. This is not a proof of complete graph equivalence: activation
precision, prefill execution, cache representation, and auxiliary transformations
can differ.

The LiteRT model remains
`litert-community/gemma-4-E2B-it-litert-lm` at
`b3ca0d2f076785a8f4b2219ddbd2bdb99954eae1`, executed by LiteRT-LM 0.17.1.

## Implementation

- Original QAT Safetensors are read directly. The setup helper writes only Kidi
  YAML, including the original quantization policy. The original BF16 model and
  opt-in PTQ paths remain available and unchanged by model selection.
- Q2/Q4 offset bytes are changed to signed packed bytes in memory; Q8 signed bytes
  retain their values. No trained scale is recalculated.
- Quantized embeddings gather and dequantize only selected rows, including the
  per-layer embedding table's independent layer scales.
- Static activation rounding uses round-to-even, signed INT8 clipping, and the
  trained scale. A zero scale bypasses rounding. K/V cache scales are honored.
- CPU calibrated projections use YNNPACK's native integer dot path directly at
  the trained activation scale. Removing redundant dynamic requantization improved
  the initial native-QAT CPU decode trial from about 26 to 55 tokens/s.
- Metal rounds each input once into persistent operator scratch. Packed GEMV
  consumes that input; output rounding is fused into the projection. Function
  constants remove unused calibration branches. Keeping rounding within the
  prepared projection avoids exceeding the bounded eager operator cache.
- For calibrated multi-row Metal prefill, prepared operators cache an expanded
  FP16 weight matrix and use native MPS matrix multiplication, retaining output
  calibration. `--packed-prefill` instead uses packed tiled GEMM. The cached path
  improved measured GPU prefill from roughly 412 to 1238 tokens/s. Decode remains
  packed in either case.

The separate unquantized per-layer model projection stays at its checkpoint dtype.
CUDA/NPU execution, multimodal encoders, and MTP are not added by this change.

## Method and Limits

Hardware: Apple M5, 10 CPU cores, 16 GiB unified memory, macOS 26.6.2. CPU threads:
four for both runtimes. Batch: one. Context: 2048. Kidi prefill chunk: 128.
Two warmups and three measured repeats in one persistent process per case,
sequential runs without competing builds or inference. Tables are medians, not
independent-process confidence intervals. Prompt token IDs are checked against
LiteRT's tokenizer, accounting for its session-inserted BOS.

Both receive a 65-token output budget. Kidi obtains the first token from prefill
and times 64 recurrent calls; LiteRT reports 65 decode tokens. Kidi's decoder
timer excludes host selection; LiteRT uses native benchmark counters. The phase
boundaries therefore are not identical. These are warm phase rates, not cold or
end-to-end application throughput. All final Kidi repeats reported zero additional
operator preparation. Speculative decoding is disabled in these comparisons.

Eight-thread tests made decode slower: Kidi measured 38.81 / 31.42 tokens/s for
short/long prompts, versus LiteRT's 46.35 / 45.82. Four threads remain the better
measured decode setting. Increasing Kidi's prefill chunk to 256 did not close the
prefill gap. A cropped-range CPU scheduling experiment was slower and removed.

Process peak RSS was about 6.0-6.1 GB for Kidi CPU and 6.7-7.4 GB for GPU. Original
mapped checkpoint pages, in-memory signed representations, and prepared buffers
all contribute. The default GPU path retains expanded FP16 prefill matrices;
it is not packed-only execution or a low-bit-only memory budget. RSS is not a
uniform measure of total GPU/unified allocation across the two runtimes.

## Correctness

The existing offline Gemma test now also reads a separate randomly generated QAT
fixture made with official Transformers quantized layers. It checks all 80 logits
through full-prefix, incremental, chunked, and optional packed-prefill execution
on CPU and Metal. Tests cover Q2/Q4/Q8, offset encoding conversion, per-layer
embedding scales, activation calibration, and rounded K/V caches. Dedicated
operator tests cover rounding ties and saturation. Random model calibration
constants avoid accidental exact half-step dot-product degeneracies; FP32 and
integer accumulation can otherwise cross a rounding threshold by tiny errors.

A full-checkpoint FP32 reference using Transformers 5.17.0 and PyTorch 2.14.0
was also evaluated on 29 forced positions, with the same trained cache rounding:

| Backend | Next-Token Agreement | Mean KL | NLL Delta |
|---|---:|---:|---:|
| CPU | 28/29 (96.55%) | 0.05721 | +0.00884 |
| GPU | 29/29 (100%) | 0.03653 | -0.00579 |

This small probe supports the implementation, not a claim of broad quality parity
or exact logits. CPU integer arithmetic, the BF16 auxiliary projection, and GPU
FP16 prefill differ from a fully FP32 reference. The probe appends a synthetic EOS
for the last next-token score; absolute NLL is not a corpus perplexity estimate.
Wider quality evaluation remains necessary. Real-model CPU/GPU smoke tests produce
the expected Paris answer. All 16 CTest targets pass; strict RTG CPU and Metal
regressions retain chrF2 100 over 50 sentences.

## Reproduce

Use the existing optional benchmark environment, or install
[requirements.txt](requirements.txt) in a Python 3.12 environment.

```sh
.cache/gemma-venv/bin/hf download google/gemma-4-E2B-it-qat-mobile-transformers \
  config.json model.safetensors tokenizer.json tokenizer_config.json chat_template.jinja \
  --revision dd693ff40353f057ca5f07e945ad867f4afbf2ec \
  --local-dir ../models/gemma-4-E2B-it-qat-mobile-transformers
.cache/gemma-venv/bin/python tools/configure_gemma4.py \
  ../models/gemma-4-E2B-it-qat-mobile-transformers
build-release/kidi generate --model ../models/gemma-4-E2B-it-qat-mobile-transformers \
  --backend ynnpack --threads 4 --prompt 'What is the capital of France?' --profile
```

Do not pass `--weight-bits` for native QAT; the trained checkpoint defines it.
The loader rejects PTQ overrides instead of silently changing the model.

```sh
.cache/gemma-venv/bin/python benchmarks/gemma4/run.py \
  --runtime kidi --backend gpu --model ../models/gemma-4-E2B-it-qat-mobile-transformers \
  --prefill 128 --decode 64 --warmups 2 --runs 3 --output .cache/gemma-bench/qat-gpu.json
```

Repeat with `--runtime litert`, `--backend cpu`, and `--prefill 1024` sequentially.
The harness labels native QAT, PTQ, and floating execution separately.

To generate and evaluate the independent full-checkpoint reference, additionally
install PyTorch 2.14.0, Transformers 5.17.0, and Safetensors 0.8.0:

```sh
.cache/gemma-venv/bin/python tests/gemma4_reference.py .cache/gemma-qat-reference \
  --checkpoint ../models/gemma-4-E2B-it-qat-mobile-transformers
build-release/kidi_gemma4_quality ../models/gemma-4-E2B-it-qat-mobile-transformers gpu 0 128 \
  .cache/gemma-qat-reference/reference.safetensors
```

`kidi_metal_packed_bench` is an optional microbenchmark with numerical checks and
GPU command-buffer timestamps for the real projection shapes. It reports repeated
queued kernels, which are not end-to-end inference. Q8 vocabulary projection
reached approximately 135 GB/s; Q2 projections reached roughly 80 GB/s. Small-Q4
lane tuning, Q2 scale/accumulator rearrangement, and shared compute encoders failed
to improve model throughput and were removed. Further work should target efficient
Q2 projections and prefill without full expanded matrices, using these measurements
rather than assuming that additional fusion will close the gap.

## Prefill Follow-up

The retained changes are direct contiguous K/V writes and fused FP32 activation
calibration plus FP16 conversion for Metal's cached-matrix prefill. Calibration
still rounds in FP32 before converting to FP16. Checkpoint values, cached weight
identity, decode kernels, CPU thread policy, and the default chunk size of 128
are unchanged. No extra full-model weight copy or model graph was introduced.

Gemma profiling now skips whole warmup requests and separates embedding/PLE,
transformer body, first-token head, and recurrent decoding. The benchmark now
forwards `--packed-prefill` for native QAT independently of PTQ flags, checks the
executed policy, and records executable/config hashes and requested chunk size.

### Paired Rates

Five fresh processes per runtime/backend/prompt, two warmups and three measured
requests per process; 60 processes total. Each trial alternates before/after
order, with LiteRT also refreshed. Batch one, four CPU threads, context 2048,
native QAT, MTP off, 65 output tokens. Nothing else was built or benchmarked
concurrently. Tables use the median of the five process medians, not 15
independent samples. Absolute rates drifted downward over the run, especially on
CPU; the paired ratios are more informative than cross-session comparisons.

| Device | Input | Before Prefill | After Prefill | LiteRT Prefill | Median Paired Gain |
|---|---:|---:|---:|---:|---:|
| CPU | 128 | 460.96 | 455.13 | 658.30 | -0.7% |
| CPU | 1024 | 420.12 | 418.82 | 522.28 | -0.3% |
| GPU | 128 | 1222.35 | 1375.85 | 2200.86 | +14.0% |
| GPU | 1024 | 1110.92 | 1283.28 | 2618.09 | +15.2% |

Rates are tokens/s. GPU paired gains span 12.4-16.2% (128) and 13.2-22.2%
(1024). After-change process ranges are CPU 387.61-505.24 / 369.34-447.29,
GPU 1357.36-1422.80 / 1062.52-1322.07 for short/long prompts. CPU changes are
within run variability. **Neither CPU nor GPU prefill has reached LiteRT parity.**

Decode medians before/after: CPU 50.22/50.31 and 40.28/40.41; GPU 76.09/75.75
and 66.33/66.63. Every paired decode ratio is above 0.96. All before/after
generated token IDs match for all 60 matched measured requests, and all warm
after-change preparation counters are zero. Runtime prefill/decode counter
boundaries still differ: Kidi includes its first head in prefill, LiteRT does not.

### First-output Latency

A separate 60-process run uses `--decode 0` (one output), six requests per
process: first use is recorded, requests 1-2 discarded, requests 3-5 measured.
LiteRT's timer includes session creation, tokenization, prefill and one decode;
Kidi's includes tokenization, cache creation, prefill and first-token selection.
Python/FFI overhead and detokenization differ, so these are application-side
first-output comparisons, not identical instrumentation or pure kernel latency.

| Device | Input | Before Warm ms | After Warm ms | LiteRT Warm ms | Before/After First-use ms |
|---|---:|---:|---:|---:|---:|
| CPU | 128 | 326.62 | 310.32 | 233.26 | 1152.33 / 1145.97 |
| CPU | 1024 | 2759.05 | 2781.77 | 2188.25 | 3643.60 / 3608.44 |
| GPU | 128 | 114.37 | 99.43 | 68.91 | 3278.97 / 3248.92 |
| GPU | 1024 | 1100.90 | 959.96 | 460.54 | 4640.06 / 4491.19 |

First-use excludes loading and does not flush OS/driver caches. Load-plus-first
medians before/after are CPU 2318.97/1823.24 ms and 4294.24/4256.33 ms;
GPU 4209.31/4357.24 ms and 5869.06/5669.59 ms. Individual trials vary materially;
no cold-start speedup claim follows from these medians. No work was moved out of
the measured phases to manufacture a warm gain.

### Scaling and Rejected Trials

The cache microbenchmark copies 512-wide FP32 rows at offset 128, queues 16
updates, and checks exact values. Actual GPU command timestamps, microseconds:

| Rows | Generic Scatter | Contiguous Copy |
|---:|---:|---:|
| 1 | 6.97 | 3.05 |
| 128 | 597.42 | 1.84 |
| 256 | 2285.96 | 3.09 |
| 512 | 8909.39 | 5.28 |
| 1024 | 34918.50 | 19.56 |

General Metal scatter validates/scans indices per updated byte. Sequential
Gemma writes now avoid that cost; general scatter's semantics are untouched.
The projection benchmark also covers calibrated 128/256/512/1024-row packed
and cached-FP16 paths. The latter remains much faster. These microbenchmarks
are not model throughput, and the fenced operator profile is not GPU timing.

Chunk 256 reached about 1510 tokens/s on one exploratory long-GPU run; CPU
remained near 467. It stays opt-in, not a validated default. A 512-token full-model
trial stalled while system swap reached 12.9 GB and was stopped; full-model
1024 chunks were not attempted. An eight-thread prefill-only CPU projection
trial was inconsistent and removed. Output-round epilogue fusion, executable
sharing, and additional projection fusion were not implemented in this pass.

Process peak-RSS medians before/after: CPU 6.02/6.02 GB short, 6.09/6.09 GB long;
GPU 7.38/6.66 GB short, 7.75/8.25 GB long. These are below the 10% median-growth
budget but are not total GPU physical footprints. The machine had active swap
throughout; sustained-swap/total-device-memory acceptance is **not established**.

### Correctness and Reproduction

All 16 CTest targets pass; RTG's 50-sentence CPU and Metal gates remain chrF2
100. Range-copy tests cover bounds/failure atomicity, overlap, offset views,
aliasing, read-only storage and queued owners. Original and native-QAT fixtures
cover full, incremental and chunked prefill including a 256+3 tail.

An independent official-Transformers QAT reference now covers the 128-token
benchmark prompt using all-position logits, evaluated as multirow prefill:

| Device | Top-1 Agreement | Mean KL | NLL Delta |
|---|---:|---:|---:|
| CPU | 125/128 | 0.0138114 | -0.0069013 |
| GPU | 123/128 | 0.0130452 | -0.0096231 |

This repetitive prompt is a numerical probe, not broad quality evidence; the
synthetic final target also prevents interpreting its NLL as corpus perplexity.
No tolerances were relaxed. Full checkpoint prefill beyond 128 positions against
an independent reference remains unverified.

The full records, process trial/order labels, binary hashes and microbenchmarks
are in [prefill-results.json](prefill-results.json). Historical artifacts are
unchanged. Baseline SHA256 starts `356b50b76620`; measured after SHA256 starts
`d3e705dacaba`. To reproduce, use the command above with `--warmups 2 --runs 3`,
five fresh processes per case, alternating `--binary` paths. For first-output
latency use `--decode 0 --warmups 0 --runs 6` and select records as described.

```sh
build-release/kidi_metal_packed_bench cache
build-release/kidi_metal_packed_bench prefill
.cache/gemma-venv/bin/python tests/gemma4_reference.py .cache/gemma-qat-prefill-reference \
  --checkpoint ../models/gemma-4-E2B-it-qat-mobile-transformers \
  --prompt-tokens .cache/gemma-bench/range-gpu-128.json
build-release/kidi_gemma4_quality ../models/gemma-4-E2B-it-qat-mobile-transformers gpu 0 128 \
  .cache/gemma-qat-prefill-reference/reference.safetensors 128
```

Next work should address calibrated prefill projections and bounded intermediate
memory, then revisit chunk policy. The current CPU bottleneck is not cache-copy
scaling; copying this Metal optimization into CPU tuning will not close its gap.