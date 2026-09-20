# Metal W8A8 throughput plan

## Target and constraints

First acceptance gate: at least **1,000 generated target tokens/s aggregate**
for the canonical RTG INT8 model on Apple M5, using actual independent source
sentences, measured after loading/compilation/warmup. Report batch latency,
quality drift, memory use, and cold preparation separately. This is not a
single-sentence latency target or a projection-only benchmark.

Keep one neutral tensor graph and shared ops/layers/model. Weights and quantized
activations remain INT8 in device memory; no persistent floating-point matrix
expansion. Backend scheduling and kernel selection belong in lowering. Retain
CPU as default and Metal batching as opt-in.

## Execution plan and outcome

1. **Complete: independent batching.** Shared encoder and cross-attention accept
   source padding masks. Every decoder row has independent EOS, length, score,
   and maximum-step state. Inactive tail rows start with zero permitted steps.
   A completed row emits PAD and stops contributing score/token counts.
2. **Complete: bounded length grouping.** `--batch-size` controls device rows;
   `--batch-window` bounds buffered input. Tokenize, sort by source-token length,
   execute batches, and restore original output order. Preserve blank lines.
3. **Complete: matrix tiles.** A scalar eight-row weight-reuse shader plateaued
   around 275 tok/s and was removed. A 32x32 SIMD-group matrix tile reached
   about 804 tok/s on the 32-sentence mixed corpus without bucket caching.
4. **Complete: prepared-shape reuse.** The quantized lowerer retains four
   prepared shapes (current plus three cached) and shares resident INT8 kernel
   weights across them. Repeated four-bucket windows no longer recompile.
5. **Complete: acceptance measurement.** Three-process median of 1,156.91
   target tok/s on 128 real WMT14 French-English sentences, versus 477.25 CPU.
   Exact output and chrF checks below limit the claim: not bit-identical quality.

## Tiled arithmetic

The tiled path uses INT8 resident operands and dynamically quantized A8 rows.
Only small threadgroup tiles are converted to exactly representable half integer
values for SIMD-group matrix operations. Zero-point-corrected activations are
in [-255,255], weights in [-128,127]. Each product is an integer with magnitude
at most 32,640. The FP32 matrix accumulator is reset every 256 contraction
elements, bounding the sum of absolute products to 8,355,840, below 2^24.
These partial integer sums are converted to INT32 and accumulated across chunks;
FP32 per-row/per-channel scales and bias are applied at the end.

This is **W8A8 storage/quantization with exact bounded integer partial sums using
floating-point SIMD-group hardware**, not native INT8 matrix instructions or
verified M5 Neural Accelerator use. Width is bounded to 65,536 for INT32 safety.
Rows below four now use packed integer GEMV (see the batch-one update below). Tests cover negative
INT8 extremes, 1,025-element contractions, and row/column tails.

## Measured result

Apple M5, macOS 26.6.2, release build, eight CPU threads, greedy decoding,
50 extra tokens. First 128 WMT14/full fr-en sources, tokenized by the existing
benchmark runner. Batch size 32, length window 128. Each fresh process runs two
untimed windows, followed by three measured windows (384 sentence translations).
The stopwatch includes tokenization, detokenization, sorting, host dispatch,
and pipe overhead. Same-shape preparation is warmed; no generated tokens from
warmup, EOS, BOS, PAD, or inactive rows are counted.

| Backend | Trial target tok/s | Median tok/s | Measured target tokens | Median 128-sentence window latency |
|---|---|---:|---:|---:|
| YNNPACK INT8, serial execution | 472.41, 477.25, 478.96 | 477.25 | 9381 | 6.55 s |
| Metal W8A8, batched/tiled | 1179.40, 1156.91, 1117.88 | 1156.91 | 9348 | 2.69 s |

This is a **2.42x aggregate throughput improvement** over the measured CPU
path, not over an optimized batched CPU implementation. The CPU CLI accepts the
same window but currently executes its sentences serially. It is not thousands
of tokens/s for an individual autoregressive sequence. Buffered windows add
request waiting time; use batch/window one for interactive single requests.

Outputs were deterministic in each of the three processes. CPU and Metal matched
120/128 sentences; Metal-to-CPU chrF was 98.9705. Reference chrF was 63.1461 CPU
versus 62.8701 Metal, a 0.276-point difference on this small sample. Do not infer
deployment-wide quality from this fixture. Batch-one vs batch-four CLI checks
preserved text/ordering with normalized-score differences up to 0.0516.

A separate sampled run measured 1,136.40 tok/s, with 62 system-wide GPU samples:
mean busy 92.8%, median 97%, p95 98%. These are driver busy counters, not tensor
core occupancy. `/usr/bin/time -l` around that harness reported peak resident
set size 953,499,648 bytes (about 0.95 GB); this is not total GPU allocation.

Larger unsorted batches were worse: batch 64 measured 834 tok/s and batch 128
736 tok/s in exploratory runs. Increasing batch size alone is not the solution.
Shape grouping, bounded executable reuse, and matrix tiling were all necessary.

## Batch-one latency update

Following the throughput work, batch-one execution was optimized without
changing the shared graph, quantization equations, or decoder policy:

- Pack an additional INT8 weight view as `[output, padded_input]` during the
  first small-row preparation. Contiguous four-byte contraction reads and SIMD
  reduction replace the previous strided per-channel reads. Rows 1..3 use this
  path; larger batches keep the tiled matrix kernel. Odd input/output extents
  and row-count transitions are covered by the existing INT8 test.
- Cache two sets of MPSGraph tensor-data bindings per executable, keyed by
  buffer identity, offset, dtype, and shape. The cache owns its tensor references
  and handles alternating state buffers. Reuse stage argument vectors as well.
- Quantization and matrix dispatch share one compute encoder with an explicit
  buffer memory barrier. This removes one encoder boundary per linear op.
- Check tensor strides directly instead of allocating a temporary vector on
  every contiguity check in binding validation.

These are capture-style fixed-resource optimizations, **not full CUDA graph
capture/replay**. Command buffers are still encoded each step; callbacks retain
error reporting, and the loop still uses one completion fence per step. K/V
state remains in two alternating buffers: blindly aliasing scatter input/output
could overwrite data used by attention. In-place cache updates require a proven
dependency/lifetime transformation and are not claimed here.

Packed weights cost an additional INT8 copy for every linear that enters the
small-row path, rounded to a multiple of four input elements per output channel.
Packing is one-time preparation and excluded from warmed throughput. The
canonical INT8 layout is retained for the batched path; there is no FP32 weight
expansion.

The new production-kernel benchmark isolates linear dispatch from model/search:

```bash
cmake -S . -B build-release -DKIDI_BUILD_BENCHMARKS=ON
cmake --build build-release --target kidi_metal_int8_bench -j8
./build-release/kidi_metal_int8_bench 1
```

It warms each real layer shape, checks deterministic numerical outputs, then
reports host-observed waited latency and per-invocation latency for 16 encodes
with one trailing fence. The latter is not a GPU hardware timer or decoder
token rate. Exploratory generator `[1,768] x [768,64000]` measurements fell from
about 543 to 424 microseconds queued after packing. Clock/caching effects make
the full-model paired comparison the stronger evidence.

Same-session alternating baseline/current/CPU measurements used three fresh
processes per path, three untimed single-sentence warmups, and 128 measured
repetitions of the first corpus sentence (1024 actual target IDs per trial):

| Path | Trial tok/s | Median tok/s |
|---|---|---:|
| Saved Metal baseline | 200.79, 203.35, 199.92 | 200.79 |
| Packed Metal + binding/encoder reuse | 280.00, 279.81, 299.36 | 280.00 |
| CPU INT8 | 482.22, 459.79, 472.97 | 472.97 |

This paired check establishes a 39.4% batch-one improvement with identical
outputs. After the allocation-free stride check, a final three-process run
measured **285.67, 292.69, 297.92 tok/s (median 292.69)**. Treat this as roughly
280--300 tok/s, not an invariant 46% uplift; it was not interleaved with the
baseline again. CPU batch one is still faster.

An attempted fused quantization-plus-small-GEMV shader measured 286 tok/s versus
299 for the separate dispatches sharing an encoder and was discarded. Repeating
activation range/quantization work per output group outweighed the saved dispatch.
Binding reuse alone measured about 202 tok/s and is not claimed as a major win.

On 128 distinct corpus sentences, saved/current Metal batch-one translations
were byte-identical (3122 target IDs); one full-corpus pass improved from
16.243 s to 11.282 s of reported inference (192.21 to 276.73 tok/s), excluding
explicit loading/preparation. A batch-32 regression run retained exactly the
previous batched translations and measured 1217.77 tok/s; that is a single
regression run, not a new three-run throughput claim. None of these changes
eliminate the preexisting Metal-versus-CPU numerical differences documented above.

## Reproduce

```bash
cmake --build build-release --target kidi_cli -j8
python3 benchmarks/rtg/run.py --precisions int8 --num 128 \
  --warmup-runs 0 --repetitions 1 --workdir /tmp/kidi-batch-study-128 \
  --outdir /tmp/kidi-batch-study-reports
python3 benchmarks/metal/profile_rtg.py \
  --model ../models/rtg500eng-tfm9L6L768d-bsz720k-ens05-kidi-canonical-int8 \
  --backend mps --input /tmp/kidi-batch-study-128/input.fr.tok \
  --batch-size 32 --batch-window 128 --sentences 384 --warmups 2 --repetitions 3
```

Repeat with `--backend ynnpack` for the matched CPU protocol. To sample GPU
busy time, use a separate `--repetitions 1 --sample-ms 100` run. Do not mix
sampled timing with the uninstrumented median.

Use the actual CLI for an input file:

```bash
./build-release/kidi predict --backend mps --beam-size 1 \
  --batch-size 32 --batch-window 128 --threads 8 \
  --model /path/to/canonical-int8 --in source.tok --out translated.tok \
  --stats --profile
```

For heterogeneous corpora exceeding four recurring shapes, compilation may recur
and cold wall time will be higher than the warmed measurement. Keep explicit
preparation timing separate; this four-shape cache is a bounded baseline, not
a full serving scheduler.

## Next gates toward multiple thousands

- Bound padding more precisely with dynamic active-row compaction and target
  cache-length buckets; retain independent stop and output-order semantics.
- Fuse normalization/quantization and decoder projections where profiling shows
  benefit, and cache binding metadata to reduce per-step host encoding work.
- Evaluate Metal 4 tensor/Neural Accelerator kernels with real hardware counters;
  do not equate INT8 weights or GPU busy time with native INT8 matrix throughput.
- Validate larger, unseen multilingual corpora and a latency/memory budget before
  changing defaults. The measured 1k gate is reached; 2k+ and single-sequence
  thousand-token throughput have **not** been demonstrated.