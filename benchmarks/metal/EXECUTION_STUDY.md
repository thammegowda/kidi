# Prepared Operators vs Compiled Regions

Historical experiment, superseded by the eager-only migration. The comparison
target and model graph/lowering APIs below were removed with the lazy runtime;
the raw measurements are retained. See [EAGER_MIGRATION.md](EAGER_MIGRATION.md)
for the current implementation and its measured costs. Reproduction commands
below apply to the pre-migration implementation only.

## Decision

Keep compilation available, but do not make graph-building machinery the model
authoring API. Larger regions help some workloads; they are not the source of
all our performance. This experiment does not justify replacing the existing
runtime or building a second set of model equations.

On Apple M5, region compilation gave about **5-10% higher step rate for batch-one
INT8**, **less than 1% for batched INT8**, and **8-31% for FP32**. The next useful
design change is a readable, bound-layer API backed by the existing compiler.
Implementing a full eager backend should wait for a concrete need, not a claim
that either execution mode is universally faster.

## Experiment

`execution_bench.cpp` builds the same shared `TransformerLayers::decode_layer`
equations with real canonical checkpoint weights. It runs either one block or
all six decoder blocks plus final normalization and the 64,000-way generator.
Inputs are deterministic synthetic embeddings, self K/V caches, and projected
encoder K/V tensors. Self and source context lengths are equal; the current
position is the last cache slot. Every invocation starts from the same inputs.
This is a decoder neural-body benchmark, not generated translation throughput:
embedding lookup, source encoding, token selection, and EOS control are excluded.

Both variants use the same Metal partition executor, resident input/output and
intermediate buffers, custom INT8 kernels, MPSGraph floating-point execution,
and one final host completion fence per invocation. No timed tensor transfers,
weight packing, graph compilation, or tensor-buffer allocations are required.
Host command encoding and normal runtime bookkeeping remain timed.

- **Region:** default partitioning, with floating-point regions separated only
  where custom INT8 projections require them. FP32 compiles as one region.
- **Operator:** materialize at normalization, linear-with-bias, attention
  output, GELU, residual-add, and K/V-update boundaries. Each operation is
  prepared once; normalization and attention can still optimize internally.
  Independent operations may share a stage. Boundaries are selected from the
  recorded shared-layer output names, not a duplicate implementation.

This is a prepared coarse-operator proxy for efficient eager execution, not a
libtorch benchmark. It intentionally does not pay per-call graph compilation
or per-operation host waits. It also does not measure the dynamic dispatch,
shape checking, or general allocation behavior of a complete eager frontend.

## Results

Measured 2026-09-19 on Apple M5, release build. Three fresh processes per case;
10 untimed warmup pairs and 80 measured pairs per process, alternating which
variant runs first. Ratios below are the median of the three paired process
ratios (`operator_us / region_us`). A ratio greater than one favors regions.

| Precision | Batch | Context 16 | Context 128 |
|---|---:|---:|---:|
| INT8 | 1 | 1.05x | 1.10x |
| INT8 | 32 | 1.006x | 1.004x |
| FP32 | 1 | 1.12x | 1.08x |
| FP32 | 32 | 1.11x | 1.31x |

The sub-1% differences should be treated as practically tied. Batch-one ratios
varied across processes: INT8 context 128 ranged 1.05-1.11x, and FP32 context 16
ranged 1.06-1.14x. Batched FP32 context 128 was more consistent at 1.30-1.32x.
These are measurements on one machine and fixed synthetic activation states,
not confidence intervals or claims about all inference workloads.

All 24 full-body comparisons passed elementwise checks of logits and updated
caches: `abs(error) <= 0.001 + 0.001 * abs(reference)`, with finite values
required. A final largest-case rerun reported exact INT8 equality and FP32
maximum absolute error `2.098e-05`.

Raw repeated measurements are in [execution-study-20260919.txt](execution-study-20260919.txt).
The original run labels precision/trial on the preceding `case` line; the
final executable also includes precision directly in each result line.

## What Changed

For full INT8 decoding, both modes invoke **37 identical custom quantized
projections**. Region compilation reduces floating-point stages from **67 to
37**, not to one fully fused decoder. This explains why our existing fused
projection kernels retain much of their performance in operator mode.

For FP32, the comparison is **104 operator stages versus one compiled region**.
These are executable stages, not measured hardware kernel-launch counts. The
largest case reduced median host submission from about 2.43 ms to 0.93 ms;
wall latency also improved. We have not isolated exactly which compiler fusion
or scheduling decisions account for the remaining benefit.

Preparation is separate from inference. In these runs INT8 preparation ranged
roughly 120-228 ms for regions versus 169-299 ms for operators; FP32 roughly
137-328 ms versus 298-379 ms. Region preparation always runs first, so these
numbers are not an order-controlled cold-compilation comparison.

`buffer_bytes` counts explicit stage outputs, not total or peak GPU memory. It
excludes weights, inputs, kernel scratch, and MPSGraph internal intermediates;
do not infer that whole-region FP32 halves actual memory use from this counter.
`submission_us` measures host encoding; `completion_us` measures the trailing
commit/wait. GPU execution can overlap encoding, so completion time is not GPU
execution time. Hardware timestamps and kernel-launch counts were not measured.

## Readable Model Direction

Illustrative API sketch, not a newly implemented frontend:

```cpp
auto hidden = input + self_attention(self_norm(input), state.self, self_mask);
hidden = hidden + cross_attention(cross_norm(hidden), state.source, source_mask);
return hidden + feed_forward(feed_forward_norm(hidden));
```

Layer construction binds checkpoint tensors, dimensions, and precision once.
The model expresses equations; the execution context chooses recording or
dispatch. Start by supporting this style through the current graph backend.
Concrete eager tensors, symbolic values, and cache mutation still need explicit
contracts; readable syntax alone is not an eager implementation.

## Reproduction

From the Kidi repository root:

```bash
cmake --preset release -DKIDI_BUILD_BENCHMARKS=ON
cmake --build build-release --target kidi_metal_execution_bench -j8
./build-release/kidi_metal_execution_bench \
  ../models/rtg500eng-tfm9L6L768d-bsz720k-ens05-kidi-canonical-int8 \
  1 16 decoder 80
```

Arguments after the package are batch, context, `block|decoder`, and iterations.
Repeat with batches 1/32, contexts 16/128, and the canonical FP32 package without
the `-int8` suffix. Use three fresh processes per configuration. The optional
benchmark target is Apple-only. Production lowering defaults are unchanged;
`runtime::mps::lower_partitioned` exposes boundaries and optional metrics for
this experiment. No tensor-wide eager dispatcher or new model API was added.