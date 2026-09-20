# Metal lowering benchmark

Current runtime: [eager migration results](EAGER_MIGRATION.md). The model graph
implementation has been removed. The projection/kernel benchmarks below remain
available, but model throughput results on this page describe the earlier lazy
implementation unless explicitly stated otherwise.

For the eager-versus-graph question, see the
[prepared-operator execution study](EXECUTION_STUDY.md): compiling larger
regions helps FP32 substantially in some cases, but adds less than 1% to the
tested batched INT8 workload. Both variants reuse the same shared equations.

Latest: the batched shared W8A8 path reaches **1,157 aggregate target tokens/s**
on Apple M5 with a warmed 128-sentence input window and batch size 32.
See [the throughput plan](THROUGHPUT_PLAN.md) for three-run results, the exact
arithmetic contract, quality caveats, and reproduction. Earlier measurements
below describe unbatched execution or projection-only experiments.

This benchmark compares the RTG generator projection
`[rows, 768] x [768, 64000] + bias` on:

- Kidi's 8-thread YNNPACK CPU backend.
- A compiled Apple MPSGraph executable on the default Metal device.

The graph, constants, command queue, input buffer, and output buffer are created
once. Warmups and graph compilation are excluded from measured execution. No
host copies occur in the timed region.

Three Metal measurements are reported:

- `metal_sync_median_us`: one invocation followed by a completion wait. This is
  the relevant mode while decoding returns logits to host-side argmax/top-k on
  every token.
- `metal_device_resident_median_us`: several invocations encoded into one
  persistent `MPSCommandBuffer`, followed by one wait. This isolates device
  throughput when host synchronization is amortized.
- `metal_async_trailing_median_us`: separate command buffers queued without
  blocking, preallocated result buffers, completion callbacks, and one trailing
  fence. This models a GPU producer with a CPU observer behind it.

Build and run:

```bash
cmake --preset release -DKIDI_BUILD_BENCHMARKS=ON
cmake --build build-release --target kidi_metal_lowering_bench -j8
./build-release/kidi_metal_lowering_bench \
  [rows] [iterations] [warmups] [inflight] [logits|argmax] [fp32|bf16]
```

## Apple M5 result

Measured 2026-09-19. Each value below is the median of three fresh processes;
each process alternated CPU and Metal samples. Numeric output matched YNNPACK
within the benchmark's `1e-3` maximum-absolute-error gate.

| Rows | YNN sync us | Metal sync us | Sync speedup | YNN sustained us | Metal device-resident us | Resident speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1594.729 | 1871.792 | 0.851x | 1584.711 | 1599.701 | 0.991x |
| 4 | 1704.188 | 1828.146 | 0.932x | 1733.703 | 1591.366 | 1.097x |
| 32 | 3496.625 | 5335.417 | 0.655x | 3841.953 | 5064.984 | 0.757x |
| 64 | 5571.208 | 5769.833 | 0.971x | 6190.359 | 5541.042 | 1.117x |
| 96 | 8059.771 | 6661.458 | 1.211x | 8172.734 | 6482.385 | 1.262x |
| 128 | 10431.125 | 7381.646 | 1.413x | 10498.688 | 7091.818 | 1.480x |

Kidi greedy decoding uses one row and beam-4 uses at most four rows for this
projection. A standalone synchronized Metal offload therefore does not improve
target-token throughput. On the measured FP32 corpus, the generator accounts
for about 1.221 s of 3.712 s across 750 decoder steps. Applying the measured
row-1 ratio to only that phase predicts about 3.924 s, or 183 target tokens/s,
versus the measured CPU rate of 193 target tokens/s.

Metal becomes useful for this operation near 96 rows. An accelerated Kidi path
must therefore lower a larger graph boundary: keep weights, encoder memory, and
K/V caches resident; combine projections, attention, FFN, and generator work in
one compiled graph; and move argmax/top-k and cache updates onto the device so a
host wait is not required for each operation. A backend should be enabled only
after an end-to-end target-token benchmark beats YNNPACK.

## GPU selection and asynchronous decoding

The `argmax` mode keeps all 64,000 logits on the GPU and exposes only one
`int32` token per row. CPU timing includes its equivalent host argmax. With
eight command buffers allowed ahead of the CPU observer and five fresh
processes, every completion callback fired and exact token selection matched
for every shared output buffer:

| Shape | YNNPACK sustained us | Metal async trailing us | Speedup |
|---|---:|---:|---:|
| Greedy row 1 | 1765.479 | 1585.404 | 1.122x |
| Beam rows 4 | 1988.268 | 1627.138 | 1.228x |

This reverses the small-workload result without introducing a timed copy. The
gain comes from keeping selection on-device and removing the per-step blocking
wait, not from unified memory alone.

The observer is notified after every step; eight is queue backpressure, not an
EOS-check interval. It must not become a model tuning parameter. The production design is specified in
[`ASYNC_DECODING.md`](ASYNC_DECODING.md): preferably compile the complete
autoregressive loop with MPSGraph's GPU `while` operation; otherwise use a
serial Metal queue with system backpressure, sticky GPU `finished` state, and a
CPU observer that stops future submission. Already committed steps become
no-ops after EOS.

## BF16 and INT8

Metal support depends on the API and operation:

- MPSGraph executes native BF16 matrix multiplication. Its BF16 logits are cast
  to FP32 before `argmax`, because direct BF16 `argmax` was incorrect on the
  measured SDK.
- MPSGraph supports INT8 quantize/dequantize operations, but rejects a raw
  INT8-by-INT8 matrix multiplication because `matmul` requires floating-point
  or complex operands.
- LiteRT's Metal backend has a specialized quantized fully connected kernel for
  affine INT8 constant weights. This is not generic MPSGraph INT8 execution.

The LiteRT comparison used a fully delegated, bias-free `[rows, 768]` to
`[rows, 64000]` fully connected model with FP32 inputs and outputs. INT8 used
per-output-channel symmetric weights, so this is W8 rather than proven A8W8.
Each timing is the median warm average from five fresh processes with 101
iterations; the first invocation in each process was excluded.

| Rows | MPSGraph BF16 full logits us | LiteRT FP16 us | LiteRT INT8 W8 us | W8 vs BF16 | W8 vs FP16 |
|---:|---:|---:|---:|---:|---:|
| 1 | 977.565 | 1144.155 | 673.946 | 1.451x | 1.698x |
| 4 | 978.977 | 1188.981 | 680.976 | 1.438x | 1.746x |

The BF16 benchmark includes bias and uses MPSGraph, while the FP16/INT8 pair is
bias-free and uses LiteRT. The same-runtime FP16-to-W8 comparison is therefore
the cleaner speed ratio. Enabling LiteRT's dynamic source-quantization option
measured `695.244 us` for row 1 and `695.568 us` for row 4, both slower than W8.
It also produced byte-identical output to W8, so A8W8 activation was not
established and those measurements must not be labeled A8W8.

Synthetic output quality is similar for FP16 and W8:

| Rows | Precision | Relative RMS error vs FP32 | Cosine similarity | FP32 argmax matches |
|---:|---|---:|---:|---:|
| 1 | FP16 | 0.002684860 | 0.999996398 | 0/1 |
| 1 | INT8 W8 | 0.002656390 | 0.999996494 | 0/1 |
| 4 | FP16 | 0.002732293 | 0.999996268 | 0/4 |
| 4 | INT8 W8 | 0.002698387 | 0.999996363 | 1/4 |

The synthetic FP32 top-two margin is only `0.000221`, so the argmax changes do
not establish a model-quality regression. Real-model corpus output and quality
must be measured before enabling W8. This result also applies only to the
generator projection: unsupported general INT8 tensors may be dequantized, and
quantized attention/cache execution is not yet a complete LiteRT path.

[`generate_fc_models.py`](generate_fc_models.py) creates deterministic FP32,
FP16, and per-channel INT8 TFLite models for this comparison. It requires the
generated TFLite FlatBuffers Python bindings and FlatBuffers Python runtime on
`PYTHONPATH`:

```bash
PYTHONPATH=/path/to/tflite-bindings:/path/to/flatbuffers/python \
  python3 benchmarks/metal/generate_fc_models.py \
  --output-dir /tmp/kidi-metal-fc --rows 1 --inputs 768 --outputs 64000
```

## Resident A8W8 implementation

The subsequent 2026-09-19 implementation enables full greedy RTG decoding of
INT8 packages with `--backend mps`. It supersedes the unsupported-INT8 status
in the historical study below. It does **not** yet outperform CPU INT8.

The shared `QUANTIZED_LINEAR` operation now lowers to Metal compute kernels:

- Weights stay resident as signed INT8 in canonical `[input, output]` layout.
- Each activation row uses a zero-inclusive min/max range, a 255-level scale,
  nudged zero point, and reciprocal-multiply rounding to quantize to INT8.
- Dot products use integer accumulation with zero-point correction; per-channel
  scales and bias produce FP32 outputs. Attention/residual/normalization remain
  floating point. Only selected embedding rows are dequantized, not entire tables.
- The lowerer partitions by quantized-op dependencies. MPSGraph floating-point
  regions and integer kernels encode in order into a shared `MPSCommandBuffer`.
  There is one explicit CPU completion fence per sequence, not per linear op;
  MPSGraph may internally commit command buffers. K/V and intermediate buffers
  stay on GPU. Buffer allocation, parameter transfer, and pipeline compilation
  happen during preparation and are reused for the same shape.
- This is ordinary Metal compute with INT8 storage and integer accumulation,
  **not proof of native INT8 matrix instructions or Neural Accelerator use**.

Implementation is confined to backend lowering/runtime plus shared embedding
semantics; no second model or Transformer implementation was introduced.
Quantized parameters must be constants. Input contraction width is limited to
65,536 to keep worst-case signed accumulation within int32, with additional
32-bit indexing bounds checked. Greedy decoding only; source embedding stays
on CPU as in the existing Metal path. Executable instances are mutable and are
not intended for concurrent invocations.

### Matched decoding measurements

Apple M5, release binary, same canonical INT8 package, same 32 source sentences,
eight CPU threads, greedy, 50 extra tokens, length penalty 0.6. Three fresh
processes for each backend, alternating CPU then Metal; reported inference excludes
package loading and explicit graph preparation. These are warm-file/driver-cache
measurements, not cold-start latency.

| Backend | Inference seconds per trial | Median target tok/s | Tokens | Decoder steps | Median graph preparation s |
|---|---|---:|---:|---:|---:|
| YNNPACK INT8 | 1.4863, 1.3870, 1.3769 | 522.69 | 725 | 757 | 1.437 |
| Metal A8W8 | 3.6635, 3.5731, 3.5902 | 202.22 | 726 | 758 | 8.071 |

Outputs were deterministic across the three processes for each backend.
31/32 sentences matched exactly. Metal-to-CPU chrF was 99.8496; reference chrF
was 59.0371 for CPU and 58.9765 for Metal. This small fixture is insufficient to
establish general model quality; retain the CPU default and validate a larger
corpus before deployment. Differences in floating-point reductions and backend
rounding can change autoregressive selections.

A warmed repeated-source measurement gave approximately 187 target tok/s
(64 measured sentences/process, three untimed warmups, three-process median).
An attempted four-channel weight-read vectorization measured 187.7 versus 187.1
tok/s and was discarded as no convincing improvement. A separate sampled run of
256 warmed sentences measured 180.8 tok/s with 88 whole-device GPU samples:
mean busy 85.8%, median 87%, p95 88%, maximum 90%. These counters do not measure
occupancy or Neural Accelerator use. Higher GPU busy time than BF16 did not
mean faster decoding.

### Validation and next steps

The existing INT8 test now covers direct CPU/Metal output comparison, a shared
mixed graph with two quantized linears and floating-point regions, odd dimensions
(7x35 and 35x5), zero/constant/mixed-sign rows, changing row counts, persistent
execution, and quantized embedding gathers. Command-batch completion errors
are propagated before buffers are reused.

The next throughput work is independent-sentence batching and a tiled,
matrix-engine-oriented kernel/provider. The initial shader rereads weights for
each activation row and is not a batched GEMM optimization. Minimize partition
and binding overhead, cache shape buckets, and verify any Metal 4 tensor path
with suitable hardware counters. Do not interpret this implementation as reaching
the advertised peak GPU AI throughput.

```bash
cmake --build build-release --target kidi_cli -j8
./build-release/kidi predict --backend mps --beam-size 1 --threads 8 \
  --model /path/to/canonical-int8 --in /path/to/source.tok --stats --profile
python3 benchmarks/metal/profile_rtg.py \
  --model /path/to/canonical-int8 --backend mps --input /path/to/source.tok \
  --sentences 256 --repetitions 1 --sample-ms 100
```

## INT8 decoding and GPU utilization study

Measured 2026-09-19 on Apple M5 (10 GPU cores), macOS 26.6.2, release build,
eight CPU threads, greedy decoding, maximum 50 extra tokens. This study predates
the resident A8W8 implementation above: full RTG INT8 decoding was CPU-only,
and the Metal lowerer did not implement `QUANTIZED_LINEAR`. Loading quantized weights and
expanding them to floating point would not establish an INT8 GPU speedup.

### End-to-end results

The existing 32-sentence corpus benchmark excludes model loading and explicit
graph compilation/preparation. Counts come from target token IDs, not words.

| Path | Target tokens | Inference seconds | Target tokens/s | Trials |
|---|---:|---:|---:|---|
| YNNPACK INT8 | 725 | 1.3898 | 521.66 | median of 3 fresh processes after 1 discarded trial |
| MPS INT8 | n/a | n/a | n/a | unsupported |
| MPS BF16, fused loop predicate | 718 | 2.8920 | 248.28 | one validation run |

Do not treat the last row as an INT8-to-INT8 comparison. BF16 Metal output
matched all 32 saved BF16 translations and executed 750 decoder steps. The INT8
CPU run executed 757 steps. The Metal run separately reported 15.626 seconds of
graph preparation across changing source shapes, excluded above. Avoid comparing
this single run with the earlier cold Metal result as an optimization ratio:
driver caches and specialization affect first execution.

For a cleaner scheduling comparison, `profile_rtg.py` repeatedly translates the
same first corpus sentence in a persistent process. Three untimed warmups precede
256 measured sentences; results below are medians of three fresh processes.
All paths returned the same eight target tokens for this sentence. The stopwatch
includes tokenization, detokenization, and pipe overhead, but excludes loading,
initial compilation, and warmups. GPU sampling is disabled for these timings.

| Steady-state path | Target tokens/s |
|---|---:|
| YNNPACK INT8 | 485.35 |
| MPS BF16, separate predicate dispatch | 298.34 |
| MPS BF16, fused predicate | 319.30 |

Predicate fusion improved this measurement by 7.0%. It is implemented in the
generic graph control-flow executor: evaluate the predicate on initial state,
then append the predicate on updated state to each body execution's outputs.
This removes a separate GPU invocation and completion wait on every iteration.
Zero-iteration, limit, finished-state, and repeated-run behavior are tested.
The body still synchronizes once per step; it is not an asynchronous native loop.

### Device-wide utilization

IORegistry `AGXAccelerator/PerformanceStatistics` exposes busy percentages
without privileges. The sampler requests a sample every 100 ms (plus query
time). These are system-wide driver counters, not per-process utilization,
shader occupancy, memory bandwidth, or Neural Accelerator occupancy.

Separate instrumented runs used 512 measured repetitions after warmup:

| Path | Samples | Mean GPU busy % | Median % | p95 % | Instrumented tok/s |
|---|---:|---:|---:|---:|---:|
| CPU INT8 control | 69 | 0.0 | 0.0 | 0 | 470.78 |
| MPS BF16 before fusion | 116 | 72.8 | 76.0 | 81 | 281.27 |
| MPS BF16 after fusion | 108 | 64.1 | 66.0 | 73 | 300.70 |

The instrumented runs are slower than uninstrumented runs. Higher busy time is
not the objective: removing unnecessary GPU work improved throughput while
reducing busy time. Other applications can contaminate these counters.
Detailed Instruments/Metal counter profiling was unavailable because only
Command Line Tools were selected and `xcrun --find xctrace` failed. No claim
about matrix-engine utilization can be made from these counters alone.

### Metal W8 batch scaling

The official LiteRT Metal plugin fully delegated the synthetic, bias-free
`[rows,768] x [768,64000]` projection with per-channel INT8 weights, float I/O,
and FP16 precision selected. Three fresh processes, 101 invocations per process,
first invocation excluded from each mean; the table uses median warm means.
This is W8 storage, not verified A8W8 arithmetic or Neural Accelerator execution.

| Rows | Warm microseconds | Projection rows/s | Dense-equivalent TOPS |
|---:|---:|---:|---:|
| 1 | 634.468 | 1576.1 | 0.1549 |
| 4 | 654.808 | 6108.7 | 0.6005 |
| 16 | 699.963 | 22858.3 | 2.2471 |
| 64 | 2067.342 | 30957.6 | 3.0433 |
| 256 | 7476.009 | 34242.9 | 3.3662 |

Projection rows/s are **not decoder tokens/s**. Dense-equivalent TOPS are
`2 * rows * 768 * 64000 / seconds / 1e12`; this is a workload count, not a
hardware instruction counter. A separate long-running sampled invocation loop,
starting after input-buffer setup and discarding the first second of samples,
reported median GPU busy readings of 94%, 90%, and 99% for rows 1, 16, and 256
respectively (89, 65, and 81 samples). High busy time did not imply high TOPS.

For context, the real CPU INT8 corpus profile spent 0.309 seconds in the
generator across 757 steps, about 408 microseconds/step. This is not an exact
cross-runtime microbenchmark, but the 634-microsecond synthetic W8 result does
not justify offloading the generator alone.

### Interpreting peak compute

[Apple's M5 announcement](https://www.apple.com/newsroom/2025/10/apple-unleashes-m5-the-next-big-leap-in-ai-performance-for-apple-silicon/)
describes GPU Neural Accelerators, direct access through Metal 4 Tensor APIs,
and 153 GB/s unified-memory bandwidth. It does not specify the quoted 70 TOPS
number or its precision/sparsity conditions. Neither our busy counters nor the
W8 benchmark establish access to that peak.

At batch one, the generator streams approximately 49.15 MB of W8 weights for
98.30 million dense operations. Ignoring caches, scales, activations, and output
traffic, arithmetic intensity is only 2 operations per weight byte. At 153 GB/s,
that gives a streaming roofline of approximately 0.306 TOPS, not 70 TOPS.
Batching `B` independent rows raises this idealized intensity to `2B`; even the
weight-only estimate would need roughly 229 rows to feed 70 TOPS at 153 GB/s.
Real output/cache traffic makes the requirement stricter. Beam four is not a
substitute for a substantial batch of independent sentences.

The current six-layer decoder plus generator has about 89.26 million matrix
weights read per step (excluding preprojected source K/V). Pure W8 weight
streaming alone would take at least about 0.583 ms at 153 GB/s. This is an ideal
bandwidth estimate, not a promised token rate; attention, normalization, cache
traffic, quantization, selection, and dispatch add work.

### Next optimization order

1. Lower shared `QUANTIZED_LINEAR` and quantized embedding ops to a real resident
  W8/A8W8 provider. Keep quantized weights resident; do not silently label a
  floating-point expansion as INT8 acceleration. Verify output quality and
  actual kernel selection. The known LiteRT FC kernel is a candidate, not yet
  a full-decoder backend.
2. Add independent-sentence batching and length buckets in shared model/search
  code. The projection sweep supports batching, but full-decoder speedup must
  still be measured. Keep top-k/argmax and K/V updates on-device.
3. Evaluate Metal 4 tensor kernels with appropriate counters. The installed
  SDK's matrix-multiplication and quantization headers expose quantize/dequantize
  but no dedicated INT8 matrix-multiply declaration. Confirm Neural Accelerator
  use rather than inferring it from weight dtype or GPU busy time; compiler
  fusion and newer APIs require separate verification.
4. Cache prepared executables by bounded shape buckets, reuse bindings and
  buffers, and reduce remaining host fences. Preserve a single shared graph
  and put scheduling/kernel choices in backend lowering.

Reproduce steady-state timing and sampling (run them separately):

```bash
python3 benchmarks/metal/profile_rtg.py \
  --model /path/to/canonical-int8 --backend ynnpack \
  --input /path/to/input.fr.tok --sentences 256 --repetitions 3
python3 benchmarks/metal/profile_rtg.py \
  --model /path/to/canonical-bf16 --backend mps \
  --input /path/to/input.fr.tok --sentences 256 --repetitions 3
python3 benchmarks/metal/profile_rtg.py \
  --model /path/to/canonical-bf16 --backend mps \
  --input /path/to/input.fr.tok --sentences 512 --repetitions 1 --sample-ms 100
```

For the projection sweep, use `generate_fc_models.py --rows N` at
`N=1,4,16,64,256`, then the LiteRT example with `--accelerator=gpu`,
`--require_registered_accelerator=gpu`, `--auto_register_accelerators=gpu`,
`--gpu_backend=metal`, `--gpu_precision=fp16`, `--gpu_buffer_storage=buffer`,
and `--iterations=101`. Keep `KIDI_LITERT_A8W8` unset in the temporary runner.
The model generator is versioned; the official runtime/plugin and example
runner remain external dependencies, not part of Kidi's production backend.

## Official LiteRT sanity check

The official prebuilt `libLiteRtMetalAccelerator.dylib` was loaded through
LiteRT's source-built C++ API example. Its MobileNet test model reported
`Compiled model fully accelerated: yes`. Across three fresh processes with 101
iterations each, excluding the first invocation from each average:

| Backend | Warm average us | Speedup vs CPU |
|---|---:|---:|
| LiteRT CPU, 8 threads | 1911.129 | 1.000x |
| LiteRT Metal FP32 | 1171.194 | 1.632x |
| LiteRT Metal FP16 | 705.872 | 2.707x |

All 1,001 outputs were compared. Metal FP32 had maximum absolute error `2e-6`;
Metal FP16 had maximum absolute error `0.003043`.

This confirms that LiteRT Metal lowering works and can accelerate a fully
resident, fully delegated graph. It does not establish a Kidi token-rate gain:
Kidi's current decode boundary is much smaller and synchronizes with host search
every token. The local LiteRT checkout also predates the upstream
`sdpa_transposed` Flash-Decode path; current upstream selects Flash-Decode only
for head dimension 128, while this RTG model uses head dimension 64.
