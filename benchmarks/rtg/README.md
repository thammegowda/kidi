# RTG precision benchmark

This benchmark compares kidi FP32, BF16, and per-channel INT8 packages with the
same RTG model and tokenized input. It follows the CPU protocol used by
`tahoma/benchmarks/ssru-marian-vs/run_cpu.py`:

- one fresh kidi process per trial;
- explicit YNNPACK CPU execution (`--backend ynnpack`), independent of automatic backend selection;
- eight total CPU threads by default;
- warm-file-cache discarded warmups;
- round-robin measured repetitions;
- equal input and output line counts;
- median and p95 inference time, with model-load and wall time reported separately;
- lines/s and exact generated target tokens/s;
- exact matches and chrF against FP32;
- optional chrF against aligned references;
- retained commands, outputs, logs, CSV, and Markdown report.

The default fixture is the first 32 lines of SacreBLEU's WMT14 French-English
test set. On first use, the runner installs pinned `sacrebleu==2.6.0` into
`benchmarks/rtg/.cache/python` with Python 3.9-compatible
`portalocker==3.2.0`, downloads the test set into
`benchmarks/rtg/.cache/datasets`, and applies SacreBLEU's Moses-compatible 13a
tokenizer to the source. Nothing is installed globally. The default decode is
greedy (`--beam-size 1`) so the benchmark isolates precision effects rather
than beam-search branching.

Target-token throughput comes directly from `Translation::token_ids` reported
by `kidi translate --stats`. It excludes BOS, EOS, and PAD tokens and does not
infer token counts from decoded whitespace. Throughput uses aggregate
`Translator::translate` time from `--profile`, excluding package loading and
graph compilation.

```bash
cmake --preset release
cmake --build --preset release --target kidi_cli -j
python3 benchmarks/rtg/run.py --warmup-runs 1 --repetitions 3
```

Use the full test set for a longer run:

```bash
python3 benchmarks/rtg/run.py --num 0 --repetitions 3
```

Use another SacreBLEU set or language pair with `--test-set` and `--lang-pair`.
`--input` remains available for an already-tokenized local override. Run
`python3 benchmarks/rtg/run.py --help` for model, corpus, cache, decode,
thread-count, timeout, and artifact path overrides. `--threads N` sets total
YNNPACK threads including the caller and records the choice in the report; the
default is 8.

Model packages must use the current nested `model`/`decode` manifest. Older local
exports with `model_type`/`architecture` at the top level are incompatible. Regenerate
them with `python -m kidi.converters.rtg`, or pass current packages through
`--fp32-model`, `--bf16-model`, and `--int8-model`. Older BF16 exports may also have
an incompatible generator-weight layout; changing YAML alone does not fix that.

For a PyTorch/RTG CPU comparison using the exported model's bundled RTG code,
run `pytorch_baseline.py` with the model's matching Python environment. It
reports exact hypothesis IDs after removing BOS, EOS, and PAD. Set
`--batch-sentences 1` to match kidi's line-at-a-time CLI or increase it to
measure RTG's batched throughput. Its throughput timer starts only after the
PyTorch model is loaded; model-load and process wall time are separate fields.

## 2026-09-21 Baseline

Apple M5, eight YNNPACK threads, 32 WMT14 French-English sentences, greedy
decoding, one warmup and three measured fresh-process trials per precision:

| Precision | Median inference s | Target tok/s | Different sentences vs FP32 | chrF2 vs FP32 |
|---|---:|---:|---:|---:|
| FP32 | 4.0744 | 176.22 | 0 | 100.0000 |
| BF16 | 2.6030 | 275.84 | 0 | 100.0000 |
| INT8 | 1.5805 | 462.50 | 5 | 98.6110 |

FP32 and INT8 used existing canonical exports through temporary current-format
manifests; BF16 used the validated `rtg-500eng-v1` Hub package. Original weights
were not modified. These are precision comparisons, not before/after optimization
results. INT8 generated 731 target tokens versus 718 for FP32/BF16.

## Reusing Gemma Optimizations

RTG already benefits from the shared Metal output pool, bounded prepared-operator
cache, CPU arena, direct pointwise Metal kernels, and persistent model state.
It also already fuses residual addition with LayerNorm. Its CPU scatter updates
only the selected cache rows; it does not copy the entire cache each token.

The remaining candidates are:

- **Contiguous KV writes:** RTG appends one row at a known position, but still
	uses indexed scatter. Gemma's `Context::copy_slice_` could remove index-tensor
	handling and use Metal blits. The existing 512-wide cache microbenchmark measured
	6.13 us for one-row scatter versus 2.68 us for slice-copy. RTG is 768-wide;
	this is a lead, not an end-to-end RTG speedup. CPU's existing row-copy scatter
	leaves much less room for improvement.
- **Device-side greedy selection:** select from RTG's 64,000 logits before the
	existing completion fence, reusing `Context::greedy_token` and preselected-token
	decoder support. Keep full logits for beam search and scored decoding, and
	preserve tie, nonfinite-value, EOS and padding behavior. Host search was only
	about 1-3% of the CPU baseline, so large gains should not be assumed.
- **Reduce Metal dispatch work:** a four-repeat BF16 sentence profile showed
	substantial host time in attention, LayerNorm/residual normalization, casts,
	and slices. Decoder BF16 linear calls also issued 36 casts per token. Direct
	kernels or folding casts into projections may help, but must preserve BF16
	rounding and LayerNorm/GELU numerics. The CPU profile instead spent about 72%
	of measured operator time in decoder/output linear projections.

These profiles include initialization and use one repeated sentence. Metal host
timings measure dispatch, not GPU execution; per-operator synchronized timings
include extra fences and must not be treated as normal end-to-end performance.
Any proposed change needs paired unprofiled trials and output-parity checks.

Gemma's shared-KV prefill layer skipping does not transfer: RTG's encoder output
feeds every decoder layer, and its decoder layers have independent caches. RTG
uses full attention rather than Gemma's sliding local window, LayerNorm and
sinusoidal positions rather than RMSNorm/RoPE, and an ungated GELU feed-forward
block. Gemma's fused RMS/RoPE and gated-GELU kernels are therefore not substitutes.
Its trained Q2/Q4/Q8 QAT scales and cached FP16 expansion path also cannot be
applied unchanged to RTG's dynamic per-channel INT8 format.

### Quick Transfer Checks

Two small implementation probes were tested independently against a saved baseline,
using BF16, eight CPU threads, batch one, and the first WMT14 sentence. Each backend
used two alternating fresh-process pairs, two warmup requests, and 16 measured
requests per process through `profile_rtg.py`. Timing excludes warmups and includes
IPC and text processing; this is a quick screen, not a statistically strong result.

| Probe | Backend | Baseline tok/s | Candidate tok/s | Change |
|---|---|---:|---:|---:|
| KV slice-copy | CPU | 265.1 | 272.5 | +2.8% |
| KV slice-copy | Metal | 288.7 | 290.5 | +0.6% |
| Device greedy selection | Metal | 286.4 | 283.7 | -0.9% |

Cache-copy passed the existing Transformer and FP32/BF16/INT8 precision tests;
timed translations matched the baseline. GPU selection matched baseline output
on eight varied sentences for batch one, batch four, scored decoding, and beam two.
Neither probe showed a compelling speed gain, so both were removed. The runtime
remains unchanged; direct LayerNorm/cast work remains untested rather than being
expanded into a larger optimization effort.