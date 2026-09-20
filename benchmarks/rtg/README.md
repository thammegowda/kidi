# RTG precision benchmark

This benchmark compares kidi FP32, BF16, and per-channel INT8 packages with the
same RTG model and tokenized input. It follows the CPU protocol used by
`tahoma/benchmarks/ssru-marian-vs/run_cpu.py`:

- one fresh kidi process per trial;
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
by `kidi predict --stats`. It excludes BOS, EOS, and PAD tokens and does not
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

For a PyTorch/RTG CPU comparison using the exported model's bundled RTG code,
run `pytorch_baseline.py` with the model's matching Python environment. It
reports exact hypothesis IDs after removing BOS, EOS, and PAD. Set
`--batch-sentences 1` to match kidi's line-at-a-time CLI or increase it to
measure RTG's batched throughput. Its throughput timer starts only after the
PyTorch model is loaded; model-load and process wall time are separate fields.