# Eager Migration

## Outcome

Kidi now has one eager model API. The neutral graph/IR, model compiler, lowerers,
partitioner, recorded loops, graph-generation policy, and separate CPU model
builders were deleted. We retained backend-private prepared operators, resident
INT8 kernels, tensor storage, and numerical fixtures. No lazy fallback remains.

Model code binds weights into reusable layers and executes ordinary tensor
operations. CPU and Metal use the same Transformer. Greedy and beam search use
the generic host decoder; batched greedy is also an ordinary host loop.

This favors simplicity, as requested. It is slower than the earlier optimized
lazy implementation and does not preserve every INT8 generated sentence.

## Performance

Apple M5, release build, 2026-09-19. Three paired fresh-process trials per case;
three untimed warmup requests and 96 measured sentences per trial. Batch-one
repeats the first fixture sentence; batch-32 repeats the entire 32-sentence
fixture three times. IPC, tokenization, and detokenization are timed, while
loading and warmup/preparation are excluded. CPU uses eight threads.

| INT8 workload | Previous tok/s | Eager tok/s | Change |
|---|---:|---:|---:|
| CPU batch 1 | 467.9 | 320.7 | -31.5% |
| Metal batch 1 | 312.7 | 204.1 | -34.8% |
| Metal batch 32 | 803.8 | 729.8 | -9.2% |

These are medians of three process rates. The older 1,157 tok/s figure used a
different 128-sentence window; it is not the baseline for this comparison.
Each paired workload generated identical token counts, and repeated requests
were stable within each process. Each pair ran previous then eager; trial order
was not randomized, so small differences should not be overinterpreted.
No builds, tests, or other Kidi inference ran concurrently with this final set.
An earlier overlapping timing attempt was discarded. CPU eager trials ranged
305-344 tok/s; Metal eager batch-one ranged 200-205, and batch-32 726-736.

The gap is larger than the earlier prepared-operator experiment. That experiment
used a prearranged execution schedule and persistent bindings; this implementation
also performs ordinary tensor dispatch, cache lookup, allocation/pooling, and
host token selection. The new runtime has not received comparable optimization.
These results are not evidence that all eager runtimes must have this gap.

## Numerical Behavior

On the 32-sentence saved corpus:

- CPU FP32: 32/32 exact translations, 718 target tokens.
- CPU BF16: 32/32 exact translations, 718 target tokens.
- CPU INT8: 28/32 exact translations, 731 versus 725 target tokens.
- Metal INT8 batch 32: 31/32 exact translations, 725 tokens in both; the changed
  sentence differs in `center` versus `centre`.
- Two sampled CPU INT8 beam-4 translations match text, but scores differ slightly
  (`-2.44394` to `-2.44979`, `-4.89863` to `-4.90579`).

The CPU INT8 differences are not all spelling: one sentence changes substantially.
Matching CPU precision flags and reproducing the old sentinel/cache layout did
not remove it. Fixed numerical layer tests pass, but there is no claim of exact
INT8 autoregressive equivalence or a broad quality assessment. INT8 rounding and
operator-boundary effects remain a compatibility caveat, not a hidden baseline
update. Existing saved reference outputs were not overwritten.
The optional chrF scorer could not run: its cached `lxml` extension was not
importable under the selected Python interpreter. No quality score is claimed.

All 15 tests pass after migration. The previous layer fixtures remain, including
BF16 arithmetic, attention, embedding/offset behavior, dynamic INT8 quantization,
odd tails, zero and constant rows, long contractions, and repeated shapes.
Eager-specific tests cover retained outputs, functional cache updates, operand
errors, and unavailable devices. Tests now exercise real eager tensors.
CLI checks also passed partial batches and blank-line ordering (identical FP32
text at batch sizes one and four), and a Metal beam-four smoke translation.

## Reproduction

Build the current executable:

```bash
cmake --preset release
cmake --build build-release --target kidi_cli -j8
```

Use the existing request-timing harness:

```bash
python3 benchmarks/metal/profile_rtg.py \
  --binary ./build-release/kidi \
  --model ../models/rtg500eng-tfm9L6L768d-bsz720k-ens05-kidi-canonical-int8 \
  --backend mps --input benchmarks/rtg/work-20260919-000334/input.fr.tok \
  --batch-size 32 --batch-window 32 --warmups 3 --sentences 96 --repetitions 3
```

For batch-one use batch/window 1; for CPU set `--backend ynnpack`. The saved
pre-migration executable used for comparison is `/tmp/kidi-before-eager` in the
development session, not a distributed artifact. Raw trial summaries are in
[eager-migration-timings.jsonl](eager-migration-timings.jsonl).

Further performance work should target individual eager operators, attention
fusion, buffer/dispatch overhead, and cache updates. It must not reintroduce a
second model execution system. See [../../ARCHITECTURE.md](../../ARCHITECTURE.md)
for synchronization, ownership, and preparation contracts.