# CPU Iteration Harness

The driver reuses `kidi_android_baseline` on desktop CPU. Each thread-count
configuration loads the model once, runs explicit warmups, then measures
repeated independent requests. It uses the same fixed chat prompt as the
CPU slowdown reproduction by default. Existing Android/native baseline
defaults are unchanged.

From the repository root, build and record a baseline:

```bash
python benchmarks/cpu/run.py --build --threads 8 \
  --output .cache/benchmarks/cpu/baseline.json
```

The default model is `@google/gemma-4-E4B-it-qat-mobile-transformers`, resolved
through the existing installed `kidi[hf]` package. Pass `--model /local/model`
for offline operation without Python package dependencies. The driver itself
uses the standard library. The default workload generates 64 tokens, ignores
EOS to keep the work fixed, and runs one warmup plus three measured requests.
It uses a 16,384-token context budget, 128-token prefill chunks, and unpacked
prefill, matching the chat defaults relevant to this investigation.

After changing native code, incrementally rebuild and compare:

```bash
python benchmarks/cpu/run.py --build --threads 8 \
  --baseline .cache/benchmarks/cpu/baseline.json \
  --output .cache/benchmarks/cpu/candidate.json
```

Use the same model and protocol for comparisons. The harness requires exact
input/output token IDs within repetitions and against the baseline. A failed
or mismatched run exits nonzero and preserves its report and raw logs. It
rejects prompt KV reuse so repeated requests still measure real prefill. It
will not overwrite the baseline file. Timing summaries exclude warmups;
cold first-token time and model-load time remain in the report. The table
shows median warm decode speed, its observed range, warm first-token latency,
peak process RSS, and baseline speedup when comparing. Peak RSS includes
model loading and warmup and is not an incremental allocation measure.

Thread sweep, with configurations run sequentially:

```bash
python benchmarks/cpu/run.py --threads 4 8 16 --tokens 128 --repeats 5 \
  --output .cache/benchmarks/cpu/threads.json
```

For operator-level diagnostics, disable replay so individual operators remain
visible in the profile:

```bash
python benchmarks/cpu/run.py --threads 8 --profile-ops --replay off \
  --output .cache/benchmarks/cpu/operators.json
```

Operator profiles are written to the report's per-configuration stderr logs.
The report also aggregates them by phase/operator, sorted by execution time;
the five largest groups are printed after each configuration.
Profiling changes timing, so profiled and unprofiled reports cannot be compared.
The driver clears inherited replay/profiling overrides and forces CPU placement.
Other `KIDI_*` environment values are recorded. With `--workload both`, the
long prompt repeats the supplied prompt 16 times; operator profiling skips
only the initial warmup requests, not later workload transitions.

Reports record the model path, native binary and harness hashes, Git revisions,
working-diff hash, selected CMake flags, CPU affinity, load average, native
architecture names and raw flags, and every native JSON record. Compare runs
under similar host load; the harness does not pin cores. On multi-socket Linux
hosts the benchmark binary confines itself to one NUMA node, as the CLI does
(`KIDI_NUMA=off` disables it), and records the node as `numa_node` in its runtime
record. The node is the one the process happens to start on, and the nodes of a
shared host can differ noticeably in speed, so compare runs that report the same
`numa_node`, or pin both with `taskset`. No timing threshold is part of the unit tests.

Use `--binary /path/to/kidi_android_baseline` to test an existing binary without
building. `--build-dir` selects another CMake build tree; `--build` preserves
its backend choices while enabling benchmarks and Release mode. The per-process
timeout defaults to 900 seconds and preserves partial output on timeout.