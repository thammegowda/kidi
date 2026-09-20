# kidi

Kidi is a small C++23 inference toolkit. The first milestone runs exported RTG
Transformer NMT models directly with YNNPACK on CPU and memory-mapped
Safetensors weights.

Repository: [thammegowda/kidi](https://github.com/thammegowda/kidi).

Kidi-owned code follows [CODING_GUIDELINES.md](CODING_GUIDELINES.md).

The MVP accepts and returns Moses-tokenized UTF-8 text. Raw-text normalization
and detokenization are deliberately outside the first runtime contract.

## Build

The build requires CMake 3.25+, a C++23 compiler, Ninja, and Python 3.10+ for
YNNPACK's source generators. The model converter separately requires the
trusted RTG PyTorch environment.

Clone all pinned source dependencies with the repository:

```bash
git clone --recurse-submodules https://github.com/thammegowda/kidi.git
cd kidi
```

For an existing clone:

```bash
git submodule update --init --recursive
```

```bash
cmake --preset debug
cmake --build --preset debug
ctest --preset debug
```

All C and C++ dependencies are submodules under `third_party/`. Configuration
does not download source code after the recursive clone.

### Apple Silicon

On arm64 macOS, kidi detects CPU features through `sysctl` and enables
YNNPACK's matching NEON, dot-product, BF16, I8MM, SME, and SME2 kernels at
runtime. Unsupported instructions remain disabled, so the same binary can run
on older Apple Silicon. `kidi inspect` prints the selected CPU features and
`kidi predict --profile` includes both their names and YNNPACK bitmask.

YNNPACK is a CPU runtime and cannot dispatch work to Metal or the Apple Neural
Engine. `--backend mps --beam-size 1` selects the eager Metal backend
for FP32, BF16, or INT8 packages. YNNPACK remains the default. ANE
execution requires a separate Core ML model graph, so kidi does not claim NPU
execution until an incremental decoder with explicit K/V state is available.

## Tensor Backends

`kidi::tensor::Tensor` is the common dtype, shape, stride, storage, and device
handle used by model weights and inference activations. Backend-owned storage
can move through `Tensor::to()` between these device kinds:

| Device | Backend | Storage and transfer | RTG eager execution |
|---|---|---|---|
| `cpu:0` | YNNPACK | yes | yes |
| `a_gpu:0` | Metal/MPS | yes on Apple platforms | FP32/BF16/INT8 greedy and beam; experimental |
| `cuda:N` | CUDA/cuDNN | yes when both shared libraries and device `N` are available | not yet |
| `q_npu:N` | Qualcomm QNN/Hexagon | provider slot; register an SDK-backed `Backend` | not yet |

The backend registry never substitutes CPU execution silently. An unavailable
device returns a structured `UNSUPPORTED` error, and a backend reports storage
availability separately from execution capability. `kidi inspect` prints
the detected backend inventory for the current process.

```cpp
auto cpu = kidi::tensor::Tensor::from_host({2, 3}, values);
auto gpu = cpu->to(kidi::tensor::Device::apple_gpu());
auto restored = gpu->to(kidi::tensor::Device::cpu());
```

### Eager Architecture

```text
Model -> Bound Layers -> Eager Tensor Operations -> Device Backend
```

`ops::Context` operates on real `tensor::Tensor` values. There are no model
placeholders, graph IDs, model compilation, or recorded loops. CPU calls execute
immediately; Metal calls enqueue work and `context.synchronize()` makes results
ready for host inspection. Model inference synchronizes before returning logits.

```cpp
layers::Linear linear = std::make_shared<layers::LinearImpl>(weights, "projection", encoding);
auto projected = linear->forward(context, input);
auto activated = context.gelu(projected);
context.synchronize();
auto values = activated.data<float>();
```

Use `context.add(input, other)` for a nonmutating result, and
`context.add_(input, other)` to update `input` in place. The underscore variants
return the same tensor by reference; existing aliases observe the writes.
The scoped thread-local `ops::is_inplace` mode is handled internally. Decoder
cache updates use `scatter_`; explicit synchronization is still required before
host inspection of queued Metal operations.

Weights are bound once into reusable `Linear`, `LayerNorm`, `Attention`,
`FeedForward`, `EncoderBlock`, and `DecoderBlock` layers. Both devices run the
same `model::Transformer`. Generic greedy, batched greedy, and beam search use
ordinary C++ loops in `inference::Decoder`; K/V caches are explicit tensor state.

Every neural implementation is a `Module` subclass named `<Name>Impl`;
`KIDI_MODULE(Name)` provides its shared-pointer alias. `forward(...)` performs
computation. Registered parameters and child modules support `state_dict()` and
validated `load_state_dict(...)`; typed `ModuleList` and `ModuleMap` containers
register their children automatically. No shared-pointer copies are needed in
forward calls.

YNNPACK and MPSGraph may prepare small backend-private operators. These are
cached implementation details, not a second model execution API. Metal retains
the existing resident INT8 kernels, including packed GEMV and tiled GEMM. There
is no claim of native M5 Neural Accelerator use.

Use `--batch-size 32 --batch-window 32` for batched execution. Length sorting
and original output order, including blank lines, are preserved. See
[ARCHITECTURE.md](ARCHITECTURE.md) for contracts and
[the eager migration report](benchmarks/metal/EAGER_MIGRATION.md) for measured
performance and numerical differences. Earlier graph throughput reports are
historical and do not describe the current eager implementation.

## Model Package

```text
model.yaml
model.safetensors
tokenizer.src.json[.gz]
tokenizer.tgt.json[.gz]
```

`--model` points to this directory. Kidi loads `model.yaml` by convention, then
resolves its weights and tokenizer paths relative to the model directory.

Inspect a package manifest with:

```bash
kidi inspect --model /path/to/model
```

Predict Moses-tokenized text from standard input, one sentence per line:

```bash
printf '%s\n' 'Comment allez @-@ vous ?' | kidi predict --model /path/to/model
```

Read and write files explicitly:

```bash
kidi predict --model /path/to/model --in source.txt --out translation.txt
```

Override decoding defaults or append the normalized hypothesis score:

```bash
kidi predict --beam-size 2 --max-extra-tokens 20 --length-penalty 0.6 --score \
  --model /path/to/model --in source.txt --out translation.txt
```

`--batch-size N` permits up to 256 independent sentences per device batch.
`--batch-window N` bounds length-sorting lookahead (batch size through 4096,
defaulting to batch size). Metal uses padding masks and independent per-row EOS;
CPU currently processes the window serially. A final partial batch is supported.

The default input type is `text`. `--inp-type jsonl` is reserved for future
chat-model input and is not yet supported.

Use `--stats` to emit exact input, translation, and generated target-token
counts to stderr. Use `--profile` to emit aggregate package-load, graph-build,
encoder, decoder, generator, and host-search timings in nanoseconds. Both
options leave translation output unchanged. `--threads N` sets total YNNPACK
threads including the calling thread.

Use `kidi --help` and `kidi predict --help` for the complete command syntax.

## Convert RTG

The converter runs in a trusted Python environment with RTG's PyTorch version:

```bash
python3 tools/convert_rtg.py /path/to/exported-rtg-model /path/to/kidi-model
```

Select the matrix and embedding weight encoding during conversion:

```bash
python3 tools/convert_rtg.py --precision bf16 /path/to/rtg-model /path/to/kidi-bf16
python3 tools/convert_rtg.py --precision int8 /path/to/rtg-model /path/to/kidi-int8
```

| Precision | Matrix and embedding storage | Other parameters | Execution |
|---|---|---|---|
| `fp32` | FP32 | FP32 | FP32 |
| `bf16` | BF16 | FP32 | BF16 dot with FP32 accumulation |
| `int8` | symmetric per-channel INT8 plus FP32 scales | FP32 | dynamic INT8 activations, FP32 output |

All linear weights use `[input, output]` storage. Embeddings keep
`[vocabulary, hidden]`. FP32 and BF16 exports store the tied target embedding
only once; the output projection uses it with transposed access. Older BF16
packages with a separate output matrix remain supported. INT8 retains separate
output storage and uses one scale per output channel or embedding row.

## Model Regression Test

```bash
make setup-test
make test
```

`tests/setup.sh` prepares an isolated Python environment and downloads a pinned
BF16 [RTG 500 model](https://huggingface.co/thammegowda/rtg-500eng-v1). The model
is stored under `$KIDI_HUB/rtg-500eng-v1`; `KIDI_HUB` defaults to
`~/.cache/kidi/model-hub`. Override it for both commands when using another cache:

```bash
make setup-test KIDI_HUB=/path/to/model-hub
make test KIDI_HUB=/path/to/model-hub
make test BACKEND=mps
```

Setup writes `._OK` only after each step succeeds: one in `.cache/test-venv/`
and one in the model directory. Repeated runs skip completed setup, including
offline runs. Changed dependency requirements or model revisions invalidate
their markers. Remove a marker to repeat its setup step. If the Hub repository
requires authentication, use `.cache/test-venv/bin/hf auth login` and rerun setup.
Python 3.9+ with `venv` and `pip` is required for this test tooling; use
`make setup-test PYTHON=/path/to/python` to select an interpreter.

`make test` (also `make regression-test`) builds the release CLI, translates
50 fixed multilingual inputs, and writes
`tests/data/sample.eng.rtg-500eng-v1.out.txt`. It compares chrF2 against the
checked-in expected output and fails below **99.0** or on missing/extra lines.
Use `MIN_CHRF=100` for a stricter threshold or `THREADS=N` to select CPU threads.
The expected output is never regenerated by the test. The output file is ignored
by Git. This integration test is separate from the fast CTest unit tests.

See [tests/data/README.md](tests/data/README.md) for fixture provenance and
baseline settings. chrF here measures agreement with the baseline, not accuracy
against human translations.

## Benchmark

The reproducible FP32/BF16/INT8 process benchmark is under
[`benchmarks/rtg`](benchmarks/rtg/README.md).

The Apple Metal lowering benchmark, including synchronized and device-resident
results, is under [`benchmarks/metal`](benchmarks/metal/README.md).

RTG checkpoints use Python pickle and must only be converted from a trusted
source. The resulting runtime package contains no pickle or PyTorch files.