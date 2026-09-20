# kidi

Kidi is a small C++23 inference toolkit for RTG translation and Gemma 4 text
generation, using YNNPACK on CPU, Metal on Apple GPUs, and memory-mapped
Safetensors weights.

Repository: [thammegowda/kidi](https://github.com/thammegowda/kidi).

Kidi-owned code follows [CODING_GUIDELINES.md](CODING_GUIDELINES.md).

RTG translation accepts and returns Moses-tokenized UTF-8 text; normalization
and detokenization remain outside that contract. Gemma accepts ordinary UTF-8
prompts and uses the original model tokenizer.

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
for FP32, BF16, or INT8 packages. The default `--backend auto` selects an
available GPU execution backend, otherwise YNNPACK CPU. Use `--backend ynnpack`
to require CPU execution. ANE
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
ModuleScope construction(tensor::DType::BF16, false);
layers::Linear linear(768, 2048);
ops::require(linear->set_state(weights));
ops::Context context(linear->device());
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

Constructors declare dimensions and structure without checkpoint or prefix arguments.
Registered children determine state names automatically. `set_state(weights)`
binds weights after construction, validates shapes/dtypes atomically, and transfers
only when the source device differs. Checkpoint keys must match the registered
names; RTG import-name mapping happens once at package loading.

`ModuleScope` temporarily sets thread-local `module_dtype`, `allocate_parameters`,
and `module_device`, inherited by nested constructors and restored on scope exit.
Defaults are FP32, allocated parameters, and an available GPU execution device
(CPU otherwise). `ModuleScope(tensor::Device::cpu())` overrides only the device.
New threads start with their own defaults. Existing modules retain their device;
scopes do not alter `forward()` behavior. Allocated weights/biases start at zero,
LayerNorm scales and INT8 scales at one. Initialize floating weights as needed
through their writable `state_dict()` handles before using them; automatic random
initialization, autograd, and training are not implemented.

Both devices run the
same `model::Transformer`. Generic greedy, batched greedy, and beam search use
ordinary C++ loops in `inference::Decoder`; K/V caches are explicit tensor state.

Every neural implementation is a `Module` subclass named `<Name>Impl`;
`KIDI_MODULE(Name)` provides a `ModuleHolder<NameImpl>` with shared ownership and
constructor forwarding. Construct layers directly as `Linear(input_size, output_size)`;
copying a holder shares the same module. Use `Linear{nullptr}` for an empty holder.
`forward(...)` performs computation. Registered parameters and child modules support `state_dict()` and
validated `set_state(...)` and copying `load_state_dict(...)`; typed `ModuleList` and `ModuleMap` containers
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

## Gemma 4 Text Generation

Gemma 4 E2B-it runs directly from the original Hugging Face Safetensors checkpoint.
The setup helper only creates `model.yaml`; it does not convert, requantize, or
rewrite weights or tokenizers. Key adaptation and small normalization-parameter
promotions happen in memory when loading. Large embedding tables remain mapped
on CPU, including for Metal execution.

With the Hugging Face CLI and PyYAML installed:

```bash
hf download google/gemma-4-E2B-it config.json model.safetensors tokenizer.json \
  --revision 3e22461f65e89153144f8adb70e3b8c2cc9845a7 \
  --local-dir ../models/gemma-4-E2B-it
python tools/configure_gemma4.py ../models/gemma-4-E2B-it
build-release/kidi inspect --model ../models/gemma-4-E2B-it
build-release/kidi generate --model ../models/gemma-4-E2B-it --backend mps \
  --prompt 'What is the capital of France?' --max-new-tokens 64 --profile
```

Use `--backend ynnpack --threads 4` for CPU. The default backend is `auto`.
Omitting `--prompt` reads a prompt from standard input. Single-user-turn chat
formatting is applied by default, with thinking disabled; `--raw-prompt` accepts
an already serialized prompt. Generation is greedy and batch size one. The
current implementation is text-only: no image/audio encoders or speculative
decoding are loaded. E2B-it is verified on the 16 GiB Apple M5; E4B has not been
validated end to end on this machine.

`--context-size` and `--max-new-tokens` override the YAML decoding defaults
(2048 and 256). `--prefill-chunk-size` defaults to 128. The prompt and requested
generation must fit the context. `--runs`, `--warmups`, `--ignore-eos`, and
`--profile` support repeatable performance measurements. Profile records separate
prefill, recurrent decoding, and operator preparation; cold and warm numbers
should not be mixed.

Low-bit execution is opt-in: `--weight-bits 8` uses per-channel Q8 projections;
`--weight-bits 4 --group-size 32` uses grouped Q4 MLPs and per-channel Q8 for
other projections. Packing occurs in memory at load time; the checkpoint is
never rewritten. Embeddings stay at original precision. Multi-row prefill uses
original floating weights by default, with `--packed-prefill` available for
packed GEMM. Retaining original weights means this is not a low-bit-only memory
footprint. Q4 changes predictions; BF16 remains the default, and Q8 is the more
conservative tested low-bit option. See [the packed execution study](benchmarks/gemma/LOW_BIT.md)
for measured speed, memory, loading cost, and quality differences.

CPU local attention skips old masked history in stable buckets. This improves
long-context speed but may change floating-point reductions and generated tokens.
Use `--full-attention-cache` to retain the previous full-history numerical path.

For faster execution, the native `google/gemma-4-E2B-it-qat-mobile-transformers`
checkpoint is supported directly. Run the same config-only setup helper in its
model directory, then use `kidi generate` without `--weight-bits`: trained mixed
Q2/Q4/Q8 weights and activation/cache scales come from the checkpoint. The loader
rejects PTQ precision overrides. Default GPU prefill caches expanded FP16 matrices;
`--packed-prefill` avoids that cache but is slower. See [the native QAT report](benchmarks/gemma/QAT.md)
for downloads, correctness checks, memory costs, and the remaining LiteRT-LM gap.

See [the Gemma benchmark report](benchmarks/gemma/README.md) for measured CPU/GPU
comparisons with LiteRT-LM, including MTP, precision differences, and the applied
executable-reuse optimization. LiteRT-LM remains faster in the measured cases.

## RTG Model Package

```text
model.yaml
model.safetensors
tokenizer.src.json[.gz]
tokenizer.tgt.json[.gz]
```

`--model` points to this directory. Kidi loads `model.yaml` by convention, then
resolves its weights and tokenizer paths relative to the model directory.

Configuration is plain YAML. Model settings belong together under `model`;
decoding defaults belong under `decode`. For example:

```yaml
format_version: 1
weights_file: model.safetensors
tokenizers: {source: tokenizer.src.json.gz, target: tokenizer.tgt.json.gz}
io: {input_format: moses_tokenized, output_format: moses_tokenized}
model:
  type: rtg_transformer_nmt
  encoder_layers: 9
  decoder_layers: 6
  hidden_size: 768
  feed_forward_size: 2048
  attention_heads: 12
  source_vocabulary_size: 512000
  target_vocabulary_size: 64000
  activation: gelu
  attention_bias: true
  tied_embeddings: one-way
  layer_norm_epsilon: 0.00001
  position_encoding: sinusoidal
  maximum_position: 5000
  source_tokens: 160
decode:
  beam_size: 4
  maximum_extra_tokens: 50
  length_penalty: 0.6
```

Models receive `config["model"]` directly and validate their own settings.
The package infers special-token IDs from the source and target tokenizer files;
do not duplicate those IDs in the YAML configuration.
Safetensors is the supported checkpoint format; each layer reads the weight dtype
and any INT8 scales from the checkpoint. No format or encoding field is needed.
CLI decoding options override YAML defaults without config-imposed upper limits.
The former flat `model_type`/`architecture` layout is not supported.

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

Human-readable diagnostics use [spdlog](https://github.com/gabime/spdlog) on
stderr; set `SPDLOG_LEVEL` to control their level. Translation output and
machine-readable `--stats`/`--profile` records are separate from logging.

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
exports must be regenerated with the nested config. INT8 retains separate output
storage and uses one scale per output channel or embedding row.

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