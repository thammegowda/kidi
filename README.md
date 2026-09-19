# kidi

Kidi is a small C++23 inference toolkit. The first milestone runs exported RTG
Transformer NMT models directly with YNNPACK on CPU and memory-mapped
Safetensors weights.

Kidi-owned code follows [CODING_GUIDELINES.md](CODING_GUIDELINES.md).

The MVP accepts and returns Moses-tokenized UTF-8 text. Raw-text normalization
and detokenization are deliberately outside the first runtime contract.

## Build

The build requires CMake 3.25+, a C++23 compiler, Ninja, and Python 3.10+ for
YNNPACK's source generators. The model converter separately requires the
trusted RTG PyTorch environment.

Clone all pinned source dependencies with the repository:

```bash
git clone --recurse-submodules REPOSITORY_URL kidi
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
Engine. ANE execution requires a separate Core ML model graph. The current RTG
package and decoder are not Core ML assets, so kidi does not claim NPU use;
an ANE backend must export and load an incremental decoder with explicit K/V
state before it can be benchmarked fairly against the CPU path.

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
`[vocabulary, hidden]`; reduced-precision tied output projection values are
stored separately in linear layout. INT8 uses one scale per output channel or
embedding row.

## Benchmark

The reproducible FP32/BF16/INT8 process benchmark is under
[`benchmarks/rtg`](benchmarks/rtg/README.md).

RTG checkpoints use Python pickle and must only be converted from a trusted
source. The resulting runtime package contains no pickle or PyTorch files.