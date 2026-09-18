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

## Model Package

```text
model.yaml
model.safetensors
tokenizer.src.json[.gz]
tokenizer.tgt.json[.gz]
```

Inspect a package manifest with:

```bash
kidi inspect /path/to/model.yaml
```

Translate Moses-tokenized text with the package defaults:

```bash
kidi translate /path/to/model.yaml 'Comment allez @-@ vous ?'
```

## Convert RTG

The converter runs in a trusted Python environment with RTG's PyTorch version:

```bash
python3 tools/convert_rtg.py /path/to/exported-rtg-model /path/to/kidi-model
```

RTG checkpoints use Python pickle and must only be converted from a trusted
source. The resulting runtime package contains no pickle or PyTorch files.