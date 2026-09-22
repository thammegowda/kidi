# Kidi <a href="docs/kidi-logo.png"><img src="docs/kidi-logo-small.png" alt="Kidi logo" width="48" height="48"></a>

**Run neural models locally, from a terminal, a browser, or C++.**

Kidi ("spark" in Kannada) is a lightweight C++23 inference toolkit. The same
model code runs on CPU, Apple Metal, and WebAssembly CPU. It currently supports
Gemma 4 text generation and RTG translation, with reusable tensors and neural
layers for developers building other architectures.

[Quick Start](#quick-start) | [Browser](#webassembly) | [Chat](#interactive-chat) |
[Translation](#rtg-model-package) | [For Developers](#for-developers) |
[Getting Started Guide](docs/getting-started.md)

## Why Kidi?

- **Local inference.** Models run on your device; prompts are not sent to an
  inference server. Download compatible models from Hugging Face or load local files.
- **A small runtime.** One native executable or Python wheel, without PyTorch or
  Transformers for inference. Measured macOS arm64 release sizes are **8.16 MiB**
  for the executable and **2.46 MiB** for the wheel download. Model weights and
  runtime memory are separate; sizes vary by build.
- **Interactive or scripted.** Streaming chat, cancellation, file processing,
  and translation share the same runtime.
- **One implementation across backends.** YNNPACK handles CPU execution, Metal
  handles Apple GPUs, and the browser uses WebAssembly SIMD with optional threads.

## What Works Today

| Model | Capabilities |
|---|---|
| Gemma 4 E2B-it | Text chat and JSONL generation; original floating-point and mixed 2/4/8-bit mobile QAT checkpoints |
| RTG Transformer | Translation with greedy or beam decoding; FP32, BF16, and INT8 packages |

The tested native setup is **Apple Silicon, macOS 26+, and Python 3.12+**.
Gemma E2B has been tested on a 16 GiB Apple M5. Browser inference has been tested
in current 64-bit Chromium. Native Linux/Windows and Gemma E4B have not been
validated end to end here.

Model support is explicit: an arbitrary Hugging Face URL does not make a new
architecture compatible. Chat is currently text-only and greedy; image/audio
inference, tool execution, CUDA, and Apple Neural Engine execution are not implemented.

## Quick Start

Install from source in an activated Python 3.12+ environment. You need Git and a
C++23 compiler; on macOS, use current Xcode Command Line Tools.

```bash
git clone --recurse-submodules https://github.com/thammegowda/kidi.git
cd kidi
python -m pip install '.[hf]'
python -m kidi chat -m @google/gemma-4-E2B-it-qat-mobile-transformers
```

The first launch downloads about **2.49 GB** of model files and configures them.
Later launches reuse the download cache. Metal is selected when available,
otherwise CPU; add `--backend ynnpack` to require CPU execution.

Download size is not RAM usage. The native defaults are a 16,384-token context
and up to 8,192 output tokens. For a smaller memory budget, add
`--context-size 2048 --max-new-tokens 256`.

See the [getting-started guide](docs/getting-started.md) for environment setup,
authentication, pinned revisions, offline use, and troubleshooting.

### Python Installation

The installed `kidi` command and `python -m kidi` expose the same CLI.
The `hf` extra enables Hub downloads. For local model files only, use
`python -m pip install .` from the checkout.

You can also install directly from Git, optionally replacing `main` with a tag
or commit. This still compiles from source:

```bash
python -m pip install "kidi[hf] @ git+https://github.com/thammegowda/kidi.git@main"
```

If you have a compatible prebuilt wheel, install it without a compiler:

```bash
python -m pip install '/path/to/kidi.whl[hf]'
```

Use the actual wheel filename for your OS and architecture. Source installs
include build dependencies through pip; wheels still require compatible OS
libraries. The standalone C++ executable needs neither Python nor a separate
inference runtime, but accepts local model paths only.

### Interactive Chat

Type a message and replies stream as they are generated. The model stays loaded
between turns. Each completed reply reports tokens, speed, latency, and memory.

| Action | Command |
|---|---|
| Clear conversation | `/clear` |
| Set a system instruction | `/system Be concise.` |
| Enter multiple lines | `/multiline`, then `/send` |
| Cancel a reply | Ctrl-C |
| Help / exit | `/help` / `/exit` |

Conversation history and the requested reply must fit within the context.
Use `/clear` or lower the output budget when it fills up; history is not silently
truncated. See the [chat guide](docs/getting-started.md#2-start-chat) for more.

### WebAssembly

The browser app runs Gemma E2B locally, with streaming replies, saved chats,
Markdown, code highlighting, diagrams, and generation statistics. It downloads
the original pinned Google mobile-QAT checkpoint and caches it in the browser.
Once the complete model is cached, it loads automatically on later visits.

Build and serve it from the checkout:

```bash
brew install emscripten node  # macOS; CMake, Ninja, and Python are also required
make wasm
make serve
```

Open **http://localhost:8080/**. Node and Emscripten are build-time dependencies;
the finished app needs only static hosting. No npm install is required.

Use a current 64-bit Chromium browser with ample memory. The Wasm heap can
approach its **4 GiB limit**, and long generations may exhaust it. The browser
defaults to 1,024 output tokens, allows up to 8,192, and shares a 9,216-token
context between the conversation and reply. Its cache is separate from the CLI's.

The [WebAssembly guide](web/README.md) covers browser requirements, static hosting,
GitHub Pages deployment, caching, and measured performance.

### RTG Model Package

The public [RTG 500 model](https://huggingface.co/thammegowda/rtg-500eng-v1)
translates from 500 languages into English. With the `hf` extra installed,
translate Moses-tokenized text, one sentence per line:

```bash
kidi translate -m @thammegowda/rtg-500eng-v1 -i input.txt -o output.txt --beam-size 1
```

The first run downloads the model; later runs reuse the cache. No Hugging Face
login or manual conversion is required. Use `-m /path/to/rtg-model` for an
existing local package. Text normalization and detokenization remain separate
steps; see the [translation guide](docs/getting-started.md#rtg-translation-from-hugging-face).

### Gemma 4 Text Generation

For scripts, `generate` reads chat requests as JSONL and returns one JSON response
per input line, in order:

```bash
printf '%s\n' '{"messages":[{"role":"user","content":"What is binary search?"}],"max_tokens":128}' | \
  kidi generate -m @google/gemma-4-E2B-it-qat-mobile-transformers
```

Use `-i requests.jsonl -o responses.jsonl` for files. Responses include the
assistant message and token IDs. `generate` is for chat models; use `translate`
for RTG. Run either command with `--help` for decoding and batching options.

## For Developers

### Build

Native builds require **CMake 3.25+, Ninja, a C++23 compiler, and Python 3.10+**
for YNNPACK's source generators. From a checkout:

```bash
git submodule update --init --recursive
cmake --preset release
cmake --build --preset release --target kidi_cli
build-release/kidi --help
```

C/C++ dependencies are pinned submodules under `third_party/`. To build a Python
wheel on the target OS/architecture:

```bash
python -m pip wheel --no-deps . --wheel-dir dist
```

Wheels use CPython's 3.12 stable ABI (`abi3`). The Python package exposes the CLI
and model converters, not a Python tensor API. See
[the architecture](ARCHITECTURE.md) for the runtime interfaces.

### Extend the Toolkit

```text
Model -> Neural Layers -> Eager Tensor Operations -> CPU / Metal Backend
```

Models are ordinary C++ composed from reusable layers and concrete tensors.
There is no model graph export or separate conversion pipeline. Safetensors
weights are memory-mapped; backend-specific operator preparation is internal.

Adding an architecture requires its model implementation and weight mapping.
Gemma and RTG are working examples, not the limits of the tensor/layer APIs.

- [Architecture](ARCHITECTURE.md): module ownership, eager execution, state, and backend contracts.
- [Coding guidelines](CODING_GUIDELINES.md): APIs, naming, formatting, and tests.
- [Getting started](docs/getting-started.md#optional-manual-setup): local Gemma setup and converter entry points.

Gemma setup writes configuration only; original weights and tokenizers are not
rewritten. Legacy RTG conversion needs the `convert` extra and a **trusted**
training export because PyTorch checkpoints can contain executable pickle data.

### Tests

Run the native unit tests without downloading a model:

```bash
cmake --preset debug
cmake --build --preset debug
ctest --preset debug
```

Browser loader and generated-glue tests use Node's built-in runner:

```bash
node --test tests/web/model_cache_test.mjs
```

`make test` additionally builds the release CLI and runs the 50-sentence RTG
regression, downloading a pinned public model on the first run. No Hub login
is required. See [test fixtures and methodology](tests/data/README.md).

### Benchmark

Results are specific to hardware, precision, and workload; they are not a
promise of performance on other systems.

- [Gemma 4](benchmarks/gemma4/README.md): CPU/Metal comparisons with LiteRT-LM, quality, and memory.
- [RTG](benchmarks/rtg/README.md): FP32/BF16/INT8 translation benchmarks.
- [Apple Metal](benchmarks/metal/README.md): backend measurements and execution studies.

### Development

The [Pages workflow](.github/workflows/pages.yml) tests and builds both Wasm
variants on pull requests to `main`; merges to `main` deploy the browser app.
Model weights are downloaded by the browser, not included in the site artifact.
See the [deployment guide](web/README.md#github-pages) for the one-time Pages setup.