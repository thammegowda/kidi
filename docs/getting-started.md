# Getting Started: Gemma 4 Chat

This guide takes you from a Kidi checkout to a local chat session using Google's
instruction-tuned Gemma 4 E2B mobile QAT checkpoint. QAT means quantization-aware
training: the model already contains trained low-bit weights and calibration
scales. Kidi loads them directly, without an offline weight conversion.

## The Model

The checkpoint used for Kidi's chat examples was downloaded from Hugging Face:

- **Repository:** [google/gemma-4-E2B-it-qat-mobile-transformers](https://huggingface.co/google/gemma-4-E2B-it-qat-mobile-transformers).
- **Tested revision:** [`dd693ff40353f057ca5f07e945ad867f4afbf2ec`](https://huggingface.co/google/gemma-4-E2B-it-qat-mobile-transformers/tree/dd693ff40353f057ca5f07e945ad867f4afbf2ec).
- **Format:** original Safetensors, native mixed Q2/Q4/Q8 mobile QAT.
- **Download:** approximately 2.49 GB for the files below, mostly model weights.
- **License:** the model page lists Apache-2.0; consult the upstream model card
  for its terms and limitations.

Use this exact `qat-mobile-transformers` model for these instructions. The BF16,
GGUF, compressed-tensors and LiteRT-LM releases are different artifacts; do not
substitute them into this example. Do not add `--weight-bits` or `--group-size`
overrides: the native QAT checkpoint supplies its own quantization policy.

## 1. Install Kidi

The tested setup is Apple Silicon, macOS 26+, and standard CPython 3.12 or later.
Both Metal and YNNPACK CPU execution have been tested on an Apple M5 with 16 GiB
of unified memory. That is not a guarantee of low memory pressure: runtime caches
can use substantially more memory than the download size. Close memory-heavy
applications. Gemma defaults to a 16,384-token context with up to 8,192 output
tokens; for a smaller memory budget, override both with `--context-size 2048
--max-new-tokens 256`.

The examples rely on automatic backend selection: Metal when available, otherwise
YNNPACK CPU. You still need a build or wheel for your OS and architecture; Linux
and Windows builds have not been validated in this guide. Shell examples use
bash/zsh syntax; on Windows, use your shell's virtual-environment activation and
line-continuation syntax, or enter commands on one line.

Source installation needs Git and a C++23 compiler. On macOS, install current
Xcode Command Line Tools. Pip supplies the Python build dependencies, including
CMake, Ninja when needed, scikit-build-core and nanobind. PyTorch, Transformers,
and LiteRT-LM are not needed to run Kidi.

```bash
git clone --recurse-submodules https://github.com/thammegowda/kidi.git
cd kidi
python3.12 -m venv .cache/venv
source .cache/venv/bin/activate
python -m pip install --upgrade pip
python -m pip install '.[hf]'
python -m kidi --version
python -m kidi chat --help
```

Use your installed Python 3.12+ executable in place of `python3.12` if necessary.
Keep this environment activated. For an existing non-recursive checkout, run
`git submodule update --init --recursive` before `pip install '.[hf]'`.
Quote extras such as `'.[hf]'` in shells like zsh. The `hf` extra installs
Hugging Face Hub, PyYAML and filelock; it does not install PyTorch or Transformers.

If you have a compatible prebuilt Kidi wheel, install that wheel instead of
`pip install '.[hf]'`, requesting its extra, for example
`python -m pip install '/path/to/kidi-0.1.0-cp312-abi3-macosx_26_0_arm64.whl[hf]'`.
No compiler or source checkout is needed to use that wheel, including automatic
model setup. See the
[installation reference](../README.md#python-installation) for wheel and Git-URL
installation. A Git install compiles from source; it is not a prebuilt download.

## 2. Start Chat

Run this from any directory. The `@` prefix means a Hugging Face model ID:

```bash
python -m kidi chat \
  --model @google/gemma-4-E2B-it-qat-mobile-transformers \
  --threads 4 \
  --system "Be helpful and concise."
```

The backend and color settings both default to `auto`. To force CPU
execution, use `--backend ynnpack`; to require Apple Metal, use `--backend mps`.
You can also use
`kidi chat` instead of `python -m kidi chat` when `kidi` is the pip-installed
launcher. On first use, Kidi downloads the necessary files into
`~/.cache/kidi/model-hub/`, creates `model.yaml`, and starts chat. No separate
`hf download` or configuration command is needed. Original weights and tokenizer
files remain unchanged. Models already packaged with `model.yaml` use that file.
Unsupported architectures/formats fail explicitly; a Hugging Face ID alone does
not make every model compatible with Kidi.

The model is publicly listed. For authentication or anonymous-download limits,
run `hf auth login` in your terminal using a token with read access, then retry.
Complete any access requirements shown on the model page. Never put tokens in
Git commits or shared command transcripts. Download progress goes to stderr.

The model loads once. Before the first `You>` prompt, Kidi prints load time and
memory statistics. Type a question such as `What is the capital of France?` and
press Enter. Replies stream as they are generated; each completed turn prints
token speed, latency and memory statistics. Follow-up messages include the
conversation history and use the original checkpoint chat template.

| Command | Action |
|---|---|
| `/help` | List shell commands |
| `/clear` | Reset history, retaining the system instruction |
| `/system TEXT` | Change the system instruction and reset history |
| `/multiline` | Enter multiple lines; finish with `/send` or discard with `/cancel` |
| `/exit` or `/quit` | Exit chat |

Ctrl-C cancels the current reply; Ctrl-D exits at the prompt. Use `--color never`
or set `NO_COLOR` to disable automatic colors. Chat is currently greedy text-only
generation with thinking disabled; it does not execute tools or process images
or audio. History is re-prefilled each turn, not retained as a cross-turn KV cache.

## Cache and Revisions

`chat`, `generate` and `inspect` accept `-c/--cache`. It controls the Hub download
cache, not the model's KV cache or `--cache-tokens` setting:

```bash
python -m kidi chat -m @google/gemma-4-E2B-it-qat-mobile-transformers \
  --cache "$HOME/models/kidi-cache"
```

Kidi uses Hugging Face's cache layout:
`CACHE/models--google--gemma-4-E2B-it-qat-mobile-transformers/snapshots/COMMIT/`.
The printed `Model ready` path is the resolved local package. Downloads are reused,
and the generated `model.yaml` is written atomically. Local generated configurations
that exactly match the old 256-output/2048-context defaults are upgraded to
8192/16384; customized configurations and downloaded manifests are preserved.
Do not move a snapshot directory alone: its files can link to the
cache's shared blobs. Copy the entire cache or use the manual download below.

Without a revision, online launches resolve the repository's current default
branch. To reproduce the tested checkpoint exactly, append `@REVISION`:

```bash
python -m kidi inspect \
  --model @google/gemma-4-E2B-it-qat-mobile-transformers@dd693ff40353f057ca5f07e945ad867f4afbf2ec
```

Inspection should include `model: gemma4_text` and `native mobile QAT: yes`.
After the download completes, run offline without Hub checks:

```bash
HF_HUB_OFFLINE=1 python -m kidi chat \
  --model @google/gemma-4-E2B-it-qat-mobile-transformers@dd693ff40353f057ca5f07e945ad867f4afbf2ec
```

Use the same `--cache` when using a custom location. Missing files in a partial
cache produce an error; rerun online to complete the download. A local path such
as `--model /path/to/model` bypasses Hub resolution and needs no `hf` extra.
The standalone C++ executable accepts local paths only; Hub downloads are provided
by the Python launcher.

## Optional Manual Setup

For a standalone model directory or the native C++ executable:

```bash
MODEL_DIR="../models/gemma-4-E2B-it-qat-mobile-transformers"
hf download google/gemma-4-E2B-it-qat-mobile-transformers \
  config.json model.safetensors tokenizer.json tokenizer_config.json chat_template.jinja \
  --revision dd693ff40353f057ca5f07e945ad867f4afbf2ec \
  --local-dir "$MODEL_DIR"
python -m kidi.converters.gemma4 "$MODEL_DIR"
python -m kidi chat --model "$MODEL_DIR"
```

The setup helper creates only `model.yaml` and refuses to overwrite it. Keep all
six files together. Image/audio processor files are not needed for text-only chat.
The converter is also importable as `from kidi.converters.gemma4 import configure`.
For trusted legacy RTG exports, install the separate `convert` extra and use
`python -m kidi.converters.rtg --help`. Automatic Hub setup never runs downloaded
Python code or unpickles RTG training checkpoints.

## RTG Translation from Hugging Face

The ready-made RTG package is
[thammegowda/rtg-500eng-v1](https://huggingface.co/thammegowda/rtg-500eng-v1).
It already includes `model.yaml` and compressed tokenizers, so no conversion is
needed. This repository is currently private: your Hugging Face account must have
access, and `hf auth login` must authenticate that account.

```bash
python -m kidi inspect -m @thammegowda/rtg-500eng-v1
printf '%s\n' 'Comment allez @-@ vous ?' | \
  python -m kidi translate -m @thammegowda/rtg-500eng-v1 \
    --beam-size 1
```

RTG uses `generate`, not `chat`, and expects Moses-tokenized text, one sentence per
line. The same `--cache` and offline options apply. Kidi downloads the declared
Safetensors and tokenizer files, preserving the repository's configuration.

## Troubleshooting

- **`python` or `kidi` not found:** activate `.cache/venv` again. Use
  `python -m kidi` to run the package installed in that environment.
- **No `chat` command:** install a Kidi revision or wheel containing chat support;
  an older installed package does not update when you only rebuild a native binary.
- **Missing submodule during installation:** run
  `git submodule update --init --recursive`, then retry `python -m pip install '.[hf]'`.
- **Hub dependencies missing:** install `'.[hf]'` from the checkout, or install the
  matching wheel with `[hf]`. Plain `pip install .` intentionally omits Hub support.
- **Download denied (401/403):** verify the exact model ID, log in with
  `hf auth login`, and check account access on the model page.
- **Missing `model.yaml` with a local path:** use the optional manual helper above,
  or supply the `@owner/model` ID for automatic setup.
- **Missing chat template/tokenizer configuration:** rerun the download command
  with all five filenames. Do not replace the checkpoint template with hand-written
  role markers.
- **Prompt and generation do not fit the context:** use `/clear`, shorten the
  prompt, or lower `--max-new-tokens`. History plus the output budget must fit;
  increasing context also increases memory use.
- **High memory use:** download size is not runtime memory use. Metal may retain
  expanded FP16 prefill weights. Setting `KIDI_PREFILL_CACHE_BYTES=0` before the
  chat command avoids retaining those expanded matrices, at a performance cost;
  it does not cap total process memory. See the
  [memory architecture](../ARCHITECTURE.md#native-gemma-4-mobile-qat).

Once cached, inference runs locally; use `HF_HUB_OFFLINE=1` to disable Hub lookups.
For ordered JSONL file processing instead of a terminal conversation,
see [`kidi generate`](../README.md#gemma-4-text-generation).