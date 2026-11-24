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
For an [NVIDIA CUDA build](../README-dev.md#cuda), use `--backend cuda:0`:

```bash
python -m kidi chat -m @google/gemma-4-E4B-it-qat-mobile-transformers --backend cuda:0
```

`cuda` also selects GPU 0; `cuda:INDEX` selects a different device. CUDA is an
explicit choice rather than part of `auto` selection and requires a package
built with `KIDI_ENABLE_CUDA=ON`.

You can also use
`kidi chat` instead of `python -m kidi chat` when `kidi` is on `PATH`. On first
use with `@owner/model`, Kidi runs `hf download --local-dir` into
`~/.cache/kidi/model-hub/<owner>/<model>/` and starts chat directly from the
downloaded `config.json` and `model.safetensors` files without modifying the
checkpoint. Unsupported architectures/formats fail explicitly; a Hugging Face ID
alone does not make every model compatible with Kidi.

For image prompts on macOS, `--backend mps` keeps both the language model and the low-bit vision tower on Apple GPU;
the vision tower does not copy intermediate blocks back to CPU during normal inference.

For Gemma, projection and KV-cache precision are separate load-time policies. The defaults
preserve the checkpoint projection format and automatically use the calibrated
INT8 KV cache when the QAT model and backend support it. CPU verification can
select `--precision fp32` or `--precision qat-fp32`; the latter preserves the
checkpoint's trained activation rounding and, on Apple CPU, enables the deliberately slow official-PyTorch parity
math used by the vision benchmark. Explicit `q2a16`, `q4a16`, and
`q8a16` policies requantize every packed projection to that weight width with
BF16 activations; `q*ae4m3` and `q*ae5m2` select FP8 activations. Use
`--kv-cache-precision fp32` to disable INT8 cache storage or
`--kv-cache-precision int8` to require it.
Unsupported model/backend combinations fail instead of falling back.

Apple CPU users can select `--precision lowbit-parity` to retain packed QAT weights while matching the FP32
accumulation oracle through bounded dequantization tiles. This is a correctness-oriented vision mode and is slower than
the native checkpoint kernels.

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
or set `NO_COLOR` to disable automatic colors. Interactive chat is currently greedy text-only
generation with thinking disabled; it does not execute tools or process multimodal input.
History is re-prefilled each turn, not retained as a cross-turn KV cache.

## Inkling-2 on CPU

The CPU-only build can run Inkling-2-Mini, including reasoning, tool calls and the coding agent:

```bash
./build-release/kidi code -m @thinkingmachines/Inkling-2-Mini --backend ynnpack --restart
# Or start a server explicitly:
./build-release/kidi serve -m @thinkingmachines/Inkling-2-Mini --backend ynnpack \
  --precision q8a16 --threads 8 --context-size 2048
```

`--backend ynnpack` selects CPU; a CPU-only binary cannot serve `--backend cuda:7`.
Both the default `checkpoint` precision and explicit `q8a16` convert NVFP4/MXFP8 experts to
signed INT8 with FP32 per-row scales. Gate/up rows are rearranged at load time, and only the
selected experts are evaluated. Dense projections and the unembedding are also packed to INT8, and
matmuls quantize their input rows to INT8 (W8A8, AVX-512 VNNI where available) with FP32 accumulation.
Element-wise activations stay BF16/F32; CPU KV caches store signed INT8 with one FP32 scale per 32 elements
(groups divide each head). This uses 56.25% of BF16 KV storage, including scales, for Inkling-2-Mini.
Prefill batches expert work without per-expert tensor gathers/scatters, and attention shares cached rows across
four query tokens. `kidi code` starts its CPU server with `--threads` (default: up to 16).
On 16 pinned Sapphire Rapids cores, a roughly 4K-token prompt prefills in about 61 seconds
(previously 96 seconds) and decodes at about 19 tokens/s; short-context decode is about 23 tokens/s.

The first CPU load converts weights in parallel across the configured threads and uses Kidi's checkpoint
preparation framework to save `kidi-inkling2-cpu-int8-v1/model.safetensors` beside the source checkpoint.
Later processes memory-map those prepared weights without repeating conversion. The cache is rebuilt when
the source size/mtime, model configuration or prepared format changes. The first start also pays the cache
write cost and needs about 28 GiB of extra disk space for Inkling-2-Mini; the source checkpoint is unchanged.
CPU INT8 needs more RAM than CUDA's
native compact storage, including prepared CPU weight tiles and context buffers. Keep the context small
for smoke tests. CPU `--precision bf16` is not supported; CUDA retains native, INT8 and BF16 modes.

### Inkling-2 speculative decoding

CPU Inkling2 supports **prompt lookup**: no assistant checkpoint is needed. It copies candidate continuations
from matching token sequences in the prompt/generated history and verifies them with the full INT8 target.
This is useful for repeating existing code during edits, not a guarantee of higher acceptance on every coding task.

```bash
taskset -c 16-31 ./build-release/kidi code -m @thinkingmachines/Inkling-2-Mini \
  --backend ynnpack -j 16 --prompt-lookup --draft-tokens 8 --temperature 0 --restart
# Explicit REST server:
taskset -c 16-31 ./build-release/kidi serve -m @thinkingmachines/Inkling-2-Mini \
  --backend ynnpack -j 16 --prompt-lookup --draft-tokens 8 --max-active 1
```

Send REST requests with `temperature: 0`. `generate` and `chat` accept the same lookup/draft flags.
The per-request budget starts at 1, doubles after full acceptance, shrinks to the accepted prefix (minimum 1)
after rejection, and is bounded by `--draft-tokens` (default 3, maximum 16) and remaining generation capacity.
No match proposes nothing, so the target decodes one token normally. Verification restores the accepted
convolution-ring prefix without recomputing the target. KV storage remains group-32 INT8.
Profile output includes `draft_tokens`, `accepted_draft_tokens`, `verification_steps`, `draft_ns` and `verify_ns`.
Compare decode time as well as acceptance: lookup misses and verification overhead can make this slower.
This mode is opt-in, CPU/text/greedy-only and limited to one active request. It cannot be combined with
`--assistant-model`, which still denotes the existing Gemma-specific hidden-state assistant.
That assistant now also uses the adaptive budget.

## Whisper Transcription

Kidi supports the multilingual `openai/whisper-tiny`, `openai/whisper-base`, `openai/whisper-small`, and
`openai/whisper-large-v3`, and `openai/whisper-large-v3-turbo` layouts from their Hugging Face `config.json`, Safetensors, tokenizer, preprocessor,
and generation metadata. It does not require `model.yaml` or modify the original checkpoint.
FP16/BF16 weights are prepared once into a reusable native INT8 cache; original FP32 models still load directly.

```bash
python -m kidi transcribe \
  --model @openai/whisper-tiny \
  --in speech.wav \
  --language auto \
  --max-new-tokens 128
```

The input must be a 16 kHz PCM WAV. Multichannel files are averaged to mono. Language
`auto` selects among the checkpoint's multilingual language tokens; pass an explicit code to skip detection. Add
`--task translate` to request speech-to-English translation. Whisper currently runs on YNNPACK CPU only.
For long recordings, the native CLI reuses one loaded model across bounded, non-overlapping windows
and flushes one transcript line after each window. `--chunk-seconds` defaults to 30 (range 1-30).
Use `--max-new-tokens 444` for dense, full-length windows; the default is 128 tokens per window.
Reaching the limit in a multi-window file fails with the completed output preserved.
Window boundaries can affect words; windows do not share text context, and timestamps and speaker
diarization are not supported. Text-only decoding can also stop early after pauses inside a window;
use Whisper's timestamp-aware long-form decoder when transcript completeness is required.
Individual embedded/microphone requests still accept at most 30 seconds.

```bash
python -m kidi transcribe --model @openai/whisper-large-v3-turbo \
  --in recording-16k.wav --out transcript.txt --max-new-tokens 444 --profile
```

Large-v3-Turbo downloads about 1.62 GB and needs additional disk space for the derived INT8 cache.
For other audio formats or sample rates, convert to 16 kHz PCM WAV before running Kidi.
On macOS, CoreAudio-supported inputs can be converted locally with:

```bash
afconvert -f WAVE -d LEI16@16000 -c 1 recording.m4a recording-16k.wav
```

Classic and extensible PCM WAV containers are supported.

In the browser, choose the speech model by Hub ID. Draft transcription replaces the composer text while recording; the
last pass after stop may revise that draft.

## Cache and Revisions

`chat`, `generate`, `transcribe` and `inspect` accept `-c/--cache`. It controls the Hub download
cache, not the model's KV cache or `--cache-tokens` setting:

```bash
python -m kidi chat -m @google/gemma-4-E2B-it-qat-mobile-transformers \
  --cache "$HOME/models/kidi-cache"
```

Kidi stores Hub models as plain regular directories:
`CACHE/google/gemma-4-E2B-it-qat-mobile-transformers/`.
The printed `Model ready` path is a normal directory containing `config.json`,
`model.safetensors`, and tokenizer files directly (no symlinks or blob hashes),
so you can inspect files directly or pass that directory path to `--model`.

Without a revision, online launches use the repository's default branch. To
fetch a specific commit or tag, append `@REVISION`:

```bash
kidi inspect \
  --model @google/gemma-4-E2B-it-qat-mobile-transformers@dd693ff40353f057ca5f07e945ad867f4afbf2ec
```

Inspection should include `model: gemma4_text` and `native mobile QAT: yes`.
Once a model is cached locally, `@owner/model` (or passing the cached directory
directly) runs offline without invoking `hf`.

Use the same `--cache` when using a custom location. Missing files in a partial
cache produce an error; rerun online to complete the download. A local path such
as `--model /path/to/model` bypasses Hub resolution and needs no `hf` CLI in `PATH`.

## Optional Manual Setup

To download a standalone model directory manually without `@owner/model`:

```bash
MODEL_DIR="../models/gemma-4-E2B-it-qat-mobile-transformers"
hf download google/gemma-4-E2B-it-qat-mobile-transformers \
  config.json model.safetensors tokenizer.json tokenizer_config.json chat_template.jinja \
  --local-dir "$MODEL_DIR"
kidi chat --model "$MODEL_DIR"
```

No conversion step is needed; Kidi reads `config.json` directly.
For trusted legacy RTG exports, use `python tools/converters/rtg.py --help`.
Automatic Hub setup never runs downloaded Python code or unpickles RTG training
checkpoints.

## RTG Translation from Hugging Face

The public, ready-made RTG package is
[thammegowda/rtg-500eng-v1](https://huggingface.co/thammegowda/rtg-500eng-v1).
It includes `config.json` and compressed tokenizers, so no conversion is
needed. With `hf` in `PATH` (for example via `kidi[hf]`), the first command
downloads the model automatically; later runs reuse the cache. No Hugging Face
login is required.

```bash
printf '%s\n' 'Comment allez @-@ vous ?' | \
  kidi translate -m @thammegowda/rtg-500eng-v1 \
    --beam-size 1
```

RTG uses `translate`, not `chat`, and expects Moses-tokenized text, one sentence per
line. The same `--cache` option applies.

## Troubleshooting

- **`python` or `kidi` not found:** activate `.cache/venv` again. Use
  `python -m kidi` to run the package installed in that environment.
- **No `chat` command:** install a Kidi revision or wheel containing chat support;
  an older installed package does not update when you only rebuild a native binary.
- **Missing submodule during installation:** run
  `git submodule update --init --recursive`, then retry `python -m pip install '.[hf]'`.
- **`'hf' was not found in PATH`:** activate an environment with `huggingface-hub[cli]`
  installed (`pip install 'kidi[hf]'`), or download the model manually and pass its local directory path.
- **Download denied (401/403):** verify the exact model ID, log in with
  `hf auth login`, and check account access on the model page.
- **Missing `config.json` with a local path:** ensure `config.json` is inside the
  model directory, or supply `@owner/model` for automatic download.
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
see [`kidi generate`](../README.md#gemma-4-generation).