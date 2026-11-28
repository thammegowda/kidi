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
substitute them into this example. Do not add a `--precision q8` or `q4` override: the native QAT checkpoint
supplies its own quantization policy.

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
python -m kidi agent --help
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
[installation reference](../README.md#quick-start) for Git-URL
installation. A Git install compiles from source; it is not a prebuilt download.

## 2. Start Chat

Run this from any directory. The `@` prefix means a Hugging Face model ID:

```bash
python -m kidi agent \
  --model @google/gemma-4-E2B-it-qat-mobile-transformers \
  --permissions none \
  --system "Be helpful and concise."
```

`kidi agent` starts (or reuses) a background inference server for the model and opens the chat shell;
`--permissions none` chats without workspace tools. The backend and color settings both default to `auto`. To force CPU
execution, use `--backend ynnpack`; to require Apple Metal, use `--backend mps`.
For an [NVIDIA CUDA build](../README-dev.md#cuda), use `--backend cuda:0`:

```bash
python -m kidi agent -m @google/gemma-4-E4B-it-qat-mobile-transformers --backend cuda:0
```

`cuda` also selects GPU 0; `cuda:INDEX` selects a different device. CUDA
requires a build with `KIDI_ENABLE_CUDA=ON`, such as `make release-cuda`; in
such a build, `auto` selects CUDA before Metal and CPU.

You can also use
`kidi agent` instead of `python -m kidi agent` when `kidi` is on `PATH`. On first
use with `@owner/model`, Kidi runs `hf download --local-dir` into
`~/.cache/kidi/model-hub/<owner>/<model>/` and starts chat directly from the
downloaded `config.json` and `model.safetensors` files without modifying the
checkpoint. Unsupported architectures/formats fail explicitly; a Hugging Face ID
alone does not make every model compatible with Kidi.

For image prompts on macOS, `--backend mps` keeps both the language model and the low-bit vision tower on Apple GPU;
the vision tower does not copy intermediate blocks back to CPU during normal inference.

For Gemma, projection and KV-cache precision are separate load-time policies. The defaults
preserve the checkpoint projection format and automatically use the calibrated
INT8 KV cache when the QAT model and backend support it. On CPU, `--precision bf16` runs a BF16 reference that
dequantizes QAT weights. For BF16 checkpoints, `q8` packs weights to INT8 per output channel at load, and `q4` packs
them to INT4 in groups of 128, falling back to per-channel INT8 where a projection width does not divide by 128.
On CUDA, `q8` keeps BF16 activations and also accepts calibrated INT8 QAT checkpoints. Use
`--kv-cache-precision fp32` to disable INT8 cache storage or
`--kv-cache-precision int8` to require it.
Unsupported model/backend combinations fail instead of falling back.

Developers verifying QAT kernels can use the `qat-fp32` oracle through `kidi_eval_kl` or the vision benchmark. It
dequantizes weights for FP32 accumulation, keeps the trained activation rounding and mirrors the reductions of the
pinned Apple-CPU PyTorch build. It is deliberately slow and is not offered by the `kidi` CLI.

The model is publicly listed. For authentication or anonymous-download limits,
run `hf auth login` in your terminal using a token with read access, then retry.
Complete any access requirements shown on the model page. Never put tokens in
Git commits or shared command transcripts. Download progress goes to stderr.

The first run starts a background server that loads the model once; later `kidi agent` sessions with the same
model and settings reuse it until it has been idle for 10 minutes. Type a question such as
`What is the capital of France?` and press Enter. Replies stream as they are generated, and the status line shows
context usage and token speed. Follow-up messages include the conversation history and use the checkpoint's chat
template.

| Command | Action |
|---|---|
| `/help` | List shell commands |
| `/reset` | Clear the conversation |
| `/context` | Show exact prompt and context token counts |
| `/compact` | Summarize older history |
| `/exit` | Exit |

Ctrl-C cancels the current reply; Ctrl-D exits at the prompt. Use `--color never` or set `NO_COLOR` to disable
colors. Sampling follows the agent configuration (temperature 0.6, medium reasoning effort); pass `--temperature 0`
for greedy replies or `--reasoning-effort` to change how long the model thinks. Without `--permissions none`, the
same shell can also read and edit files in `--workspace` after asking for approval.

## Inkling-2 on NVIDIA CUDA

```bash
build-release-cuda/kidi agent -m @thinkingmachines/Inkling-2-Mini --backend cuda
# Or start a server explicitly:
CUDA_VISIBLE_DEVICES=0 build-release-cuda/kidi serve -m @thinkingmachines/Inkling-2-Mini --backend cuda \
  --port 18097 --context-size 65536 --cache-tokens 65536 --prefill-chunk-size 512
```

CUDA defaults to native NVFP4/MXFP8 expert storage, interleaved gate/up kernels and BF16 KV caches, fitting below
24 GiB in the tested 8K-context configuration. On `serve` and `generate`, `--precision q8` explicitly selects INT8
experts on CUDA or CPU; `--precision bf16` is a CUDA-only comparison mode. `--kv-cache-precision int8` stores
group-32 INT8 KV caches (one F32 scale per 32 values, 56% of BF16 storage); `fp32` is rejected on CUDA.

As on Metal, the 20 sliding-window layers keep ring caches of their 1,024-token window plus 4,096 rows, and only
the four full-attention layers (8 visits with the loop) grow with the context: about 32 KiB of KV per token instead
of 192 KiB, or 18 KiB with INT8. On a 48 GB RTX A6000, `kidi agent` sizes a 694,272-token context (previously
119,808) from the same 29 GiB of free memory, and the model's full 1,048,576 tokens with INT8. A turn reuses the
previous prompt's cache only if the reply since then fits in the 4,096 spare rows; longer replies make the next
turn prefill the conversation again.

On an 85K-token prompt, BF16 caches prefilled at 899 tokens/s and decoded at 30.1 tokens/s; INT8 caches at 751 and
29.1. INT8 currently trades some prefill speed for memory: decode attention at this length is limited by
parallelism rather than cache bandwidth.

Inputs of any length run as steps of at most 512 tokens (power-of-two lengths on CUDA), so a large file in the
conversation keeps activation memory bounded, and varied prompt lengths reuse the buffers of a few step lengths.

## Inkling-2 on Apple Silicon

```bash
./build-release/kidi agent -m @thinkingmachines/Inkling-2-Mini --backend mps
```

The managed server defaults to `vq3d-hybrid` weights and BF16 KV storage for Inkling-2 on Metal.
With prepared VQ checkpoints, the input embedding table stays file-backed; only the requested rows are staged
on the GPU. Aligned packed dense projections bind directly to read-only file mappings on Metal. The packed
expert banks are still copied at startup to avoid slowing down decode. Use `--kv-cache-precision int8` to
reduce KV memory for longer conversations. When the server grows an Inkling-2 Metal cache, it preserves the
reusable prefix and continues prefilling only the new suffix.

For larger prompt batches, Metal pairs tokens routed to the same expert and reuses each VQ code across the
pair. It keeps the same VQ weights and FP32 accumulation order; single-token decode uses the original path.

## Inkling-2 on CPU

The CPU-only build can run Inkling-2-Mini, including reasoning, tool calls and the coding agent:

```bash
./build-release/kidi agent -m @thinkingmachines/Inkling-2-Mini --backend ynnpack --restart
# Or start a server explicitly:
./build-release/kidi serve -m @thinkingmachines/Inkling-2-Mini --backend ynnpack \
  --precision q8 --threads 8 --context-size 2048
```

`--backend ynnpack` selects CPU; a CPU-only binary cannot serve `--backend cuda:7`.
Both the default `checkpoint` precision and explicit `q8` convert NVFP4/MXFP8 experts to
signed INT8 with FP32 per-row scales. Gate/up rows are rearranged at load time, and only the
selected experts are evaluated. Dense projections and the unembedding are also packed to INT8, and
matmuls quantize their input rows to INT8 (W8A8, AVX-512 VNNI where available) with FP32 accumulation.
Element-wise activations stay BF16/F32; CPU KV caches store signed INT8 with one FP32 scale per 32 elements
(groups divide each head). This uses 56.25% of BF16 KV storage, including scales, for Inkling-2-Mini.
Prefill batches expert work without per-expert tensor gathers/scatters, and attention shares cached rows across
four query tokens. `kidi agent` starts its CPU server with `--threads` (default: up to 16).
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
taskset -c 16-31 ./build-release/kidi agent -m @thinkingmachines/Inkling-2-Mini \
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

## Scripted Generation and the Server API

For scripted pipelines, `kidi generate` streams JSONL requests to JSONL responses:

```bash
printf '%s\n' '{"messages":[{"role":"user","content":"Explain binary search in three bullet points."}],"max_tokens":128}' | \
  kidi generate -m @google/gemma-4-E2B-it-qat-mobile-transformers
```

For structured, text-only conversations, include `template_args` (even `{}`) to pass message fields and template
options through to the model's chat template:

```json
{"messages":[{"role":"user","content":"Weather in Paris?"},{"role":"assistant","content":null,"tool_calls":[{"id":"c1","function":{"name":"weather","arguments":{"city":"Paris"}}}]},{"role":"tool","tool_call_id":"c1","content":"Sunny"}],"template_args":{"tools":[{"type":"function","function":{"name":"weather","parameters":{"type":"object","properties":{"city":{"type":"string"}}}}}],"reasoning_effort":"low"}}
```

Template options and tool-call argument formats are model-specific; Inkling expects `arguments` as an object, not a
JSON-encoded string. The structured path uses the template's own thinking default unless explicitly overridden.
Requests without `template_args` retain the existing simple-chat behavior. User messages may also carry OpenAI
`image_url` parts with base64 `data:image/...` URLs when the checkpoint has a supported vision tower (Gemma 4 E2B
QAT); `kidi serve` reports this as `capabilities.vision` in `/v1/models`. `generate` responses contain raw assistant text; use `serve` for
parsed reasoning and tool calls.

Add `--assistant-model` (with `@owner/model` or a local directory) to enable Gemma CPU speculative decoding in
`generate` or `serve`:

```bash
printf '%s\n' '{"messages":[{"role":"user","content":"Explain binary search."}],"max_tokens":128}' | \
kidi generate -m @google/gemma-4-E2B-it \
  --assistant-model @google/gemma-4-E2B-it-assistant \
  --precision q8 --draft-tokens 4
```

`kidi serve` and `kidi agent`'s background server run the same server module (`src/kidi/server`): one engine thread owns the model and
serves every request, and the same routes expose an OpenAI-compatible `/v1/chat/completions` endpoint (streaming or
not) for agent harnesses. It supports OpenAI `tools`, string tool-call `arguments`, `chat_template_kwargs`,
`reasoning_effort` and `max_tokens`. Decoding is greedy unless a request sets `temperature`; `top_p` and `top_k` are
also accepted. Responses return parsed `reasoning_content` and `tool_calls` when the model emits reasoning or
tool-call markup (Inkling and Gemma 4 do), and plain `content` otherwise. The server listens on `127.0.0.1:8000` by default and has no API
keys or TLS, so use it only with trusted local clients.

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

## Text-to-Speech

Synthesize natural speech using **OmniVoice** (multilingual with voice design attributes) or **Kokoro-82M** (compact US English):

```bash
# OmniVoice: use a published INT8 package directly, or export offline with tools/converters/omnivoice.py
build-release/kidi synthesize \
  --model @thammegowda/OmniVoice-q8 \
  --text 'Hello from Kidi.' \
  --language English \
  --voice gender=female --voice 'age=young adult' --voice accent=british \
  --out hello.wav
```

```bash
# Kokoro-82M: convert offline with selected voice pack(s) and synthesize
SOURCE="$(hf download hexgrad/Kokoro-82M config.json kokoro-v1_0.pth voices/af_heart.pt README.md VOICES.md)"
python tools/converters/kokoro.py "$SOURCE" .cache/kokoro-int8 --voices af_heart

build-release/kidi synthesize \
  --model .cache/kokoro-int8 \
  --text 'Hello world.' \
  --language English \
  --voice name=af_heart --voice speed=1.0 \
  --out kokoro.wav
```

> **Note on model licenses:** Kokoro-82M and Misaki are Apache-2.0. Pretrained OmniVoice weights are **CC-BY-NC** (non-commercial) and bundle the Higgs Audio 2 decoder; review the generated `NOTICE` file before redistributing converted checkpoints.

## Cache and Revisions

`generate`, `serve`, `transcribe` and `inspect` accept `-c/--cache`. It controls the Hub download
cache, not the model's KV cache or `--cache-tokens` setting:

```bash
python -m kidi inspect -m @google/gemma-4-E2B-it-qat-mobile-transformers \
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
kidi agent --model "$MODEL_DIR"
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
- **No `agent` command:** install a Kidi revision or wheel built on Linux or macOS with agent support;
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
- **Prompt and generation do not fit the context:** use `/reset` or `/compact`, shorten the
  prompt, or lower `--max-new-tokens`. History plus the output budget must fit;
  increasing context also increases memory use.
- **High memory use:** download size is not runtime memory use. Metal may retain
  expanded FP16 prefill weights. `--metal-prefill-cache-mib 0` (on `generate` and
  `serve`) avoids retaining those expanded matrices, at a performance cost;
  it does not cap total process memory. See the
  [memory architecture](../ARCHITECTURE.md#native-gemma-4-mobile-qat).

Once cached, inference runs locally; use `HF_HUB_OFFLINE=1` to disable Hub lookups.
For ordered JSONL file processing instead of a terminal conversation,
see [Scripted Generation and the Server API](#scripted-generation-and-the-server-api).