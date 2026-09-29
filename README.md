# Kidi <a href="docs/kidi-logo.png"><img src="docs/kidi-logo-small.png" alt="Kidi logo" width="48" height="48"></a>

**Local AI chat, image questions, speech transcription, and translation.**

Kidi ("spark" in Kannada) runs models on your device. Use the Android app for
chat, photos, and dictation, the browser app for local chat and dictation, or
the command line for chat, audio files, and translation.

[Quick Start](#quick-start) | [Android](#android) | [Browser](#webassembly) | [Chat](#interactive-chat) |
[Transcription](#whisper-transcription) | [Translation](#rtg-model-package) | [Developer Guide](README-dev.md) |
[Getting Started Guide](docs/getting-started.md)

## Why Kidi?

- **Private inference.** Prompts are processed locally, not sent to a remote
  inference server. Internet is needed to download models, not to run them afterward.
- **Chat and more.** Stream replies, ask follow-up questions, dictate a message,
  or ask about a photo in the Android app.◊
- **No heavyweight inference install.** Running Kidi does not require PyTorch or
  Transformers. Model downloads and memory requirements are separate from app size.
- **Interactive or scripted.** Use a chat interface or process text and audio
  files from the command line.

## What Works Today

| Model | Capabilities |
|---|---|
| Gemma 4 E2B-it | Text chat and generation; Android photo questions with the default mobile model |
| Whisper Tiny/Base/Small | Multilingual speech transcription and speech-to-English translation |
| RTG Transformer | Text translation, including the public 500-language-to-English model |

The tested command-line setup is **Apple Silicon, macOS 26+, and Python 3.12+**.
Gemma E2B has been tested on a 16 GiB Apple M5. Browser inference has been tested
in current 64-bit Chromium. Native Linux/Windows and Gemma E4B have not been
validated end to end here.

Model support is explicit: an arbitrary Hugging Face URL does not make a new
architecture compatible. Terminal chat and JSONL generation are currently
text-only and greedy. Photo input is available in the Android app, not every
interface. Tool execution, video/document input, CUDA, and Apple Neural Engine
execution are not supported. Whisper runs on CPU and accepts at most 30 seconds
of 16 kHz audio per request.

## Quick Start

For the command line, use Python 3.12 or later. The Git installation below
compiles Kidi, so it also needs Git and a C++23 compiler; on macOS, install
current Xcode Command Line Tools. It does not install PyTorch or Transformers.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install "kidi[hf] @ git+https://github.com/thammegowda/kidi.git@main"
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

The installed `kidi` command and `python -m kidi` expose the same CLI. The `hf`
extra enables Hugging Face downloads. Replace `python3.12` with your installed
Python 3.12+ executable as needed. These shell examples use bash/zsh syntax.

If you have a compatible prebuilt wheel, install it without a compiler:

```bash
python -m pip install '/path/to/kidi.whl[hf]'
```

Use the actual wheel filename for your OS and architecture; availability depends
on the build you received. Wheels still require compatible OS libraries. The
standalone executable needs no Python but accepts local model paths only.
For checkout installation and building your own wheel, see the
[developer guide](README-dev.md#python-package).

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

The self-hosted browser app runs Gemma E2B locally, with streaming replies, saved
chats, Markdown, code highlighting, diagrams, and Whisper microphone dictation.
In a hosted instance, open Model settings and download the chat model; speech
dictation has a separate model download. Models are cached in your browser.
Once the complete model is cached, it loads automatically on later visits.

Microphone access requires permission and a secure browser context (HTTPS or
localhost). Clearing site data removes downloaded models and saved browser data.
To run your own instance, follow the [browser build instructions](README-dev.md#browser).

Use a current 64-bit Chromium browser with ample memory. The Wasm heap can
approach its **4 GiB limit**, and long generations may exhaust it. The browser
defaults to 1,024 output tokens, allows up to 8,192, and shares a 9,216-token
context between the conversation and reply. Its cache is separate from the CLI's.

See the [browser guide](web/README.md) for detailed requirements and caching behavior.

### Android

The Android app supports streaming chat, searchable conversation history, camera
and photo-picker questions, and microphone dictation. It needs an **ARM64 phone
running Android 10 or later**, enough RAM for the selected model, and several GB
of free storage. Not every compatible phone will have the same performance.

Use an APK or Google Play testing invitation supplied by the maintainer. A Play
test requires joining with the invited Google account; do not assume the beta
is available through public store search.

1. Open **Set up models** and download the default chat model (about **2.49 GB**).
2. Start a chat, or attach a photo from the camera or image picker.
3. Optionally download speech and allow microphone access to dictate. The current
  Whisper Small Q8 download is about **267 MB**, with a separate **249 MB** prepared
  cache. Existing Tiny/Base selections are not automatically replaced.
4. Tap the logo to find earlier conversations. After setup, installed models
  reopen offline.

Downloads require confirmation; model weights are not bundled in the APK. Keep
the app open during initial downloads. Chats and copied photos are stored
locally. Uninstalling or clearing app data removes local conversations and model
downloads. A differently signed APK may require uninstalling an existing build,
so check with the maintainer before replacing one.

For more about the app, see the [Android guide](android/README.md). Building APKs
and publishing Play test releases are covered in [README-dev.md](README-dev.md#android).

### Whisper Transcription

Whisper Tiny, Base, and Small load directly from their original Hugging Face directories; no export or Kidi
manifest is created.
Input must be a mono or multichannel 16 kHz PCM WAV of at most 30 seconds. Channels are averaged to mono.

```bash
python -m kidi transcribe \
  --model @openai/whisper-tiny \
  --in speech.wav \
  --language auto
```

Use an ISO language code such as `en`, `es`, `de`, `ja`, or `hi` to bypass detection. `--task translate` translates
supported speech into English; the default task transcribes in the detected or selected language. Whisper is currently
CPU-only, so use `--backend ynnpack` rather than Metal. Hub downloads handle the
required files for you. Advanced local-checkpoint import is documented in
[README-dev.md](README-dev.md#ggml-and-gguf-import).

The browser settings accept Hugging Face model IDs rather than file URLs. Loading resolves the repository's current
`main` revision to an immutable commit before downloading. The speech model field offers `openai/whisper-tiny`,
`openai/whisper-base`, and `openai/whisper-small`; larger models trade substantially more download, memory, and latency
for accuracy. During recording, the composer shows replaceable draft text and runs a final refinement after stop.

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

### Gemma 4 Generation

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

Build instructions, C++ architecture, checkpoint formats, tests, benchmarks,
and release workflows are in [README-dev.md](README-dev.md).

## AI Notice

Kidi is experimental. AI-generated responses, photo interpretations, and speech
transcriptions can be wrong. Double-check important information against reliable
sources; do not rely on Kidi for professional advice or emergencies.

[Android Privacy Policy](android/PRIVACY.md) | [Android Terms of Use](android/TERMS.md)