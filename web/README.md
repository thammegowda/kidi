# Kidi in Your Browser

Chat, ask about images, dictate messages, and listen to replies, with AI models
running locally on your device.

**[Open Kidi](https://gowda.ai/kidi/)** — no installation needed.

## Get Started

1. Open Kidi in a current, 64-bit Chrome or Edge browser.
2. Click **Load** next to **Chat / LLM** in the setup card.
   The first download can take a few minutes.
3. Type a message and send it. Load **ASR** for dictation or **TTS** for spoken
   replies when you want them.

Use the language control at the top to choose input and output languages.
Open **Settings** to change models, execution options, or manage cached downloads.

## Good to Know

- **Inference stays on your device.** Messages are processed locally, not by a
  remote AI service. Model downloads come from Hugging Face.
- **Models are large.** Allow several gigabytes of storage and enough free
  memory. A capable computer with WebGPU is recommended; speed varies by device.
- **Downloads are cached.** Later visits can reuse them, but clearing browser
  data or browser storage eviction may require another download.
- **Speech is experimental.** Dictation needs microphone permission. Speech
  models are optional and use additional storage and memory.
- **A first-visit reload is normal.** The app may reload once to enable its CPU
  backend.
- **Check important answers.** AI can make mistakes, and model licenses apply
  separately from Kidi.

## Building or Contributing?

See the [browser developer guide](README-dev.md) for builds, hosting, tests,
and technical details, or the [project README](../README.md) for other Kidi apps.
