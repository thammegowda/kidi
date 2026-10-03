# Kidi for Android

Kidi's Android app is a native ARM64 chat client for Gemma 4 with Whisper microphone dictation. Jetpack Compose owns
the interface, a small JNI bridge owns application/runtime translation, and the existing C++ `inference::Generator`,
`inference::Transcriber`, and YNNPACK backend perform inference. There is no WebView, remote inference service, or
second implementation of either model.

## Requirements

- a recursive Kidi checkout;
- JDK 17;
- Android SDK Platform 36 and Build Tools 36.0.0;
- Android NDK 28.0.13004108;
- Android CMake 3.31.6.

Android Studio can install those SDK packages. With the Android command-line tools on macOS:

```bash
brew install openjdk@17 android-commandlinetools
export JAVA_HOME="$(brew --prefix openjdk@17)/libexec/openjdk.jdk/Contents/Home"
sdkmanager --sdk_root="$HOME/Library/Android/sdk" \
	'platforms;android-36' 'build-tools;36.0.0' 'cmake;3.31.6' 'ndk;28.0.13004108'
```

## Build and Run

For distribution through Google Play, follow the
[developer release guide](../README-dev.md#google-play-releases). It covers signed
app bundles, tester enrollment, updates, and Play-readiness checks.

From the repository root:

```bash
make apk
```

This performs a clean optimized release build and lint, aligns/signs/verifies the
APK, and writes `dist/kidi-release.apk`. It does not install anything or change
phone data. By default it uses your existing `~/.android/debug.keystore`, so this
is a release-mode APK signed for **local testing**, not public distribution.
If that key does not exist, build `assembleDeveloper` once using the commands below.

For an APK signed with your private release/upload key:

```bash
make apk-release
```

This defaults to `~/.local/share/kidi/keys/upload.keystore`, alias `kidi-upload`,
and prompts for its password in the terminal. It never falls back to the debug
key. The verified output is `dist/kidi-release-signed.apk`, kept separate from
the local-testing APK. Override the key with `APK_KEYSTORE` and `APK_KEY_ALIAS`.
An upload-key-signed APK is for direct distribution; it does not necessarily
match Google's app signing key. Google Play's new-app workflow uses a signed
AAB, not this APK; follow the developer release guide above.

On macOS the script finds Homebrew JDK 17 (or a registered JDK 17) and defaults
to `~/Library/Android/sdk`. Set `JAVA_HOME` and `ANDROID_HOME` to override these;
`ANDROID_SDK_ROOT` is also accepted when `ANDROID_HOME` is unset. Build Tools
default to 36.0.0, overridable with `ANDROID_BUILD_TOOLS_VERSION`.

For your own signing key, set `APK_KEYSTORE` and `APK_KEY_ALIAS`. Optional exported
`APK_STORE_PASSWORD` and `APK_KEY_PASSWORD` are read by `apksigner` from the
environment; otherwise it prompts in the terminal. Do not commit signing keys or
passwords. Installing with `adb install -r dist/kidi-release.apk` preserves data
only when the signing key matches the installed app.

For the optimized developer build and installation:

```bash
export JAVA_HOME=/path/to/jdk-17
export ANDROID_HOME=/path/to/Android/sdk
./android/gradlew -p android :app:assembleDeveloper
./android/gradlew -p android :app:installDeveloper
```

The APK is written to
`android/app/build/outputs/apk/developer/app-developer.apk`. It uses
the same optimized/minified configuration as Release; only its package identity,
launcher resources, version suffix, and debug-key signing differ.

Developer builds use the standard Android debug key and application ID
`ai.gowda.kidi.dev`, so `installDeveloper` installs beside the release app
without replacing it. The launcher identifies it as **Kidi Dev** with an amber
icon and a red `!` badge. Release builds remain `ai.gowda.kidi` with the normal
Kidi icon.

For Java/Kotlin debugger sessions, `assembleDebug` produces the deliberately
unoptimized `ai.gowda.kidi.debug` variant. Do not use that variant for model
startup or throughput measurements. Open `android/` in Android Studio for the
IDE workflow. The Gradle project references the repository root, so moving only
the `android/` directory is not a supported build layout.

The Kotlin namespace is `ai.gowda.kidi`. The app targets Android 16/API 36,
requires Android 10/API 29 or newer, and currently packages only `arm64-v8a`.
Gemma 4 needs a 64-bit device with several gigabytes of available storage and
memory.

## Model and Privacy

See the [Android Privacy Policy](PRIVACY.md) for data handling, permissions,
third-party downloads, and deletion, and the [Terms of Use](TERMS.md) for AI
limitations and responsible use. AI-generated content can be wrong; verify
important information before relying on it.

A published APK does not require sideloaded model files. On a clean install, **Set up models** opens the in-app model
downloads. Each download requires confirmation and displays an approximate transfer/storage size; the app does not
silently start multi-gigabyte downloads on launch. The default public repositories work without a Hugging Face account.
Chat and optional dictation downloads are separate. Internet is needed for setup; already installed models reopen
offline. ADB commands in the development instructions are for building/testing, not the user installation flow.

The default chat model is `google/gemma-4-E2B-it-qat-mobile-transformers`. Model settings accepts another public dense
Gemma 4 repository with the same supported file layout. Loading performs these steps:

1. Resolve `main` through the Hugging Face metadata API to a 40-character immutable revision.
2. Download the original config, Safetensors, tokenizer, tokenizer config, and chat template into app-private storage.
3. Resume a partial file with HTTP ranges and verify LFS files against their published SHA-256 digest.
4. Generate `model.yaml` locally and memory-map the unchanged checkpoint through Kidi.

The default download is about 2.49 GB. Download progress can be cancelled and resumed. Removing the local model deletes
the downloaded files but preserves the current conversation. Android removes the app-private model directory when the
application is uninstalled. Prompts, responses, and model files are not sent to a Kidi server.

File downloads retry transient network failures up to three attempts, resuming
the saved `.part` file with a validated HTTP range. Idle reads time out after
30 seconds. Settings show Connecting, Downloading, Verifying and Retrying phases;
checksum verification is cancellable and remains visible after transfer reaches
100%. Failed chat and speech operations retain separate, selectable errors in
their model sections, rather than relying on a snackbar behind the settings
sheet. Download again to resume valid saved progress; corrupt partial files are
discarded after checksum failure.

Downloads currently run in the app process, not a persistent background service.
Keep the app in the foreground for large downloads. If Android terminates it,
open settings and start the download again; downloaded partial bytes are retained.
This retry handling does not guarantee continuation through process death.

The current permission set is `INTERNET` and runtime `RECORD_AUDIO`. App-private
model/image storage needs no shared-storage permission. The system photo picker
and camera activity use URI grants, so Kidi does not request broad gallery access
or direct camera permission. Adding storage permissions will not fix a slow or
interrupted multi-GB transfer.

The app uses CPU inference with 1-8 threads, a shared 9,216-token context, and a configurable output limit.
Generation is streamed one bounded native step at a time and can be stopped between steps. The current conversation and
runtime preferences are stored locally.

Completed and stopped chat turns retain one in-memory KV cache, up to 512 MiB. A subsequent turn reuses the matching
token prefix, including processed assistant tokens, and evaluates only the remaining input. Reuse moves the existing
buffers instead of copying them. Changed history reuses only its unchanged prefix; incompatible runtime settings,
model reload, or process exit discard the cache. This is one recent conversation prefix, not a KV cache for every saved
thread. Stored chat history is independent of this temporary optimization. First turns still require ordinary prefill.

JNI completion records expose `reused_prompt_tokens`, `prefix_cache_bytes`, `prefix_reserved_bytes`, `prefill_ms`, and
`first_token_ms` so cache hits and first-response latency can be measured independently of decode speed.

### Photos

With the default Gemma 4 mobile model loaded, the composer offers camera and photo-picker actions. Capture through the
system camera app or select an image through Android's photo picker, inspect/remove the preview, and send a question.
Sending only a photo uses "What is in this image?". Tap a photo in the conversation to view it larger.

The app copies selected images into private storage, applies orientation/color conversion, and bounds them to a
2048-pixel longest edge before saving JPEG. No broad photo-library permission is requested; camera capture uses a
temporary, narrowly shared FileProvider URI. The private copy stays with the message when reopening a chat. Audio,
video, and document attachments are not model inputs yet.

Tahoma Vision decodes JPEG/PNG in the shared C++ core. Gemma's vision tower, image projection, and text-model image
embeddings run on device using the vision weights already present in the default checkpoint. The tower loads lazily,
so text-only startup does not pay its cost. Image features are reused for follow-up questions, and exact image content
is part of KV-cache compatibility. A changed photo cannot reuse KV from a different image with identical placeholders.
The native request limit is eight images and 32 MiB of encoded input. Only supported causal Gemma vision configurations
are enabled; this does not add remote vision inference or restore Gemma audio input.

The APK bundles the Tahoma Vision and codec redistribution notices. PDF/SVG, codec command-line tools, and Python
bindings are disabled in the embedded dependency build.

### Speech Dictation

The default speech model is **Whisper Small INT8**. Selecting `openai/whisper-small`
downloads `ggml-small-q8_0.bin` from `ggerganov/whisper.cpp`, plus the original
config, tokenizer, preprocessor and generation JSON from `openai/whisper-small`.
No FP32 weight file is downloaded. Both repositories resolve `main` to immutable
revisions, and both revisions are retained in the installation identity. The
Q8 file is verified and stored locally as `ggml-model.bin`.

The download is about **267 MB** including metadata. First load imports the
GGML weights into a separate `ggml-model.bin.kidi-int8-v1` native cache of about
249 MB. Source and cache are retained, totaling about 519 MB; allow **600 MB**
free for installation. Subsequent loads reuse the cache, and removing the model
removes that installation and its cache. The input convolution, normalization,
positions, biases, and scales remain FP32; other weights, including the tied embedding/output table, are INT8.
YNNPACK dynamically quantizes projection inputs and executes integer math on CPU.
This is not merely INT8 storage expanded to FP16. GGML block scales are repacked
to the native per-output-channel layout; numerical quality is not guaranteed to
be identical to the original checkpoint.

Already-installed FP32 Small and its `kidi-int8-v2` cache remain loadable and are
not deleted or redownloaded automatically. Remove that speech model and download
Small again to switch formats. Existing Tiny/Base selections are unchanged and
continue to use their original Safetensors files. Chat and speech downloads have
separate progress, cancellation, restore and removal controls.

The earlier FP32-to-INT8 Small path matched Small FP32 quality on a 40-clip/776-word English clean-speech confirmation sample (2.32% versus
2.45% word error rate; Tiny 9.79%). This small sample does not establish multilingual, noisy-speech, or accent coverage.
Those measurements do not certify GGML Q8 re-quantization.

Speech runs on the CPU INT8 path in Auto mode; see [Accelerators](#accelerators) for chat.
Release builds enable R8/resource shrinking and discard unused native dependency sections while preserving JNI
entry points and partial-transcript callbacks.

Speech has its own native lock and runtime thread, like the web app's speech worker, so it loads alongside Gemma and
dictation never queues behind chat loading. Loading memory-maps the cached INT8 checkpoint and then transcribes a second
of silence, so weight packing and operator preparation (about 1.4 seconds on the SM8750 phone) happen before the first
recording. Speech is ready about 1.8 seconds after launch, and the microphone stays available while the chat model loads.
A new model or missing INT8 cache still incurs the one-time conversion cost. Typing stays editable during model loading,
and sending is enabled once chat is ready even when speech is preparing.

The microphone button requests `RECORD_AUDIO` permission when first used. Recording captures mono PCM at 16 kHz for at
most 30 seconds. During recording, the bottom-right send button becomes a red stop control; tap it to finish earlier.
It returns to Send after transcript refinement, while generation uses the same button to stop the response.
Kidi automatically detects the language. Like the web app, each pass encodes only the recorded audio plus at least a
second of silence, rounded up to 128-position encoder slices, instead of a padded 30-second window; output that repeats
itself is decoded again over the full window. Drafts run every 1.2 seconds while recording, and as soon as a pause of
300 ms follows new speech; a stale draft still running then is cancelled so the covering one starts at once. Partial
text is published during token decoding and replaces the dictation suffix in the composer while recording. A draft
failure is shown as an error rather than silently hiding updates; recording continues and the final pass is still
attempted. The transcript is never sent to Gemma automatically. Audio remains in memory only for the current
transcription and is not uploaded or saved.

Stopping reuses a completed (or nearly complete) draft when it already covers all detected speech plus 200 ms, so the
final transcript usually appears immediately. Speech is tracked from 20 ms frame energy against a slowly rising noise
floor and a fraction of the loudest speech; the thresholds err toward counting speech, which only costs a final pass.
Otherwise drafts in progress are cancelled and one final pass runs over the whole recording. On the SM8750 phone,
LibriSpeech clips played through a speaker and stopped after a natural pause took 37-86 ms from Stop to final text for
5.5-17.7 seconds of audio; stopping mid-sentence ran the final pass in 0.44 seconds for 5.8 seconds of audio. The
earlier full-window final pass took about 3.2 seconds plus any draft already in progress.

Provisional speech is light gray in the composer, including while the final pass is decoding. Text typed before dictation
keeps its normal color. Successful finalization restores the normal text color; a failed final pass leaves its draft
provisional until the user edits or sends it.
Repeated hypotheses do not replace the editor value. A new decoding pass retains the existing draft until it catches up;
completed corrections may replace it. The composer uses a fixed three-line, internally scrollable viewport and a remembered
text transformation, so timer updates and shorter hypotheses do not resize or reset the input field.

ASR skips FFT/mel work for the zero padded tail. Convolution writes into a reusable im2col buffer with power-of-two
capacity growth; feature transposition writes directly into its tensor. Forced decoder-prefix tokens populate KV without
unnecessary vocabulary projections. Decoder attention over the cached encoder keys and values uses a dedicated
short-query CPU kernel instead of re-laying out every key and value per token, which cut decoding from about 14.5 to
8.8 ms per token. Decoder self-KV and projected encoder K/V are reused within one transcription. They are not reused
across changing audio, since Whisper's encoder is bidirectional; encoding remains the main cost of a final pass.

## Interface

The chat interface follows the system light/dark theme and uses the canonical Kidi logo. A model-status strip opens
settings, which separates model management from inference controls. Repository details and pinned revisions are
expandable, and downloads are distinguished from native model preparation.

Replies render Markdown, with selectable text and Android's contextual Copy action instead of permanent copy icons.
During generation, reply text and token counters are batched at 100 ms intervals, with an immediate flush on completion,
stop or failure. Inference does not wait for the UI interval. Closed LaTeX equations render while the response streams;
unfinished equations remain text. Existing math drawables are retained as later text arrives, avoiding repeated
placeholder/layout changes. Theme and text-size changes rebuild the renderer and its drawables.
During generation, a fixed status strip shows the current phase, elapsed time, generated tokens, and live decode tok/s.
It does not imply a completion percentage or use the small bouncing response bar. Live decode speed excludes image
analysis and prompt prefill; elapsed time includes them. Final per-message metrics remain available after completion.
Remote images are not loaded. Recording status stays above the composer so live transcript text remains readable.
The microphone has a larger glyph and a 48 dp touch target; model names and device-status text are not repeated beside it.
The composer respects keyboard insets, and conversation width is constrained on larger displays.

The chat footer and the bottom of settings show an experimental-AI notice with
Privacy and Terms links. Both documents open in a scrollable, selectable in-app
reader and are bundled from [PRIVACY.md](PRIVACY.md) and [TERMS.md](TERMS.md), so
they remain available offline. Links between those policies stay inside the app;
external links open only when selected. The notice sets expectations, not a
guarantee of accuracy, safety, legal protection, or user consent to new data uses.

Lato is bundled for offline typography under the [SIL Open Font License](app/src/main/res/raw/lato_license.txt).
Markdown uses [Markwon](https://github.com/noties/Markwon), licensed under Apache 2.0.

## Chat History and Storage

The header logo opens the left chat drawer. It initially reads 30 recent thread summaries, then loads the next page
as the list scrolls. Search uses an on-device full-text index over all saved message bodies, not just the visible page;
words support prefix matching. Opening a thread reads its latest 50 messages, with earlier messages fetched on demand.
Starting a new chat preserves the previous thread. The formerly saved single conversation is imported once on startup;
conversations discarded by older app versions cannot be recovered.

The SQLite database is private to the app and uses foreign keys, transactions, and write-ahead logging:

| Record | Ownership and purpose |
|---|---|
| `threads` | Stable ID, title, last-message preview, creation/activity timestamps |
| `participants` | User or agent identity, display name, optional model ID |
| `thread_participants` | Thread membership; multiple agents can belong to a thread |
| `messages` | Individual message ID, stable ordering, thread, sender, text, completion status, generation metrics |
| `attachments` | Message-owned ordered image/document/audio/video references, MIME type, name, size, optional dimensions/duration |
| `message_search` | Transactionally maintained full-text index |
| `app_state` | Active thread and one-time legacy-import marker |

Messages are appended individually, not serialized as one growing conversation blob. A sender must belong to the
thread; participant kind/model identity cannot silently change. Thread deletion cascades to message and attachment
metadata and the search index. Thread and message pagination use stable cursors rather than large SQL offsets.

Attachment rows reference local `content://` or `file://` URIs; media bytes do not live in message/database rows.
Image imports own private file copies; removing an unsent photo deletes that copy. Other attachment types remain
storage-only and are rejected as model input. Agent membership records identify actual senders; they do not
implement automatic agent scheduling. The current UI sends to the loaded agent, preserving previous agent identities
when a different model is used in the same thread. Failed/stopped replies remain in history.

## Physical Hardware

Read-only inspection on 2026-09-27 identified a Motorola Razr Ultra 2025 running Android 16/API 36:

| Component | Reported capability |
|---|---|
| SoC | Qualcomm SM8750 (Snapdragon 8 Elite) |
| CPU | Six cores up to 3.5328 GHz; two up to 4.32 GHz |
| ARM instructions | NEON, FP16 arithmetic, integer dot product, I8MM, BF16; no SVE/SME advertised |
| RAM | 15,540,640 KiB reported by Android, approximately 16 GB installed |
| GPU | Adreno 830 v2; Vulkan 1.3.284, FP16/INT8 shaders and integer dot products |
| GPU limits | 64-lane subgroups, 32 KiB workgroup memory, 2,147,483,647-byte storage-buffer range |

Small INT8 JNI inference, cache reuse, partial transcripts, and error recovery are checked on the physical phone.
Direct microphone/UI testing requires an unlocked screen. Standalone benchmark setup is documented in
[benchmarks/android](../benchmarks/android/README.md).

## Accelerators

Settings > Inference has **Chat accelerator** and **Speech accelerator**: Auto (default), CPU, GPU, or NPU. The
selection persists, reloads the affected model, and each row shows where the loaded model actually runs. All inference
work is in C++ (`inference::select_device`, `Generator::load`); the app only passes the preference.

The same page includes **Hardware diagnostics** with the detected SoC/CPU and
YNNPACK features, Vulkan physical-device name, and Hexagon HTP generation.
GPU/NPU rows explicitly report Available, Recognized (but unavailable), or Not
recognized so backend-selection failures are distinguishable from unknown
hardware.

**Run accelerator benchmark** opens an in-app benchmark that unloads normal
chat/speech models, runs identical warmup and measured requests on CPU, GPU, and
NPU, and reports model-load time, prompt tokens, prefill time/rate, time to first
token, incremental decode time/rate, app PSS, system memory headroom, backend
errors, and relative decode speedup. Returning to chat recreates the main
activity so its models reload normally.

Example from the SM8750 test phone with a 204-token prompt and 63 measured
incremental tokens:

| Backend | Prefill | Time to first token | Incremental decode | System memory headroom |
| --- | ---: | ---: | ---: | ---: |
| CPU | 192.0 tok/s | 1.06 s | 19.20 tok/s | 6.43 GiB |
| Vulkan GPU | 225.1 tok/s | 0.91 s | 8.82 tok/s | 6.49 GiB |
| QNN HTP NPU | 51.6 tok/s | 3.95 s | 40.11 tok/s | 3.11 GiB |

Thus the current app path delivers **2.09x CPU incremental decode** on NPU.
Short-prompt NPU prefill and first-token latency are still worse because loading
the large cached HTP decode context consumes time and memory; the benchmark
reports this rather than folding it into the decode rate. Long, steady 128-token
NPU prefill graphs separately measure about 1,000 tok/s, but that advantage is
not yet representative of a short interactive request.

- **Auto** tries the Qualcomm NPU, Vulkan GPU, then CPU for chat, falling back only when loading fails. Speech stays on
  the quality-checked CPU INT8 path.
- **CPU, GPU, NPU** are explicit: if that accelerator is unavailable or fails to load, the error is shown and the
  model stays offline rather than silently running elsewhere.
- **GPU (experimental)** runs Gemma 4 on Adreno through Kidi's Vulkan backend (built into the APK; no extra files).
  It is currently slower than the CPU and its greedy output can drift from the CPU's, so Auto uses it only when the NPU
  is unavailable or cannot load.
- **NPU** records Gemma 4 steps on the CPU, then runs complete 128-token prefill
  and single-token decode graphs on Hexagon through QNN. The CPU gathers the two
  token embedding rows so their 1.27 GiB packed tables do not consume the cDSP
  virtual address space; all transformer layers, attention/KV updates,
  vocabulary projection, and argmax execute on HTP. The APK includes the
  matching runtime, prepare library, stub, system library, and V79 skel.
  Prepare is 81 MiB installed but compresses to about 35 MiB in the APK; it is
  what makes a fresh install able to compile its first graph without manual
  provisioning.

  Compilation runs in the background while CPU replay remains available.
  Matching context binaries are persistent: measured reload is about 4 seconds
  for prefill and 8 seconds for decode. The 512-key prefill/decode caches consume
  about 319/804 MiB.

Chat loading finishes with a three-token warm-up request using the serving shapes, so CPU weight packing and, on the
NPU, the decode context load (or the start of its first compilation) happen before the model reports ready rather than
during the first reply. A device that cannot complete the warm-up counts as a failed load, so Auto moves on to the next
accelerator. On the SM8750 phone with the NPU, chat becomes ready about 11 seconds after launch (about 8.5 seconds of
it warm-up); the first reply's first token then arrived in 0.40 seconds and a follow-up's in 0.07 seconds, decoding at
about 43 tokens/s.

## Validation

Run the focused on-device JNI test on a connected ARM64 device or emulator:

```bash
./android/gradlew -p android :app:connectedDebugAndroidTest \
	-Pandroid.testInstrumentationRunnerArguments.class=ai.gowda.kidi.NativeRuntimeTest
```

The test loads `libkidi_android.so`, initializes the YNNPACK thread pool, and checks the Gemma and Whisper JNI error
boundaries, including non-BMP Unicode round-tripping. Keep the app and test APK variants matched; do not install debug
instrumentation over a minified release app. Run Android lint and the release build with:

The opt-in real NPU chat test needs an app-accessible Gemma directory and a build
configured with the local QAIRT SDK:

```bash
./android/gradlew -p android :app:connectedDebugAndroidTest \
  -Pandroid.testInstrumentationRunnerArguments.class=ai.gowda.kidi.NativeRuntimeTest#generatesWithNpu \
  -Pandroid.testInstrumentationRunnerArguments.runNpuGemma=true \
  -Pandroid.testInstrumentationRunnerArguments.gemmaDirectory=/sdcard/Android/data/ai.gowda.kidi/files/models/gemma4
```

```bash
./android/gradlew -p android :app:lintDeveloper :app:assembleRelease
```

`NativeRuntimeTest.transcribesCachedSmallInt8` is opt-in: pass instrumentation arguments `whisperSmallDirectory` and
`whisperSmallAudio` pointing to app-accessible copies of the Small source checkpoint and `benchmarks/android/.cache/speech.wav`.
Generate the audio locally with `bash benchmarks/android/prepare.sh` before staging it on the phone.
It verifies INT8 cache reuse, draft/final transcription, streaming callbacks, callback-error recovery, and exact tokens after reload. `selectWhisperSmall=true` additionally
uses the app's regular verified download/repository flow to make Small the installed speech choice; it is not set by default.