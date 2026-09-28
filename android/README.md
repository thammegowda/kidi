# Kidi for Android

Kidi's Android app is a native ARM64 chat client for Gemma 4 with Whisper microphone dictation. Jetpack Compose owns
the interface, a small JNI bridge owns application/runtime translation, and the existing C++ `inference::Generator`,
`inference::Transcriber`, and YNNPACK backend perform inference. There is no WebView, remote inference service, or
second implementation of either model.

## Requirements

- a recursive Kidi checkout;
- JDK 17;
- Android SDK Platform 35 and Build Tools 35.0.0;
- Android NDK 28.0.13004108;
- Android CMake 3.31.6.

Android Studio can install those SDK packages. With the Android command-line tools on macOS:

```bash
brew install openjdk@17 android-commandlinetools
export JAVA_HOME="$(brew --prefix openjdk@17)/libexec/openjdk.jdk/Contents/Home"
sdkmanager --sdk_root="$HOME/Library/Android/sdk" \
	'platforms;android-35' 'build-tools;35.0.0' 'cmake;3.31.6' 'ndk;28.0.13004108'
```

## Build and Run

From the repository root:

```bash
export JAVA_HOME=/path/to/jdk-17
export ANDROID_HOME=/path/to/Android/sdk
./android/gradlew -p android :app:assembleDebug
./android/gradlew -p android :app:installDebug
```

The APK is written to `android/app/build/outputs/apk/debug/app-debug.apk`. Open `android/` in Android Studio for the
IDE workflow. The Gradle project references the repository root, so moving only the `android/` directory is not a
supported build layout.

The application ID and Kotlin namespace are `ai.gowda.kidi`. The app requires Android 10/API 29 or newer and currently
packages only `arm64-v8a`. Gemma 4 needs a 64-bit device with several gigabytes of available storage and memory.

## Model and Privacy

The default chat model is `google/gemma-4-E2B-it-qat-mobile-transformers`. Model settings accepts another public dense
Gemma 4 repository with the same supported file layout. Loading performs these steps:

1. Resolve `main` through the Hugging Face metadata API to a 40-character immutable revision.
2. Download the original config, Safetensors, tokenizer, tokenizer config, and chat template into app-private storage.
3. Resume a partial file with HTTP ranges and verify LFS files against their published SHA-256 digest.
4. Generate `model.yaml` locally and memory-map the unchanged checkpoint through Kidi.

The default download is about 2.49 GB. Download progress can be cancelled and resumed. Removing the local model deletes
the downloaded files but preserves the current conversation. Android removes the app-private model directory when the
application is uninstalled. Prompts, responses, and model files are not sent to a Kidi server.

The app uses CPU inference with 1-8 threads, a shared 9,216-token context, and a configurable output limit.
Generation is streamed one bounded native step at a time and can be stopped between steps. The current conversation and
runtime preferences are stored locally.

### Speech Dictation

The default speech model is **Whisper Small INT8**, from `openai/whisper-small`. Existing user-selected Tiny/Base models
are not automatically replaced. Runtime settings can independently download `openai/whisper-tiny`, `openai/whisper-base`, or
`openai/whisper-small`. The app pins and verifies the untouched Hugging Face config, Safetensors, tokenizer,
preprocessor config, and generation config. Chat and speech model downloads have separate progress, cancellation,
restore, and removal controls.

Small's first load builds a separate `kidi-int8-v2` cache in the C++ core. The compact checkpoint is about 249 MB;
the roughly 967 MB upstream checkpoint is retained unchanged. Subsequent loads reuse the cache. Removing the model
also removes its cache. Allow roughly 1.3 GB for download and conversion. The input convolution, normalization,
positions, biases, and scales remain FP32; other weights, including the tied embedding/output table, are INT8.
YNNPACK dynamically quantizes projection inputs and executes integer math on CPU. Tiny and Base retain FP32 loading.
This is not merely INT8 storage expanded to FP16.

Small INT8 matched Small FP32 quality on a 40-clip/776-word English clean-speech confirmation sample (2.32% versus
2.45% word error rate; Tiny 9.79%). This small sample does not establish multilingual, noisy-speech, or accent coverage.
Small is slower than Tiny: four-thread phone runs averaged roughly 2.75 seconds per complete segment versus 0.82 seconds
for Tiny. Live dictation still replaces drafts with one inference in flight; draft latency is model-dependent.

All Android inference uses CPU; no vendor accelerator SDK, calibration audio, or device graph caches are needed.
Release builds enable R8/resource shrinking and discard unused native dependency sections while preserving JNI
entry points and partial-transcript callbacks.

Model loading memory-maps the cached INT8 checkpoint without a throwaway startup transcription. CPU operators prepare
on the first real ASR request. A new model or missing INT8 cache still incurs the one-time conversion cost. Typing stays
editable during model loading, and sending is enabled once chat is ready even when speech is preparing. Native operations
remain serialized.

The microphone button requests `RECORD_AUDIO` permission when first used. Recording captures mono PCM at 16 kHz for at
most 30 seconds. Tap the stop control to finish earlier. Kidi automatically detects the language and runs one native
Whisper prefix transcription at a time on the shared runtime thread. Partial text is published during token decoding and
replaces the dictation suffix in the composer while recording. The final pass after Stop also publishes partial text,
then replaces it with the completed transcript. A draft failure is shown as an error rather than silently hiding updates;
recording continues and the final pass is still attempted. The first text still waits for audio encoding to complete.
The transcript is never sent to Gemma automatically. Audio
remains in memory only for the current transcription and is not uploaded or saved.

## Interface

The chat interface follows the system light/dark theme and uses the canonical Kidi logo. A model-status strip opens
settings, which separates model management from inference controls. Repository details and pinned revisions are
expandable, and downloads are distinguished from native model preparation.

Replies render Markdown, with selectable text and Android's contextual Copy action instead of permanent copy icons.
Remote images are not loaded. Recording status stays above the composer so live transcript text remains readable.
The composer respects keyboard insets, and conversation width is constrained on larger displays.

Lato is bundled for offline typography under the [SIL Open Font License](app/src/main/res/raw/lato_license.txt).
Markdown uses [Markwon](https://github.com/noties/Markwon), licensed under Apache 2.0.

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

The Android app executes on CPU, regardless of which vendor libraries the phone exposes. Small INT8 JNI inference,
cache reuse, partial transcripts, and error recovery are checked on the physical phone. Direct microphone/UI testing
requires an unlocked screen. Standalone benchmark setup is documented in [benchmarks/android](../benchmarks/android/README.md).

## Validation

Run the focused on-device JNI test on a connected ARM64 device or emulator:

```bash
./android/gradlew -p android :app:connectedDebugAndroidTest \
	-Pandroid.testInstrumentationRunnerArguments.class=ai.gowda.kidi.NativeRuntimeTest
```

The test loads `libkidi_android.so`, initializes the YNNPACK thread pool, and checks the Gemma and Whisper JNI error
boundaries, including non-BMP Unicode round-tripping. Keep the app and test APK variants matched; do not install debug
instrumentation over a minified release app. Run Android lint and the release build with:

```bash
./android/gradlew -p android :app:lintDebug :app:assembleRelease
```

`NativeRuntimeTest.transcribesCachedSmallInt8` is opt-in: pass instrumentation arguments `whisperSmallDirectory` and
`whisperSmallAudio` pointing to app-accessible copies of the Small source checkpoint and `benchmarks/android/.cache/speech.wav`.
Generate the audio locally with `bash benchmarks/android/prepare.sh` before staging it on the phone.
It verifies INT8 cache reuse, draft/final transcription, streaming callbacks, callback-error recovery, and exact tokens after reload. `selectWhisperSmall=true` additionally
uses the app's regular verified download/repository flow to make Small the installed speech choice; it is not set by default.