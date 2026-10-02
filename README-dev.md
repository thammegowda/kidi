# Kidi Developer Guide

For installation and everyday use, start with [README.md](README.md).
This guide covers working on Kidi, building its applications, and preparing releases.

[Native Build](#native-build) | [Python](#python-package) | [Android](#android) |
[Browser](#browser) | [Architecture](#architecture-and-models) |
[Checkpoint Import](#ggml-and-gguf-import) | [Tests](#tests) | [Benchmarks](#benchmarks)

## Checkout and Dependencies

Native builds require **CMake 3.25+, Ninja, a C++23 compiler, and Python 3.10+**
for YNNPACK's source generators. Python packaging requires **CPython 3.12+**.
On macOS, use current Xcode Command Line Tools. Native CPU and Apple Metal have
been tested on Apple Silicon/macOS 26; do not assume Linux or Windows parity
without running their build and numerical checks.

```bash
git clone --recurse-submodules https://github.com/thammegowda/kidi.git
cd kidi
git submodule update --init --recursive
```

C/C++ dependencies are pinned submodules under `third_party/`.
[ynnpack-dev](https://github.com/thammegowda/ynnpack-dev) owns nested Slinky,
cpuinfo, GoogleTest, and Google Benchmark dependencies. Initialize recursively;
do not edit vendored dependencies as part of unrelated application changes.

## Native Build

```bash
cmake --preset release
cmake --build --preset release --target kidi_cli
build-release/kidi --help
```

The standalone executable needs no Python or separate inference runtime, but
accepts local model paths only. The Python launcher supplies Hugging Face Hub
resolution. Executable/wheel sizes vary by platform and build; model weights and
runtime memory must be measured separately from artifact size.

## Python Package

From an activated Python 3.12+ environment in the checkout:

```bash
python -m pip install '.[hf]'
python -m kidi --version
python -m pip wheel --no-deps . --wheel-dir dist
```

Use `python -m pip install .` for local model files without Hub support. Pip
supplies Python build dependencies. Wheel builds target the current OS and
architecture and use CPython's 3.12 stable ABI (`abi3`). The package exposes the
CLI and model converters, not a Python tensor API. The installed `kidi` command
and `python -m kidi` share the same native implementation.

The [getting-started guide](docs/getting-started.md) covers authentication,
revision selection, offline use, and manual model setup. Gemma setup writes
configuration only; original weights and tokenizers are not rewritten. Legacy
RTG conversion needs the `convert` extra and a **trusted** training export:
PyTorch checkpoints can contain executable pickle data.

## Android

Requirements: JDK 17, SDK Platform 36, Build Tools 36.0.0, NDK 28.0.13004108,
and Android CMake 3.31.6. The app ID is `ai.gowda.kidi`, minimum API is 29,
target API is 36, and the current build packages ARM64 only.

From the repository root:

```bash
make apk
make apk-release
```

- `make apk`: clean optimized release build and lint, signed with the existing
  Android debug key by default; writes `dist/kidi-release.apk` for local testing.
- `make apk-release`: same optimized build, signed with your private key;
  defaults to `~/.local/share/kidi/keys/upload.keystore`, alias `kidi-upload`,
  and writes `dist/kidi-release-signed.apk`. There is no debug-key fallback.
- Both verify signatures and ZIP alignment; neither installs the app. A signing
  key change prevents ordinary updates over a differently signed installation.
  Uninstalling removes local chats, images and model downloads.

Set `JAVA_HOME` and `ANDROID_HOME` when automatic discovery is unsuitable.
Override signing with `APK_KEYSTORE` and `APK_KEY_ALIAS`. Enter passwords in the
terminal, or provide them securely using the documented environment variables;
never commit keys or passwords. Back up release/upload keys securely.

For an optimized side-by-side developer install:

```bash
./android/gradlew -p android :app:assembleDeveloper
./android/gradlew -p android :app:installDeveloper
```

This variant is signed by the standard debug key, uses `ai.gowda.kidi.dev`, and
otherwise inherits the optimized/minified Release configuration. `assembleDebug` is reserved for Java/Kotlin
debugger sessions and uses the separate `ai.gowda.kidi.debug` package; it is not
a performance build. Use matching developer app and AndroidTest APKs for
instrumentation; do not install test APKs over a minified release.
See [android/README.md](android/README.md) for SDK setup, JNI behavior, device
tests and model storage. Moving only the Android directory is not supported:
the project builds the shared repository C++ core.

### Google Play Releases

The APK targets above do **not** produce the Play upload bundle. For a Play beta:

1. Increment `versionCode` for every new uploaded build; keep `ai.gowda.kidi`
  stable. Use the registered upload key, not the Android debug key.
2. Build the bundle and run lint from the repository root with JDK 17 and the
  Android SDK configured:

  ```bash
  ./android/gradlew -p android :app:bundleRelease :app:lintRelease
  ```

3. Sign the AAB with the upload key using `jarsigner`, or use Android Studio's
  **Generate Signed Bundle / APK > Android App Bundle** wizard. The current
  Gradle release configuration does not sign it automatically. Do not use
  `apksigner` on an AAB. Follow Google's [app-signing guide](https://developer.android.com/studio/publish/app-signing)
  for key creation, certificate registration and secure backups.
4. In Play Console, create an **Internal testing** release, enable Play App
  Signing, upload the signed AAB, add tester accounts and share the opt-in link
  when available. Google signs the installed APKs; that app signing key may
  differ from the upload key used by `make apk-release`.
5. Verify a fresh Play install and an update that preserves chats/models. Do not
  uninstall an existing sideloaded build without warning about local data loss.
6. Complete privacy, Data safety, content-rating and AI-content reporting
  requirements, and test native-library/device compatibility. A successful
  local build is not proof of Play policy compliance.

See Google's [testing-track guide](https://support.google.com/googleplay/android-developer/answer/9845334)
and [personal-account testing requirements](https://support.google.com/googleplay/android-developer/answer/14151465).
Newer personal accounts require a closed test with at least 12 continuously
opted-in testers for 14 days before applying for production access; internal
testing does not satisfy that requirement.

#### Privacy and Legal Publication

The Android [Privacy Policy](android/PRIVACY.md) and
[Terms of Use](android/TERMS.md) are the source documents. Before publication:

- Confirm the publisher's identity matches the Play listing and provide a
  monitored privacy contact. The documents currently identify the Kidi project
  maintainers and link to a public GitHub issue tracker; use a private contact
  channel for requests involving personal information.
- Have qualified counsel review the terms and privacy disclosures for the
  publisher's circumstances and intended markets. Disclaimers do not eliminate
  statutory responsibilities or guarantee protection from liability.
- Publish a stable, publicly accessible, read-only HTML privacy-policy page
  without a sign-in requirement, and enter its URL in Play Console. Publish the
  terms alongside it. Repository Markdown alone does not configure hosting.
- Add accessible Privacy Policy and Terms of Use entries inside the Android
  app. The source documents do not themselves add an in-app legal screen.
- Complete Data safety against the actual release, its dependencies, and
  Google's current definitions. Review third-party model-download traffic and
  platform diagnostics separately from local-only chat processing; do not
  infer every form answer from the phrase "we do not collect data".
- Verify the effective dates and repeat this review when data handling,
  dependencies, or model-hosting services change. Check Google's current
  [User Data policy](https://support.google.com/googleplay/android-developer/answer/10144311)
  and the separate AI-content requirements before submitting.

## Browser

Requirements: Emscripten 6.0.9+, CMake, Ninja, Python 3.10+, and Node.js.

```bash
brew install emscripten node
make wasm
make serve
```

Open **http://localhost:8080/**. Override server settings with
`make serve PORT=8081 WEB_DIR=build-pages PYTHON=python3`. Node and Emscripten are
build-time dependencies; the generated app needs only static hosting. No npm
install is required. Deploy generated assets, not build scripts.

The [browser guide](web/README.md) covers execution modes, caches, browser
requirements, and deployment. Model weights are downloaded by the browser and
are not embedded in the site artifact. Browser caches are separate from native
and Android model storage.

## Architecture and Models

```text
Inference -> Models -> Neural Layers -> Eager Tensor Operations -> Runtime Backend
Checkpoint I/O -> Model-owned configuration and preparation hooks
```

Kidi uses C++23 with a shared model implementation across backends. CPU uses
YNNPACK; Apple GPUs use Metal. Models are ordinary C++ composed from reusable
layers and concrete tensors, not exported model graphs. Backend-specific
operator preparation and fusion remain internal to the runtime.

`kidi::model` owns network definitions, validation, parameter binding and
model-specific checkpoint policy. `kidi::checkpoint` owns generic format/file
I/O and atomic cache preparation. Format readers have their own `ggml/` and
`safetensors/` directories and namespaces. Model hooks supply schema, required
sidecars, compatibility checks, conversion/export rules and cache identity.

Adding an architecture requires its model implementation, weight mappings and
applicable checkpoint hooks. Gemma, Whisper and RTG are examples, not automatic
support for arbitrary Hugging Face architectures.

- [ARCHITECTURE.md](ARCHITECTURE.md): ownership, execution, state and backend contracts.
- [CODING_GUIDELINES.md](CODING_GUIDELINES.md): APIs, naming, formatting and testing rules.
- [Manual model setup](docs/getting-started.md#optional-manual-setup): configuration and converter entry points.

## GGML and GGUF Import

`checkpoint::Weights::load` reads Safetensors, little-endian GGUF v2/v3 and the
legacy Whisper GGML container, detected by magic rather than filename extension.
GGML/GGUF encodings supported by the reader are F32, F16, BF16 (GGUF), Q4_0,
Q4_1, Q5_0, Q5_1 and Q8_0. Unsupported encodings, malformed shapes/offsets,
overlapping GGUF tensors and non-finite decoded values are rejected.
Imported tensors decode individually to F32 on demand; Safetensors keeps its
zero-copy memory mapping.

For Whisper, download `ggml-small-q8_0.bin` from `ggerganov/whisper.cpp` and put
the matching `openai/whisper-small` sidecars beside it: `config.json`,
`tokenizer.json`, `preprocessor_config.json`, and `generation_config.json`.
Pass the binary directly, or name it `ggml-model.bin` in a directory without
`model.safetensors`:

```bash
build-release/kidi transcribe --model ./models/whisper-small/ggml-small-q8_0.bin \
  --in speech.wav --language en
```

The first load validates the GGML header against the HF config and converts one
tensor at a time into Kidi's per-output-channel INT8 layout. Generic preparation
invokes Whisper's hooks and atomically writes `<filename>.kidi-int8-v1/`.
Subsequent loads reuse that cache. The original source is retained.

Small Q8 weights are about 264.5 MB, and the derived checkpoint is 248.7 MB:
roughly 519 MB total with both sidecar copies, not 249 MB total. No 967 MB FP32
checkpoint is needed for this import. Requantization may change outputs; it is
not lossless conversion or a broad ASR quality guarantee. Changes to source
size/mtime invalidate the cache, and stale caches fail explicitly.

GGUF container support does **not** automatically provide Gemma or other model
architecture/tokenizer mappings. GGUF tensor names are retained, with the
existing state-mapping API available to loaders. This importer does not execute
GGML kernels or retain GGML block quantization for inference. Android Small
downloads now combine GGML Q8 weights with matching HF sidecars, with both
repository revisions tracked. The browser does the same for Tiny, Base and Small,
converting in its in-memory file system on first load and then keeping the
converted checkpoint in Cache Storage instead of the GGML file; the Python Hub
path remains unchanged.

The small adapted [reference codec](src/kidi/checkpoint/ggml/dequantize.h)
contains upstream credits, revision and MIT terms. No GGML runtime, backend,
submodule or build system is linked. Keep packaged license notices intact.
Smaller downloads come from quantization, not from container overhead alone.

## Tests

Native unit tests, without downloading a model:

```bash
make native-test
```

This configures Debug with integration tests disabled. For the complete suite:

```bash
make test
```

`make test` builds required artifacts and runs native tests, installed-wheel
Python tests, browser tests, and the 50-sentence RTG regression through one CTest
invocation. The pinned public model downloads on the first run; no Hub login is
required. Individual targets are `native-test`, `python-test`, `web-test`, and
`regression-test`. See [tests/data/README.md](tests/data/README.md) for fixtures
and numerical contracts.

Browser loader/generated-glue tests can also run directly:

```bash
node --test tests/web/*.mjs
```

Android model-download contracts run locally without installing over a phone app:

```bash
./android/gradlew -p android :app:testDebugUnitTest --tests ai.gowda.kidi.ModelDownloaderTest
```

These use a local HTTP test server for interrupted/resumed transfers, offsets
beyond 2 GB, timeouts, HTTP failures, checksum validation and cancellation.
Compose download-status/error regressions are in the existing `ChatUiTest`
instrumentation suite. Keep instrumentation and app signing/build variants
matched, and do not uninstall a tester's app merely to run them: that deletes
its chats and downloaded models.

## Benchmarks

Results depend on hardware, precision, memory pressure and workload. They are
not promises for other devices or proof of quality equivalence across quantizers.

- [Gemma 4](benchmarks/gemma4/README.md): CPU/Metal comparisons with GGML and LiteRT-LM, quality and memory.
- [RTG](benchmarks/rtg/README.md): FP32/BF16/INT8 translation benchmarks.
- [Apple Metal](benchmarks/metal/README.md): backend measurements and execution studies.
- [Android](benchmarks/android/README.md): reproducible device benchmark setup.

Keep generated model files, local reports and large binary artifacts out of Git.

## CI and Deployment

The [Pages workflow](.github/workflows/pages.yml) builds/tests Wasm variants for
relevant changes, reusing SDK/compiler caches; merges to `main` deploy the
browser app. The separate [native workflow](.github/workflows/native.yml) runs
native unit tests and installed Python-wheel smoke tests on macOS 26 without
downloading models. One Release build serves both: the wheel build configures
`build-ci` with tests enabled, and the test programs reuse its objects. Compiled
objects (ccache) are saved even when tests fail. It does not block Pages deployment.

See [GitHub Pages setup](web/README.md#github-pages) for one-time configuration.
The generated browser directory is self-contained for static hosting; model
weights are downloaded at runtime, not committed or shipped in site artifacts.