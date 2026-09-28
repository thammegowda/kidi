# Android Hardware Benchmarks

Standalone CPU inference benchmarks and an optional Vulkan projection probe for ARM64 Android devices.
These measure native Release executables, not APK startup or UI latency. Only source and setup instructions belong
in Git; generated audio, model downloads, shaders, logs, and measurement reports stay in the ignored `.cache/` directory.

## Build

From the repository root, with the NDK installed:

```sh
export ANDROID_HOME="$HOME/Library/Android/sdk"
export NDK="$ANDROID_HOME/ndk/28.0.13004108"
cmake -S . -B build-android-baseline -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE="$NDK/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-29 \
  -DCMAKE_BUILD_TYPE=Release -DKIDI_BUILD_BENCHMARKS=ON \
  -DKIDI_BUILD_TESTS=OFF -DBUILD_TESTING=OFF
cmake --build build-android-baseline --target kidi_android_baseline -j 8
```

## Prepare Data

Generate the speech fixture locally; no audio binary is checked in:

```sh
bash benchmarks/android/prepare.sh
```

The script requires macOS `say` with the Samantha voice and `afconvert`. It synthesizes the text embedded in the script
at 145 words/minute and writes `.cache/speech.wav` as mono PCM16 at 16 kHz. Voice/OS versions can change samples; reuse
one generated file for comparisons. On other hosts, copy a prepared file into that same ignored path. The runner checks
that the phone has the identical WAV and records its SHA256.

Use `main`, not a commit-pinned URL, and keep model directories stable. For example, with the Hugging Face CLI and
Kidi's existing conversion dependencies installed:

```sh
MODELS=benchmarks/android/.cache/models
hf download google/gemma-4-E2B-it-qat-mobile-transformers --revision main --local-dir "$MODELS/gemma4" \
  --include config.json model.safetensors tokenizer.json tokenizer_config.json chat_template.jinja
python -m kidi.converters.gemma4 "$MODELS/gemma4" --upgrade-defaults
hf download openai/whisper-tiny --revision main --local-dir "$MODELS/whisper-tiny" \
  --include config.json model.safetensors tokenizer.json preprocessor_config.json generation_config.json
```

Refresh and restage the models before measuring a new `main`. Reports retain actual weight hashes for provenance;
the runner does not download or replace models during a timing run.

## Run

Use Python 3.12+; the runner needs only its standard library. Keep USB debugging authorized, use a consistent screen/power
state, and avoid competing workloads. These commands do not change governors, affinity, or thermal policy.

```sh
ADB="$ANDROID_HOME/platform-tools/adb"
REMOTE=/data/local/tmp/kidi-baseline
"$ADB" -s SERIAL shell mkdir -p "$REMOTE/models"
"$ADB" -s SERIAL push build-android-baseline/kidi_android_baseline "$REMOTE/runner"
"$ADB" -s SERIAL push benchmarks/android/.cache/speech.wav "$REMOTE/speech.wav"
"$ADB" -s SERIAL push "$MODELS/gemma4" "$MODELS/whisper-tiny" "$REMOTE/models/"
"$ADB" -s SERIAL shell chmod 755 "$REMOTE/runner"
python benchmarks/android/run.py SERIAL benchmarks/android/.cache/cpu-results.json
```

The CPU runner generates the JSON report with thermal/battery snapshots, source/input hashes, exact-token comparisons,
and three warm repeats after one warmup. A repeated four-thread anchor helps expose drift. Process peak RSS is cumulative
and includes mapped pages, not just live heap allocations. Keep results local rather than committing dated report files.

## Multi-Turn Cache Check

Use the optimized native harness to compare the same second turn with and without prefix reuse. It keeps the model
and prepared operators loaded, verifies exact token/text parity and streamed output, and reports prefill, first-token,
and completion latency separately. KV retention is capped at 512 MiB. Results remain local:

```sh
"$ADB" -s SERIAL shell "timeout 90 $REMOTE/runner chat $REMOTE/models/gemma4 4 1" \
  > benchmarks/android/.cache/chat-turns.jsonl
```

The first turn includes cold operator preparation; do not compare it directly with the cached second turn. Compare the
`cached` and `uncached` records, which use identical conversation input and generation limits in the same process.
This checks the same shared serving core used by the app, not Compose/UI overhead or debug-build inference performance.

## Optional Vulkan Probe

This standalone projection microbenchmark is not an app backend or a full-model speed comparison.

```sh
cmake --build build-android-baseline --target kidi_vulkan_probe -j 8
SHADERS="$NDK/shader-tools/darwin-x86_64"
"$SHADERS/glslc" --target-env=vulkan1.3 -O benchmarks/android/packed_projection.comp \
  -o benchmarks/android/.cache/packed_projection.spv
"$SHADERS/glslc" --target-env=vulkan1.3 -O -DUSE_SUBGROUP=1 -DLOCAL_SIZE=64 \
  benchmarks/android/packed_projection.comp -o benchmarks/android/.cache/packed_subgroup.spv
"$SHADERS/glslc" --target-env=vulkan1.3 -O -DUSE_SUBGROUP=1 -DUSE_DOT=1 -DLOCAL_SIZE=64 \
  benchmarks/android/packed_projection.comp -o benchmarks/android/.cache/packed_dot.spv
"$SHADERS/spirv-val" --target-env vulkan1.3 benchmarks/android/.cache/packed_dot.spv
"$ADB" -s SERIAL push build-android-baseline/kidi_vulkan_probe "$REMOTE/vulkan_probe"
"$ADB" -s SERIAL push benchmarks/android/.cache/*.spv "$REMOTE/"
"$ADB" -s SERIAL shell chmod 755 "$REMOTE/vulkan_probe"
"$ADB" -s SERIAL shell "timeout 30 $REMOTE/vulkan_probe $REMOTE/packed_dot.spv dot" \
  > benchmarks/android/.cache/vulkan-results.json
```

## Checks

Device-free runner checks:

```sh
python -m unittest discover -s tests -p android_benchmark_test.py -v
```