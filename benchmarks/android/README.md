# Android Hardware Benchmarks

Standalone CPU inference benchmarks and optional accelerator benchmarks for ARM64 Android devices.
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

## Image Chat Check

Place a JPEG/PNG test image in the ignored cache and stage it on the device. The image mode asks for a short description,
then a color follow-up; the second turn must reuse KV and encoded image features. It uses the checkpoint's existing
vision weights and does not download another model:

```sh
"$ADB" -s SERIAL push benchmarks/android/.cache/photo.jpg "$REMOTE/photo.jpg"
"$ADB" -s SERIAL shell "timeout 150 $REMOTE/runner image $REMOTE/models/gemma4 4 1 $REMOTE/photo.jpg" \
  > benchmarks/android/.cache/image-chat.jsonl
```

With the existing PyTorch/Transformers reference environment, generate small numerical fixtures locally:

```sh
python tests/gemma4_reference.py tests/.cache/gemma4-vision --vision
python tests/gemma4_reference.py tests/.cache/gemma4-vision-qat --vision --qat
cmake --build --preset debug --target kidi_gemma4_image_test
build-debug/kidi_gemma4_image_test tests/.cache/gemma4-vision
build-debug/kidi_gemma4_image_test tests/.cache/gemma4-vision-qat qat
```

These compare projected image features independently of the real model's natural-language answer. Generated tensors,
photos, and reports remain ignored and are not committed.

## Low-Bit CPU, GPU, and NPU Projections

`kidi_lowbit_bench` measures one quantized projection at the precisions Kidi models use: INT8 activations times W8
(Whisper Small) or W4/W2 (Gemma 4 QAT) weights with per-channel scales, INT32 accumulation, and INT8 requantization with
a static output scale. Every backend receives the same deterministic data and is checked against an exact integer
reference: CPU and GPU must match exactly, and HTP may differ by one quantum from its requantization rounding. A backend
never falls back to another processor.

- CPU: Kidi's YNNPACK graph wrapper and thread pool with native INT8 x INT8/INT4/INT2 dot kernels.
- GPU: Vulkan compute on Adreno with hardware packed INT8 dot products. Weights stay packed in memory and are
  sign-extended in registers. The NDK `glslc` compiles the shaders, which are embedded at build time.
- NPU: QNN HTP loaded with `dlopen`; nothing links against the SDK. Each graph is prepared once into a context binary,
  and measurements load it from a runtime-only directory without the 81 MB `libQnnHtpPrepare.so`.

NPU support is optional and needs a local QAIRT SDK in the ignored cache; SDK files are never committed. Add
`-DKIDI_QNN_SDK="$PWD/benchmarks/android/.cache/qairt/2.50.0.260828"` to the Android configure command above, then:

```sh
cmake --build build-android-baseline --target kidi_lowbit_bench -j 8
"$NDK/toolchains/llvm/prebuilt/darwin-x86_64/bin/llvm-strip" \
  -o build-android-baseline/kidi_lowbit_bench.stripped build-android-baseline/kidi_lowbit_bench
python benchmarks/android/lowbit.py SERIAL benchmarks/android/.cache/lowbit-results.json \
  --binary build-android-baseline/kidi_lowbit_bench.stripped \
  --qnn-sdk benchmarks/android/.cache/qairt/2.50.0.260828
```

The runner covers the Whisper Small FFN (1,500 encoder rows and single-row decoding) and the Gemma 4 E2B MLPs
(single-row decoding and 128-row prefill). Gemma gate and up projections share an input and run as one concatenated
projection. The CPU uses the app's default 4 threads; pass `--threads 8` for its maximum. Use `--cases` to filter and
omit `--qnn-sdk` to skip the NPU. The JSON report includes thermal snapshots, context binary sizes, and whether the
prepare library was loaded.

A context binary only loads in a runtime at least as new as the SDK that prepared it. The SM8750 test phone ships QNN
2.29 in `/vendor/lib64`, which rejects 2.50 contexts, so an app must ship the matching runtime (about 17 MB for V79:
`libQnnHtp.so`, the stub, and the skel) or prepare with an SDK no newer than every target device's runtime.

### Captured Decode FFN Stack

`--workload ffn` measures launch overhead for autoregressive decoding: the Gemma 4 E2B decode FFN stack (35 MLPs;
W4 with intermediate 6144 in layers 0-14, W2 with 12288 in layers 15-34; 495 MB of packed weights) for one token. Each
MLP runs gate/up, GELU-multiply, and down with INT8 activations. Every mode uses preallocated buffers; the host rewrites
the input in place between steps and nothing is reallocated or rebound.

| Mode | Backends | Launches per step |
|---|---|---|
| `eager` | all | one per operator (105), each completed before the next |
| `eager-async` | GPU | one submission per operator, one wait per step |
| `encode` | GPU | the step re-recorded into one command buffer every step |
| `replay` | all | the step captured once: one YNNPACK graph, one Vulkan command buffer, or chained HTP graphs |
| `replay-queued` | GPU | several captured steps submitted back to back behind one fence |

NPU steps are captured as HTP graphs in a context binary and bound to registered FastRPC shared memory, so executions
copy nothing. The prepare library runs out of memory finalizing all 35 MLPs as one graph, so replay captures
`--npu-graph-layers` (default 18) layers per graph and chains them through dedicated shared buffers.

```sh
python benchmarks/android/lowbit.py SERIAL benchmarks/android/.cache/ffn-replay-results.json --workload ffn \
  --binary build-android-baseline/kidi_lowbit_bench.stripped \
  --qnn-sdk benchmarks/android/.cache/qairt/2.50.0.260828
```

Modes repeat round-robin (`--repeats`, default 3) and report the median and range. Weights are synthetic but sampled from
the checkpoint's measured W4/W2 code histograms; activation scales are calibrated statically. The stack omits attention,
normalization, and residuals, so its values are a numerical fixture: without RMSNorm the gated product squares
activation magnitude and deep stacks decay toward zero. Correctness is therefore gated as follows:

- CPU and GPU must match the integer reference exactly at full depth.
- Replay must be bit-identical to eager and must distinguish two inputs on a live 2-layer stack for every backend.
- HTP rounding differs from the reference, so each eager HTP graph is compared with the reference applied to its own
  inputs and must stay within one INT8 quantum RMS. `KIDI_FFN_DIAGNOSE=1` also reports accumulated error per stage.

This is a benchmark prototype. Kidi's runtime remains eager; see [ARCHITECTURE.md](../../ARCHITECTURE.md).

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

## Vulkan GPU backend

Build the Android benchmark with `KIDI_ENABLE_VULKAN=ON` (the Android default) and run Gemma with
`KIDI_ACCELERATOR=gpu` to select `Device::vulkan()` when the phone reports accelerated packed signed INT8 dot products:

```sh
LD_LIBRARY_PATH=/data/local/tmp/kidi-vk \
KIDI_ACCELERATOR=gpu \
/data/local/tmp/kidi-vk/kidi_android_baseline gemma /data/local/tmp/kidi-baseline/models/gemma4 4 3
```

`kidi_vulkan_operator_test` exercises the Vulkan eager backend and skips successfully when no suitable device is
available.  The current backend keeps all tensors host-visible for captured-step input updates and accelerates calibrated
packed projections/fused Gemma MLPs with Vulkan compute; unsupported operators execute through the CPU bridge on the same
Vulkan tensor storage.

## Qualcomm NPU (QNN HTP) backend

Build the Android benchmark with `-DKIDI_ENABLE_QNN=ON -DKIDI_QNN_SDK=/path/to/qairt`.  At runtime push `libQnnHtp.so`, `libQnnHtpPrepare.so` for first-use compilation, `libQnnHtpV79Stub.so`, `libQnnSystem.so`, and `libQnnHtpV79Skel.so` from the same QAIRT release to the phone.  Run with, for example:

```sh
LD_LIBRARY_PATH=/data/local/tmp/kidi-qnn \
ADSP_LIBRARY_PATH=/data/local/tmp/kidi-qnn \
KIDI_QNN_LIBRARY_DIR=/data/local/tmp/kidi-qnn \
KIDI_QNN_CACHE_DIR=/data/local/tmp/kidi-qnn/cache \
KIDI_ACCELERATOR=npu \
./kidi_android_baseline gemma /data/local/tmp/kidi-baseline/models/gemma4 4 3
```

`libQnnHtpPrepare.so` is only needed when the cache does not already contain a
matching context binary. The phone vendor QNN runtime must not be mixed with
context binaries generated by a different QAIRT version.

The production compiler lowers the complete Gemma captured step: packed
embeddings, every body layer, grouped-query attention, KV writes, the vocabulary
head, and argmax all execute on HTP. Prefill uses 128-token chunks; decode uses
power-of-two attention buckets beginning at 512 rows. The host only binds inputs,
synchronizes a cache prefix when the bucket changes, launches the chained
graphs, copies new KV rows, and reads the token.

SM8750 / Gemma 4 E2B QAT steady-state results (four CPU threads):

| Workload | CPU | Whole-step QNN | Speedup |
| --- | ---: | ---: | ---: |
| Prefill, 128-token chunk | ~313 tok/s | 1,026-1,045 tok/s | 3.3x |
| Decode after 384-token prompt | 20.49 tok/s | 37.92-42.49 tok/s | 1.9-2.1x |

The first 16 generated tokens after NPU prefill and NPU decode matched CPU
exactly. The 512-key prefill/decode contexts are 319/804 MiB, compile in about
114/92 seconds, and reload in about 4/8 seconds.
