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

## Dictation Check

The dictation mode mirrors the app's speech path: it prepares the same INT8 cache from a Whisper checkpoint directory,
warms up, then transcribes growing prefixes of the WAV at the app's draft cadence (0.8 s, then every 1.2 s) and the whole
file, reporting feature, encode, and decode time for each pass. Set `KIDI_FIT_AUDIO=0` or `KIDI_WARM_UP=0` to compare
against a padded 30-second window or a cold first request:

```sh
"$ADB" -s SERIAL shell "timeout 600 $REMOTE/runner dictation $REMOTE/models/whisper-small 4 2 $REMOTE/speech.wav" \
  > benchmarks/android/.cache/dictation.jsonl
```

On the SM8750 phone the 9.1-second speech fixture's final pass took about 3.2 seconds with the padded window, 1.05
seconds with fitted audio, and 0.84 seconds after the short-query decoder attention kernel (8.8 ms per token).

## Image Chat Check

Place a JPEG/PNG test image in the ignored cache and stage it on the device. The image mode asks for a short description,
then a color follow-up; the second turn must reuse KV and encoded image features. It uses the checkpoint's existing
vision weights and does not download another model:

```sh
"$ADB" -s SERIAL push benchmarks/android/.cache/photo.jpg "$REMOTE/photo.jpg"
"$ADB" -s SERIAL shell "timeout 150 $REMOTE/runner image $REMOTE/models/gemma4 4 1 $REMOTE/photo.jpg" \
  > benchmarks/android/.cache/image-chat.jsonl
```

`image` is an end-to-end check: it loads the full Gemma generator, encodes the image, prefills visual/text tokens,
decodes an answer, and runs a cached follow-up. To optimize the vision frontend without LM load, prefill, decode, or
prefix-cache noise, run the vision-only mode:

```sh
"$ADB" -s SERIAL shell "timeout 150 $REMOTE/runner vision $REMOTE/models/gemma4 4 3 $REMOTE/photo.jpg" \
  > benchmarks/android/.cache/vision-only.jsonl
```

It reports image preparation, vision checkpoint binding, and each full frontend pass (patch projection, all 16 encoder
blocks, 3x3 spatial pooling, and the projection to Gemma's text width). Iteration zero is marked as warm-up. One model
load can benchmark several images:

```sh
"$ADB" -s SERIAL shell "timeout 300 $REMOTE/runner vision $REMOTE/models/gemma4 4 3 \
  $REMOTE/vision-suite/square-grid.png \
  $REMOTE/vision-suite/landscape-shapes.png \
  $REMOTE/vision-suite/portrait-shapes.png"
```

Generate those deterministic images and element-wise reference tensors with the official Google checkpoint and
Transformers implementation. The script pins its Python dependencies and the exact checkpoint revision:

```sh
uv run --script benchmarks/android/gemma4_vision_reference.py \
  --output benchmarks/android/.cache/vision-suite
```

Each Safetensors reference contains the unpadded input patches and position IDs, all 16 unpooled block outputs, the
first configured layers' internal stages, compact axial RoPE angles, the final unpooled encoder output, pooled 768-wide
tokens, and projected 1536-wide visual embeddings. `--stage-layers N` controls the diagnostic depth.
`--verify-determinism` reruns every captured tensor with the thread counts from `--determinism-threads` and fails on
any elementwise change. Push the images and references to the phone, then set `KIDI_VISION_REFERENCE_DIR` for
element-wise preprocessing and final-output metrics:

The module hooks are registered through [`torch_intercept.py`](./torch_intercept.py), a reusable context manager
adapted from the earlier forward-interception script. Model-specific code only declares which module inputs and
outputs receive stable tensor names.

```sh
"$ADB" -s SERIAL shell "mkdir -p $REMOTE/vision-suite/images $REMOTE/vision-suite/references"
for file in benchmarks/android/.cache/vision-suite/images/*.png; do
  "$ADB" -s SERIAL push "$file" "$REMOTE/vision-suite/images/"
done
for file in benchmarks/android/.cache/vision-suite/references/*.safetensors; do
  "$ADB" -s SERIAL push "$file" "$REMOTE/vision-suite/references/"
done
"$ADB" -s SERIAL shell "KIDI_VISION_REFERENCE_DIR=$REMOTE/vision-suite/references timeout 300 \
  $REMOTE/runner vision $REMOTE/models/gemma4 4 3 \
  $REMOTE/vision-suite/images/square-grid.png \
  $REMOTE/vision-suite/images/landscape-shapes.png \
  $REMOTE/vision-suite/images/portrait-shapes.png"
```

Set `KIDI_VISION_USE_REFERENCE_INPUT=1` as well to feed the saved official patch tensor into Kidi. This isolates model
inference correctness from decoder/resizer differences; without it, the benchmark measures the real Kidi preprocessing
path and reports its patch error separately. When references are enabled, the timed passes compare final projected
embeddings and one additional untimed diagnostic pass reports element-wise error after every encoder block.
`KIDI_VISION_STAGE_LAYERS` changes the number of layers reported by that diagnostic pass (0-16).
`KIDI_VISION_DUMP_DIR` writes matching Kidi tensors to Safetensors for offline analysis; existing files are never
overwritten.

On macOS, set `KIDI_ACCELERATOR=gpu` to run the standalone tower on Apple GPU. The native W8 path quantizes each
shared activation once, keeps activation and weight codes integral through 64x64 tiled projections, and applies scales
only after accumulation. Q/K/V reuse one quantized activation, Q/K use fused axial RMSNorm/RoPE, and large equal-head
attention uses MPSGraph's fused SDPA operation with the checkpoint's scale of 1.0. A combined Q/K/V dispatch was
removed after measuring slower than the three projections with shared quantization. At checkpoint binding, vision gate
and up rows are validated and combined into the existing gated-FFN representation. The FFN computes both halves
together, applies GELU/multiply on-chip, and writes the down projection's INT8 input directly; see the illustrated
[FFN fusion walkthrough](../../ffn-fuse.md):

```sh
KIDI_ACCELERATOR=gpu KIDI_VISION_PRECISION=checkpoint \
  build-release/kidi_android_baseline vision MODEL 4 3 image.png
```

On an Apple M5, the three 2,304-2,376-patch fixtures take about 0.38-0.42 s after warm-up. The fixed photo produces the
same CPU and Metal answers (`A cat.` and `Orange/Ginger.`); the deterministic shape fixtures preserve the same objects
and colors with minor wording differences. Intermediate features are not numerically identical—the residual difference
can compound through the sensitive tower—so Metal quality is gated by short-answer equivalence rather than FP32 tensor
parity.

The committed pre-optimization path measured about 0.67-0.70 s. Profiling attributed most of it to 112 packed
projections per image: each output tile repeated FP32 activation rounding, and the 32x32 tile reread activations and
weights more often. Captured-step replay can skip host-side operator lookup on later identical calls, but it does not
fuse these Metal kernels and cannot reduce first-image latency; the backend fusions above address the measured GPU work
directly.

The packed-kernel benchmark has a vision-shape mode for the 768-wide attention projections and 3,072-wide MLP:

```sh
build-release/kidi_metal_packed_bench vision
build-release/kidi_metal_packed_bench vision-ffn
```

`KIDI_VISION_PRECISION` selects the projection compute policy. The default, `checkpoint`, uses the checkpoint's native
precision. `fp32` disables activation quantization and dequantizes packed weights for FP32 computation; `bf16` uses
BF16 activations and computation. `q2a16`, `q4a16`, and `q8a16` explicitly requantize weights to that width before BF16
compute. Their `q*ae4m3` and `q*ae5m2` variants round activations through the selected FP8 format and use FP32
accumulation; `i8a8` is an alias for `q8ae4m3`. `qat-fp32` is the deliberately slow numerical verification policy: it
preserves trained activation SRQ while dequantizing each layer's checkpoint weights for FP32 accumulation. On Apple
CPU it also mirrors the pinned PyTorch build's RMS reduction, SLEEF RoPE math, and softmax reduction; it is an oracle,
not a performance mode. `lowbit-parity` retains packed weights, dequantizes only 192 output channels at a time into
scratch, and uses the same parity math; it currently requires Apple Accelerate. Unsupported backend combinations never
fall back silently. For example:

```sh
KIDI_VISION_PRECISION=qat-fp32 "$REMOTE/runner" vision "$REMOTE/models/gemma4" 4 3 image.png
```

Compare matching official and Kidi dumps, then attribute projection differences to local computation versus propagated
inputs:

```sh
uv run --script benchmarks/android/gemma4_vision_diagnose.py \
  --official benchmarks/android/.cache/vision-suite/references/square-grid.safetensors \
  --kidi /path/to/kidi-dump/square-grid.safetensors \
  --checkpoint benchmarks/android/.cache/gemma4-official \
  --projection 0:q_proj --projection 0:gate_proj
```

For the pinned Apple/PyTorch oracle, `qat-fp32` produces elementwise-identical outputs after all 16 encoder blocks on
the square, landscape, and portrait fixtures. Final projected embeddings remain within roughly `2.4e-7` relative RMSE
because pooling/projector reductions are not part of the parity path. The native `checkpoint` mode intentionally
retains its much faster integer projection kernels and is not expected to be bit-identical to dequantized PyTorch.
The original drift was seeded by different RMS mean reduction and RoPE/softmax arithmetic; sparse one-quantum SRQ
changes were then amplified by large learned norm scales and the gated MLP. The parity modes support PyTorch-compatible
small-range RoPE angles (absolute angle below 125), which covers this three-image suite.

On an Apple M5 CPU with four threads, `lowbit-parity` keeps all 16 block outputs elementwise identical on all three
fixtures and final embedding relative RMSE below `2.4e-7`. Steady-state encoding is about 1.72-1.76 s for 2,304-2,376
patches, versus about 1.25 s for native `checkpoint` and 3.96 s for full `qat-fp32`. Square-image peak RSS is about
0.89 GiB rather than about 2.03 GiB for cached FP32 weights.

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

### Kidi HTP op package (fused RMSNorm and GELU*up)

`src/kidi/runtime/qnn/htp/KidiOpsPackage` builds the custom HTP operators that
keep Kidi's CPU fusions on the NPU. The DSP library needs the Hexagon SDK 6.3
toolchain (x86-64 Linux); the host library uses the NDK:

```sh
cd src/kidi/runtime/qnn/htp/KidiOpsPackage
make all sim-fused-test   # build/hexagon-v79/libQnnKidiOpsHtp.so + V79 simulator checks
make aarch64              # build/aarch64-android/libQnnKidiOps.so
```

Push `libQnnKidiOpsHtp.so` next to the V79 skel (on `ADSP_LIBRARY_PATH`) and
`libQnnKidiOps.so` on `LD_LIBRARY_PATH`, then add
`KIDI_QNN_OP_PACKAGE=libQnnKidiOps.so KIDI_QNN_OP_PACKAGE_HTP=libQnnKidiOpsHtp.so`
to the run. With a 384-token prompt and a 9216-token cache,
`kidi_npu_prefill MODEL_DIR 384 128 20` measured:

| Workload | Stock QNN lowering | KidiOps fused | Speedup |
| --- | ---: | ---: | ---: |
| Prefill, steady 128-token chunk | 179.4 ms | 63.7 ms | 2.8x |
| Decode | 28.3 tok/s | 33.2 tok/s | 1.17x |

All 20 greedy tokens matched the CPU run.
