# Kidi WebAssembly Demo

This directory builds a self-contained browser chat app for Gemma 4 E2B IT with image input and Whisper microphone dictation.
It uses the released `google/gemma-4-E2B-it-qat-mobile-transformers` checkpoint,
keeps its trained mixed 2/4/8-bit values, and runs Kidi's existing C++ Gemma
model through WebAssembly.

## Build

Requirements:

- a recursive Kidi checkout;
- Emscripten 6.0.9 or later;
- CMake, Ninja, Python 3.10+, and Node.js.

```bash
brew install emscripten node
node web/build.mjs build-web
```

## Serve

Once built, only Python (or another static host) is needed:

```bash
make serve
```

Run from the repository root. This serves `build-web/` using Python's standard
HTTP server on loopback port 8080. Override settings with
`make serve PORT=8081 WEB_DIR=build-pages PYTHON=python3`.
Without Make, run `python -m http.server 8080 --bind 127.0.0.1 --directory build-web`.

Open `http://localhost:8080`. The first visit may reload automatically to enable
multithreading. Open Model settings and load the model. Node.js and Emscripten
are build-time dependencies; deployment needs only a static HTTP server.
The required libraries and icons are vendored in
[`src/web/libs/`](../src/web/libs/README.md), with versions and licenses recorded
there. No npm installation or dependency download is needed for the web build.

The JavaScript files have distinct roles:

| Files | Run in | Purpose |
|---|---|---|
| `app.mjs`, `inference-worker.mjs`, `asr-worker.mjs`, `model-cache.mjs`, `images.mjs` | Browser | Chat UI, local image attachments, isolated Gemma/Whisper execution, Hub downloads and cache |
| `build.mjs`, `wasm-glue.mjs` | Node.js during build | Compile/package the wasm32 variants and fix their generated heap indexing above 2 GiB |
| `src/web/libs/` | Build inputs | Pinned isolation helper, selected icons, and JavaScript parser dependencies |

Only browser assets and their licenses are copied to the deployment directory.
Do not deploy build scripts or the build-only parsers. Serving via Python does not replace the
JavaScript that runs in the browser. No Node server or model-export step is used.

## Model Source

The default chat model ID is `google/gemma-4-E2B-it-qat-mobile-transformers`. No local model, export, HF token, or
republished weights are required. Load resolves the Hub repository's current `main` revision to an immutable commit,
then the browser downloads the original Safetensors
file in 8 MiB HTTP ranges and normalizes the packed integer representation (a sign-bit flip) once, before caching
each range, so reloads copy weights without rewriting them. This is lossless, not re-quantization. Ranges cached by
earlier versions as upstream bytes are rewritten once on the next load; see
[Memory Layout](#memory-layout) for what enters the Wasm heap. Supported vision checkpoints accept images through the
shared C++ core; audio input remains the separate Whisper dictation workflow.

Settings accept a public `OWNER/REPO` Hub model ID. The resolved commit is cached separately from the mutable ID so
byte ranges from different revisions cannot mix. Automatic chat startup remains offline: it reuses the last resolved commit
only when every required file is cached. Previously exported manifests and pinned config URLs remain readable for
compatibility, but are not shown in settings.

Microphone dictation accepts `openai/whisper-tiny`, `openai/whisper-base`, or `openai/whisper-small` (default). Like
the Android app, the browser downloads config and tokenizer files from that repository and the weights from
whisper.cpp's Q8_0 GGML file of the same size (`ggerganov/whisper.cpp`, pinned revision): 264 MB for Small instead of
the 967 MB FP32 Safetensors. The GGML bytes are cached beside the model's own files, so the cache manager lists and
deletes them together; a previously cached FP32 checkpoint is removed. The selected speech model loads at startup,
downloading missing files as needed. Whisper runs in one reusable CPU Wasm worker with its own linear heap, including
when Gemma uses WebGPU. Model settings shows speech loading, ready,
or error status. Preloading does not request microphone permission; permission is requested only when recording starts.
Changing the speech model or CPU thread count replaces that worker; chat-backend changes and completed recordings retain it.

## Execution Modes

On first use, the app selects WebGPU when a hardware adapter is available; otherwise it uses CPU.
An explicitly selected backend is saved and takes precedence on later visits.

- **WebAssembly CPU, one thread:** uses the single-thread SIMD module.
- **WebAssembly CPU, 2-8 threads:** uses pthreads and the YNNPACK/Slinky
  scheduler. The selected count includes the calling thread.
- **WebGPU:** uses the hardware GPU from the single-thread module. It needs
  only WebGPU; without it the option is disabled with the reason in its
  tooltip. Captured decode steps reuse prepared operators and GPU output
  buffers; per-call inputs upload explicitly.

Wasm never waits for the GPU, so every browser uses the same protocol and no
JSPI or Asyncify is involved. C++ records and submits GPU work synchronously
(pipelines use `createComputePipeline`), and `synchronize()` only submits.
Values the host needs, such as the selected token, are copied back
asynchronously. The worker calls `kidi_step`, awaits the runtime's
`synchronize()`, and calls again; serving runs with
`ServingOptions::deferred_tokens`, so each step first accepts the token
selected by the previous one and then submits new work. The CPU variants use
the same loop. Image encoding has no mid-step reads: patch pooling runs on the
device and the position table stays on the host. Kernels that use
`dot4I8Packed` fall back to an equivalent WGSL function when a browser lacks the
`packed_4x8_integer_dot_product` extension. Recorded work is submitted every
128 dispatches, so the GPU starts on a step while the rest is still recorded.

Calibrated projections of one to three rows (decode) do not use `dot4I8Packed`,
which Apple GPUs emulate. Each workgroup quantizes the activation row while
staging it as floats, and the weights are decoded with one mask per value. The
sums are exact integers, so the outputs match the INT8 dot product exactly, and
no separate quantization dispatch is needed. Calibrated packed WebGPU FFNs with
aligned decode shapes use two dispatches: the fused gate/up projection plus
GELU-product/requantization, and the down projection. The first stage writes
INT8 activations directly.
Prefill keeps the existing projection path because the measured fused prefill
kernel was slower. Unsupported fusion shapes and uncalibrated models retain
their existing operators. The backend test page's **Benchmark FFN** control
compares both paths with exact output checks and GPU timestamps.
On the tested Apple GPU, the full-model short-prompt comparison improved warm
decode from 36.85 to 40.04 tokens/s with identical output tokens. A 145-token
prompt's single sample was unchanged; this is not a general throughput guarantee.

Gemma's model/layer equations are shared with native execution. CPU WASM uses
the native YNNPACK operators. Backend support is checked explicitly; WebGPU
does not silently fall back to CPU.

## Browser Workflow

The main screen reserves the viewport for messages and the bottom composer.
Model source, thread count, cache and token controls live in the runtime settings dialog.
Valid thread-count and output-token changes are remembered in local storage.
If isolation is unavailable, the current session uses one thread without replacing
the saved thread preference. The composer identifies inference as running locally
inside the browser.
New chat creates a blank conversation; completed chats are stored in local
browser storage and can be reopened or deleted from the history rail. Chat
content is not sent to a server.

### Images

With a supported Gemma vision checkpoint loaded, the plus button
attaches JPEG or PNG files. Paste and drop into the composer are also
supported. Previews can be removed before sending or opened at a larger size.
Messages may contain an image alone or text with images. Later turns retain
the conversation's images; missing locally cached data is reported explicitly.

The browser stores and transfers the original image bytes unchanged. It does
not resize, re-encode, or generate model features. The same C++ code used by
Android, Python, and the CLI decodes through Tahoma Vision, applies orientation
and pixel conversion, resizes for Gemma, and builds patches. The shared vision
tower produces image embeddings, which enter the normal text decoder.
Image features are reused for unchanged images and the same image-token budget.
The common default is 280 soft image tokens. Kidi configures Tahoma's existing
decode options with `max_pixels = 24'000'000` and a 96,000,000-byte decoded
limit, allowing four-channel input. JPEG and PNG enforce those limits before pixel allocation. The existing
16,384-pixel axis guard runs after decoding and before resize scratch allocation.
These are shared C++ resource limits, not UI resizing rules. The Wasm heap has a 4 GiB ceiling; a 2,469-token
request with a 280-token image peaked at 2.32 GiB.

**Image resize limit (pixels)** in Model settings defaults to 3,000,000 and can
be reduced to 161,280. It is saved locally and sent with each request; C++
validates the value and selects the largest supported token budget that fits
both the pixel ceiling and the requested token budget. The ceiling does not
force upscaling: the default 280-token budget uses at most 645,120 pixels.
For example, a 500,000-pixel ceiling reduces that budget to 140 tokens. Changed
limits invalidate reusable image features and image-dependent prefix state.

At most eight images and 32 MiB of original image data are allowed per
conversation. Image data lives in
a separate local Cache Storage cache, not in chat JSON or the model cache.
Deleting chats or removing drafts removes unreferenced attachments. Model-cache
deletion does not delete chat images. Temporary files in the WASM filesystem
are removed after request admission, including on errors. Images are never
uploaded to Hugging Face or another inference service. Browser storage eviction
and memory limits still apply.

The microphone button records at most 30 seconds, resamples captured mono PCM to 16 kHz, and transfers snapshots to the
isolated Whisper worker. Automatic language detection is enabled. While recording, replaceable draft hypotheses appear
in the composer; stopping runs a final pass that may refine them. The transcript is not sent to Gemma until submitted.

The first load converts the GGML weights to Kidi's per-channel INT8 checkpoint through the same preparation as native
builds, in the in-memory file system. The worker then stores that checkpoint in Cache Storage in place of the GGML
download (about the same size, 251 MB for Small), so later loads map it straight into the heap. Every load transcribes
a second of silence (`Transcriber::warm_up`) so the first recording does not pay weight packing. Whisper always encodes a 30-second window;
the browser instead encodes the recording plus at least one second of silence (`TranscriptionOptions::fit_audio`), in
2.56-second steps. If that output repeats itself, a known failure of shortened windows, it is decoded again over the
full window. On 43 LibriSpeech clean utterances (7.6 s on average) this matched the full window (Small 2.46% vs 2.58%
WER, Tiny 8.09% vs 8.68%) while encoding about 3.5 times faster. The encoder runs in 128-row slices, so its retained
buffers are small and shared by every recording length.

| Whisper Small, Node, Apple M5, 4 threads | Before (FP32, full window) | Now |
|---|---:|---:|
| Download | 967 MB | 267 MB |
| First load (convert + warm-up) | 0.56 s | 2.3 s |
| Reload (warm-up only) | 0.56 s | 0.75 s |
| Transcribe 6.6 s of speech | 3.1 s first, then 2.5 s | 0.63 s |
| Wasm heap after transcribing | 3.29 GiB | 0.66-0.76 GiB |

In Edge the app reports speech ready 1.5-2 s after opening with a cached model, and transcribes 3.9, 6.6 and 22.6
seconds of speech in 0.34, 0.63 and 2.2 s.

`benchmarks/web/asr_probe.mjs` runs the same loader and runtime in Node against a local Hugging Face cache.

Loading ends with a one-token warm-up (`Generator::warm_up`) so the first message
does not pay one-time work: on CPU, YNNPACK packs every projection weight on first
use; on WebGPU, kernels are prepared and host-resident weights such as `lm_head`
are uploaded. On an Apple M5 this moves about 1 s (one CPU thread) into loading;
the first token of a short prompt arrives in 0.53 s with four CPU threads (was
1.56 s) and WebGPU prefill takes 0.19 s.

Reloading a cached Gemma model, measured in Edge on an Apple M5 from worker start to ready:

| Backend | Before | Now |
|---|---:|---:|
| WebGPU | 4.0-4.3 s | 2.2-2.6 s |
| CPU, 4 threads | 6.1 s | 2.7-2.8 s |
| CPU, 1 thread | 6.3 s | 2.9 s |

Three changes produced this: cached ranges are no longer hashed again (0.93 s), packed weights are cached already
sign-flipped (0.86 s), and all variants use native Wasm exceptions. With Emscripten's JavaScript exception fallback, every
call that might throw went through a JavaScript trampoline, and the CPU builds parsed the 32 MB Gemma tokenizer in 1.9 s
instead of 0.5 s. What remains is reading 2.5 GB from Cache Storage (0.8-1.1 s), the tokenizer, and the warm-up, whose
YNNPACK weight packing exists only in memory and cannot be cached.

On startup, a fully cached chat model loads automatically. The last successfully
loaded source is remembered. Empty, partial, or unavailable caches leave the
chat model offline; automatic chat loading never downloads missing model data.
Use Load model to repair an incomplete or corrupt chat cache. Speech preloading
is independent and can download its selected model at startup.

Messages render Markdown, including lists, links, tables, and fenced code,
using vendored Marked and DOMPurify. Raw Markdown is kept in chat history and
model requests. Executable HTML, embedded media, and remote images are removed.
Long completed replies have Show more / Show less controls; active streams stay
expanded. Fenced code uses Highlight.js common languages, with plain-text fallback.
Mermaid fences render after completion using a lazy-loaded local bundle in a
sandbox with network access blocked. Diagrams retain a source disclosure and
fall back to source when invalid or too large to preview.

Each new assistant reply keeps a small token-count, elapsed-time, and decode-speed
footer in saved history. Cancelled partial replies keep their available stats.
Older saved replies without timing data do not show invented measurements.
During generation, the header shows live decode speed, output token count, and
elapsed time. Speed excludes prompt preparation; final summaries use the native
runtime counters. First-token latency is available in the footer tooltip.

The header shows allocated Wasm linear memory and growth headroom below the
4 GiB limit, refreshed during loading and generation. These are not process
RAM or free device memory, and headroom does not include reusable space already
inside the heap or guarantee that the browser can allocate more memory.

Output defaults to 1,024 tokens and can be set from 1 to 8,192. The formatted
conversation and requested output must fit within a shared 9,216-token context.
At the maximum output setting, 1,024 tokens remain for the formatted conversation.
The runtime reserves prompt plus requested output capacity in 128-token
buckets, capped by the context limit. It preserves the requested context's
cache precision policy. Short requests no longer reserve all 9,216 positions;
large output limits and conversations still require corresponding memory.

During generation, the send button becomes a square stop control. Stop is
cooperative at the next model-step boundary; a long prefill can therefore take
several seconds to yield. A partial assistant response is retained only if text
was emitted. Closing or navigating away from the page sends cancellation and
terminates the inference worker immediately.

## Memory Layout

All browser variants are wasm32 with a 4 GiB maximum linear memory, a 64 MiB
initial memory, and a 2 MiB stack. C++ keeps 64-bit pointers and `size_t`;
Emscripten lowers them to wasm32 (`MEMORY64=2`), so every current engine can
run the modules, including Safari without Memory64. Emscripten 6.0.9's
generated glue indexes heap views with signed shifts in this mode, which breaks
above 2 GiB; [`wasm-glue.mjs`](wasm-glue.mjs) rewrites only those heap indices
in the copied glue.

Gemma fits in that budget because only data the runtime reads stays resident:

- The loader writes a compacted Safetensors image into the heap: tensors start
  on 64-byte boundaries, the unused audio tower is skipped, and each layer's
  gate projection sits directly before its up projection, so the runtime views
  the fused gate/up matrix without copying it.
- The 1.09 GiB per-layer embedding table stays in JavaScript, in at most
  256 MiB row-aligned shards. `model.yaml` declares it under
  `external_tensors`; each step copies only the 4,480 bytes per token it reads.
  WebGPU uploads those rows instead of holding a 1.12 GiB table buffer.
- With WebGPU, the loader uploads the decoder's packed projection weights
  (0.63 GiB) straight into GPU buffers, masked and interleaved into the
  kernels' layout in JavaScript, gate and up fused into one buffer. C++ adopts
  them through the same `external_tensors` declarations, so they never enter
  the heap and the GPU holds one copy.
- Wasm builds skip the CPU fused gate/up/down kernel, whose transposed weight
  tiles were a second FFN copy, and run single-row calibrated projections
  through the same INT8 dot as prefill. YNNPACK then packs each projection
  once for both. Prefill and decode speed were unchanged in measurement.

| 2,211 prompt + 32 output tokens, CPU | Peak heap |
|---|---:|
| Previous wasm64 build | 4.38 GiB |
| Compacted heap and external per-layer table | 2.84 GiB |
| Plus shared packing and no fused CPU FFN (current) | 2.23 GiB |

| WebGPU, Chromium on an Apple GPU, 296-token prompt | Peak heap | GPU buffers | Prefill / decode tok/s |
|---|---:|---:|---:|
| JSPI build, weights uploaded from the heap | 1.44 GiB | 1.53 GiB | 58 / 11.2 |
| Weights streamed to GPU, no JSPI (current) | 0.82 GiB | 0.95 GiB | 84 / 14.8 |

At 2,211 prompt tokens the WebGPU heap stays at 0.82 GiB with 1.04 GiB of GPU
buffers. Safari 26.6, which has no JSPI, ran the same WebGPU build at 74 / 16
tokens/s and described an attached photo correctly. The CPU table is Node
measurements of the same modules. Safari 26.6 ran
the current wasm32 build on the same prompt with the same 32 token IDs: one
thread at 18.8 prompt / 4.4 decode tokens/s and four threads at 52.7 / 11.8,
both peaking at 2.24 GiB. Native wasm64 parity and an image request are in the
[optimization journal](../benchmarks/gemma4/JOURNAL.md).

### Memory diagnostics

Model settings has a **Memory diagnostics** section, updated during loading and
generation: heap size and limit, bytes allocated and free inside the allocator,
its peak footprint, weights inside and outside the heap, KV cache, reusable
image features, and WebGPU buffers. Allocator statistics walk the heap, so they
refresh when a phase ends (load, reply, cancel or error); heap and GPU sizes stay
live. Below them, allocations are grouped by the work that made them: tokenizer,
weight binding and vision weights at load. These are net allocations for that
work and do not shrink when workspaces are later released. **Copy diagnostics**
copies the JSON for bug reports.

To measure without a browser, run the same loader and module in Node against a
local Hugging Face snapshot:

```bash
node benchmarks/web/heap_probe.mjs --glue build-web/single/kidi.mjs \
  --snapshot ~/.cache/kidi/model-hub/models--google--gemma-4-E2B-it-qat-mobile-transformers/snapshots/<sha> \
  [--threads 4] [--words 1600] [--image photo.jpg] [--json result.json]
```

The heap ceiling is not an upfront allocation or a guarantee of available RAM;
the per-layer table also uses about 1.1 GiB of browser memory outside the heap.
Long requests and simultaneous model workers can still exhaust memory.
Allocation wrappers use JavaScript numbers, and `ccall` pointer arguments use
its `pointer` type. Check the built ABI with an allocation that crosses 2 GiB:

```bash
KIDI_TEST_WASM=build-web/single/kidi.mjs node --test tests/web/model_cache_test.mjs
```

## Hosting and Cache

Pthreads require a secure, cross-origin-isolated page. The bundled, pinned
[`coi-serviceworker`](https://github.com/gzuidhof/coi-serviceworker) adds the
isolation headers through a service worker on servers that cannot configure
them, including Python's standard server and GitHub Pages. It must remain beside
the index page on the same origin. First installation can trigger a page reload;
subsequent visits reuse it. Browsers that cannot establish isolation fall back
to one thread. HTTPS is required except on localhost/loopback.

### GitHub Pages

The [WebAssembly Pages workflow](../.github/workflows/pages.yml) tests the browser
loader and builds the single-thread, pthread, and WebGPU variants for relevant pull requests and pushes to `main`.
Only `main` deploys, using the `github-pages` environment and GitHub's Pages
artifact service. It can also be run manually from the Actions tab on `main`.
The Linux build uses Emscripten 6.0.9 and Node.js 24; npm is not required.

The workflow caches the versioned Emscripten SDK (including system libraries)
and up to 500 MB of compiler objects through `ccache`. The first build is cold;
later runs reuse matching objects, but still configure and link all three variants.
All runtime sources remain included. Python bindings, native tests, and
benchmarks are disabled for the Pages build. Native unit tests and Python wheel
checks run independently in the [native workflow](../.github/workflows/native.yml).
Python-only and top-level documentation changes do not trigger Pages; stale PR
builds are cancelled, while deployments are allowed to finish.

One-time repository setup:

1. Open [Settings > Pages](https://github.com/thammegowda/kidi/settings/pages).
2. Set **Build and deployment > Source** to **GitHub Actions**.
3. Push the workflow to `main`, or run **WebAssembly Pages** from Actions.

After a successful deployment, the app is served at
[thammegowda.github.io/kidi/](https://thammegowda.github.io/kidi/).
No access token or Hugging Face secret needs to be configured.

To build the same app-only artifact locally:

```bash
make wasm WEB_DIR=build-pages
```

Publish `build-pages/`, not source, build tools, or an old `build-web/model/`
directory. The app uses relative asset and worker URLs, so repository subpaths
such as `https://OWNER.github.io/REPO/` work. The service worker is scoped to that
subpath. The checkpoint is fetched directly from HF, not stored on Pages. HF and
its redirected CDN must permit CORS and byte-range requests; the loader checks
for an exact `206 Content-Range` response rather than accepting an accidental
full-file download. No secret should be embedded in the static app.

Servers with configurable headers can instead supply:

```text
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
Cross-Origin-Resource-Policy: same-origin
```

Hub ranges are cached by pinned URL and byte interval. Download integrity relies on HTTPS and the pinned revision.
Reuse checks each cached range's size, which catches truncated entries but not altered bytes: hashing every range again
cost about 1 s of each Gemma reload. Previously exported manifests supply expected SHA-256 hashes, checked when their
chunks are downloaded. Reloads fetch only missing or truncated parts. The app requests
persistent storage and shows cached versus downloaded bytes; browser quota and
eviction policy still apply. Cache Storage belongs to the app origin, so changing
hostname or port starts a separate cache. Model settings lists each cached Hub
revision, its files and sizes, and whether the download is complete. Delete
removes one Hub model; Clear all also removes legacy manifest caches. Deleting a
loaded model does not interrupt the current runtime, but the next load downloads
it again.

No checkpoint code or pickle data is executed. The browser reads YAML/JSON,
Safetensors, tokenizer data, and static Wasm/JavaScript assets.

## Validation

### Current Backend Checks

Run `make web-test` for loader, cache, generated-glue, and speech-resampling checks.
After `make wasm`, explicitly rebuild the C++ WebGPU check; the app build disables
test targets and an older test binary can otherwise remain on disk:

```bash
emcmake cmake -S . -B build-webgpu -DKIDI_BUILD_TESTS=ON \
  -DKIDI_WASM_WEBGPU=ON -DKIDI_WASM_THREADS=OFF
cmake --build build-webgpu --target kidi_webgpu_test -j8
python3 -m http.server 8081 --bind 127.0.0.1 --directory .
```

Open `http://127.0.0.1:8081/tests/web/backend_test.html?run=1` in the integrated
browser. Unlike the app, this developer test links JSPI (so it needs a browser
with JSPI, such as current Chromium): its checks read GPU results inline through
a test-only `set_web_gpu_wait` hook. Disable HTTP caching when iterating on shader modules. Checks cover
packed projections, attention, float/QAT Gemma references, replay rebinding,
token readback, cache scatter aliasing and errors, and buffer lifetime/budgets.
Full-model observations and performance limitations are recorded in the
[optimization journal](../benchmarks/gemma4/JOURNAL.md).

### Historical CPU Measurements

Validated on 2026-09-21 with Emscripten 6.0.9, Chromium 148, and an Apple M5
with 16 GiB RAM. The prompt was `What is the capital of France? Answer briefly.`
with 19 prompt tokens and a 24-token output limit.

| Mode | Result | Total generation | Decode |
|---|---|---:|---:|
| Wasm CPU, 1 thread | `The capital of France is Paris.` | 4.76 s | 5.6 tok/s |
| Wasm CPU, 4 threads | `The capital of France is Paris.` | 2.62 s | 14.3 tok/s |

CPU emitted exactly the native Kidi token IDs. Native Kidi's 16 CTest cases also
pass after these changes.

The direct upstream path was additionally tested with an unmodified Python
server at a nested URL: service-worker isolation, one/four-thread inference, the
full 2,458,111,846-byte weight file, and cached reloads with zero downloaded
bytes in 4.9-5.6 seconds. `Count from one to five in words.` produced the expected eight IDs
`4906,236764,1156,236764,1806,236764,2390,236764`. GitHub Pages itself has not
been published in this session. Loader and generated-glue regression checks:

```bash
node --test tests/web/model_cache_test.mjs
```

### Scheduling follow-up

WebAssembly now compiles row-one packed projections of 512 KiB or less with a
single-thread YNNPACK scheduler. Larger attention, MLP, and vocabulary matrices
still use the selected CPU thread pool. This avoids dispatching several workers
for projections whose work is smaller than the scheduling overhead.

On the same browser session, four-thread CPU with the fixed prompt
`Name three practical uses of binary search.` and a 32-token limit improved
from 10.4 s to 8.85 s total (15%) and from 3.4 to 4.1 decode tok/s. The 32 token
IDs were identical. This is a focused warm comparison, not a cross-browser
performance claim.
