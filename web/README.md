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
| `build.mjs`, `wasm-glue.mjs` | Node.js during build | Compile/package Wasm and fix large-memory generated glue |
| `src/web/libs/` | Build inputs | Pinned isolation helper, selected icons, and JavaScript parser dependencies |

Only browser assets and their licenses are copied to the deployment directory.
Do not deploy build scripts or the build-only parsers. Serving via Python does not replace the
JavaScript that runs in the browser. No Node server or model-export step is used.

## Model Source

The default chat model ID is `google/gemma-4-E2B-it-qat-mobile-transformers`. No local model, export, HF token, or
republished weights are required. Load resolves the Hub repository's current `main` revision to an immutable commit,
then the browser downloads the original Safetensors
file in 8 MiB HTTP ranges, caches those original bytes, and normalizes the packed
integer representation once in the final Wasm buffer. This is lossless, not
re-quantization. All checkpoint tensors, including vision/audio weights, are
preserved. Supported vision checkpoints accept images through the shared C++ core; audio
input remains the separate Whisper dictation workflow.

Settings accept a public `OWNER/REPO` Hub model ID. The resolved commit is cached separately from the mutable ID so
byte ranges from different revisions cannot mix. Automatic startup remains offline: it reuses the last resolved commit
only when every required file is cached. Previously exported manifests and pinned config URLs remain readable for
compatibility, but are not shown in settings.

Microphone dictation accepts `openai/whisper-tiny`, `openai/whisper-base`, or `openai/whisper-small`. The browser caches
the selected model's five upstream files without conversion or a generated manifest. Whisper runs in an on-demand CPU
Wasm worker with its own linear heap, including when Gemma uses WebGPU.

## Execution Modes

- **WebAssembly CPU, one thread:** uses the single-thread SIMD module.
- **WebAssembly CPU, 2-8 threads:** uses pthreads and the YNNPACK/Slinky
  scheduler. The selected count includes the calling thread.
- **WebGPU:** uses the hardware GPU through the single-thread WASM/JSPI bridge.
  It requires WebGPU and WebAssembly JSPI support. Captured decode steps reuse
  prepared operators and GPU output buffers; per-call inputs upload explicitly.

Calibrated packed WebGPU FFNs with aligned decode shapes use three dispatches:
input quantization, fused gate/up projection plus GELU-product/requantization,
and down projection. The fused middle stage writes INT8 activations directly.
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
These are shared C++ resource limits, not UI resizing rules. CPU WASM still has a 4 GiB heap ceiling and may
run out of memory with a full model and image input.

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

On startup, a fully cached model loads automatically. The last successfully
loaded source is remembered. Empty, partial, or unavailable caches leave the
model offline; automatic loading never downloads missing model data. Use Load
model to repair an incomplete or corrupt cache.

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

The E2B live Wasm heap crosses 2 GiB. Builds therefore use Emscripten's 64-bit
pointer compatibility lowering (`MEMORY64=2`), a 4 GiB maximum memory, and a
2 MiB stack. Pthread control blocks are reserved before the model occupies high
addresses. A generation can grow the Wasm heap to roughly 3.9 GiB; use a
64-bit current Chromium browser and a machine with ample available memory.
Long generations can still exhaust the 4 GiB limit as working buffers grow.
The direct loader accepts checkpoints over 2 GiB, subject to that total memory
budget. The build also fixes signed heap indexing in Emscripten 6.0.9's generated
`MEMORY64=2` JavaScript, including pthread heap-view wrappers. Without that fix,
mapping bookkeeping above 2 GiB can return an incorrect pointer. This transform
is limited to generated glue; native C++ and dependency sources are unchanged.

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

Hub ranges are cached by pinned URL and byte interval. A SHA-256 digest computed
at download time detects corruption on cache reuse; it is not an independent
upstream checksum. Initial download integrity relies on HTTPS and the pinned
revision. Previously exported manifests supply expected SHA-256 hashes for
their chunks. Reloads fetch only missing or corrupt parts. The app requests
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
browser. Disable HTTP caching when iterating on shader modules. Checks cover
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
