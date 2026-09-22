# Kidi WebAssembly Demo

This directory builds a self-contained browser chat app for Gemma 4 E2B IT.
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
| `app.mjs`, `inference-worker.mjs`, `model-cache.mjs` | Browser | Chat UI, Wasm execution, Hub downloads and cache |
| `build.mjs`, `wasm-glue.mjs` | Node.js during build | Compile/package Wasm and fix large-memory generated glue |
| `src/web/libs/` | Build inputs | Pinned isolation helper, selected icons, and JavaScript parser dependencies |

Only browser assets and their licenses are copied to the deployment directory.
Do not deploy build scripts or the build-only parsers. Serving via Python does not replace the
JavaScript that runs in the browser. No Node server or model-export step is used.

## Model Source

The default source is Google's public upstream checkpoint pinned to revision
`dd693ff40353f057ca5f07e945ad867f4afbf2ec`. No local model, export, HF token, or
republished weights are required. The browser downloads the original Safetensors
file in 8 MiB HTTP ranges, caches those original bytes, and normalizes the packed
integer representation once in the final Wasm buffer. This is lossless, not
re-quantization. All checkpoint tensors, including vision/audio weights, are
preserved; inference currently supports text only.

Model source accepts a public Hub URL of the form
`https://huggingface.co/OWNER/REPO/resolve/COMMIT_SHA/config.json` for a compatible
single-file mobile-QAT checkpoint. Previously exported manifests remain readable
for compatibility, but no export tooling is shipped. Mutable Hub
revisions such as `main` are rejected so cached ranges cannot mix revisions.

## Execution Modes

- **WebAssembly CPU, one thread:** uses the single-thread SIMD module.
- **WebAssembly CPU, 2-8 threads:** uses pthreads and the YNNPACK/Slinky
  scheduler. The selected count includes the calling thread.

Four-thread Wasm CPU is the recommended mode on the tested system. Gemma's
model/layer equations and YNNPACK operators are shared with native CPU execution.

## Browser Workflow

The main screen reserves the viewport for messages and the bottom composer.
Model source, thread count, cache and token controls live in the runtime settings dialog.
New chat creates a blank conversation; completed chats are stored in local
browser storage and can be reopened or deleted from the history rail. Chat
content is not sent to a server.

Output defaults to 1,024 tokens and can be set from 1 to 8,192. The formatted
conversation and requested output must fit within a shared 9,216-token context.
At the maximum output setting, 1,024 tokens remain for the formatted conversation.
The runtime reserves the full context cache for each generation, so even short
responses need the corresponding memory headroom.

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

Build to a fresh app-only directory and publish its contents as a Pages artifact:

```bash
node web/build.mjs build-pages
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
hostname or port starts a separate cache. The trash button clears both formats.

No checkpoint code or pickle data is executed. The browser reads YAML/JSON,
Safetensors, tokenizer data, and static Wasm/JavaScript assets.

## Validation

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
