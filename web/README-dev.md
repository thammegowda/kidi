# Kidi Browser Developer Guide

For a quick introduction and usage instructions, see the [browser README](README.md).
This guide covers building, hosting, testing, and the browser runtime internals.

This directory builds the WASM host for Kidi's shared multi-view interface with
Gemma chat and vision, Whisper microphone dictation, and OmniVoice read-aloud
and continuous voice mode.
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
node web/tools/build.mjs build-web
```

The packaged site contains two runtimes: the primary non-pthread WebGPU module
and a pthread CPU fallback. Extension packages additionally include a
single-thread CPU fallback for browsers that cannot isolate extension pages.

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
While loading, the workspace header shows progress even with settings closed.
Browser runtime settings show the progress bar, loaded/total bytes, separate
download and cache byte counts, and the current file. Model initialization
uses an indeterminate indicator rather than a misleading completed download bar.
Cached automatic restores report the same progress as manual loads.
The required libraries and icons are vendored in
[`web/libs/`](./libs/README.md), with versions and licenses recorded
there. No npm installation or dependency download is needed for the web build.

On phones, the 52px header keeps Settings in view and combines the input/output
language pickers into one compact control. Matching selections show one name
(`Auto`, `English`); different selections show `Kannada → English`. Tap it to
search either language; tap elsewhere or press Escape to collapse it. Phone
screens use icon-only Chat and Reader tabs, retaining their accessible labels.
The far-left panel toggle opens/closes conversation navigation on phones and
collapses/expands it on desktop. The drawer opens below the header so the toggle
remains accessible. Secondary backend telemetry remains in the desktop header,
while runtime status appears beside the title in the slim conversation bar.
Model setup uses compact `Load` actions with component-specific accessible names.
The disclaimer sits immediately above the bottom-anchored message composer;
the message list resizes when the draft grows so responses are not covered.

### Deployment regression tests

`make web-test` runs the browser unit tests, including the real site packager
with small runtime fixtures. The packaging test checks local module imports,
re-exports, literal dynamic imports, and worker/asset URLs in the generated
application, and links the packaged LiteApp entry point. LiteApp's complete
browser module tree is shipped rather than a hand-maintained list of imports.

To test a built site in a real browser:

```bash
npm ci --prefix web --ignore-scripts --no-audit --no-fund
npm --prefix web exec -- playwright install chromium
make wasm WEB_DIR=build-pages
make web-smoke-test WEB_DIR=build-pages
```

The smoke test serves only the generated artifact under `/kidi/`, without
COOP/COEP server headers, using a fresh Chromium context. It verifies the
isolation service worker, backend initialization, settings and view navigation,
the shared runtime worker's request/response transport, and native configuration
and memory calls in both actual WebAssembly modules
(including CPU pthread startup). JavaScript errors, failed assets, and external
requests fail the test. It does not download model weights or require a GPU;
model inference and hardware WebGPU execution remain separate tests.

The same check emulates Android touch input at 320, 360, 393, and 412 CSS pixels,
plus landscape and desktop breakpoints. It checks actual element bounds and
hit targets, sidebar toggling, language-panel search/dismissal/persistence,
compact model-load button sizes, and disclaimer/composer ordering with short
and multiline drafts. Matching names and differing language pairs must fit;
Settings must remain visible and tappable.

The Pages workflow runs both test layers on pull requests and before uploading
the exact tested artifact for deployment. Source-only tests and checking that
the HTML and Wasm files exist do not establish that the deployed app can start.

The JavaScript files have distinct roles:

| Files | Run in | Purpose |
|---|---|---|
| `web/ui/` | Browser | Shared multi-view Kidi shell, Chat UI, Markdown, images, voice controls, settings, and persistence |
| `backend.js`, `runtime-client.mjs` | Browser main thread | Shared UI adapter and logical GPU model leases |
| `gpu-runtime-worker.mjs`, `runtime-scheduler.mjs`, `model-residency.mjs`, `scoped-model.mjs` | One GPU worker | One device/Wasm heap, isolated native model sessions, priority scheduling, scoped files/weights, and residency |
| `inference-worker.mjs`, `asr-worker.mjs`, `tts-worker.mjs`, `model-cache.mjs` | Workers | CPU execution, standalone benchmark/reference execution, Hub downloads, and persistent model cache |
| `build.mjs`, `package-site.mjs`, `wasm-glue.mjs` | Node.js during build | Compile/package the wasm32 variants and fix their generated heap indexing above 2 GiB |
| `web/libs/` | Build inputs | Pinned isolation helper, selected icons, and JavaScript parser dependencies |

Only browser assets and their licenses are copied to the deployment directory.
Do not deploy build scripts or the build-only parsers. Serving via Python does not replace the
JavaScript that runs in the browser. No Node server or model-export step is used.

## Browser extensions

Kidi packages a chat-first writing assistant as Manifest V3 extensions for Chromium browsers
and Firefox:

```bash
make extension
```

The build compiles WebGPU, pthread CPU, and single-thread CPU Wasm runtimes,
then produces:

```text
build-extension/chromium/
build-extension/firefox/
dist/kidi-chromium.zip
dist/kidi-firefox.zip
```

Chrome and Edge use the Chromium package. Load
`build-extension/chromium/` through the browser's extension developer page.
For Firefox development, open `about:debugging`, choose **This Firefox**, and
load `build-extension/firefox/manifest.json` as a temporary add-on.

The panel keeps a conversation in memory while it is open. Type a message and
press **Send** (Enter; Shift+Enter inserts a newline). **New chat** clears the
conversation and detaches source context without unloading the model. Follow-up
instructions reuse completed user/assistant turns; conversations that exceed
the turn, byte, or model context budget fail explicitly rather than silently
dropping messages.

The extension also adds **Polish with Kidi**, **Explain with Kidi**, and
**Summarize with Kidi** to the selection context menu. Chromium opens a side
panel; Firefox opens a sidebar. Editable input, textarea, and contenteditable
selections can be replaced only while the captured text still matches. Other
results remain available to copy. The compact **Prompt** control exposes the
configured system prompt for each action. Edits are saved automatically per
action in extension-local storage; **Default** removes the override and restores
the shared YAML prompt. All defaults are under the top-level `prompts` map:
`system` supplies the shared chat system prompt, `translate` supplies reader
translation, and keys such as `polish`, `email_reply`, and `page_summarize`
supply action prompts. `runtime.wasm.prompts` may override individual defaults.
The old `models.chat.system_prompt`, `translation_prompt`, and `editor.prompts`
locations must be moved to this map; they are rejected rather than silently
ignored. Existing locally customized prompts are retained.
The action selector lives inside the prompt inspector,
not in the main chat UI. Prompt overrides remain separate for chat, email
composition, email polishing, email replies, and the existing text actions.

### Gmail writing (M1)

1. Open `https://mail.google.com/` and open the Kidi panel.
2. Click **Enable Gmail** and grant the optional permission for that origin.
   No Gmail OAuth or blanket access to other websites is requested.
3. Open a compose window or an email. Suggestions update as Gmail changes.
4. For a compose window, choose **Polish draft** or **Compose from notes**.
   With multiple drafts, choose the draft by subject first. A selected passage
   is used instead of the complete draft when it is inside that editor.
5. For an opened email, choose **Draft a reply**, optionally entering guidance
   in the chat composer first.
6. Review and refine the result in chat. Click **Replace draft** or
   **Insert into reply** to apply it, or **Copy** to use it elsewhere.

The Gmail bridge observes only draft/email availability until an action
explicitly captures content. It extracts the latest visible opened message,
not inbox snippets, hidden messages, or the complete thread. Use **View attached
text** to inspect exactly what is sent to the local model.

Insertion is plain text and preserves signature and quoted-history nodes.
It targets the captured compose window, validates the original document URL
and draft content, and refuses stale edits. To insert a suggested reply, open
an empty inline reply in the same Gmail thread; existing reply text is never
overwritten. Kidi never clicks Send, creates recipients, or changes a subject.
After applying a result, recapture the draft before another replacement.

Source snapshots live in extension session storage (or volatile memory when
that API is unavailable), expire after 30 minutes, and are bounded to 16
contexts. Email text and conversations are not saved in local storage.
Revoking Gmail permission unregisters its content script and rejects further
captures or insertions. Browser reloads can require recapturing a draft.
Gmail's DOM is not a public editing API; retain manual Gmail smoke coverage
for autosave, inline replies, and multiple-account layouts when Gmail changes.

Once the model is ready, a compact diagnostics strip reports the active backend,
host RAM, WebGPU buffer memory, live decode speed, output-token count, and
time-to-first-token. Memory values come from the runtime allocator and WebGPU
buffer accounting rather than browser-wide estimates. **GPU mem** is allocated
buffer memory, not a utilization percentage; browsers do not expose portable
GPU-utilization telemetry. Assistant results use the shared vendored
Marked/DOMPurify renderer for sanitized streaming Markdown, links, lists,
tables, and fenced code.

The extension contains executable JavaScript and Wasm locally. Production
configuration downloads only model data from Hugging Face; the explicit
development bridge described below can instead populate Cache Storage from the
native loopback cache. The Chromium manifest opts into
cross-origin isolation for pthread CPU inference. Firefox prefers WebGPU and
uses the packaged single-thread CPU runtime when its extension page is not
isolated.

Run the installed Chromium smoke test after packaging:

```bash
make extension-smoke-test
```

The smoke test loads the unpacked extension in Chromium and verifies its
service worker, chat page, extension origin, cross-origin isolation, prompt
persistence/reset, and the actual shared WebGPU worker's editor/document service routes.
It does not download the model.

Additional Gmail/chat checks:

```bash
make extension-gmail-test
```

The Gmail test uses an isolated temporary extension and a synthetic mailbox;
it never opens a real account or downloads model data. Its backend fixture
tests UI transport, not model quality. The separate CPU Wasm test loads the
small checked-in Gemma numerical fixture and runs real inference for
chat, email composition/polishing/replies, conversational history, and document
processing. A controlled in-memory variant emits a word followed by EOS to
verify ordered translation, rolling summary replacement, changing native
request IDs, input/output token budgets, and output-limit retries. Tests
also cover optional permissions, multiple windows/drafts, stale captures,
protected signatures/quotes, and explicit insertion.

### Articles, translation, and read aloud (M2/M3)

1. Open an HTTP/HTTPS article and invoke Kidi using its toolbar icon. The panel
   offers **Summarize page**, **Translate page**, and **Read article** outside
   Gmail. You can also invoke these actions from the page's right-click menu.
2. **Summarize page** extracts the article and produces a summary. Readability
   runs locally against a document clone with navigation, hidden content, and
   forms removed. A selection takes precedence over the full article.
3. **Translate page** attaches the source and opens a language picker. Choose
   the target, then click **Translate**. Its preference is saved locally.
4. Expand **View attached text** to review the exact read-only source. Only the
   first 64 KiB is retained for unusually long pages, with a persistent warning.
   Kidi translates the extracted text in the panel, not page navigation,
   advertisements, or an in-place rewrite of the website.
5. Use **Read article** for the source, or **Read aloud** under a result. On first
   use, approve the separate local voice model download/setup. No voice files
   are fetched before approval. Cached files and voice calibration are reused.
   Unknown source languages require an explicit narration-language choice;
   supported pronunciation/translation quality depends on the installed models.
6. Use **Pause**/**Resume** and **Stop audio** in the compact playback strip.
   Stopping discards buffered audio immediately and cancels pending synthesis.
   Closing or clearing the conversation stops narration.

Original article narration does not load or invoke the chat model, including
when chat files are already cached. The extension loads chat only on explicit
**Load** or a text-generation action; other web hosts keep automatic cache
restore. Starting
from a right-click **Read article** request attaches the source first; click
the panel's reading action to activate audio. Voice loading shows real byte
progress, initialization/calibration state, and synthesis steps when available.
The player remains active until queued audio is actually played, not merely
until synthesis finishes.

No blanket website permission is requested. `activeTab` access comes from
invoking the toolbar/context menu; a panel left open while browsing does not
automatically gain access to a new origin. On access errors, invoke Kidi on
that page again. Chromium/Edge use the explicit action-click handler to open
the side panel; do not enable `openPanelOnActionClick`, because Edge can open
the panel through that behavior without retaining the `activeTab` grant needed
by a later page action. Browser-internal pages, PDFs without accessible page text,
and articles that Readability cannot identify may require selecting/pasting
text. Mailbox-page extraction is explicitly refused; use Gmail writing actions.

The C++ document pipeline processes every captured section rather than
silently dropping text beyond one model window. Translation preserves section
and paragraph order, publishes completed sections, and retries output-limited
sections in smaller pieces. Summaries maintain one cumulative result: each
4096-token input combines the previous summary with the next source segment,
and the model returns a revised summary of at most 1024 tokens. The panel
replaces its visible summary after every segment, so useful output appears
before the document finishes. Segment progress accompanies generation, and
token/throughput statistics aggregate the underlying inference calls.

After a summary, follow-up chat explicitly uses that generated summary instead
of the complete article; the source stays attached for another transformation.
Without a summary, normal context-budget errors remain explicit. Summary and
translation have separate editable/default-reset prompts in **Prompt**.

Run the installed-package article checks:

```bash
make extension-article-test
```

These checks serve only synthetic articles and use a fixture backend/PCM, not
pretrained language/voice models. They verify source filtering and Unicode
limits, summary follow-up context, target-language routing, progressive results,
download approval, narration controls, and cancellation. Unit tests also
exercise bounded playback, late events, and drain-before-completion. Live model
quality/performance, real Gmail autosave, and Firefox runtime behavior still
need device/browser smoke testing; synthetic output is not a quality benchmark.

For a network-dependent Microsoft Edge check against the current Wikipedia
article, run:

```bash
make extension-edge-e2e-test
```

This first runs every deterministic installed-extension test in Edge, then
copies the packaged extension unchanged to a temporary Edge profile, uses the
DevTools Protocol `Extensions.triggerAction` command to invoke its real toolbar
action and grant `activeTab`, verifies Readability extraction with the
shipped background/content scripts, and exercises the actual side-panel
Summarize button with the deterministic backend fixture, so it never downloads
Gemma. Temporary profiles and copies are deleted. Set `KIDI_E2E_KEEP_OPEN=1`
to keep the one successful test window available for inspection, and
`KIDI_E2E_STEP_DELAY_MS=1500` to make each visible transition easier to follow.
The live portion detects upstream page/DOM and toolbar-permission regressions
but is intentionally outside `web-test` because it requires Edge and network
access.

For manual testing with the real backend and a reusable model cache, run:

```bash
make extension-edge-open
```

This opens (or reuses) exactly one isolated profile at
`~/.cache/kidi/edge-extension-profile`. Unlike automated test profiles, it is
not deleted, so the extension's Cache Storage, model ranges, voice calibration,
prompt overrides, and other browser-local settings survive Edge restarts. The
script reconnects to an already-running profile instead of opening duplicates.
It loads the real package from `build-extension/chromium`; deterministic E2E
tests continue to use disposable fixture profiles and never alter this cache.

To fill that browser cache from Kidi's existing native Hub cache instead of
downloading model bytes from Hugging Face, use:

```bash
make extension-edge-open-local
```

The launcher starts a loopback-only bridge on `127.0.0.1:8765`, stores that
development override in this extension's local settings, and reuses the same
stable Edge profile. The bridge exposes only validated cached Hub revisions and
supports the exact byte-range/CORS/CORP contract used by the browser loader.
Model bytes remain on the machine; the normal loader still performs its layout
normalization and writes the browser's persistent Cache Storage once. Later
loads use the browser cache without the bridge. Running
`make extension-edge-open` removes the override and returns the profile to the
configured HTTPS Hub. Shipped ZIPs retain the HTTPS default and never enable
the bridge automatically.

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

The speech loader accepts `openai/whisper-tiny`, `openai/whisper-base`, or `openai/whisper-small`; the browser UI defaults
to Whisper Small INT8, matching Android and desktop. Like
the Android app, the browser downloads config and tokenizer files from that repository and the weights from
whisper.cpp's Q8_0 GGML file of the same size (`ggerganov/whisper.cpp`, pinned revision): 264 MB for Small instead of
the 967 MB FP32 Safetensors. The GGML bytes are cached beside the model's own files, so the cache manager lists and
deletes them together; a previously cached FP32 checkpoint is removed. Dictation loads the model on demand and uses the
shared GPU runtime when WebGPU is selected. Initialization failure is reported before retrying on a reusable CPU worker.
CPU pthreads are used only on isolated pages. Microphone permission is requested only when recording starts.
Small has the primary multilingual quality coverage; Tiny retains fast numerical and timing coverage, while Base
GPU quality/performance has not been measured in this redesign.

Microphone capture keeps a bounded 500 ms pre-roll locally and does not send PCM
to Whisper until the configured amount of consecutive speech crosses VAD.
Sub-threshold frames reset that consecutive-speech timer, so intermittent noise
cannot accumulate into a false start. Voice mode finalizes after end silence
and shows provisional recognition in the composer after speech starts.
The header explains that a 1.5-second pause sends the turn. The final transcript
remains editable and visible for 600 ms before voice mode submits it. Manual
dictation retains the same VAD-gated partial text without automatic submission.
Stopping before VAD confirms speech cancels cleanly and reports that no speech
was detected instead of decoding silence.

## Execution Modes

On first use, the app selects WebGPU when a hardware adapter is available; otherwise it uses CPU.
An explicitly selected backend is saved and takes precedence on later visits.

- **WebAssembly CPU, 1-8 configured threads:** uses the pthread module and the
  YNNPACK/Slinky scheduler. The selected count includes the calling thread and
  requires cross-origin isolation even when configured to one.
- **WebGPU:** uses the non-pthread `gpu` module. It requires a hardware WebGPU
  adapter and a secure context, but not cross-origin isolation, JSPI, or Asyncify.
  Chat, TTS, ASR, and vision share one device and Wasm heap. CPU threading remains
  an independent fallback choice.

Wasm never waits for the GPU, so every browser uses the same protocol and no
JSPI or Asyncify is involved. C++ records and submits GPU work synchronously
(pipelines use `createComputePipeline`), and `synchronize()` only submits.
Values the host needs, such as the selected token, are copied back
asynchronously. The worker calls chat, synthesis, or transcription start/step APIs, awaits the runtime's
`synchronize()`, and calls again; chat serving runs with
`ServingOptions::deferred_tokens`, so each step first accepts the token
selected by the previous one and then submits new work. The CPU variants use
the same loop. Image encoding has no mid-step reads: patch pooling runs on the
device and the position table stays on the host. Kernels that use
`dot4I8Packed` fall back to an equivalent WGSL function when a browser lacks the
`packed_4x8_integer_dot_product` extension. Recorded work is submitted every
128 dispatches, so the GPU starts on a step while the rest is still recorded.
The standalone CPU workers use the same speech start/step ABI and simply
complete on the first call. There are no separate synchronous TTS/ASR Wasm
exports.

### Shared GPU execution and residency

Logical clients use LiteApp's generic request/event/cancellation transport; no AI
logic is added to LiteApp. C++ model-session handles isolate generator, synthesizer,
transcriber, and pending coroutine state. Loader files and external-weight tables
belong to `/models/<kind>-<id>`; image staging directories are request-scoped.

Captured regions lower the existing graph IR to reusable GPU command descriptions,
patching input handles/offsets rather than crossing the Wasm/JavaScript bridge per
operator. Alias-aware liveness planning shares scratch buffers. Four-layer Qwen
and vision regions bound first-capture allocation as well as warm execution.
Direct in-place arithmetic avoids temporary-then-copy mutations.
Multi-query online-softmax attention is selected when materialized scores and
probabilities would exceed 32 MiB; smaller shapes retain the faster reference path.

The scheduler serializes GPU regions and gives interactive speech higher priority.
Reservations can delay work when co-residency does not fit. Up to three native
models remain warm, with a five-minute idle policy and LRU pressure eviction even
when logical clients remain attached. An evicted logical session reloads from the
persistent cache on use. When no native models remain, the device and Wasm module
are released and rebuilt lazily.

Current GPU-worker limits are 3 GiB of tracked GPU buffers, 3 GiB of accounted
resident host model memory, a 1 GiB prepared-weight cache, a 128 MiB free-buffer
pool, and bounded pipeline/bind-group/replay caches. Explicit upload staging and
readback are limited to 64 MiB each; at most four idle readback buffers, totaling
16 MiB, are retained. GPU accounting includes storage, uniforms, explicit staging,
readback, and profiling buffers, not driver/compiler allocations or total process
RAM. Host accounting does not make unused Wasm pages shrink, nor include separate
CPU fallback workers. The Wasm heap itself remains capped at 4 GiB.

Cancellation waits for submitted reads before releasing their destinations.
Device loss and shader/validation failure invalidate all GPU leases explicitly;
the UI marks chat unavailable rather than retaining a stale Ready state.
Reloading recreates the runtime from cached model data. Idle eviction does not
delete persistent cache entries or make a logical model unavailable.

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

The application shell, Chat view, EPUB Reader, message rendering,
Markdown, attachments, settings, and voice-mode controller come from
[`web/ui/`](./ui/), which is also embedded in the native desktop
application. This directory contains only the WASM host adapter, model workers,
cache implementation, and build packaging.

The Chat view reserves the viewport for messages and the bottom composer.
Model source, thread count, cache and token controls live in the runtime settings dialog.
Valid thread-count and output-token changes are remembered in local storage.
If isolation is unavailable, WebGPU remains usable but the CPU backend reports
that its pthread runtime is unavailable. The composer identifies inference as
running locally inside the browser.
New chat creates a blank conversation; completed chats are stored in local
browser storage and can be reopened or deleted from the history rail. Chat
content is not sent to a server.

### Text to speech

Browser runtime settings include a Model cache view listing files and their
combined cached sizes. Ranges and speech-model chunks are grouped by original
file; old chunks without a manifest appear as Unreferenced package data.
The file explorer opens directories with a click and provides breadcrumbs and
an Up button. Folders show aggregate sizes and support confirmed recursive
deletion of all cached files/chunks beneath them, at any directory level.
Refresh updates the inventory; Delete removes all cached parts of a file, and Clear model cache
removes only model storage, not conversations or attached images. Both deletion
actions require confirmation. Running models stay loaded, but later loads need
to fetch deleted data again. Keep the same origin (scheme, host, and port) to
manage the same cache.

Settings use a two-panel layout: section navigation on the left and independently
scrolling details on the right. Sections are LLM (including the system prompt
and browser model-loading controls), ASR, TTS, and finally Model Cache.
Model Cache and browser model-loading controls are browser-only.

Chat and Reader are top-level destinations in the application header. Chat owns
its conversation sidebar; Reader uses the full workspace for its library.
The library supports local EPUB import, search, sorting, removal, chapter
navigation, reading progress, and TTS narration. Browser EPUB parsing handles
stored or deflated ZIP entries locally with archive, entry, expanded-size,
path, and CRC checks. Original EPUB blobs use a dedicated Cache Storage cache
at the application origin; bounded metadata and progress remain in local
storage. Reloading reparses a cached package and requested chapter without a
server or model load. The in-memory provider retains at most two parsed
archives and lazily decompresses chapter content.
The open-book toolbar includes diffusion steps, delivery, gender, age, pitch,
style, accent, and browser GPU precision where available. Changes persist
through the shared TTS settings and restart active narration at the current text
position. Model loading and diagnostics remain in the full TTS settings panel.

When the local OmniVoice package is available during `make wasm`, the build
creates a chunked browser manifest under `models/omnivoice`, with chunks of up
to 256 MiB (three weight chunks for the current OmniVoice model). Smaller
support files remain separate. This reduces request overhead while bounding
the temporary buffers used for download, checksum verification, and caching;
it does not accelerate CPU synthesis. Existing 8 MiB packages remain supported.
Rebuilding changes chunk identities, so the new package needs a one-time cache
fill even if the old package was cached. The browser caches
those chunks at the application origin. With WebGPU selected, OmniVoice uses the
shared `gpu/kidi.wasm` runtime for diffusion and the complete Higgs/DAC waveform
decoder, including Conv1d, transposed-convolution projection/folding, and Snake.
CPU and native Metal retain their established CPU codec paths.

Guidance, candidate selection, schedule updates, and codebook embedding sums stay
on the GPU. CPU-generated random draws preserve the established seeded stream.
The 16-step reference's full-logit readback falls from 45,132,800 bytes to 1,376
bytes of final codes; PCM is read back separately. Each diffusion step yields to
JavaScript before the next C++ call. No production JSPI or Asyncify is used.
If WebGPU TTS initialization fails,
the UI reports the fallback and retries with up to eight WASM CPU threads when
cross-origin isolation is available.
The UI reports model loading, synthesis, and playback as separate phases.
Browser playback accepts transferable Float32 PCM directly, avoiding the old
Float32-to-Int16-to-Float32 round trip. Native binary PCM framing remains supported.
Completion follows audio delivery, and stream IDs fence cancelled chunks and
stale drain notifications.
Read aloud sends only the assistant response to OmniVoice, excluding hidden UI
labels such as the long-response disclosure.

Browser speech is synthesized in bounded phrases, not as one complete reply.
One shared `SpeechController` implements this behavior for Wasm and native
desktop; backend adapters expose only chat/ASR/TTS primitives and transport.
In voice mode, text deltas feed the phrase queue while chat generation continues;
playback starts when the first phrase arrives. Sentence and clause boundaries
are preferred. Complete sentences up to 192 weighted units remain whole; only
longer sentences use soft targets of 128 units for the first phrase and 160
thereafter. CJK graphemes cost three units. Text offsets and
segment IDs connect synthesis to transient message highlights: ready/played
phrases have a subtle background, and the phrase actually playing is emphasized.
CSS Custom Highlights preserve the Markdown DOM; older browsers use removable
inline marks. Cancellation and playback completion clear the highlights.

The controller is transport-independent and depends only on the versioned
command/event/PCM contract. This keeps a future Android host adapter possible
without moving browser-specific WebSocket, Worker, Cache Storage, or DOM details
into the state machine.

OmniVoice voice-design attributes describe a class of speaker, not one fixed
speaker identity. During TTS loading, the browser deterministically synthesizes
one short calibration sentence from the selected voice attributes and a voice
seed. It retains the matching transcript plus model-native 8-codebook tokens as
an OmniVoice voice prompt. Every visible segment, including segment zero, uses
that same immutable prompt; no sentence derives its identity from another
sentence.

The roughly 2.4 KiB prompt is persisted in a dedicated, versioned Cache Storage
namespace. Its key covers the model package, execution precision, voice
attributes, calibration text/language, voice seed, and diffusion parameters.
Cached prompts are shape/range/checksum validated before a narrow Wasm import.
A miss regenerates the deterministic prompt; corrupt or unavailable cache data
never falls back to an unconditioned voice. Model-cache deletion also clears
voice prompts. This uses OmniVoice's documented reusable prompt conditioning
instead of relying on [voice-design attributes](https://github.com/k2-fsa/OmniVoice/blob/master/docs/voice-design.md)
alone.

Playback applies backpressure at six seconds of queued audio with exactly one
segment of lookahead. The next segment starts synthesizing as soon as the
current PCM is queued; if it finishes early, its one PCM buffer waits for
playback capacity before a third synthesis can start. Cumulative consumed-frame
reports include chunks still in transit, so fast cache replays cannot bypass
that bound. The response speaker button is a three-state control: speaker starts
playback, pause stops consumption at the exact sample while bounded synthesis
may continue, and play resumes from that sample. Pausing a manual read detaches
it from the foreground request, re-enabling the composer and dictation while
the bounded speech stream remains resumable. Selecting another response's
speaker control cancels and clears the old playing or paused stream before
starting the newly selected response. A speech or audio-device failure stops
its speech stream without discarding an ongoing text reply. The GPU decoder
retains one waveform shape and Qwen retains one working set of four-layer
regions, rather than accumulating buffers for every phrase length.
Chat releases its prefill-only reservation after the first token; the existing
3 GiB GPU quota is unchanged. TTS admission reserves 768 MiB up to 128 text
units and 1.25 GiB for 129-192. These are pressure estimates, not allocations.

Recent synthesized PCM is cached in memory for replay: at most 16 MiB, 48 phrases,
and ten minutes from synthesis, with LRU eviction. Keys include source, selected
backend, text, effective synthesis settings, and the deterministic voice key.
Cached storage owns its bytes independently of playback transfers; a hit needs
no speech worker/model load. The cache is not written to disk and is cleared by
a page reload. Model storage and saved conversations are unchanged.

Segmentation reduces first-audio latency and bounds memory; it does not make
a slower-than-realtime profile gapless. Keep the chosen voice, steps, pace, and
precision when comparing RTF, and include model loading separately.

Browser TTS settings expose three compute profiles, without changing saved
diffusion steps or voice settings:

- **FP32 activations:** unchanged comparison baseline over the existing INT8
  projection weights.
- **Mixed FP16:** caches dequantized FP16 weights and stages FP16 activations,
  with FP32 products/accumulation. Faster, but uses additional GPU memory.
- **W8A8:** dynamically quantizes each activation row with an affine INT8 scale
  and zero point. INT8 weights remain packed; cached column sums provide exact
  zero-point correction. Production uses the portable floating-dot kernel;
  packed integer dots remain an explicit diagnostic option for adapters where
  full-model measurement proves they help. This is the default browser WebGPU
  profile; FP32 remains selectable as the quality/reference baseline.

`packed_4x8_integer_dot_product` exposes `dot4I8Packed`, not a guarantee of native
INT8 throughput. [Dawn's Metal lowering](https://github.com/google/dawn/blob/main/src/tint/lang/msl/writer/raise/raise.cc)
polyfills packed dots. The floating alternative keeps 8-bit weights/activations
in storage and decodes tiles for floating-point arithmetic. For the inspected
OmniVoice checkpoint, all 169 INT8 rank-two tensors satisfy
`128 * max_row_sum(abs(weight)) = 13,161,216 < 2^24`, so those integer products
and partial sums fit exactly in FP32. Other checkpoints need independent checks.

Controlled Apple M5 / Chromium 150 results, same checkpoint, English,
female/middle-aged/moderate, seed zero, pace one, 16 steps, no cached voice
reference, and 46,080 samples at 24 kHz (1.92 s including padding):

| Execution | Warm synthesis RTF | GPU storage buffers |
|---|---:|---:|
| FP32 activation baseline | 1.28-1.37 | about 0.83 GB |
| Mixed FP16 | 0.79-0.86 | about 1.76 GB |
| W8A8 packed intrinsic | 1.07-1.08 | about 0.84 GB |
| W8A8 selected floating kernel | 0.79-0.80 | about 0.84 GB |
| Native Metal reference | 0.35-0.36 | not compared |

An October 18 reproduction using the identical checkpoint and fixture did not
reproduce the original W8A8 number: the historical `28467d9` binary measured
1.13-1.15 RTF and the current binary measured 1.09-1.35 in a fresh Chromium
156 profile. VS Code's Electron/Chromium 150 browser measured about 1.5 RTF.
Installing the current 39-frame cached voice prompt increased one controlled
run from 1.92 to 2.17 RTF. These comparisons rule out a current W8A8 fallback
or a diffusion-kernel regression, but show that the earlier 0.79-0.80 result is
not a stable cross-browser performance expectation.

RTF includes generation and required codec work, not model load, playback, or
diagnostic CPU reference decoding. Warm preparation misses were zero. These are
one-adapter observations, not a universal INT8/FP16 speed claim or native parity.
The prepared representation is about 919 MB for FP16 versus 1.5 MB of W8A8
column sums. Small English, Spanish, French, and German backtranscription checks
cover finite/non-silent speech and up to two sentences, but are not a listening
score or a comprehensive quality evaluation. Existing browser WebGPU settings
from the former FP32-default release migrate once to W8A8; subsequent explicit
profile choices are preserved.

The adapter releases a logical TTS lease after delivery; the shared worker retains
bounded weights/plans for reuse rather than destroying the device every utterance.
The application never substitutes the browser's platform-dependent
`speechSynthesis` voices.

Continuous voice mode applies a hardware-dispatched WebGPU profile rather than
the quality-oriented benchmark settings above: constrained browsers use four
diffusion steps and a 96-unit first-segment ceiling, while higher-capacity
browsers use five steps and 112 units. Manual read-aloud retains the selected quality setting.
Voice calibration uses one persistent six-step reference independent of request
step count, so switching between real-time and read-aloud modes does not
regenerate or duplicate the voice prompt. Single PCM segments transfer through
the shared speech controller without an additional combine copy.

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
deletion does not delete chat images. Request-scoped temporary files in the WASM filesystem
are removed after request admission, including on errors. Captured four-layer
encoder regions and online attention reduced a rejected 5.7 GB cold capture to
about 2.4 GB of GPU storage buffers on the 192x192 synthetic-image test. Five
distinct images reached a memory plateau. These figures are not total process RAM.
Images are never
uploaded to Hugging Face or another inference service. Browser storage eviction
and memory limits still apply.

The microphone button records at most 30 seconds, resamples captured mono PCM to 16 kHz, and transfers snapshots to the
selected Whisper backend. Automatic language detection is enabled. While recording, replaceable draft hypotheses appear
in the composer; stopping runs a final pass that may refine them. The transcript is not sent to Gemma until submitted.

The first load converts the GGML weights to Kidi's per-channel INT8 checkpoint through the same preparation as native
builds, in the in-memory file system. The worker then stores that checkpoint in Cache Storage in place of the GGML
download (about the same size, 251 MB for Small), so later loads map it straight into the heap. CPU loads transcribe
a second of silence (`Transcriber::warm_up`) so the first recording does not pay weight packing. GPU execution uses
stepped encoder/language/decoder sessions, captured graphs, device-side suppression/selection, and compact token
readback. Native synchronous transcription drives the same coroutine session.
Whisper supports a 30-second window;
the browser instead encodes the recording plus at least one second of silence (`TranscriptionOptions::fit_audio`), in
2.56-second steps. If that output repeats itself, a known failure of shortened windows, it is decoded again over the
full window. On 43 LibriSpeech clean utterances (7.6 s on average) this matched the full window (Small 2.46% vs 2.58%
WER, Tiny 8.09% vs 8.68%) while encoding about 3.5 times faster. CPU projections use 128-row slices; GPU projections
run whole and use bounded-memory attention and graph scratch reuse.

On the M5, GPU Tiny transcribed the 11-second JFK fixture in about 0.40 s warm,
with the expected 24-token final transcript for English and automatic language
detection. That measurement includes **two attempts**: the shortened-window
repetition retry remains enabled. It is not a broad WER benchmark.

The following Small/Node numbers are earlier CPU-only measurements:

| Whisper Small, Node, Apple M5, 4 threads | Before (FP32, full window) | Now |
|---|---:|---:|
| Download | 967 MB | 267 MB |
| First load (convert + warm-up) | 0.56 s | 2.3 s |
| Reload (warm-up only) | 0.56 s | 0.75 s |
| Transcribe 6.6 s of speech | 3.1 s first, then 2.5 s | 0.63 s |
| Wasm heap after transcribing | 3.29 GiB | 0.66-0.76 GiB |

In Edge the app reports speech ready 1.5-2 s after opening with a cached model, and transcribes 3.9, 6.6 and 22.6
seconds of speech in 0.34, 0.63 and 2.2 s.

`benchmarks/browser/asr_probe.mjs` runs the same loader and runtime in Node against a local Hugging Face cache.

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
Use Load model to repair an incomplete or corrupt chat cache. Speech loading is
independent and can download missing files when first requested.

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

Worker diagnostics report allocated Wasm linear memory and growth headroom below the
4 GiB limit. These are not process
RAM or free device memory, and headroom does not include reusable space already
inside the heap or guarantee that the browser can allocate more memory.

Browser output defaults to 512 tokens and can be set from 1 to 8,192. The formatted
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
destroys the page's workers. Stopping one logical GPU request does not terminate
the shared worker or unrelated model sessions.

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

Standalone worker/benchmark results expose allocator, heap, transfer, dispatch,
preparation, and GPU buffer counters. Shared logical clients also accept
`runtime.status` for resident model ownership, global buffer totals, and host
accounting. `runtime.evict` releases idle native models without deleting their
persistent cache; attached logical clients remain reloadable. `runtime.reset`
deliberately destroys the device to exercise failure/recreation.
These are diagnostic worker methods, not new LiteApp or native-wire methods.
Allocator statistics walk the heap and are sampled at phase boundaries, not per
operator. `allocatedBytes`/`peakAllocatedBytes` count storage buffers;
`totalBufferBytes` additionally counts explicit staging, readback, uniforms,
and profiling buffers. Neither is total browser/GPU-driver memory.

To measure without a browser, run the same loader and module in Node against a
local Hugging Face snapshot:

```bash
node benchmarks/browser/heap_probe.mjs --glue build-web/threads/kidi.mjs \
  --snapshot ~/.cache/kidi/model-hub/models--google--gemma-4-E2B-it-qat-mobile-transformers/snapshots/<sha> \
  [--threads 4] [--words 1600] [--image photo.jpg] [--json result.json]
```

The heap ceiling is not an upfront allocation or a guarantee of available RAM;
the per-layer table also uses about 1.1 GiB of browser memory outside the heap.
Long requests and CPU fallback workers can still exhaust available process memory;
the GPU-worker quotas are not a promise that the machine has that much free RAM.
Allocation wrappers use JavaScript numbers, and `ccall` pointer arguments use
its `pointer` type. Check the built ABI with an allocation that crosses 2 GiB:

```bash
KIDI_TEST_WASM=build-web/gpu/kidi.mjs node --test tests/browser/model_cache_test.mjs
```

## Hosting and Cache

Pthreads require a secure, cross-origin-isolated page. The bundled, pinned
[`coi-serviceworker`](https://github.com/gzuidhof/coi-serviceworker) adds the
isolation headers through a service worker on servers that cannot configure
them, including Python's standard server and GitHub Pages. It must remain beside
the index page on the same origin. First installation can trigger a page reload;
subsequent visits reuse it. Browsers that cannot establish isolation fall back
to WebGPU when available; the CPU backend requires isolation. HTTPS is required
except on localhost/loopback.

### GitHub Pages

The [WebAssembly Pages workflow](../.github/workflows/pages.yml) tests the browser
loader and builds the pthread CPU fallback and unified WebGPU primary runtime for
relevant pull requests and pushes to `main`.
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
It also covers residency, cancellation/read lifetimes, Float32 delivery, playback
stream fencing, benchmark accounting, and kernel program shapes.
Only `*_test.mjs` files are passed to Node's test runner. The exported GPU
numerical/integration harnesses are run in a browser, not counted as successful
Node tests merely because importing them succeeds. Kernel-planning tests assert
shapes, binding layout, and memory bounds; numerical behavior is checked by
execution rather than shader-source substrings.
Real GPU numerical checks run through the production asynchronous path after
`make wasm`:

```js
const {runNumericalChecks} = await import('./webgpu-checks.mjs');
await runNumericalChecks();
await runNumericalChecks({portable: true});
await runNumericalChecks({matrixPrecision: 'mixed-f16'});
await runNumericalChecks({matrixPrecision: 'w8a8'});
await runNumericalChecks({matrixPrecision: 'w8a8', w8a8Kernel: 'float', portable: true});
```

Use the built page at `http://127.0.0.1:10010/` for model benchmarks so cached
weights keep the same origin. `runtime-bench.mjs` provides `benchmarkWorker`
(cold/first/warm measurements, fingerprints, PCM/token hashes, real-duration
RTF, counters) and `evaluateTtsProfiles` (resampling and explicit ASR WER/CER
proxies). A request array sweeps fresh inputs without disguising cached image
embeddings as encoder performance. Optional GPU timestamps are diagnostic only:
profiling changes pass boundaries and must not be used for headline latency.

```js
const {benchmarkWorker} = await import('./runtime-bench.mjs');
await benchmarkWorker({
  kind: 'tts', workerUrl: './tts-worker.mjs',
  load: {source: './models/omnivoice/manifest.json', backend: 'webgpu',
    threads: 1, cacheOnly: true, matrixPrecision: 'w8a8', w8a8Kernel: 'float'},
  request: {type: 'synthesize', text: 'The logo is for "kidi".',
    options: {language: 'English', steps: 16, seed: 0, pace: 1,
      use_voice_prompt: false, cache_voice_prompt: false, return_tokens: true,
      voice: {gender: 'female', age: 'middle-aged', pitch: 'moderate'}}},
  warmups: 1, trials: 3,
});
```

For shared-runtime benchmarks use
`workerFactory: () => createGpuWorker('tts')` from `runtime-client.mjs` instead
of `workerUrl`. Keep models sequential for controlled timing; test interleaved
work separately. Current full-model measurements are from Chromium/Apple M5,
not claimed validated on every browser/vendor. Native and portable-feature-disabled
numerical checks remain required when testing other adapters.

`gpu-runtime-checks.mjs` exports `runGpuRuntimeIntegration({speech})`, where
`speech` is the decoded 16 kHz Float32 JFK fixture buffer. It checks chat/TTS/ASR
co-residency, cancellation/reuse, eviction with attached logical clients, lazy
reload, device-loss invalidation, recreation, and final resource release.

The older C++ check page is an additional developer reference:
After `make wasm`, explicitly rebuild the C++ WebGPU check; the app build disables
test targets and an older test binary can otherwise remain on disk:

```bash
emcmake cmake -S . -B build-webgpu -DKIDI_BUILD_TESTS=ON \
  -DKIDI_WASM_WEBGPU=ON -DKIDI_WASM_THREADS=OFF
cmake --build build-webgpu --target kidi_webgpu_test -j8
python3 -m http.server 8081 --bind 127.0.0.1 --directory .
```

Open `http://127.0.0.1:8081/tests/browser/backend_test.html?run=1` in the integrated
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
node --test tests/browser/model_cache_test.mjs
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
