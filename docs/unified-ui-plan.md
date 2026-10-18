# Unified Kidi UI and Multimodal Feature Plan

## Objective

Build one shared, multi-view Kidi application that is packaged with two
execution backends:

1. The native desktop application served by LiteApp.
2. The browser application powered by WebAssembly and WebGPU.

Both applications must present the same visual design, interaction model,
settings experience, and multimodal capabilities. Chat is the first Kidi view,
not the application shell itself. The architecture must support future
first-class views such as an e-book reader without duplicating the application
framework or coupling global navigation to conversations. Backend differences
should be visible only where runtime configuration or diagnostics are
inherently platform-specific.

## Status Legend

- [ ] Not started
- [x] Complete
- Use nested checkboxes for independently verifiable deliverables.
- A milestone is complete only when all of its deliverables and acceptance
  checks are complete.

## Current-State Review

- [x] Identify the two independent UI implementations.
  - Shared Kidi UI: [`web/ui/`](../web/ui/)
  - Desktop host: [`web/hosts/desktop/`](../web/hosts/desktop/)
  - WASM host: [`web/hosts/wasm/`](../web/hosts/wasm/)
- [x] Confirm that both applications already share the native inference core.
- [x] Confirm that both build paths use vendored browser libraries from
  [`web/libs/`](../web/libs/).
- [x] Identify duplicated UI responsibilities:
  - Conversation storage
  - History navigation
  - Message rendering
  - Markdown sanitization and highlighting
  - Composer behavior
  - Dictation controls
  - Runtime state and error presentation
  - Responsive layout
- [x] Identify current feature differences:
  - WASM supports image attachments and browser model downloads.
  - Desktop supports read-aloud TTS, streaming TTS, continuous voice mode, and
    runtime system-prompt editing.
  - The two applications use different styles, settings layouts, persistence
    schemas, event formats, and runtime controls.

## Product Requirements

### Shared functionality

- [ ] Streaming text generation
- [ ] Sanitized streaming Markdown
- [ ] Syntax-highlighted code
- [ ] Conversation creation, selection, deletion, and persistence
- [ ] Image attachment by picker, paste, and drag-and-drop
- [ ] Image previews and per-conversation image persistence
- [ ] Microphone dictation with partial and final transcripts
- [ ] Read-aloud TTS
- [ ] Continuous ASR-to-chat-to-TTS voice mode
- [ ] Editable runtime system prompt
- [ ] TTS language, voice, pace, and quality settings
- [ ] Copy-response action
- [ ] Generation timing and token statistics
- [ ] Cancellation at every inference stage
- [ ] Responsive desktop, tablet, and mobile behavior
- [ ] Accessible labels, focus handling, and keyboard operation
- [ ] Offline operation after required models are installed or cached

### Shared visual and interaction design

- [ ] One multi-view application shell
- [ ] One primary navigation system
- [ ] One route and view lifecycle system
- [ ] One color and typography system
- [ ] View-specific secondary navigation, such as chat history or a book table
  of contents
- [ ] One compact message-bubble design
- [ ] One composer and attachment experience
- [ ] One play/copy action design
- [ ] One settings-dialog framework
- [ ] One status, progress, notice, and error system
- [ ] One responsive breakpoint strategy
- [ ] One set of loading, empty, listening, generating, and speaking states

### Platform-specific runtime surfaces

These use shared UI components but expose different values.

- [ ] Desktop runtime section
  - YAML path
  - Native model directories
  - Accelerator placement
  - Metal cache limits
  - Thread count
  - Idle model eviction
- [ ] WASM runtime section
  - Hugging Face model IDs
  - Model download and cache status
  - CPU/WebGPU selection
  - Worker thread count
  - WASM heap diagnostics
  - Browser cache management

## Multi-View Kidi Application Model

Kidi must be organized as an application containing independently mounted
feature views. Chat must not own the root shell, global navigation, settings,
runtime lifecycle, model services, or persistence infrastructure.

### Initial and future views

- [ ] Chat view
  - Conversation history
  - Multimodal messages
  - Dictation
  - Read aloud
  - Continuous voice mode
- [ ] Reader view
  - Book library
  - EPUB/document import
  - Chapter and page navigation
  - Reading-position persistence
  - Text selection
  - Translate selection
  - Explain selection
  - Read selection, paragraph, chapter, or book aloud
  - Reader-specific AI context
  - Reader annotations and bookmarks
- [ ] Define a stable view-registration API for additional future workflows.

### Application-shell responsibilities

- [ ] Primary navigation between Kidi views.
- [ ] Route parsing, deep links, and back/forward navigation.
- [ ] View mounting, suspension, restoration, and disposal.
- [ ] Shared runtime readiness and model status.
- [ ] Shared global settings.
- [ ] Shared notification and error presentation.
- [ ] Shared account/privacy/about surfaces if added later.
- [ ] Responsive adaptation of primary and secondary navigation.

### View responsibilities

- [ ] Each view owns its domain state and secondary navigation.
- [ ] Chat owns conversations and chat history.
- [ ] Reader owns books, reading position, table of contents, bookmarks, and
  annotations.
- [ ] Views consume shared Kidi services instead of owning model workers or
  native transports.
- [ ] Views declare required capabilities when mounted.
- [ ] Views release view-specific resources when suspended or disposed.
- [ ] A failure in one view must not corrupt or reset another view.

### Shared Kidi services

- [ ] Text generation
- [ ] Translation
- [ ] Summarization and explanation
- [ ] Vision/image understanding
- [ ] Speech recognition
- [ ] Speech synthesis
- [ ] Audio playback
- [ ] Model availability and lifecycle
- [ ] Runtime settings
- [ ] Content and blob persistence

The service layer must accept task-specific context rather than assuming every
request is a chat conversation. For example, Reader can submit a selected
passage, source language, target language, chapter metadata, and surrounding
context without converting the operation into a fake user chat message.

### Routing model

- [ ] Reserve stable routes such as:
  - `/chat`
  - `/chat/:conversationId`
  - `/reader`
  - `/reader/:bookId`
  - `/reader/:bookId/:location`
- [ ] Keep route definitions in Kidi, not LiteApp.
- [ ] Permit LiteApp to provide a generic router/view-host primitive.
- [ ] Preserve active view and view state across reloads.
- [ ] Support deep links in both the native loopback host and static WASM host.

## Target Architecture

```text
third_party/liteapp/
├── web/
│   ├── core/
│   │   ├── application.mjs
│   │   ├── lifecycle.mjs
│   │   └── capabilities.mjs
│   ├── transport/
│   │   ├── channel.mjs
│   │   ├── websocket-channel.mjs
│   │   ├── worker-channel.mjs
│   │   └── service-router.mjs
│   ├── storage/
│   │   ├── records.mjs
│   │   ├── blobs.mjs
│   │   └── migrations.mjs
│   ├── media/
│   │   ├── microphone.mjs
│   │   ├── capture-worklet.js
│   │   ├── playback.mjs
│   │   ├── playback-worklet.js
│   │   └── vad.mjs
│   ├── components/
│   │   ├── shell.mjs
│   │   ├── sidebar.mjs
│   │   ├── dialog.mjs
│   │   ├── composer.mjs
│   │   └── notice.mjs
│   ├── styles/
│   └── build/
│
web/
├── ui/
├── app.js
├── router.mjs
├── view-registry.mjs
├── navigation.mjs
├── kidi-client.mjs
├── kidi-services.mjs
├── state/
│   ├── application.mjs
│   ├── preferences.mjs
│   └── migrations.mjs
├── shared/
│   ├── attachments.mjs
│   ├── dictation.mjs
│   ├── playback.mjs
│   ├── voice-mode.mjs
│   ├── markdown.js
│   └── settings.mjs
├── views/
│   ├── chat/
│   │   ├── view.mjs
│   │   ├── conversations.mjs
│   │   ├── history.mjs
│   │   ├── messages.mjs
│   │   └── composer.mjs
│   └── reader/
│       ├── view.mjs
│       ├── library.mjs
│       ├── document.mjs
│       ├── navigation.mjs
│       └── reading-tools.mjs
│   ├── style.css
│   └── icons/
├── hosts/
│   ├── desktop/
│   │   ├── backend.js
│   │   ├── capture-worklet.js
│   │   └── playback-worklet.js
│   └── wasm/
│       ├── backend.js
│       ├── inference-worker.mjs
│       ├── asr-worker.mjs
│       ├── tts-worker.mjs
│       └── model-cache.mjs
├── libs/
└── tools/
```

## Framework Ownership and Reuse Rules

LiteApp is a general-purpose application framework. It must remain usable by
applications that have no AI, chat, speech, image, or Kidi functionality.

### LiteApp may own

- [ ] Application bootstrap, lifecycle, readiness, shutdown, and disposal.
- [ ] Generic request/event channels over WebSocket, Worker, and MessagePort.
- [ ] Request IDs, event sequence validation, cancellation, limits, and
  backpressure.
- [ ] Generic service registration and routing.
- [ ] Versioned record, key/value, and blob persistence primitives.
- [ ] Generic schema-migration mechanics; applications own migration contents.
- [ ] Asset staging, deterministic archives, static output, MIME handling, CSP,
  and cross-origin-isolation configuration.
- [ ] Accessible, unbranded UI primitives such as shells, sidebars, dialogs,
  composers, buttons, progress indicators, and notices.
- [ ] Optional browser media primitives:
  - Microphone permission and lifecycle
  - AudioWorklet capture
  - Downmix and resampling
  - PCM playback and drain notification
  - Voice activity detection
- [ ] Provider interfaces that applications may implement for speech
  recognition or synthesis.

### LiteApp must not own

- [ ] Kidi branding, copy, colors, or product layout.
- [ ] Chat, ASR, TTS, image, or settings service names.
- [ ] Gemma, Whisper, OmniVoice, or Hugging Face knowledge.
- [ ] Kidi conversation or message schemas.
- [ ] Kidi voice-mode policy.
- [ ] Kidi system-prompt policy.
- [ ] Kidi model selection, package layout, cache keys, or runtime placement.
- [ ] Kidi-specific WebSocket or worker payloads.
- [ ] Assumptions that an application uses AI, audio, images, or a sidebar.

### Optionality requirements

- [ ] LiteApp core must not import storage, media, UI, or transport modules
  unless the application explicitly selects them.
- [ ] Speech and playback support must be optional modules.
- [ ] UI primitives must be usable independently rather than requiring one
  prescribed application shell.
- [ ] Native-only C++ libraries must not be pulled into WASM/static-web builds.
- [ ] Browser-only modules must not become dependencies of native applications
  that do not embed a web frontend.
- [ ] Every LiteApp module must be testable without Kidi source or fixtures.

### Kidi owns

- [ ] The shared Kidi application composition and theme.
- [ ] Kidi view registration, routing, and primary navigation.
- [ ] Kidi services and payload validation.
- [ ] Conversation and attachment schemas.
- [ ] Reader book, location, annotation, and bookmark schemas.
- [ ] Multimodal chat orchestration.
- [ ] Reader translation, explanation, and read-aloud workflows.
- [ ] System-prompt behavior.
- [ ] Model-specific settings.
- [ ] ASR/TTS provider implementations.
- [ ] Continuous voice-mode state and product behavior.
- [ ] Native and WASM model workers/services.

## Shared Contracts

### LiteApp channel surface

LiteApp should provide a generic channel API without application service names:

```js
channel.request(method, payload, {
  signal,
  onEvent
})
```

- [ ] Implement a WebSocket channel.
- [ ] Implement a Worker channel.
- [ ] Implement a MessagePort channel.
- [ ] Implement a deterministic fake channel for tests.
- [ ] Keep message envelopes generic and application payloads opaque.
- [ ] Permit transferable binary buffers without copying where the host allows.

### Kidi service surface

- [ ] Define documented Kidi services over the LiteApp channel.
- [ ] Define normalized capability metadata.
- [ ] Define normalized settings metadata.
- [ ] Define lifecycle and disposal behavior.
- [ ] Require explicit errors instead of silent feature fallback.

Proposed logical surface:

```js
backend.chat.generate(request, callbacks, signal)
backend.chat.cancel(requestId)
backend.images.prepare(attachments, signal)
backend.asr.start(options, callbacks, signal)
backend.asr.stop()
backend.tts.prepare(settings, signal)
backend.tts.synthesize(text, settings, callbacks, signal)
backend.tts.cancel(requestId)
backend.settings.read()
backend.settings.update(change)
backend.runtime.subscribe(listener)
backend.dispose()
```

### Normalized events

- [ ] Define `started`.
- [ ] Define `progress`.
- [ ] Define `partial`.
- [ ] Define `delta`.
- [ ] Define `completed`.
- [ ] Define `cancelled`.
- [ ] Define `error`.
- [ ] Define ordering and request-ID guarantees.
- [ ] Define binary payload ownership and transfer rules.

### Canonical conversation schema

- [ ] Version the schema.
- [ ] Assign stable IDs to every chat and message.
- [ ] Support optional image attachments.
- [ ] Normalize generation statistics.
- [ ] Preserve raw Markdown as the message source.
- [ ] Exclude incomplete pending messages from durable storage.

Proposed shape:

```js
{
  version: 2,
  id,
  title,
  updatedAt,
  messages: [{
    id,
    role,
    content,
    images: [{
      id,
      name,
      mimeType,
      size
    }],
    stats: {
      tokenCount,
      promptTokens,
      decodeTokens,
      elapsedMs,
      prefillMs,
      decodeMs,
      firstTokenMs
    }
  }]
}
```

## Milestone 0: Baseline and Safety Net

**Goal:** Capture current behavior before reorganizing either application.

- [ ] Add representative stored-conversation fixtures for both existing schemas.
- [ ] Add fixtures containing Markdown, code, tables, Unicode, and images.
- [ ] Record desktop and mobile visual baselines for both applications.
- [ ] Add a fake backend that can emit deterministic lifecycle events.
- [ ] Add tests for:
  - Streaming deltas
  - Completion
  - Cancellation
  - Errors
  - Reconnection
  - Partial ASR
  - Streaming PCM
- [ ] Document current public build commands.
- [ ] Confirm baseline commands pass:
  - `make native-test`
  - `make web-test`
  - `make app`
  - `make wasm`

### Milestone 0 acceptance

- [ ] Existing behavior can be compared before and after each migration.
- [ ] Test fixtures contain no external network dependency.
- [ ] No user history is modified yet.

## Milestone L1: General-Purpose LiteApp Web Runtime

**Goal:** Extend LiteApp into a reusable native/static-web application
framework before moving Kidi product code onto it.

### Core and transport

- [ ] Define LiteApp's JavaScript module boundaries and public API.
- [ ] Implement generic application lifecycle and disposal.
- [x] Implement request/event envelopes.
- [x] Implement request IDs and monotonic event sequences.
- [x] Implement cancellation with `AbortSignal`.
- [ ] Implement bounded text and binary payload handling.
- [ ] Implement WebSocket, Worker, and MessagePort channels.
- [x] Implement generic service routing.
- [ ] Keep cpp-httplib and native WebSocket details behind existing LiteApp
  boundaries.

### Storage

- [ ] Implement optional local record storage.
- [ ] Implement optional IndexedDB blob storage.
- [ ] Implement generic versioned migration transactions.
- [ ] Surface quota, serialization, and transaction errors explicitly.
- [ ] Avoid defining any application schema in LiteApp.

### Optional media

- [ ] Implement reusable microphone capture.
- [ ] Implement reusable downmix and resampling helpers.
- [ ] Implement reusable PCM playback with backpressure and drain events.
- [ ] Implement optional VAD.
- [ ] Define provider interfaces for transcription and synthesis.
- [ ] Do not include or select an ASR/TTS engine inside LiteApp.

### Optional UI primitives

- [ ] Implement accessible unbranded controls.
- [ ] Implement dialog and focus management.
- [ ] Implement responsive sidebar behavior.
- [ ] Implement generic progress, status, and notice components.
- [ ] Keep all visual tokens overrideable by applications.
- [ ] Do not ship Kidi branding or product copy in LiteApp.

### Packaging

- [ ] Add a reusable CMake function for embedded web applications.
- [ ] Add a reusable static-web packaging tool.
- [ ] Support workers, worklets, WASM files, and ordinary assets.
- [ ] Support strict CSP generation.
- [ ] Support cross-origin-isolation configuration for applications that opt in.
- [ ] Keep native HTTP/database/filesystem/platform targets optional and
  excluded from static-web builds.

### Standalone LiteApp validation

- [ ] Add a non-Kidi example application.
- [x] Test every currently implemented module using LiteApp-owned generic
  fixtures.
- [ ] Build and test LiteApp as a standalone top-level project.
- [ ] Verify no LiteApp source imports or references Kidi.
- [ ] Document module-level opt-in behavior and dependency costs.

### Milestone L1 acceptance

- [ ] A small non-AI application can use LiteApp core and UI primitives.
- [ ] A static browser application can use Worker transport without native
  LiteApp libraries.
- [ ] A native application can use embedded assets and WebSocket transport.
- [ ] Media, storage, and UI modules are optional.
- [ ] LiteApp contains no Kidi business logic, names, models, schemas, or
  branding.

## Milestone 1: Shared Design System and Markdown

**Goal:** Establish one visual source of truth without changing backend logic.

- [x] Create [`web/ui/`](../web/ui/).
- [ ] Build the Kidi UI from LiteApp primitives without placing Kidi styles in
  LiteApp.
- [x] Move shared color, typography, spacing, radius, and elevation tokens into
  one stylesheet.
- [x] Move message bubbles, history, composer, dialogs, controls, notices, and
  responsive rules into the shared stylesheet.
- [x] Adopt the compact desktop conversation design as the common baseline.
- [x] Consolidate the two Markdown implementations.
- [ ] Preserve:
  - Sanitization
  - Safe links
  - GFM tables
  - Syntax highlighting
  - Incremental Markdown
  - Open-fence handling
  - Long-response disclosure
  - Mermaid rendering
- [x] Consolidate shared icons and their licenses.
- [x] Align vendored module names and paths for both builds.

### Milestone 1 acceptance

- [ ] Desktop and WASM render the same fixture conversation identically.
- [x] Both applications use the same Markdown module and CSS.
- [x] No duplicated Markdown renderer remains.
- [ ] CSP remains strict and no inline script or style exemptions are added.

## Milestone 2: Shared Multi-View Shell and Chat View

**Goal:** Replace duplicated browser application structure with one extensible
Kidi shell and mount Chat as its first feature view.

- [x] Implement the shared application shell.
- [x] Implement primary application navigation.
- [ ] Implement the Kidi router and view registry.
- [ ] Implement view mount, suspend, restore, and dispose lifecycles.
- [x] Implement Chat as a registered view.
- [x] Implement Chat-specific history rendering.
- [x] Implement Chat-specific message rendering.
- [x] Implement Chat-specific composer behavior.
- [x] Implement shared copy and playback action placement.
- [x] Implement shared follow-scroll behavior.
- [x] Implement shared responsive primary navigation.
- [x] Implement view-specific responsive secondary navigation.
- [x] Implement shared notice and error presentation.
- [x] Implement shared loading, listening, generating, and speaking states.
- [x] Implement shared settings-dialog layout.
- [x] Reduce both host pages to minimal bootstrap documents.
- [x] Ensure the mounted DOM is equivalent across desktop and WASM.
- [x] Add a placeholder Reader registration test to prove Chat is not coupled
  to the root shell.

### Milestone 2 acceptance

- [x] Both applications use the same shell, views, controls, and styles.
- [x] Platform-specific code does not create message or navigation DOM.
- [ ] Desktop and WASM entry points only mount the shared UI with an adapter.
- [ ] Chat can be mounted and disposed through the generic view lifecycle.
- [ ] Registering a second non-chat view requires no changes to Chat modules.

## Milestone 3: Shared Persistence and Migration

**Goal:** Use one durable conversation format without losing existing history.

- [x] Namespace persisted records by Kidi view/domain.
- [x] Keep application preferences independent of Chat state.
- [x] Implement canonical schema validation.
- [x] Implement migrations from `kidi-desktop-v1`.
- [x] Implement migrations from `kidi-chats-v1`.
- [x] Migrate the active-chat selection.
- [x] Preserve image references from WASM conversations.
- [x] Preserve generation statistics where possible.
- [x] Add a storage-version marker.
- [x] Make migrations idempotent.
- [x] Retain a rollback-safe copy until migration succeeds.
- [ ] Apply one shared retention limit and ordering policy.

### Milestone 3 acceptance

- [x] Existing desktop conversations survive migration.
- [x] Existing WASM conversations and images survive migration.
- [x] Re-running migration does not duplicate or corrupt data.
- [x] Pending or malformed messages do not enter durable history.

## Milestone 4: Backend Adapters

**Goal:** Preserve both execution paths behind one contract.

### Desktop adapter

- [ ] Configure a LiteApp WebSocket channel.
- [ ] Register Kidi-specific desktop services over the generic channel.
- [x] Normalize native chat events.
- [x] Normalize ASR events.
- [x] Normalize TTS events.
- [x] Preserve binary PCM framing and backpressure.
- [x] Preserve reconnect behavior.
- [x] Preserve request cancellation and voice epochs.
- [x] Normalize native statistics into the canonical schema.

### WASM adapter

- [ ] Configure LiteApp Worker and MessagePort channels.
- [ ] Register Kidi-specific WASM services over the generic channels.
- [x] Normalize worker chat events.
- [x] Normalize model-download progress.
- [x] Normalize ASR events.
- [x] Preserve WebGPU/CPU selection.
- [x] Preserve worker restart and cancellation behavior.
- [x] Normalize WASM statistics into the canonical schema.

### Milestone 4 acceptance

- [ ] Shared UI tests pass against both adapters.
- [x] Backend-specific event formats do not leak into UI modules.
- [x] Cancellation behaves consistently on both platforms.
- [ ] Kidi transport code does not reimplement LiteApp request IDs, event
  sequencing, cancellation, or service routing.

## Milestone 5: Image Parity

**Goal:** Support the same vision workflow on desktop and WASM.

### Shared image UX

- [x] Implement picker, paste, and drag-and-drop.
- [x] Implement previews and removal.
- [x] Implement image-only user messages.
- [x] Implement image gallery rendering in conversation history.
- [x] Apply common count and byte limits.
- [ ] Show common preparation and encoding progress.

### Native image input

- [ ] Introduce a native image-input abstraction that can accept encoded bytes
  without trusting browser-provided filesystem paths.
- [x] Preserve path-based loading for CLI callers.
- [x] Extend the desktop protocol with bounded image upload frames.
- [ ] Validate MIME type, size, dimensions, request ownership, and ordering.
- [x] Release uploaded bytes after completion, cancellation, or disconnect.
- [x] Connect image inputs to `ChatMessage` and multimodal generation.

### Milestone 5 acceptance

- [x] The same image prompt produces a multimodal request on both backends.
- [x] Image limits and errors are presented consistently.
- [x] Cancellation releases image memory and temporary resources.
- [x] No arbitrary local filesystem path can be supplied by the browser.

## Milestone 6: System Prompt and Settings Parity

**Goal:** Give both applications the same configurable behavior.

- [x] Add runtime system-prompt support to the WASM adapter.
- [x] Inject system prompts only inside trusted backend code.
- [x] Reject browser-provided system/developer conversation roles.
- [x] Share the system-prompt editor.
- [x] Share reset, modified, applying, and error states.
- [x] Define the WASM prompt reset and persistence policy.
- [x] Share response-token and image-limit controls.
- [x] Share ASR language controls.
- [x] Share TTS language, voice, pace, and step controls.
- [x] Render desktop- and WASM-specific runtime fields with shared components.

### Milestone 6 acceptance

- [x] Prompt changes affect the next turn on both backends.
- [x] Reset restores the configured default.
- [x] Settings validation and error behavior are equivalent.

## Milestone 7: OmniVoice TTS in WASM

**Goal:** Provide the same read-aloud model and voice behavior in the browser.

- [x] Confirm OmniVoice dependencies compile under Emscripten.
- [x] Add TTS bindings to [`src/kidi/platform/wasm/bindings.cpp`](../src/kidi/platform/wasm/bindings.cpp).
- [x] Add a dedicated TTS worker.
- [x] Add OmniVoice package download and cache support.
- [ ] Display model size, license notice, download progress, and cache state.
- [ ] Implement persistent voice-prompt caching inside the worker.
- [x] Support auto-language inference.
- [x] Support the same voice attributes and pace presets.
- [ ] Port Unicode-safe streaming segmentation.
- [ ] Port first-segment quality and latency settings.
- [x] Stream transferable PCM buffers to the shared playback worklet.
- [ ] Port adaptive silence trimming and overlap joining.
- [x] Add idle TTS worker eviction.
- [x] Report explicit model-load and memory failures.

### Memory requirements

- [ ] Measure Gemma, Whisper, and OmniVoice workers independently.
- [ ] Measure combined browser process memory.
- [x] Ensure no individual WASM heap exceeds the 4 GiB limit.
- [x] Define constrained-device model eviction behavior.
- [x] Never silently substitute browser `speechSynthesis`.

### Milestone 7 acceptance

- [x] Read aloud uses OmniVoice on both platforms.
- [ ] The same voice profile is stable across segments and turns.
- [ ] Kannada and other multibyte text synthesize without invalid UTF-8.
- [ ] Playback begins before long responses finish synthesizing.
- [ ] Internal segment boundaries have no artificial long silence.

## Milestone 8: Shared Continuous Voice Mode

**Goal:** Run one voice-mode state machine on both adapters.

- [x] Extract the desktop voice epoch and stop-loop protections.
- [x] Implement the shared state machine:
  - Idle
  - Listening
  - Finalizing ASR
  - Generating
  - Speaking
  - Draining playback
  - Returning to listening
- [x] Normalize VAD configuration and state.
- [x] Keep ASR transcripts visible in conversation history.
- [x] Stream chat text while generating.
- [ ] Start TTS before complete chat output.
- [x] Cancel correctly during every stage.
- [x] Prevent stale timers, worker messages, and WebSocket events from
  restarting a stopped session.
- [x] Apply identical status text and controls on both platforms.

### Milestone 8 acceptance

- [x] Both applications complete ASR-to-chat-to-TTS-to-ASR loops.
- [x] Repeated stop/start cycles do not produce loops.
- [x] Stopping during listening, generation, or playback remains stopped.
- [x] Errors return to a recoverable idle state.

## Milestone 9: Build and Test Integration

**Goal:** Package the same UI reliably in both products.

- [ ] Replace Kidi-specific asset staging with LiteApp's embedded application
  packaging.
- [ ] Replace Kidi-specific static staging with LiteApp's static-web packaging.
- [x] Update [`CMakeLists.txt`](../CMakeLists.txt) to configure the Kidi desktop
  application through LiteApp.
- [x] Update [`web/tools/build.mjs`](../web/tools/build.mjs) to configure the Kidi WASM
  application through LiteApp.
- [x] Keep public commands in [`Makefile`](../Makefile):
  - `make app`
  - `make wasm`
  - `make serve`
  - `make web-test`
- [ ] Expand `make web-test` with:
  - Shared UI tests
  - Backend contract tests
  - Persistence migration tests
  - Voice state-machine tests
  - Markdown security tests
- [ ] Add visual snapshots for desktop and mobile widths.
- [ ] Add desktop WebSocket multimodal integration tests.
- [x] Add WASM chat, image, ASR, and TTS smoke tests.
- [ ] Confirm no build requires a network request after dependencies and model
  fixtures are available.

### Milestone 9 acceptance

- [x] `make native-test` passes.
- [x] `make web-test` passes.
- [x] `make app` passes.
- [x] `make wasm` passes.
- [x] Desktop and WASM use byte-identical shared UI assets.

## Milestone 10: Remove Legacy Duplication

**Goal:** Finish the migration and prevent future UI drift.

- [x] Delete the old desktop-only message renderer.
- [x] Delete the old WASM-only message renderer.
- [x] Delete duplicate Markdown modules.
- [x] Delete duplicate conversation persistence code.
- [x] Delete duplicate shell and settings styles.
- [ ] Delete unused icons and staged asset aliases.
- [x] Update desktop and WASM documentation.
- [x] Document the backend contract for future platforms.
- [ ] Add a repository check that prevents reintroducing duplicate shared UI
  assets.

### Milestone 10 acceptance

- [x] Visible UI behavior comes exclusively from [`web/ui/`](../web/ui/).
- [x] Desktop-specific browser code contains transport/runtime logic only.
- [x] WASM-specific browser code contains worker/runtime logic only.
- [ ] No known feature or visual parity gaps remain.

## Final End-to-End Acceptance Checklist

Run every item once with the desktop backend and once with the WASM backend.

- [ ] Start a new conversation.
- [ ] Edit and reset the system prompt.
- [ ] Send a text-only message.
- [ ] Attach an image with the file picker.
- [ ] Paste an image.
- [ ] Drop an image into the composer.
- [ ] Receive streamed Markdown containing headings, lists, code, and a table.
- [ ] Copy the raw response.
- [ ] Read the response aloud.
- [ ] Change voice, language, pace, and quality.
- [ ] Dictate a message and edit the transcript.
- [ ] Send while dictation is active.
- [ ] Enter continuous voice mode.
- [ ] Stop voice mode while listening.
- [ ] Stop voice mode while generating.
- [ ] Stop voice mode while speaking.
- [ ] Reopen a persisted conversation.
- [ ] Reload/restart and confirm configured prompt restoration.
- [ ] Work offline with cached or installed models.
- [ ] Recover from a model-load failure.
- [ ] Recover from an inference failure.
- [ ] Verify keyboard-only operation.
- [ ] Verify narrow mobile layout.
- [ ] Verify no unexpected memory growth after repeated multimodal turns.

## Risks and Required Measurements

- [ ] Measure WASM OmniVoice compatibility before committing to a delivery date.
- [ ] Measure browser memory with separate chat, ASR, and TTS workers.
- [ ] Verify OmniVoice browser distribution and license presentation.
- [ ] Verify image-byte transfer does not duplicate large buffers unnecessarily.
- [ ] Verify WebGPU resources are released when workers are evicted.
- [ ] Verify desktop binary size after embedding shared assets.
- [ ] Verify local-storage and IndexedDB migration against real existing data.
- [ ] Verify CSP compatibility for shared modules, workers, and worklets.

## Definition of Done

The project is complete when Kidi has one shared multi-view browser application
source, Chat is an independently registered view, both backends pass the final
acceptance checklist, and existing user history survives migration. Neither
platform may contain an independent implementation of the application shell,
conversation interface, or multimodal workflow. Adding Reader or another
future view must not require restructuring Chat or either backend transport.
LiteApp must remain a standalone general-purpose framework with optional web,
transport, storage, media, and UI modules and no dependency on Kidi business
logic.
