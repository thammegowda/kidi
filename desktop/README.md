# Kidi Desktop Streaming Protocol

`kidi-app CONFIG` loads one immutable YAML configuration, starts a
capability-protected loopback server, and serves the embedded interface. HTTP is
used only for bootstrap assets and `/api/status`; interactive work uses one
authenticated WebSocket at `/ws`.

For development, `kidi-app CONFIG --port 10010 --dev` keeps the loopback bind
and security headers but disables the one-time capability and cookie exchange.
It serves the shared UI, desktop host adapter, worklets, icons, and browser
libraries directly from the repository, so HTML, JavaScript, and CSS edits only
need a browser reload. Development assets are resolved from the current working
directory. Development mode prints a warning and serves a stable
clickable `http://127.0.0.1:10010/` URL without opening a browser or application
window. Never use `--dev` with an exposed or forwarded port.

`make app-dev` builds and runs this mode on port `10010`. Override the defaults
with `APP_CONFIG=path/to/config.yaml` or `APP_PORT=another-loopback-port`.

## Text messages

Every command and event is a JSON object:

```json
{"v":1,"id":"request UUID","type":"chat.start","seq":0,"payload":{}}
```

Client commands are `chat.start`, `image.start`, `image.finish`, `asr.start`,
`asr.stop`, `tts.start`, `settings.system_prompt.set`, and `cancel`. Server events are `session.ready`,
`settings.system_prompt.updated`, modality-specific `started`, `partial`/`delta`,
and `completed` events, `request.cancelled`, and `error`. Sequence numbers are
monotonic within each request.

Images use bounded binary frames with kind `3`. The browser announces a unique
stream, MIME type, and total size, sends ordered 64 KiB chunks, and finishes the
stream before starting chat. The server accepts JPEG and PNG data only, limits
each upload to 32 MiB, writes an unguessable request-scoped temporary file for
the existing multimodal generator API, and removes it after completion,
cancellation, or disconnect. Browser-provided filesystem paths are never
accepted.

Chat publishes each tokenizer-safe `GenerationEvent.text` delta immediately.
ASR accepts rolling microphone audio and emits bounded partial snapshots before
the final result. TTS emits each sentence-sized waveform chunk in playback order
and does not retain the complete waveform in native memory.

In voice mode, completed LLM sentences and natural clauses enter a bounded
four-segment TTS queue while later chat tokens are still generating. Chat and
TTS use separate model locks, so PCM can begin before `chat.completed` without
unbounded buffering. Markdown markers and link destinations are removed from
spoken segments.

The chat system prompt is injected server-side as the first message; clients
cannot submit system or developer roles. The settings UI can replace it for
future turns with `settings.system_prompt.set`. That override is held only in
process memory, so restarting the app restores the YAML value. `tts.start` may
include validated per-request language, diffusion-step, gender, age, pitch,
style, and accent overrides without changing the immutable model configuration.

## Binary frames

Audio uses PCM16 little-endian mono data. Each binary WebSocket message starts
with this 24-byte little-endian header:

| Offset | Size | Meaning |
|---:|---:|---|
| 0 | 4 | ASCII `KDA1` |
| 4 | 1 | Protocol version (`1`) |
| 5 | 1 | Kind: `1` ASR input, `2` TTS output, `3` image input |
| 6 | 2 | Reserved |
| 8 | 4 | Stream ID |
| 12 | 4 | Sequence number |
| 16 | 4 | Sample rate |
| 20 | 4 | PCM payload byte count |

ASR input is always 16 kHz. The server rejects missing/out-of-order frames,
recordings over 30 seconds, and audio queued beyond the YAML limit. TTS frames
carry the model sample rate; the browser resamples to its `AudioContext` rate
before adding samples to the playback worklet.

## Scheduling

The shared JavaScript `SpeechController` owns sentence segmentation, logical
streams, PCM replay, one-segment lookahead, highlighting metadata, pause/resume,
replacement, and terminal sequencing for both desktop and Wasm. The native
server exposes only one-segment TTS primitives.

The desktop adapter uses one WebSocket for chat/ASR/images and a second for
primitive TTS. The native request gate allows one chat and one TTS request to
overlap, matching the former native voice pipeline. Distinct model kinds may
overlap; duplicate requests of one kind are rejected. Cancellation and
disconnects are checked between generation steps, Whisper stages, and TTS
diffusion steps.
Every cancel receives a terminal acknowledgement, including an already-finished
native request.

The browser uses an `AudioWorklet` for microphone downsampling and another
worklet for TTS playback. Audio never enters JSON, and `WebSocket.bufferedAmount`
provides client-side backpressure before the server's bounded queue limit.

## Voice mode

Read aloud and continuous voice mode share the same frontend speech controller,
PCM framing, playback queue, cancellation, and voice-preference path. The
controller drives the turn sequence:

```text
ASR final -> chat.start -> shared text segmentation -> primitive tts.start -> PCM playback
```

The capture worklet computes frame RMS for browser-side endpointing. Once speech
has exceeded the configured minimum and silence reaches the YAML VAD threshold,
capture finalizes automatically. ASR text and the assistant response remain in
normal conversation history, and listening resumes only after the TTS worklet
reports that playback drained. Clicking the waveform again cancels the current
stage and exits voice mode.

Every voice-mode activation increments a client epoch carried through ASR,
chat, TTS, playback drain, and delayed listen restarts. Exiting voice mode
invalidates the epoch before sending cancellation, so stale completion events
or timers cannot re-enter the loop.

### Voice latency benchmark

Development builds record user-visible timing for every completed voice turn.
The clock starts at the last microphone frame above the VAD threshold and ends
when the playback worklet begins consuming the first synthesized segment. Open
the browser developer console while running with `--dev`, complete several
voice turns, then inspect:

```js
kidiVoiceLatency.samples
kidiVoiceLatency.summary()
```

Each completed turn also emits a `kidi_voice_latency|{...}` console record.
`endpoint_ms` measures VAD silence, `transcription_ms` measures final ASR,
`dispatch_ms` covers the ASR-to-chat handoff, `first_token_ms` and
`end_to_end_ms` are measured from the user's last speech, and the remaining
fields isolate text segmentation, TTS first-PCM, and browser playback queueing.
Call `kidiVoiceLatency.clear()` before a benchmark run. The summary reports p50
and p95 across the last 100 completed turns.

## Accelerator placement

The example uses `auto` for Gemma and OmniVoice and CPU for Whisper. On Apple,
`auto` selects Metal; on systems without a supported GPU it falls back to CPU.
Whisper remains CPU-only. On the tested 16 GB Apple M5:

- Gemma load plus 64-token generation: 5.00 s CPU, 3.78 s Metal.
- OmniVoice 2-second, 10-step synthesis: 2.23 s CPU, 0.88 s Metal.
- Warm voice pipeline: first chat delta about 0.40-0.50 s and first PCM about 1.16 s.
- Combined process after a real voice turn: about 5.55 GB.
- After speech-model idle eviction: about 4.69 GB.

Gemma is warmed before the UI opens. When `runtime.preload_speech` is enabled,
Whisper and the default OmniVoice voice prompt warm in the background after the
loopback server starts. The first voice-mode recording therefore overlaps model
preparation instead of paying it after chat completion.

Desktop and Wasm now use the same sentence policy: the first segment is capped
at 96 weighted units with a 64-unit natural-clause target, while later complete
sentences remain whole through 192 units. Exactly one segment may synthesize
ahead of playback. The same controller emits source offsets and segment IDs for
highlighting on both backends.

Metal FP16 projection expansion is bounded separately per constructed model:
512 MiB for Gemma and 1 GiB for OmniVoice in the example. An unlimited cache
made Gemma peak near 8.1 GB by itself; 512 MiB retained most of its speed
(4.22 s instead of 3.78 s in the same load-plus-64-token check) while lowering
that isolated peak to about 4.75 GB. OmniVoice requires roughly 898 MiB for its
expanded cache, so lower limits fail explicitly rather than silently falling
back.

Every single-segment synthesis now releases request workspaces after its PCM has
been handed to the stream. This removed the largest long-lived allocation after
voice turns. Whisper, OmniVoice, and cached voice prompts are also unloaded
after `runtime.idle_model_timeout_seconds`; the next voice turn reloads them
lazily. CPU chat plus Metal TTS and Metal chat plus CPU TTS were both worse
resident-memory placements on this machine (about 11 GB and 15 GB respectively),
so the default keeps both large generative models on bounded Metal.
