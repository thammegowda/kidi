# Kidi Desktop Streaming Protocol

`kidi-app CONFIG` loads one immutable YAML configuration, starts a
capability-protected loopback server, and serves the embedded interface. HTTP is
used only for bootstrap assets and `/api/status`; interactive work uses one
authenticated WebSocket at `/ws`.

## Text messages

Every command and event is a JSON object:

```json
{"v":1,"id":"request UUID","type":"chat.start","seq":0,"payload":{}}
```

Client commands are `chat.start`, `asr.start`, `asr.stop`, `tts.start`,
`settings.system_prompt.set`, and `cancel`. Server events are `session.ready`,
`settings.system_prompt.updated`, modality-specific `started`, `partial`/`delta`,
and `completed` events, `request.cancelled`, and `error`. Sequence numbers are
monotonic within each request.

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

## Binary audio frames

Audio uses PCM16 little-endian mono data. Each binary WebSocket message starts
with this 24-byte little-endian header:

| Offset | Size | Meaning |
|---:|---:|---|
| 0 | 4 | ASCII `KDA1` |
| 4 | 1 | Protocol version (`1`) |
| 5 | 1 | Kind: `1` ASR input, `2` TTS output |
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

The default scheduler permits one active chat, ASR, or TTS inference request.
Network callbacks only validate and enqueue work. Model execution runs on a
background task, and WebSocket writes are serialized. Cancellation and socket
disconnects are checked between generation steps, Whisper stages, and TTS
diffusion steps.

The browser uses an `AudioWorklet` for microphone downsampling and another
worklet for TTS playback. Audio never enters JSON, and `WebSocket.bufferedAmount`
provides client-side backpressure before the server's bounded queue limit.

## Voice mode

Read aloud and continuous voice mode share the same TTS request,
PCM-framing, playback queue, cancellation, and voice-preference path. Read aloud
starts `tts.start` from a selected assistant message. The waveform control
starts a persistent turn orchestrator:

```text
ASR final -> chat.start -> streamed chat completion -> tts.start -> PCM playback
```

The capture worklet computes frame RMS for browser-side endpointing. Once speech
has exceeded the configured minimum and silence reaches the YAML VAD threshold,
capture finalizes automatically. ASR text and the assistant response remain in
normal conversation history, and listening resumes only after the TTS worklet
reports that playback drained. The single-inference scheduler keeps stages
sequential. Clicking the waveform again cancels the current stage and exits
voice mode.

Every voice-mode activation increments a client epoch carried through ASR,
chat, TTS, playback drain, and delayed listen restarts. Exiting voice mode
invalidates the epoch before sending cancellation, so stale completion events
or timers cannot re-enter the loop.

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

For voice mode, the first speech segment is word-bounded near 48 characters and
uses eight diffusion steps. Later segments use natural clause/sentence
boundaries, a 120-character cap, and the configured quality schedule. Once the
first segment enters the queue, Gemma briefly yields Metal submission until the
server reports `voice.audio.first`; generation then resumes while speech plays.
In the measured pipeline first PCM arrived about 0.74 seconds after the first
displayed token. A more aggressive 36-character/six-step profile improved only
about 22 ms and was rejected to preserve first-phrase quality.

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
