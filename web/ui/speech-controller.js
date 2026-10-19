import {READER_BLOCK_BREAK} from './reader-position.js';

export const MAX_SPEECH_SEGMENT_UNITS = 192;
export const SPEECH_BLOCK_PAUSE_MS = 180;
const FIRST_MAX_SPEECH_SEGMENT_UNITS = 128;
const FIRST_TARGET_UNITS = 48;
const NEXT_TARGET_UNITS = 160;
const wideScript = /[\p{Script=Han}\p{Script=Hiragana}\p{Script=Katakana}\p{Script=Hangul}]/u;
const graphemes = new Intl.Segmenter(undefined, {granularity: 'grapheme'});
const sentences = new Intl.Segmenter(undefined, {granularity: 'sentence'});
const abbreviation = /\b(?:[A-Z]|Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|(?:[A-Za-z]\.)+[A-Za-z])\.$/u;

// Host adapter contract: send one-segment `tts.start`/speech-scoped `cancel` commands, forward
// `tts.started|progress|completed`, `request.cancelled`, and `error` events to handleEvent, and forward ordered
// Float32 PCM frames `{streamId, sequence, sampleRate, samples}` to handlePcm. The controller owns all user-visible
// speech behavior and deliberately has no DOM, Worker, WebSocket, Cache Storage, or native-runtime dependency.

export function speechUnits(text) {
  let units = 0;
  for (const {segment} of graphemes.segment(text)) units += wideScript.test(segment) ? 3 : segment.length;
  return units;
}

export function plainSpeechText(source) {
  return source
    .replace(/!\[([^\]]*)\]\([^)]*\)/gu, '$1')
    .replace(/\[([^\]]+)\]\([^)]*\)/gu, '$1')
    .replace(/<[^>]*>/gu, ' ')
    .replace(/^\s{0,3}(?:(?:#{1,6}|[-+*]|\d+[.)])(?:\s+|$)|>\s*)/gmu, '')
    .replace(/^\s*(?:[-*_]\s*){3,}$/gmu, '')
    .replace(/[*_`~]+/gu, '')
    .replace(/\|+/gu, ' ')
    .replace(/\s+/gu, ' ')
    .trim();
}

function safeMarkdownBoundary(text, proposed) {
  const label = text.lastIndexOf('[', proposed - 1);
  if (label < 0) return proposed;
  const destination = text.indexOf('](', label + 1);
  if (destination < 0 || destination >= proposed) return proposed;
  const close = text.indexOf(')', destination + 2);
  if (close < proposed) return proposed;
  const construct = label > 0 && text[label - 1] === '!' ? label - 1 : label;
  return construct ? construct : close + 1;
}

const LATENCY_MILESTONES = [
  'speech_end',
  'capture_stop',
  'asr_complete',
  'chat_request',
  'first_token',
  'tts_request',
  'first_pcm',
  'playback_start',
];

function roundedLatency(value) {
  return Math.round(value * 10) / 10;
}

function latencyPercentile(values, fraction) {
  if (!values.length) return null;
  const sorted = [...values].sort((left, right) => left - right);
  return roundedLatency(sorted[Math.max(0, Math.ceil(sorted.length * fraction) - 1)]);
}

export class VoiceLatencyBenchmark {
  #now;
  #logger;
  #maximumSamples;
  #active = null;

  constructor({now = () => performance.now(), logger = sample =>
    console.info(`kidi_voice_latency|${JSON.stringify(sample)}`), maximumSamples = 100} = {}) {
    if (typeof now !== 'function' || typeof logger !== 'function' ||
        !Number.isSafeInteger(maximumSamples) || maximumSamples < 1)
      throw new Error('Invalid voice latency benchmark options');
    this.#now = now;
    this.#logger = logger;
    this.#maximumSamples = maximumSamples;
    this.samples = [];
  }

  begin(speechEnd, captureStop = this.#now()) {
    if (![speechEnd, captureStop].every(Number.isFinite) || captureStop < speechEnd)
      throw new Error('Invalid voice latency start');
    this.#active = {speech_end: speechEnd, capture_stop: captureStop};
  }

  mark(name, at = this.#now()) {
    if (!this.#active || !LATENCY_MILESTONES.includes(name) || !Number.isFinite(at)) return;
    if (this.#active[name] === undefined) this.#active[name] = at;
  }

  note(values) {
    if (!this.#active || !values || typeof values !== 'object') return;
    this.#active.notes ||= {};
    for (const [name, value] of Object.entries(values))
      if (Number.isFinite(value) && this.#active.notes[name] === undefined) this.#active.notes[name] = value;
  }

  complete(at = this.#now()) {
    this.mark('playback_start', at);
    if (!this.#active?.playback_start) return null;
    const marks = this.#active;
    this.#active = null;
    const sinceEnd = name => marks[name] === undefined ? null :
      roundedLatency(marks[name] - marks.speech_end);
    const between = (start, end) => marks[start] === undefined || marks[end] === undefined ? null :
      roundedLatency(marks[end] - marks[start]);
    const sample = {
      endpoint_ms: between('speech_end', 'capture_stop'),
      transcription_ms: between('capture_stop', 'asr_complete'),
      dispatch_ms: between('asr_complete', 'chat_request'),
      first_token_ms: sinceEnd('first_token'),
      text_to_tts_ms: between('first_token', 'tts_request'),
      tts_first_pcm_ms: between('tts_request', 'first_pcm'),
      playback_queue_ms: between('first_pcm', 'playback_start'),
      end_to_end_ms: sinceEnd('playback_start'),
      ...(marks.notes || {}),
    };
    this.samples.push(sample);
    if (this.samples.length > this.#maximumSamples) this.samples.shift();
    this.#logger(sample);
    return sample;
  }

  cancel() {
    this.#active = null;
  }

  clear() {
    this.cancel();
    this.samples.length = 0;
  }

  summary() {
    const result = {turns: this.samples.length};
    for (const name of Object.keys(this.samples[0] || {})) {
      const values = this.samples.map(sample => sample[name]).filter(Number.isFinite);
      result[name] = {p50: latencyPercentile(values, 0.5), p95: latencyPercentile(values, 0.95)};
    }
    return result;
  }
}

export class SpeechSegments {
  #buffer = '';
  #offset = 0;
  #count = 0;
  #finished = false;
  #firstMaximumUnits;
  #firstTargetUnits;
  #nextTargetUnits;
  #maximumUnits;

  constructor({
    firstMaximumUnits = FIRST_MAX_SPEECH_SEGMENT_UNITS,
    firstTargetUnits = FIRST_TARGET_UNITS,
    nextTargetUnits = NEXT_TARGET_UNITS,
    maximumUnits = MAX_SPEECH_SEGMENT_UNITS,
  } = {}) {
    const values = [firstMaximumUnits, firstTargetUnits, nextTargetUnits, maximumUnits];
    if (!values.every(value => Number.isSafeInteger(value) && value > 0) ||
        firstTargetUnits > firstMaximumUnits || nextTargetUnits > maximumUnits)
      throw new Error('Invalid speech segmentation profile');
    this.#firstMaximumUnits = firstMaximumUnits;
    this.#firstTargetUnits = firstTargetUnits;
    this.#nextTargetUnits = nextTargetUnits;
    this.#maximumUnits = maximumUnits;
  }

  push(text, finished = false) {
    if (this.#finished || typeof text !== 'string') throw new Error('Invalid speech text stream');
    this.#buffer += text;
    const result = [];
    for (;;) {
      const leading = this.#buffer.match(/^[^\S\u2029]+/u)?.[0].length || 0;
      this.#offset += leading;
      this.#buffer = this.#buffer.slice(leading);
      if (!this.#buffer) break;
      const target = this.#count ? this.#nextTargetUnits : this.#firstTargetUnits;
      const limit = this.#count ? this.#maximumUnits : this.#firstMaximumUnits;
      const block = this.#buffer.indexOf(READER_BLOCK_BREAK);
      if (block === 0) {
        this.#buffer = this.#buffer.slice(READER_BLOCK_BREAK.length);
        this.#offset += READER_BLOCK_BREAK.length;
        continue;
      }
      if (block > 0 && speechUnits(this.#buffer.slice(0, block)) <= limit) {
        const value = this.#buffer.slice(0, block).trimEnd();
        const spoken = plainSpeechText(value);
        if (spoken) result.push({
          id: this.#count++, text: spoken, start: this.#offset,
          end: this.#offset + value.length, pauseAfterMs: SPEECH_BLOCK_PAUSE_MS,
        });
        this.#buffer = this.#buffer.slice(block + READER_BLOCK_BREAK.length);
        this.#offset += block + READER_BLOCK_BREAK.length;
        continue;
      }
      let bounded = 0, units = 0, word = 0, clause = 0;
      for (const {segment, index} of graphemes.segment(this.#buffer)) {
        const next = units + (wideScript.test(segment) ? 3 : segment.length);
        if (next > limit) break;
        units = next;
        bounded = index + segment.length;
        if (/\s/u.test(segment)) {
          word = index;
        }
        if (!clause && units >= target && /[,;:，；：]/u.test(segment)) clause = bounded;
      }
      if (!bounded) throw new Error('Speech grapheme exceeds the segment budget');
      let end = 0;
      for (const {segment, index} of sentences.segment(this.#buffer)) {
        const candidate = this.#buffer.slice(0, index + segment.length).trimEnd();
        if (speechUnits(candidate) > limit) break;
        if (abbreviation.test(candidate)) continue;
        if ((finished || index + segment.length < this.#buffer.length ||
            /[.!?。！？]["'”’)\]]*\s+$/u.test(segment)) && /[.!?。！？]["'”’)\]]*\s*$/u.test(segment))
          end = candidate.length;
        break;
      }
      if (!end && clause) end = clause;
      if (!end && bounded < this.#buffer.length) end = word || bounded;
      if (!end && finished) end = this.#buffer.length;
      if (!end) break;
      end = safeMarkdownBoundary(this.#buffer, end);
      if (!end) break;
      const raw = this.#buffer.slice(0, end), value = raw.trimEnd();
      const blockTail = this.#buffer.slice(end).match(/^[^\S\u2029]*\u2029/u)?.[0] || '';
      const spoken = plainSpeechText(value);
      if (spoken) result.push({id: this.#count++, text: spoken,
        start: this.#offset, end: this.#offset + value.length,
        ...(blockTail ? {pauseAfterMs: SPEECH_BLOCK_PAUSE_MS} : {})});
      const consumed = end + blockTail.length;
      this.#buffer = this.#buffer.slice(consumed);
      this.#offset += consumed;
    }
    this.#finished = finished;
    return result;
  }
}

export class RecentAudio {
  #entries = new Map();
  #bytes = 0;

  constructor({maximumBytes = 16 * 1024 * 1024, maximumEntries = 48,
    lifetimeMs = 600000, now = Date.now} = {}) {
    if (![maximumBytes, maximumEntries, lifetimeMs].every(value => Number.isSafeInteger(value) && value > 0) ||
        typeof now !== 'function') throw new Error('Invalid recent audio limits');
    this.maximumBytes = maximumBytes;
    this.maximumEntries = maximumEntries;
    this.lifetimeMs = lifetimeMs;
    this.now = now;
  }

  get bytes() { return this.#bytes; }

  get(key) {
    const entry = this.#entries.get(key);
    if (!entry) return null;
    if (this.now() - entry.created >= this.lifetimeMs) {
      this.#remove(key);
      return null;
    }
    this.#entries.delete(key);
    this.#entries.set(key, entry);
    return {audio: entry.audio.slice(0), sample_rate: entry.sampleRate, cached: true};
  }

  put(key, audio, sampleRate) {
    if (!(audio instanceof ArrayBuffer) || !audio.byteLength || audio.byteLength % 4 ||
        !Number.isInteger(sampleRate) || sampleRate <= 0 ||
        new Float32Array(audio).some(value => !Number.isFinite(value)))
      throw new Error('Invalid replay audio');
    this.#remove(key);
    if (audio.byteLength > this.maximumBytes) return;
    for (const [oldest, entry] of this.#entries)
      if (this.now() - entry.created >= this.lifetimeMs) this.#remove(oldest);
    while (this.#entries.size >= this.maximumEntries || this.#bytes + audio.byteLength > this.maximumBytes)
      this.#remove(this.#entries.keys().next().value);
    const copy = audio.slice(0);
    this.#entries.set(key, {audio: copy, sampleRate, created: this.now()});
    this.#bytes += copy.byteLength;
  }

  #remove(key) {
    const entry = this.#entries.get(key);
    if (!entry) return;
    this.#entries.delete(key);
    this.#bytes -= entry.audio.byteLength;
  }
}

function combine(chunks) {
  if (chunks.length === 1) {
    const chunk = chunks[0];
    return chunk.byteOffset || chunk.byteLength !== chunk.buffer.byteLength ? chunk.slice() : chunk;
  }
  const length = chunks.reduce((total, chunk) => total + chunk.length, 0);
  const result = new Float32Array(length);
  let offset = 0;
  for (const chunk of chunks) {
    result.set(chunk, offset);
    offset += chunk.length;
  }
  return result;
}

function appendSilence(samples, sampleRate, milliseconds) {
  const frames = Math.max(0, Math.round(sampleRate * milliseconds / 1000));
  if (!frames) return samples;
  const result = new Float32Array(samples.length + frames);
  result.set(samples);
  return result;
}

export class SpeechController {
  #send;
  #nextWireStream;
  #onStarted;
  #onProgress;
  #onSegment;
  #onRequest;
  #onWirePcm;
  #onPrimitiveCompleted;
  #onPcm;
  #onCompleted;
  #onError;
  #onCancelled;
  #cache = new RecentAudio();
  #current = null;

  constructor({send, nextWireStream, onStarted, onProgress, onSegment, onRequest, onWirePcm,
    onPrimitiveCompleted, onPcm, onCompleted, onError, onCancelled}) {
    this.#send = send;
    this.#nextWireStream = nextWireStream;
    this.#onStarted = onStarted;
    this.#onProgress = onProgress;
    this.#onSegment = onSegment;
    this.#onRequest = onRequest;
    this.#onWirePcm = onWirePcm;
    this.#onPrimitiveCompleted = onPrimitiveCompleted;
    this.#onPcm = onPcm;
    this.#onCompleted = onCompleted;
    this.#onError = onError;
    this.#onCancelled = onCancelled;
  }

  get active() { return this.#current; }

  start({id, streamId, text = '', options = {}, voice = false, finished = true, cacheIdentity = '',
    segmentation}) {
    if (this.#current) throw new Error('Another speech stream is active');
    const current = {
      id, streamId, options, voice, cacheIdentity, split: new SpeechSegments(segmentation), queue: [],
      ended: false, primitive: null, delivery: null, delivering: false, finishing: false,
      sequence: 0, samples: 0, chunks: 0, sampleRate: null, cancelled: false,
    };
    this.#current = current;
    this.#onStarted?.(current);
    this.append(id, text, finished);
    return current;
  }

  append(id, text, finished = false) {
    const current = this.#current;
    if (!current || current.id !== id || current.cancelled) return;
    current.queue.push(...current.split.push(text, finished));
    current.ended = finished;
    this.#pump(current);
  }

  async cancel(id, message = 'Speech playback stopped') {
    const current = this.#current;
    if (!current || current.id !== id) return;
    if (current.cancelPromise) return current.cancelPromise;
    current.cancelled = true;
    current.queue.length = 0;
    if (!current.primitive) return this.#finishCancelled(current, message);
    current.cancelPromise = new Promise(resolve => { current.cancelResolve = resolve; });
    try {
      this.#send('cancel', current.primitive.requestId, {scope: 'speech', message});
    } catch (error) {
      this.#finishCancelled(current, message);
      throw error;
    }
    return current.cancelPromise;
  }

  reset(message = 'Speech transport disconnected') {
    const current = this.#current;
    if (!current) return;
    current.cancelled = true;
    current.queue.length = 0;
    this.#finishCancelled(current, message);
  }

  handleEvent(event) {
    const current = this.#current;
    const primitive = current?.primitive;
    if (!primitive || event.id !== primitive.requestId) return false;
    if (event.type === 'tts.progress') {
      this.#onProgress?.(current, event.payload || {});
      return true;
    }
    if (event.type === 'tts.started') return true;
    if (event.type === 'tts.completed') {
      if (current.cancelled) {
        this.#finishCancelled(current, event.payload?.message || 'Speech playback stopped');
        return true;
      }
      this.#onPrimitiveCompleted?.(current, primitive, event.payload || {});
      this.#completePrimitive(current, primitive, event.payload || {}).catch(error => this.#fail(current, error));
      return true;
    }
    if (event.type === 'request.cancelled') {
      this.#finishCancelled(current, event.payload?.message || 'Speech playback stopped');
      return true;
    }
    if (event.type === 'error') {
      this.#fail(current, new Error(event.payload?.message || 'Speech synthesis failed'));
      return true;
    }
    return false;
  }

  handlePcm({streamId, sequence, sampleRate, samples}) {
    const primitive = this.#current?.primitive;
    if (!primitive || primitive.wireStreamId !== streamId) return false;
    if (!(samples instanceof Float32Array) || !samples.length || !Number.isFinite(sampleRate) || sampleRate <= 0 ||
        sequence !== primitive.sequence++)
      throw new Error('Invalid or out-of-order speech PCM');
    if (primitive.sampleRate !== null && primitive.sampleRate !== sampleRate)
      throw new Error('Speech segment changed sample rate');
    primitive.sampleRate = sampleRate;
    primitive.audio.push(samples);
    this.#onWirePcm?.(this.#current, primitive, {sequence, sampleRate, samples});
    return true;
  }

  #pump(current) {
    if (this.#current !== current || current.cancelled || current.primitive ||
        current.delivering || current.finishing) return;
    const segment = current.queue.shift();
    if (!segment) {
      if (current.ended) this.#finish(current);
      return;
    }
    segment.cacheKey = JSON.stringify([
      'shared-speech-v1', current.cacheIdentity, current.options, segment.text,
    ]);
    const cached = this.#cache.get(segment.cacheKey);
    if (cached) {
      this.#deliverInOrder(current, segment, cached.sample_rate, new Float32Array(cached.audio), true)
        .catch(error => this.#fail(current, error));
      return;
    }
    const primitive = {
      requestId: crypto.randomUUID(),
      wireStreamId: this.#nextWireStream(),
      segment,
      sequence: 0,
      sampleRate: null,
      audio: [],
    };
    current.primitive = primitive;
    try {
      this.#onRequest?.(current, segment, primitive);
      this.#send('tts.start', primitive.requestId, {
        stream_id: primitive.wireStreamId,
        text: segment.text,
        options: current.options,
      });
    } catch (error) {
      current.primitive = null;
      this.#fail(current, error);
    }
  }

  async #completePrimitive(current, primitive, payload) {
    if (this.#current !== current || current.primitive !== primitive || current.cancelled) return;
    if (!primitive.audio.length) throw new Error('Speech synthesis completed without PCM');
    const sampleRate = primitive.sampleRate || payload.sample_rate;
    const samples = combine(primitive.audio);
    current.primitive = null;
    this.#cache.put(primitive.segment.cacheKey, samples.buffer, sampleRate);
    await this.#deliverInOrder(current, primitive.segment, sampleRate, samples, false);
  }

  async #deliverInOrder(current, segment, sampleRate, samples, cached) {
    if (current.delivering) throw new Error('Speech delivery overlap');
    // append() may run while delivery waits for playback capacity; keep later segments behind this one.
    current.delivering = true;
    try {
      await this.#deliver(current, segment, sampleRate, samples, cached);
    } finally {
      current.delivering = false;
    }
    this.#pump(current);
  }

  async #deliver(current, segment, sampleRate, samples, cached) {
    const previous = current.delivery;
    current.delivery = null;
    if (previous) await previous;
    if (this.#current !== current || current.cancelled) return;
    if (current.sampleRate !== null && current.sampleRate !== sampleRate)
      throw new Error('Speech segments changed sample rate');
    current.sampleRate = sampleRate;
    this.#onSegment?.(current, segment, {cached});
    const delivered = segment.pauseAfterMs ?
      appendSilence(samples, sampleRate, segment.pauseAfterMs) : samples;
    current.delivery = Promise.resolve(this.#onPcm?.(current, {
      streamId: current.streamId,
      sequence: current.sequence++,
      sampleRate,
      samples: delivered,
      segmentId: segment.id,
    }));
    current.samples += delivered.length;
    current.chunks++;
  }

  #finish(current) {
    if (current.finishing) return;
    if (current.delivery) {
      current.finishing = true;
      const delivery = current.delivery;
      current.delivery = null;
      delivery.then(() => {
        if (this.#current !== current || current.cancelled) return;
        current.finishing = false;
        this.#finish(current);
      }, error => this.#fail(current, error));
      return;
    }
    this.#current = null;
    this.#onCompleted?.(current);
  }

  #finishCancelled(current, message) {
    if (this.#current !== current) return;
    this.#current = null;
    current.cancelResolve?.();
    this.#onCancelled?.(current, message);
  }

  #fail(current, error) {
    if (this.#current !== current) return;
    if (current.primitive) {
      try { this.#send('cancel', current.primitive.requestId, {scope: 'speech', message: error.message}); }
      catch (cancelError) { console.warn('Unable to cancel failed speech request:', cancelError); }
    }
    this.#current = null;
    current.cancelResolve?.();
    this.#onError?.(current, error);
  }
}
