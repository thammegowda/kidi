import {isModelCached, webGpuSupport, listCachedFiles, deleteCachedFile, deleteCachedFiles, clearModelCache} from './model-cache.mjs';
import {createGpuWorker} from './runtime-client.mjs';
import {SpeechSegments} from './speech-segments.mjs';
import {RecentAudio} from './recent-audio.mjs';

const SETTINGS_KEY = 'kidi-wasm-host-v1';
const DEFAULTS = {
  model: 'google/gemma-4-E2B-it-qat-mobile-transformers',
  backend: 'webgpu',
  threads: 4,
  maximumTokens: 512,
  systemPrompt: 'You are a helpful local AI assistant. Respond directly, briefly, and clearly.',
};

class WasmBackend {
  #callbacks;
  #asr;
  #components = {
    chat: {
      enabled: true,
      directory: 'Browser model cache',
      accelerator: 'wasm',
      precision: 'checkpoint',
      kv_cache_precision: 'auto',
      system_prompt: DEFAULTS.systemPrompt,
      configured_system_prompt: DEFAULTS.systemPrompt,
      maximum_tokens: DEFAULTS.maximumTokens,
      context_tokens: 9216,
      prefix_cache_mb: 0,
    },
    asr: {
      enabled: true,
      accelerator: 'wasm-cpu',
      directory: 'openai/whisper-tiny',
      settings: {
        language: 'auto',
        max_tokens: 128,
        vad: {threshold: 0.012, minimum_speech_ms: 250, silence_ms: 650},
      },
    },
    tts: {
      enabled: true,
      accelerator: 'wasm-cpu',
      directory: './models/omnivoice/manifest.json',
      settings: {
        language: 'auto',
        steps: 10,
        pace: 1.15,
        gpu_precision: 'f32',
        reference_language: 'English',
        reference_text: 'Hello, I am Kidi, your helpful local assistant.',
        voice: {gender: 'female', age: 'young adult', pitch: 'moderate', style: '', accent: ''},
      },
    },
  };
  #current;
  #ready = false;
  #settings = {...DEFAULTS};
  #speechReady;
  #speechWorker;
  #speechMode;
  #ttsPending = new Map();
  #ttsReady;
  #ttsWorker;
  #ttsLoadingId;
  #ttsLoadReject;
  #ttsMode;
  #recentAudio = new RecentAudio();
  #worker;
  #resident = false;
  #reloading = false;
  #nextAudioStream = 0x80000000;

  constructor() {
    try {
      const saved = JSON.parse(localStorage.getItem(SETTINGS_KEY) || '{}');
      if (typeof saved.model === 'string') this.#settings.model = saved.model;
      if (['cpu', 'webgpu'].includes(saved.backend)) this.#settings.backend = saved.backend;
      if (Number.isInteger(saved.threads)) this.#settings.threads = saved.threads;
      if (Number.isInteger(saved.maximumTokens)) this.#settings.maximumTokens = saved.maximumTokens;
      if (typeof saved.systemPrompt === 'string') {
        this.#settings.systemPrompt = saved.systemPrompt;
        this.#components.chat.system_prompt = saved.systemPrompt;
      }
    } catch {}
  }

  #ensureTts(id, options) {
    const precision = options?.gpu_precision || 'f32';
    if (!['f32', 'mixed-f16', 'w8a8'].includes(precision))
      return Promise.reject(new Error('Unsupported browser speech precision'));
    const mode = JSON.stringify({backend: this.#settings.backend, precision});
    if (this.#ttsReady && this.#ttsMode !== mode) {
      this.#ttsWorker?.terminate();
      this.#ttsWorker = null;
      this.#ttsReady = null;
    }
    if (this.#ttsReady) return this.#ttsReady;
    this.#ttsMode = mode;
    this.#ttsLoadingId = id;
    this.#ttsReady = new Promise((resolve, reject) => {
      this.#ttsLoadReject = reject;
      const availableThreads = Number(globalThis.navigator?.hardwareConcurrency);
      const preferredGpu = this.#settings.backend === 'webgpu' && !webGpuSupport();
      const start = useGpu => {
        const worker = useGpu ? createGpuWorker('tts') :
          new Worker(new URL('./tts-worker.mjs', import.meta.url), {type: 'module'});
        this.#ttsWorker = worker;
        let loaded = false;
        const failLoad = message => {
          if (worker !== this.#ttsWorker) return;
          worker.terminate();
          if (useGpu) {
            this.#event(id, 'tts.progress', {
              phase: 'fallback',
              message: `WebGPU speech unavailable; using CPU (${message})`,
            });
            start(false);
          } else {
            this.#ttsWorker = null;
            this.#ttsReady = null;
            this.#ttsLoadingId = null;
            this.#ttsLoadReject = null;
            reject(new Error(message));
          }
        };
        worker.onmessage = ({data}) => {
          if (worker !== this.#ttsWorker) return;
          if (data.type === 'ready') {
            loaded = true;
            this.#components.tts.threads = data.threads;
            this.#components.tts.accelerator = data.backend ||
              (data.threads > 1 ? 'wasm-threads' : 'wasm-cpu');
            this.#components.tts.gpu_precision = data.matrix_precision;
            this.#ttsLoadingId = null;
            this.#ttsLoadReject = null;
            resolve();
          } else if (data.type === 'progress') {
            this.#event(id, 'tts.progress', {phase: 'loading', ...data});
          } else if (data.type === 'generating') {
            this.#event(id, 'tts.progress', {
              phase: 'synthesizing',
              step: data.step,
              steps: data.steps,
            });
          } else if (data.type === 'audio') {
            this.#ttsAudio(data);
          } else if (data.type === 'error') {
            if (!loaded) {
              failLoad(data.error || 'TTS worker failed');
            } else {
              const pending = this.#ttsPending.get(data.requestId) || this.#ttsPending.values().next().value;
              if (pending) this.#failSpeech(pending, data.error || 'Speech synthesis failed');
            }
          }
        };
        worker.onerror = event => {
          if (worker !== this.#ttsWorker) return;
          if (!loaded) failLoad(event.message || 'TTS worker failed');
          else {
            const pending = this.#ttsPending.values().next().value;
            if (pending) this.#failSpeech(pending, event.message || 'TTS worker failed', 'tts_worker_failed');
          }
        };
        const threads = !useGpu && globalThis.crossOriginIsolated
          ? Math.min(8, Number.isInteger(availableThreads) && availableThreads > 0
            ? availableThreads : this.#settings.threads)
          : 1;
        worker.postMessage({
          type: 'load',
          source: this.#components.tts.directory,
          backend: useGpu ? 'webgpu' : 'cpu',
          threads,
          cacheOnly: false,
          matrixPrecision: precision,
          w8a8Kernel: 'float',
        });
      };
      start(preferredGpu);
    });
    return this.#ttsReady;
  }

  #synthesize(id, streamId, text, options, voice, finished = true) {
    if (this.#ttsPending.size) {
      this.#event(id, voice ? 'voice.audio.error' : 'error', {
        code: 'busy', message: 'Another speech synthesis request is active',
      });
      return false;
    }
    const defaults = this.#components.tts.settings;
    const pending = {id, streamId, voice,
      options: {...defaults, ...options, voice: {...(options?.voice || defaults.voice)},
        use_voice_prompt: false, cache_voice_prompt: false},
      started: performance.now(), split: new SpeechSegments(),
      queue: [], ready: false, ended: false, inFlight: false, delivering: false,
      sequence: 0, samples: 0, chunks: 0, sampleRate: null};
    this.#ttsPending.set(id, pending);
    this.#appendSpeech(id, text, finished);
    return true;
  }

  #appendSpeech(id, text, finished = false) {
    const pending = this.#ttsPending.get(id);
    if (!pending) return;
    try {
      pending.queue.push(...pending.split.push(text, finished));
      pending.ended = finished;
      this.#pumpSpeech(pending);
    } catch (error) { this.#failSpeech(pending, error.message); }
  }

  #pumpSpeech(pending) {
    if (this.#ttsPending.get(pending.id) !== pending || pending.loading || pending.inFlight || pending.delivering) return;
    const segment = pending.queue[0];
    if (segment) {
      const options = pending.options;
      segment.cacheKey = JSON.stringify([this.#components.tts.directory, this.#settings.backend, options, segment.text]);
      const cached = this.#recentAudio.get(segment.cacheKey);
      if (!cached && !pending.ready) {
        pending.loading = true;
        this.#ensureTts(pending.id, options).then(() => {
          if (this.#ttsPending.get(pending.id) !== pending) return;
          pending.loading = false;
          pending.ready = true;
          this.#pumpSpeech(pending);
        }).catch(error => this.#failSpeech(pending, error.message, 'tts_load_failed'));
        return;
      }
      pending.queue.shift();
      pending.active = segment;
      pending.inFlight = true;
      pending.synthesisStarted = performance.now();
      if (cached) {
        this.#ttsAudio({requestId: pending.id, ...cached});
        return;
      }
      this.#event(pending.id, 'tts.progress', {phase: 'synthesizing', segment_id: segment.id});
      try {
        this.#ttsWorker.postMessage({
          type: 'synthesize', requestId: pending.id, streamSegment: true, text: segment.text, options,
        });
      } catch (error) { this.#failSpeech(pending, error.message); }
    } else if (pending.ended) {
      if (!pending.chunks) return this.#failSpeech(pending, 'No speech text was produced');
      this.#ttsPending.delete(pending.id);
      this.#releaseTts();
      this.#event(pending.id, pending.voice ? 'voice.audio.completed' : 'tts.completed', {
        stream_id: pending.streamId, sample_rate: pending.sampleRate, samples: pending.samples,
        chunks: pending.chunks, voice_prompt_reused: false,
      });
    }
  }

  #releaseTts() {
    this.#ttsLoadReject?.(new DOMException('Speech request stopped', 'AbortError'));
    this.#ttsLoadReject = null;
    this.#ttsWorker?.terminate();
    this.#ttsWorker = null;
    this.#ttsReady = null;
    this.#ttsLoadingId = null;
  }

  #cancelSpeech(id) {
    if (!this.#ttsPending.has(id) && this.#ttsLoadingId !== id) return false;
    this.#ttsPending.delete(id);
    if (this.#current?.id === id) this.#current.speak = false;
    this.#releaseTts();
    return true;
  }

  #failSpeech(pending, message, code = 'tts_failed') {
    if (!pending || this.#ttsPending.get(pending.id) !== pending) return;
    this.#ttsPending.delete(pending.id);
    if (this.#current?.id === pending.id) this.#current.speak = false;
    this.#releaseTts();
    this.#event(pending.id, pending.voice ? 'voice.audio.error' : 'error', {code, message});
  }

  async #ttsAudio(data) {
    const pending = this.#ttsPending.get(data.requestId);
    if (!pending) return;
    try {
      if (!pending.inFlight || pending.delivering) throw new Error('Unexpected speech segment audio');
      pending.inFlight = false;
      pending.delivering = true;
      if (!(data.audio instanceof ArrayBuffer) || !data.audio.byteLength || data.audio.byteLength % 4 ||
          !Number.isInteger(data.sample_rate) || data.sample_rate <= 0)
        throw new Error('Speech worker returned invalid PCM');
      const count = data.audio.byteLength / 4;
      const samples = new Float32Array(data.audio);
      if (!data.cached) this.#recentAudio.put(pending.active.cacheKey, data.audio, data.sample_rate);
      if (pending.sampleRate !== null && pending.sampleRate !== data.sample_rate)
        throw new Error('Speech segments changed sample rate');
      pending.sampleRate = data.sample_rate;
      const segment = pending.active;
      this.#event(pending.id, 'tts.segment', {stream_id: pending.streamId, segment_id: segment.id,
        text: segment.text, start: segment.start, end: segment.end,
        cached: data.cached === true, synthesis_ms: performance.now() - pending.synthesisStarted});
      if (pending.voice && !pending.chunks) this.#event(pending.id, 'voice.audio.first', {
        stream_id: pending.streamId, elapsed_ms: performance.now() - pending.started,
      });
      if (this.#callbacks?.onPcm) {
        await this.#callbacks.onPcm({streamId: pending.streamId, sequence: pending.sequence++,
          sampleRate: data.sample_rate, samples, segmentId: segment.id});
      } else {
        if (!this.#callbacks?.onAudio) throw new Error('No speech audio consumer is connected');
        for (let offset = 0; offset < samples.length; offset += 4096) {
          if (this.#ttsPending.get(pending.id) !== pending) return;
          const length = Math.min(4096, samples.length - offset);
          const frame = new ArrayBuffer(24 + length * 2);
          new Uint8Array(frame).set([0x4b, 0x44, 0x41, 0x31, 1, 2], 0);
          const header = new DataView(frame);
          header.setUint32(8, pending.streamId, true);
          header.setUint32(12, pending.sequence++, true);
          header.setUint32(16, data.sample_rate, true);
          header.setUint32(20, length * 2, true);
          const pcm = new Int16Array(frame, 24, length);
          for (let index = 0; index < length; index++)
            pcm[index] = Math.max(-32768, Math.min(32767, Math.round(samples[offset + index] * 32768)));
          await this.#callbacks.onAudio(frame);
        }
      }
      if (this.#ttsPending.get(pending.id) !== pending) return;
      pending.samples += count;
      pending.chunks++;
    } catch (error) {
      this.#failSpeech(pending, error.message, 'tts_audio_failed');
    } finally {
      pending.delivering = false;
      if (this.#ttsPending.get(pending.id) === pending) this.#pumpSpeech(pending);
    }
  }

  get bufferedAmount() {
    return 0;
  }

  async status() {
    return this.#snapshot();
  }

  listCachedFiles() { return listCachedFiles(); }
  deleteCachedFile(id) { return deleteCachedFile(id); }
  deleteCachedFiles(ids) { return deleteCachedFiles(ids); }
  clearModelCache() { return clearModelCache(); }

  connect(callbacks) {
    this.#callbacks = callbacks;
    queueMicrotask(() => {
      callbacks.onOpen?.();
      callbacks.onEvent?.({
        v: 1, id: '', type: 'session.ready', seq: 0,
        payload: {
          ready: this.#ready,
          backend: this.#ready ? this.#settings.backend : '',
          components: this.#components,
          runtime: this.#runtime(),
          config: 'Browser-local settings',
          protocol: 1,
          wire_audio: 'float32',
        },
      });
      isModelCached(this.#settings.model).then(cached => {
        if (cached && !this.#ready && !this.#worker)
          this.#load(crypto.randomUUID(), {...this.#settings, cacheOnly: true});
      }).catch(() => {});
    });
  }

  send(type, id, payload = {}) {
    if (type === 'runtime.load') {
      this.#load(id, {
        model: payload.model,
        backend: payload.backend,
        threads: payload.threads,
        maximumTokens: payload.maximum_tokens,
        cacheOnly: false,
      });
      return;
    }
    if (type === 'tts.start') {
      if (!Number.isInteger(payload.stream_id) || payload.stream_id < 0 || payload.stream_id > 0xffffffff ||
          typeof payload.text !== 'string' || !payload.text)
        return this.#error(id, 'invalid_tts', 'TTS requires text and an unsigned stream_id');
      this.#event(id, 'tts.started', {stream_id: payload.stream_id,
        format: this.#callbacks?.onPcm ? 'pcm_f32le' : 'pcm_s16le'});
      this.#synthesize(id, payload.stream_id, payload.text, payload.options || this.#components.tts.settings, false);
      return;
    }
    if (type === 'settings.system_prompt.set') {
      if (typeof payload.prompt !== 'string' || payload.prompt.length > 32768)
        return this.#error(id, 'invalid_system_prompt', 'System prompt must contain at most 32768 characters');
      this.#settings.systemPrompt = payload.prompt;
      this.#components.chat.system_prompt = payload.prompt;
      this.#save();
      this.#event(id, 'settings.system_prompt.updated', {prompt: payload.prompt});
      return;
    }
    if (type === 'chat.start') {
      if (!this.#ready || !this.#worker) return this.#error(id, 'not_ready', 'Load the browser model first');
      if (this.#current) return this.#error(id, 'busy', 'Another inference request is active');
      const messages = [...(payload.messages || [])];
      if (this.#settings.systemPrompt)
        messages.unshift({role: 'system', content: this.#settings.systemPrompt});
      this.#current = {
        id,
        text: '',
        speak: payload.speak === true,
        streamId: payload.speak === true ? this.#nextAudioStream++ : null,
        ttsOptions: payload.tts_options || this.#components.tts.settings,
      };
      if (this.#current.speak)
        this.#event(id, 'voice.audio.started', {
          stream_id: this.#current.streamId,
          format: this.#callbacks?.onPcm ? 'pcm_f32le' : 'pcm_s16le',
        });
      if (this.#current.speak)
        this.#current.speak = this.#synthesize(id, this.#current.streamId, '', this.#current.ttsOptions, true, false);
      this.#event(id, 'chat.started');
      this.#worker.postMessage({
        type: 'generate',
        messages,
        maximumTokens: this.#settings.maximumTokens,
        imageMaxPixels: 3000000,
      });
      return;
    }
    if (type === 'asr.start') {
      if (this.#asr) return this.#error(id, 'busy', 'Speech recognition is already active');
      if (!payload.stream_id || !Number.isInteger(payload.stream_id))
        return this.#error(id, 'invalid_stream', 'ASR stream_id must be an unsigned integer');
      this.#asr = {
        id,
        streamId: payload.stream_id,
        samples: [],
        sequence: 0,
        transcribing: false,
        finalRequested: false,
        partialSequence: 0,
        lastPartialSamples: 0,
      };
      this.#ensureSpeech().then(() => {
        if (this.#asr?.id === id)
          this.#event(id, 'asr.started', {stream_id: payload.stream_id, sample_rate: 16000});
      }).catch(error => {
        if (this.#asr?.id === id) this.#asr = null;
        this.#error(id, 'asr_load_failed', error.message);
      });
      return;
    }
    if (type === 'asr.stop') {
      if (this.#asr?.id !== id) return;
      this.#asr.finalRequested = true;
      if (!this.#asr.transcribing) this.#transcribe(true);
      return;
    }
    if (type === 'cancel') {
      if (payload.scope === 'speech') {
        const pending = this.#ttsPending.get(id);
        if (pending) {
          this.#cancelSpeech(id);
          this.#event(id, pending.voice ? 'voice.audio.error' : 'request.cancelled',
            {message: payload.message || 'Speech playback stopped'});
        }
        return;
      }
      if (this.#current?.id === id) this.#worker?.postMessage({type: 'cancel'});
      if (this.#asr?.id === id) {
        this.#speechWorker?.postMessage({type: 'cancel'});
        this.#asr = null;
        this.#event(id, 'request.cancelled');
      }
      if (this.#ttsLoadingId === id || this.#ttsPending.has(id)) {
        this.#cancelSpeech(id);
        this.#event(id, 'request.cancelled');
      }
      return;
    }
    this.#error(id, 'unsupported', `The browser backend does not support ${type} yet`);
  }

  sendBinary(frame) {
    if (!(frame instanceof Uint8Array) || frame.byteLength < 24) throw new Error('Invalid binary frame');
    const header = new DataView(frame.buffer, frame.byteOffset, frame.byteLength);
    if (header.getUint8(0) !== 0x4b || header.getUint8(1) !== 0x44 ||
        header.getUint8(2) !== 0x41 || header.getUint8(3) !== 0x31 ||
        header.getUint8(4) !== 1 || header.getUint8(5) !== 1)
      throw new Error('The browser backend accepts only ASR PCM frames');
    const streamId = header.getUint32(8, true);
    const sequence = header.getUint32(12, true);
    const sampleRate = header.getUint32(16, true);
    const bytes = header.getUint32(20, true);
    if (!this.#asr || streamId !== this.#asr.streamId || sequence !== this.#asr.sequence++ ||
        sampleRate !== 16000 || bytes !== frame.byteLength - 24 || bytes % 2)
      throw new Error('ASR PCM frame is invalid or out of order');
    const pcm = new Int16Array(frame.buffer, frame.byteOffset + 24, bytes / 2);
    for (const sample of pcm) this.#asr.samples.push(sample / 32768);
    if (!this.#asr.transcribing && !this.#asr.finalRequested &&
        this.#asr.samples.length >= this.#asr.lastPartialSamples + 12800)
      this.#transcribe(false);
  }

  #load(id, requested) {
    if (this.#current) {
      const previous = this.#current.id;
      this.#cancelSpeech(previous);
      this.#current = null;
      this.#event(previous, 'request.cancelled');
    }
    if (this.#worker) this.#worker.terminate();
    this.#ready = false;
    this.#resident = false;
    this.#reloading = false;
    const threads = Number.isInteger(requested.threads) && requested.threads >= 1 && requested.threads <= 8
      ? requested.threads : 1;
    const selectedBackend = requested.backend === 'webgpu' && !webGpuSupport() ? 'webgpu' : 'cpu';
    this.#settings = {
      ...this.#settings,
      model: requested.model || DEFAULTS.model,
      backend: selectedBackend,
      threads,
      maximumTokens: Number.isInteger(requested.maximumTokens) ? requested.maximumTokens : 512,
    };
    this.#save();
    const worker = selectedBackend === 'webgpu' ? createGpuWorker('chat') :
      new Worker(new URL('./inference-worker.mjs', import.meta.url), {type: 'module'});
    this.#worker = worker;
    this.#event(id, 'runtime.loading', {runtime: this.#runtime()});
    worker.onmessage = ({data}) => this.#workerEvent(id, worker, data);
    worker.onerror = event => {
      if (worker !== this.#worker) return;
      const request = this.#current?.id || id;
      this.#cancelSpeech(request);
      this.#ready = this.#resident = this.#reloading = false;
      this.#worker = null;
      this.#current = null;
      const message = event.message || 'Browser model worker failed';
      this.#event('', 'runtime.availability', {available: false, resident: false, message});
      this.#error(request, 'worker_error', message);
    };
    worker.postMessage({
      type: 'load',
      manifest: this.#settings.model,
      threads: selectedBackend === 'webgpu' ? 1 : threads,
      backend: selectedBackend,
      cacheOnly: requested.cacheOnly === true,
      reload: requested.cacheOnly !== true,
    });
  }

  #ensureSpeech() {
    const preferredGpu = this.#settings.backend === 'webgpu' && !webGpuSupport();
    const mode = JSON.stringify({gpu: preferredGpu, threads: this.#settings.threads});
    if (this.#speechMode !== mode) {
      this.#speechWorker?.terminate();
      this.#speechWorker = null;
      this.#speechReady = null;
      this.#speechMode = mode;
    }
    if (this.#speechReady) return this.#speechReady;
    this.#speechReady = new Promise((resolve, reject) => {
      const start = useGpu => {
        const worker = useGpu ? createGpuWorker('asr') :
          new Worker(new URL('./asr-worker.mjs', import.meta.url), {type: 'module'});
        this.#speechWorker = worker;
        let loaded = false;
        const fail = message => {
          if (worker !== this.#speechWorker) return;
          worker.terminate();
          if (!loaded && useGpu) {
            this.#event(this.#asr?.id || '', 'asr.progress', {
              phase: 'fallback', message: `WebGPU speech recognition unavailable; using CPU (${message})`,
            });
            start(false);
            return;
          }
          this.#speechWorker = null;
          this.#speechReady = null;
          if (!loaded) reject(new Error(message));
          else if (this.#asr) {
            const id = this.#asr.id;
            this.#asr = null;
            this.#error(id, 'asr_failed', message);
          }
        };
        worker.onmessage = ({data}) => {
          if (worker !== this.#speechWorker) return;
          if (data.type === 'ready') {
            loaded = true;
            this.#components.asr.accelerator = data.backend || 'wasm-cpu';
            this.#components.asr.threads = data.threads;
            resolve();
          } else if (data.type === 'result') {
            this.#speechResult(data);
          } else if (data.type === 'partial' && this.#asr && String(data.requestId).includes(this.#asr.id)) {
            this.#event(this.#asr.id, 'asr.partial', {
              stable: '', provisional: data.text || '', language: data.language || 'auto',
            });
          } else if (data.type === 'error') fail(data.error || 'Speech recognition failed');
          else if ((data.type === 'progress' || data.type === 'residency') && this.#asr) {
            this.#event(this.#asr.id, 'asr.progress', {phase: 'loading', ...data});
          }
        };
        worker.onerror = event => fail(event.message || 'Speech worker failed');
        worker.postMessage({
          type: 'load',
          source: this.#components.asr.directory,
          backend: useGpu ? 'webgpu' : 'cpu',
          threads: !useGpu && globalThis.crossOriginIsolated ? Math.max(1, this.#settings.threads) : 1,
        });
      };
      start(preferredGpu);
    });
    return this.#speechReady;
  }

  #transcribe(final) {
    const request = this.#asr;
    if (!request || request.transcribing || !request.samples.length) return;
    request.transcribing = true;
    if (!final) request.lastPartialSamples = request.samples.length;
    const audio = Float32Array.from(request.samples);
    const requestId = final ? `final:${request.id}` : `partial:${request.id}:${request.partialSequence++}`;
    this.#speechWorker.postMessage({
      type: 'transcribe',
      requestId,
      audio: audio.buffer,
      language: 'auto',
      maximumTokens: 128,
    }, [audio.buffer]);
  }

  #speechResult(data) {
    const request = this.#asr;
    if (!request || !String(data.requestId).includes(request.id)) return;
    request.transcribing = false;
    if (String(data.requestId).startsWith('final:')) {
      this.#event(request.id, 'asr.completed', {
        text: data.text || '',
        language: data.language || 'auto',
        stats: {
          feature_ms: data.feature_ms || 0,
          encode_ms: data.encode_ms || 0,
          decode_ms: data.decode_ms || 0,
        },
      });
      this.#asr = null;
    } else {
      this.#event(request.id, 'asr.partial', {
        stable: '',
        provisional: data.text || '',
        language: data.language || 'auto',
      });
      if (request.finalRequested) this.#transcribe(true);
    }
  }

  #workerEvent(loadId, worker, data) {
    if (worker !== this.#worker) return;
    if (data.type === 'residency') {
      this.#resident = data.state === 'ready';
      this.#reloading = data.state === 'loading';
      if (this.#ready) this.#event('', 'runtime.availability', {
        available: true, resident: this.#resident, reloading: this.#reloading,
      });
      return;
    }
    if (data.type === 'progress' || data.type === 'initializing') {
      this.#event(loadId, 'runtime.progress', {
        stage: data.stage || data.file || (data.type === 'initializing' ? 'Initializing model' : 'Loading model'),
        ...data,
      });
      return;
    }

    if (data.type === 'ready') {
      this.#ready = true;
      this.#resident = true;
      this.#reloading = false;
      this.#components.chat = {
        ...this.#components.chat,
        accelerator: data.backend,
        directory: this.#settings.model,
        maximum_tokens: this.#settings.maximumTokens,
      };
      this.#event(loadId, 'runtime.ready', {
        backend: data.backend,
        runtime: this.#runtime(),
        components: this.#components,
      });
      this.#callbacks?.onEvent?.({
        v: 1, id: '', type: 'session.ready', seq: 0,
        payload: {
          ready: true,
          backend: data.backend,
          components: this.#components,
          runtime: this.#runtime(),
          config: 'Browser-local settings',
          protocol: 1,
          wire_audio: 'float32',
        },
      });
      return;
    }
    if (data.type === 'step' && this.#current) {
      for (const item of data.events || []) {
        if (item.text) {
          this.#current.text += item.text;
          this.#event(this.#current.id, 'chat.delta', {text: item.text});
          if (this.#current.speak) this.#appendSpeech(this.#current.id, item.text);
        }
        if (item.completed) {
          const completed = item.completed;
          const current = this.#current;
          const text = completed.text || current.text;
          this.#event(this.#current.id, 'chat.completed', {
            text: completed.text || this.#current.text,
            stats: {
              elapsed_ms: data.stats?.elapsedMs || completed.generation_ms || 0,
              prompt_tokens: completed.prompt_tokens || 0,
              decode_tokens: completed.decode_tokens || data.stats?.decodeTokens || 0,
              prefill_ms: 0,
              decode_ms: completed.decode_ms || data.stats?.decodeMs || 0,
              preparation_ms: 0,
            },
          });
          this.#current = null;
          if (current.speak) {
            if (text.startsWith(current.text)) this.#appendSpeech(current.id, text.slice(current.text.length), true);
            else if (text.trim() === current.text.trim()) this.#appendSpeech(current.id, '', true);
            else this.#failSpeech(this.#ttsPending.get(current.id), 'The streamed response was revised after speech started');
          }
        }
      }
      return;
    }
    if (data.type === 'cancelled' && this.#current) {
      this.#cancelSpeech(this.#current.id);
      this.#event(this.#current.id, 'request.cancelled');
      this.#current = null;
      return;
    }
    if (data.type === 'error') {
      const request = this.#current?.id || loadId;
      this.#cancelSpeech(request);
      this.#error(request, 'wasm_error', data.error || 'Browser runtime failed');
      this.#current = null;
      if (data.fatal) {
        this.#ready = this.#resident = false;
        this.#event('', 'runtime.availability', {
          available: false, resident: false, message: data.error || 'Browser runtime failed',
        });
      }
    }
  }

  #runtime() {
    return {
      host: 'wasm',
      threads: this.#settings.threads,
      concurrency: this.#settings.backend === 'webgpu' ? 'shared-gpu' : 'single',
      resident: this.#resident,
      reloading: this.#reloading,
      preload_speech: false,
      idle_model_timeout_seconds: 300,
      model_id: this.#settings.model,
      requested_backend: this.#settings.backend,
      maximum_tokens: this.#settings.maximumTokens,
      loading: Boolean(this.#worker && !this.#ready),
    };
  }

  #snapshot() {
    return {
      ready: this.#ready,
      config: 'Browser-local settings',
      runtime: this.#runtime(),
      components: this.#components,
      model: this.#settings.model,
      backend: this.#ready ? this.#settings.backend : '',
      vision: false,
    };
  }

  #event(id, type, payload = {}) {
    this.#callbacks?.onEvent?.({v: 1, id, type, seq: 0, payload});
  }

  #error(id, code, message) {
    this.#event(id, 'error', {code, message});
  }

  #save() {
    try { localStorage.setItem(SETTINGS_KEY, JSON.stringify(this.#settings)); } catch {}
  }
}

export const backend = new WasmBackend();
