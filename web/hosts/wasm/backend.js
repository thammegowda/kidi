import {isModelCached, webGpuSupport, listCachedFiles, deleteCachedFile, deleteCachedFiles, clearModelCache} from './model-cache.mjs';

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
  #ttsPending = new Map();
  #ttsReady;
  #ttsWorker;
  #worker;
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

  #ensureTts() {
    if (this.#ttsReady) return this.#ttsReady;
    this.#ttsReady = new Promise((resolve, reject) => {
      const worker = new Worker(new URL('./tts-worker.mjs', import.meta.url), {type: 'module'});
      this.#ttsWorker = worker;
      let loaded = false;
      worker.onmessage = ({data}) => {
        if (data.type === 'ready') {
          loaded = true;
          resolve();
        } else if (data.type === 'audio') {
          this.#ttsAudio(data);
        } else if (data.type === 'error') {
          if (!loaded) {
            worker.terminate();
            this.#ttsReady = null;
            reject(new Error(data.error || 'TTS worker failed'));
          } else {
            const pending = this.#ttsPending.get(data.requestId);
            if (pending) {
              this.#ttsPending.delete(data.requestId);
              this.#event(pending.id, pending.voice ? 'voice.audio.error' : 'error', {
                code: 'tts_failed',
                message: data.error || 'Speech synthesis failed',
              });
            }
          }
        }
      };
      worker.onerror = event => {
        this.#ttsReady = null;
        reject(new Error(event.message || 'TTS worker failed'));
      };
      worker.postMessage({
        type: 'load',
        source: this.#components.tts.directory,
        threads: 1,
        cacheOnly: false,
      });
    });
    return this.#ttsReady;
  }

  #synthesize(id, streamId, text, options, voice) {
    this.#ensureTts().then(() => {
      if (voice) this.#event(id, 'voice.audio.first', {stream_id: streamId, elapsed_ms: 0});
      this.#ttsPending.set(id, {id, streamId, voice});
      this.#ttsWorker.postMessage({
        type: 'synthesize',
        requestId: id,
        text,
        options: {
          ...this.#components.tts.settings,
          ...(options || {}),
          use_voice_prompt: false,
          cache_voice_prompt: false,
        },
      });
    }).catch(error => {
      this.#event(id, voice ? 'voice.audio.error' : 'error', {
        code: 'tts_load_failed',
        message: error.message,
      });
    });
  }

  #ttsAudio(data) {
    const pending = this.#ttsPending.get(data.requestId);
    if (!pending || !(data.audio instanceof ArrayBuffer)) return;
    this.#ttsPending.delete(data.requestId);
    const samples = new Float32Array(data.audio);
    let sequence = 0;
    for (let offset = 0; offset < samples.length; offset += 4096) {
      const count = Math.min(4096, samples.length - offset);
      const frame = new ArrayBuffer(24 + count * 2);
      const bytes = new Uint8Array(frame);
      bytes.set([0x4b, 0x44, 0x41, 0x31, 1, 2], 0);
      const header = new DataView(frame);
      header.setUint32(8, pending.streamId, true);
      header.setUint32(12, sequence++, true);
      header.setUint32(16, data.sample_rate, true);
      header.setUint32(20, count * 2, true);
      const pcm = new Int16Array(frame, 24, count);
      for (let index = 0; index < count; index++)
        pcm[index] = Math.max(-32768, Math.min(32767, Math.round(samples[offset + index] * 32768)));
      this.#callbacks?.onAudio?.(frame);
    }
    this.#event(pending.id, pending.voice ? 'voice.audio.completed' : 'tts.completed', {
      stream_id: pending.streamId,
      sample_rate: data.sample_rate,
      samples: samples.length,
      chunks: data.chunks || 1,
      voice_prompt_reused: false,
    });
    this.#ttsWorker?.terminate();
    this.#ttsWorker = null;
    this.#ttsReady = null;
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
      if (!Number.isInteger(payload.stream_id) || typeof payload.text !== 'string' || !payload.text)
        return this.#error(id, 'invalid_tts', 'TTS requires text and an unsigned stream_id');
      this.#event(id, 'tts.started', {stream_id: payload.stream_id, format: 'pcm_s16le'});
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
          format: 'pcm_s16le',
        });
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
      if (this.#current?.id === id) this.#worker?.postMessage({type: 'cancel'});
      if (this.#asr?.id === id) {
        this.#asr = null;
        this.#event(id, 'request.cancelled');
      }
      if (this.#ttsPending.has(id)) {
        this.#ttsPending.delete(id);
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
    if (this.#worker) this.#worker.terminate();
    this.#ready = false;
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
    const worker = new Worker(new URL('./inference-worker.mjs', import.meta.url), {type: 'module'});
    this.#worker = worker;
    this.#event(id, 'runtime.loading', {runtime: this.#runtime()});
    worker.onmessage = ({data}) => this.#workerEvent(id, worker, data);
    worker.onerror = event => this.#error(id, 'worker_error', event.message || 'Browser model worker failed');
    worker.postMessage({
      type: 'load',
      manifest: this.#settings.model,
      threads: selectedBackend === 'webgpu' ? 1 : threads,
      backend: selectedBackend,
      cacheOnly: requested.cacheOnly === true,
    });
  }

  #ensureSpeech() {
    if (this.#speechReady) return this.#speechReady;
    this.#speechReady = new Promise((resolve, reject) => {
      const worker = new Worker(new URL('./asr-worker.mjs', import.meta.url), {type: 'module'});
      this.#speechWorker = worker;
      worker.onmessage = ({data}) => {
        if (data.type === 'ready') {
          resolve();
        } else if (data.type === 'result') {
          this.#speechResult(data);
        } else if (data.type === 'error') {
          if (!this.#asr) reject(new Error(data.error || 'Speech worker failed'));
          else {
            const id = this.#asr.id;
            this.#asr = null;
            this.#error(id, 'asr_failed', data.error || 'Speech recognition failed');
          }
        }
      };
      worker.onerror = event => reject(new Error(event.message || 'Speech worker failed'));
      worker.postMessage({
        type: 'load',
        source: this.#components.asr.directory,
        threads: Math.max(1, this.#settings.threads),
      });
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
    if (data.type === 'progress' || data.type === 'initializing') {
      this.#event(loadId, 'runtime.progress', {
        stage: data.stage || data.file || (data.type === 'initializing' ? 'Initializing model' : 'Loading model'),
        ...data,
      });
      return;
    }

    if (data.type === 'ready') {
      this.#ready = true;
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
        }
        if (item.completed) {
          const completed = item.completed;
          const current = this.#current;
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
          if (current.speak)
            this.#synthesize(current.id, current.streamId, completed.text || current.text,
              current.ttsOptions, true);
        }
      }
      return;
    }
    if (data.type === 'cancelled' && this.#current) {
      this.#event(this.#current.id, 'request.cancelled');
      this.#current = null;
      return;
    }
    if (data.type === 'error') {
      const request = this.#current?.id || loadId;
      this.#error(request, 'wasm_error', data.error || 'Browser runtime failed');
      this.#current = null;
      if (data.fatal) this.#ready = false;
    }
  }

  #runtime() {
    return {
      host: 'wasm',
      threads: this.#settings.threads,
      concurrency: 'single',
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
