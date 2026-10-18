import {isModelCached, webGpuSupport, listCachedFiles, deleteCachedFile, deleteCachedFiles, clearModelCache} from './model-cache.mjs';
import {createGpuWorker} from './runtime-client.mjs';
import {clearVoicePromptCache, describeVoicePrompt, loadVoicePrompt, storeVoicePrompt}
  from './voice-prompt-cache.mjs';

const SETTINGS_KEY = 'kidi-wasm-host-v1';
const VOICE_REFERENCES = [
  'Hello, I am Kidi, your local assistant.',
  'A quiet breeze moves through the bright garden.',
  'Clear morning light falls across the open window.',
];
const DEFAULTS = {
  model: 'google/gemma-4-E2B-it-qat-mobile-transformers',
  backend: 'webgpu',
  threads: 4,
  maximumTokens: 512,
  systemPrompt: 'You are a helpful local AI assistant. Respond directly, briefly, and in the language or language mix used by the user.',
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
      directory: 'openai/whisper-small',
      settings: {
        language: 'auto',
        max_tokens: 128,
        vad: {threshold: 0.012, minimum_speech_ms: 250, silence_ms: 800},
      },
    },
    tts: {
      enabled: true,
      accelerator: 'auto',
      directory: './models/omnivoice/manifest.json',
      settings: {
        language: 'auto',
        steps: 10,
        realtime_steps: 6,
        reference_steps: 6,
        pace: 1.15,
        gpu_precision: 'w8a8',
        reference_language: 'English',
        reference_text: VOICE_REFERENCES[0],
        reference_sentences: VOICE_REFERENCES,
        voice_seed: 0,
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
  #ttsRequest;
  #ttsReady;
  #ttsWorker;
  #ttsLoadingId;
  #ttsLoadReject;
  #ttsMode;
  #ttsUseGpu = false;
  #ttsModelInfo;
  #ttsVoiceKey;
  #ttsVoiceSettingsKey;
  #ttsControls = new Map();
  #worker;
  #resident = false;
  #reloading = false;

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

  #voiceDefinition(options, modelInfo = {}) {
    const voiceSeed = Number.isSafeInteger(options.voice_seed) ? options.voice_seed : 0;
    const configured = Array.isArray(options.reference_sentences) ?
      options.reference_sentences.filter(value => typeof value === 'string' && value.trim()) : [];
    const references = configured.length ? configured :
      typeof options.reference_text === 'string' && options.reference_text.trim() ?
        [options.reference_text.trim()] : VOICE_REFERENCES;
    const index = ((voiceSeed % references.length) + references.length) % references.length;
    const text = references[index];
    const language = options.reference_language || 'English';
    const precision = options.gpu_precision || 'f32';
    const referenceSteps = Number(this.#components.tts.settings.reference_steps) || 6;
    const settings = {
      model: this.#components.tts.directory,
      precision,
      codebooks: 8,
      frameRate: 25,
      vocabulary: 1024,
      maximumSeconds: 20,
      voice: options.voice || {},
      reference: {
        language,
        text,
        seed: voiceSeed,
        steps: referenceSteps,
        guidanceScale: Number(options.guidance_scale ?? 2),
        timeShift: Number(options.time_shift ?? 0.1),
        layerPenalty: Number(options.layer_penalty ?? 5),
        positionTemperature: Number(options.position_temperature ?? 5),
      },
    };
    const identity = {
      ...settings,
      model: modelInfo.id || settings.model,
      execution: modelInfo.backend || this.#settings.backend,
      precision: modelInfo.matrix_precision || (modelInfo.backend === 'webgpu' ? precision : 'f32'),
    };
    return {identity, settingsKey: JSON.stringify(settings), text, language, voiceSeed, precision, referenceSteps};
  }

  #controlTts(worker, data, expected) {
    const requestId = `voice:${crypto.randomUUID()}`;
    data.requestId = requestId;
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.#ttsControls.delete(requestId);
        reject(new Error(`TTS voice preparation timed out: ${data.type}`));
      }, 300000);
      this.#ttsControls.set(requestId, {expected, resolve, reject, timer});
      try {
        const transfer = data.tokens instanceof ArrayBuffer ? [data.tokens] : [];
        worker.postMessage(data, transfer);
      } catch (error) {
        clearTimeout(timer);
        this.#ttsControls.delete(requestId);
        reject(error);
      }
    });
  }

  #handleTtsControl(data, id) {
    if (data.type === 'generating' && this.#ttsControls.size) {
      this.#event(id, 'tts.progress', {phase: 'calibrating-voice', step: data.step, steps: data.steps});
      return true;
    }
    const control = this.#ttsControls.get(data.requestId);
    if (!control) return false;
    if (data.type !== 'error' && data.type !== control.expected) return true;
    clearTimeout(control.timer);
    this.#ttsControls.delete(data.requestId);
    if (data.type === 'error') control.reject(new Error(data.error || 'TTS voice preparation failed'));
    else control.resolve(data);
    return true;
  }

  async #prepareTtsVoice(worker, id, options, modelInfo) {
    const definition = this.#voiceDefinition(options, modelInfo);
    const descriptor = await describeVoicePrompt(definition.identity);
    if (worker !== this.#ttsWorker) throw new DOMException('TTS worker changed', 'AbortError');
    let cached = null;
    try {
      cached = await loadVoicePrompt(descriptor);
    } catch (error) {
      this.#event(id, 'tts.progress', {phase: 'voice-cache', message: `Ignoring cached voice: ${error.message}`});
    }
    if (cached) {
      const buffer = cached.tokens.slice().buffer;
      await this.#controlTts(worker, {type: 'voice-prompt', text: cached.text, tokens: buffer},
        'voice-prompt-ready');
    } else {
      this.#event(id, 'tts.progress', {phase: 'calibrating-voice'});
      const result = await this.#controlTts(worker, {
        type: 'synthesize',
        streamSegment: true,
        text: definition.text,
        options: {
          ...options,
          language: definition.language,
          pace: 1,
          steps: definition.referenceSteps,
          seed: definition.voiceSeed,
          use_voice_prompt: false,
          cache_voice_prompt: true,
          return_tokens: true,
        },
      }, 'audio');
      if (result.voice_prompt_used === true || result.voice_prompt_cached !== true ||
          !Array.isArray(result.token_ids) || !result.token_ids.length)
        throw new Error('OmniVoice did not return a reusable calibration prompt');
      const prompt = {text: definition.text, tokens: Int32Array.from(result.token_ids)};
      const buffer = prompt.tokens.slice().buffer;
      await this.#controlTts(worker, {type: 'voice-prompt', text: prompt.text, tokens: buffer},
        'voice-prompt-ready');
      try {
        await storeVoicePrompt(descriptor, prompt);
      } catch (error) {
        this.#event(id, 'tts.progress', {phase: 'voice-cache', message: `Voice cache unavailable: ${error.message}`});
      }
    }
    if (worker !== this.#ttsWorker) throw new DOMException('TTS worker changed', 'AbortError');
    this.#ttsVoiceKey = descriptor.id;
    this.#ttsVoiceSettingsKey = definition.settingsKey;
    this.#event(id, 'tts.progress', {phase: 'voice-ready', cached: Boolean(cached)});
  }

  #ensureTts(id, options) {
    const definition = this.#voiceDefinition(options);
    const precision = definition.precision;
    if (!['f32', 'mixed-f16', 'w8a8'].includes(precision))
      return Promise.reject(new Error('Unsupported browser speech precision'));
    const mode = JSON.stringify({backend: this.#settings.backend, precision});
    if (this.#ttsWorker && this.#ttsMode !== mode) this.#releaseTts();
    if (this.#ttsWorker && this.#ttsModelInfo) {
      if (this.#ttsReady && this.#ttsVoiceSettingsKey === definition.settingsKey) return this.#ttsReady;
      this.#ttsLoadingId = id;
      this.#ttsVoiceSettingsKey = definition.settingsKey;
      this.#ttsReady = new Promise((resolve, reject) => {
        this.#ttsLoadReject = reject;
        this.#prepareTtsVoice(this.#ttsWorker, id, options, this.#ttsModelInfo).then(resolve, reject);
      }).finally(() => {
        this.#ttsLoadingId = null;
        this.#ttsLoadReject = null;
      });
      return this.#ttsReady;
    }
    if (this.#ttsReady) return this.#ttsReady;
    this.#ttsMode = mode;
    this.#ttsVoiceSettingsKey = definition.settingsKey;
    this.#ttsLoadingId = id;
    this.#ttsReady = new Promise((resolve, reject) => {
      this.#ttsLoadReject = reject;
      const availableThreads = Number(globalThis.navigator?.hardwareConcurrency);
      const preferredGpu = this.#settings.backend === 'webgpu' && !webGpuSupport();
      const start = useGpu => {
        const worker = useGpu ? createGpuWorker('tts') :
          new Worker(new URL('./tts-worker.mjs', import.meta.url), {type: 'module'});
        this.#ttsWorker = worker;
        this.#ttsUseGpu = useGpu;
        let loaded = false;
        const failLoad = message => {
          if (worker !== this.#ttsWorker) return;
          worker.terminate();
          this.#ttsWorker = null;
          this.#ttsModelInfo = null;
          if (useGpu) {
            this.#event(id, 'tts.progress', {
              phase: 'fallback',
              message: `WebGPU speech unavailable; using CPU (${message})`,
            });
            start(false);
          } else {
            this.#ttsReady = null;
            this.#ttsLoadingId = null;
            this.#ttsLoadReject = null;
            reject(new Error(message));
          }
        };
        worker.onmessage = ({data}) => {
          if (worker !== this.#ttsWorker) return;
          if (this.#handleTtsControl(data, id)) return;
          if (data.type === 'ready') {
            loaded = true;
            this.#ttsModelInfo = data;
            this.#components.tts.threads = data.threads;
            this.#components.tts.accelerator = data.backend ||
              (data.threads > 1 ? 'wasm-threads' : 'wasm-cpu');
            this.#components.tts.gpu_precision = data.matrix_precision;
            this.#prepareTtsVoice(worker, id, options, data).then(() => {
              if (worker !== this.#ttsWorker) return;
              this.#ttsLoadingId = null;
              this.#ttsLoadReject = null;
              resolve();
            }, error => failLoad(error.message));
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
          } else if (data.type === 'residency' && data.state === 'evicted') {
            this.#ttsReady = null;
            this.#ttsVoiceKey = null;
          } else if (data.type === 'error') {
            if (!loaded) {
              failLoad(data.error || 'TTS worker failed');
            } else {
              const request = this.#ttsRequest;
              if (request) this.#failSpeech(request, data.error || 'Speech synthesis failed');
            }
          }
        };
        worker.onerror = event => {
          if (worker !== this.#ttsWorker) return;
          if (!loaded) failLoad(event.message || 'TTS worker failed');
          else {
            const request = this.#ttsRequest;
            if (request) this.#failSpeech(request, event.message || 'TTS worker failed', 'tts_worker_failed');
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

  #synthesize(id, streamId, text, options) {
    if (this.#ttsRequest) {
      this.#error(id, 'busy', 'Another speech synthesis request is active');
      return;
    }
    const defaults = this.#components.tts.settings;
    const effectiveOptions = {...defaults, ...options, voice: {...(options?.voice || defaults.voice)}};
    const request = {id, streamId, text, options: effectiveOptions, started: performance.now()};
    this.#ttsRequest = request;
    this.#ensureTts(id, effectiveOptions).then(() => {
      if (this.#ttsRequest !== request) return;
      this.#event(id, 'tts.progress', {phase: 'synthesizing'});
      this.#ttsWorker.postMessage({
        type: 'synthesize', requestId: id, streamSegment: true, text,
        options: {...effectiveOptions, use_voice_prompt: true, cache_voice_prompt: false},
      });
    }).catch(error => this.#failSpeech(request, error.message, 'tts_load_failed'));
  }

  #releaseTts() {
    this.#ttsLoadReject?.(new DOMException('Speech request stopped', 'AbortError'));
    this.#ttsLoadReject = null;
    for (const control of this.#ttsControls.values()) {
      clearTimeout(control.timer);
      control.reject(new DOMException('Speech request stopped', 'AbortError'));
    }
    this.#ttsControls.clear();
    this.#ttsWorker?.terminate();
    this.#ttsWorker = null;
    this.#ttsReady = null;
    this.#ttsLoadingId = null;
    this.#ttsModelInfo = null;
    this.#ttsUseGpu = false;
    this.#ttsVoiceKey = null;
    this.#ttsVoiceSettingsKey = null;
    this.#components.tts.accelerator = 'auto';
    delete this.#components.tts.threads;
    delete this.#components.tts.gpu_precision;
  }

  #cancelSpeech(id) {
    if (this.#ttsRequest?.id !== id && this.#ttsLoadingId !== id) return false;
    this.#ttsRequest = null;
    this.#releaseTts();
    return true;
  }

  #failSpeech(request, message, code = 'tts_failed') {
    if (!request || this.#ttsRequest !== request) return;
    this.#ttsRequest = null;
    this.#releaseTts();
    this.#error(request.id, code, message);
  }

  async #ttsAudio(data) {
    const request = this.#ttsRequest;
    if (!request || request.id !== data.requestId) return;
    try {
      if (!(data.audio instanceof ArrayBuffer) || !data.audio.byteLength || data.audio.byteLength % 4 ||
          !Number.isInteger(data.sample_rate) || data.sample_rate <= 0)
        throw new Error('Speech worker returned invalid PCM');
      const samples = new Float32Array(data.audio);
      const sampleCount = samples.length;
      if (data.voice_prompt_used !== true)
        throw new Error('OmniVoice lost its prepared voice prompt');
      if (this.#callbacks?.onPcm) {
        await this.#callbacks.onPcm({streamId: request.streamId, sequence: 0,
          sampleRate: data.sample_rate, samples});
      } else {
        if (!this.#callbacks?.onAudio) throw new Error('No speech audio consumer is connected');
        for (let offset = 0; offset < samples.length; offset += 4096) {
          if (this.#ttsRequest !== request) return;
          const length = Math.min(4096, samples.length - offset);
          const frame = new ArrayBuffer(24 + length * 2);
          new Uint8Array(frame).set([0x4b, 0x44, 0x41, 0x31, 1, 2], 0);
          const header = new DataView(frame);
          header.setUint32(8, request.streamId, true);
          header.setUint32(12, offset / 4096, true);
          header.setUint32(16, data.sample_rate, true);
          header.setUint32(20, length * 2, true);
          const pcm = new Int16Array(frame, 24, length);
          for (let index = 0; index < length; index++)
            pcm[index] = Math.max(-32768, Math.min(32767, Math.round(samples[offset + index] * 32768)));
          await this.#callbacks.onAudio(frame);
        }
      }
      if (this.#ttsRequest !== request) return;
      this.#ttsRequest = null;
      if (!this.#ttsUseGpu) this.#releaseTts();
      this.#event(request.id, 'tts.completed', {
        stream_id: request.streamId,
        sample_rate: data.sample_rate,
        samples: sampleCount,
        chunks: 1,
        voice_prompt_reused: true,
        synthesis_ms: performance.now() - request.started,
        stats: {
          generation_ms: data.generation_ms,
          decode_ms: data.decode_ms,
          preparation_ms: data.preparation_ms,
          audio_tokens: data.audio_tokens,
        },
      });
    } catch (error) {
      this.#failSpeech(request, error.message, 'tts_audio_failed');
    }
  }

  get bufferedAmount() {
    return 0;
  }

  async status() {
    return this.#snapshot();
  }

  listCachedFiles() { return listCachedFiles(); }
  async deleteCachedFile(id) {
    const result = await deleteCachedFile(id);
    await clearVoicePromptCache();
    return result;
  }
  async deleteCachedFiles(ids) {
    const result = await deleteCachedFiles(ids);
    await clearVoicePromptCache();
    return result;
  }
  async clearModelCache() {
    const [models] = await Promise.all([clearModelCache(), clearVoicePromptCache()]);
    return models;
  }

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
      const languages = String(payload.languages || '');
      if (languages && !/^[a-z]{2,3}(?:,[a-z]{2,3})*$/u.test(languages))
        return this.#error(id, 'invalid_languages', 'languages must contain comma-separated language codes');
      const languageNames = String(payload.language_names || languages);
      if (languageNames.length > 256 || /[\u0000-\u001f]/u.test(languageNames))
        return this.#error(id, 'invalid_languages', 'language_names contains invalid characters');
      if (this.#settings.systemPrompt) {
        const instruction = !languages ? '' : languages.includes(',')
          ? `\nThe user may code-switch among ${languageNames}. Respond in the same language or language mix as the latest user message.`
          : `\nThe user speaks ${languageNames}. Respond in ${languageNames} only.`;
        messages.unshift({role: 'system', content: this.#settings.systemPrompt + instruction});
      }
      this.#current = {
        id,
        text: '',
      };
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
      const language = payload.language ?? this.#components.asr.settings.language ?? 'en';
      if (typeof language !== 'string' || !/^(?:auto|[a-z]{2,3}(?:,[a-z]{2,3})*)$/u.test(language))
        return this.#error(id, 'invalid_asr', 'ASR language must be auto or comma-separated language codes');
      this.#asr = {
        id,
        streamId: payload.stream_id,
        samples: [],
        sequence: 0,
        transcribing: false,
        finalRequested: false,
        language,
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
        const request = this.#ttsRequest?.id === id ? this.#ttsRequest : null;
        if (request) {
          this.#cancelSpeech(id);
          this.#event(id, 'request.cancelled', {message: payload.message || 'Speech playback stopped'});
        }
        return;
      }
      if (this.#current?.id === id) this.#worker?.postMessage({type: 'cancel'});
      if (this.#asr?.id === id) {
        this.#speechWorker?.postMessage({type: 'cancel'});
        this.#asr = null;
        this.#event(id, 'request.cancelled');
      }
      if (this.#ttsLoadingId === id || this.#ttsRequest?.id === id) {
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
    if (selectedBackend !== this.#settings.backend) this.#releaseTts();
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
          } else if (data.type === 'partial' && this.#asr &&
                     String(data.requestId).includes(this.#asr.id)) {
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
    if (!request || request.transcribing) return;
    if (!request.samples.length) {
      if (final) {
        this.#event(request.id, 'asr.completed', {
          text: '', language: request.language, stats: {feature_ms: 0, encode_ms: 0, decode_ms: 0},
        });
        this.#asr = null;
      }
      return;
    }
    request.transcribing = true;
    if (!final) request.lastPartialSamples = request.samples.length;
    const audio = Float32Array.from(request.samples);
    const requestId = final ? `final:${request.id}` : `partial:${request.id}:${request.partialSequence++}`;
    this.#speechWorker.postMessage({
      type: 'transcribe',
      requestId,
      audio: audio.buffer,
      language: request.language,
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
        }
        if (item.completed) {
          const completed = item.completed;
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
      speech_backend: this.#ttsWorker ? (this.#ttsUseGpu ? 'webgpu' : 'cpu') :
        (this.#settings.backend === 'webgpu' ? 'webgpu' : 'cpu'),
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
