import {loadAppConfig} from './app-config.mjs';

const config = await loadAppConfig();
const requests = [];
let callbacks;
let active;
let ready = false;
let speech;
const snapshot = () => ({
  ready,
  model: config.models.chat.model_id,
  backend: 'webgpu',
  components: {chat: {editor_prompts: config.prompts},
    tts: {directory: config.models.tts.model_id, settings: config.models.tts}},
  runtime: {
    model_id: config.models.chat.model_id, requested_backend: 'webgpu', threads: 1,
    memory: {host_bytes: 128 * 1024 ** 2, gpu_bytes: 64 * 1024 ** 2},
  },
});
const emit = (type, id, payload = {}) => callbacks.onEvent({type, id, payload});
export const backend = {
  requests,
  async status() { return snapshot(); },
  connect(value, options) {
    if (options?.loadCachedModel !== false) throw new Error('The extension must keep cached chat loading explicit');
    callbacks = value;
    queueMicrotask(() => {
      callbacks.onOpen();
      emit('session.ready', '', snapshot());
    });
  },
  send(type, id, payload = {}) {
    requests.push(structuredClone({type, id, payload}));
    if (type === 'runtime.load') {
      emit('runtime.loading', id);
      ready = true;
      queueMicrotask(() => emit('runtime.ready', id, snapshot()));
    } else if (type === 'cancel') {
      if (payload.scope === 'speech') speech = null;
      else active = null;
      emit('request.cancelled', id);
    } else if (type === 'tts.start') {
      speech = id;
      emit('tts.started', id, {stream_id: payload.stream_id});
      emit('tts.progress', id, {phase: 'loading', loadedBytes: 50, totalBytes: 100,
        downloadedBytes: 50, cachedBytes: 0, file: 'synthetic-voice.safetensors'});
      setTimeout(() => {
        if (speech !== id) return;
        emit('tts.progress', id, {phase: 'synthesizing', step: 2, steps: 6});
        callbacks.onPcm({streamId: payload.stream_id, sequence: 0, sampleRate: 8000,
          samples: new Float32Array(16000)});
        speech = null;
        emit('tts.completed', id, {sample_rate: 8000});
      }, 250);
    } else if (type === 'document.run') {
      active = id;
      const text = payload.mode === 'translate' ? 'Premier paragraphe traduit.\n\nDeuxième paragraphe traduit.'
        : 'The article explains a local-first approach and reports its key results.';
      emit('document.progress', id, {part: 1, parts: 2, round: 1, translating: payload.mode === 'translate'});
      if (payload.instruction === 'Wait for cancellation') return;
      setTimeout(() => {
        if (active !== id) return;
        emit('document.delta', id, {text: text.slice(0, 25)});
        emit('document.progress', id, {part: 2, parts: 2, round: 1, translating: payload.mode === 'translate'});
        setTimeout(() => {
          if (active !== id) return;
          emit('document.completed', id, {text, stats: {decode_tokens: 12, decode_ms: 500, token_count: 13}});
          active = null;
        }, 40);
      }, 40);
    } else if (type === 'editor.run') {
      active = id;
      const text = payload.instruction === 'Show Markdown'
        ? '**Bold result**\n\n- First item\n- Second item\n\n[Safe link](https://example.com)\n\n<script>alert(1)</script>' :
        payload.task === 'email_reply' ? 'Thanks for the update. Could we meet on Monday instead?' :
        payload.task === 'chat' ? 'A concise local answer.' : 'Hello Sam,\n\nThe report will be ready on Monday.';
      if (payload.instruction === 'Wait for cancellation') return;
      const stats = {decode_tokens: 12, decode_ms: 500, token_count: 13, first_token_ms: 100};
      setTimeout(() => {
        if (active !== id) return;
        emit('editor.delta', id, {text: text.slice(0, 15), stats});
        queueMicrotask(() => {
          if (active !== id) return;
          emit('editor.completed', id, {text, stats});
          active = null;
        });
      }, 20);
    } else throw new Error(`Unexpected fixture method: ${type}`);
  },
};
