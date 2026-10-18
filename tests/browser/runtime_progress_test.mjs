import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';

async function backendSource() {
  let source = await readFile(new URL('../../web/hosts/wasm/backend.js', import.meta.url), 'utf8');
  source = source.replace("import {loadAppConfig} from './app-config.mjs';",
    `const loadAppConfig = async () => ({version:1,model_hub:{remote:'https://huggingface.co',local:'cache'},
      runtime:{threads:4},models:{
        chat:{enabled:true,model_id:'google/test',accelerator:'auto',precision:'checkpoint',
          kv_cache_precision:'auto',system_prompt:'Test',
          translation_prompt:'Translate into {target_language}; return only the translation.',
          generation:{max_tokens:128,context_tokens:1024,prefix_cache_mb:0}},
        asr:{enabled:true,model_id:'openai/whisper-small',language:'auto',max_tokens:128,
          vad:{threshold:.012,minimum_speech_ms:250,silence_ms:800}},
        tts:{enabled:true,model_id:'test/tts',language:'auto',steps:10,realtime_steps:6,pace:1.15,
          reference_language:'English',reference_text:'Hello.',voice:{}}
      }});`);
  source = source.replace(
    "import {clearVoicePromptCache, describeVoicePrompt, loadVoicePrompt, storeVoicePrompt}\n  from './voice-prompt-cache.mjs';",
    `const clearVoicePromptCache = async () => {};
     const describeVoicePrompt = async config => ({id: 'a'.repeat(64), metadata: '{}', config});
     const loadVoicePrompt = async () => ({text: 'Hello, I am Kidi.', tokens: new Int32Array(600)});
     const storeVoicePrompt = async () => {};`);
  return source;
}

async function waitFor(predicate) {
  for (let attempt = 0; attempt < 100; ++attempt) {
    if (predicate()) return;
    await new Promise(resolve => setImmediate(resolve));
  }
  throw new Error('Timed out waiting for backend state');
}

test('automatic cache restore announces its request before progress and readiness', async () => {
  const source = await backendSource();
  const workers = [];
  globalThis.localStorage = {getItem: () => null, setItem() {}};
  globalThis.Worker = class {
    constructor() { workers.push(this); }
    postMessage() {}
    terminate() {}
  };
  try {
    const isolated = source
      .replace("import {createGpuWorker} from './runtime-client.mjs';",
        'const createGpuWorker = kind => new Worker(kind + "-gpu");')
      .replace("import {configureModelHub, isModelCached, webGpuSupport, listCachedFiles, deleteCachedFile, deleteCachedFiles,\n  clearModelCache} from './model-cache.mjs';",
        'const configureModelHub = () => {}; const isModelCached = async () => true; const webGpuSupport = () => null;')
      .replace("new URL('./inference-worker.mjs', import.meta.url)", "'inference-worker.mjs'");
    const {backend} = await import(`data:text/javascript,${encodeURIComponent(isolated)}`);
    const events = [];
    backend.connect({onEvent: event => events.push(event)});
    await new Promise(resolve => setImmediate(resolve));
    const started = events.find(event => event.type === 'runtime.loading');
    assert.ok(started?.id);
    assert.equal(started.payload.runtime.loading, true);
    assert.equal(workers.length, 1);
    const metrics = {type: 'progress', totalBytes: 100, loadedBytes: 75, cachedBytes: 50, downloadedBytes: 25};
    workers[0].onmessage({data: metrics});
    const progress = events.at(-1);
    assert.equal(progress.type, 'runtime.progress');
    assert.equal(progress.id, started.id);
    assert.equal(progress.payload.loadedBytes, 75);
    assert.equal(progress.payload.cachedBytes, 50);
    workers[0].onmessage({data: {type: 'initializing'}});
    assert.equal(events.at(-1).payload.type, 'initializing');
    workers[0].onmessage({data: {type: 'ready', backend: 'webgpu'}});
    const ready = events.find(event => event.type === 'runtime.ready');
    assert.equal(ready.id, started.id);
    assert.equal(ready.payload.runtime.loading, false);
  } finally {
    delete globalThis.localStorage;
    delete globalThis.Worker;
  }
});

test('reader translation uses its configured prompt and dedicated events', async () => {
  const source = await backendSource();
  const workers = [];
  globalThis.localStorage = {getItem: () => null, setItem() {}};
  globalThis.Worker = class {
    messages = [];
    constructor() { workers.push(this); }
    postMessage(message) { this.messages.push(message); }
    terminate() {}
  };
  try {
    const isolated = source
      .replace("import {createGpuWorker} from './runtime-client.mjs';",
        'const createGpuWorker = kind => new Worker(kind + "-gpu");')
      .replace("import {configureModelHub, isModelCached, webGpuSupport, listCachedFiles, deleteCachedFile, deleteCachedFiles,\n  clearModelCache} from './model-cache.mjs';",
        'const configureModelHub = () => {}; const isModelCached = async () => false; const webGpuSupport = () => null;')
      .replace("new URL('./inference-worker.mjs', import.meta.url)", "'inference-worker.mjs'");
    const {backend} = await import(`data:text/javascript,${encodeURIComponent(isolated)}#translation`);
    const events = [];
    backend.connect({onEvent: event => events.push(event)});
    backend.send('runtime.load', 'load', {model: 'google/test', backend: 'cpu', threads: 2});
    workers[0].onmessage({data: {type: 'ready', backend: 'cpu'}});
    backend.send('reader.translate', 'translate', {
      text: 'The old house.', previous: 'A storm ended.', next: 'The garden bloomed.',
      target_language: 'kn', target_language_name: 'Kannada',
    });
    const generation = workers[0].messages.at(-1);
    assert.equal(generation.type, 'generate');
    assert.equal(generation.messages[0].content,
      'Translate into Kannada; return only the translation.');
    assert.match(generation.messages[1].content,
      /<preceding_context>\nA storm ended\.\n<\/preceding_context>/);
    assert.match(generation.messages[1].content,
      /<text_to_translate>\nThe old house\.\n<\/text_to_translate>/);
    workers[0].onmessage({data: {type: 'step', events: [
      {text: 'ಹಳೆಯ ಮನೆ.'},
      {completed: {text: 'ಹಳೆಯ ಮನೆ.', prompt_tokens: 20, decode_tokens: 4}},
    ]}});
    assert.deepEqual(events.slice(-3).map(event => event.type), [
      'reader.translation.started',
      'reader.translation.delta',
      'reader.translation.completed',
    ]);
    assert.equal(events.at(-1).payload.text, 'ಹಳೆಯ ಮನೆ.');
  } finally {
    delete globalThis.localStorage;
    delete globalThis.Worker;
  }
});

test('isolated browsers use available CPU threads for TTS', async () => {
  const source = await backendSource();
  const workers = [];
  const previousIsolation = globalThis.crossOriginIsolated;
  globalThis.crossOriginIsolated = true;
  globalThis.localStorage = {getItem: () => JSON.stringify({backend: 'cpu', threads: 4}), setItem() {}};
  globalThis.Worker = class {
    constructor(url) { this.url = url; workers.push(this); }
    postMessage(message) { this.message = message; }
    terminate() {}
  };
  try {
    const isolated = source
      .replace("import {createGpuWorker} from './runtime-client.mjs';",
        'const createGpuWorker = kind => new Worker(kind + "-gpu");')
      .replace("import {configureModelHub, isModelCached, webGpuSupport, listCachedFiles, deleteCachedFile, deleteCachedFiles,\n  clearModelCache} from './model-cache.mjs';",
        'const configureModelHub = () => {}; const isModelCached = async () => false; const webGpuSupport = () => null;')
      .replace("new URL('./tts-worker.mjs', import.meta.url)", "'tts-worker.mjs'");
    const {backend} = await import(`data:text/javascript,${encodeURIComponent(isolated)}#tts`);
    const events = [];
    backend.connect({onEvent: event => events.push(event)});
    backend.send('tts.start', 'speech', {stream_id: 7, text: 'Hello'});
    assert.equal(workers.length, 1);
    const expectedThreads = Math.min(8, globalThis.navigator?.hardwareConcurrency || 4);
    assert.equal(workers[0].message.threads, expectedThreads);
    workers[0].onmessage({data: {
      type: 'progress', totalBytes: 100, loadedBytes: 75, cachedBytes: 75, downloadedBytes: 0,
    }});
    const progress = events.at(-1);
    assert.equal(progress.type, 'tts.progress');
    assert.equal(progress.payload.phase, 'loading');
    workers[0].onmessage({data: {type: 'ready', threads: expectedThreads}});
    await waitFor(() => workers[0].message.type === 'voice-prompt');
    workers[0].onmessage({data: {type: 'voice-prompt-ready', requestId: workers[0].message.requestId}});
    await waitFor(() => workers[0].message.type === 'synthesize');
    assert.equal(events.at(-1).type, 'tts.progress');
    assert.equal(events.at(-1).payload.phase, 'synthesizing');
    assert.equal(workers[0].message.type, 'synthesize');
    workers[0].onmessage({data: {type: 'generating', step: 3, steps: 8}});
    assert.equal(events.at(-1).payload.step, 3);
    assert.equal(events.at(-1).payload.steps, 8);
    assert.equal((await backend.status()).components.tts.accelerator, 'wasm-threads');
  } finally {
    if (previousIsolation === undefined) delete globalThis.crossOriginIsolated;
    else globalThis.crossOriginIsolated = previousIsolation;
    delete globalThis.localStorage;
    delete globalThis.Worker;
  }
});

test('WebGPU TTS initialization failure reports and uses threaded CPU fallback', async () => {
  const source = await backendSource();
  const workers = [];
  const previousIsolation = globalThis.crossOriginIsolated;
  globalThis.crossOriginIsolated = true;
  globalThis.localStorage = {getItem: () => JSON.stringify({backend: 'webgpu', threads: 4}), setItem() {}};
  globalThis.Worker = class {
    constructor(url) { this.url = url; workers.push(this); }
    postMessage(message) { this.message = message; }
    terminate() { this.terminated = true; }
  };
  try {
    const isolated = source
      .replace("import {createGpuWorker} from './runtime-client.mjs';",
        'const createGpuWorker = kind => new Worker(kind + "-gpu");')
      .replace("import {configureModelHub, isModelCached, webGpuSupport, listCachedFiles, deleteCachedFile, deleteCachedFiles,\n  clearModelCache} from './model-cache.mjs';",
        'const configureModelHub = () => {}; const isModelCached = async () => false; const webGpuSupport = () => null;')
      .replace("new URL('./tts-worker.mjs', import.meta.url)", "'tts-worker.mjs'");
    const {backend} = await import(`data:text/javascript,${encodeURIComponent(isolated)}#tts-fallback`);
    const events = [];
    backend.connect({onEvent: event => events.push(event)});
    backend.send('tts.start', 'speech', {stream_id: 8, text: 'Hello'});
    assert.equal(workers[0].message.backend, 'webgpu');
    workers[0].onmessage({data: {type: 'error', error: 'JSPI unavailable'}});
    assert.equal(workers[0].terminated, true);
    assert.equal(workers.length, 2);
    assert.equal(workers[1].message.backend, 'cpu');
    assert.equal(workers[1].message.threads, Math.min(8, globalThis.navigator?.hardwareConcurrency || 4));
    const fallback = events.find(event => event.type === 'tts.progress' && event.payload.phase === 'fallback');
    assert.match(fallback.payload.message, /JSPI unavailable/);
    workers[1].onmessage({data: {type: 'ready', threads: workers[1].message.threads, backend: 'wasm-threads'}});
    await waitFor(() => workers[1].message.type === 'voice-prompt');
    workers[1].onmessage({data: {type: 'voice-prompt-ready', requestId: workers[1].message.requestId}});
    await waitFor(() => workers[1].message.type === 'synthesize');
    assert.equal(workers[1].message.type, 'synthesize');
  } finally {
    if (previousIsolation === undefined) delete globalThis.crossOriginIsolated;
    else globalThis.crossOriginIsolated = previousIsolation;
    delete globalThis.localStorage;
    delete globalThis.Worker;
  }
});
