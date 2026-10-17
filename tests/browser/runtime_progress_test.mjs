import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';

test('automatic cache restore announces its request before progress and readiness', async () => {
  const source = await readFile(new URL('../../web/hosts/wasm/backend.js', import.meta.url), 'utf8');
  const workers = [];
  globalThis.localStorage = {getItem: () => null, setItem() {}};
  globalThis.Worker = class {
    constructor() { workers.push(this); }
    postMessage() {}
    terminate() {}
  };
  try {
    const isolated = source
      .replace("import {isModelCached, webGpuSupport, listCachedFiles, deleteCachedFile, deleteCachedFiles, clearModelCache} from './model-cache.mjs';",
        'const isModelCached = async () => true; const webGpuSupport = () => null;')
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
