import assert from 'node:assert/strict';
import test from 'node:test';
import {readFile} from 'node:fs/promises';

test('CPU and WebGPU workers both route typed editor requests to C++', async () => {
    const cpu = await readFile(new URL('../../web/hosts/wasm/inference-worker.mjs', import.meta.url), 'utf8');
    const gpu = await readFile(new URL('../../web/hosts/wasm/gpu-runtime-worker.mjs', import.meta.url), 'utf8');
    assert.match(cpu, /data\.type === 'editor'/);
    assert.match(cpu, /kidi_editor_enqueue/);
    assert.match(gpu, /router\.register\('editor'/);
    assert.match(gpu, /kidi_editor_enqueue/);
    assert.match(gpu, /streamGeneration\(request/);
});
