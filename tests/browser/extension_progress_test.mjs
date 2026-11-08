import assert from 'node:assert/strict';
import test from 'node:test';

import {formatBytes, modelProgressDetails, runtimeDiagnostics} from '../../web/extension/progress.mjs';

test('extension model progress reports real byte totals and network/cache sources', () => {
    const details = modelProgressDetails({
        type: 'progress',
        totalBytes: 2 * 1024 ** 3,
        loadedBytes: 1024 ** 3,
        downloadedBytes: 768 * 1024 ** 2,
        cachedBytes: 256 * 1024 ** 2,
        file: 'model.safetensors',
    }, 30_000);
    assert.equal(details.stage, 'Downloading model');
    assert.equal(details.percent, 50);
    assert.equal(details.bytes, '1.00 GB of 2.00 GB');
    assert.match(details.transfer, /768 MB network/);
    assert.match(details.transfer, /256 MB cached/);
    assert.match(details.transfer, /MB\/s/);
    assert.match(details.transfer, /remaining/);
    assert.equal(details.file, 'model.safetensors');
    assert.equal(details.determinate, true);
});

test('extension model progress distinguishes cache loading and initialization', () => {
    const cached = modelProgressDetails({
        type: 'progress',
        totalBytes: 100,
        loadedBytes: 75,
        downloadedBytes: 0,
        cachedBytes: 75,
    }, 1000);
    assert.equal(cached.stage, 'Reading cached model');
    assert.equal(cached.percent, 75);
    const initializing = modelProgressDetails({
        type: 'initializing',
        totalBytes: 100,
        loadedBytes: 100,
    }, 5000);
    assert.equal(initializing.stage, 'Initializing inference engine');
    assert.equal(initializing.percent, 100);
    assert.match(initializing.transfer, /Download complete/);
    assert.equal(formatBytes(1536), '2 KB');
});

test('extension model progress names metadata work before totals are known', () => {
    const details = modelProgressDetails({type: 'progress', file: 'config.json'}, 500);
    assert.equal(details.stage, 'Loading model metadata');
    assert.equal(details.percent, null);
    assert.equal(details.determinate, false);
});

test('extension runtime diagnostics report backend, memory, speed, tokens, and latency', () => {
    const diagnostics = runtimeDiagnostics({
        backend: 'webgpu',
        runtime: {
            requested_backend: 'webgpu',
            memory: {host_bytes: 768 * 1024 ** 2, gpu_bytes: 1024 ** 3},
        },
    }, {
        decode_tokens: 128,
        decode_ms: 4000,
        first_token_ms: 750,
    });
    assert.deepEqual(diagnostics, [
        {key: 'backend', label: 'Backend', value: 'WebGPU'},
        {key: 'ram', label: 'RAM', value: '768 MB'},
        {key: 'gpu', label: 'GPU mem', value: '1.00 GB'},
        {key: 'speed', label: 'Speed', value: '32.0 tok/s'},
        {key: 'tokens', label: 'Output', value: '128 tok'},
        {key: 'latency', label: 'TTFT', value: '0.75 s'},
    ]);
});
