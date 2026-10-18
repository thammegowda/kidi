import test from 'node:test';
import assert from 'node:assert/strict';
import {webcrypto} from 'node:crypto';
import {benchmarkWorker, measuredResult, transcriptionErrorRates} from '../../benchmarks/browser/runtime_bench.mjs';

test('benchmark reports synthesis RTF from actual samples and delta GPU counters', () => {
    const result = measuredResult('tts', {
        samples: 46080, sample_rate: 24000,
        gpuStats: {dispatches: 100, readBytes: 1200, allocatedBytes: 4096},
    }, 3840, {dispatches: 20, readBytes: 400});
    assert.equal(result.audioSeconds, 1.92);
    assert.equal(result.rtf, 2);
    assert.equal(result.gpu.dispatches, 80);
    assert.equal(result.gpu.readBytes, 800);
    assert.equal(result.gpuAllocatedBytes, 4096);
});

test('diagnostic GPU timestamps are reported as per-trial deltas', () => {
    const result = measuredResult('chat', {
        gpuStats: {gpuMilliseconds: {attention: 12, linear: 9}},
    }, 100, {gpuMilliseconds: {attention: 7}});
    assert.deepEqual(result.gpuMilliseconds, {attention: 5, linear: 9});
});

test('speech quality metrics normalize punctuation but preserve content and Unicode letters', () => {
    assert.equal(transcriptionErrorRates('Please open the window.', 'please OPEN the window!').wordErrorRate, 0);
    const changed = transcriptionErrorRates('Open the window', 'Open a window');
    assert.equal(changed.wordErrors, 1);
    assert.equal(changed.wordErrorRate, 1 / 3);
    assert.equal(transcriptionErrorRates('Open the window', '').wordErrorRate, 1);
    assert.equal(transcriptionErrorRates('abc', 'axc').characterErrorRate, 1 / 3);
    assert.equal(transcriptionErrorRates('caf\u00e9', 'cafe\u0301').characterErrorRate, 0);
    assert.throws(() => transcriptionErrorRates('', 'text'), /Reference/);
});

test('fresh-input sweeps load once, record each input, and terminate the worker', async () => {
    const messages = [];
    let terminated = false;
    class Worker {
        postMessage(data) {
            messages.push(data);
            queueMicrotask(() => this.onmessage({data: data.type === 'load'
                ? {type: 'ready', gpuStats: {}}
                : {type: 'completed', text: data.messages[0].content, token_ids: [1], gpuStats: {}}}));
        }
        terminate() { terminated = true; }
    }
    const globals = {Worker, navigator: {userAgent: 'test', hardwareConcurrency: 1},
        crossOriginIsolated: false, crypto: webcrypto};
    const previous = new Map(Object.keys(globals).map(key => [key, Object.getOwnPropertyDescriptor(globalThis, key)]));
    try {
        for (const [key, value] of Object.entries(globals))
            Object.defineProperty(globalThis, key, {configurable: true, value});
        const request = ['green', 'blue'].map(content => ({type: 'generate', messages: [{role: 'user', content}]}));
        const observed = [];
        const result = await benchmarkWorker({kind: 'chat', workerFactory: () => new Worker(), load: {},
            request, trials: 3, warmups: 0, onResult: (data, trial) => observed.push([data.text, trial.warmup])});
        assert.equal(messages.filter(message => message.type === 'load').length, 1);
        assert.deepEqual(result.results.map(result => [result.inputIndex, result.text]), [[0, 'green'], [1, 'blue'], [0, 'green']]);
        assert.match(result.inputHash, /^[a-f0-9]{64}$/);
        assert.match(result.results[0].tokenHash, /^[a-f0-9]{64}$/);
        assert.deepEqual(observed, [['green', false], ['blue', false], ['green', false]]);
        assert.equal(terminated, true);
        await assert.rejects(benchmarkWorker({kind: 'chat', request: [], trials: 1}), /Invalid benchmark/);
    } finally {
        for (const [key, descriptor] of previous)
            if (descriptor) Object.defineProperty(globalThis, key, descriptor);
            else delete globalThis[key];
    }
});
