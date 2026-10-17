import test from 'node:test';
import assert from 'node:assert/strict';
import {evictionCandidate, modelIdentity, protectedRepresentationOwners, synthesisGpuReserve} from '../../web/hosts/wasm/model-residency.mjs';

const model = (key, lastUsed, extra = {}) => ({key, lastUsed, pins: 0, clients: new Set(), ...extra});

test('idle attached logical clients do not pin a native model forever', () => {
    const idle = model('chat', 0, {clients: new Set(['chat-tab'])});
    const recent = model('asr', 299999);
    assert.equal(evictionCandidate([recent, idle], {now: 300000}), idle);
    assert.equal(evictionCandidate([recent], {now: 300000}), undefined);
});

test('memory pressure evicts the least recent unpinned model, not a loading or active request', () => {
    const idle = model('tts', 20);
    const entries = [model('active', 0, {active: true}), model('loading', 1, {loading: true}),
        model('pending', 2, {pins: 1}), model('closing', 3, {closing: true}), idle, model('new', 30)];
    assert.equal(evictionCandidate(entries, {now: 40, pressure: true}), idle);
    assert.equal(evictionCandidate(entries.slice(0, 4), {now: 400000, pressure: true}), undefined);
});

test('resident model count is bounded before a fourth model finishes loading', () => {
    const oldest = model('old', 10);
    const entries = [oldest, model('second', 20), model('third', 30), model('incoming', 40, {loading: true})];
    assert.equal(evictionCandidate(entries, {now: 40}), oldest);
    assert.equal(evictionCandidate(entries.slice(1), {now: 40}), undefined);
});

test('admission cannot free representations that the requested model immediately needs again', () => {
    const requested = model('tts', 20), active = model('asr', 10, {executing: true});
    const loading = model('chat', 30, {admitted: true}), idle = model('other', 0);
    assert.deepEqual([...protectedRepresentationOwners([idle, requested, active, loading], requested)],
        ['tts', 'asr', 'chat']);
});

test('model reuse includes execution precision and validates sources before device creation', () => {
    assert.equal(modelIdentity('tts', {source: 'speech'}), modelIdentity('tts', {manifest: 'speech', matrixPrecision: 'f32'}));
    assert.notEqual(modelIdentity('tts', {source: 'speech'}),
        modelIdentity('tts', {source: 'speech', matrixPrecision: 'mixed-f16'}));
    assert.equal(modelIdentity('tts', {source: 'speech', matrixPrecision: 'w8a8'}),
        modelIdentity('tts', {source: 'speech', matrixPrecision: 'w8a8', w8a8Kernel: 'float'}));
    assert.notEqual(modelIdentity('tts', {source: 'speech', matrixPrecision: 'w8a8'}),
        modelIdentity('tts', {source: 'speech', matrixPrecision: 'w8a8', w8a8Kernel: 'packed'}));
    assert.throws(() => modelIdentity('unknown', {source: 'speech'}), /valid kind/);
    assert.throws(() => modelIdentity('chat', {}), /valid kind/);
    assert.throws(() => modelIdentity('tts', {source: 'speech', matrixPrecision: 'invalid'}), /precision/);
});

test('speech reserves new-shape scratch and any evicted prepared weights', () => {
    const mib = 1024 * 1024;
    const data = {text: 'Hello', options: {use_voice_prompt: false, cache_voice_prompt: false}};
    const cold = {matrixPrecision: 'mixed-f16', warmed: false};
    assert.equal(synthesisGpuReserve(cold, data, 0).bytes, 1536 * mib);
    const warm = {...cold, warmed: true, lastChunks: 1, lastSynthesis: synthesisGpuReserve(cold, data, 0).signature,
        requiredPreparedBytes: 900 * mib};
    assert.equal(synthesisGpuReserve(warm, data, 900 * mib).bytes, 128 * mib);
    assert.equal(synthesisGpuReserve(warm, data, 600 * mib).bytes, 428 * mib);
    assert.equal(synthesisGpuReserve(warm, {...data, text: 'A longer utterance'}, 900 * mib).bytes, 768 * mib);
    assert.equal(synthesisGpuReserve({...warm, lastChunks: 2}, data, 900 * mib).bytes, 768 * mib);
    assert.equal(synthesisGpuReserve(warm, {...data, text: 'A new short phrase', streamSegment: true}, 900 * mib).bytes, 768 * mib);
    assert.throws(() => synthesisGpuReserve(warm, {...data, text: 'x'.repeat(300), streamSegment: true}, 0), /text budget/);
});
