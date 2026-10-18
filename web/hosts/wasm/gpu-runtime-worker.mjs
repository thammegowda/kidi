import {MessageTransport, ServiceRouter} from '../../../third_party/liteapp/web/liteapp.mjs';
import {createWebGpu} from './webgpu.mjs';
import {loadModel, storeConvertedSpeech} from './model-cache.mjs';
import {stageImages} from './images.js';
import {scopedModel} from './scoped-model.mjs';
import {RuntimeScheduler, awaitWithSignal} from './runtime-scheduler.mjs';
import {evictionCandidate, modelIdentity, protectedRepresentationOwners, synthesisGpuReserve,
    HOST_MODEL_BUDGET} from './model-residency.mjs';

const scheduler = new RuntimeScheduler();
const preparations = new RuntimeScheduler();
const maintenance = new RuntimeScheduler();
const clients = new Map();
const models = new Map();
let runtime;
let initialize;
let nextDirectory = 1;
let currentEmitter;
let reportedFailure;

function cancelled(signal) {
    signal?.throwIfAborted();
}

async function ensureRuntime(options) {
    if (!initialize) initialize = (async () => {
        const {default: create} = await import('./gpu/kidi.mjs');
        const module = await create({printErr: text => currentEmitter?.('worker', {type: 'log', text})});
        const gpu = await createWebGpu(module, {measure: options.measure === true, profile: options.profile === true,
            lowerGraphs: options.lowerGraphs !== false, onlineAttention: options.onlineAttention ?? 'auto'});
        const instance = {module, gpu, intentional: false};
        runtime = instance;
        reportedFailure = null;
        gpu.device.lost.then(info => {
            if (runtime === instance && !instance.intentional)
                reportFailure(`WebGPU device lost: ${info.message || info.reason}`);
        });
        return instance;
    })().catch(error => {
        initialize = null;
        throw error;
    });
    return initialize;
}

function reportFailure(message) {
    if (reportedFailure) return;
    reportedFailure = message;
    for (const model of models.values()) model.failure = message;
    self.postMessage({type: 'runtime.error', message});
}

function residency(model, state) {
    for (const session of model.clients)
        if (clients.has(session)) self.postMessage({type: 'runtime.residency', session, state, kind: model.kind});
}

async function call(model, name, types = [], args = [], signal) {
    return scheduler.run(model.kind === 'tts' || model.kind === 'asr' ? 2 : 1, async () => {
        cancelled(signal);
        if (model.failure) throw new Error(model.failure);
        const {module, gpu} = runtime;
        gpu.matrixPrecision = model.matrixPrecision;
        gpu.w8a8Kernel = model.data.w8a8Kernel ?? 'float';
        if (gpu.packedDotAvailable && model.data.packedDot !== false) gpu.features.add('packed-dot');
        else gpu.features.delete('packed-dot');
        currentEmitter = model.emit;
        module.kidiExternal = model.scope.kidiExternal;
        const selected = JSON.parse(module.ccall('kidi_session_select', 'string', ['number'], [model.handle]));
        if (selected.error) throw new Error(selected.error);
        try {
            const result = JSON.parse(gpu.withOwner(model.key, () => module.ccall(name, 'string', types, args)));
            await gpu.synchronize();
            if (result.error) throw new Error(result.error);
            return result;
        } catch (error) {
            if (gpu.failure) reportFailure(gpu.failure);
            throw error;
        } finally { currentEmitter = null; }
    }, signal);
}

async function disposeModel(model) {
    if (model.disposed || model.active || model.loading || model.pins || model.closing) return false;
    model.closing = true;
    model.closed = new Promise(resolve => { model.finishClose = resolve; });
    try {
        if (runtime) await scheduler.run(0, async () => {
        const {module, gpu} = runtime;
        await gpu.synchronize();
        if (model.handle !== null) {
            module.kidiExternal = model.scope?.kidiExternal;
            const result = JSON.parse(module.ccall('kidi_session_close', 'string', ['number'], [model.handle]));
            if (result.error) throw new Error(result.error);
            await gpu.synchronize();
        }
        for (const value of model.scope?.kidiExternal?.gpu?.values() || [])
            if (value.handle) gpu.release(value.handle);
        model.scope?.dispose();
        module.kidiExternal = null;
        gpu.trimBuffers(0);
        gpu.forgetOwner(model.key);
        });
        residency(model, 'evicted');
        for (const session of model.clients) {
            const client = clients.get(session);
            if (client?.model === model) client.model = null;
        }
        model.clients.clear();
        model.disposed = true;
        models.delete(model.key);
        return true;
    } catch (error) {
        reportFailure(`GPU model cleanup failed: ${error.message}`);
        throw error;
    } finally {
        model.finishClose();
    }
}

async function memoryTotals() {
    if (!runtime) return {gpu: 0, host: 0};
    const native = await scheduler.run(0, () =>
        JSON.parse(runtime.module.ccall('kidi_memory_stats', 'string', [], [])));
    if (native.error) throw new Error(native.error);
    return {gpu: runtime.gpu.stats.totalBufferBytes,
        host: (native.malloc_in_use_bytes || 0) +
            [...models.values()].reduce((bytes, model) => bytes + (model.cache?.externalWeightBytes || 0), 0)};
}

function hostOwned(model) {
    return (model.scope?.allocatedBytes || 0) + (model.cache?.externalWeightBytes || 0);
}

async function reserve(model = null, gpuBytes = 0, hostBytes = 0, signal) {
    for (;;) {
        const waiting = await maintenance.run(model?.kind === 'tts' || model?.kind === 'asr' ? 2 : 0, async () => {
            cancelled(signal);
            if (!runtime) return null;
            await scheduler.run(0, () => runtime.gpu.synchronize());
            runtime.gpu.trimBuffers(0);
            const protectedOwners = protectedRepresentationOwners(models.values(), model);
            for (;;) {
                const totals = await memoryTotals();
                const others = [...models.values()].filter(entry => entry !== model && entry.reservation);
                const futureGpu = others.reduce((bytes, entry) => bytes +
                    Math.max(0, entry.reservation.gpu - runtime.gpu.ownedBytes(entry.key)), 0);
                const futureHost = others.reduce((bytes, entry) => bytes +
                    Math.max(0, entry.reservation.host - hostOwned(entry)), 0);
                const excessGpu = totals.gpu + futureGpu + gpuBytes - runtime.gpu.memoryBudgetBytes;
                const pressure = excessGpu > 0 || totals.host + futureHost + hostBytes > HOST_MODEL_BUDGET;
                if (excessGpu > 0 && runtime.gpu.stats.preparedWeightBytes) {
                    const before = runtime.gpu.stats.preparedWeightBytes;
                    runtime.gpu.trimPreparedWeights(Math.max(0, before - excessGpu), protectedOwners);
                    runtime.gpu.trimBuffers(0);
                    if (runtime.gpu.stats.preparedWeightBytes < before) continue;
                }
                const candidate = evictionCandidate(models.values(), {now: performance.now(), pressure});
                if (candidate && candidate !== model) {
                    await disposeModel(candidate);
                    continue;
                }
                if (pressure) {
                    const blockers = [...models.values()].filter(entry =>
                        entry !== model && (entry.executing || entry.admitted));
                    if (blockers.length) return blockers.map(entry => entry.idle);
                    throw new Error('The GPU runtime memory budget is in use; release an idle model or use the CPU backend');
                }
                if (model) {
                    model.reservation = {gpu: runtime.gpu.ownedBytes(model.key) + gpuBytes,
                        host: hostOwned(model) + hostBytes};
                    if (model.loading) model.admitted = true;
                    else model.executing = true;
                } else if (!models.size) {
                    runtime.intentional = true;
                    runtime.gpu.dispose();
                    runtime = initialize = null;
                }
                return null;
            }
        }, signal);
        if (!waiting) return;
        await awaitWithSignal(Promise.race(waiting), signal);
    }
}

function finishRequest(model, client) {
    model.active = model.executing = false;
    model.reservation = null;
    model.pins--;
    model.lastUsed = performance.now();
    client.requesting = false;
    client.finishRequest?.();
    client.finishRequest = null;
    model.emit = null;
    model.finishIdle?.();
    model.finishIdle = null;
}

async function release(payload) {
    const client = clients.get(payload.session);
    if (!client) return {released: true};
    clients.delete(payload.session);
    client.model?.clients.delete(payload.session);
    if (client.model) {
        client.model.lastUsed = performance.now();
        if (client.model.loading && !client.model.clients.size) client.model.controller.abort();
    }
    await reserve();
    return {released: true};
}

async function prepareModel(model) {
    const signal = model.controller.signal;
    const mib = 1024 * 1024;
    let instance;
    try {
        cancelled(signal);
        instance = await ensureRuntime(model.data);
        const {module, gpu} = instance;
        model.scope ||= scopedModel(module, model.directory);
        if (model.matrixPrecision === 'mixed-f16' && !gpu.features.has('shader-f16'))
            throw new Error('Mixed FP16 matrices are unavailable; select the portable F32 profile');
        await reserve(model, (model.kind === 'chat' ? 1024 : model.kind === 'tts' ? 640 : 128) * mib,
            (model.kind === 'chat' ? 2048 : model.kind === 'tts' ? 768 : 256) * mib, signal);
        const created = await scheduler.run(0, () =>
            JSON.parse(module.ccall('kidi_session_create', 'string', [], [])), signal);
        if (created.error) throw new Error(created.error);
        model.handle = created.session_id;
        await call(model, 'kidi_configure', ['number'], [1], signal);
        const ownedGpu = new Proxy(gpu, {get(target, key) {
            const value = target[key];
            return typeof value === 'function' ? (...args) => gpu.withOwner(model.key, () => value.apply(gpu, args)) : value;
        }});
        model.cache = await loadModel(model.scope, model.source,
            metrics => model.emit?.('worker', {type: 'progress', ...metrics}),
            {cacheOnly: model.data.cacheOnly === true, gpu: ownedGpu});
        model.emit?.('worker', {type: 'initializing', ...model.cache});
        const result = await call(model, model.kind === 'chat' ? 'kidi_load' :
            model.kind === 'tts' ? 'kidi_load_tts' : 'kidi_load_asr', ['string'], [model.directory], signal);
        if (model.kind === 'asr') await storeConvertedSpeech(model.scope, model.cache);
        cancelled(signal);
        model.ready = {type: 'ready', ...result, ...model.cache, threads: 1,
            matrix_precision: model.matrixPrecision, gpuStats: {...gpu.stats}};
    } catch (error) {
        model.failure = error.message;
        if (instance?.gpu.failure) reportFailure(instance.gpu.failure);
        throw error;
    } finally {
        model.loading = model.admitted = false;
        model.reservation = null;
        model.finishLoading?.();
        model.emit = null;
        if (!model.clients.size && !model.pins && !reportedFailure)
            maintenance.run(0, () => disposeModel(model)).catch(error =>
                reportFailure(`Cancelled model cleanup failed: ${error.message}`));
    }
}

async function loadClient(client, emit, signal, pin = false) {
    cancelled(signal);
    if (reportedFailure) throw new Error(reportedFailure);
    let model = models.get(client.key);
    if (model?.closing) {
        await awaitWithSignal(model.closed, signal);
        model = models.get(client.key);
        if (reportedFailure) throw new Error(reportedFailure);
    }
    if (model?.failure || model?.controller.signal.aborted) {
        if (model.loading) await awaitWithSignal(model.idle, signal);
        await maintenance.run(0, () => disposeModel(model), signal);
        model = models.get(client.key);
    }
    const reused = Boolean(model?.ready);
    if (!model) {
        const directory = `/models/${client.kind}-${nextDirectory++}`;
        model = {key: client.key, source: client.data.manifest || client.data.source,
            kind: client.kind, matrixPrecision: client.data.matrixPrecision ?? 'f32', data: client.data,
            directory, scope: null, handle: null, clients: new Set(), loading: true, active: false,
            executing: false, pins: 0, lastUsed: performance.now(), emit, controller: new AbortController()};
        model.idle = new Promise(resolve => { model.finishLoading = resolve; });
        models.set(model.key, model);
        model.scope = runtime ? scopedModel(runtime.module, directory) : null;
        model.preparation = preparations.run(0, () => prepareModel(model));
        model.preparation.catch(error => {
            if (error.name !== 'AbortError') console.error('GPU model preparation failed:', error);
        });
    }
    client.model = model;
    model.clients.add(client.id);
    model.emit = emit;
    if (pin) model.pins++;
    if (model.loading) residency(model, 'loading');
    try {
        await awaitWithSignal(model.preparation, signal);
        cancelled(signal);
        if (clients.get(client.id) !== client) throw new DOMException('Model session was released', 'AbortError');
        model.lastUsed = performance.now();
        residency(model, 'ready');
        return {model, reused};
    } catch (error) {
        if (pin) model.pins--;
        model.clients.delete(client.id);
        if (client.model === model) client.model = null;
        if (model.loading && !model.clients.size) model.controller.abort();
        if (!model.loading && !model.active && !model.pins && !reportedFailure)
            await maintenance.run(0, () => disposeModel(model));
        throw error;
    }
}

async function modelFor(payload, emit, signal) {
    const client = clients.get(payload.session);
    if (!client) throw new Error('Load the model session before inference');
    if ((client.waiting || 0) >= 4) throw new Error('Too many queued requests for this logical model session');
    client.waiting = (client.waiting || 0) + 1;
    let model;
    let claimed = false;
    try {
        while (client.requesting) await awaitWithSignal(client.finished, signal);
        cancelled(signal);
        client.waiting--;
        claimed = true;
        client.requesting = true;
        client.finished = new Promise(resolve => { client.finishRequest = resolve; });
        ({model} = await loadClient(client, emit, signal, true));
        while (model.active) await awaitWithSignal(model.idle, signal);
        cancelled(signal);
        model.active = true;
        model.idle = new Promise(resolve => { model.finishIdle = resolve; });
        model.emit = emit;
        return {model, client};
    } catch (error) {
        if (model) model.pins--;
        if (claimed) {
            client.requesting = false;
            client.finishRequest?.();
            client.finishRequest = null;
        } else client.waiting--;
        throw error;
    }
}

const router = new ServiceRouter(new MessageTransport(self));
router.register('runtime.status', async () => ({
    type: 'runtime-status', memory: await memoryTotals(), gpuStats: runtime ? {...runtime.gpu.stats} : null,
    gpuRegions: runtime?.gpu.regionStats() || [],
    logicalSessions: clients.size,
    models: [...models.values()].map(model => ({
        kind: model.kind, source: model.source, matrixPrecision: model.matrixPrecision,
        active: model.active, loading: model.loading, pins: model.pins, clients: model.clients.size,
        gpuBytes: runtime?.gpu.ownedBytes(model.key) || 0,
    })),
}));
router.register('runtime.evict', async ({payload}) => {
    let count = 0;
    await maintenance.run(0, async () => {
        for (const model of [...models.values()])
            if ((payload.data?.all || model.kind === payload.kind) && await disposeModel(model)) count++;
    });
    await reserve();
    return {type: 'runtime-evicted', count};
});
router.register('runtime.reset', () => {
    if (runtime) runtime.gpu.device.destroy();
    return {type: 'runtime-reset'};
});
router.register('release', ({payload}) => release(payload));
router.register('load', async ({payload, emit, signal}) => {
    const key = modelIdentity(payload.kind, payload.data);
    const previous = clients.get(payload.session);
    if (previous) throw new Error('This logical model session is already loaded');
    if (clients.size >= 16) throw new Error('The shared GPU runtime supports at most 16 logical model sessions');
    const client = {id: payload.session, key, kind: payload.kind, data: payload.data, model: null, requesting: false};
    clients.set(client.id, client);
    try {
        const {model, reused} = await loadClient(client, emit, signal);
        return {...model.ready, gpuStats: {...runtime.gpu.stats}, warm: reused};
    } catch (error) {
        clients.delete(client.id);
        throw error;
    }
});

router.register('generate', async ({payload, emit, signal}) => {
    const {model, client} = await modelFor(payload, emit, signal);
    let requestId;
    let completed = false;
    try {
        const data = payload.data;
        const hasImages = data.messages?.some(message => message.images?.length);
        await reserve(model, (hasImages ? 1536 : 1024) * 1024 * 1024,
            (hasImages ? 512 : 64) * 1024 * 1024, signal);
        const staged = await stageImages(runtime.module, data.messages, {directory: `/images/request-${nextDirectory++}`});
        try {
            if (staged.messages.some(message => message.images.length))
                emit('worker', {type: 'encoding-images'});
            const enqueued = await call(model, 'kidi_enqueue', ['string', 'number', 'number'],
                [JSON.stringify(staged.messages), data.maximumTokens, data.imageMaxPixels ?? 3000000], signal);
            requestId = enqueued.request_id;
            cancelled(signal);
        } finally { staged.dispose(); }
        const started = performance.now();
        let firstTokenMs = null, decodeMs = 0, decodeTokens = 0, tokens = 0;
        for (;;) {
            const stepStart = performance.now();
            const step = await call(model, 'kidi_step', [], [], signal);
            completed = !step.pending;
            cancelled(signal);
            const count = step.events.filter(event => event.token !== undefined).length;
            if (count && model.reservation) {
                model.reservation = {gpu: runtime.gpu.ownedBytes(model.key) + 64 * 1024 * 1024,
                    host: hostOwned(model) + 16 * 1024 * 1024};
            }
            if (firstTokenMs !== null) { decodeMs += performance.now() - stepStart; decodeTokens += count; }
            else if (count) firstTokenMs = performance.now() - started;
            tokens += count;
            emit('worker', {
                type: 'step', ...step,
                stats: {tokenCount: tokens, elapsedMs: performance.now() - started, decodeMs, decodeTokens, firstTokenMs},
                gpuStats: {...runtime.gpu.stats},
            });
            if (completed) { model.warmed = true; return; }
        }
    } finally {
        try {
            if (!completed && requestId !== undefined && !reportedFailure)
                await call(model, 'kidi_cancel', ['number'], [requestId]);
        } finally { finishRequest(model, client); }
    }
});

router.register('synthesize', async ({payload, emit, signal}) => {
    const {model, client} = await modelFor(payload, emit, signal);
    let completed = false;
    try {
        const data = payload.data;
        const reservation = synthesisGpuReserve(model, data, runtime.gpu.preparedBytes(model.key));
        model.lastSynthesis = null;
        await reserve(model, reservation.bytes, 128 * 1024 * 1024, signal);
        let result = await call(model, 'kidi_tts_start', ['string', 'string'],
            [data.text, JSON.stringify(data.options || {})], signal);
        while (result.pending) {
            cancelled(signal);
            emit('worker', {type: 'generating', step: result.step, steps: result.steps});
            result = await call(model, 'kidi_tts_step', [], [], signal);
        }
        const pointer = Number(result.audio_address);
        const samples = Number(result.samples);
        if (!Number.isSafeInteger(pointer) || pointer <= 0 || !Number.isSafeInteger(samples) || samples <= 0 ||
            pointer + samples * 4 > runtime.module.HEAPU8.byteLength)
            throw new Error('TTS runtime returned invalid audio');
        const audio = new Float32Array(samples);
        try { audio.set(new Float32Array(runtime.module.HEAPU8.buffer, pointer, samples)); }
        finally { runtime.module._free(pointer); }
        completed = true;
        model.warmed = true;
        model.lastSynthesis = reservation.signature;
        model.lastChunks = result.chunks;
        model.requiredPreparedBytes = Math.max(model.requiredPreparedBytes || 0, runtime.gpu.preparedBytes(model.key));
        cancelled(signal);
        emit('worker', {type: 'audio', requestId: data.requestId, ...result, audio: audio.buffer,
            gpuStats: {...runtime.gpu.stats}}, [audio.buffer]);
    } finally {
        try {
            if (!completed && !reportedFailure) await call(model, 'kidi_tts_cancel');
        } finally { finishRequest(model, client); }
    }
});

router.register('voice-prompt', async ({payload, emit, signal}) => {
    const {model, client} = await modelFor(payload, emit, signal);
    let pointer = 0;
    try {
        if (model.kind !== 'tts') throw new Error('Voice prompts require a TTS model session');
        const data = payload.data;
        if (!(data.tokens instanceof ArrayBuffer) || !data.tokens.byteLength || data.tokens.byteLength % 4 ||
            typeof data.text !== 'string' || !data.text.trim())
            throw new Error('Invalid cached voice prompt');
        await reserve(model, 64 * 1024 * 1024, 16 * 1024 * 1024, signal);
        const tokens = new Int32Array(data.tokens);
        pointer = Number(runtime.module._malloc(tokens.byteLength));
        if (!pointer || pointer + tokens.byteLength > runtime.module.HEAPU8.byteLength)
            throw new Error('Cannot allocate cached voice prompt');
        new Int32Array(runtime.module.HEAPU8.buffer, pointer, tokens.length).set(tokens);
        const result = await call(model, 'kidi_tts_set_voice_prompt', ['string', 'pointer', 'number'],
            [data.text, pointer, tokens.length], signal);
        cancelled(signal);
        return {type: 'voice-prompt-ready', requestId: data.requestId, ...result};
    } finally {
        if (pointer) runtime.module._free(pointer);
        finishRequest(model, client);
    }
});

router.register('transcribe', async ({payload, emit, signal}) => {
    const {model, client} = await modelFor(payload, emit, signal);
    let pointer = 0;
    let completed = false;
    try {
        const data = payload.data;
        await reserve(model, (model.warmed ? 256 : 512) * 1024 * 1024, 128 * 1024 * 1024, signal);
        if (!(data.audio instanceof ArrayBuffer) || !data.audio.byteLength || data.audio.byteLength % 4)
            throw new Error('Invalid speech samples');
        const samples = new Float32Array(data.audio);
        pointer = Number(runtime.module._malloc(data.audio.byteLength));
        if (!pointer || pointer + data.audio.byteLength > runtime.module.HEAPU8.byteLength)
            throw new Error('Cannot allocate speech samples');
        new Float32Array(runtime.module.HEAPU8.buffer, pointer, samples.length).set(samples);
        let result = await call(model, 'kidi_asr_start', ['pointer', 'number', 'string', 'number'],
            [pointer, samples.length, data.language, data.maximumTokens], signal);
        while (result.pending) {
            cancelled(signal);
            emit('worker', {type: 'partial', requestId: data.requestId, text: result.text, language: result.language});
            result = await call(model, 'kidi_asr_step', [], [], signal);
        }
        completed = true;
        model.warmed = true;
        cancelled(signal);
        emit('worker', {type: 'result', requestId: data.requestId, ...result, gpuStats: {...runtime.gpu.stats}});
    } finally {
        try {
            if (!completed && !reportedFailure) await call(model, 'kidi_asr_cancel');
        } finally {
            if (pointer) runtime.module._free(pointer);
            finishRequest(model, client);
        }
    }
});

setInterval(() => reserve().catch(error => console.error('GPU model eviction failed:', error)), 30000);
