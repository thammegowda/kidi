import {loadModel, MAX_WASM_MEMORY, webGpuSupport} from './model-cache.mjs';
import {stageImages} from './images.mjs';

let module;
let active = false;
let ready = false;
let requestId;
let generation;
let generationStarted;
let gpu;
let inFlight = false;
let cancelRequested = false;
let loaded = {};
let nativeMemory;
// Native allocator statistics walk the heap (tens of milliseconds), so they refresh only when a phase ends; heap and
// GPU sizes below stay live.
function memory(fresh = false) {
    if (!module) return undefined;
    if (fresh) {
        try {
            nativeMemory = JSON.parse(module.ccall('kidi_memory_stats', 'string', [], []));
        } catch {}
    }
    return {...nativeMemory, heap_bytes: module.HEAPU8.byteLength,
        heap_limit_bytes: nativeMemory?.heap_limit_bytes ?? MAX_WASM_MEMORY,
        weights_heap_bytes: loaded.heapWeightBytes, weights_external_bytes: loaded.externalWeightBytes,
        weights_gpu_bytes: loaded.gpuWeightBytes,
        gpu_buffer_bytes: gpu?.stats.allocatedBytes, gpu_pooled_bytes: gpu?.stats.pooledBytes};
}
function post(message, fresh = false) {
    self.postMessage({...message, heapBytes: module?.HEAPU8?.byteLength, gpuStats: gpu?.stats, memory: memory(fresh)});
}
// Wasm never waits on the GPU: each call submits work, and its results (such as the selected token reported by the
// next kidi_step) are complete once the runtime's synchronize() resolves here, between calls.
async function call(name, types = [], args = []) {
    const result = JSON.parse(module.ccall(name, 'string', types, args));
    if (gpu) await gpu.synchronize();
    if (result.error) throw new Error(result.error);
    return result;
}
async function cancel() {
    if (!active || inFlight) return;
    inFlight = true;
    try {
        await call('kidi_cancel', ['number'], [requestId]);
        active = false;
        cancelRequested = false;
        generation.elapsedMs = performance.now() - generationStarted;
        post({type: 'cancelled', stats: generation}, true);
    } catch (error) { fail(error, true); }
    finally { inFlight = false; }
}
async function step() {
    if (!active || inFlight) return;
    if (cancelRequested) return cancel();
    inFlight = true;
    try {
        const started = performance.now();
        const result = await call('kidi_step');
        const elapsed = performance.now() - started;
        const tokens = result.events.filter(event => event.token !== undefined).length;
        if (generation.firstTokenMs !== null) {
            generation.decodeMs += elapsed;
            generation.decodeTokens += tokens;
        } else if (tokens) generation.firstTokenMs = performance.now() - generationStarted;
        generation.tokenCount += tokens;
        generation.elapsedMs = performance.now() - generationStarted;
        const completed = result.events.find(event => event.completed)?.completed;
        if (completed) Object.assign(generation, {tokenCount: completed.token_ids.length,
            elapsedMs: completed.generation_ms, decodeMs: completed.decode_ms, decodeTokens: completed.decode_tokens});
        post({type: 'step', ...result, stats: generation}, Boolean(completed));
        active = result.pending > 0;
        if (active) setTimeout(step, 0);
    } catch (error) { fail(error, true); }
    finally { inFlight = false; }
}
function fail(error, fatal = false) {
    active = false;
    if (fatal) ready = false;
    const fallback = `${error?.constructor?.name || typeof error}: ${String(error)}`;
    post({type: 'error', error: error?.stack || error?.message || fallback, fatal, stats: generation}, true);
}
self.onmessage = async ({data}) => {
    try {
        if (data.type === 'load') {
            if (module) throw new Error('Restart the worker to change runtime settings');
            const started = performance.now();
            const useGpu = data.backend === 'webgpu';
            if (!useGpu && data.threads > 1 && !self.crossOriginIsolated)
                throw new Error('Multiple threads require COOP/COEP response headers');
            const threaded = !useGpu && data.threads > 1;
            const unsupported = useGpu && webGpuSupport();
            if (unsupported) throw new Error(`WebGPU is unavailable: ${unsupported}. Use the WebAssembly CPU backend.`);
            const {default: createKidi} = await import(useGpu ? './gpu/kidi.mjs' : threaded ? './threads/kidi.mjs' : './single/kidi.mjs');
            module = await createKidi({printErr: text => self.postMessage({type: 'log', text})});
            if (useGpu) {
                const {createWebGpu} = await import('./webgpu.mjs');
                gpu = await createWebGpu(module, {profile: data.profile === true});
                gpu.device.lost.then(info => fail(new Error(`WebGPU device lost: ${info.message || info.reason}`), true));
            }
            await call('kidi_configure', ['number'], [useGpu ? 1 : data.threads]);
            const cache = await loadModel(module, data.manifest,
                metrics => post({type: 'progress', ...metrics}),
                {cacheOnly: data.cacheOnly, gpu});
            loaded = cache;
            post({type: 'initializing', ...cache}, true);
            const result = await call('kidi_load', ['string'], ['/model']);
            ready = true;
            post({type: 'ready', ...result, ...cache, loadMs: performance.now() - started}, true);
        } else if (data.type === 'generate') {
            if (!ready || active || inFlight) throw new Error('Runtime is not ready for a new request');
            active = true;
            inFlight = true;
            cancelRequested = false;
            generationStarted = performance.now();
            generation = {tokenCount: 0, elapsedMs: 0, decodeMs: 0, decodeTokens: 0, firstTokenMs: null};
            const staged = await stageImages(module, data.messages);
            let result;
            try {
                if (staged.messages.some(message => message.images.length))
                    post({type: 'encoding-images'});
                result = await call('kidi_enqueue', ['string', 'number', 'number'],
                    [JSON.stringify(staged.messages), data.maximumTokens, data.imageMaxPixels ?? 3000000]);
            } finally { staged.dispose(); }
            requestId = result.request_id;
            inFlight = false;
            setTimeout(step, 0);
        } else if (data.type === 'cancel' && active) {
            cancelRequested = true;
            if (!inFlight) await cancel();
        }
    } catch (error) { inFlight = false; fail(error, data.type === 'load'); }
};