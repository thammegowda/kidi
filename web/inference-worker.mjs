import {loadModel} from './model-cache.mjs';

let module;
let active = false;
let ready = false;
let requestId;
let generation;
let generationStarted;
let gpu;
let inFlight = false;
let cancelRequested = false;
async function call(name, types = [], args = []) {
    const result = JSON.parse(await module.ccall(name, 'string', types, args, gpu ? {async: true} : {}));
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
        self.postMessage({type: 'cancelled', stats: generation, heapBytes: module.HEAPU8.byteLength, gpuStats: gpu?.stats});
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
        self.postMessage({type: 'step', ...result, stats: generation, heapBytes: module.HEAPU8.byteLength, gpuStats: gpu?.stats});
        active = result.pending > 0;
        if (active) setTimeout(step, 0);
    } catch (error) { fail(error, true); }
    finally { inFlight = false; }
}
function fail(error, fatal = false) {
    active = false;
    if (fatal) ready = false;
    const fallback = `${error?.constructor?.name || typeof error}: ${String(error)}`;
    self.postMessage({type: 'error', error: error?.stack || error?.message || fallback, fatal,
        heapBytes: module?.HEAPU8?.byteLength, stats: generation});
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
            const {default: createKidi} = await import(useGpu ? './gpu/kidi.mjs' : threaded ? './threads/kidi.mjs' : './single/kidi.mjs');
            module = await createKidi({printErr: text => self.postMessage({type: 'log', text})});
            if (useGpu) {
                const {createWebGpu} = await import('./webgpu.mjs');
                gpu = await createWebGpu(module, {profile: data.profile === true});
                gpu.device.lost.then(info => fail(new Error(`WebGPU device lost: ${info.message || info.reason}`), true));
            }
            await call('kidi_configure', ['number'], [useGpu ? 1 : data.threads]);
            const cache = await loadModel(module, data.manifest,
                metrics => self.postMessage({type: 'progress', ...metrics, heapBytes: module.HEAPU8.byteLength}),
                {cacheOnly: data.cacheOnly});
            self.postMessage({type: 'initializing', ...cache});
            const result = await call('kidi_load', ['string'], ['/model']);
            ready = true;
            self.postMessage({type: 'ready', ...result, ...cache, loadMs: performance.now() - started,
                heapBytes: module.HEAPU8.byteLength, gpuStats: gpu?.stats});
        } else if (data.type === 'generate') {
            if (!ready || active || inFlight) throw new Error('Runtime is not ready for a new request');
            active = true;
            inFlight = true;
            cancelRequested = false;
            generationStarted = performance.now();
            generation = {tokenCount: 0, elapsedMs: 0, decodeMs: 0, decodeTokens: 0, firstTokenMs: null};
            const result = await call('kidi_enqueue', ['string', 'number'], [JSON.stringify(data.messages), data.maximumTokens]);
            requestId = result.request_id;
            inFlight = false;
            setTimeout(step, 0);
        } else if (data.type === 'cancel' && active) {
            cancelRequested = true;
            if (!inFlight) await cancel();
        }
    } catch (error) { inFlight = false; fail(error, data.type === 'load'); }
};