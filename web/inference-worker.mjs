import {loadModel} from './model-cache.mjs';

let module;
let active = false;
let ready = false;
let requestId;
let generation;
let generationStarted;
function call(name, types = [], args = []) {
    const result = JSON.parse(module.ccall(name, 'string', types, args));
    if (result.error) throw new Error(result.error);
    return result;
}
function step() {
    if (!active) return;
    try {
        const started = performance.now();
        const result = call('kidi_step');
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
        self.postMessage({type: 'step', ...result, stats: generation, heapBytes: module.HEAPU8.byteLength});
        active = result.pending > 0;
        if (active) setTimeout(step, 0);
    } catch (error) { fail(error, true); }
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
            if (data.threads > 1 && !self.crossOriginIsolated)
                throw new Error('Multiple threads require COOP/COEP response headers');
            const threaded = data.threads > 1;
            const {default: createKidi} = await import(threaded ? './threads/kidi.mjs' : './single/kidi.mjs');
            module = await createKidi({printErr: text => self.postMessage({type: 'log', text})});
            call('kidi_configure', ['number'], [data.threads]);
            const cache = await loadModel(module, data.manifest,
                metrics => self.postMessage({type: 'progress', ...metrics, heapBytes: module.HEAPU8.byteLength}),
                {cacheOnly: data.cacheOnly});
            self.postMessage({type: 'initializing', ...cache});
            const result = call('kidi_load', ['string'], ['/model']);
            ready = true;
            self.postMessage({type: 'ready', ...result, ...cache, loadMs: performance.now() - started,
                heapBytes: module.HEAPU8.byteLength});
        } else if (data.type === 'generate') {
            if (!ready || active) throw new Error('Runtime is not ready for a new request');
            generationStarted = performance.now();
            generation = {tokenCount: 0, elapsedMs: 0, decodeMs: 0, decodeTokens: 0, firstTokenMs: null};
            const result = call('kidi_enqueue', ['string', 'number'], [JSON.stringify(data.messages), data.maximumTokens]);
            requestId = result.request_id;
            active = true;
            setTimeout(step, 0);
        } else if (data.type === 'cancel' && active) {
            call('kidi_cancel', ['number'], [requestId]);
            active = false;
            generation.elapsedMs = performance.now() - generationStarted;
            self.postMessage({type: 'cancelled', stats: generation, heapBytes: module.HEAPU8.byteLength});
        }
    } catch (error) { fail(error, data.type === 'load'); }
};