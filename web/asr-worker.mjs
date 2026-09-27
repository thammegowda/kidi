import {loadModel} from './model-cache.mjs';

let module;

async function call(name, types = [], args = []) {
    const result = JSON.parse(module.ccall(name, 'string', types, args));
    if (result.error) throw new Error(result.error);
    return result;
}

self.onmessage = async ({data}) => {
    try {
        if (data.type === 'load') {
            if (module) throw new Error('Speech runtime is already loaded');
            const threaded = data.threads > 1;
            const {default: createKidi} = await import(threaded ? './threads/kidi.mjs' : './single/kidi.mjs');
            module = await createKidi({printErr: text => self.postMessage({type: 'log', text})});
            await call('kidi_configure', ['number'], [data.threads]);
            const cache = await loadModel(module, data.source,
                metrics => self.postMessage({type: 'progress', ...metrics}), {cacheOnly: false});
            const result = await call('kidi_load_asr', ['string'], ['/model']);
            self.postMessage({type: 'ready', ...result, ...cache});
        } else if (data.type === 'transcribe') {
            if (!module || !(data.audio instanceof ArrayBuffer)) throw new Error('Speech runtime is not ready');
            const audio = new Float32Array(data.audio);
            const pointer = Number(module._malloc(audio.byteLength));
            if (!Number.isSafeInteger(pointer) || pointer <= 0 || pointer + audio.byteLength > module.HEAPU8.byteLength)
                throw new Error('Unable to allocate speech input in WebAssembly memory');
            try {
                new Float32Array(module.HEAPU8.buffer, pointer, audio.length).set(audio);
                const result = await call('kidi_transcribe', ['number', 'number', 'string', 'number'],
                    [pointer, audio.length, data.language, data.maximumTokens]);
                self.postMessage({type: 'result', requestId: data.requestId, ...result});
            } finally { module._free(pointer); }
        }
    } catch (error) {
        self.postMessage({type: 'error', error: error?.stack || error?.message || String(error)});
    }
};