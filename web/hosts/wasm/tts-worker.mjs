import {loadModel} from './model-cache.mjs';

let module;
let ready = false;

async function call(name, types = [], args = []) {
    const result = JSON.parse(module.ccall(name, 'string', types, args));
    if (result.error) throw new Error(result.error);
    return result;
}

async function synthesize(text, options) {
    const result = await call('kidi_synthesize', ['string', 'string'], [text, JSON.stringify(options || {})]);
    const pointer = Number(result.audio_address);
    const samples = Number(result.samples);
    if (!Number.isSafeInteger(pointer) || pointer <= 0 || !Number.isSafeInteger(samples) || samples <= 0 ||
        pointer + samples * 4 > module.HEAPU8.byteLength)
        throw new Error('TTS runtime returned invalid audio');
    const audio = new Float32Array(samples);
    audio.set(new Float32Array(module.HEAPU8.buffer, pointer, samples));
    module._free(pointer);
    return {result, audio};
}

self.onmessage = async ({data}) => {
    try {
        if (data.type === 'load') {
            if (module) throw new Error('TTS runtime is already loaded');
            const threaded = data.threads > 1;
            const {default: createKidi} = await import(threaded ? './threads/kidi.mjs' : './single/kidi.mjs');
            module = await createKidi({printErr: text => self.postMessage({type: 'log', text})});
            await call('kidi_configure', ['number'], [data.threads]);
            const cache = await loadModel(module, data.source,
                metrics => self.postMessage({type: 'progress', ...metrics}), {cacheOnly: data.cacheOnly === true});
            const result = await call('kidi_load_tts', ['string'], ['/model']);
            ready = true;
            self.postMessage({type: 'ready', ...result, ...cache});
        } else if (data.type === 'synthesize') {
            if (!ready) throw new Error('TTS runtime is not ready');
            const {result, audio} = await synthesize(data.text, data.options);
            self.postMessage({type: 'audio', requestId: data.requestId, ...result, audio: audio.buffer}, [audio.buffer]);
        }
    } catch (error) {
        self.postMessage({
            type: 'error',
            requestId: data.requestId,
            error: error?.stack || error?.message || String(error),
        });
    }
};
