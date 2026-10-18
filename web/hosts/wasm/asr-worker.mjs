import {loadModel, storeConvertedSpeech} from './model-cache.mjs';

let module;
let gpu;

async function call(name, types = [], args = []) {
    const result = JSON.parse(module.ccall(name, 'string', types, args));
    if (gpu) await gpu.synchronize();
    if (result.error) throw new Error(result.error);
    return result;
}

self.onmessage = async ({data}) => {
    try {
        if (data.type === 'load') {
            if (module) throw new Error('Speech runtime is already loaded');
            const useGpu = data.backend === 'webgpu';
            if (!useGpu && !self.crossOriginIsolated)
                throw new Error('The CPU backend requires cross-origin isolation');
            const {default: createKidi} = await import(useGpu ? './gpu/kidi.mjs' : './threads/kidi.mjs');
            module = await createKidi({printErr: text => self.postMessage({type: 'log', text})});
            if (useGpu) {
                const {createWebGpu} = await import('./webgpu.mjs');
                gpu = await createWebGpu(module, {measure: data.measure === true, profile: data.profile === true,
                    lowerGraphs: data.lowerGraphs !== false, onlineAttention: data.onlineAttention ?? 'auto',
                    matrixPrecision: data.matrixPrecision, packedDot: data.packedDot, w8a8Kernel: data.w8a8Kernel});
            }
            await call('kidi_configure', ['number'], [useGpu ? 1 : data.threads]);
            const cache = await loadModel(module, data.source,
                metrics => self.postMessage({type: 'progress', ...metrics}),
                {cacheOnly: data.cacheOnly === true, gpu});
            const result = await call('kidi_load_asr', ['string'], ['/model']);
            self.postMessage({type: 'ready', ...result, ...cache, gpuStats: gpu?.stats});
            // Transcription can start meanwhile: storing only reads the converted file, now outside the Wasm heap.
            await storeConvertedSpeech(module, cache).catch(error =>
                self.postMessage({type: 'log', text: `Converted speech model not cached: ${error.message}`}));
            self.postMessage({type: 'cached'});
        } else if (data.type === 'transcribe') {
            if (!module || !(data.audio instanceof ArrayBuffer)) throw new Error('Speech runtime is not ready');
            const audio = new Float32Array(data.audio);
            const pointer = Number(module._malloc(audio.byteLength));
            if (!Number.isSafeInteger(pointer) || pointer <= 0 || pointer + audio.byteLength > module.HEAPU8.byteLength)
                throw new Error('Unable to allocate speech input in WebAssembly memory');
            try {
                new Float32Array(module.HEAPU8.buffer, pointer, audio.length).set(audio);
                let result = await call('kidi_asr_start', ['pointer', 'number', 'string', 'number'],
                    [pointer, audio.length, data.language, data.maximumTokens]);
                while (result.pending) {
                    self.postMessage({type: 'partial', requestId: data.requestId, text: result.text, language: result.language});
                    result = await call('kidi_asr_step');
                }
                self.postMessage({type: 'result', requestId: data.requestId, ...result, gpuStats: gpu?.stats});
            } finally { module._free(pointer); }
        }
    } catch (error) {
        self.postMessage({type: 'error', error: error?.stack || error?.message || String(error)});
    }
};