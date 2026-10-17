import {loadModel} from './model-cache.mjs';

let module;
let ready = false;
let gpu;

async function call(name, types = [], args = []) {
    const result = JSON.parse(module.ccall(name, 'string', types, args));
    if (gpu) await gpu.synchronize();
    if (result.error) throw new Error(result.error);
    return result;
}

async function synthesize(text, options) {
    let result = await call('kidi_tts_start', ['string', 'string'], [text, JSON.stringify(options || {})]);
    while (result.pending) {
        self.postMessage({type: 'generating', step: result.step, steps: result.steps});
        result = await call('kidi_tts_step');
    }
    const pointer = Number(result.audio_address);
    const samples = Number(result.samples);
    if (!Number.isSafeInteger(pointer) || pointer <= 0 || !Number.isSafeInteger(samples) || samples <= 0 ||
        pointer + samples * 4 > module.HEAPU8.byteLength)
        throw new Error('TTS runtime returned invalid audio');
    const audio = new Float32Array(samples);
    audio.set(new Float32Array(module.HEAPU8.buffer, pointer, samples));
    module._free(pointer);
    if (gpu && options?.diagnostics) {
        const tokens = new Int32Array(result.token_ids);
        const source = Number(module._malloc(tokens.byteLength));
        if (!source) throw new Error('Cannot allocate codec validation tokens');
        let reference;
        try {
            new Int32Array(module.HEAPU8.buffer, source, tokens.length).set(tokens);
            reference = await call('kidi_tts_decode_reference', ['pointer', 'number'], [source, tokens.length]);
        } finally { module._free(source); }
        const referencePointer = Number(reference.audio_address);
        try {
            if (reference.samples !== samples || referencePointer <= 0 ||
                referencePointer + samples * 4 > module.HEAPU8.byteLength)
                throw new Error('Reference waveform shape differs from WebGPU output');
            const expected = new Float32Array(module.HEAPU8.buffer, referencePointer, samples);
            let signal = 0, error = 0, maximum = 0;
            for (let index = 0; index < samples; ++index) {
                const delta = audio[index] - expected[index];
                signal += expected[index] ** 2;
                error += delta ** 2;
                maximum = Math.max(maximum, Math.abs(delta));
            }
            result.codec_quality = {
                samples, maxAbsoluteError: maximum, rmse: Math.sqrt(error / samples),
                snrDb: error ? 10 * Math.log10(signal / error) : null,
            };
            result.reference_audio = expected.slice().buffer;
        } finally { module._free(referencePointer); }
    }
    return {result, audio};
}

self.onmessage = async ({data}) => {
    try {
        if (data.type === 'load') {
            if (module) throw new Error('TTS runtime is already loaded');
            const useGpu = data.backend === 'webgpu';
            const threaded = !useGpu && data.threads > 1;
            const {default: createKidi} = await import(
                useGpu ? './gpu/kidi.mjs' : threaded ? './threads/kidi.mjs' : './single/kidi.mjs');
            module = await createKidi({printErr: text => self.postMessage({type: 'log', text})});
            if (useGpu) {
                const {createWebGpu} = await import('./webgpu.mjs');
                gpu = await createWebGpu(module, {profile: data.profile === true, measure: data.measure === true,
                    lowerGraphs: data.lowerGraphs !== false, onlineAttention: data.onlineAttention ?? 'auto',
                    matrixPrecision: data.matrixPrecision, packedDot: data.packedDot, w8a8Kernel: data.w8a8Kernel});
                gpu.device.lost.then(info => self.postMessage({
                    type: 'error',
                    error: `WebGPU device lost: ${info.message || info.reason}`,
                }));
            }
            await call('kidi_configure', ['number'], [data.threads]);
            const cache = await loadModel(module, data.source,
                metrics => self.postMessage({type: 'progress', ...metrics}),
                {cacheOnly: data.cacheOnly === true, gpu});
            const result = await call('kidi_load_tts', ['string'], ['/model']);
            ready = true;
            self.postMessage({type: 'ready', threads: data.threads, ...result, ...cache,
                gpuStats: gpu?.stats, matrix_precision: gpu?.matrixPrecision, packed_dot: gpu?.features.has('packed-dot')});
        } else if (data.type === 'synthesize') {
            if (!ready) throw new Error('TTS runtime is not ready');
            const {result, audio} = await synthesize(data.text, data.options);
            const transfer = [audio.buffer];
            if (result.reference_audio instanceof ArrayBuffer) transfer.push(result.reference_audio);
            self.postMessage({type: 'audio', requestId: data.requestId, ...result, audio: audio.buffer,
                gpuStats: gpu?.stats}, transfer);
        }
    } catch (error) {
        self.postMessage({
            type: 'error',
            requestId: data.requestId,
            error: error?.stack || error?.message || String(error),
        });
    }
};
