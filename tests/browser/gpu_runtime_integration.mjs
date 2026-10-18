import {createGpuWorker} from '../../web/hosts/wasm/runtime-client.mjs';

function session(kind) {
    const worker = createGpuWorker(kind), pending = new Set(), events = [], errors = [];
    let lost;
    const loss = new Promise(resolve => { lost = resolve; });
    const client = {
        worker, events, errors, loss, onGenerating: null,
        request(data, accepts) {
            return new Promise((resolve, reject) => {
                const entry = {data, accepts, resolve, reject};
                entry.timer = setTimeout(() => {
                    pending.delete(entry);
                    reject(new Error(`GPU integration request timed out: ${data.type}`));
                }, 180000);
                pending.add(entry);
                try { worker.postMessage(data); }
                catch (error) {
                    clearTimeout(entry.timer);
                    pending.delete(entry);
                    reject(error);
                }
            });
        },
    };
    worker.onmessage = ({data}) => {
        events.push({type: data.type, state: data.state, requestId: data.requestId});
        if (data.type === 'generating') client.onGenerating?.(data);
        for (const entry of pending) {
            if (data.requestId !== undefined && data.requestId !== entry.data.requestId) continue;
            if (data.type !== 'error' && !entry.accepts(data)) continue;
            clearTimeout(entry.timer);
            pending.delete(entry);
            if (data.type === 'error') entry.reject(new Error(data.error));
            else entry.resolve(data);
            break;
        }
    };
    worker.onerror = event => {
        errors.push(event.message);
        for (const entry of pending) {
            clearTimeout(entry.timer);
            entry.reject(new Error(event.message));
        }
        pending.clear();
        lost(event.message);
    };
    return client;
}

const ready = data => data.type === 'ready';
const audio = data => data.type === 'audio';
const status = data => data.type === 'runtime-status';
const evicted = data => data.type === 'runtime-evicted';
const completed = data => data.type === 'step' && !data.pending;
const require = (value, message) => { if (!value) throw new Error(message); };

export async function runGpuRuntimeIntegration({speech, chat = true} = {}) {
    if (!(speech instanceof ArrayBuffer) || !speech.byteLength)
        throw new Error('Integration checks require the 16 kHz JFK speech fixture');
    const clients = [], results = {};
    const create = kind => { const client = session(kind); clients.push(client); return client; };
    const tts = create('tts'), asr = create('asr'), inspector = create('tts');
    const ttsLoad = {type: 'load', source: './models/omnivoice/manifest.json', cacheOnly: true,
        backend: 'webgpu', threads: 1, matrixPrecision: 'w8a8', w8a8Kernel: 'float'};
    const synthesis = requestId => ({type: 'synthesize', requestId, text: 'The logo is for "kidi".',
        options: {language: 'English', steps: 16, seed: 0, pace: 1,
            use_voice_prompt: false, cache_voice_prompt: false,
            voice: {gender: 'female', age: 'middle-aged', pitch: 'moderate'}}});
    try {
        results.ttsLoad = await tts.request(ttsLoad, ready);
        await asr.request({type: 'load', source: 'openai/whisper-tiny', backend: 'webgpu', threads: 1, cacheOnly: true}, ready);
        let llm;
        if (chat) {
            llm = create('chat');
            await llm.request({type: 'load', manifest: 'google/gemma-4-E2B-it-qat-mobile-transformers',
                backend: 'webgpu', threads: 1, cacheOnly: true}, ready);
        }
        const operations = [
            tts.request(synthesis('shared-speech'), audio),
            asr.request({type: 'transcribe', requestId: 'shared-asr', audio: speech, language: 'en', maximumTokens: 128},
                data => data.type === 'result'),
        ];
        if (llm) operations.push(llm.request({type: 'generate',
            messages: [{role: 'user', content: 'Reply only with Hello.', images: []}], maximumTokens: 16}, completed));
        const [spoken, recognized, generated] = await Promise.all(operations);
        require(spoken.samples === 46080 && spoken.audio.byteLength === 46080 * 4, 'Shared TTS output shape changed');
        require(recognized.token_ids.length === 24 && recognized.text.includes('fellow Americans'),
            'Shared ASR did not preserve the JFK transcription');
        if (generated) require(generated.events.some(event => event.completed?.text.includes('Hello')),
            'Shared chat did not complete correctly');
        results.mixed = {samples: spoken.samples, transcript: recognized.text, attempts: recognized.attempts,
            chat: generated?.events.find(event => event.completed)?.completed.text};
        results.resident = await inspector.request({type: 'runtime.status'}, status);
        require(results.resident.gpuStats.totalBufferBytes <= results.resident.gpuStats.memoryBudgetBytes,
            'Mixed inference exceeded the GPU buffer quota');

        let generating;
        const started = new Promise(resolve => { generating = resolve; });
        tts.onGenerating = () => { tts.onGenerating = null; generating(); };
        const cancelled = tts.request(synthesis('cancel-old'), data => data.type === 'cancelled' || data.type === 'audio');
        await started;
        tts.worker.postMessage({type: 'cancel'});
        const next = tts.request(synthesis('after-cancel'), audio);
        require((await cancelled).type === 'cancelled', 'Cancellation was not acknowledged');
        require((await next).samples === 46080, 'Synthesis did not recover after cancellation');
        results.cancel = true;

        results.eviction = await inspector.request({type: 'runtime.evict', all: true}, evicted);
        const empty = await inspector.request({type: 'runtime.status'}, status);
        require(empty.models.length === 0 && empty.memory.gpu === 0, 'Eviction retained a native model or device');
        const beforeReload = tts.events.length;
        require((await tts.request(synthesis('after-eviction'), audio)).samples === 46080,
            'Attached logical session did not reload after eviction');
        require(tts.events.slice(beforeReload).some(event => event.type === 'residency' && event.state === 'ready'),
            'Lazy reload did not restore resident availability');
        results.lazyReload = true;

        inspector.worker.postMessage({type: 'runtime.reset'});
        await Promise.all(clients.map(client => client.loss));
        require(clients.every(client => client.errors.length === 1 && client.worker.closed),
            'Device loss did not invalidate every logical lease exactly once');
        const replacement = create('tts');
        await replacement.request(ttsLoad, ready);
        require((await replacement.request(synthesis('after-device-loss'), audio)).samples === 46080,
            'GPU runtime did not recreate successfully after device loss');
        results.deviceRecovery = true;
        await replacement.request({type: 'runtime.evict', all: true}, evicted);
        results.final = await replacement.request({type: 'runtime.status'}, status);
        require(results.final.memory.gpu === 0 && results.final.models.length === 0,
            'Integration checks left GPU model resources allocated');
        return results;
    } finally {
        for (const client of clients) client.worker.terminate();
    }
}
