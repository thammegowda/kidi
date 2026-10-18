import test from 'node:test';
import assert from 'node:assert/strict';
import {setTimeout as delay, setImmediate} from 'node:timers/promises';

let nextModule = 0;
const TEST_APP_CONFIG = {
    version: 1,
    model_hub: {remote: 'https://huggingface.co', local: '~/.cache/kidi/model-hub'},
    runtime: {threads: 4},
    models: {
        chat: {enabled: true, model_id: 'google/test', accelerator: 'auto', precision: 'checkpoint',
            kv_cache_precision: 'auto', system_prompt: 'Test prompt',
            generation: {max_tokens: 128, context_tokens: 1024, prefix_cache_mb: 0}},
        asr: {enabled: true, model_id: 'openai/whisper-small', language: 'auto', max_tokens: 128,
            vad: {threshold: .012, minimum_speech_ms: 250, silence_ms: 800}},
        tts: {enabled: true, model_id: 'test/omnivoice', language: 'auto', steps: 10, realtime_steps: 6,
            pace: 1.15, reference_language: 'English', reference_text: 'Hello.',
            voice: {gender: 'female', age: 'young adult', pitch: 'moderate', style: '', accent: ''}},
    },
};
function deferred() {
    let resolve;
    const promise = new Promise(yes => { resolve = yes; });
    return {promise, resolve};
}

function asrFrame(streamId, sequence, samples = 320) {
    const frame = new ArrayBuffer(24 + samples * 2);
    new Uint8Array(frame).set([0x4b, 0x44, 0x41, 0x31, 1, 1]);
    const header = new DataView(frame);
    header.setUint32(8, streamId, true);
    header.setUint32(12, sequence, true);
    header.setUint32(16, 16000, true);
    header.setUint32(20, samples * 2, true);
    new Int16Array(frame, 24).fill(1024);
    return new Uint8Array(frame);
}

const visibleSynthesis = worker => worker.calls.filter(call =>
    call.type === 'synthesize' && call.options?.return_tokens !== true);
async function waitFor(predicate, message = 'Timed out waiting for test state') {
    const deadline = Date.now() + 2000;
    while (Date.now() < deadline) {
        const value = predicate();
        if (value) return value;
        await delay(1);
    }
    throw new Error(message);
}

async function withBackend(action, {ready = true} = {}) {
    const workers = [], events = [], cacheEntries = new Map();
    class Worker {
        constructor(url) {
            this.url = String(url);
            this.calls = [];
            workers.push(this);
        }
        postMessage(data) {
            this.calls.push(data);
            if (data.type === 'load' && ready)
                queueMicrotask(() => this.emit({type: 'ready', backend: 'wasm-cpu', threads: 1, id: 'tts-model'}));
            else if (data.type === 'voice-prompt')
                queueMicrotask(() => this.emit({type: 'voice-prompt-ready', requestId: data.requestId,
                    voice_prompt: true, tokens: new Int32Array(data.tokens).length}));
            else if (data.type === 'synthesize' && data.options?.return_tokens)
                queueMicrotask(() => this.emit({type: 'audio', requestId: data.requestId, sample_rate: 24000,
                    chunks: 1, audio: new Float32Array([0]).buffer,
                    token_ids: Array.from({length: 8 * 75}, (_, index) => index % 1024),
                    voice_prompt_cached: true, voice_prompt_used: false}));
        }
        emit(data) { this.onmessage?.({data}); }
        audio(id, values = [0.123456789, -0.25, 0.5],
            details = {voice_prompt_cached: true, voice_prompt_used: true}) {
            this.emit({type: 'audio', requestId: id, sample_rate: 24000, chunks: 1,
                audio: Float32Array.from(values).buffer, ...details});
        }
        terminate() { this.terminated = true; }
    }
    const globals = {
        Worker,
        navigator: {hardwareConcurrency: 1},
        crossOriginIsolated: false,
        localStorage: {getItem: () => JSON.stringify({backend: 'cpu'}), setItem() {}},
        caches: {
            open: async name => ({
                match: async key => cacheEntries.get(`${name}:${key}`)?.clone(),
                put: async (key, value) => cacheEntries.set(`${name}:${key}`, value.clone()),
                keys: async () => [...cacheEntries.keys()].filter(key => key.startsWith(`${name}:`))
                    .map(key => ({url: key.slice(name.length + 1)})),
            }),
            delete: async name => {
                let deleted = false;
                for (const key of [...cacheEntries.keys()]) if (key.startsWith(`${name}:`)) {
                    cacheEntries.delete(key);
                    deleted = true;
                }
                return deleted;
            },
        },
        fetch: async () => { throw new Error('Tests must not download models'); },
        KIDI_APP_CONFIG: TEST_APP_CONFIG,
    };
    const previous = new Map(Object.keys(globals).map(key => [key, Object.getOwnPropertyDescriptor(globalThis, key)]));
    try {
        for (const [key, value] of Object.entries(globals))
            Object.defineProperty(globalThis, key, {configurable: true, value});
        const url = new URL('../../web/hosts/wasm/backend.js', import.meta.url);
        url.searchParams.set('audio-test', String(++nextModule));
        const {backend} = await import(url.href);
        await action({backend, workers, events, connect: callbacks =>
            backend.connect({onEvent: event => events.push(event), ...callbacks})});
    } finally {
        for (const worker of workers) worker.terminate();
        for (const [key, descriptor] of previous)
            if (descriptor) Object.defineProperty(globalThis, key, descriptor);
            else delete globalThis[key];
    }
}

test('browser PCM is Float32 and completion waits for transferable audio delivery', async () => {
    await withBackend(async ({backend, workers, events, connect}) => {
        const delivered = deferred(), frames = [];
        connect({onPcm: frame => { frames.push(frame); return delivered.promise; }});
        backend.send('tts.start', 'first', {stream_id: 1, text: 'Hello', options: {gpu_precision: 'w8a8'}});
        const worker = await waitFor(() => workers.find(worker => worker.url.includes('tts-worker')));
        await waitFor(() => visibleSynthesis(worker).length);
        assert.equal(worker.calls[0].matrixPrecision, 'w8a8');
        assert.equal(worker.calls[0].w8a8Kernel, 'float');
        worker.audio('first', [0.123456789, -0.25, 0.5], {
            voice_prompt_cached: true, voice_prompt_used: true,
            generation_ms: 120, decode_ms: 50, preparation_ms: 10, audio_tokens: 48,
        });
        await setImmediate();
        assert.equal(frames[0].samples[0], Math.fround(0.123456789));
        assert.equal(frames[0].sampleRate, 24000);
        assert.equal(events.some(event => event.type === 'tts.completed'), false);
        structuredClone(frames[0].samples, {transfer: [frames[0].samples.buffer]});
        delivered.resolve();
        await setImmediate();
        const completed = events.find(event => event.type === 'tts.completed').payload;
        assert.equal(completed.samples, 3);
        assert.deepEqual(completed.stats, {
            generation_ms: 120,
            decode_ms: 50,
            preparation_ms: 10,
            audio_tokens: 48,
        });
        assert.equal(worker.terminated, true);
    });
});

test('cancelling a model load cannot later start stale synthesis', async () => {
    await withBackend(async ({backend, workers, events, connect}) => {
        connect({onPcm() {}});
        backend.send('tts.start', 'old', {stream_id: 1, text: 'Old speech'});
        await setImmediate();
        const old = workers.find(worker => worker.url.includes('tts-worker'));
        backend.send('cancel', 'old');
        backend.send('tts.start', 'new', {stream_id: 2, text: 'New speech'});
        await setImmediate();
        const current = workers.at(-1);
        old.emit({type: 'ready', backend: 'wasm-cpu', threads: 1});
        current.emit({type: 'ready', backend: 'wasm-cpu', threads: 1, id: 'tts-model'});
        await waitFor(() => visibleSynthesis(current).length);
        assert.equal(old.calls.some(call => call.type === 'synthesize'), false);
        assert.equal(visibleSynthesis(current)[0].requestId, 'new');
        current.audio('new');
        await setImmediate();
        assert.equal(events.some(event => event.type === 'tts.completed' && event.id === 'old'), false);
    }, {ready: false});
});

test('late audio delivery and errors from a cancelled worker cannot close its replacement', async () => {
    await withBackend(async ({backend, workers, events, connect}) => {
        const delivered = deferred();
        connect({onPcm: frame => frame.streamId === 1 ? delivered.promise : undefined});
        backend.send('tts.start', 'old', {stream_id: 1, text: 'Old speech'});
        const old = await waitFor(() => workers.at(-1));
        await waitFor(() => visibleSynthesis(old).length);
        old.audio('old');
        await setImmediate();
        backend.send('cancel', 'old');
        backend.send('tts.start', 'new', {stream_id: 2, text: 'New speech'});
        const current = await waitFor(() => workers.length > 1 && workers.at(-1));
        await waitFor(() => visibleSynthesis(current).length);
        old.onerror({message: 'Late worker error'});
        delivered.resolve();
        await setImmediate();
        assert.notEqual(current.terminated, true);
        assert.equal(events.some(event => event.type === 'tts.completed' && event.id === 'old'), false);
        current.audio('new');
        await setImmediate();
        assert.equal(events.some(event => event.type === 'tts.completed' && event.id === 'new'), true);
    });
});

test('audio consumer failures are reported instead of a successful completion', async () => {
    await withBackend(async ({backend, workers, events, connect}) => {
        connect({onPcm: async () => { throw new Error('Speaker unavailable'); }});
        backend.send('tts.start', 'first', {stream_id: 1, text: 'Hello'});
        const worker = await waitFor(() => workers.at(-1));
        await waitFor(() => visibleSynthesis(worker).length);
        worker.audio('first');
        await setImmediate();
        assert.equal(events.find(event => event.type === 'error').payload.message, 'Speaker unavailable');
        assert.equal(events.some(event => event.type === 'tts.completed'), false);
    });
});

test('idle eviction remains available and worker failure clears ready state', async () => {
    await withBackend(async ({backend, workers, events, connect}) => {
        connect({});
        backend.send('runtime.load', 'load', {model: 'test/model', backend: 'cpu', threads: 1});
        await setImmediate();
        const worker = workers.find(worker => worker.url.includes('inference-worker'));
        assert.equal((await backend.status()).ready, true);
        worker.emit({type: 'residency', state: 'evicted'});
        assert.equal((await backend.status()).ready, true);
        assert.equal(events.filter(event => event.type === 'runtime.availability').at(-1).payload.resident, false);
        worker.onerror({message: 'GPU device lost'});
        assert.equal((await backend.status()).ready, false);
        assert.equal(events.filter(event => event.type === 'runtime.availability').at(-1).payload.available, false);
    });
});

test('chat system prompt follows pinned and code-switched language settings', async () => {
    await withBackend(async ({backend, workers, connect}) => {
        connect({});
        backend.send('runtime.load', 'load', {model: 'test/model', backend: 'cpu', threads: 1});
        await setImmediate();
        const worker = workers.find(value => value.url.includes('inference-worker'));
        backend.send('chat.start', 'pinned', {
            messages: [{role: 'user', content: 'Hello'}], languages: 'kn', language_names: 'Kannada',
        });
        assert.match(worker.calls.at(-1).messages[0].content, /speaks Kannada.*Kannada only/u);
        worker.emit({type: 'step', events: [{completed: {text: 'ok', stats: {}}}]});
        backend.send('chat.start', 'mixed', {
            messages: [{role: 'user', content: 'Hello'}], languages: 'en,kn',
            language_names: 'English, Kannada',
        });
        assert.match(worker.calls.at(-1).messages[0].content, /code-switch.*English, Kannada/u);
    });
});

test('voice mode and dictation both expose partial recognition after VAD-gated audio', async () => {
    await withBackend(async ({backend, workers, events, connect}) => {
        connect({});
        backend.send('asr.start', 'voice', {stream_id: 11, voice_mode: true, language: 'en,kn,te,hi'});
        await setImmediate();
        const worker = workers.find(value => value.url.includes('asr-worker'));
        for (let sequence = 0; sequence < 40; ++sequence)
            backend.sendBinary(asrFrame(11, sequence));
        const partial = worker.calls.find(call => call.type === 'transcribe');
        assert.match(partial.requestId, /^partial:voice:/);
        assert.equal(partial.language, 'en,kn,te,hi');
        worker.emit({type: 'result', requestId: partial.requestId, text: 'Hello', language: 'en'});
        assert.equal(events.at(-1).type, 'asr.partial');
        assert.equal(events.at(-1).payload.provisional, 'Hello');
        backend.send('asr.stop', 'voice');
        assert.equal(worker.calls.at(-1).requestId, 'final:voice');
        worker.emit({type: 'result', requestId: 'final:voice', text: 'Hello', language: 'en'});
        assert.equal(events.at(-1).type, 'asr.completed');

        backend.send('asr.start', 'dictation', {stream_id: 12, voice_mode: false});
        await setImmediate();
        for (let sequence = 0; sequence < 40; ++sequence)
            backend.sendBinary(asrFrame(12, sequence));
        assert.match(worker.calls.at(-1).requestId, /^partial:dictation:/);
    });

});

test('ASR rejects malformed recognition language preferences', async () => {
    await withBackend(async ({backend, workers, events, connect}) => {
        connect({});
        backend.send('asr.start', 'invalid-language', {stream_id: 14, language: 'English'});
        assert.equal(workers.some(worker => worker.url.includes('asr-worker')), false);
        const error = events.find(event => event.id === 'invalid-language' && event.type === 'error');
        assert.equal(error.payload.code, 'invalid_asr');
    });
});

test('speech models can be preloaded without recording or synthesis', async () => {
    await withBackend(async ({backend, workers, events, connect}) => {
        connect({});
        backend.send('asr.preload', 'load-asr');
        await waitFor(() => events.some(event => event.id === 'load-asr' && event.type === 'asr.ready'));
        const asr = workers.find(worker => worker.url.includes('asr-worker'));
        assert.ok(asr);
        assert.equal(asr.calls[0].type, 'load');
        assert.equal(asr.calls.some(call => call.type === 'transcribe'), false);
        const asrProgress = events.filter(event => event.type === 'asr.progress').length;
        asr.emit({type: 'residency', state: 'ready'});
        assert.equal(events.filter(event => event.type === 'asr.progress').length, asrProgress);

        backend.send('tts.preload', 'load-tts', {options: {gpu_precision: 'w8a8'}});
        await waitFor(() => events.some(event => event.id === 'load-tts' && event.type === 'tts.ready'));
        const tts = workers.find(worker => worker.url.includes('tts-worker'));
        assert.ok(tts);
        assert.equal(tts.calls[0].type, 'load');
        assert.equal(visibleSynthesis(tts).length, 0);

        const status = await backend.status();
        assert.equal(status.components.asr.loaded, true);
        assert.equal(status.components.tts.loaded, true);
    });
});

test('stopping empty ASR completes without asking Whisper to decode silence', async () => {
    await withBackend(async ({backend, workers, events, connect}) => {
        connect({});
        backend.send('asr.start', 'quiet', {stream_id: 13});
        await setImmediate();
        const worker = workers.find(value => value.url.includes('asr-worker'));
        backend.send('asr.stop', 'quiet');
        assert.equal(worker.calls.some(call => call.type === 'transcribe'), false);
        const completed = events.find(event => event.id === 'quiet' && event.type === 'asr.completed');
        assert.equal(completed.payload.text, '');
    });
});

test('a persisted voice prompt skips calibration after the worker is recreated', async () => {
    await withBackend(async ({backend, workers, connect}) => {
        connect({onPcm() {}});
        backend.send('tts.start', 'one', {
            stream_id: 1, text: 'First uncached sentence.', options: {steps: 4, gpu_precision: 'w8a8'},
        });
        const first = await waitFor(() => workers[0]);
        await waitFor(() => visibleSynthesis(first).length);
        const calibration = first.calls.filter(call => call.type === 'synthesize' && call.options?.return_tokens);
        assert.equal(calibration.length, 1);
        assert.equal(calibration[0].options.steps, 6);
        assert.equal(visibleSynthesis(first)[0].options.steps, 4);
        first.audio('one');
        await setImmediate();

        backend.send('tts.start', 'two', {
            stream_id: 2, text: 'A different uncached sentence.', options: {steps: 10, gpu_precision: 'w8a8'},
        });
        const second = await waitFor(() => workers.length === 2 && workers[1]);
        await waitFor(() => visibleSynthesis(second).length);
        assert.equal(second.calls.filter(call => call.type === 'voice-prompt').length, 1);
        assert.equal(second.calls.filter(call => call.type === 'synthesize' && call.options?.return_tokens).length, 0);
        assert.equal(visibleSynthesis(second)[0].options.steps, 10);
        second.audio('two');
        await setImmediate();
    });
});

test('legacy binary consumers retain the signed-16 PCM framing contract', async () => {
    await withBackend(async ({backend, workers, events, connect}) => {
        const frames = [];
        connect({onAudio: frame => frames.push(frame)});
        backend.send('tts.start', 'first', {stream_id: 7, text: 'Hello'});
        const worker = await waitFor(() => workers.at(-1));
        await waitFor(() => visibleSynthesis(worker).length);
        worker.audio('first');
        await setImmediate();
        const header = new DataView(frames[0]);
        assert.equal(header.getUint32(8, true), 7);
        assert.equal(header.getUint32(16, true), 24000);
        assert.equal(header.getUint32(20, true), 6);
        assert.equal(new Int16Array(frames[0], 24)[0], Math.round(Math.fround(0.123456789) * 32768));
        assert.equal(events.some(event => event.type === 'tts.completed'), true);
    });
});
