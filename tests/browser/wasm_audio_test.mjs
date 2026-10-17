import test from 'node:test';
import assert from 'node:assert/strict';
import {setImmediate} from 'node:timers/promises';
import {SpeechSegments} from '../../web/hosts/wasm/speech-segments.mjs';

let nextModule = 0;
function deferred() {
    let resolve;
    const promise = new Promise(yes => { resolve = yes; });
    return {promise, resolve};
}

async function withBackend(action, {ready = true} = {}) {
    const workers = [], events = [];
    class Worker {
        constructor(url) {
            this.url = String(url);
            this.calls = [];
            workers.push(this);
        }
        postMessage(data) {
            this.calls.push(data);
            if (data.type === 'load' && ready)
                queueMicrotask(() => this.emit({type: 'ready', backend: 'wasm-cpu', threads: 1}));
        }
        emit(data) { this.onmessage?.({data}); }
        audio(id, values = [0.123456789, -0.25, 0.5]) {
            this.emit({type: 'audio', requestId: id, sample_rate: 24000, chunks: 1,
                audio: Float32Array.from(values).buffer});
        }
        terminate() { this.terminated = true; }
    }
    const globals = {
        Worker,
        navigator: {hardwareConcurrency: 1},
        crossOriginIsolated: false,
        localStorage: {getItem: () => JSON.stringify({backend: 'cpu'}), setItem() {}},
        caches: {open: async () => ({match: async () => undefined, keys: async () => []})},
        fetch: async () => { throw new Error('Tests must not download models'); },
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
        await setImmediate();
        const worker = workers.find(worker => worker.url.includes('tts-worker'));
        assert.equal(worker.calls[0].matrixPrecision, 'w8a8');
        assert.equal(worker.calls[0].w8a8Kernel, 'float');
        worker.audio('first');
        await setImmediate();
        assert.equal(frames[0].samples[0], Math.fround(0.123456789));
        assert.equal(frames[0].sampleRate, 24000);
        assert.equal(events.some(event => event.type === 'tts.completed'), false);
        structuredClone(frames[0].samples, {transfer: [frames[0].samples.buffer]});
        delivered.resolve();
        await setImmediate();
        assert.equal(events.find(event => event.type === 'tts.completed').payload.samples, 3);
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
        current.emit({type: 'ready', backend: 'wasm-cpu', threads: 1});
        await setImmediate();
        assert.equal(old.calls.some(call => call.type === 'synthesize'), false);
        assert.equal(current.calls.find(call => call.type === 'synthesize').requestId, 'new');
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
        await setImmediate();
        const old = workers.at(-1);
        old.audio('old');
        await setImmediate();
        backend.send('cancel', 'old');
        backend.send('tts.start', 'new', {stream_id: 2, text: 'New speech'});
        await setImmediate();
        const current = workers.at(-1);
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
        await setImmediate();
        workers.at(-1).audio('first');
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

test('bounded segments deliver early, keep one worker warm, and respect audio backpressure', async () => {
    await withBackend(async ({backend, workers, events, connect}) => {
        const text = 'The small wooden bridge crosses the river near the old mill and follows a winding path through the trees while a bright bird sings softly in the morning.';
        const expected = new SpeechSegments().push(text, true);
        const delivery = deferred(), frames = [];
        connect({onPcm: frame => { frames.push(frame); return frame.sequence === 0 ? delivery.promise : undefined; }});
        backend.send('tts.start', 'segments', {stream_id: 4, text});
        await setImmediate();
        const worker = workers.at(-1);
        assert.equal(worker.calls.at(-1).text, expected[0].text);
        worker.audio('segments');
        await setImmediate();
        assert.equal(frames.length, 1);
        assert.equal(worker.calls.filter(call => call.type === 'synthesize').length, 1);
        assert.equal(worker.terminated, undefined);
        assert.equal(events.some(event => event.type === 'tts.completed'), false);
        delivery.resolve();
        await setImmediate();
        for (let index = 1; index < expected.length; ++index) {
            assert.equal(worker.calls.at(-1).text, expected[index].text);
            assert.equal(worker.calls.at(-1).streamSegment, true);
            worker.audio('segments');
            await setImmediate();
        }
        assert.deepEqual(frames.map(frame => frame.sequence), expected.map((_, index) => index));
        assert.equal(events.find(event => event.type === 'tts.completed').payload.chunks, expected.length);
        assert.equal(workers.length, 1);
        assert.equal(worker.terminated, true);
    });
});

test('speech overlaps chat generation and finishes only when the text stream ends', async () => {
    await withBackend(async ({backend, workers, events, connect}) => {
        const frames = [];
        connect({onPcm: frame => frames.push(frame)});
        backend.send('runtime.load', 'load', {model: 'test/model', backend: 'cpu', threads: 1});
        await setImmediate();
        const chat = workers.at(-1);
        backend.send('chat.start', 'chat', {messages: [{role: 'user', content: 'Say two sentences'}], speak: true});
        chat.emit({type: 'step', pending: 1, events: [{text: 'First sentence. '}]});
        await setImmediate();
        const speech = workers.find(worker => worker.url.includes('tts-worker'));
        assert.equal(speech.calls.at(-1).text, 'First sentence.');
        speech.audio('chat');
        await setImmediate();
        assert.equal(frames.length, 1);
        assert.equal(events.some(event => event.type === 'chat.completed'), false);
        assert.equal(events.some(event => event.type === 'voice.audio.completed'), false);
        chat.emit({type: 'step', pending: 1, events: [{text: 'Second sentence. '}]});
        await setImmediate();
        speech.audio('chat');
        await setImmediate();
        assert.equal(frames.length, 2);
        assert.equal(events.some(event => event.type === 'voice.audio.completed'), false);
        chat.emit({type: 'step', pending: 0, events: [{completed: {text: 'First sentence. Second sentence.'}}]});
        await setImmediate();
        assert.equal(events.find(event => event.type === 'voice.audio.completed').payload.chunks, 2);
        assert.ok(events.findIndex(event => event.type === 'tts.segment') <
            events.findIndex(event => event.type === 'chat.completed'));
    });
});

test('replay uses cached audio without creating or loading another worker', async () => {
    await withBackend(async ({backend, workers, events, connect}) => {
        const frames = [];
        connect({onPcm: frame => frames.push([...frame.samples])});
        backend.send('tts.start', 'one', {stream_id: 1, text: 'Hello'});
        await setImmediate();
        workers[0].audio('one');
        await setImmediate();
        backend.send('tts.start', 'two', {stream_id: 2, text: 'Hello'});
        await setImmediate();
        assert.equal(workers.length, 1);
        assert.deepEqual(frames[1], frames[0]);
        assert.equal(events.find(event => event.type === 'tts.segment' && event.id === 'two').payload.cached, true);
        assert.equal(events.some(event => event.type === 'tts.completed' && event.id === 'two'), true);
    });
});

test('changed synthesis settings do not reuse cached speech', async () => {
    await withBackend(async ({backend, workers, events, connect}) => {
        connect({onPcm() {}});
        backend.send('tts.start', 'one', {stream_id: 1, text: 'Hello', options: {steps: 10}});
        await setImmediate();
        workers.at(-1).audio('one');
        await setImmediate();
        backend.send('tts.start', 'two', {stream_id: 2, text: 'Hello', options: {steps: 16}});
        await setImmediate();
        assert.equal(workers.length, 2);
        workers.at(-1).audio('two');
        await setImmediate();
        assert.equal(events.find(event => event.type === 'tts.segment' && event.id === 'two').payload.cached, false);
    });
});

test('speech-only cancellation leaves ongoing chat generation alive', async () => {
    await withBackend(async ({backend, workers, events, connect}) => {
        connect({onPcm() {}});
        backend.send('runtime.load', 'load', {model: 'test/model', backend: 'cpu', threads: 1});
        await setImmediate();
        const chat = workers.at(-1);
        backend.send('chat.start', 'chat', {messages: [], speak: true});
        chat.emit({type: 'step', pending: 1, events: [{text: 'First sentence. '}]});
        await setImmediate();
        const speech = workers.find(worker => worker.url.includes('tts-worker'));
        backend.send('cancel', 'chat', {scope: 'speech', message: 'Speaker failed'});
        assert.equal(speech.terminated, true);
        assert.equal(chat.calls.some(call => call.type === 'cancel'), false);
        chat.emit({type: 'step', pending: 0, events: [{text: 'More text.', completed: {text: 'First sentence. More text.'}}]});
        await setImmediate();
        assert.equal(events.find(event => event.type === 'voice.audio.error').payload.message, 'Speaker failed');
        assert.equal(events.find(event => event.type === 'chat.completed').payload.text, 'First sentence. More text.');
        assert.equal(events.some(event => event.type === 'request.cancelled'), false);
    });
});

test('legacy binary consumers retain the signed-16 PCM framing contract', async () => {
    await withBackend(async ({backend, workers, events, connect}) => {
        const frames = [];
        connect({onAudio: frame => frames.push(frame)});
        backend.send('tts.start', 'first', {stream_id: 7, text: 'Hello'});
        await setImmediate();
        workers.at(-1).audio('first');
        await setImmediate();
        const header = new DataView(frames[0]);
        assert.equal(header.getUint32(8, true), 7);
        assert.equal(header.getUint32(16, true), 24000);
        assert.equal(header.getUint32(20, true), 6);
        assert.equal(new Int16Array(frames[0], 24)[0], Math.round(Math.fround(0.123456789) * 32768));
        assert.equal(events.some(event => event.type === 'tts.completed'), true);
    });
});
