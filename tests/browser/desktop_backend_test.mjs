import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {setImmediate} from 'node:timers/promises';

class Socket extends EventTarget {
    static instances = [];
    static CONNECTING = 0;
    static OPEN = 1;
    static CLOSING = 2;
    static CLOSED = 3;
    readyState = 0;
    bufferedAmount = 0;
    sent = [];
    constructor(url) {
        super();
        this.url = url;
        Socket.instances.push(this);
    }
    send(value) {
        if (this.readyState !== 1) throw new Error('socket is not open');
        this.sent.push(value);
    }
    open() {
        this.readyState = 1;
        this.dispatchEvent(new Event('open'));
    }
    receive(data) { this.dispatchEvent(new MessageEvent('message', {data})); }
    close() {
        if (this.readyState === 3) return;
        this.readyState = 3;
        this.dispatchEvent(new Event('close'));
    }
}

const message = (socket, index = -1) => JSON.parse(socket.sent.at(index));
const event = (id, sequence, type, payload) => JSON.stringify({
    version: 1, kind: 'event', id, sequence, type, payload,
});

test('desktop adapter uses LiteApp requests, binary uploads, and isolated speech', async () => {
    const previous = new Map(['WebSocket', 'location', 'crypto'].map(key =>
        [key, Object.getOwnPropertyDescriptor(globalThis, key)]));
    try {
        Object.defineProperty(globalThis, 'WebSocket', {configurable: true, value: Socket});
        Object.defineProperty(globalThis, 'location', {configurable: true,
            value: {protocol: 'http:', host: '127.0.0.1:10010'}});
        let uuid = 0;
        Object.defineProperty(globalThis, 'crypto', {configurable: true,
            value: {randomUUID: () => `generated-${++uuid}`}});
        const imageSource = 'export async function readImage() {' +
            "return new Blob([new Uint8Array([1,2,3])], {type:'image/png'});}";
        const imageModule = `data:text/javascript;base64,${Buffer.from(imageSource).toString('base64')}`;
        const source = (await readFile(new URL('../../web/hosts/desktop/backend.js', import.meta.url), 'utf8'))
            .replace("from './images.js'", `from '${imageModule}'`)
            .replace("from './channel.mjs'",
                `from '${new URL('../../third_party/liteapp/web/transport/channel.mjs', import.meta.url).href}'`)
            .replace("from './event-client.mjs'",
                `from '${new URL('../../third_party/liteapp/web/host/event-client.mjs', import.meta.url).href}'`)
            .replace("from './websocket-transport.mjs'",
                `from '${new URL('../../third_party/liteapp/web/transport/websocket-transport.mjs', import.meta.url).href}'`);
        const {backend} = await import(`data:text/javascript,${encodeURIComponent(source)}#${Date.now()}`);
        const events = [], audio = [];
        backend.connect({onEvent: value => events.push(value), onAudio: value => audio.push(value)});
        assert.equal(Socket.instances.length, 2);
        const [main, speech] = Socket.instances;
        main.open();
        speech.open();
        await setImmediate();

        backend.send('chat.start', 'chat', {messages: []});
        backend.send('tts.start', 'speech', {stream_id: 9, text: 'Hello'});
        assert.deepEqual(message(main), {
            version: 1, kind: 'request', id: 'chat', method: 'chat.start', payload: {messages: []},
        });
        assert.equal(message(speech).method, 'tts.start');
        main.receive(event('chat', 0, 'chat.delta', {text: 'Hi'}));
        main.receive(event('chat', 1, 'completed', {text: 'Hi', stats: {}}));
        await setImmediate();
        assert.deepEqual(events.slice(0, 2).map(value => value.type), ['chat.delta', 'chat.completed']);

        backend.send('reader.translate', 'translate', {
            text: 'A house.', previous: '', next: 'A garden.',
            target_language: 'kn', target_language_name: 'Kannada',
        });
        assert.equal(message(main).method, 'reader.translate');
        main.receive(event('translate', 0, 'reader.translation.delta', {text: 'ಮನೆ'}));
        main.receive(event('translate', 1, 'completed', {text: 'ಒಂದು ಮನೆ.', stats: {}}));
        await setImmediate();
        assert.deepEqual(events.slice(2, 4).map(value => value.type),
            ['reader.translation.delta', 'reader.translation.completed']);
        assert.equal(events[3].payload.text, 'ಒಂದು ಮನೆ.');

        backend.send('settings.system_prompt.set', 'settings', {prompt: 7});
        main.receive(event('settings', 0, 'error',
            {code: 'invalid_system_prompt', message: 'Prompt must be text'}));
        await setImmediate();
        assert.equal(events.at(-1).payload.code, 'invalid_system_prompt');
        backend.send('asr.stop', 'asr');
        const stopped = message(main);
        assert.equal(stopped.method, 'asr.stop');
        assert.equal(stopped.payload.request_id, 'asr');
        main.receive(event(stopped.id, 0, 'completed', {stopped: true}));

        speech.receive(event('speech', 0, 'tts.started', {stream_id: 9}));
        backend.send('cancel', 'speech', {scope: 'speech'});
        assert.deepEqual(message(speech), {version: 1, kind: 'cancel', id: 'speech'});
        await setImmediate();
        assert.equal(events.at(-1).type, 'request.cancelled');
        const audioFrame = new ArrayBuffer(24);
        main.receive(audioFrame);
        assert.equal(audio[0], audioFrame);

        const prepared = backend.prepareMessages([
            {role: 'user', content: 'describe', images: [{name: 'photo.png'}]},
        ]);
        await setImmediate();
        const started = message(main);
        assert.equal(started.method, 'image.start');
        main.receive(event(started.id, 0, 'completed', {stream_id: started.payload.stream_id}));
        await setImmediate();
        assert.ok(main.sent.at(-2) instanceof Uint8Array);
        const finished = message(main);
        assert.equal(finished.method, 'image.finish');
        main.receive(event(finished.id, 0, 'completed', {stream_id: started.payload.stream_id}));
        assert.deepEqual(await prepared, [
            {role: 'user', content: 'describe', images: [started.payload.stream_id]},
        ]);

        const importing = backend.importEbook(new Blob([new Uint8Array([4, 5, 6])]), 'sample.epub');
        await setImmediate();
        const ebookStarted = message(main);
        assert.equal(ebookStarted.method, 'ebook.start');
        main.receive(event(ebookStarted.id, 0, 'completed', {stream_id: ebookStarted.payload.stream_id}));
        await setImmediate();
        const ebookFrame = main.sent.at(-2);
        assert.ok(ebookFrame instanceof Uint8Array);
        assert.equal(ebookFrame[5], 4);
        const ebookFinished = message(main);
        assert.equal(ebookFinished.method, 'ebook.finish');
        main.receive(event(ebookFinished.id, 0, 'completed', {
            book_id: 'native-book',
            title: 'Sample',
            author: 'Writer',
            language: 'en',
            chapters: [{id: 'one', title: 'One'}],
        }));
        assert.equal((await importing).book_id, 'native-book');

        const chapter = backend.loadEbookChapter('native-book', 0);
        await setImmediate();
        const chapterRequest = message(main);
        assert.equal(chapterRequest.method, 'ebook.chapter');
        main.receive(event(chapterRequest.id, 0, 'completed', {
            book_id: 'native-book', chapter: 0, title: 'One', html: '<p>Hello</p>',
        }));
        assert.equal((await chapter).html, '<p>Hello</p>');

        speech.close();
        await setImmediate();
        assert.equal(main.readyState, Socket.CLOSED);
    } finally {
        Socket.instances.length = 0;
        for (const [key, descriptor] of previous)
            if (descriptor) Object.defineProperty(globalThis, key, descriptor);
            else delete globalThis[key];
    }
});
