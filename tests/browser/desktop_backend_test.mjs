import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';

test('desktop adapter isolates primitive speech on its own WebSocket', async () => {
    const previous = new Map(['WebSocket', 'location'].map(key =>
        [key, Object.getOwnPropertyDescriptor(globalThis, key)]));
    const sockets = [];
    class WebSocket {
        static CONNECTING = 0;
        static OPEN = 1;
        static CLOSING = 2;
        static CLOSED = 3;
        readyState = WebSocket.CONNECTING;
        listeners = new Map();
        sent = [];
        constructor(url) { this.url = url; sockets.push(this); }
        addEventListener(type, callback) { this.listeners.set(type, callback); }
        send(message) { this.sent.push(message); }
        open() { this.readyState = WebSocket.OPEN; this.listeners.get('open')?.(); }
        message(data) { this.listeners.get('message')?.({data}); }
        close() { this.readyState = WebSocket.CLOSED; this.listeners.get('close')?.(); }
    }
    try {
        Object.defineProperty(globalThis, 'WebSocket', {configurable: true, value: WebSocket});
        Object.defineProperty(globalThis, 'location', {configurable: true,
            value: {protocol: 'http:', host: '127.0.0.1:10010'}});
        const source = (await readFile(new URL('../../web/hosts/desktop/backend.js', import.meta.url), 'utf8'))
            .replace("from './images.js'", `from '${new URL('../../web/ui/images.js', import.meta.url).href}'`);
        const {backend} = await import(`data:text/javascript,${encodeURIComponent(source)}#${Date.now()}`);
        const events = [], audio = [];
        backend.connect({onOpen() {}, onEvent: event => events.push(event), onAudio: frame => audio.push(frame)});
        assert.equal(sockets.length, 2);
        const [main, speech] = sockets;
        main.open();
        speech.open();

        backend.send('chat.start', 'chat', {messages: []});
        backend.send('tts.start', 'speech', {stream_id: 9, text: 'Hello'});
        backend.send('cancel', 'speech', {scope: 'speech'});
        assert.deepEqual(main.sent.map(message => JSON.parse(message).type), ['chat.start']);
        assert.deepEqual(speech.sent.map(message => JSON.parse(message).type), ['tts.start', 'cancel']);

        speech.message(JSON.stringify({v: 1, id: '', type: 'session.ready', payload: {}}));
        assert.equal(events.length, 0);
        const frame = new ArrayBuffer(24);
        speech.message(frame);
        assert.equal(audio[0], frame);
        speech.message(JSON.stringify({v: 1, id: 'speech', type: 'request.cancelled', payload: {}}));
        assert.equal(events.at(-1).type, 'request.cancelled');
        speech.close();
        assert.equal(main.readyState, WebSocket.CLOSED);
    } finally {
        for (const [key, descriptor] of previous)
            if (descriptor) Object.defineProperty(globalThis, key, descriptor);
            else delete globalThis[key];
    }
});
