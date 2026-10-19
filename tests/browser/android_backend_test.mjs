import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {setImmediate} from 'node:timers/promises';

class Transport {
    binaryListeners = new Set();
    listeners = new Set();
    open = true;
    sent = [];
    sentBinary = [];
    bufferedAmount = 0;

    constructor() {
        this.closed = new Promise(resolve => { this.finishClosed = resolve; });
    }

    post(value) { this.sent.push(value); }
    postBinary(value) { this.sentBinary.push(new Uint8Array(value)); }
    subscribe(listener) { this.listeners.add(listener); return () => this.listeners.delete(listener); }
    subscribeBinary(listener) {
        this.binaryListeners.add(listener);
        return () => this.binaryListeners.delete(listener);
    }
    receive(value) { for (const listener of this.listeners) listener(value); }
    receiveBinary(value) { for (const listener of this.binaryListeners) listener(value); }
    async waitForDrain() {}
    close() { this.open = false; this.finishClosed(); }
}

const completed = (request, payload = {}) => ({
    version: 1, kind: 'event', id: request.id, sequence: 0, type: 'completed', payload,
});

test('Android adapter uses LiteApp requests, native binary streams, and browser EPUB parsing', async () => {
    const transport = new Transport();
    globalThis.__kidiAndroidTestTransport = transport;
    try {
        const imageSource = 'export async function readImage() {' +
            "return new Blob([new Uint8Array([1,2,3])], {type:'image/png'});}";
        const epubSource = 'export async function openBrowserEpub() {' +
            "return {metadata:{book_id:'book',title:'Book',author:'Writer',language:'en',chapters:[{id:'one',title:'One'}]}," +
            "chapter:async index=>({book_id:'book',chapter:index,title:'One',html:'<p>Hello</p>'})};}";
        const transportSource =
            'export async function connectAndroid(){return globalThis.__kidiAndroidTestTransport;}';
        const module = value => `data:text/javascript;base64,${Buffer.from(value).toString('base64')}`;
        const source = (await readFile(new URL('../../web/hosts/android/backend.js', import.meta.url), 'utf8'))
            .replace("from './images.js'", `from '${module(imageSource)}'`)
            .replace("from './browser-epub.js'", `from '${module(epubSource)}'`)
            .replace("from './channel.mjs'",
                `from '${new URL('../../third_party/liteapp/web/transport/channel.mjs', import.meta.url).href}'`)
            .replace("from './event-client.mjs'",
                `from '${new URL('../../third_party/liteapp/web/host/event-client.mjs', import.meta.url).href}'`)
            .replace("from './android-transport.mjs'", `from '${module(transportSource)}'`);
        const {backend} = await import(`data:text/javascript,${encodeURIComponent(source)}#${Date.now()}`);

        const statusPromise = backend.status();
        await setImmediate();
        const statusRequest = transport.sent.at(-1);
        assert.equal(statusRequest.method, 'session.status');
        transport.receive(completed(statusRequest, {ready: false, runtime: {host: 'android'}}));
        assert.equal((await statusPromise).runtime.host, 'android');

        const events = [], audio = [];
        backend.connect({onEvent: event => events.push(event), onAudio: value => audio.push(value)});
        await setImmediate();
        assert.equal(backend.nativeAudioCapture, true);
        assert.equal(backend.nativeTtsPlayback, true);
        const permission = backend.requestMicrophonePermission();
        await setImmediate();
        const permissionRequest = transport.sent.at(-1);
        assert.equal(permissionRequest.method, 'permissions.microphone.request');
        transport.receive(completed(permissionRequest, {granted: true}));
        assert.deepEqual(await permission, {granted: true});

        backend.send('runtime.load', 'load', {model: 'owner/model'});
        const load = transport.sent.at(-1);
        assert.equal(load.method, 'runtime.load');
        transport.receive({
            version: 1, kind: 'event', id: 'load', sequence: 0,
            type: 'runtime.progress', payload: {file: 'model.safetensors'},
        });
        transport.receive({...completed(load, {backend: 'android-cpu'}), sequence: 1});
        await setImmediate();
        assert.deepEqual(events.map(event => event.type), ['runtime.progress', 'runtime.ready']);

        const prepared = backend.prepareMessages([
            {role: 'user', content: 'Describe this', images: [{name: 'photo.png'}]},
        ]);
        await setImmediate();
        const start = transport.sent.at(-1);
        assert.equal(start.method, 'image.start');
        transport.receive(completed(start, {stream_id: start.payload.stream_id}));
        await setImmediate();
        assert.equal(transport.sentBinary.length, 1);
        assert.equal(transport.sentBinary[0][5], 3);
        const finish = transport.sent.at(-1);
        assert.equal(finish.method, 'image.finish');
        transport.receive(completed(finish, {stream_id: start.payload.stream_id}));
        assert.deepEqual(await prepared, [{
            role: 'user', content: 'Describe this', images: [start.payload.stream_id],
        }]);

        const book = await backend.importEbook(new Blob([new Uint8Array([1])]));
        assert.equal(book.book_id, 'book');
        assert.equal((await backend.loadEbookChapter('book', 0)).html, '<p>Hello</p>');
        const frame = new Uint8Array([1, 2, 3]).buffer;
        transport.receiveBinary(frame);
        assert.equal(audio[0], frame);
    } finally {
        delete globalThis.__kidiAndroidTestTransport;
    }
});
