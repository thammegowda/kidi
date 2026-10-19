import {readImage} from './images.js';
import {openBrowserEpub} from './browser-epub.js';
import {RequestChannel} from './channel.mjs';
import {EventClient} from './event-client.mjs';
import {connectAndroid} from './android-transport.mjs';

const COMPLETION_EVENTS = new Map([
  ['runtime.restore', 'runtime.ready'],
  ['runtime.load', 'runtime.ready'],
  ['asr.preload', 'asr.ready'],
  ['tts.preload', 'tts.ready'],
  ['settings.system_prompt.set', 'settings.system_prompt.updated'],
  ['reader.translate', 'reader.translation.completed'],
  ['chat.start', 'chat.completed'],
  ['asr.start', 'asr.completed'],
  ['tts.start', 'tts.completed'],
]);

class AndroidBackend {
  #callbacks;
  #channel;
  #client;
  #connection;
  #ebooks = new Map();
  #nextImageStream = 1;
  #transport;

  get bufferedAmount() {
    return this.#transport?.bufferedAmount || 0;
  }

  get nativeAudioCapture() {
    return true;
  }

  get nativeTtsPlayback() {
    return true;
  }

  async requestMicrophonePermission() {
    await this.#ensureConnection();
    return this.#channel.request('permissions.microphone.request');
  }

  legalDocuments() {
    return {
      privacy: './legal/PRIVACY.md',
      terms: './legal/TERMS.md',
    };
  }

  async #ensureConnection() {
    if (!this.#connection) {
      this.#connection = connectAndroid().then(transport => {
        this.#transport = transport;
        this.#channel = new RequestChannel(transport);
        return transport;
      });
    }
    await this.#connection;
  }

  async status() {
    await this.#ensureConnection();
    return this.#channel.request('session.status');
  }

  connect(callbacks) {
    this.#callbacks = callbacks;
    this.#ensureConnection().then(() => {
      const emit = event => callbacks.onEvent?.({
        v: 1, id: event.id, type: event.type, seq: event.sequence, payload: event.payload,
      });
      this.#client = new EventClient(this.#channel, {
        completionEvents: COMPLETION_EVENTS,
        onEvent: emit,
      });
      this.#transport.subscribeBinary(data => callbacks.onAudio?.(data));
      this.#transport.closed.then(() => {
        this.#client = null;
        this.#channel = null;
        this.#transport = null;
        this.#connection = null;
        callbacks.onClose?.();
      });
      callbacks.onOpen?.();
      this.#client.request('runtime.restore', 'android-runtime-restore', {});
    }).catch(error => {
      callbacks.onError?.(error);
      callbacks.onClose?.();
    });
  }

  send(type, id, payload = {}) {
    if (type === 'cancel') {
      if (!this.#client?.cancel(id))
        this.#callbacks?.onEvent?.({v: 1, id, type: 'request.cancelled', seq: 0, payload});
      return;
    }
    if (type === 'asr.stop') {
      if (!this.#channel) throw new Error('The Android host is reconnecting');
      this.#channel.request('asr.stop', {request_id: id}).catch(error => {
        if (error.name !== 'AbortError')
          this.#callbacks?.onEvent?.({v: 1, id, type: 'error', seq: 0,
            payload: {code: error.code || 'service_error', message: error.message}});
      });
      return;
    }
    if (!this.#client || !this.#transport?.open)
      throw new Error('The Android host is reconnecting');
    this.#client.request(type, id, payload);
  }

  sendBinary(frame) {
    if (!this.#transport?.open) throw new Error('The Android host is reconnecting');
    this.#transport.postBinary(frame);
  }

  async importEbook(blob) {
    const provider = await openBrowserEpub(blob);
    this.#ebooks.set(provider.metadata.book_id, provider);
    return provider.metadata;
  }

  async loadEbookChapter(bookId, chapter) {
    const provider = this.#ebooks.get(bookId);
    if (!provider) throw new Error('Re-import this EPUB to continue reading it');
    return provider.chapter(chapter);
  }

  async prepareMessages(messages) {
    const prepared = [];
    for (const message of messages) {
      const images = [];
      for (const image of message.images || []) {
        const blob = await readImage(image);
        const streamId = this.#nextImageStream++;
        if (!this.#channel || !this.#transport?.open)
          throw new Error('The Android host is reconnecting');
        await this.#channel.request('image.start', {
          stream_id: streamId,
          size: blob.size,
          content_type: blob.type,
          name: image.name,
        });
        const bytes = new Uint8Array(await blob.arrayBuffer());
        let sequence = 0;
        for (let offset = 0; offset < bytes.length; offset += 64 * 1024) {
          const payload = bytes.subarray(offset, Math.min(bytes.length, offset + 64 * 1024));
          const frame = new Uint8Array(24 + payload.length);
          frame.set([0x4b, 0x44, 0x41, 0x31, 1, 3], 0);
          const header = new DataView(frame.buffer);
          header.setUint32(8, streamId, true);
          header.setUint32(12, sequence++, true);
          header.setUint32(16, 0, true);
          header.setUint32(20, payload.length, true);
          frame.set(payload, 24);
          this.#transport.postBinary(frame);
          await this.#transport.waitForDrain();
        }
        await this.#channel.request('image.finish', {stream_id: streamId});
        images.push(streamId);
      }
      prepared.push({...message, images});
    }
    return prepared;
  }
}

export const backend = new AndroidBackend();
