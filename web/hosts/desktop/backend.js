import {readImage} from './images.js';
import {RequestChannel} from './channel.mjs';
import {EventClient} from './event-client.mjs';
import {connectWebSocket} from './websocket-transport.mjs';

const COMPLETION_EVENTS = new Map([
  ['settings.system_prompt.set', 'settings.system_prompt.updated'],
  ['chat.start', 'chat.completed'],
  ['asr.start', 'asr.completed'],
  ['tts.start', 'tts.completed'],
]);

class DesktopBackend {
  #callbacks;
  #connection = 0;
  #disconnecting = false;
  #client;
  #nextEbookStream = 0x40000000;
  #nextImageStream = 1;
  #transport;
  #channel;
  #speechClient;
  #speechTransport;
  #speechChannel;

  get bufferedAmount() {
    return Math.max(this.#transport?.bufferedAmount || 0, this.#speechTransport?.bufferedAmount || 0);
  }

  async status() {
    const response = await fetch('/api/status', {headers: {'Content-Type': 'application/json'}});
    const body = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(body.error?.message || `Request failed (${response.status})`);
    return body;
  }

  connect(callbacks) {
    this.#callbacks = callbacks;
    const connection = ++this.#connection;
    this.#disconnecting = false;
    this.#client?.dispose('The local session was replaced');
    this.#speechClient?.dispose('The native speech channel was replaced');
    this.#client = null;
    this.#speechClient = null;
    this.#channel = null;
    this.#speechChannel = null;
    this.#transport = null;
    this.#speechTransport = null;
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const address = `${protocol}//${location.host}/ws`;
    Promise.all([connectWebSocket(address), connectWebSocket(address)]).then(([transport, speechTransport]) => {
      if (connection !== this.#connection) {
        transport.close();
        speechTransport.close();
        return;
      }
      const channel = new RequestChannel(transport);
      const speechChannel = new RequestChannel(speechTransport);
      const emit = event => callbacks.onEvent?.({
        v: 1, id: event.id, type: event.type, seq: event.sequence, payload: event.payload,
      });
      const mapError = speech => error => ({
        code: this.#disconnecting && speech ? 'speech_disconnected'
          : this.#disconnecting ? 'disconnected'
            : error.code || 'service_error',
        message: error.message || 'Local service request failed',
        ...(error.details === undefined ? {} : {details: error.details}),
      });
      const client = new EventClient(channel, {
        completionEvents: COMPLETION_EVENTS,
        mapError: mapError(false),
        onEvent: emit,
      });
      const speechClient = new EventClient(speechChannel, {
        completionEvents: COMPLETION_EVENTS,
        mapError: mapError(true),
        onEvent: emit,
      });
      this.#transport = transport;
      this.#channel = channel;
      this.#client = client;
      this.#speechTransport = speechTransport;
      this.#speechChannel = speechChannel;
      this.#speechClient = speechClient;
      transport.subscribeBinary(data => callbacks.onAudio?.(data));
      speechTransport.subscribeBinary(data => callbacks.onAudio?.(data));
      let closed = false;
      const disconnected = speech => {
        if (closed || connection !== this.#connection) return;
        closed = true;
        this.#disconnecting = true;
        this.#transport = null;
        this.#speechTransport = null;
        client.dispose(speech
          ? 'The native speech channel disconnected'
          : 'The local session disconnected');
        speechClient.dispose(speech
          ? 'The native speech channel disconnected'
          : 'The local session disconnected');
        this.#client = null;
        this.#speechClient = null;
        this.#channel = null;
        this.#speechChannel = null;
        callbacks.onClose?.();
      };
      transport.closed.then(() => disconnected(false));
      speechTransport.closed.then(() => disconnected(true));
      callbacks.onOpen?.();
    }).catch(error => {
      if (connection !== this.#connection) return;
      callbacks.onError?.(error);
      callbacks.onClose?.();
    });
  }

  send(type, id, payload = {}) {
    if (type === 'cancel') {
      if (!this.#client?.cancel(id) && !this.#speechClient?.cancel(id))
        this.#callbacks?.onEvent?.({v: 1, id, type: 'request.cancelled', seq: 0, payload});
      return;
    }
    if (type === 'asr.stop') {
      if (!this.#channel) throw new Error('The local session is reconnecting');
      this.#channel.request('asr.stop', {request_id: id}).catch(error => {
        if (error.name !== 'AbortError')
          this.#callbacks?.onEvent?.({v: 1, id, type: 'error', seq: 0,
            payload: {code: error.code || 'service_error', message: error.message}});
      });
      return;
    }
    const speech = type === 'tts.start';
    const client = speech ? this.#speechClient : this.#client;
    if (!client || !(speech ? this.#speechTransport : this.#transport)?.open)
      throw new Error(speech
        ? 'The native speech channel is reconnecting'
        : 'The local session is reconnecting');
    client.request(type, id, payload);
  }

  sendBinary(frame) {
    if (!this.#transport?.open)
      throw new Error('The local session is reconnecting');
    this.#transport.postBinary(frame);
  }

  async importEbook(blob, name = 'Book.epub') {
    if (!(blob instanceof Blob) || !blob.size || blob.size > 64 * 1024 * 1024)
      throw new Error('Choose an EPUB no larger than 64 MiB');
    if (!this.#channel || !this.#transport?.open)
      throw new Error('The local session is reconnecting');
    const streamId = this.#nextEbookStream++;
    await this.#channel.request('ebook.start', {
      stream_id: streamId,
      size: blob.size,
      name: String(name).slice(0, 256),
    });
    let sequence = 0;
    for (let offset = 0; offset < blob.size; offset += 64 * 1024) {
      const payload = new Uint8Array(await blob.slice(offset, offset + 64 * 1024).arrayBuffer());
      const frame = new Uint8Array(24 + payload.length);
      frame.set([0x4b, 0x44, 0x41, 0x31, 1, 4], 0);
      const header = new DataView(frame.buffer);
      header.setUint32(8, streamId, true);
      header.setUint32(12, sequence++, true);
      header.setUint32(16, 0, true);
      header.setUint32(20, payload.length, true);
      frame.set(payload, 24);
      this.#transport.postBinary(frame);
      await this.#transport.waitForDrain();
    }
    return this.#channel.request('ebook.finish', {stream_id: streamId});
  }

  async loadEbookChapter(bookId, chapter) {
    if (!this.#channel || !this.#transport?.open)
      throw new Error('The local session is reconnecting');
    return this.#channel.request('ebook.chapter', {book_id: bookId, chapter});
  }

  async prepareMessages(messages) {
    const prepared = [];
    for (const message of messages) {
      const images = [];
      for (const image of message.images || []) {
        const blob = await readImage(image);
        const streamId = this.#nextImageStream++;
        if (!this.#channel || !this.#transport?.open)
          throw new Error('The local session is reconnecting');
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
      prepared.push(images.length
        ? {role: message.role, content: message.content, images}
        : {role: message.role, content: message.content});
    }
    return prepared;
  }

}

export const backend = new DesktopBackend();
