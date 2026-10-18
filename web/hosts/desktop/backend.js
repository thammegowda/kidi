import {readImage} from './images.js';

class DesktopBackend {
  #callbacks;
  #nextImageStream = 1;
  #uploads = new Map();
  #socket;
  #speechSocket;
  #speechQueue = [];
  #speechRequests = new Set();

  get bufferedAmount() {
    return Math.max(this.#socket?.bufferedAmount || 0, this.#speechSocket?.bufferedAmount || 0);
  }

  async status() {
    const response = await fetch('/api/status', {headers: {'Content-Type': 'application/json'}});
    const body = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(body.error?.message || `Request failed (${response.status})`);
    return body;
  }

  connect(callbacks) {
    this.#callbacks = callbacks;
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const address = `${protocol}//${location.host}/ws`;
    const socket = new WebSocket(address);
    const speech = new WebSocket(address);
    this.#socket = socket;
    this.#speechSocket = speech;
    socket.binaryType = 'arraybuffer';
    speech.binaryType = 'arraybuffer';
    socket.addEventListener('open', () => callbacks.onOpen?.());
    speech.addEventListener('open', () => {
      if (this.#speechSocket !== speech) return;
      for (const message of this.#speechQueue.splice(0)) speech.send(message);
    });
    const receive = (event, speechChannel) => {
      if (typeof event.data === 'string') {
        try {
          const message = JSON.parse(event.data);
          if (speechChannel && message.type === 'session.ready') return;
          const upload = this.#uploads.get(message.id);
          if (upload && (message.type === 'image.completed' || message.type === 'error')) {
            this.#uploads.delete(message.id);
            if (message.type === 'image.completed') upload.resolve();
            else upload.reject(new Error(message.payload?.message || 'Image upload failed'));
            return;
          }
          if (speechChannel && ['tts.completed', 'request.cancelled', 'error'].includes(message.type))
            this.#speechRequests.delete(message.id);
          callbacks.onEvent?.(message);
        }
        catch { callbacks.onError?.(new Error('Received an invalid local protocol message')); }
      } else {
        callbacks.onAudio?.(event.data);
      }
    };
    socket.addEventListener('message', event => receive(event, false));
    speech.addEventListener('message', event => receive(event, true));
    socket.addEventListener('close', () => {
      if (this.#socket === socket) this.#socket = null;
      if (this.#speechSocket === speech && speech.readyState < WebSocket.CLOSING) speech.close();
      const error = new Error('The local session disconnected during image upload');
      for (const upload of this.#uploads.values()) upload.reject(error);
      this.#uploads.clear();
      callbacks.onClose?.();
    });
    speech.addEventListener('close', () => {
      if (this.#speechSocket !== speech) return;
      this.#speechSocket = null;
      this.#speechQueue.length = 0;
      for (const id of this.#speechRequests)
        callbacks.onEvent?.({v: 1, id, type: 'error', seq: 0,
          payload: {code: 'speech_disconnected', message: 'The native speech channel disconnected'}});
      this.#speechRequests.clear();
      if (this.#socket?.readyState < WebSocket.CLOSING) this.#socket.close();
    });
    socket.addEventListener('error', () => socket.close());
    speech.addEventListener('error', () => speech.close());
  }

  send(type, id, payload = {}) {
    const message = JSON.stringify({v: 1, id, type, payload});
    const speechRequest = type === 'tts.start' ||
      (type === 'cancel' && payload.scope === 'speech' && this.#speechRequests.has(id));
    if (speechRequest) {
      if (!this.#speechSocket || this.#speechSocket.readyState === WebSocket.CLOSING ||
          this.#speechSocket.readyState === WebSocket.CLOSED)
        throw new Error('The native speech channel is reconnecting');
      if (type === 'tts.start') this.#speechRequests.add(id);
      if (this.#speechSocket.readyState === WebSocket.OPEN) this.#speechSocket.send(message);
      else this.#speechQueue.push(message);
      return;
    }
    if (!this.#socket || this.#socket.readyState !== WebSocket.OPEN)
      throw new Error('The local session is reconnecting');
    this.#socket.send(message);
  }

  sendBinary(frame) {
    if (!this.#socket || this.#socket.readyState !== WebSocket.OPEN)
      throw new Error('The local session is reconnecting');
    this.#socket.send(frame);
  }

  async prepareMessages(messages) {
    const prepared = [];
    for (const message of messages) {
      const images = [];
      for (const image of message.images || []) {
        const blob = await readImage(image);
        const streamId = this.#nextImageStream++;
        const uploadId = crypto.randomUUID();
        const completed = new Promise((resolve, reject) => this.#uploads.set(uploadId, {resolve, reject}));
        this.send('image.start', uploadId, {
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
          this.sendBinary(frame);
          await this.#waitForDrain();
        }
        this.send('image.finish', uploadId, {stream_id: streamId});
        await completed;
        images.push(streamId);
      }
      prepared.push(images.length
        ? {role: message.role, content: message.content, images}
        : {role: message.role, content: message.content});
    }
    return prepared;
  }

  async #waitForDrain() {
    const started = performance.now();
    while (this.#socket?.readyState === WebSocket.OPEN && this.#socket.bufferedAmount > 512 * 1024) {
      if (performance.now() - started > 30000)
        throw new Error('Image upload could not keep up with the local server');
      await new Promise(resolve => setTimeout(resolve, 5));
    }
    if (!this.#socket || this.#socket.readyState !== WebSocket.OPEN)
      throw new Error('The local session disconnected during image upload');
  }
}

export const backend = new DesktopBackend();
