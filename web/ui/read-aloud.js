import {resample} from './audio-playback.js';
import {SpeechController} from './speech-controller.js';

export class ReadAloud {
  #speech;
  #player;
  #preparing;
  #current = null;
  #nextStream = 1;
  #onState;
  #onError;

  constructor({send, onState, onError}) {
    this.#onState = onState;
    this.#onError = onError;
    this.#speech = new SpeechController({
      send,
      nextWireStream: () => this.#nextStream++,
      onProgress: (_current, progress) => this.#publish('Synthesizing speech', progress),
      onSegment: (_current, segment) => {
        if (this.#current) this.#current.segment = segment.id + 1;
      },
      onPcm: (_current, frame) => this.#deliver(frame),
      onCompleted: current => {
        if (this.#current?.id !== current.id) return;
        this.#current.completed = true;
        this.#publish('Finishing audio');
        this.#player.node.port.postMessage({type: 'end', streamId: current.streamId});
      },
      onCancelled: current => {
        if (this.#current?.id === current.id) this.#reset('Reading stopped');
      },
      onError: (_current, error) => this.#fail(error),
    });
  }

  prepare() {
    if (this.#player?.context.state === 'closed') {
      this.#player = null;
      this.#preparing = null;
    }
    if (this.#player) return this.#player.context.resume().then(() => this.#player);
    if (this.#preparing) return this.#preparing;
    const context = new AudioContext();
    this.#preparing = Promise.all([
      context.resume(), context.audioWorklet.addModule(new URL('./playback-worklet.js', import.meta.url)),
    ]).then(() => {
      const node = new AudioWorkletNode(context, 'kidi-playback', {outputChannelCount: [1]});
      node.connect(context.destination);
      node.port.onmessage = ({data}) => {
        const current = this.#current;
        if (!current || data.streamId !== current.streamId) return;
        if (data.type === 'buffered') {
          current.playedFrames = data.playedFrames;
          for (const resolve of current.waiters.splice(0)) resolve();
        } else if (data.type === 'segment-started') {
          this.#publish(`Reading segment ${data.segmentId + 1}`);
        } else if (data.type === 'drained' && current.completed) {
          this.#reset('Finished reading');
        }
      };
      node.onprocessorerror = () => this.#fail(new Error('The speech audio processor failed'));
      context.onstatechange = () => {
        if (this.#current && ['closed', 'interrupted'].includes(context.state))
          this.#fail(new Error('Audio playback was interrupted. Click Read aloud to retry.'));
      };
      this.#player = {context, node};
      return this.#player;
    }).catch(error => {
      this.#preparing = null;
      if (context.state !== 'closed')
        context.close().catch(closeError => console.error('Unable to close failed audio:', closeError));
      throw error;
    });
    return this.#preparing;
  }

  async start({text, language, cacheIdentity = ''}) {
    if (typeof text !== 'string' || !text.trim() || typeof language !== 'string' || !language.trim())
      throw new Error('Read aloud requires text and an explicit narration language');
    // Resume inside the click gesture, before waiting for cancellation or model loading.
    const player = this.prepare();
    const stopped = this.stop();
    const current = {
      id: crypto.randomUUID(), streamId: this.#nextStream++, paused: false,
      started: false, completed: false, sentFrames: 0, playedFrames: 0, sequence: 0, waiters: [],
    };
    this.#current = current;
    this.#publish('Preparing local speech');
    try {
      const [audio] = await Promise.all([player, stopped]);
      if (this.#current !== current) return;
      audio.node.port.postMessage({type: 'start', streamId: current.streamId});
      this.#speech.start({
        id: current.id, streamId: current.streamId,
        text: text.replace(/\n\s*\n/gu, '\u2029'), options: {language},
        cacheIdentity, finished: true,
      });
    } catch (error) {
      if (this.#current === current) this.#fail(error);
      else console.warn('Stopped speech setup failed:', error);
    }
  }

  handleEvent(event) { return this.#speech.handleEvent(event); }
  handlePcm(frame) { return this.#speech.handlePcm(frame); }

  async togglePause() {
    const current = this.#current;
    if (!current?.started) throw new Error('There is no playing audio to pause');
    if (current.paused) await this.#player.context.resume();
    if (this.#current !== current) return;
    current.paused = !current.paused;
    this.#player.node.port.postMessage({type: current.paused ? 'pause' : 'resume', streamId: current.streamId});
    this.#publish(current.paused ? 'Paused' : 'Reading');
  }

  stop() {
    this.#reset('Reading stopped');
    const current = this.#speech.active;
    return current ? this.#speech.cancel(current.id) : Promise.resolve();
  }

  async close() {
    const stopped = this.stop();
    const player = this.#player || await this.#preparing;
    await stopped;
    if (player && player.context.state !== 'closed') await player.context.close();
  }

  #publish(message, progress) {
    const current = this.#current;
    if (!current) return;
    this.#onState({active: true, paused: current.paused, started: current.started,
      message: current.paused ? 'Paused' : message, progress});
  }

  #reset(message) {
    const current = this.#current;
    this.#current = null;
    this.#player?.node.port.postMessage({type: 'reset'});
    for (const resolve of current?.waiters.splice(0) || []) resolve();
    if (current) this.#onState({active: false, message});
  }

  #fail(error) {
    this.stop().catch(cancelError => this.#onError(cancelError));
    this.#onError(error);
  }

  async #deliver({streamId, sequence, sampleRate, samples, segmentId}) {
    const current = this.#current;
    if (!current || current.streamId !== streamId) return;
    if (!(samples instanceof Float32Array) || !samples.length || samples.some(value => !Number.isFinite(value)) ||
        !Number.isInteger(sampleRate) || sampleRate <= 0 || sequence !== current.sequence++)
      throw new Error('Invalid or out-of-order read-aloud audio');
    const {context, node} = this.#player;
    const output = resample(samples, sampleRate, context.sampleRate);
    const maximumFrames = context.sampleRate * 6;
    for (let offset = 0; offset < output.length;) {
      if (this.#current !== current) return;
      const capacity = maximumFrames - (current.sentFrames - current.playedFrames);
      if (capacity <= 0) {
        await new Promise(resolve => current.waiters.push(resolve));
        continue;
      }
      const length = Math.min(capacity, Math.ceil(context.sampleRate / 4), output.length - offset);
      const chunk = output.slice(offset, offset + length);
      current.sentFrames += length;
      offset += length;
      node.port.postMessage({type: 'chunk', streamId, segmentId, samples: chunk.buffer}, [chunk.buffer]);
      if (!current.started) {
        current.started = true;
        this.#publish('Reading');
      }
    }
  }
}
