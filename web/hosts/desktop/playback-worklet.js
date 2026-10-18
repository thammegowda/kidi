class KidiPlaybackProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.queue = [];
    this.offset = 0;
    this.ending = false;
    this.paused = false;
    this.streamId = null;
    this.frames = 0;
    this.playedFrames = 0;
    this.reported = 0;
    this.port.onmessage = event => {
      const data = event.data;
      if (data.type === 'start') {
        this.queue = []; this.offset = 0; this.ending = false; this.paused = false; this.streamId = data.streamId;
        this.frames = 0; this.reported = 0;
        this.playedFrames = 0;
      }
      if (data.type === 'chunk' && data.streamId === this.streamId) {
        const samples = new Float32Array(data.samples);
        this.queue.push({samples, segmentId: data.segmentId, started: false});
        this.frames += samples.length;
        this.port.postMessage({type: 'buffered', streamId: this.streamId, frames: this.frames, playedFrames: this.playedFrames});
      }
      if (data.type === 'end' && data.streamId === this.streamId) this.ending = true;
      if (data.type === 'pause' && data.streamId === this.streamId) this.paused = true;
      if (data.type === 'resume' && data.streamId === this.streamId) this.paused = false;
      if (data.type === 'reset') {
        this.queue = []; this.offset = 0; this.ending = false; this.paused = false; this.streamId = null;
        this.frames = 0; this.reported = 0;
        this.playedFrames = 0;
      }
    };
  }

  process(_inputs, outputs) {
    const output = outputs[0][0];
    output.fill(0);
    if (this.paused) return true;
    let written = 0;
    while (written < output.length && this.queue.length) {
      const current = this.queue[0];
      if (!current.started) {
        current.started = true;
        if (current.segmentId !== undefined)
          this.port.postMessage({type: 'segment-started', streamId: this.streamId, segmentId: current.segmentId});
      }
      const count = Math.min(output.length - written, current.samples.length - this.offset);
      output.set(current.samples.subarray(this.offset, this.offset + count), written);
      written += count;
      this.offset += count;
      this.frames -= count;
      this.playedFrames += count;
      this.reported += count;
      if (this.offset === current.samples.length) {
        if (current.segmentId !== undefined)
          this.port.postMessage({type: 'segment-ended', streamId: this.streamId, segmentId: current.segmentId});
        this.queue.shift(); this.offset = 0;
      }
    }
    if (this.reported >= 2048 || (written && !this.frames)) {
      this.reported = 0;
      this.port.postMessage({type: 'buffered', streamId: this.streamId, frames: this.frames, playedFrames: this.playedFrames});
    }
    if (this.ending && !this.queue.length) {
      this.ending = false;
      this.port.postMessage({type: 'drained', streamId: this.streamId});
    }
    return true;
  }
}

registerProcessor('kidi-playback', KidiPlaybackProcessor);
