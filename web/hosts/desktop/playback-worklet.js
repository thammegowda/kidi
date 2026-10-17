class KidiPlaybackProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.queue = [];
    this.offset = 0;
    this.ending = false;
    this.port.onmessage = event => {
      if (event.data.type === 'chunk') this.queue.push(new Float32Array(event.data.samples));
      if (event.data.type === 'end') this.ending = true;
      if (event.data.type === 'reset') { this.queue = []; this.offset = 0; this.ending = false; }
    };
  }

  process(_inputs, outputs) {
    const output = outputs[0][0];
    output.fill(0);
    let written = 0;
    while (written < output.length && this.queue.length) {
      const current = this.queue[0];
      const count = Math.min(output.length - written, current.length - this.offset);
      output.set(current.subarray(this.offset, this.offset + count), written);
      written += count;
      this.offset += count;
      if (this.offset === current.length) { this.queue.shift(); this.offset = 0; }
    }
    if (this.ending && !this.queue.length) {
      this.ending = false;
      this.port.postMessage({type: 'drained'});
    }
    return true;
  }
}

registerProcessor('kidi-playback', KidiPlaybackProcessor);
