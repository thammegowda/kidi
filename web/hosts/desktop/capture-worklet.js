class KidiCaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.phase = 0;
    this.sum = 0;
    this.count = 0;
    this.output = new Int16Array(320);
    this.offset = 0;
  }

  process(inputs) {
    const channels = inputs[0];
    if (!channels?.length || !channels[0]?.length) return true;
    for (let index = 0; index < channels[0].length; index++) {
      let value = 0;
      for (const channel of channels) value += channel[index] / channels.length;
      this.sum += value;
      this.count++;
      this.phase += 16000;
      if (this.phase < sampleRate) continue;
      this.phase -= sampleRate;
      const filtered = this.sum / this.count;
      this.sum = 0;
      this.count = 0;
      this.output[this.offset++] = Math.max(-32768, Math.min(32767, Math.round(filtered * 32768)));
      if (this.offset !== this.output.length) continue;
      const frame = this.output.buffer;
      this.port.postMessage(frame, [frame]);
      this.output = new Int16Array(320);
      this.offset = 0;
    }
    return true;
  }
}

registerProcessor('kidi-capture', KidiCaptureProcessor);
