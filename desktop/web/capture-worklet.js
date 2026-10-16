class KidiCaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.phase = 0;
    this.output = new Int16Array(320);
    this.offset = 0;
  }

  process(inputs) {
    const channels = inputs[0];
    if (!channels?.length || !channels[0]?.length) return true;
    const rate = 16000 / sampleRate;
    for (let index = 0; index < channels[0].length; index++) {
      let value = 0;
      for (const channel of channels) value += channel[index] / channels.length;
      this.phase += rate;
      if (this.phase < 1) continue;
      this.phase -= 1;
      this.output[this.offset++] = Math.max(-32768, Math.min(32767, Math.round(value * 32768)));
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
