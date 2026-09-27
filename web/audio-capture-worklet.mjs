class KidiAudioCapture extends AudioWorkletProcessor {
    process(inputs) {
        const channels = inputs[0];
        if (!channels?.length || !channels[0].length) return true;
        const samples = new Float32Array(channels[0].length);
        for (const channel of channels)
            for (let index = 0; index < samples.length; index++) samples[index] += channel[index] / channels.length;
        this.port.postMessage(samples.buffer, [samples.buffer]);
        return true;
    }
}

registerProcessor('kidi-audio-capture', KidiAudioCapture);