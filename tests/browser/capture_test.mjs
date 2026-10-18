import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import vm from 'node:vm';

const source = await readFile(new URL('../../web/hosts/desktop/capture-worklet.js', import.meta.url), 'utf8');

function capture(rate = 48000) {
    const frames = [];
    let Processor;
    vm.runInNewContext(source, {
        Int16Array,
        sampleRate: rate,
        AudioWorkletProcessor: class {
            port = {postMessage: frame => frames.push(new Int16Array(frame))};
        },
        registerProcessor(name, type) {
            assert.equal(name, 'kidi-capture');
            Processor = type;
        },
    });
    return {processor: new Processor(), frames};
}

test('capture worklet anti-aliases while downsampling 48 kHz PCM', () => {
    const {processor, frames} = capture();
    const input = Float32Array.from({length: 960}, (_, index) => [1, -0.5, -0.5][index % 3]);
    assert.equal(processor.process([[input]]), true);
    assert.equal(frames.length, 1);
    assert.equal(frames[0].length, 320);
    assert.ok(frames[0].every(sample => Math.abs(sample) <= 1));
});

test('capture worklet preserves speech-band DC level and frame size', () => {
    const {processor, frames} = capture();
    assert.equal(processor.process([[new Float32Array(960).fill(0.25)]]), true);
    assert.equal(frames.length, 1);
    assert.ok(frames[0].every(sample => sample === 8192));
});
