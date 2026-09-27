import test from 'node:test';
import assert from 'node:assert/strict';
import {resampleAudio} from '../../web/speech.mjs';

test('captured speech resamples to 16 kHz and respects the duration cap', () => {
    const input = Float32Array.from({length: 48000}, (_, index) => index / 48000);
    const output = resampleAudio([input], 48000);
    assert.equal(output.length, 16000);
    assert.ok(Math.abs(output[8000] - 0.5) < 1e-6);
    assert.equal(resampleAudio([new Float32Array(600000)], 16000).length, 480000);
});