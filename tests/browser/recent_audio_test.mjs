import test from 'node:test';
import assert from 'node:assert/strict';
import {RecentAudio} from '../../web/ui/speech-controller.js';

test('cached audio owns its bytes and replay is safe to transfer', () => {
    const cache = new RecentAudio();
    const source = new Float32Array([0.1, 0.2]).buffer;
    cache.put('voice-and-text', source, 24000);
    structuredClone(source, {transfer: [source]});
    const first = cache.get('voice-and-text');
    structuredClone(first.audio, {transfer: [first.audio]});
    assert.equal(cache.get('voice-and-text').audio.byteLength, 8);
    assert.equal(cache.get('different-voice'), null);
    assert.equal(cache.bytes, 8);
});

test('replay cache enforces bytes, LRU count, and expiry', () => {
    let now = 0;
    const cache = new RecentAudio({maximumBytes: 16, maximumEntries: 2, lifetimeMs: 10, now: () => now});
    const audio = new Float32Array(2).buffer;
    cache.put('one', audio, 24000);
    cache.put('two', audio, 24000);
    cache.get('one');
    cache.put('three', audio, 24000);
    assert.equal(cache.get('two'), null);
    assert.equal(cache.bytes, 16);
    now = 10;
    assert.equal(cache.get('one'), null);
    assert.equal(cache.get('three'), null);
    assert.equal(cache.bytes, 0);
});
