import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import vm from 'node:vm';

const source = await readFile(new URL('../../web/hosts/desktop/playback-worklet.js', import.meta.url), 'utf8');

function playback() {
    const messages = [];
    let Processor;
    vm.runInNewContext(source, {
        Float32Array,
        AudioWorkletProcessor: class {
            port = {postMessage: message => messages.push(message)};
        },
        registerProcessor(name, type) {
            assert.equal(name, 'kidi-playback');
            Processor = type;
        },
    });
    const processor = new Processor();
    const send = data => processor.port.onmessage({data});
    const chunk = (streamId, samples, segmentId) => send({type: 'chunk', streamId, segmentId, samples: Float32Array.from(samples).buffer});
    const render = count => {
        const output = new Float32Array(count);
        assert.equal(processor.process([], [[output]]), true);
        return [...output];
    };
    return {send, chunk, render, messages};
}

test('playback preserves Float32 samples and only drains after every chunk', () => {
    const player = playback();
    player.send({type: 'start', streamId: 7});
    player.chunk(7, [0.123456789, 0.25, 0.5]);
    player.chunk(7, [-0.5, -0.25]);
    player.send({type: 'end', streamId: 7});
    assert.deepEqual(player.render(4), [Math.fround(0.123456789), 0.25, 0.5, -0.5]);
    assert.equal(player.messages.filter(message => message.type === 'drained').length, 0);
    assert.deepEqual(player.render(4), [-0.25, 0, 0, 0]);
    const drained = player.messages.filter(message => message.type === 'drained');
    assert.equal(drained.length, 1);
    assert.equal(drained[0].streamId, 7);
    player.render(4);
    assert.equal(player.messages.filter(message => message.type === 'drained').length, 1);
});

test('reset fences stale chunks and completion from a cancelled stream', () => {
    const player = playback();
    player.send({type: 'start', streamId: 11});
    player.chunk(11, [1, 1]);
    player.send({type: 'end', streamId: 11});
    player.send({type: 'reset'});
    player.send({type: 'start', streamId: 12});
    player.chunk(12, [0.25]);
    player.chunk(11, [-1, -1]);
    player.send({type: 'end', streamId: 11});
    assert.deepEqual(player.render(4), [0.25, 0, 0, 0]);
    assert.equal(player.messages.filter(message => message.type === 'drained').length, 0);
    player.send({type: 'end', streamId: 12});
    player.render(4);
    assert.equal(player.messages.find(message => message.type === 'drained').streamId, 12);
});

test('playback reports actual segment boundaries and buffered frames', () => {
    const player = playback();
    player.send({type: 'start', streamId: 5});
    player.chunk(5, [0.1, 0.2, 0.3], 0);
    player.chunk(5, [0.4, 0.5], 1);
    player.render(4);
    assert.deepEqual(player.messages.filter(message => message.type.startsWith('segment-'))
        .map(message => [message.type, message.segmentId]),
    [['segment-started', 0], ['segment-ended', 0], ['segment-started', 1]]);
    player.render(4);
    assert.equal(player.messages.filter(message => message.type === 'buffered').at(-1).frames, 0);
    assert.equal(player.messages.filter(message => message.type === 'segment-ended').at(-1).segmentId, 1);
    assert.equal(player.messages.filter(message => message.type === 'buffered').at(-1).playedFrames, 5);
});

test('buffer reports include consumed frames even when newer chunks are in transit', () => {
    const player = playback();
    player.send({type: 'start', streamId: 6});
    player.chunk(6, new Array(4096).fill(0.1), 0);
    const first = player.messages.at(-1);
    player.chunk(6, new Array(4096).fill(0.2), 1);
    assert.equal(first.frames, 4096);
    assert.equal(first.playedFrames, 0);
    assert.equal(8192 - first.playedFrames, 8192);
    player.render(2048);
    assert.equal(player.messages.at(-1).playedFrames, 2048);
    assert.equal(player.messages.at(-1).frames, 6144);
});
