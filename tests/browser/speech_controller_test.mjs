import test from 'node:test';
import assert from 'node:assert/strict';
import {SPEECH_BLOCK_PAUSE_MS, SpeechController} from '../../web/ui/speech-controller.js';
import {READER_BLOCK_BREAK} from '../../web/ui/reader-position.js';

function harness({blockFirstDelivery = false, failStart = false} = {}) {
  const sent = [], segments = [], pcm = [], completed = [], cancelled = [], errors = [];
  let wire = 100, release;
  const blocked = new Promise(resolve => { release = resolve; });
  const controller = new SpeechController({
    send: (type, id, payload) => {
      sent.push({type, id, payload});
      if (failStart && type === 'tts.start') throw new Error('Transport unavailable');
    },
    nextWireStream: () => wire++,
    onSegment: (_stream, segment, detail) => segments.push({...segment, ...detail}),
    onPcm: (_stream, frame) => {
      pcm.push(frame);
      return blockFirstDelivery && pcm.length === 1 ? blocked : undefined;
    },
    onCompleted: stream => completed.push(stream),
    onCancelled: (stream, message) => cancelled.push({stream, message}),
    onError: (stream, error) => errors.push({stream, error}),
  });
  const primitive = index => sent.filter(item => item.type === 'tts.start')[index];
  const audio = (index, values = [0.1, 0.2]) => {
    const request = primitive(index);
    controller.handlePcm({streamId: request.payload.stream_id, sequence: 0, sampleRate: 24000,
      samples: Float32Array.from(values)});
    controller.handleEvent({id: request.id, type: 'tts.completed',
      payload: {stream_id: request.payload.stream_id, sample_rate: 24000}});
  };
  return {controller, sent, segments, pcm, completed, cancelled, errors, primitive, audio, release};
}

test('shared controller segments, tags, and pipelines exactly one request ahead', async () => {
  const run = harness({blockFirstDelivery: true});
  run.controller.start({id: 'outer', streamId: 7,
    text: 'First sentence. Second sentence. Third sentence.', finished: true});
  assert.equal(run.primitive(0).payload.text, 'First sentence.');
  run.audio(0);
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(run.primitive(1).payload.text, 'Second sentence.');
  run.audio(1);
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(run.sent.filter(item => item.type === 'tts.start').length, 2);
  assert.equal(run.pcm.length, 1);
  run.release();
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(run.pcm.length, 2);
  assert.equal(run.primitive(2).payload.text, 'Third sentence.');
  run.audio(2);
  await new Promise(resolve => setImmediate(resolve));
  assert.deepEqual(run.segments.map(segment => segment.id), [0, 1, 2]);
  assert.deepEqual(run.pcm.map(frame => [frame.streamId, frame.sequence, frame.segmentId]),
    [[7, 0, 0], [7, 1, 1], [7, 2, 2]]);
  assert.equal(run.completed.length, 1);
  assert.equal(run.errors.length, 0);
});

test('unfinished speech streams accept translated text while playback is active', async () => {
  const run = harness();
  run.controller.start({
    id: 'translation', streamId: 8, text: 'First translated sentence. ', finished: false,
  });

  test('block boundaries add silence without changing synthesized text', async () => {
    const run = harness();
    run.controller.start({
      id: 'blocks', streamId: 12,
      text: `A short title${READER_BLOCK_BREAK}The first sentence.`, finished: true,
    });
    assert.equal(run.primitive(0).payload.text, 'A short title');
    run.audio(0, [0.25, -0.25]);
    await new Promise(resolve => setImmediate(resolve));
    assert.equal(run.pcm[0].samples.length, 2 + 24000 * SPEECH_BLOCK_PAUSE_MS / 1000);
    assert.deepEqual([...run.pcm[0].samples.slice(0, 2)], [0.25, -0.25]);
    assert.ok(run.pcm[0].samples.slice(2).every(value => value === 0));
    assert.equal(run.primitive(1).payload.text, 'The first sentence.');
  });
  assert.equal(run.primitive(0).payload.text, 'First translated sentence.');
  run.audio(0);
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(run.completed.length, 0);
  run.controller.append('translation', 'Second translated sentence.', true);
  assert.equal(run.primitive(1).payload.text, 'Second translated sentence.');
  run.audio(1);
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(run.completed.length, 1);
});

test('appended translation cannot overtake a segment waiting for playback capacity', async () => {
  const run = harness({blockFirstDelivery: true});
  run.controller.start({
    id: 'translation-order', streamId: 13,
    text: 'First translated segment. Second translated segment. ',
    finished: false,
  });
  run.audio(0);
  await new Promise(resolve => setImmediate(resolve));
  run.audio(1);
  await new Promise(resolve => setImmediate(resolve));
  run.controller.append('translation-order', 'Third translated segment.', true);
  assert.equal(run.primitive(2), undefined);
  assert.deepEqual(run.pcm.map(frame => frame.segmentId), [0]);
  run.release();
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(run.primitive(2).payload.text, 'Third translated segment.');
  run.audio(2);
  await new Promise(resolve => setImmediate(resolve));
  assert.deepEqual(run.pcm.map(frame => frame.segmentId), [0, 1, 2]);
  assert.equal(run.completed.length, 1);
});

test('replacement waits for backend cancellation acknowledgement', async () => {
  const run = harness();
  run.controller.start({id: 'old', streamId: 1, text: 'Old sentence.', finished: true});
  const old = run.primitive(0);
  const waiting = run.controller.cancel('old', 'Replaced');
  assert.deepEqual(run.sent.at(-1), {
    type: 'cancel', id: old.id, payload: {scope: 'speech', message: 'Replaced'},
  });
  let settled = false;
  waiting.then(() => { settled = true; });
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(settled, false);
  run.controller.handleEvent({id: old.id, type: 'request.cancelled', payload: {}});
  await waiting;
  assert.equal(run.cancelled.length, 1);
  run.controller.start({id: 'new', streamId: 2, text: 'New sentence.', finished: true});
  assert.equal(run.primitive(1).payload.text, 'New sentence.');
});

test('cached speech preserves the same logical segment contract', async () => {
  const run = harness();
  const request = {id: 'one', streamId: 3, text: 'Repeat me.', options: {voice: 'same'},
    cacheIdentity: 'native', finished: true};
  run.controller.start(request);
  run.audio(0, [0.25, -0.25]);
  await new Promise(resolve => setImmediate(resolve));
  run.controller.start({...request, id: 'two', streamId: 4});
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(run.sent.filter(item => item.type === 'tts.start').length, 1);
  assert.equal(run.segments.at(-1).cached, true);
  assert.deepEqual([...run.pcm.at(-1).samples], [0.25, -0.25]);
});

test('native playback completion advances without returning PCM to the UI', async () => {
  const run = harness();
  run.controller.start({
    id: 'native', streamId: 14,
    text: 'First native sentence. Second native sentence.', finished: true,
  });
  const first = run.primitive(0);
  run.controller.handleEvent({
    id: first.id,
    type: 'tts.completed',
    payload: {native_playback: true, sample_rate: 24000, samples: 12000, chunks: 1},
  });
  await new Promise(resolve => setImmediate(resolve));
  const second = run.primitive(1);
  assert.ok(second);
  run.controller.handleEvent({
    id: second.id,
    type: 'tts.completed',
    payload: {native_playback: true, sample_rate: 24000, samples: 16000, chunks: 1},
  });
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(run.pcm.length, 0);
  assert.deepEqual(run.segments.map(segment => [segment.id, segment.native]), [[0, true], [1, true]]);
  assert.equal(run.completed.length, 1);
  assert.equal(run.completed[0].samples, 28000);
});

test('transport reset releases an active stream without another command', () => {
  const run = harness();
  run.controller.start({id: 'active', streamId: 9, text: 'Interrupted sentence.', finished: true});
  const commands = run.sent.length;
  run.controller.reset('Disconnected');
  assert.equal(run.controller.active, null);
  assert.equal(run.sent.length, commands);
  assert.equal(run.cancelled[0].message, 'Disconnected');
});

test('completion racing cancellation resolves every cancellation waiter', async () => {
  const run = harness();
  run.controller.start({id: 'race', streamId: 10, text: 'Race sentence.', finished: true});
  const primitive = run.primitive(0);
  const first = run.controller.cancel('race', 'Stopped');
  const second = run.controller.cancel('race', 'Stopped');
  run.controller.handlePcm({streamId: primitive.payload.stream_id, sequence: 0, sampleRate: 24000,
    samples: Float32Array.from([0.1])});
  run.controller.handleEvent({id: primitive.id, type: 'tts.completed', payload: {sample_rate: 24000}});
  await Promise.all([first, second]);
  assert.equal(run.controller.active, null);
  assert.equal(run.cancelled.length, 1);
});

test('synchronous transport failure cannot wedge the next stream', () => {
  const failed = harness({failStart: true});
  failed.controller.start({id: 'failed', streamId: 11, text: 'Cannot start.', finished: true});
  assert.equal(failed.controller.active, null);
  assert.match(failed.errors[0].error.message, /Transport unavailable/);

  const recovered = harness();
  recovered.controller.start({id: 'next', streamId: 12, text: 'Can start.', finished: true});
  assert.equal(recovered.primitive(0).payload.text, 'Can start.');
});
