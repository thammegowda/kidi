import assert from 'node:assert/strict';
import test from 'node:test';
import {ReadAloud} from '../../web/ui/read-aloud.js';
import {resample} from '../../web/ui/audio-playback.js';

const tick = async () => {
  for (let count = 0; count < 12; count++) await new Promise(resolve => setImmediate(resolve));
};
function harness(t, failResume = false) {
  const originalContext = globalThis.AudioContext, originalNode = globalThis.AudioWorkletNode;
  const messages = [], commands = [], states = [], errors = [];
  let node, audio;
  globalThis.AudioContext = class {
    state = 'running';
    sampleRate = 8000;
    destination = {};
    audioWorklet = {addModule: async url => assert.match(url.pathname, /playback-worklet\.js$/)};
    constructor() { audio = this; }
    async resume() { if (failResume) throw new Error('Audio permission denied'); }
    async close() { this.state = 'closed'; }
  };
  globalThis.AudioWorkletNode = class {
    port = {postMessage: message => messages.push(message)};
    constructor() { node = this; }
    connect() {}
  };
  t.after(() => {
    globalThis.AudioContext = originalContext;
    globalThis.AudioWorkletNode = originalNode;
  });
  const reader = new ReadAloud({
    send: (type, id, payload) => {
      commands.push({type, id, payload});
      if (type === 'cancel') queueMicrotask(() => reader.handleEvent({type: 'request.cancelled', id}));
    },
    onState: state => states.push(state),
    onError: error => errors.push(error.message),
  });
  const receive = data => node.port.onmessage({data});
  const finish = async (seconds = 1) => {
    const command = commands.filter(command => command.type === 'tts.start').at(-1);
    assert.ok(reader.handlePcm({streamId: command.payload.stream_id, sequence: 0, sampleRate: 8000,
      samples: new Float32Array(seconds * 8000)}));
    reader.handleEvent({type: 'tts.completed', id: command.id, payload: {sample_rate: 8000}});
    await tick();
  };
  return {reader, messages, commands, states, errors, receive, finish, audio: () => audio};
}

test('read aloud bounds queued audio to six seconds and finishes only after playback drains', async t => {
  const env = harness(t);
  await env.reader.start({text: 'A short sentence.', language: 'English'});
  await env.finish(10);
  const streamId = env.messages.find(message => message.type === 'start').streamId;
  const queued = () => env.messages.filter(message => message.type === 'chunk')
    .reduce((total, message) => total + message.samples.byteLength / 4, 0);
  assert.equal(queued(), 6 * 8000, 'large native segments cannot exceed the playback budget');
  assert.equal(env.messages.some(message => message.type === 'end'), false);
  await env.reader.togglePause();
  assert.equal(env.messages.at(-1).type, 'pause');
  await env.reader.togglePause();
  assert.equal(env.messages.at(-1).type, 'resume');
  env.receive({type: 'buffered', streamId, playedFrames: 4 * 8000});
  await tick();
  assert.equal(queued() - 4 * 8000, 6 * 8000);
  assert.equal(env.messages.at(-1).type, 'end');
  assert.equal(env.states.at(-1).active, true, 'synthesis completion does not finish playing audio');
  env.receive({type: 'drained', streamId});
  assert.equal(env.states.at(-1).message, 'Finished reading');
  assert.equal(env.states.at(-1).active, false);
  assert.deepEqual(env.errors, []);
  await env.reader.close();
});

test('stop releases backpressure and ignores late audio and completion', async t => {
  const env = harness(t);
  await env.reader.start({text: 'A short sentence.', language: 'French'});
  const command = env.commands[0];
  await env.finish(12);
  await env.reader.stop();
  const count = env.messages.length;
  await tick();
  assert.equal(env.messages.length, count);
  assert.equal(env.reader.handlePcm({streamId: command.payload.stream_id, sequence: 1, sampleRate: 8000,
    samples: new Float32Array(80)}), false);
  assert.equal(env.reader.handleEvent({type: 'tts.completed', id: command.id}), false);
  assert.equal(env.states.at(-1).active, false);
  assert.deepEqual(env.errors, []);
  await env.reader.close();
});

test('speech cancellation is scoped, and another reading waits for its acknowledgement', async t => {
  const env = harness(t);
  await env.reader.start({text: 'First sentence.', language: 'French'});
  await env.reader.start({text: 'A different sentence.', language: 'Spanish'});
  assert.deepEqual(env.commands.map(command => command.type), ['tts.start', 'cancel', 'tts.start']);
  assert.equal(env.commands[1].payload.scope, 'speech');
  assert.equal(env.commands[2].payload.options.language, 'Spanish');
  await env.reader.stop();
  assert.deepEqual(env.errors, []);
  await env.reader.close();
});

test('unknown language is never silently replaced and audio initialization errors are visible', async t => {
  const env = harness(t, true);
  await assert.rejects(env.reader.start({text: 'Text.', language: ''}), /explicit narration language/);
  await env.reader.start({text: 'Text.', language: 'Kannada'});
  assert.deepEqual(env.errors, ['Audio permission denied']);
  assert.equal(env.commands.length, 0);
  assert.equal(env.audio().state, 'closed');
});

test('the shared resampler preserves same-rate PCM and interpolates different rates', () => {
  const source = new Float32Array([0, 1, 0, -1]);
  assert.equal(resample(source, 8000, 8000), source);
  assert.deepEqual([...resample(source, 8000, 16000)], [0, .5, 1, .5, 0, -.5, -1, -1]);
});
