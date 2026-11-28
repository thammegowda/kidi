// Dictation (microphone → Whisper → composer text) and read-aloud (reply text → OmniVoice → speakers).
import {call} from './rpc.js';
import {CAPTURE_RATE, openMicrophone, openPlayer, pcm16ToFloat32, resample} from './liteapp/media/audio.mjs';
import {base64ToBytes, bytesToBase64} from './liteapp/media/base64.mjs';

const MAX_SECONDS = 120;
const MIC_ICON = '<svg viewBox="0 0 16 16" width="13" height="13" aria-hidden="true"><path fill="currentColor" ' +
  'd="M8 1a2.5 2.5 0 0 0-2.5 2.5v4a2.5 2.5 0 0 0 5 0v-4A2.5 2.5 0 0 0 8 1Zm-4.5 6a.5.5 0 0 0-1 0 5.5 5.5 0 0 0 ' +
  '5 5.48V14H5.5a.5.5 0 0 0 0 1h5a.5.5 0 0 0 0-1H8.5v-1.52A5.5 5.5 0 0 0 13.5 7a.5.5 0 0 0-1 0 4.5 4.5 0 0 1-9 0Z"/></svg>';
const SPEAKER_ICON = '<svg viewBox="0 0 16 16" width="13" height="13" aria-hidden="true"><path fill="currentColor" ' +
  'd="M7.6 2.2a.5.5 0 0 1 .4.5v10.6a.5.5 0 0 1-.8.4L4.3 11H2.5A1.5 1.5 0 0 1 1 9.5v-3A1.5 1.5 0 0 1 2.5 5h1.8l2.9-2.7' +
  'a.5.5 0 0 1 .4-.1ZM10.5 5a.5.5 0 0 1 .7 0 4.2 4.2 0 0 1 0 6 .5.5 0 1 1-.7-.7 3.2 3.2 0 0 0 0-4.6.5.5 0 0 1 0-.7Z' +
  'm1.8-1.8a.5.5 0 0 1 .7 0 6.8 6.8 0 0 1 0 9.6.5.5 0 0 1-.7-.7 5.8 5.8 0 0 0 0-8.2.5.5 0 0 1 0-.7Z"/></svg>';
const STOP_ICON = '<svg viewBox="0 0 16 16" width="11" height="11" aria-hidden="true">' +
  '<rect x="3" y="3" width="10" height="10" rx="1.5" fill="currentColor"/></svg>';

let available = {transcribe: false, speak: false};
let onError = () => {};

// ---- Dictation ----
const micButton = document.getElementById('dictate');
const prompt = document.getElementById('prompt');
let recording = null;

async function startRecording() {
  const frames = [];
  let samples = 0;
  const microphone = await openMicrophone({
    onFrame: frame => {
      frames.push(frame);
      samples += frame.length;
      if (samples >= MAX_SECONDS * CAPTURE_RATE) stopRecording();
    },
  });
  recording = {microphone, frames, started: performance.now()};
  micButton.classList.add('recording');
  micButton.title = 'Stop and transcribe';
  recording.timer = setInterval(() => {
    const seconds = Math.floor((performance.now() - recording.started) / 1000);
    prompt.placeholder = `Listening… ${seconds}s (click the microphone to finish)`;
  }, 250);
}

async function stopRecording() {
  if (!recording) return;
  const {microphone, frames, timer} = recording;
  recording = null;
  clearInterval(timer);
  await microphone.close();
  micButton.classList.remove('recording');
  micButton.classList.add('working');
  micButton.title = 'Dictate';
  const placeholder = prompt.placeholder;
  prompt.placeholder = 'Transcribing… (the first use downloads and loads Whisper)';
  try {
    const total = Math.min(frames.reduce((sum, frame) => sum + frame.length, 0), MAX_SECONDS * CAPTURE_RATE);
    if (total < CAPTURE_RATE / 4) return;
    const pcm = new Int16Array(total);
    let offset = 0;
    for (const frame of frames) {
      if (offset >= total) break;
      const part = frame.subarray(0, total - offset);
      pcm.set(part, offset);
      offset += part.length;
    }
    const {text} = await call('speech.transcribe', {audio: bytesToBase64(new Uint8Array(pcm.buffer))});
    const spoken = text.trim();
    if (!spoken) return;
    const before = prompt.value.slice(0, prompt.selectionStart);
    const after = prompt.value.slice(prompt.selectionEnd);
    const joiner = before && !/\s$/.test(before) ? ' ' : '';
    prompt.value = before + joiner + spoken + after;
    prompt.dispatchEvent(new Event('input'));
    prompt.focus();
  } catch (error) {
    onError(`Dictation failed: ${error.message}`);
  } finally {
    micButton.classList.remove('working');
    prompt.placeholder = placeholder.startsWith('Listening') ? 'Ask Kidi… (/ for commands)' : placeholder;
  }
}

micButton.innerHTML = MIC_ICON;
micButton.addEventListener('click', async () => {
  if (micButton.classList.contains('working')) return;
  if (recording) return stopRecording();
  try {
    await startRecording();
  } catch (error) {
    onError(`Microphone unavailable: ${error.message}`);
  }
});

// ---- Read-aloud ----
let playing = null;

function stopPlayback() {
  if (!playing) return;
  playing.abort.abort();
  playing.player.then(({context}) => context.close()).catch(() => {});
  playing.button.innerHTML = SPEAKER_ICON;
  playing.button.classList.remove('active');
  playing = null;
}

// Prose only: code blocks are skipped, as a listener cannot follow them.
function readableText(body) {
  const copy = body.cloneNode(true);
  for (const code of copy.querySelectorAll('pre, .mermaid')) code.remove();
  return copy.innerText.replace(/\s+\n/g, '\n').trim();
}

async function readAloud(button, body) {
  const wasPlaying = playing?.button === button;
  stopPlayback();
  if (wasPlaying) return;
  const text = readableText(body);
  if (!text) return;
  // Opened inside the click gesture so that the audio context may start.
  const current = {button, abort: new AbortController(), player: openPlayer()};
  playing = current;
  button.innerHTML = STOP_ICON;
  button.classList.add('active');
  try {
    const {context, node} = await current.player;
    node.port.onmessage = ({data}) => {
      if (data.type === 'drained' && playing === current) stopPlayback();
    };
    node.port.postMessage({type: 'start', streamId: 0});
    await call('speech.speak', {text}, {
      signal: current.abort.signal,
      onEvent: ({type, payload}) => {
        if (type !== 'audio' || playing !== current) return;
        const pcm = new Int16Array(base64ToBytes(payload.data).buffer);
        const samples = resample(pcm16ToFloat32(pcm), payload.sample_rate, context.sampleRate);
        node.port.postMessage({type: 'chunk', streamId: 0, samples: samples.buffer}, [samples.buffer]);
      },
    });
    if (playing === current) node.port.postMessage({type: 'end', streamId: 0});
  } catch (error) {
    if (playing === current) {
      stopPlayback();
      onError(`Read-aloud failed: ${error.message}`);
    }
  }
}

// A read-aloud button for a reply's rendered Markdown; hidden when the server has no TTS.
export function readAloudButton(body) {
  const button = document.createElement('button');
  button.type = 'button';
  button.className = 'icon read-aloud';
  button.title = 'Read aloud (the first use downloads and loads OmniVoice)';
  button.innerHTML = SPEAKER_ICON;
  button.hidden = !available.speak;
  button.addEventListener('click', () => readAloud(button, body));
  return button;
}

export function setupSpeech(speech, errorHandler) {
  available = {transcribe: !!speech?.transcribe, speak: !!speech?.speak};
  onError = errorHandler;
  micButton.hidden = !available.transcribe;
  for (const button of document.querySelectorAll('.read-aloud')) button.hidden = !available.speak;
}
