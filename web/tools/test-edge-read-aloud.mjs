import {readFile} from 'node:fs/promises';
import {homedir} from 'node:os';
import {join, resolve} from 'node:path';

import {chromium} from 'playwright';

const profile = resolve(process.env.KIDI_EDGE_PROFILE || join(homedir(), '.cache/kidi/edge-extension-profile'));
const timeout = Number(process.env.KIDI_TTS_LOAD_TIMEOUT_MS) || 10 * 60 * 1000;
const delay = milliseconds => new Promise(resolveDelay => setTimeout(resolveDelay, milliseconds));
const [port] = (await readFile(join(profile, 'DevToolsActivePort'), 'utf8')).split(/\r?\n/u);
const browser = await chromium.connectOverCDP(`http://127.0.0.1:${port}`);
const cdp = await browser.newBrowserCDPSession();
const extensions = await cdp.send('Extensions.getExtensions');
const installed = extensions.extensions.find(item => item.name === 'Kidi AI');
const current = await cdp.send('Target.getTargets', {filter: [{}]});
const source = current.targetInfos.find(target => target.type === 'tab' && /^https?:/u.test(target.url));
if (!installed?.id || !source) throw new Error('Kidi or an active webpage is unavailable');
await cdp.send('Extensions.triggerAction', {id: installed.id, targetId: source.targetId});
let panel;
const panelDeadline = Date.now() + 15000;
while (!panel && Date.now() < panelDeadline) {
  const targets = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json();
  panel = targets.find(target => target.type === 'page' &&
    /^chrome-extension:\/\/[^/]+\/editor\.html$/u.test(target.url));
  if (!panel) await delay(100);
}
if (!panel?.webSocketDebuggerUrl) throw new Error('Kidi side panel target is unavailable');

const socket = new WebSocket(panel.webSocketDebuggerUrl);
let sequence = 0;
const pending = new Map();
await new Promise((resolveOpen, reject) => {
  socket.addEventListener('open', resolveOpen, {once: true});
  socket.addEventListener('error', () => reject(new Error('Unable to connect to the Edge panel')), {once: true});
});
socket.addEventListener('message', event => {
  const message = JSON.parse(event.data);
  if (!message.id || !pending.has(message.id)) return;
  const request = pending.get(message.id);
  pending.delete(message.id);
  clearTimeout(request.timer);
  if (message.error || message.result?.exceptionDetails) {
    const details = message.result?.exceptionDetails;
    request.reject(new Error(message.error?.message || details?.exception?.description ||
      details?.exception?.value || details?.text || 'Edge panel evaluation failed'));
  }
  else request.resolve(message.result.result?.value);
});
const evaluate = (expression, userGesture = false) => new Promise((resolveEvaluate, reject) => {
  const id = ++sequence;
  const timer = setTimeout(() => {
    pending.delete(id);
    reject(new Error('Edge read-aloud evaluation timed out'));
  }, 30000);
  pending.set(id, {resolve: resolveEvaluate, reject, timer});
  socket.send(JSON.stringify({
    id,
    method: 'Runtime.evaluate',
    params: {expression, awaitPromise: true, returnByValue: true, userGesture},
  }));
});

const actionDeadline = Date.now() + 10000;
let actionReady = false;
while (!actionReady && Date.now() < actionDeadline) {
  actionReady = await evaluate(`(() => {
    const button = [...document.querySelectorAll('button')]
      .find(item => item.textContent.trim() === 'Read article');
    return Boolean(button && !button.disabled);
  })()`);
  if (!actionReady) await delay(100);
}
if (!actionReady) throw new Error('Read article is unavailable');
await evaluate(`([...document.querySelectorAll('button')]
  .find(item => item.textContent.trim() === 'Read article')).click()`, true);
await delay(500);
if (await evaluate(`document.querySelector('#speech-dialog')?.open === true`)) {
  await evaluate(`(() => {
    const language = document.querySelector('#speech-language');
    language.value = 'en';
    language.dispatchEvent(new Event('change', {bubbles: true}));
    document.querySelector('#speech-form').requestSubmit();
    return true;
  })()`, true);
}

const started = Date.now();
let state, previous = '';
while (Date.now() - started < timeout) {
  state = await evaluate(`(async () => {
    const {backend} = await import('./backend.js');
    const snapshot = await backend.status();
    return {
      status: document.querySelector('#speech-status')?.textContent || '',
      playerVisible: !document.querySelector('#speech-player')?.hidden,
      pauseDisabled: document.querySelector('#speech-pause')?.disabled ?? true,
      dialogOpen: document.querySelector('#speech-dialog')?.open || false,
      notice: document.querySelector('#notice')?.textContent || '',
      noticeError: document.querySelector('#notice')?.classList.contains('error') || false,
      loaded: snapshot.components?.tts?.loaded === true,
      accelerator: snapshot.components?.tts?.accelerator,
      memory: snapshot.runtime?.memory,
      modelHub: snapshot.runtime?.model_hub_remote,
    };
  })()`);
  const visible = JSON.stringify(state);
  if (visible !== previous) {
    console.log(visible);
    previous = visible;
  }
  if (state.playerVisible && !state.pauseDisabled) break;
  if (state.noticeError) throw new Error(state.notice || state.status || 'Read aloud failed');
  await delay(500);
}
socket.close();
if (!state?.playerVisible || state.pauseDisabled)
  throw new Error(`Read aloud did not start within ${timeout} ms`);
console.log(JSON.stringify({playing: true, elapsed_ms: Date.now() - started, ...state}, null, 2));
process.exit(0);
