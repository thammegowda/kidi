import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {homedir} from 'node:os';
import {join, resolve} from 'node:path';

import {chromium} from 'playwright';

const profile = resolve(process.env.KIDI_EDGE_PROFILE || join(homedir(), '.cache/kidi/edge-extension-profile'));
const timeout = Number(process.env.KIDI_MODEL_LOAD_TIMEOUT_MS) || 10 * 60 * 1000;
const runSummary = process.env.KIDI_TEST_SUMMARY === '1';
const translateLanguage = process.env.KIDI_TEST_TRANSLATE || '';
const sourceUrl = process.env.KIDI_EDGE_SOURCE_URL || '';
const delay = milliseconds => new Promise(resolveDelay => setTimeout(resolveDelay, milliseconds));
const [port] = (await readFile(join(profile, 'DevToolsActivePort'), 'utf8')).split(/\r?\n/u);
const browser = await chromium.connectOverCDP(`http://127.0.0.1:${port}`);
const cdp = await browser.newBrowserCDPSession();
const extensions = await cdp.send('Extensions.getExtensions');
const installed = extensions.extensions.find(item => item.name === 'Kidi AI');
assert.ok(installed?.enabled, 'Kidi AI is not enabled in the stable Edge profile');
const targets = await cdp.send('Target.getTargets', {filter: [{}]});
const source = targets.targetInfos.find(target => target.type === 'tab' &&
  (sourceUrl ? target.url === sourceUrl : /^https?:/u.test(target.url)));
assert.ok(source, 'Open an HTTP or HTTPS page before testing the extension');
await cdp.send('Extensions.triggerAction', {id: installed.id, targetId: source.targetId});

const editorUrl = `chrome-extension://${installed.id}/editor.html`;
let panel;
const panelDeadline = Date.now() + 15000;
while (!panel && Date.now() < panelDeadline) {
  const current = await cdp.send('Target.getTargets', {filter: [{}]});
  panel = current.targetInfos.find(target => target.type === 'page' && target.url === editorUrl);
  if (!panel) await delay(100);
}
assert.ok(panel, 'Kidi side panel did not open');
const {sessionId} = await cdp.send('Target.attachToTarget', {targetId: panel.targetId, flatten: false});
let sequence = 0;
const pending = new Map();
cdp.on('Target.receivedMessageFromTarget', event => {
  if (event.sessionId !== sessionId) return;
  const message = JSON.parse(event.message);
  if (!message.id || !pending.has(message.id)) return;
  const request = pending.get(message.id);
  pending.delete(message.id);
  clearTimeout(request.timer);
  if (message.error) request.reject(new Error(message.error.message));
  else request.resolve(message.result || {});
});
const send = (method, params = {}) => new Promise((resolveSend, reject) => {
  const id = ++sequence;
  const timer = setTimeout(() => {
    pending.delete(id);
    reject(new Error(`${method} timed out`));
  }, 30000);
  pending.set(id, {resolve: resolveSend, reject, timer});
  cdp.send('Target.sendMessageToTarget', {
    sessionId, message: JSON.stringify({id, method, params}),
  }).catch(error => {
    pending.delete(id);
    clearTimeout(timer);
    reject(error);
  });
});
const evaluate = async expression => {
  const result = await send('Runtime.evaluate', {expression, awaitPromise: true, returnByValue: true});
  if (result.exceptionDetails) throw new Error(result.exceptionDetails.text || 'Panel evaluation failed');
  return result.result?.value;
};
await send('Runtime.enable');
const state = () => evaluate(`(async () => {
  const {backend} = await import('./backend.js');
  return {
    snapshot: await backend.status(),
    status: document.querySelector('#runtime-status-label')?.textContent || '',
    notice: document.querySelector('#notice')?.textContent || '',
    noticeError: document.querySelector('#notice')?.classList.contains('error') || false,
    loadDisabled: document.querySelector('#load-model')?.disabled || false,
    progressHidden: document.querySelector('#model-progress')?.hidden ?? true,
    stage: document.querySelector('#load-stage')?.textContent || '',
    percent: document.querySelector('#load-percent')?.textContent || '',
    bytes: document.querySelector('#load-bytes')?.textContent || '',
    transfer: document.querySelector('#load-transfer')?.textContent || '',
    warning: document.querySelector('#load-warning')?.textContent || '',
    diagnostics: document.querySelector('#diagnostics')?.textContent || '',
  };
})()`);

let current = await state();
if (!current.snapshot.ready) {
  await evaluate(`document.querySelector('#load-model').click()`);
  const started = Date.now();
  let previous = '';
  while (Date.now() - started < timeout) {
    current = await state();
    const status = JSON.stringify({
      status: current.status, notice: current.notice, stage: current.stage,
      percent: current.percent, bytes: current.bytes, transfer: current.transfer,
    });
    if (status !== previous) {
      console.log(status);
      previous = status;
    }
    if (current.snapshot.ready) break;
    if (current.noticeError && !current.loadDisabled) throw new Error(current.notice || 'Model loading failed');
    await delay(500);
  }
  if (!current.snapshot.ready) throw new Error(`Model load timed out after ${timeout} ms`);
}
assert.equal(current.snapshot.runtime.model_hub_remote, 'http://127.0.0.1:8765');
const report = {
  ready: true,
  backend: current.snapshot.backend,
  model: current.snapshot.runtime.model_id,
  modelHub: current.snapshot.runtime.model_hub_remote,
  memory: current.snapshot.runtime.memory,
  diagnostics: current.diagnostics,
  finalStatus: current.status,
  warning: current.warning,
};
if (runSummary || translateLanguage) {
  await evaluate(`document.querySelector('#new-chat').click()`);
  await delay(250);
  if (runSummary) {
    await evaluate(`(() => {
      const button = [...document.querySelectorAll('button')]
        .find(item => item.textContent.trim() === 'Summarize page');
      if (!button || button.disabled) throw new Error('Summarize page is unavailable');
      button.click();
    })()`);
  } else {
    await evaluate(`(() => {
      const button = [...document.querySelectorAll('button')]
        .find(item => item.textContent.trim() === 'Translate page');
      if (!button || button.disabled) throw new Error('Translate page is unavailable');
      button.click();
    })()`);
    const language = JSON.stringify(translateLanguage);
    const deadline = Date.now() + 10000;
    let ready = false;
    while (!ready && Date.now() < deadline) {
      ready = await evaluate(`!document.querySelector('#translation-tools').hidden &&
        [...document.querySelector('#target-language').options].some(option => option.value === ${language})`);
      if (!ready) await delay(100);
    }
    if (!ready) throw new Error(`Translation language is unavailable: ${translateLanguage}`);
    await evaluate(`(() => {
      const select = document.querySelector('#target-language');
      select.value = ${language};
      select.dispatchEvent(new Event('change', {bubbles: true}));
      document.querySelector('#send').click();
    })()`);
  }
  const started = Date.now();
  let previous = '';
  let result;
  while (Date.now() - started < timeout) {
    result = await evaluate(`(async () => {
      const {backend} = await import('./backend.js');
      const turns = [...document.querySelectorAll('.turn.assistant')];
      return {
        snapshot: await backend.status(),
        notice: document.querySelector('#notice')?.textContent || '',
        noticeError: document.querySelector('#notice')?.classList.contains('error') || false,
        status: turns.at(-1)?.querySelector('.turn-status')?.textContent || '',
        text: turns.at(-1)?.querySelector('.turn-text')?.textContent || '',
        completed: Boolean(turns.at(-1)?.querySelector('.result-actions')),
      };
    })()`);
    const status = JSON.stringify({notice: result.notice, status: result.status,
      characters: result.text.length, memory: result.snapshot.runtime.memory});
    if (status !== previous) {
      console.log(status);
      previous = status;
    }
    if (result.completed) break;
    if (result.noticeError) throw new Error(`${result.notice}; ${result.status}`);
    await delay(500);
  }
  const operation = runSummary ? 'summary' : 'translation';
  if (!result?.completed) throw new Error(`Wikipedia ${operation} timed out after ${timeout} ms`);
  report[operation] = {
    elapsed_ms: Date.now() - started,
    characters: result.text.length,
    preview: result.text.slice(0, 300),
    memory: result.snapshot.runtime.memory,
  };
}
console.log(JSON.stringify(report, null, 2));
process.exit(0);
