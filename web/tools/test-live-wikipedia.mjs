import assert from 'node:assert/strict';
import {spawn} from 'node:child_process';
import {cp, mkdtemp, readFile, realpath, rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {fileURLToPath} from 'node:url';

import {chromium} from 'playwright';

const root = fileURLToPath(new URL('../../', import.meta.url));
const source = join(root, 'build-extension/chromium');
const articleUrl = 'https://en.wikipedia.org/wiki/Artificial_intelligence';
const edge = '/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge';
const keepOpen = process.env.KIDI_E2E_KEEP_OPEN === '1';
const stepDelay = Math.max(0, Number(process.env.KIDI_E2E_STEP_DELAY_MS) || 0);
const delay = milliseconds => new Promise(resolve => setTimeout(resolve, milliseconds));

async function waitFor(check, message, timeout = 30000) {
  const deadline = Date.now() + timeout;
  while (Date.now() < deadline) {
    const value = await check();
    if (value) return value;
    await delay(100);
  }
  throw new Error(message);
}

const temporary = await mkdtemp(join(tmpdir(), 'kidi-live-wikipedia-'));
const extension = join(temporary, 'extension');
const profile = join(temporary, 'profile');
await cp(source, extension, {recursive: true});
await cp(join(root, 'tests/browser/fixtures/extension-backend.mjs'), join(extension, 'backend.js'));

const edgeProcess = spawn(edge, [
  `--user-data-dir=${profile}`,
  '--remote-debugging-port=0',
  '--enable-unsafe-extension-debugging',
  `--disable-extensions-except=${extension}`,
  `--load-extension=${extension}`,
  '--no-first-run',
  '--no-default-browser-check',
  '--new-window',
  articleUrl,
], {stdio: 'ignore'});

let browser, browserCdp, sessionId, closed = false, succeeded = false;
const exited = new Promise(resolve => edgeProcess.once('exit', (code, signal) => {
  closed = true;
  resolve({code, signal});
}));

try {
  const portFile = join(profile, 'DevToolsActivePort');
  const port = await waitFor(async () => {
    try { return (await readFile(portFile, 'utf8')).split(/\r?\n/u)[0]; }
    catch { return null; }
  }, 'Edge did not expose its DevTools endpoint');
  browser = await chromium.connectOverCDP(`http://127.0.0.1:${port}`);
  browserCdp = await browser.newBrowserCDPSession();
  const context = browser.contexts()[0];
  assert.ok(context, 'Edge did not expose its persistent browser context');
  const article = await waitFor(
    () => context.pages().find(page => page.url() === articleUrl),
    'Edge did not open the Wikipedia article',
  );
  await article.locator('h1').first().waitFor({state: 'visible', timeout: 60000});
  const articleErrors = [];
  article.on('pageerror', error => articleErrors.push(error.message));

  const extensions = await browserCdp.send('Extensions.getExtensions');
  const installed = extensions.extensions.find(item => item.name === 'Kidi AI');
  assert.ok(installed?.id, 'Kidi AI is not loaded in Edge');
  assert.equal(await realpath(installed.path), await realpath(extension));
  const extensionUrl = `chrome-extension://${installed.id}/editor.html`;
  const targets = await browserCdp.send('Target.getTargets', {filter: [{}]});
  const articleTarget = targets.targetInfos.find(target => target.type === 'tab' && target.url === articleUrl);
  assert.ok(articleTarget, 'Edge did not expose the Wikipedia tab target');
  await browserCdp.send('Extensions.triggerAction', {
    id: installed.id, targetId: articleTarget.targetId,
  });
  await delay(stepDelay);

  const panelTarget = await waitFor(async () => {
    const current = await browserCdp.send('Target.getTargets', {filter: [{}]});
    return current.targetInfos.find(target => target.type === 'page' && target.url === extensionUrl);
  }, 'Kidi toolbar action did not open its side panel');
  ({sessionId} = await browserCdp.send('Target.attachToTarget', {
    targetId: panelTarget.targetId, flatten: false,
  }));

  let command = 0;
  const pending = new Map();
  const panelErrors = [];
  browserCdp.on('Target.receivedMessageFromTarget', event => {
    if (event.sessionId !== sessionId) return;
    const message = JSON.parse(event.message);
    if (message.id && pending.has(message.id)) {
      const {resolve, reject, timer} = pending.get(message.id);
      pending.delete(message.id);
      clearTimeout(timer);
      if (message.error) reject(new Error(message.error.message));
      else resolve(message.result || {});
    } else if (message.method === 'Runtime.exceptionThrown') {
      panelErrors.push(message.params?.exceptionDetails?.text || 'Uncaught side-panel exception');
    }
  });
  const send = (method, params = {}) => new Promise((resolve, reject) => {
    const id = ++command;
    const timer = setTimeout(() => {
      pending.delete(id);
      reject(new Error(`${method} timed out`));
    }, 30000);
    pending.set(id, {resolve, reject, timer});
    browserCdp.send('Target.sendMessageToTarget', {
      sessionId, message: JSON.stringify({id, method, params}),
    }).catch(error => {
      clearTimeout(timer);
      pending.delete(id);
      reject(error);
    });
  });
  const evaluate = async expression => {
    const result = await send('Runtime.evaluate', {expression, awaitPromise: true, returnByValue: true});
    if (result.exceptionDetails) throw new Error(result.exceptionDetails.text || 'Side-panel evaluation failed');
    return result.result?.value;
  };
  await send('Runtime.enable');
  await waitFor(
    () => evaluate(`document.readyState === 'complete' &&
      [...document.querySelectorAll('button')].some(button => button.textContent.trim() === 'Summarize page')`),
    'The Summarize page action did not become available',
  );
  await evaluate(`(() => {
    globalThis.__kidiProgress = [];
    new MutationObserver(() => {
      for (const element of document.querySelectorAll('.turn-status'))
        if (element.textContent && !globalThis.__kidiProgress.includes(element.textContent))
          globalThis.__kidiProgress.push(element.textContent);
    }).observe(document.body, {subtree: true, childList: true, characterData: true});
    const button = [...document.querySelectorAll('button')]
      .find(item => item.textContent.trim() === 'Summarize page');
    button.click();
    return true;
  })()`);
  await delay(stepDelay);
  try {
    await waitFor(
      () => evaluate(`document.querySelectorAll('.turn.assistant .result-actions').length === 1`),
      'Wikipedia summary did not complete',
      60000,
    );
  } catch (error) {
    const diagnosis = await evaluate(`(async () => {
      const requests = (await import('./backend.js')).backend.requests;
      return {
        notice: document.querySelector('#notice').textContent,
        contextStatus: document.querySelector('#context-status').textContent,
        contextLabel: document.querySelector('#context-label').textContent,
        runtime: document.querySelector('#runtime-status-label').textContent,
        setupHidden: document.querySelector('#setup').hidden,
        turns: [...document.querySelectorAll('.turn')].map(turn => ({
          role: turn.className, text: turn.querySelector('.turn-text')?.textContent,
          status: turn.querySelector('.turn-status')?.textContent,
        })),
        requests: requests.map(request => ({
          type: request.type, id: request.id, mode: request.payload?.mode,
          textBytes: typeof request.payload?.text === 'string'
            ? new TextEncoder().encode(request.payload.text).length : 0,
        })),
      };
    })()`);
    throw new Error(`${error.message}: ${JSON.stringify({...diagnosis, panelErrors})}`);
  }
  const result = await evaluate(`(async () => {
    const requests = (await import('./backend.js')).backend.requests;
    const request = requests.find(item => item.type === 'document.run');
    return {
      notice: document.querySelector('#notice').textContent,
      context: document.querySelector('#context-label').textContent,
      warning: document.querySelector('#context-warning').textContent,
      progress: globalThis.__kidiProgress,
      summary: document.querySelector('.turn.assistant .turn-text:last-child')?.textContent ||
        [...document.querySelectorAll('.turn.assistant .turn-text')].at(-1)?.textContent,
      request,
    };
  })()`);
  assert.equal(result.request?.payload?.mode, 'summarize');
  assert.match(result.request.payload.text, /artificial intelligence/iu);
  assert.equal(result.notice, 'Completed on this device.');
  assert.match(result.context, /Artificial intelligence - Wikipedia/);
  assert.match(result.summary, /local-first approach/);
  assert.deepEqual(articleErrors, []);
  assert.deepEqual(panelErrors, []);
  const report = {
    browser: 'Microsoft Edge',
    extensionId: installed.id,
    activeTab: true,
    source: {
      url: articleUrl,
      bytes: new TextEncoder().encode(result.request.payload.text).length,
      truncated: Boolean(result.warning),
      warning: result.warning,
    },
    progress: result.progress,
    summary: result.summary,
    profile,
    devToolsPort: Number(port),
    keptOpen: keepOpen,
  };
  console.log(JSON.stringify(report, null, 2));
  succeeded = true;
  if (keepOpen) {
    edgeProcess.unref();
    process.exit(0);
  }
} finally {
  if (!keepOpen || !succeeded) {
    if (browserCdp) await browserCdp.send('Browser.close').catch(() => {});
    await Promise.race([exited, delay(3000)]);
    if (!closed) edgeProcess.kill('SIGTERM');
    await rm(temporary, {recursive: true, force: true});
  }
}
