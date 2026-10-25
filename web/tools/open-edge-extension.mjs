import assert from 'node:assert/strict';
import {spawn} from 'node:child_process';
import {mkdir, readFile, realpath} from 'node:fs/promises';
import {homedir} from 'node:os';
import {join, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';

import {chromium} from 'playwright';

const root = fileURLToPath(new URL('../../', import.meta.url));
const extension = resolve(process.argv[2] || join(root, 'build-extension/chromium'));
const profile = resolve(process.env.KIDI_EDGE_PROFILE || join(homedir(), '.cache/kidi/edge-extension-profile'));
const startUrl = process.env.KIDI_EDGE_START_URL ||
  'https://en.wikipedia.org/wiki/Artificial_intelligence';
const edge = '/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge';
const forceReload = process.env.KIDI_EDGE_RELOAD === '1';
const modelHubKey = 'kidi-model-hub-remote-v1';
const localCache = process.env.KIDI_LOCAL_MODEL_CACHE ?
  resolve(process.env.KIDI_LOCAL_MODEL_CACHE.replace(/^~(?=\/)/u, homedir())) : '';
const localHub = process.env.KIDI_LOCAL_MODEL_HUB || 'http://127.0.0.1:8765';
const localHubUrl = new URL(localHub);
if (localCache && (localHubUrl.protocol !== 'http:' || localHubUrl.hostname !== '127.0.0.1' ||
    (localHubUrl.pathname !== '/' && localHubUrl.pathname !== '') || localHubUrl.search || localHubUrl.hash))
  throw new Error('KIDI_LOCAL_MODEL_HUB must be an IPv4 loopback HTTP origin');
const delay = milliseconds => new Promise(resolve => setTimeout(resolve, milliseconds));

async function localBridge() {
  if (!localCache) return null;
  const expected = await realpath(localCache);
  const healthy = async () => {
    try {
      const response = await fetch(`${localHub}/__kidi_health__`, {signal: AbortSignal.timeout(1000)});
      if (!response.ok) return false;
      const status = await response.json();
      if (status.root !== expected) throw new Error(`Local model bridge serves ${status.root}, expected ${expected}`);
      return true;
    } catch (error) {
      if (error.message?.startsWith('Local model bridge serves')) throw error;
      return false;
    }
  };
  if (!await healthy()) {
    const server = spawn(process.execPath, [
      fileURLToPath(new URL('./serve-model-cache.mjs', import.meta.url)),
      '--root', expected, '--host', '127.0.0.1', '--port', localHubUrl.port || '80',
    ], {detached: true, stdio: 'ignore'});
    server.unref();
    const deadline = Date.now() + 10000;
    while (!await healthy() && Date.now() < deadline) await delay(100);
    if (!await healthy()) throw new Error('Local model bridge did not become ready');
  }
  return {url: localHub, root: expected};
}

async function endpoint() {
  try {
    const [port] = (await readFile(join(profile, 'DevToolsActivePort'), 'utf8')).split(/\r?\n/u);
    const url = `http://127.0.0.1:${port}`;
    const response = await fetch(`${url}/json/version`, {signal: AbortSignal.timeout(1000)});
    return response.ok ? url : null;
  } catch {
    return null;
  }
}

const bridge = await localBridge();
await mkdir(profile, {recursive: true});
let devTools = await endpoint();
const reused = Boolean(devTools);
if (!devTools) {
  const child = spawn(edge, [
    `--user-data-dir=${profile}`,
    '--remote-debugging-port=0',
    '--enable-unsafe-extension-debugging',
    `--disable-extensions-except=${extension}`,
    `--load-extension=${extension}`,
    '--no-first-run',
    '--no-default-browser-check',
    '--new-window',
    startUrl,
  ], {detached: true, stdio: 'ignore'});
  child.unref();
  const deadline = Date.now() + 15000;
  while (!devTools && Date.now() < deadline) {
    await delay(100);
    devTools = await endpoint();
  }
}
assert.ok(devTools, 'Microsoft Edge did not expose its debugging endpoint');

const browser = await chromium.connectOverCDP(devTools);
const cdp = await browser.newBrowserCDPSession();
let extensions = await cdp.send('Extensions.getExtensions');
let installed = extensions.extensions.find(item => item.name === 'Kidi AI');
if (!installed) {
  const loaded = await cdp.send('Extensions.loadUnpacked', {path: extension});
  extensions = await cdp.send('Extensions.getExtensions');
  installed = extensions.extensions.find(item => item.id === loaded.id);
}
assert.ok(installed?.enabled, 'Kidi AI is not enabled in the stable Edge profile');
assert.equal(await realpath(installed.path), await realpath(extension));

const context = browser.contexts()[0];
assert.ok(context, 'Microsoft Edge did not expose its persistent context');
let settingsPage = context.pages().find(item =>
  item.url().startsWith(`chrome-extension://${installed.id}/`));
const temporarySettingsPage = !settingsPage;
if (!settingsPage) {
  settingsPage = await context.newPage();
  await settingsPage.goto(`chrome-extension://${installed.id}/editor.html`, {waitUntil: 'domcontentloaded'});
}
const previousHub = await settingsPage.evaluate(
  async key => (await chrome.storage.local.get(key))[key] || null, modelHubKey);
if (bridge)
  await settingsPage.evaluate(async ({key, value}) => chrome.storage.local.set({[key]: value}),
    {key: modelHubKey, value: bridge.url});
else
  await settingsPage.evaluate(async key => chrome.storage.local.remove(key), modelHubKey);
const configuredHub = bridge?.url || null;
if (reused && (previousHub !== configuredHub || forceReload)) {
  await settingsPage.evaluate(() => chrome.runtime.reload()).catch(() => {});
  await delay(500);
} else if (temporarySettingsPage) await settingsPage.close();
let page = context.pages().find(item => item.url() === startUrl);
if (!page) {
  page = await context.newPage();
  await page.goto(startUrl, {waitUntil: 'domcontentloaded', timeout: 60000});
}
await page.bringToFront();

console.log(JSON.stringify({
  browser: 'Microsoft Edge',
  profile,
  reused,
  extensionId: installed.id,
  extension: installed.path,
  url: page.url(),
  modelCache: 'Persistent in this profile across launches',
  modelHub: bridge ? {mode: 'local', ...bridge} : {mode: 'remote', url: 'https://huggingface.co'},
}, null, 2));
process.exit(0);
