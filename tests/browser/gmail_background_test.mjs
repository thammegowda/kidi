import assert from 'node:assert/strict';
import test from 'node:test';
import {readFile} from 'node:fs/promises';
import {webcrypto} from 'node:crypto';
import vm from 'node:vm';

class ExtensionEvent {
  listeners = [];
  addListener(listener) { this.listeners.push(listener); }
  fire(...args) { for (const listener of this.listeners) listener(...args); }
}
async function waitFor(predicate) {
  for (let count = 0; count < 100; count++) {
    if (predicate()) return;
    await new Promise(resolve => setImmediate(resolve));
  }
  throw new Error('Timed out waiting for extension state');
}
async function background() {
  const source = await readFile(new URL('../../web/extension/background.mjs', import.meta.url), 'utf8');
  const storage = {}, registered = new Map(), notifications = [], injections = [], sent = [], steps = [];
  const tabs = new Map([
    [5, {id: 5, windowId: 7, url: 'https://mail.google.com/mail/u/0/#inbox/thread-one', title: 'First Gmail'}],
    [6, {id: 6, windowId: 8, url: 'https://example.org/article', title: 'Another window'}],
    [9, {id: 9, windowId: 7, url: 'https://mail.google.com.attacker.invalid/', title: 'Not Gmail'}],
  ]);
  const state = {enabled: false};
  const api = {
    runtime: {
      id: 'test-extension',
      getURL: path => `chrome-extension://test-extension/${path}`,
      onInstalled: new ExtensionEvent(), onStartup: new ExtensionEvent(), onMessage: new ExtensionEvent(),
      sendMessage: async message => { notifications.push(message); },
    },
    action: {onClicked: new ExtensionEvent()},
    sidePanel: {
      open: async ({tabId}) => { steps.push(`open:${tabId}`); },
      setPanelBehavior: async () => { steps.push('automatic-panel-behavior'); },
    },
    contextMenus: {
      onClicked: new ExtensionEvent(),
      removeAll: async () => {},
      create() {},
    },
    permissions: {
      contains: async () => state.enabled,
      onAdded: new ExtensionEvent(), onRemoved: new ExtensionEvent(),
    },
    storage: {session: {
      get: async key => structuredClone({[key]: storage[key]}),
      set: async value => { Object.assign(storage, structuredClone(value)); },
    }},
    scripting: {
      getRegisteredContentScripts: async () => [...registered.values()],
      registerContentScripts: async descriptors => {
        for (const item of descriptors) registered.set(item.id, item);
      },
      unregisterContentScripts: async ({ids}) => { for (const id of ids) registered.delete(id); },
      executeScript: async request => { injections.push(request); steps.push(`inject:${request.target.tabId}`); },
    },
    tabs: {
      get: async id => {
        if (!tabs.has(id)) throw new Error('Tab closed');
        return tabs.get(id);
      },
      query: async query => [...tabs.values()].filter(tab => {
        if (query.url) return tab.url.startsWith('https://mail.google.com/');
        if (query.windowId !== undefined) return tab.windowId === query.windowId && tab.id !== 9;
        return tab.id === 5;
      }),
      create: async () => {},
      sendMessage: async (tabId, message, options) => {
        sent.push({tabId, message, options});
        if (message.type === 'kidi.page.capture')
          return state.captureError ? {ok: false, error: state.captureError}
            : {ok: true, kind: 'article', title: 'An article', text: 'Readable source paragraphs.',
              language: 'en', sourceUrl: state.captureUrl || tabs.get(tabId).url, operation: null};
        if (message.type === 'kidi.gmail.inspect')
          return {ok: true, drafts: [{id: 'draft-one', title: 'Launch plan'}], activeDraftId: 'draft-one'};
        if (message.type === 'kidi.gmail.capture')
          return {ok: true, kind: 'gmail-draft', title: 'Launch plan', text: 'Private draft text',
            operation: null};
        if (message.type === 'liteapp.selection.capture')
          return {ok: true, text: `Selection from ${tabId}`, editable: true};
        return {ok: true};
      },
      onActivated: new ExtensionEvent(), onUpdated: new ExtensionEvent(), onRemoved: new ExtensionEvent(),
    },
  };
  const sandbox = vm.createContext({
    chrome: api, URL, TextEncoder, Date, crypto: webcrypto,
    console: {error() {}, warn() {}, debug() {}},
  });
  vm.runInContext(source, sandbox, {filename: 'background.mjs'});
  const send = (message, sender = {
    id: api.runtime.id, url: api.runtime.getURL('editor.html'),
  }) => new Promise(resolve => {
    const handled = api.runtime.onMessage.listeners[0](message, sender,
      value => resolve(JSON.parse(JSON.stringify(value))));
    if (!handled) resolve(undefined);
  });
  return {api, send, tabs, storage, state, registered, notifications, injections, sent, steps};
}

test('Gmail bridge activation requires optional permission and an exact Gmail origin', async () => {
  const env = await background();
  const denied = await env.send({type: 'kidi.context.inspect', windowId: 7});
  assert.equal(denied.onGmail, true);
  assert.equal(denied.gmailEnabled, false);
  assert.equal(env.injections.length, 0);
  const capture = await env.send({type: 'kidi.gmail.capture', windowId: 7, kind: 'draft'});
  assert.equal(capture.ok, false);
  assert.equal(env.injections.length, 0);
  env.state.enabled = true;
  env.api.permissions.onAdded.fire({origins: ['https://mail.google.com/*']});
  await waitFor(() => env.registered.size === 1 && env.injections.length > 0);
  assert.equal(env.registered.get('kidi-gmail').js[0], 'gmail-bridge.js');
  assert.equal(env.sent.some(item => item.message.type === 'kidi.gmail.capture'), false,
    'granting access observes availability, not email bodies');
  const wrongOrigin = await env.send({type: 'kidi.gmail.capture', tabId: 9, windowId: 7, kind: 'draft'});
  assert.equal(wrongOrigin.ok, false);
  assert.equal(env.injections.some(request => request.target.tabId === 9), false);
});

test('toolbar action explicitly opens the panel without Edge automatic side-panel behavior', async () => {
  const env = await background();
  env.api.runtime.onInstalled.fire();
  await waitFor(() => env.notifications.length > 0);
  assert.equal(env.steps.includes('automatic-panel-behavior'), false);
  env.api.action.onClicked.fire(env.tabs.get(6));
  await waitFor(() => env.steps.includes('open:6'));
});

test('Gmail capture rejects webpage callers and remains copy-only after permission revocation', async () => {
  const env = await background();
  env.state.enabled = true;
  const external = await env.send({type: 'kidi.gmail.capture', windowId: 7},
    {id: env.api.runtime.id, url: 'https://mail.google.com/', tab: env.tabs.get(5)});
  assert.equal(external, undefined);
  const captured = await env.send({
    type: 'kidi.gmail.capture', windowId: 7, kind: 'draft', draftId: 'draft-one',
  });
  assert.equal(captured.ok, true);
  assert.equal(captured.task.text, 'Private draft text');
  env.state.enabled = false;
  env.api.permissions.onRemoved.fire({origins: ['https://mail.google.com/*']});
  await waitFor(() => Object.values(env.storage['kidi-extension-tasks-v2']?.tasks || {}).length === 0);
  const sent = env.sent.length;
  const result = await env.send({type: 'kidi.context.apply', id: captured.task.id, text: 'Generated text'});
  assert.equal(result.ok, false);
  assert.equal(env.sent.length, sent);
  assert.equal(await env.send({type: 'unknown.message'}), undefined);
});

test('selection requests are window scoped and open the panel before asynchronous capture', async () => {
  const env = await background();
  env.api.contextMenus.onClicked.fire({menuItemId: 'kidi-polish', selectionText: 'one'}, env.tabs.get(5));
  env.api.contextMenus.onClicked.fire({menuItemId: 'kidi-polish', selectionText: 'two'}, env.tabs.get(6));
  await waitFor(() => Object.keys(env.storage['kidi-extension-tasks-v2']?.pending || {}).length === 2);
  assert.ok(env.steps.indexOf('open:5') < env.steps.indexOf('inject:5'));
  const one = await env.send({type: 'kidi.pending.take', windowId: 7});
  const two = await env.send({type: 'kidi.pending.take', windowId: 8});
  assert.equal(one.task.text, 'Selection from 5');
  assert.equal(two.task.text, 'Selection from 6');
  assert.notEqual(one.task.target, two.task.target);
  assert.equal((await env.send({type: 'kidi.pending.take', windowId: 7})).task, null);
  env.tabs.get(5).url = 'https://mail.google.com/mail/u/1/#inbox/another-account';
  const apply = await env.send({type: 'kidi.context.apply', id: one.task.id, text: 'Replacement'});
  assert.equal(apply.ok, false);
  assert.match(apply.error, /source page changed/);
  assert.equal(env.sent.some(item => item.message.type === 'liteapp.selection.replace'), false);
});

test('page capture injects only packaged extraction scripts and stays read-only', async () => {
  const env = await background();
  const captured = await env.send({type: 'kidi.page.capture', tabId: 6, windowId: 8, task: 'page_translate'});
  assert.equal(captured.ok, true);
  assert.equal(captured.task.task, 'page_translate');
  assert.equal(captured.task.kind, 'article');
  assert.equal(captured.task.operation, null);
  assert.deepEqual([...env.injections[0].files], ['libs/readability/Readability.js', 'article-bridge.js']);
  assert.equal((await env.send({type: 'kidi.context.apply', id: captured.task.id, text: 'Replacement'})).ok, false);
  assert.equal((await env.send({type: 'kidi.context.inspect', tabId: 6, windowId: 8})).pageAvailable, true);
  assert.equal((await env.send({type: 'kidi.page.capture', tabId: 5, windowId: 7})).ok, false);
  assert.equal(await env.send({type: 'kidi.page.capture', tabId: 6},
    {id: env.api.runtime.id, url: env.tabs.get(6).url, tab: env.tabs.get(6)}), undefined);
});

test('page capture rejects navigation and restricted protocols with actionable errors', async () => {
  const env = await background();
  env.state.captureUrl = 'https://example.org/a-different-page';
  assert.match((await env.send({type: 'kidi.page.capture', tabId: 6})).error, /navigated/);
  env.tabs.get(6).url = 'file:///private/document.html';
  assert.match((await env.send({type: 'kidi.page.capture', tabId: 6})).error, /Only HTTP and HTTPS/);
  delete env.tabs.get(6).url;
  assert.match((await env.send({type: 'kidi.page.capture', tabId: 6})).error, /toolbar.*temporary page access/);
});

test('page menu failures survive panel startup, and success clears the old error', async () => {
  const env = await background();
  env.state.captureError = 'No readable article was found';
  env.api.contextMenus.onClicked.fire({menuItemId: 'kidi-page-summary'}, env.tabs.get(6));
  await waitFor(() => env.storage['kidi-extension-tasks-v2']?.errors?.['8']);
  assert.ok(env.steps.indexOf('open:6') < env.steps.indexOf('inject:6'));
  const error = await env.send({type: 'kidi.pending.take', windowId: 8});
  assert.equal(error.ok, false);
  assert.match(error.error, /No readable article/);
  assert.equal((await env.send({type: 'kidi.pending.take', windowId: 8})).task, null);
  env.state.captureError = '';
  env.api.contextMenus.onClicked.fire({menuItemId: 'kidi-page-read'}, env.tabs.get(6));
  await waitFor(() => env.storage['kidi-extension-tasks-v2']?.pending?.['8']);
  const pending = await env.send({type: 'kidi.pending.take', windowId: 8});
  assert.equal(pending.task.task, 'page_read');
  assert.equal((await env.send({type: 'kidi.pending.take', windowId: 7})).task, null);
});
