const api = globalThis.browser || globalThis.chrome;
const menus = api.contextMenus || api.menus;
const STORAGE_KEY = 'kidi-extension-tasks-v2';
const GMAIL_ORIGIN = 'https://mail.google.com/*';
const GMAIL_SCRIPT = 'kidi-gmail';
const TTL = 30 * 60 * 1000;
const COMMANDS = [
  {id: 'kidi-polish', title: 'Polish with Kidi', task: 'polish'},
  {id: 'kidi-explain', title: 'Explain with Kidi', task: 'explain'},
  {id: 'kidi-summarize', title: 'Summarize with Kidi', task: 'summarize'},
];
const PAGE_COMMANDS = [
  {id: 'kidi-page-summary', title: 'Summarize page with Kidi', task: 'page_summarize'},
  {id: 'kidi-page-translate', title: 'Translate page with Kidi', task: 'page_translate'},
  {id: 'kidi-page-read', title: 'Read article with Kidi', task: 'page_read'},
];
const UI_MESSAGES = new Set([
  'kidi.pending.take', 'kidi.context.inspect', 'kidi.gmail.capture',
  'kidi.page.capture', 'kidi.context.apply', 'kidi.context.discard',
]);
const lastTabs = new Map();
let transientState = {tasks: {}, pending: {}};
let mutations = Promise.resolve();

function isGmail(url) {
  try {
    const parsed = new URL(url);
    return parsed.protocol === 'https:' && parsed.hostname === 'mail.google.com' && !parsed.port;
  } catch { return false; }
}
function isWebPage(url) {
  try { return ['http:', 'https:'].includes(new URL(url).protocol); }
  catch { return false; }
}
function isPanel(sender) {
  return sender.id === api.runtime.id && typeof sender.url === 'string' &&
    sender.url.split(/[?#]/u)[0] === api.runtime.getURL('editor.html');
}
async function readState() {
  const state = api.storage.session
    ? (await api.storage.session.get(STORAGE_KEY))[STORAGE_KEY] || {tasks: {}, pending: {}}
    : transientState;
  for (const [id, task] of Object.entries(state.tasks))
    if (Date.now() - task.createdAt > TTL) delete state.tasks[id];
  for (const [windowId, error] of Object.entries(state.errors || {}))
    if (Date.now() - error.createdAt > TTL) delete state.errors[windowId];
  return state;
}
function updateState(update) {
  const operation = mutations.then(async () => {
    const state = await readState();
    const result = await update(state);
    const entries = Object.entries(state.tasks).sort((left, right) => right[1].createdAt - left[1].createdAt);
    state.tasks = Object.fromEntries(entries.slice(0, 16));
    if (api.storage.session) await api.storage.session.set({[STORAGE_KEY]: state});
    else transientState = state;
    return result;
  });
  mutations = operation.catch(error => { console.error('Kidi context storage failed:', error); });
  return operation;
}
function notify(message) {
  api.runtime.sendMessage(message).catch(error => {
    if (!/receiving end|connection|message port/iu.test(error.message || ''))
      console.error('Kidi panel notification failed:', error);
  });
}
async function openEditor(tab) {
  if (api.sidePanel?.open) await api.sidePanel.open({tabId: tab.id});
  else if (api.sidebarAction?.open) await api.sidebarAction.open();
  else await api.tabs.create({url: `${api.runtime.getURL('editor.html')}?tab=${tab.id}`});
}
async function currentTab(message = {}) {
  const windowId = Number.isInteger(message.windowId) ? message.windowId : undefined;
  let tab;
  if (Number.isInteger(message.tabId)) {
    tab = await api.tabs.get(message.tabId);
    if (windowId !== undefined && tab.windowId !== windowId)
      throw new Error('The requested tab belongs to another browser window.');
  } else {
    [tab] = await api.tabs.query(windowId === undefined
      ? {active: true, lastFocusedWindow: true} : {active: true, windowId});
    if (tab?.url?.startsWith(api.runtime.getURL(''))) {
      const previous = lastTabs.get(tab.windowId);
      tab = previous === undefined ? null : await api.tabs.get(previous);
    }
  }
  if (!tab?.id) throw new Error('Open a webpage to attach context.');
  return tab;
}
async function gmailEnabled() {
  return api.permissions.contains({origins: [GMAIL_ORIGIN]});
}
async function ensureGmail(tab) {
  if (!isGmail(tab.url) || !await gmailEnabled())
    throw new Error('Enable Gmail integration to access this email.');
  await api.scripting.executeScript({target: {tabId: tab.id, frameIds: [0]}, files: ['gmail-bridge.js']});
}
async function syncGmail() {
  const enabled = await gmailEnabled();
  const registered = await api.scripting.getRegisteredContentScripts({ids: [GMAIL_SCRIPT]});
  if (enabled && !registered.length) {
    await api.scripting.registerContentScripts([{
      id: GMAIL_SCRIPT, matches: [GMAIL_ORIGIN], js: ['gmail-bridge.js'],
      runAt: 'document_idle', persistAcrossSessions: true,
    }]);
  } else if (!enabled && registered.length) {
    await api.scripting.unregisterContentScripts({ids: [GMAIL_SCRIPT]});
  }
  if (enabled) {
    for (const tab of await api.tabs.query({url: GMAIL_ORIGIN})) {
      try { await ensureGmail(tab); }
      catch (error) { console.warn('Unable to attach Kidi to a Gmail tab:', error.message); }
    }
  } else {
    await updateState(state => {
      for (const [id, task] of Object.entries(state.tasks))
        if (task.kind.startsWith('gmail-')) delete state.tasks[id];
    });
  }
  notify({type: 'kidi.context.changed'});
}
async function saveTask(tab, frameId, context, pending = false) {
  if (typeof context.text !== 'string' || new TextEncoder().encode(context.text).length > 64 * 1024)
    throw new Error('Select no more than 64 KiB of text.');
  const task = {
    ...context, id: crypto.randomUUID(), tabId: tab.id, frameId,
    pageUrl: tab.url || '', createdAt: Date.now(),
  };
  await updateState(state => {
    state.tasks[task.id] = task;
    if (pending) {
      state.pending[String(tab.windowId)] = task.id;
      if (state.errors) delete state.errors[String(tab.windowId)];
    }
  });
  return task;
}
async function createSelectionTask(info, tab) {
  const command = COMMANDS.find(item => item.id === info.menuItemId);
  if (!tab?.id || !info.selectionText || !command) return;
  lastTabs.set(tab.windowId, tab.id);
  // Opening must start inside the context-menu user gesture, before capture awaits.
  const opening = openEditor(tab).then(() => null, error => error);
  const frameId = Number.isInteger(info.frameId) ? info.frameId : 0;
  const target = crypto.randomUUID();
  let selection = {text: info.selectionText, editable: false};
  let warning = '';
  try {
    await api.scripting.executeScript({
      target: {tabId: tab.id, frameIds: [frameId]}, files: ['selection-bridge.js'],
    });
    const captured = await api.tabs.sendMessage(
      tab.id, {type: 'liteapp.selection.capture', id: target}, {frameId});
    if (captured?.ok && captured.text) selection = captured;
    else warning = captured?.error || 'This selection can be copied but not replaced.';
  } catch (error) {
    warning = `Replacement is unavailable on this page: ${error.message}`;
  }
  const task = await saveTask(tab, frameId, {
    kind: 'selection', title: tab.title || 'Selected text',
    task: command.task, text: selection.text,
    operation: selection.editable ? 'selection' : null,
    operationLabel: 'Replace selection', target, warning,
  }, true);
  const openingError = await opening;
  if (openingError) throw openingError;
  notify({type: 'kidi.context.pending', windowId: tab.windowId, id: task.id});
}
async function capturePage(tab, task, pending = false) {
  if (!isWebPage(tab.url))
    throw new Error(tab.url
      ? 'Only HTTP and HTTPS webpages can be captured. Open an article or select its text.'
      : 'Click the Kidi toolbar icon on this webpage to grant temporary page access, then retry.');
  if (isGmail(tab.url)) throw new Error('Use the Gmail writing actions instead of capturing the mailbox page.');
  try {
    await api.scripting.executeScript({
      target: {tabId: tab.id, frameIds: [0]},
      files: ['libs/readability/Readability.js', 'article-bridge.js'],
    });
  } catch (error) {
    throw new Error(`Unable to read this page. Click Kidi in its toolbar or use the page's right-click menu. ${error.message}`);
  }
  const captured = await api.tabs.sendMessage(tab.id, {type: 'kidi.page.capture'}, {frameId: 0});
  if (!captured?.ok) throw new Error(captured?.error || 'Unable to extract this article.');
  if (captured.sourceUrl !== tab.url) throw new Error('The page navigated during capture. Try again.');
  return saveTask(tab, 0, {...captured, task}, pending);
}
async function createPageTask(info, tab) {
  const command = PAGE_COMMANDS.find(item => item.id === info.menuItemId);
  if (!command || !tab?.id) return;
  lastTabs.set(tab.windowId, tab.id);
  const opening = openEditor(tab).then(() => null, error => error);
  try {
    const task = await capturePage(tab, command.task, true);
    const error = await opening;
    if (error) throw error;
    notify({type: 'kidi.context.pending', windowId: tab.windowId, id: task.id});
  } catch (error) {
    await opening;
    await updateState(state => {
      const key = String(tab.windowId);
      delete state.pending[key];
      state.errors ||= {};
      state.errors[key] = {message: error.message, createdAt: Date.now()};
    });
    notify({type: 'kidi.context.error', windowId: tab.windowId});
    throw error;
  }
}
async function inspectContext(message) {
  const tab = await currentTab(message);
  const enabled = await gmailEnabled();
  const result = {
    ok: true, tabId: tab.id, windowId: tab.windowId,
    onGmail: isGmail(tab.url), gmailEnabled: enabled,
    pageAvailable: isWebPage(tab.url) && !isGmail(tab.url),
    pageUrl: tab.url || '',
  };
  if (!result.onGmail || !enabled) return result;
  await ensureGmail(tab);
  const details = await api.tabs.sendMessage(tab.id, {type: 'kidi.gmail.inspect'}, {frameId: 0});
  if (!details?.ok) throw new Error(details?.error || 'Unable to inspect the Gmail view.');
  return {...result, ...details};
}
async function handleMessage(message) {
  if (message.type === 'kidi.context.inspect') return inspectContext(message);
  if (message.type === 'kidi.pending.take') {
    const tab = await currentTab(message);
    return updateState(state => {
      const key = String(tab.windowId);
      if (state.errors?.[key]) {
        const {message: error} = state.errors[key];
        delete state.errors[key];
        return {ok: false, error};
      }
      const id = state.pending[key];
      delete state.pending[key];
      return {ok: true, task: id ? state.tasks[id] || null : null};
    });
  }
  if (message.type === 'kidi.gmail.capture') {
    const tab = await currentTab(message);
    await ensureGmail(tab);
    const captured = await api.tabs.sendMessage(tab.id, {
      type: 'kidi.gmail.capture', kind: message.kind, draftId: message.draftId,
    }, {frameId: 0});
    if (!captured?.ok) throw new Error(captured?.error || 'Unable to capture the Gmail context.');
    return {ok: true, task: await saveTask(tab, 0, captured)};
  }
  if (message.type === 'kidi.page.capture') {
    const tab = await currentTab(message);
    return {ok: true, task: await capturePage(tab, message.task || 'page_summarize')};
  }
  if (message.type === 'kidi.context.discard') {
    await updateState(state => { delete state.tasks[String(message.id || '')]; });
    return {ok: true};
  }
  if (message.type === 'kidi.context.apply') {
    const state = await readState();
    const task = state.tasks[String(message.id || '')];
    if (!task?.operation) throw new Error('This context is no longer editable. Capture it again or copy the result.');
    if (typeof message.text !== 'string' || !message.text.trim() ||
        new TextEncoder().encode(message.text).length > 64 * 1024)
      throw new Error('The result is empty or exceeds the insertion limit.');
    const tab = await api.tabs.get(task.tabId);
    if (tab.url !== task.pageUrl) throw new Error('The source page changed. Capture it again.');
    const result = await api.tabs.sendMessage(task.tabId, {
      type: 'liteapp.selection.replace',
      id: task.target, expected: task.text, replacement: message.text,
    }, {frameId: task.frameId});
    if (!result?.ok) throw new Error(result?.error || 'The source editor rejected the change.');
    await updateState(current => { delete current.tasks[task.id]; });
    return {ok: true};
  }
}
async function install() {
  await menus.removeAll();
  for (const command of COMMANDS)
    menus.create({id: command.id, title: command.title, contexts: ['selection']});
  for (const command of PAGE_COMMANDS)
    menus.create({
      id: command.id, title: command.title, contexts: ['page'],
      documentUrlPatterns: ['https://*/*', 'http://*/*'],
    });
  await syncGmail();
}
api.runtime.onInstalled.addListener(() => { install().catch(console.error); });
api.runtime.onStartup?.addListener(() => { install().catch(console.error); });
api.permissions.onAdded.addListener(() => { syncGmail().catch(console.error); });
api.permissions.onRemoved.addListener(() => { syncGmail().catch(console.error); });
menus.onClicked.addListener((info, tab) => {
  const handler = PAGE_COMMANDS.some(command => command.id === info.menuItemId) ? createPageTask : createSelectionTask;
  handler(info, tab).catch(console.error);
});
api.action?.onClicked.addListener(tab => {
  if (!tab?.id) return;
  lastTabs.set(tab.windowId, tab.id);
  openEditor(tab).catch(console.error);
});
api.tabs.onActivated.addListener(({tabId, windowId}) => {
  api.tabs.get(tabId).then(tab => {
    if (!tab.url?.startsWith(api.runtime.getURL(''))) lastTabs.set(windowId, tabId);
    notify({type: 'kidi.context.changed', windowId});
  }).catch(error => console.debug('Kidi active tab unavailable:', error.message));
});
api.tabs.onUpdated.addListener((tabId, change, tab) => {
  if (change.status === 'complete' || change.url)
    notify({type: 'kidi.context.changed', windowId: tab.windowId, tabId});
});
api.tabs.onRemoved.addListener(tabId => {
  updateState(state => {
    for (const [id, task] of Object.entries(state.tasks))
      if (task.tabId === tabId) delete state.tasks[id];
  }).catch(console.error);
});
api.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message?.type === 'kidi.gmail.changed' && sender.id === api.runtime.id && sender.tab &&
      isGmail(sender.url)) {
    notify({type: 'kidi.context.changed', windowId: sender.tab.windowId, tabId: sender.tab.id});
    return false;
  }
  if (!UI_MESSAGES.has(message?.type) || !isPanel(sender)) return false;
  handleMessage(message).then(sendResponse, error => {
    console.warn('Kidi context request failed:', message.type, error.message);
    sendResponse({ok: false, error: error.message || String(error)});
  });
  return true;
});
