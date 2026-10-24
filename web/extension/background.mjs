const api = globalThis.browser || globalThis.chrome;
const STORAGE_KEY = 'kidi-extension-tasks-v1';
const COMMANDS = [
  {id: 'kidi-polish', title: 'Polish with Kidi', task: 'polish'},
  {id: 'kidi-explain', title: 'Explain with Kidi', task: 'explain'},
  {id: 'kidi-summarize', title: 'Summarize with Kidi', task: 'summarize'},
];

const storage = () => api.storage.session || api.storage.local;
async function readState() {
  return (await storage().get(STORAGE_KEY))[STORAGE_KEY] || {tasks: {}, pending: null};
}
async function writeState(state) {
  await storage().set({[STORAGE_KEY]: state});
}
async function updateState(update) {
  const state = await readState();
  await update(state);
  const entries = Object.entries(state.tasks).sort((left, right) => right[1].createdAt - left[1].createdAt);
  state.tasks = Object.fromEntries(entries.slice(0, 16));
  await writeState(state);
  return state;
}
async function openEditor(tabId) {
  if (api.sidePanel?.open) await api.sidePanel.open({tabId});
  else if (api.sidebarAction?.open) await api.sidebarAction.open();
  else await api.tabs.create({url: api.runtime.getURL('editor.html')});
}
async function installSelectionBridge(tabId, frameId) {
  await api.scripting.executeScript({
    target: {tabId, frameIds: [frameId]},
    files: ['selection-bridge.js'],
  });
}
async function captureSelection(tabId, frameId, id, fallback) {
  try {
    await installSelectionBridge(tabId, frameId);
    const captured = await api.tabs.sendMessage(
      tabId, {type: 'liteapp.selection.capture', id}, {frameId});
    if (captured?.ok && captured.text) return captured;
  } catch {}
  return {text: fallback || '', editable: false};
}
async function createTask(info, tab) {
  if (!tab?.id || !info.selectionText) return;
  const command = COMMANDS.find(item => item.id === info.menuItemId);
  if (!command) return;
  const id = crypto.randomUUID();
  const frameId = Number.isInteger(info.frameId) ? info.frameId : 0;
  const selection = await captureSelection(tab.id, frameId, id, info.selectionText);
  const task = {
    id,
    task: command.task,
    text: selection.text || info.selectionText,
    editable: selection.editable === true,
    tabId: tab.id,
    frameId,
    pageTitle: tab.title || '',
    pageUrl: tab.url || '',
    createdAt: Date.now(),
  };
  await updateState(state => {
    state.tasks[id] = task;
    state.pending = id;
  });
  await openEditor(tab.id);
}
async function installMenus() {
  await api.contextMenus.removeAll();
  for (const command of COMMANDS)
    api.contextMenus.create({id: command.id, title: command.title, contexts: ['selection']});
  if (api.sidePanel?.setPanelBehavior)
    await api.sidePanel.setPanelBehavior({openPanelOnActionClick: true});
}
api.runtime.onInstalled.addListener(() => { installMenus().catch(console.error); });
api.runtime.onStartup?.addListener(() => { installMenus().catch(console.error); });
api.contextMenus.onClicked.addListener((info, tab) => { createTask(info, tab).catch(console.error); });
api.action?.onClicked.addListener(tab => { if (tab?.id) openEditor(tab.id).catch(console.error); });

async function handleMessage(message) {
  if (message?.type === 'kidi.pending.take') {
    let task = null;
    await updateState(state => {
      task = state.pending ? state.tasks[state.pending] || null : null;
      state.pending = null;
    });
    return {task};
  }
  if (message?.type === 'kidi.selection.replace') {
    const state = await readState();
    const task = state.tasks[String(message.id || '')];
    if (!task || !task.editable) return {ok: false, error: 'The original selection is not editable'};
    try {
      const result = await api.tabs.sendMessage(task.tabId, {
        type: 'liteapp.selection.replace',
        id: task.id,
        expected: task.text,
        replacement: String(message.text || ''),
      }, {frameId: task.frameId});
      if (!result?.ok) return {ok: false, error: result?.error || 'The page rejected the replacement'};
      await updateState(current => { delete current.tasks[task.id]; });
      return {ok: true};
    } catch (error) {
      return {ok: false, error: error.message || String(error)};
    }
  }
  if (message?.type === 'kidi.task.complete') {
    await updateState(state => { delete state.tasks[String(message.id || '')]; });
    return {ok: true};
  }
  return undefined;
}
api.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  const result = handleMessage(message);
  if (!result) return false;
  Promise.resolve(result).then(sendResponse, error => sendResponse({ok: false, error: error.message || String(error)}));
  return true;
});
