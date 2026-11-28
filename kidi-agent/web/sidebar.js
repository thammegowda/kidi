import {FileTree, renderChanges} from './liteapp/ide/file-tree.mjs';
import {call} from './rpc.js';
import {openDiff, openFile} from './editor.js';
import {clear, notice, renderHistory} from './chat.js';

const changesList = document.getElementById('changes-list');
const changesCount = document.getElementById('changes-count');
const sessionsList = document.getElementById('sessions-list');
const historyPanel = document.getElementById('history-panel');
const historyButton = document.getElementById('toggle-history');
const chatTitle = document.getElementById('chat-title');
const tree = new FileTree({call, element: document.getElementById('file-tree'), onOpen: openFile});

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

export const refreshFiles = () => tree.refresh();

// What the agent changed in this conversation, then the Git working tree; both also badge the file tree.
export async function refreshChanges() {
  try {
    const result = await call('changes.list');
    tree.setDecorations(new Map([...result.git, ...result.session].map(item => [item.path, item.status])));
    const compare = base => item => openDiff(item.path, base).catch(error => notice(error.message, 'error'));
    renderChanges(changesList, [
      {title: 'Changed in this conversation', empty: 'The agent has not edited files yet.', items: result.session,
        onOpen: compare('session')},
      {title: 'Git working tree', empty: result.repository ? 'No uncommitted changes.' : 'Not a Git repository.',
        items: result.git, onOpen: compare('git')},
    ]);
    const count = new Set([...result.git, ...result.session].map(item => item.path)).size;
    changesCount.textContent = count ? String(count) : '';
  } catch (error) {
    changesList.replaceChildren(element('p', 'empty', error.message));
  }
}

function age(timeNs) {
  const seconds = (Date.now() - timeNs / 1e6) / 1000;
  if (seconds < 60) return 'just now';
  if (seconds < 3600) return `${Math.floor(seconds / 60)} min ago`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)} h ago`;
  return new Date(timeNs / 1e6).toLocaleDateString();
}

export function setTitle(text) {
  chatTitle.textContent = text || 'New conversation';
  chatTitle.title = chatTitle.textContent;
}

function showHistory(open) {
  historyPanel.hidden = !open;
  historyButton.classList.toggle('active', open);
  if (open) refreshSessions();
}

export async function refreshSessions() {
  try {
    const {sessions} = await call('sessions.list');
    const current = sessions.find(session => session.current);
    if (current) setTitle(current.title);
    if (!sessions.length) {
      sessionsList.replaceChildren(element('p', 'empty', 'Saved conversations of this workspace appear here.'));
      return;
    }
    sessionsList.replaceChildren(...sessions.map(session => {
      const row = element('div', 'session-row' + (session.current ? ' current' : ''));
      row.append(element('div', 'session-title', session.title || '(untitled)'),
        element('div', 'session-meta', `${age(session.time_ns)} · ${session.messages} messages`));
      row.addEventListener('click', async () => {
        try {
          const {messages} = await call('sessions.open', {id: session.id});
          renderHistory(messages);
          setTitle(session.title);
          showHistory(false);
          notice('Continuing this conversation.', 'session');
          refreshSessions();
          refreshChanges();
        } catch (error) {
          notice(error.message, 'error');
        }
      });
      return row;
    }));
  } catch (error) {
    sessionsList.replaceChildren(element('p', 'empty', error.message));
  }
}

export async function newChat() {
  try {
    await call('chat.new');
    clear();
    setTitle('');
    showHistory(false);
    notice('New conversation.', 'session');
    refreshChanges();
    refreshSessions();
  } catch (error) {
    notice(error.message, 'error');
  }
}

async function newFile() {
  const path = prompt('New file path (relative to the workspace):');
  if (!path) return;
  try {
    await call('fs.write', {path, content: '', modified_ns: 0});
    await refreshFiles();
    openFile(path.replace(/^\.\//, ''));
  } catch (error) {
    notice(error.code === 'conflict' ? `${path} already exists.` : error.message, 'error');
  }
}

document.getElementById('refresh-files').addEventListener('click', refreshFiles);
document.getElementById('refresh-changes').addEventListener('click', refreshChanges);
document.getElementById('new-file').addEventListener('click', newFile);
for (const tab of document.querySelectorAll('.sidebar-tabs [role=tab]'))
  tab.addEventListener('click', () => {
    for (const other of document.querySelectorAll('.sidebar-tabs [role=tab]')) other.classList.toggle('active', other === tab);
    for (const panel of document.querySelectorAll('.sidebar .panel'))
      panel.classList.toggle('active', panel.dataset.panel === tab.dataset.panel);
    if (tab.dataset.panel === 'changes') refreshChanges();
  });

historyButton.addEventListener('click', () => showHistory(historyPanel.hidden));
document.addEventListener('keydown', event => { if (event.key === 'Escape' && !historyPanel.hidden) showHistory(false); });
