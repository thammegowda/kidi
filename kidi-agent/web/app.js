import {call, connect, on, subscribe} from './rpc.js';
import {notice, renderHistory, setBusy} from './chat.js';
import {filesChanged} from './editor.js';
import {newChat, refreshChanges, refreshFiles, refreshSessions, setTitle} from './sidebar.js';
import './terminal.js';
import './theme.js';
import {enableSplitters} from './liteapp/ide/splitters.mjs';
import './layout.js';
import './settings.js';
import {setupSpeech} from './speech.js';

const phase = document.getElementById('status-phase');
let phaseTimer = null;

function showStatus(status) {
  if (!status) return;
  document.getElementById('status-model').textContent = (status.model ?? '').split('/').pop();
  const context = status.context;
  document.getElementById('status-context').textContent = context?.context_tokens
    ? `ctx ${(100 * context.prompt_tokens / context.context_tokens).toFixed(1)}% · ` +
      `${context.prompt_tokens.toLocaleString()}/${context.context_tokens.toLocaleString()}`
    : '';
  const speed = status.speed;
  document.getElementById('status-speed').textContent = !speed ? ''
    : `${speed.tokens_per_second.toFixed(1)} tok/s` +
      (speed.decode_only ? ` · prefill ${speed.prefill_seconds.toFixed(1)}s` : '');
}

function showPhase(text, running) {
  clearInterval(phaseTimer);
  if (!running) {
    phase.textContent = text ?? '';
    phase.classList.remove('running');
    return;
  }
  const started = performance.now();
  phase.classList.add('running');
  const update = () => { phase.textContent = `${text} ${((performance.now() - started) / 1000).toFixed(1)}s`; };
  update();
  phaseTimer = setInterval(update, 100);
}

function banner(text, action) {
  const element = document.getElementById('banner');
  element.replaceChildren(text);
  if (action) {
    const button = document.createElement('button');
    button.className = 'primary small';
    button.textContent = action[0];
    button.addEventListener('click', action[1]);
    element.append(' ', button);
  }
  element.hidden = false;
}

// Subscribing delivers a "sync" event (the conversation and whether a turn is running) and then replays the running
// turn, so the page draws a consistent state whenever it connects.
let synced = null;
on('sync', state => {
  renderHistory(state.messages);
  setBusy(state.busy);
  synced?.(state);
  synced = null;
});

async function synchronize(info) {
  info ??= await call('app.info');
  showStatus(info.status);
  const state = new Promise(resolve => { synced = resolve; });
  subscribe();
  return state;
}

async function start() {
  enableSplitters();
  try {
    await connect({
      onClosed: () => banner('Connection lost. Reconnecting…'),
      onReconnected: async () => {
        document.getElementById('banner').hidden = true;
        await synchronize();
      },
    });
  } catch {
    banner('Cannot reach the agent. Open the link printed by `kidi agent --web` (it works once per server).');
    return;
  }
  on('status', status => {
    showStatus(status);
    if (status.phase === 'Ready' || status.phase === 'Stopped') showPhase(status.phase, false);
  });
  on('busy', ({phase: text}) => showPhase(text, true));
  on('user', ({text}) => { if (document.getElementById('chat-title').textContent === 'New conversation') setTitle(text); });
  on('idle', () => showPhase('', false));
  on('done', result => {
    showStatus(result.status);
    showPhase(result.ok ? 'Ready' : 'Stopped', false);
    refreshChanges();
  });
  on('files_changed', ({paths, source}) => {
    filesChanged(paths);
    refreshFiles();
    if (source === 'agent' || source === 'editor') refreshChanges();
  });
  const info = await call('app.info');
  document.title = `${info.name} — Kidi Agent`;
  const workspace = document.getElementById('workspace-name');
  workspace.textContent = info.name;
  workspace.title = info.workspace;
  document.getElementById('mode-badge').textContent = info.mode;
  setupSpeech(info.speech, text => notice(text, 'error'));
  showStatus(info.status);
  const history = await synchronize(info);
  if (!history.messages.length && !history.busy)
    notice(`Workspace ${info.workspace} · permissions: ${info.mode}. Type /help for commands.`, 'session');
  document.getElementById('new-chat').addEventListener('click', newChat);
  await refreshFiles();
  refreshChanges();
  refreshSessions();
  document.getElementById('prompt').focus();
}

start();
