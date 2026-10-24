import {modelProgressDetails, runtimeDiagnostics} from './progress.mjs';

globalThis.KIDI_SINGLE_THREAD_WASM = true;
const api = globalThis.browser || globalThis.chrome;
const {backend} = await import('./backend.js');
const $ = id => document.getElementById(id);
const PROMPT_STORAGE = 'kidi-editor-prompts-v1';
let snapshot = await backend.status();
let promptDefaults = {...(snapshot.components?.chat?.editor_prompts || {})};
let promptOverrides = {};
try {
  const saved = JSON.parse(localStorage.getItem(PROMPT_STORAGE) || '{}');
  if (saved && typeof saved === 'object' && !Array.isArray(saved)) promptOverrides = saved;
} catch {}
let ready = snapshot.ready === true;
let activeId = null;
let modelLoadId = null;
let activeTask = null;
let generated = '';
let runWhenReady = false;
let loadStartedAt = 0;
let loadMetrics = {};
let progressTimer;
let generationStats = {};

function notice(message = '', error = false) {
  $('notice').textContent = message;
  $('notice').classList.toggle('error', error);
}
function renderRuntime(stage = '') {
  const status = $('runtime-status');
  const label = $('runtime-status-label');
  status.classList.toggle('ready', ready);
  status.classList.toggle('loading', !ready && Boolean(modelLoadId));
  status.classList.toggle('offline', !ready && !modelLoadId);
  label.textContent = ready ? 'Ready' : stage || (modelLoadId ? 'Loading' : 'Offline');
  $('setup').hidden = ready;
  $('run').disabled = !ready || Boolean(activeId) || !$('source').value.trim();
  $('load-model').disabled = Boolean(modelLoadId);
  $('load-model').textContent = modelLoadId ? 'Loading…' : 'Load';
  renderDiagnostics();
}
function renderDiagnostics() {
  const container = $('diagnostics');
  container.hidden = !ready;
  if (!ready) {
    container.replaceChildren();
    return;
  }
  const fragment = document.createDocumentFragment();
  for (const item of runtimeDiagnostics(snapshot, generationStats)) {
    const metric = document.createElement('span');
    metric.className = `diagnostic ${item.key}`;
    const label = document.createElement('span');
    label.textContent = item.label;
    const value = document.createElement('strong');
    value.textContent = item.value;
    metric.append(label, value);
    fragment.append(metric);
  }
  container.replaceChildren(fragment);
}
async function refreshRuntime() {
  try {
    const current = await backend.status();
    snapshot = {...snapshot, ...current, runtime: current.runtime || snapshot.runtime};
    renderDiagnostics();
  } catch {}
}
function updateCount(input, output) {
  const count = [...$(input).value].length;
  $(output).textContent = `${count.toLocaleString()} ${count === 1 ? 'character' : 'characters'}`;
}
function taskName(task) {
  return [...$('task').options].find(option => option.value === task)?.textContent || task;
}
function defaultPrompt(task) {
  return typeof promptDefaults[task] === 'string' ? promptDefaults[task] : '';
}
function selectedPrompt(task = $('task').value) {
  const override = promptOverrides[task];
  return typeof override === 'string' && override ? override : defaultPrompt(task);
}
function storePrompts() {
  try { localStorage.setItem(PROMPT_STORAGE, JSON.stringify(promptOverrides)); } catch {}
}
function renderPrompt() {
  const task = $('task').value;
  $('prompt-label').textContent = `System prompt · ${taskName(task)}`;
  $('system-prompt').value = selectedPrompt(task);
  $('prompt-state').textContent = Object.hasOwn(promptOverrides, task)
    ? 'Customized · saved automatically'
    : 'Configured default';
}
function updatePrompt() {
  const task = $('task').value;
  const value = $('system-prompt').value.trim();
  if (!value) {
    $('prompt-state').textContent = 'Prompt cannot be empty';
    return;
  }
  if (value === defaultPrompt(task)) delete promptOverrides[task];
  else promptOverrides[task] = value;
  storePrompts();
  $('prompt-state').textContent = Object.hasOwn(promptOverrides, task)
    ? 'Customized · saved automatically'
    : 'Configured default';
}
function renderProgress() {
  if (!modelLoadId) return;
  const details = modelProgressDetails(loadMetrics, performance.now() - loadStartedAt);
  $('model-progress').hidden = false;
  $('load-stage').textContent = details.stage;
  $('load-percent').textContent = details.percent === null ? '' : `${details.percent}%`;
  $('load-bytes').textContent = details.bytes;
  $('load-transfer').textContent = details.transfer;
  $('load-file').textContent = details.file;
  $('load-warning').textContent = details.warning;
  $('load-warning').hidden = !details.warning;
  const bar = $('load-progress');
  if (details.determinate) {
    bar.max = details.total;
    bar.value = details.loaded;
  } else {
    bar.removeAttribute('value');
  }
}
function beginModelLoad() {
  loadStartedAt = performance.now();
  loadMetrics = {};
  clearInterval(progressTimer);
  progressTimer = setInterval(renderProgress, 1000);
  renderProgress();
  renderRuntime('Loading');
}
function endModelLoad() {
  clearInterval(progressTimer);
  progressTimer = undefined;
  modelLoadId = null;
}
function complete(text) {
  generated = text;
  $('result').value = text;
  updateCount('result', 'result-count');
  $('result-card').hidden = false;
  $('replace').hidden = !activeTask?.editable;
  $('cancel').hidden = true;
  activeId = null;
  renderRuntime();
}
function event(message) {
  const {type, payload = {}, id} = message;
  if (type === 'runtime.loading') {
    if (!modelLoadId) {
      modelLoadId = id;
      beginModelLoad();
    }
    if (id !== modelLoadId) return;
    renderRuntime('Loading');
  } else if (type === 'runtime.progress') {
    if (!modelLoadId) {
      modelLoadId = id;
      beginModelLoad();
    }
    if (id !== modelLoadId) return;
    loadMetrics = payload;
    renderProgress();
  } else if (type === 'runtime.ready' && (!modelLoadId || id === modelLoadId)) {
    ready = true;
    snapshot = {...snapshot, ready: true, backend: payload.backend, runtime: payload.runtime || snapshot.runtime,
      components: payload.components || snapshot.components};
    promptDefaults = {...(snapshot.components?.chat?.editor_prompts || promptDefaults)};
    renderPrompt();
    endModelLoad();
    $('model-progress').hidden = true;
    notice('Model ready.');
    renderRuntime();
    if (runWhenReady) {
      runWhenReady = false;
      run();
    }
  } else if (type === 'editor.delta' && id === activeId) {
    generated += payload.text || '';
    generationStats = payload.stats || generationStats;
    $('result').value = generated;
    updateCount('result', 'result-count');
    $('result-card').hidden = false;
    renderDiagnostics();
  } else if (type === 'editor.completed' && id === activeId) {
    generationStats = payload.stats || generationStats;
    complete(payload.text || generated);
    notice('Completed locally.');
    refreshRuntime();
  } else if (type === 'request.cancelled' && id === activeId) {
    activeId = null;
    $('cancel').hidden = true;
    notice('Cancelled.');
    renderRuntime();
  } else if (type === 'error' && id === modelLoadId) {
    endModelLoad();
    $('model-progress').hidden = true;
    notice(payload.message || 'Unable to load the local model.', true);
    renderRuntime();
  } else if (type === 'error' && (!id || id === activeId)) {
    activeId = null;
    $('cancel').hidden = true;
    notice(payload.message || 'Local model request failed.', true);
    renderRuntime();
  } else if (type === 'session.ready') {
    ready = payload.ready === true;
    snapshot = {...snapshot, ready, backend: payload.backend, runtime: payload.runtime || snapshot.runtime,
      components: payload.components || snapshot.components};
    renderRuntime();
  }
}
backend.connect({
  onOpen: () => renderRuntime(),
  onEvent: event,
  onError: error => notice(error.message || String(error), true),
  onClose: () => {
    ready = false;
    endModelLoad();
    renderRuntime('Disconnected');
  },
});

async function pendingTask() {
  const response = await api.runtime.sendMessage({type: 'kidi.pending.take'});
  if (!response?.task) return;
  activeTask = response.task;
  $('task').value = activeTask.task;
  $('source').value = activeTask.text;
  updateCount('source', 'source-count');
  $('instruction-row').hidden = activeTask.task !== 'custom';
  renderRuntime();
  if (ready) run();
  else {
    runWhenReady = true;
    notice('Load the model once to process selected text locally.');
  }
}
function loadModel() {
  if (modelLoadId) return;
  const runtime = snapshot.runtime || {};
  modelLoadId = crypto.randomUUID();
  beginModelLoad();
  notice('Downloading model files to local browser storage.');
  backend.send('runtime.load', modelLoadId, {
    model: runtime.model_id || snapshot.model,
    backend: runtime.requested_backend || 'webgpu',
    threads: runtime.threads || Math.min(4, navigator.hardwareConcurrency || 1),
    maximum_tokens: 512,
  });
}
function run() {
  const text = $('source').value.trim();
  if (!ready) {
    runWhenReady = true;
    loadModel();
    return;
  }
  if (!text || activeId) return;
  activeId = crypto.randomUUID();
  generated = '';
  generationStats = {};
  $('result').value = '';
  updateCount('result', 'result-count');
  $('result-card').hidden = false;
  $('replace').hidden = true;
  $('cancel').hidden = false;
  notice('Running locally…');
  renderRuntime();
  backend.send('editor.run', activeId, {
    task: $('task').value,
    text,
    instruction: $('task').value === 'custom' ? $('instruction').value : undefined,
    system_prompt: $('system-prompt').value.trim() || selectedPrompt(),
    length: 'preserve',
    maximum_tokens: 512,
  });
}
$('load-model').addEventListener('click', loadModel);
$('run').addEventListener('click', run);
$('cancel').addEventListener('click', () => {
  if (activeId) backend.send('cancel', activeId);
});
$('task').addEventListener('change', () => {
  $('instruction-row').hidden = $('task').value !== 'custom';
  renderPrompt();
});
$('prompt-toggle').addEventListener('click', () => {
  const opening = $('prompt-editor').hidden;
  $('prompt-editor').hidden = !opening;
  $('prompt-toggle').setAttribute('aria-expanded', String(opening));
  if (opening) $('system-prompt').focus();
});
$('system-prompt').addEventListener('input', updatePrompt);
$('prompt-default').addEventListener('click', () => {
  delete promptOverrides[$('task').value];
  storePrompts();
  renderPrompt();
  $('system-prompt').focus();
});
$('source').addEventListener('input', () => {
  updateCount('source', 'source-count');
  renderRuntime();
});
$('result').addEventListener('input', () => updateCount('result', 'result-count'));
$('copy').addEventListener('click', async () => {
  await navigator.clipboard.writeText($('result').value);
  notice('Copied.');
});
$('replace').addEventListener('click', async () => {
  if (!activeTask) return;
  const response = await api.runtime.sendMessage({
    type: 'kidi.selection.replace',
    id: activeTask.id,
    text: $('result').value,
  });
  if (!response?.ok) return notice(response?.error || 'Unable to replace the selection.', true);
  $('replace').hidden = true;
  activeTask = null;
  notice('Selection replaced.');
});
updateCount('source', 'source-count');
updateCount('result', 'result-count');
renderPrompt();
renderRuntime();
pendingTask().catch(error => notice(error.message || String(error), true));
