import {modelProgressDetails, runtimeDiagnostics} from './progress.mjs';
import {renderMarkdown, renderStreamingMarkdown} from './markdown.js';
import {parseLanguages, resolveLanguage} from './languages.js';
import {ReadAloud} from './read-aloud.js';

globalThis.KIDI_SINGLE_THREAD_WASM = true;
const api = globalThis.browser || globalThis.chrome;
const {backend} = await import('./backend.js');
const $ = id => document.getElementById(id);
const PROMPT_STORAGE = 'kidi-editor-prompts-v1';
const GMAIL_ORIGIN = 'https://mail.google.com/*';
const SPEECH_CONSENT = 'kidi-read-aloud-consent-v1';
const LANGUAGE_STORAGE = 'kidi-translation-language-v1';
const DOCUMENT_ACTIONS = new Set(['page_summarize', 'page_translate']);
const LABELS = {
  chat: 'Chat', email_compose: 'Compose email', email_polish: 'Polish draft', email_reply: 'Draft a reply',
  polish: 'Polish selection', explain: 'Explain selection', summarize: 'Summarize selection',
  formalize: 'Make formal', friendly: 'Make friendly', concise: 'Make concise', custom: 'Custom edit',
  page_summarize: 'Summarize page', page_translate: 'Translate page',
};
const REPLACE_ACTIONS = new Set(['polish', 'formalize', 'friendly', 'concise', 'custom']);
const windowInfo = await api.windows.getCurrent();
const tabParameter = new URL(location.href).searchParams.get('tab');
const sourceTabId = tabParameter !== null && /^\d+$/u.test(tabParameter) ? Number(tabParameter) : null;
const pageContext = () => ({windowId: windowInfo.id, ...(sourceTabId === null ? {} : {tabId: sourceTabId})});
let snapshot = await backend.status();
let promptDefaults = {...(snapshot.components?.chat?.editor_prompts || {})};
let promptOverrides = {};
let ready = snapshot.ready === true;
let activeRequest = null;
let context = null;
let inspection = {};
let selectedDraft = '';
let action = 'chat';
let promptAction = 'chat';
let history = [];
let queued = null;
let modelLoadId = null;
let loadStartedAt = 0;
let loadMetrics = {};
let progressTimer;
let generationStats = {};
let capturing = false;
let refreshSequence = 0;
let refreshTimer;
let languageTable = {languages: [], aliases: new Map()};
let speechApproved = false;
let pendingSpeech = null;
let speechState = {active: false};
let speechStartedAt = 0;
let speechTimer;
const readAloud = new ReadAloud({
  send: (type, id, payload) => backend.send(type, id, payload),
  onState: renderSpeechState,
  onError: error => notice(`Read aloud: ${error.message || String(error)}`, true),
});

function renderSpeechState(state = speechState) {
  speechState = state;
  if (!state.active) {
    clearInterval(speechTimer);
    speechTimer = undefined;
  } else if (!speechTimer) speechTimer = setInterval(renderSpeechState, 1000);
  $('speech-player').hidden = !state.active;
  $('speech-pause').disabled = !state.started;
  $('speech-pause').textContent = state.paused ? 'Resume' : 'Pause';
  let message = state.message;
  const progress = state.progress;
  if (progress && !state.paused) {
    if (progress.phase === 'loading') {
      const details = modelProgressDetails(progress, performance.now() - speechStartedAt);
      message = [details.stage, details.percent === null ? '' : `${details.percent}%`,
        details.bytes, details.transfer, details.file, details.warning].filter(Boolean).join(' · ');
    } else if (progress.phase === 'calibrating-voice')
      message = `Preparing voice · ${Math.floor((performance.now() - speechStartedAt) / 1000)}s elapsed`;
    else if (progress.phase === 'fallback' || progress.phase === 'voice-cache') message = progress.message;
    if (progress.steps) message += ` · ${progress.step || 0}/${progress.steps}`;
  }
  $('speech-status').textContent = message;
  if (!state.active) notice(message);
  renderDiagnostics();
}
function notice(message = '', error = false) {
  $('notice').textContent = message;
  $('notice').classList.toggle('error', error);
}
try {
  const saved = JSON.parse(localStorage.getItem(PROMPT_STORAGE) || '{}');
  if (!saved || typeof saved !== 'object' || Array.isArray(saved)) throw new Error('Invalid saved prompts');
  for (const [key, value] of Object.entries(saved))
    if (Object.hasOwn(LABELS, key) && typeof value === 'string' && value.trim()) promptOverrides[key] = value;
  speechApproved = localStorage.getItem(SPEECH_CONSENT) === 'true';
} catch (error) {
  console.warn('Unable to restore Kidi prompt overrides:', error);
  notice('Saved prompts could not be restored. Configured defaults are in use.', true);
}
function promptFor(task) {
  return promptOverrides[task] ?? defaultPromptFor(task);
}
function defaultPromptFor(task) {
  return promptDefaults[task === 'chat' ? 'system' : task] ?? '';
}
function renderPrompt() {
  $('prompt-action').value = promptAction;
  $('system-prompt').value = promptFor(promptAction);
  $('prompt-state').textContent = Object.hasOwn(promptOverrides, promptAction)
    ? 'Customized · saved automatically' : 'Configured default';
}
function savePrompt() {
  try {
    localStorage.setItem(PROMPT_STORAGE, JSON.stringify(promptOverrides));
    return true;
  } catch (error) {
    console.error('Unable to save Kidi prompts:', error);
    $('prompt-state').textContent = 'Unable to save. This change only lasts while the panel is open.';
    return false;
  }
}
function renderDiagnostics() {
  $('diagnostics').hidden = !ready && !speechState.active;
  const metrics = ready || speechState.active
    ? runtimeDiagnostics(snapshot, generationStats).filter(item => ready || item.key !== 'backend') : [];
  $('diagnostics').replaceChildren(...metrics.map(item => {
    const metric = document.createElement('span');
    metric.className = `diagnostic ${item.key}`;
    const label = document.createElement('span'), value = document.createElement('strong');
    label.textContent = item.label;
    value.textContent = item.value;
    metric.append(label, value);
    return metric;
  }));
}
function renderRuntime(stage = '') {
  $('runtime-status').className = `runtime-status ${ready ? 'ready' : modelLoadId ? 'loading' : 'offline'}`;
  $('runtime-status-label').textContent = ready ? 'Ready' : stage || (modelLoadId ? 'Loading' : 'Offline');
  $('setup').hidden = ready;
  $('load-model').disabled = Boolean(modelLoadId);
  $('load-model').textContent = modelLoadId ? 'Loading…' : 'Load';
  $('send').disabled = Boolean(activeRequest || queued || capturing) ||
    (!$('message').value.trim() && !(DOCUMENT_ACTIONS.has(action) && context?.text)) ||
    (action === 'page_translate' && !$('target-language').value);
  $('send').textContent = DOCUMENT_ACTIONS.has(action) ? action === 'page_translate' ? 'Translate' : 'Summarize' : 'Send';
  $('new-chat').disabled = capturing;
  $('clear-context').disabled = capturing;
  $('cancel').hidden = !activeRequest;
  $('action-label').textContent = LABELS[action];
  $('translation-tools').hidden = action !== 'page_translate';
  $('target-language').disabled = Boolean(activeRequest || queued || capturing);
  for (const button of $('suggestions').querySelectorAll('button'))
    button.disabled = Boolean(activeRequest || queued || capturing);
  renderDiagnostics();
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
  if (details.determinate) {
    $('load-progress').max = details.total;
    $('load-progress').value = details.loaded;
  } else $('load-progress').removeAttribute('value');
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
  $('model-progress').hidden = true;
}
function loadModel() {
  if (modelLoadId) return;
  modelLoadId = crypto.randomUUID();
  beginModelLoad();
  notice('Preparing local model files. Cached files are reused.');
  try {
    backend.send('runtime.load', modelLoadId, {
      model: snapshot.runtime?.model_id || snapshot.model,
      backend: snapshot.runtime?.requested_backend || 'webgpu',
      threads: snapshot.runtime?.threads || Math.min(4, navigator.hardwareConcurrency || 1),
      maximum_tokens: 512,
    });
  } catch (error) {
    queued = null;
    endModelLoad();
    notice(error.message || String(error), true);
    renderRuntime();
  }
}
async function sendIntegration(type, payload = {}) {
  const result = await api.runtime.sendMessage({type, ...pageContext(), ...payload});
  if (!result?.ok) throw new Error(result?.error || 'The page integration is unavailable. Reload the extension.');
  return result;
}
function chooseAction(task) {
  action = task;
  promptAction = task;
  renderPrompt();
  renderRuntime();
}
function suggestion(label, handler) {
  const button = document.createElement('button');
  button.className = 'quiet-button';
  button.type = 'button';
  button.textContent = label;
  button.disabled = Boolean(activeRequest || queued || capturing);
  button.addEventListener('click', async () => {
    try { await handler(); }
    catch (error) { notice(error.message || String(error), true); }
  });
  return button;
}
function renderSuggestions() {
  const buttons = [];
  const drafts = inspection.drafts || [];
  $('gmail-connect').hidden = !inspection.onGmail || inspection.gmailEnabled;
  $('draft-picker').hidden = drafts.length < 2;
  if (inspection.onGmail && inspection.gmailEnabled) {
    if (drafts.length) {
      buttons.push(
        suggestion('Polish draft', () => captureGmail('draft', 'email_polish')),
        suggestion('Compose from notes', () => captureGmail('draft', 'email_compose')),
      );
    } else {
      buttons.push(suggestion('Compose email', () => {
        chooseAction('email_compose');
        $('message').placeholder = 'What would you like the email to say?';
        $('message').focus();
      }));
    }
    if (inspection.emailAvailable)
      buttons.push(suggestion('Draft a reply', () => captureGmail('email', 'email_reply')));
  } else if (context?.kind === 'selection') {
    for (const task of ['polish', 'explain', 'summarize'])
      buttons.push(suggestion(LABELS[task], () => submit(task)));
  } else {
    if (!inspection.onGmail) buttons.push(
      suggestion('Summarize page', () => pageAction('page_summarize')),
      suggestion('Translate page', () => pageAction('page_translate')),
      suggestion('Read article', () => {
        readAloud.prepare().catch(error => notice(error.message, true));
        return pageAction('page_read');
      }),
    );
    buttons.push(suggestion('Compose email', () => {
      chooseAction('email_compose');
      $('message').placeholder = 'Describe the email, or paste your rough notes…';
      $('message').focus();
    }));
  }
  $('suggestions').replaceChildren(...buttons);
}
async function refreshContext() {
  const sequence = ++refreshSequence;
  try {
    const details = await sendIntegration('kidi.context.inspect');
    if (sequence !== refreshSequence) return;
    inspection = details;
    const drafts = details.drafts || [];
    if (!drafts.some(draft => draft.id === selectedDraft))
      selectedDraft = details.activeDraftId || (drafts.length === 1 ? drafts[0].id : '');
    const choices = [];
    if (drafts.length > 1) choices.push(new Option('Choose a draft…', ''));
    for (const [index, draft] of drafts.entries())
      choices.push(new Option(`${draft.title}${drafts.length > 1 ? ` (${index + 1})` : ''}`, draft.id));
    $('gmail-draft').replaceChildren(...choices);
    $('gmail-draft').value = selectedDraft;
    $('context-status').textContent = details.onGmail && details.gmailEnabled && !drafts.length && !details.emailAvailable
      ? 'Open a compose window or an email in Gmail.' : '';
    renderSuggestions();
  } catch (error) {
    if (sequence !== refreshSequence) return;
    $('context-status').textContent = error.message;
  }
}
function renderContext() {
  $('context-card').hidden = !context;
  $('context-label').textContent = context
    ? `${context.kind === 'gmail-draft' ? 'Draft' : context.kind === 'gmail-email' ? 'Email'
      : context.kind === 'article' ? 'Article' : 'Selection'}: ${context.title}`
    : '';
  $('context-label').title = context?.sourceUrl || '';
  $('context-text').textContent = context?.text || '';
  $('context-warning').textContent = context?.warning || '';
  $('context-warning').hidden = !context?.warning;
  $('context-followup').textContent = context?.summaryText
    ? 'Follow-up chat uses the generated summary, not the full article. Page actions still use the attached source.'
    : context?.kind === 'article' ? 'Read-only snapshot. Page actions use the attached text; the webpage is not changed.' : '';
  $('context-followup').hidden = !$('context-followup').textContent;
  renderSuggestions();
}
async function clearConversation({detach = true} = {}) {
  pendingSpeech = null;
  $('speech-dialog').close();
  await readAloud.stop();
  if (activeRequest) backend.send('cancel', activeRequest.id);
  activeRequest = null;
  history = [];
  queued = null;
  generationStats = {};
  $('conversation').replaceChildren($('welcome') || createWelcome());
  $('welcome').hidden = false;
  if (detach && context) {
    const previous = context;
    context = null;
    try { await sendIntegration('kidi.context.discard', {id: previous.id}); }
    catch (error) { notice(error.message, true); }
  }
  chooseAction('chat');
  $('message').placeholder = 'Ask Kidi, or describe the email you want to write…';
  renderContext();
  renderRuntime();
}
function createWelcome() {
  const welcome = document.createElement('div');
  welcome.id = 'welcome';
  welcome.className = 'welcome';
  const heading = document.createElement('h1'), paragraph = document.createElement('p');
  heading.textContent = 'How can I help?';
  paragraph.textContent = 'Chat privately, write an email, summarize an article, or translate a page.';
  welcome.append(heading, paragraph);
  return welcome;
}
async function attachContext(task, taskAction) {
  await clearConversation();
  context = task;
  chooseAction(taskAction);
  renderContext();
  if (task.warning) notice(task.warning);
}
async function captureGmail(kind, task) {
  if (capturing || activeRequest || queued) return;
  capturing = true;
  renderRuntime();
  try {
    const captured = await sendIntegration('kidi.gmail.capture', {
      tabId: inspection.tabId, kind, draftId: selectedDraft,
    });
    await attachContext(captured.task, task);
  } finally {
    capturing = false;
    renderRuntime();
  }
  submit(task);
}
async function pageAction(task) {
  if (capturing || activeRequest || queued) return;
  const current = context?.kind === 'article' && context.tabId === inspection.tabId &&
    context.sourceUrl === inspection.pageUrl;
  if (!current) {
    capturing = true;
    renderRuntime();
    try {
      const captured = await sendIntegration('kidi.page.capture', {task});
      await attachContext(captured.task, task === 'page_read' ? 'chat' : task);
    } finally {
      capturing = false;
      renderRuntime();
    }
  } else if (task !== 'page_read') chooseAction(task);
  if (task === 'page_read') return requestRead(context.text, context.language);
  else if (task === 'page_translate') {
    notice('Choose a language, then click Translate. The webpage stays unchanged.');
    $('target-language').focus();
  } else submit(task);
}
function languageFor(value) {
  const code = resolveLanguage(value, languageTable.aliases) ||
    resolveLanguage(String(value || '').split('-')[0], languageTable.aliases);
  return languageTable.languages.find(language => language.code === code);
}
function requestRead(text, language) {
  if (!languageTable.languages.length) return notice('Narration languages are unavailable. Reload the panel.', true);
  const resolved = languageFor(language);
  if (!speechApproved || !resolved) {
    pendingSpeech = {text};
    $('speech-language').value = resolved?.code || '';
    $('speech-description').textContent = speechApproved
      ? 'Choose the language of this text. Kidi will not silently switch it to English.'
      : 'Read aloud uses an additional local voice model. Its files are downloaded only with your approval; cached files are reused. Text and audio stay on this device.';
    $('speech-model').textContent = snapshot.components?.tts?.directory || 'Configured local voice model';
    $('speech-dialog').showModal();
    return;
  }
  speechStartedAt = performance.now();
  return readAloud.start({text, language: resolved.name, cacheIdentity: snapshot.components?.tts?.directory || ''});
}
async function loadLanguages() {
  try {
    const response = await fetch('./languages.tsv');
    if (!response.ok) throw new Error(`Language table returned HTTP ${response.status}`);
    languageTable = parseLanguages(await response.text());
    for (const language of languageTable.languages) {
      $('target-language').append(new Option(language.name, language.code));
      $('speech-language').append(new Option(language.name, language.code));
    }
    const saved = localStorage.getItem(LANGUAGE_STORAGE);
    $('target-language').value = languageFor(saved || navigator.language)?.code || '';
    renderRuntime();
  } catch (error) {
    notice(`Unable to load language choices: ${error.message}`, true);
  }
}
function appendTurn(role, text) {
  $('welcome').hidden = true;
  const section = document.createElement('section');
  section.className = `turn ${role}`;
  const label = document.createElement('span'), body = document.createElement('div');
  label.className = 'turn-label';
  label.textContent = role === 'user' ? 'You' : 'Kidi AI';
  body.className = 'turn-text';
  body.textContent = text;
  section.append(label, body);
  $('conversation').append(section);
  $('conversation').scrollTop = $('conversation').scrollHeight;
  return {section, body};
}
function resultActions(request) {
  const actions = document.createElement('div');
  actions.className = 'result-actions';
  actions.append(suggestion('Copy', async () => {
    await navigator.clipboard.writeText(request.text);
    notice('Copied.');
  }));
  actions.append(suggestion('Read aloud', () => requestRead(request.text, request.language)));
  if (request.context?.operation === 'selection' && REPLACE_ACTIONS.has(request.action)) {
    const apply = suggestion(request.context.operationLabel || 'Apply', async () => {
      await sendIntegration('kidi.context.apply', {id: request.context.id, text: request.text});
      apply.disabled = true;
      apply.textContent = 'Applied';
      if (context?.id === request.context.id) context.operation = null;
      notice('Replaced the captured selection in its original editor.');
    });
    actions.prepend(apply);
  }
  request.view.section.append(actions);
}
function failRequest(message, cancelled = false) {
  if (activeRequest) {
    const status = document.createElement('div');
    status.className = `turn-status${cancelled ? '' : ' error'}`;
    status.textContent = message;
    activeRequest.view.section.append(status);
  }
  activeRequest = null;
  notice(message, !cancelled);
  renderRuntime();
}
function submit(task = action, instruction = $('message').value.trim()) {
  if (activeRequest || queued || capturing) return;
  chooseAction(task);
  if (!instruction && !context?.text) {
    notice('Add a message or rough notes first.');
    $('message').focus();
    return;
  }
  if (!ready) {
    queued = {task, instruction, contextId: context?.id};
    loadModel();
    renderRuntime();
    return;
  }
  if (history.length >= 20) return notice('This conversation reached its limit. Start a new chat.', true);
  const userText = instruction || LABELS[task];
  const documentAction = DOCUMENT_ACTIONS.has(task);
  const targetLanguage = task === 'page_translate' ? languageFor($('target-language').value) : null;
  if (task === 'page_translate' && !targetLanguage) return notice('Choose a translation language first.', true);
  appendTurn('user', userText);
  const view = appendTurn('assistant', '');
  activeRequest = {
    id: crypto.randomUUID(), action: task, context: context ? {...context} : null,
    input: userText, text: '', view, documentAction,
    language: targetLanguage?.name || context?.language || '',
  };
  generationStats = {};
  $('message').value = '';
  notice('Working locally…');
  renderRuntime();
  try {
    if (documentAction) {
      backend.send('document.run', activeRequest.id, {
        mode: task === 'page_translate' ? 'translate' : 'summarize',
        text: context.text, instruction, system_prompt: promptFor(task),
        ...(targetLanguage ? {target_language: targetLanguage.name} : {}),
        maximum_tokens: 512,
      });
    } else {
      backend.send('editor.run', activeRequest.id, {
        task, text: context?.summaryText || context?.text || '', instruction,
        system_prompt: promptFor(task), history,
        length: 'preserve', maximum_tokens: 512,
      });
      if (context?.summaryText) notice('Working locally, using the generated summary as follow-up context.');
    }
  } catch (error) { failRequest(error.message || String(error)); }
}
async function refreshRuntime() {
  try {
    snapshot = await backend.status();
    renderDiagnostics();
  } catch (error) {
    console.warn('Kidi diagnostics refresh failed:', error);
    notice(`Unable to refresh diagnostics: ${error.message}`, true);
  }
}
function event(message) {
  if (readAloud.handleEvent(message)) return;
  const {type, payload = {}, id} = message;
  if (type === 'runtime.loading' || type === 'runtime.progress') {
    if (!modelLoadId) {
      modelLoadId = id;
      beginModelLoad();
    }
    if (id !== modelLoadId) return;
    if (type === 'runtime.progress') { loadMetrics = payload; renderProgress(); }
    renderRuntime('Loading');
  } else if (type === 'runtime.ready' && (!modelLoadId || id === modelLoadId)) {
    ready = true;
    snapshot = {...snapshot, ready: true, backend: payload.backend, runtime: payload.runtime || snapshot.runtime,
      components: payload.components || snapshot.components};
    promptDefaults = {...(snapshot.components?.chat?.editor_prompts || promptDefaults)};
    renderPrompt();
    endModelLoad();
    const waiting = queued;
    queued = null;
    notice('Model ready.');
    renderRuntime();
    if (waiting && waiting.contextId === context?.id) submit(waiting.task, waiting.instruction);
  } else if (type === 'session.ready') {
    ready = payload.ready === true;
    snapshot = {...snapshot, ...payload};
    renderRuntime();
  } else if (type === 'tts.ready') {
    snapshot = {...snapshot, runtime: payload.runtime || snapshot.runtime,
      components: {...snapshot.components, tts: payload.component || snapshot.components?.tts}};
    renderDiagnostics();
  } else if (type === 'document.progress' && id === activeRequest?.id) {
    if (!activeRequest.view.status) {
      const status = document.createElement('div');
      status.className = 'turn-status';
      activeRequest.view.section.append(status);
      activeRequest.view.status = status;
    }
    activeRequest.view.status.textContent = payload.translating ? `Translating section ${payload.part}/${payload.parts}`
      : `${payload.round > 1 ? 'Combining summaries' : 'Summarizing'} · section ${payload.part}/${payload.parts}` +
        (payload.round > 1 ? ` · pass ${payload.round}` : '');
  } else if ((type === 'editor.delta' || type === 'document.delta') && id === activeRequest?.id) {
    const nearBottom = $('conversation').scrollHeight - $('conversation').scrollTop - $('conversation').clientHeight < 80;
    if (payload.replace) activeRequest.text = payload.text || '';
    else activeRequest.text += payload.text || '';
    renderStreamingMarkdown(activeRequest.view.body, activeRequest.text);
    generationStats = payload.stats || generationStats;
    renderDiagnostics();
    if (nearBottom) $('conversation').scrollTop = $('conversation').scrollHeight;
  } else if ((type === 'editor.completed' || type === 'document.completed') && id === activeRequest?.id) {
    const request = activeRequest;
    request.text = payload.text || request.text;
    renderMarkdown(request.view.body, request.text);
    generationStats = payload.stats || generationStats;
    if (request.text) {
      if (request.documentAction && payload.output_limited) {
        const warning = request.view.status || document.createElement('div');
        warning.className = 'turn-status error';
        warning.textContent = 'The summary reached its output budget and may end abruptly. Try a more concise prompt.';
        request.view.section.append(warning);
      } else request.view.status?.remove();
      if (request.documentAction) {
        history = [];
        if (request.action === 'page_summarize' && context?.id === request.context?.id) {
          context.summaryText = request.text;
          renderContext();
        }
      }
      history.push({role: 'user', content: request.input}, {role: 'assistant', content: request.text});
      activeRequest = null;
      resultActions(request);
      if (request.documentAction) chooseAction('chat');
      notice('Completed on this device.');
    } else failRequest('The model returned no text. Try a more specific instruction.');
    renderRuntime();
    refreshRuntime();
  } else if (type === 'request.cancelled' && id === activeRequest?.id) {
    failRequest('Stopped. Partial text is not applied.', true);
  } else if (type === 'error') {
    if (modelLoadId && id === modelLoadId) {
      queued = null;
      endModelLoad();
      notice(payload.message || 'Unable to load the model.', true);
      renderRuntime();
    } else if (!id || id === activeRequest?.id) failRequest(payload.message || 'Local request failed.');
  } else if (type === 'runtime.availability' && payload.available === false) {
    ready = false;
    renderRuntime();
  }
}
async function takePending() {
  const response = await sendIntegration('kidi.pending.take');
  if (!response.task) return;
  if (activeRequest) {
    notice('Finish or stop the current answer before attaching a new selection.');
    return;
  }
  const task = response.task.task || 'polish';
  await attachContext(response.task, task === 'page_read' ? 'chat' : task);
  if (task === 'page_read') notice('Article attached. Click Read article to begin narration; no chat model is needed.');
  else if (task === 'page_translate') notice('Choose a language, then click Translate.');
  else submit();
}

for (const [task, label] of Object.entries(LABELS))
  $('prompt-action').append(new Option(label, task));
$('composer').addEventListener('submit', event => { event.preventDefault(); submit(); });
$('message').addEventListener('input', () => renderRuntime());
$('message').addEventListener('keydown', event => {
  if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    submit();
  }
});
$('load-model').addEventListener('click', loadModel);
$('cancel').addEventListener('click', () => {
  if (activeRequest) backend.send('cancel', activeRequest.id);
});
$('new-chat').addEventListener('click', () => {
  clearConversation().then(() => { notice(); $('message').focus(); }).catch(error => notice(error.message, true));
});
$('clear-context').addEventListener('click', () => {
  clearConversation().catch(error => notice(error.message, true));
});
$('gmail-draft').addEventListener('change', () => { selectedDraft = $('gmail-draft').value; });
$('target-language').addEventListener('change', () => {
  try { localStorage.setItem(LANGUAGE_STORAGE, $('target-language').value); }
  catch (error) { notice(`Unable to save translation language: ${error.message}`, true); }
  renderRuntime();
});
$('speech-pause').addEventListener('click', () => {
  readAloud.togglePause().catch(error => notice(error.message, true));
});
$('speech-stop').addEventListener('click', () => {
  readAloud.stop().catch(error => notice(error.message, true));
});
$('speech-form').addEventListener('submit', event => {
  event.preventDefault();
  const language = languageFor($('speech-language').value);
  if (!pendingSpeech || !language) return notice('Choose the narration language first.', true);
  const {text} = pendingSpeech;
  pendingSpeech = null;
  speechApproved = true;
  try { localStorage.setItem(SPEECH_CONSENT, 'true'); }
  catch (error) { notice(`Read-aloud approval could not be saved: ${error.message}`, true); }
  $('speech-dialog').close();
  speechStartedAt = performance.now();
  readAloud.start({text, language: language.name, cacheIdentity: snapshot.components?.tts?.directory || ''})
    .catch(error => notice(error.message, true));
});
$('speech-decline').addEventListener('click', () => { pendingSpeech = null; $('speech-dialog').close(); });
$('speech-dialog').addEventListener('cancel', () => { pendingSpeech = null; });
$('enable-gmail').addEventListener('click', () => {
  api.permissions.request({origins: [GMAIL_ORIGIN]}).then(async granted => {
    if (!granted) return notice('Gmail access was not granted. You can still paste text into chat.');
    await refreshContext();
  }).catch(error => notice(error.message || String(error), true));
});
$('prompt-toggle').addEventListener('click', () => {
  const opening = $('prompt-editor').hidden;
  $('prompt-editor').hidden = !opening;
  $('prompt-toggle').setAttribute('aria-expanded', String(opening));
  if (opening) { promptAction = action; renderPrompt(); $('system-prompt').focus(); }
});
$('prompt-action').addEventListener('change', () => {
  promptAction = $('prompt-action').value;
  renderPrompt();
});
$('system-prompt').addEventListener('input', () => {
  const value = $('system-prompt').value.trim();
  if (!value) {
    $('prompt-state').textContent = 'Prompt cannot be empty. The previous prompt remains in use.';
    return;
  }
  if (new TextEncoder().encode(value).length > 16 * 1024) {
    $('prompt-state').textContent = 'Prompt exceeds the 16 KiB limit. The previous prompt remains in use.';
    return;
  }
  if (value === defaultPromptFor(promptAction)) delete promptOverrides[promptAction];
  else promptOverrides[promptAction] = value;
  if (savePrompt()) $('prompt-state').textContent = Object.hasOwn(promptOverrides, promptAction)
    ? 'Customized · saved automatically' : 'Configured default';
});
$('prompt-default').addEventListener('click', () => {
  delete promptOverrides[promptAction];
  const saved = savePrompt();
  renderPrompt();
  if (!saved) $('prompt-state').textContent = 'Default restored for this panel, but storage is unavailable.';
});
api.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message?.windowId !== undefined && message.windowId !== windowInfo.id) return false;
  if (message?.type === 'kidi.context.pending' || message?.type === 'kidi.context.error') {
    takePending().catch(error => notice(error.message, true));
    sendResponse({ok: true});
  } else if (message?.type === 'kidi.context.changed') {
    clearTimeout(refreshTimer);
    refreshTimer = setTimeout(refreshContext, 200);
  }
  return false;
});
addEventListener('focus', refreshContext);
addEventListener('pagehide', () => {
  clearInterval(progressTimer);
  clearInterval(speechTimer);
  clearTimeout(refreshTimer);
  readAloud.close().catch(error => console.error('Unable to stop extension speech:', error));
});
backend.connect({
  onOpen: () => renderRuntime(),
  onEvent: event,
  onPcm: frame => readAloud.handlePcm(frame),
  onError: error => notice(error.message || String(error), true),
  onClose: () => {
    ready = false;
    queued = null;
    endModelLoad();
    if (activeRequest) failRequest('The local runtime disconnected.');
    readAloud.stop().catch(error => notice(error.message, true));
    renderRuntime('Disconnected');
  },
}, {loadCachedModel: false});
renderPrompt();
renderRuntime();
renderContext();
refreshContext();
loadLanguages();
takePending().catch(error => { $('context-status').textContent = error.message; });
