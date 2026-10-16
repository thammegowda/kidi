import {renderMarkdown, renderStreamingMarkdown} from './markdown.js';

const $ = id => document.getElementById(id);
const STORAGE = 'kidi-desktop-v1';
const state = {
  ready: false,
  socketReady: false,
  busy: false,
  backend: '',
  model: '',
  config: '',
  runtime: {},
  components: {},
  ttsSettings: null,
  chats: [],
  active: null,
};
const welcome = $('welcome');
let socket;
let activeRequest;
let streamId = 1;
let recording;
let playback;
const ttsStreams = new Map();
let playbackStreamId = null;
let followOutput = true;
let historyQuery = '';
let rebuildingMessages = false;
let sendAfterTranscription = false;
let ttsSettingsInitialized = false;
let systemPromptRequestId;
let voiceMode = false;
let voiceEpoch = 0;
let asrFinalizing = false;
let voiceRestartTimer;

function loadLocal() {
  try {
    const saved = JSON.parse(localStorage.getItem(STORAGE) || '{}');
    if (Array.isArray(saved.chats)) {
      state.chats = saved.chats.filter(chat => chat && typeof chat.id === 'string' &&
        typeof chat.title === 'string' && Array.isArray(chat.messages)).map(chat => ({
          ...chat,
          messages: chat.messages.filter(message => message && !message.pending &&
            ['user', 'assistant'].includes(message.role) && typeof message.content === 'string')
            .map(message => ({...message, id: message.id || crypto.randomUUID()})),
        })).slice(0, 40);
    }
    state.active = typeof saved.active === 'string' ? saved.active : null;
    if (saved.tts && typeof saved.tts === 'object') state.ttsSettings = saved.tts;
  } catch {}
  if (!state.chats.some(chat => chat.id === state.active)) state.active = state.chats[0]?.id || null;
}

function persist() {
  try {
    localStorage.setItem(STORAGE, JSON.stringify({
      chats: state.chats, active: state.active, tts: state.ttsSettings,
    }));
  }
  catch {}
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: {'Content-Type': 'application/json', ...(options.headers || {})},
  });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.error?.message || `Request failed (${response.status})`);
  return body;
}

function send(type, id, payload = {}) {
  if (!socket || socket.readyState !== WebSocket.OPEN) throw new Error('The local session is reconnecting');
  socket.send(JSON.stringify({v: 1, id, type, payload}));
}

function audioHeader(kind, id, sequence, sampleRate, payloadBytes) {
  const header = new ArrayBuffer(24);
  const bytes = new Uint8Array(header);
  bytes.set([0x4b, 0x44, 0x41, 0x31, 1, kind], 0);
  const view = new DataView(header);
  view.setUint32(8, id, true);
  view.setUint32(12, sequence, true);
  view.setUint32(16, sampleRate, true);
  view.setUint32(20, payloadBytes, true);
  return bytes;
}

function connectSocket() {
  const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
  socket = new WebSocket(`${protocol}//${location.host}/ws`);
  socket.binaryType = 'arraybuffer';
  socket.addEventListener('open', () => { state.socketReady = true; renderRuntime(); });
  socket.addEventListener('message', event => {
    if (typeof event.data === 'string') {
      try { handleEvent(JSON.parse(event.data)); }
      catch { showNotice('Received an invalid local protocol message'); }
    } else {
      handleAudio(event.data).catch(error => showNotice(error.message));
    }
  });
  socket.addEventListener('close', () => {
    if (activeRequest?.kind === 'chat') {
      const chat = activeChat();
      if (chat?.messages.at(-1)?.pending) {
        const removed = chat.messages.pop();
        document.querySelector(`[data-message-id="${CSS.escape(removed.id)}"]`)?.remove();
      }
    }
    stopCapture(false);
    playback?.node.port.postMessage({type: 'reset'});
    for (const [stream, value] of ttsStreams) finishTts(value.button, stream);
    sendAfterTranscription = false;
    voiceMode = false; asrFinalizing = false;
    systemPromptRequestId = null;
    clearTimeout(voiceRestartTimer);
    state.socketReady = false; state.busy = false; activeRequest = null;
    showNotice('The local session disconnected. Reconnecting…'); renderHistory(); renderRuntime();
    setTimeout(connectSocket, 750);
  });
  socket.addEventListener('error', () => socket.close());
}

function activeChat() { return state.chats.find(chat => chat.id === state.active); }
function titleFor(text) {
  const title = text.replace(/\s+/g, ' ').trim();
  return title.length > 48 ? `${title.slice(0, 45)}…` : title;
}
function showNotice(message = '') {
  $('notice').textContent = message;
  $('notice').hidden = !message;
}
function formatStats(stats) {
  if (!stats) return '';
  const speed = stats.decode_ms > 0 ? `${(stats.decode_tokens * 1000 / stats.decode_ms).toFixed(1)} tok/s` : '';
  return [`${stats.decode_tokens} tokens`, `${(stats.elapsed_ms / 1000).toFixed(1)} s`, speed].filter(Boolean).join(' · ');
}

function executionLabel(value) {
  if (['a_gpu', 'apple_gpu', 'mps'].includes(value)) return 'Metal GPU';
  if (value === 'cpu' || value === 'ynnpack') return 'CPU';
  if (value === 'vulkan') return 'Vulkan GPU';
  return value || 'Model ready';
}

function renderHistory() {
  const list = $('chat-list');
  list.replaceChildren();
  for (const chat of [...state.chats].sort((a, b) => b.updatedAt - a.updatedAt)) {
    if (historyQuery && !chat.title.toLowerCase().includes(historyQuery)) continue;
    const row = document.createElement('div');
    row.className = `chat-item${chat.id === state.active ? ' active' : ''}`;
    const open = document.createElement('button');
    open.type = 'button'; open.textContent = chat.title; open.title = chat.title;
    open.addEventListener('click', () => {
      state.active = chat.id; followOutput = true; persist(); render(); $('sidebar').classList.remove('open');
    });
    const remove = document.createElement('button');
    remove.type = 'button'; remove.className = 'delete-chat'; remove.textContent = '×';
    remove.setAttribute('aria-label', `Delete ${chat.title}`);
    remove.addEventListener('click', event => {
      event.stopPropagation();
      state.chats = state.chats.filter(candidate => candidate.id !== chat.id);
      if (state.active === chat.id) {
        state.active = state.chats[0]?.id || null;
        followOutput = true;
      }
      persist(); render();
    });
    row.append(open, remove); list.append(row);
  }
}

function renderMessages() {
  const container = $('messages');
  const previousTop = container.scrollTop;
  rebuildingMessages = true;
  container.replaceChildren();
  const chat = activeChat();
  if (!chat?.messages.length) {
    container.append(welcome);
    welcome.hidden = false;
    requestAnimationFrame(() => { rebuildingMessages = false; });
    return;
  }
  for (const message of chat.messages) {
    const article = document.createElement('article');
    article.className = `message ${message.role}`;
    article.dataset.messageId = message.id;
    article.setAttribute('aria-label', message.role === 'assistant' ? 'Kidi' : 'You');
    const avatar = document.createElement('div');
    avatar.className = 'avatar'; avatar.textContent = 'K'; avatar.setAttribute('aria-hidden', 'true');
    const body = document.createElement('div');
    body.className = 'message-body';
    const content = document.createElement('div');
    content.className = 'message-content';
    if (message.pending) {
      content.className = 'thinking';
      content.innerHTML = '<i></i><i></i><i></i>';
    } else if (message.role === 'assistant') {
      renderMarkdown(content, message.content);
    } else {
      content.textContent = message.content;
    }
    const bubbleRow = document.createElement('div');
    bubbleRow.className = 'message-bubble-row';
    bubbleRow.append(content);
    if (message.role === 'assistant' && !message.pending)
      appendMessageActions(bubbleRow, content, message.content);
    body.append(bubbleRow);
    const footer = document.createElement('div');
    footer.className = 'message-footer';
    if (message.stats) {
      const stats = document.createElement('div');
      stats.className = 'message-stats'; stats.textContent = formatStats(message.stats); footer.append(stats);
    }
    if (footer.childElementCount) body.append(footer);
    article.append(avatar, body); container.append(article);
  }
  requestAnimationFrame(() => {
    container.scrollTop = followOutput ? container.scrollHeight : previousTop;
    requestAnimationFrame(() => { rebuildingMessages = false; });
  });
}

function renderRuntime() {
  const asrActive = activeRequest?.kind === 'asr';
  const playingVoiceReply = voiceMode && playbackStreamId !== null;
  const label = asrActive ? (recording ? 'Listening…' :
    asrFinalizing ? 'Finalizing dictation…' : 'Preparing speech recognition…') :
    activeRequest?.kind === 'chat' && voiceMode ? 'Kidi is thinking…' :
    activeRequest?.kind === 'tts' && voiceMode ? 'Kidi is speaking…' :
    playingVoiceReply ? 'Kidi is speaking…' :
    voiceMode ? 'Voice mode ready' :
    state.busy ? 'Generating locally…' : state.ready ? executionLabel(state.backend) : 'No model loaded';
  $('runtime-label').textContent = label;
  $('runtime-dot').className = `runtime-dot${state.busy ? ' busy' : state.ready ? ' ready' : ''}`;
  $('send').disabled = (asrActive ? (!recording || sendAfterTranscription) :
    (state.busy || !$('prompt').value.trim())) || !state.ready || !state.socketReady;
  $('stop').hidden = !state.busy || asrActive || activeRequest?.kind === 'tts';
  $('send').hidden = state.busy && !asrActive;
  $('prompt').disabled = state.busy && !asrActive;
  $('prompt').readOnly = asrActive;
  $('mic').disabled = (state.busy && activeRequest?.kind !== 'asr') ||
    (asrActive && !recording) || playbackStreamId !== null ||
    !state.socketReady || !state.components.asr?.enabled;
  $('voice-mode').disabled = (state.busy && !voiceMode) || (playbackStreamId !== null && !voiceMode) || !state.socketReady ||
    !state.components.asr?.enabled || !state.components.tts?.enabled;
  $('voice-mode').classList.toggle('active', voiceMode);
  $('voice-mode').setAttribute('aria-label', voiceMode ? 'Stop voice mode' : 'Start voice mode');
  $('voice-mode').title = voiceMode ? 'Stop voice mode' : 'Start voice mode';
}

function render() {
  const chat = activeChat();
  $('conversation-title').textContent = chat?.title || 'New conversation';
  renderHistory(); renderMessages(); renderRuntime();
}

function configRow(label, value, className = '') {
  const row = document.createElement('div');
  row.className = `config-row ${className}`.trim();
  const term = document.createElement('dt'); term.textContent = label;
  const detail = document.createElement('dd'); detail.textContent = String(value ?? '—'); detail.title = detail.textContent;
  row.append(term, detail);
  return row;
}

function configSection(title, rows) {
  const section = document.createElement('section');
  const heading = document.createElement('h3'); heading.textContent = title;
  const list = document.createElement('dl'); list.append(...rows);
  section.append(heading, list);
  return section;
}

function renderConfig() {
  const target = $('config-summary');
  const chat = state.components.chat || {};
  const asr = state.components.asr || {};
  const tts = state.components.tts || {};
  target.replaceChildren(
    configSection('Runtime', [
      configRow('Configuration', state.config, 'wide'),
      configRow('Threads', state.runtime.threads),
      configRow('Concurrency', state.runtime.concurrency),
      configRow('Speech preload', state.runtime.preload_speech ? 'Enabled' : 'Disabled'),
      configRow('Speech idle unload', `${state.runtime.idle_model_timeout_seconds ?? '—'} seconds`),
    ]),
    configSection('Chat', [
      configRow('Model', chat.directory, 'wide'),
      configRow('Accelerator', chat.accelerator),
      configRow('Precision', chat.precision),
      configRow('KV cache', chat.kv_cache_precision),
      configRow('Metal prefill cache', `${chat.metal_prefill_cache_mb ?? '—'} MiB`),
      configRow('Context', `${chat.context_tokens ?? '—'} tokens`),
      configRow('Response limit', `${chat.maximum_tokens ?? '—'} tokens`),
      configRow('Prefix cache', `${chat.prefix_cache_mb ?? '—'} MiB`),
    ]),
    configSection('Speech', [
      configRow('ASR', asr.enabled ? `${asr.accelerator} · ${asr.directory}` : 'Disabled', 'wide'),
      configRow('VAD threshold', asr.settings?.vad?.threshold ?? '—'),
      configRow('End silence', `${asr.settings?.vad?.silence_ms ?? '—'} ms`),
      configRow('TTS', tts.enabled ? `${tts.accelerator} · ${tts.directory}` : 'Disabled', 'wide'),
      configRow('TTS Metal cache', `${tts.metal_prefill_cache_mb ?? '—'} MiB`),
      configRow('Language mode', tts.settings?.language || 'auto'),
      configRow('Reference language', tts.settings?.reference_language || 'English'),
      configRow('Voice reference', tts.settings?.reference_text || 'Model default', 'wide prompt'),
    ]),
  );
  const settings = state.ttsSettings || {};
  const voice = settings.voice || {};
  $('tts-language').value = settings.language || 'English';
  $('tts-steps').value = String(settings.steps || 10);
  const pace = Number(settings.pace || 1);
  $('tts-pace').value = pace >= 1.175 ? '1.2' : pace >= 1.075 ? '1.15' : pace >= .95 ? '1' : '0.9';
  $('tts-gender').value = voice.gender || 'female';
  $('tts-age').value = voice.age || 'young adult';
  $('tts-pitch').value = voice.pitch || 'moderate';
  $('tts-style').value = voice.style || '';
  $('tts-accent').value = voice.accent || '';
  $('tts-availability').textContent = tts.enabled ? 'Available' : 'Model disabled';
  $('system-prompt').value = chat.system_prompt || '';
  const configured = chat.configured_system_prompt || '';
  $('system-prompt-status').textContent =
    (chat.system_prompt || '') === configured ? 'From configuration' : 'Modified for this session';
}

function initializeTtsSettings() {
  if (ttsSettingsInitialized) return;
  const defaults = state.components.tts?.settings || {};
  const saved = state.ttsSettings || {};
  state.ttsSettings = {
    language: saved.language || defaults.language || 'English',
    steps: Number.isInteger(saved.steps) ? saved.steps : defaults.steps || 10,
    pace: Number.isFinite(saved.pace) ? saved.pace : defaults.pace || 1,
    voice: {...(defaults.voice || {}), ...(saved.voice || {})},
  };
  ttsSettingsInitialized = true;
}

function updateTtsSettings() {
  const steps = Number($('tts-steps').value);
  state.ttsSettings = {
    language: $('tts-language').value.trim() || 'English',
    steps: Number.isInteger(steps) && steps >= 1 && steps <= 128 ? steps : 10,
    pace: Number($('tts-pace').value),
    voice: {
      gender: $('tts-gender').value,
      age: $('tts-age').value,
      pitch: $('tts-pitch').value,
      style: $('tts-style').value,
      accent: $('tts-accent').value,
    },
  };
  persist();
}

function applySystemPrompt(prompt) {
  if (systemPromptRequestId) return;
  systemPromptRequestId = crypto.randomUUID();
  $('apply-system-prompt').disabled = true;
  $('reset-system-prompt').disabled = true;
  $('system-prompt-status').textContent = 'Applying…';
  try {
    send('settings.system_prompt.set', systemPromptRequestId, {prompt});
  } catch (error) {
    systemPromptRequestId = null;
    $('apply-system-prompt').disabled = false;
    $('reset-system-prompt').disabled = false;
    $('system-prompt-status').textContent = 'Not applied';
    showNotice(error.message);
  }
}

function newChat() {
  state.active = null; followOutput = true; persist(); showNotice(); render(); closeSidebar(); $('prompt').focus();
}

function closeSidebar() { $('sidebar').classList.remove('open'); }

function activeMessage() {
  if (!activeRequest?.messageId) return null;
  return activeChat()?.messages.find(message => message.id === activeRequest.messageId);
}

function streamText(delta) {
  const message = activeMessage();
  if (!message) return;
  message.content += delta;
  const content = document.querySelector(`[data-message-id="${CSS.escape(message.id)}"] .thinking, ` +
    `[data-message-id="${CSS.escape(message.id)}"] .message-content`);
  if (content) {
    content.className = 'message-content streaming';
    renderStreamingMarkdown(content, message.content);
  }
  if (followOutput) {
    requestAnimationFrame(() => {
      $('messages').scrollTop = $('messages').scrollHeight;
    });
  }
}

function finishRequest() {
  state.busy = false; activeRequest = null; renderRuntime(); $('prompt').focus();
}

function setReadAloudState(button, active) {
  if (!button) return;
  button.classList.toggle('active', active);
  button.setAttribute('aria-label', active ? 'Stop reading response' : 'Read response aloud');
  button.title = active ? 'Stop reading response' : 'Read response aloud';
}

function appendReadAloud(body, content) {
  const read = document.createElement('button');
  read.type = 'button'; read.className = 'message-action read-aloud';
  read.innerHTML = '<span class="read-aloud-icon" aria-hidden="true"></span>';
  setReadAloudState(read, false);
  read.addEventListener('click', () => {
    const stream = [...ttsStreams.entries()].find(([, value]) => value.button === read);
    if (stream) stopTts(stream[0]);
    else startTts(content.textContent, read);
  });
  body.append(read);
}

function appendMessageActions(row, content, source) {
  const actions = document.createElement('div');
  actions.className = 'message-actions';
  if (state.components.tts?.enabled) appendReadAloud(actions, content);
  const copy = document.createElement('button');
  copy.type = 'button'; copy.className = 'message-action copy-message';
  copy.innerHTML = '<span class="copy-message-icon" aria-hidden="true"></span>';
  copy.setAttribute('aria-label', 'Copy response');
  copy.title = 'Copy response';
  copy.addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText(source);
      copy.classList.add('copied');
      copy.setAttribute('aria-label', 'Response copied');
      copy.title = 'Copied';
      setTimeout(() => {
        copy.classList.remove('copied');
        copy.setAttribute('aria-label', 'Copy response');
        copy.title = 'Copy response';
      }, 1200);
    } catch {
      showNotice('Unable to copy the response');
    }
  });
  actions.append(copy);
  row.append(actions);
}

function finalizeAssistant(message) {
  const article = document.querySelector(`[data-message-id="${CSS.escape(message.id)}"]`);
  if (!article) return;
  const content = article.querySelector('.message-content, .thinking');
  content.className = 'message-content';
  renderMarkdown(content, message.content);
  const body = content.closest('.message-body');
  const bubbleRow = content.closest('.message-bubble-row');
  if (bubbleRow && !bubbleRow.querySelector('.message-actions'))
    appendMessageActions(bubbleRow, content, message.content);
  let footer = body.querySelector('.message-footer');
  if (!footer) {
    footer = document.createElement('div');
    footer.className = 'message-footer';
    body.append(footer);
  }
  let stats = footer.querySelector('.message-stats');
  if (!stats) {
    stats = document.createElement('div');
    stats.className = 'message-stats';
    footer.append(stats);
  }
  stats.textContent = formatStats(message.stats);
  if (followOutput) requestAnimationFrame(() => { $('messages').scrollTop = $('messages').scrollHeight; });
}

function handleEvent(event) {
  if (!event || event.v !== 1 || typeof event.type !== 'string') return;
  const payload = event.payload || {};
  if (event.type === 'session.ready') {
    state.components = payload.components || {};
    state.config = payload.config || state.config;
    state.runtime = payload.runtime || state.runtime;
    initializeTtsSettings();
    showNotice(); render();
    return;
  }
  if (event.type === 'settings.system_prompt.updated' && event.id === systemPromptRequestId) {
    state.components.chat = {...(state.components.chat || {}), system_prompt: payload.prompt || ''};
    systemPromptRequestId = null;
    $('apply-system-prompt').disabled = false;
    $('reset-system-prompt').disabled = false;
    renderConfig();
    return;
  }
  if (event.type === 'error' && event.id === systemPromptRequestId) {
    systemPromptRequestId = null;
    $('apply-system-prompt').disabled = false;
    $('reset-system-prompt').disabled = false;
    $('system-prompt-status').textContent = 'Not applied';
    showNotice(payload.message || 'Unable to update the system prompt');
    return;
  }
  if (event.type === 'chat.delta' && activeRequest?.id === event.id) {
    streamText(payload.text || '');
    return;
  }
  if (event.type === 'chat.completed' && activeRequest?.id === event.id) {
    const speak = voiceMode && activeRequest.voiceTurn && activeRequest.voiceEpoch === voiceEpoch;
    const message = activeMessage();
    if (message) {
      message.content = payload.text || message.content;
      message.pending = false;
      message.stats = payload.stats;
      const chat = activeChat(); chat.updatedAt = Date.now(); persist();
      finalizeAssistant(message);
    }
    renderHistory();
    if (speak && message) {
      const article = document.querySelector(`[data-message-id="${CSS.escape(message.id)}"]`);
      const button = article?.querySelector('.read-aloud');
      const stream = [...ttsStreams.entries()].find(([, value]) => value.requestId === event.id);
      if (stream) {
        stream[1].button = button;
        setReadAloudState(button, true);
        activeRequest.kind = 'voice-tts';
        activeRequest.streamId = stream[0];
        renderRuntime();
      } else {
        finishRequest();
        scheduleVoiceListening(voiceEpoch);
      }
    } else finishRequest();
    return;
  }
  if (event.type === 'asr.started' && activeRequest?.id === event.id) {
    beginCapture(payload.stream_id).catch(error => {
      showNotice(error.message); send('cancel', event.id); finishRequest();
    });
    return;
  }
  if (event.type === 'asr.partial' && activeRequest?.id === event.id) {
    $('prompt').value = `${payload.stable || ''}${payload.provisional || ''}`; resizePrompt();
    return;
  }
  if (event.type === 'asr.completed' && activeRequest?.id === event.id) {
    const voiceTurn = voiceMode && activeRequest.voiceEpoch === voiceEpoch;
    const epoch = activeRequest.voiceEpoch;
    const submit = sendAfterTranscription || voiceTurn;
    sendAfterTranscription = false;
    asrFinalizing = false;
    stopCapture(false); $('prompt').value = payload.text || ''; finishRequest(); resizePrompt();
    if (submit) {
      if ($('prompt').value.trim())
        setTimeout(() => {
          if (!voiceTurn || (voiceMode && epoch === voiceEpoch)) sendMessage();
        }, 25);
      else if (voiceTurn) scheduleVoiceListening(voiceEpoch);
      else showNotice('No speech was recognized');
    }
    return;
  }
  if (event.type === 'tts.completed') {
    const stream = ttsStreams.get(payload.stream_id);
    if (stream) {
      playback?.node.port.postMessage({type: 'end'});
      stream.completed = true;
    }
    if (activeRequest?.id === event.id) finishRequest();
    return;
  }
  if (event.type === 'voice.audio.started' && activeRequest?.id === event.id) {
    const streamId = payload.stream_id;
    playbackStreamId = streamId;
    ttsStreams.set(streamId, {
      button: null, completed: false, voiceTurn: true,
      voiceEpoch: activeRequest.voiceEpoch, requestId: event.id,
    });
    ensurePlayback().catch(error => showNotice(error.message));
    return;
  }
  if (event.type === 'voice.audio.completed' && activeRequest?.id === event.id) {
    const stream = ttsStreams.get(payload.stream_id);
    if (stream) {
      stream.completed = true;
      playback?.node.port.postMessage({type: 'end'});
    }
    return;
  }
  if (event.type === 'voice.audio.error' && activeRequest?.id === event.id) {
    const stream = [...ttsStreams.entries()].find(([, value]) => value.requestId === event.id);
    playback?.node.port.postMessage({type: 'reset'});
    if (stream) finishTts(stream[1].button, stream[0]);
    showNotice(payload.message || 'Voice playback failed');
    stopVoiceMode(false);
    finishRequest();
    return;
  }
  if (event.type === 'voice.prepare.error' && activeRequest?.id === event.id) {
    showNotice(payload.message || 'Unable to prepare voice playback');
    stopVoiceMode();
    return;
  }
  if ((event.type === 'error' || event.type === 'request.cancelled') && activeRequest?.id === event.id) {
    if (activeRequest.kind === 'chat') {
      const chat = activeChat();
      if (chat?.messages.at(-1)?.pending) {
        const removed = chat.messages.pop();
        document.querySelector(`[data-message-id="${CSS.escape(removed.id)}"]`)?.remove();
      }
    }
    if (activeRequest.kind === 'asr') stopCapture(false);
    if (activeRequest.kind === 'tts' || activeRequest.kind === 'voice-tts') {
      const stream = ttsStreams.get(activeRequest.streamId);
      playback?.node.port.postMessage({type: 'reset'});
      if (stream) finishTts(stream.button, activeRequest.streamId);
    }
    if (event.type === 'error') showNotice(payload.message || 'Inference failed');
    sendAfterTranscription = false;
    asrFinalizing = false;
    if (voiceMode) stopVoiceMode(false);
    renderHistory(); finishRequest();
  }
}

function resample(samples, sourceRate, targetRate) {
  if (sourceRate === targetRate) return samples;
  const output = new Float32Array(Math.max(1, Math.floor(samples.length * targetRate / sourceRate)));
  const scale = sourceRate / targetRate;
  for (let index = 0; index < output.length; index++) {
    const position = index * scale;
    const left = Math.floor(position);
    const fraction = position - left;
    output[index] = samples[left] * (1 - fraction) +
      samples[Math.min(left + 1, samples.length - 1)] * fraction;
  }
  return output;
}

let playbackPromise;
async function ensurePlayback(sampleRate) {
  if (playback) return playback;
  if (!playbackPromise) playbackPromise = (async () => {
    const context = sampleRate ? new AudioContext({sampleRate}) : new AudioContext();
    const resumed = context.resume();
    await context.audioWorklet.addModule('/playback-worklet.js');
    const node = new AudioWorkletNode(context, 'kidi-playback', {outputChannelCount: [1]});
    node.connect(context.destination);
    node.port.onmessage = event => {
      if (event.data.type !== 'drained' || playbackStreamId === null) return;
      const stream = ttsStreams.get(playbackStreamId);
      if (stream) finishTts(stream.button, playbackStreamId);
    };
    await resumed;
    playback = {context, node};
    return playback;
  })();
  return playbackPromise;
}

async function handleAudio(buffer) {
  if (!(buffer instanceof ArrayBuffer) || buffer.byteLength < 24) return;
  const bytes = new Uint8Array(buffer);
  if (bytes[0] !== 0x4b || bytes[1] !== 0x44 || bytes[2] !== 0x41 || bytes[3] !== 0x31 ||
      bytes[4] !== 1 || bytes[5] !== 2) return;
  const view = new DataView(buffer);
  const id = view.getUint32(8, true);
  const sampleRate = view.getUint32(16, true);
  const length = view.getUint32(20, true);
  if (!ttsStreams.has(id) || length !== buffer.byteLength - 24 || length % 2) return;
  playbackStreamId = id;
  const pcm = new Int16Array(buffer, 24, length / 2);
  const samples = new Float32Array(pcm.length);
  for (let index = 0; index < pcm.length; index++) samples[index] = pcm[index] / 32768;
  const player = await ensurePlayback(sampleRate);
  const output = resample(samples, sampleRate, player.context.sampleRate);
  player.node.port.postMessage({type: 'chunk', samples: output.buffer}, [output.buffer]);
}

async function beginCapture(id) {
  const media = await navigator.mediaDevices.getUserMedia({audio: {
    channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true,
  }});
  const context = new AudioContext();
  await context.audioWorklet.addModule('/capture-worklet.js');
  const source = context.createMediaStreamSource(media);
  const node = new AudioWorkletNode(context, 'kidi-capture');
  const silence = context.createGain(); silence.gain.value = 0;
  source.connect(node); node.connect(silence); silence.connect(context.destination);
  const vad = state.components.asr?.settings?.vad || {};
  recording = {
    id, sequence: 0, context, media, source, node,
    heardSpeech: false, speechMs: 0, silenceMs: 0,
    threshold: Number(vad.threshold || .012),
    minimumSpeechMs: Number(vad.minimum_speech_ms || 250),
    silenceLimitMs: Number(vad.silence_ms || 800),
  };
  node.port.onmessage = event => {
    const current = recording;
    if (!current || socket.bufferedAmount > 1024 * 1024) {
      if (recording) { showNotice('Microphone stream cannot keep up'); stopCapture(); }
      return;
    }
    const pcm = new Uint8Array(event.data);
    const values = new Int16Array(event.data);
    let energy = 0;
    for (const value of values) energy += (value / 32768) ** 2;
    const rms = Math.sqrt(energy / Math.max(1, values.length));
    const frameMs = values.length / 16;
    if (rms >= current.threshold) {
      current.speechMs += frameMs; current.silenceMs = 0;
      if (current.speechMs >= current.minimumSpeechMs) current.heardSpeech = true;
    } else if (current.heardSpeech) {
      current.silenceMs += frameMs;
    }
    const header = audioHeader(1, id, current.sequence++, 16000, pcm.byteLength);
    const frame = new Uint8Array(header.byteLength + pcm.byteLength);
    frame.set(header); frame.set(pcm, header.byteLength); socket.send(frame);
    if (voiceMode && current.heardSpeech && current.silenceMs >= current.silenceLimitMs) stopCapture();
  };
  await context.resume();
  $('mic').classList.add('recording'); $('mic').setAttribute('aria-label', 'Stop dictation');
  renderRuntime();
}

function stopCapture(notify = true) {
  if (!recording) return;
  const current = recording; recording = null;
  current.source.disconnect(); current.node.disconnect();
  for (const track of current.media.getTracks()) track.stop();
  current.context.close();
  $('mic').classList.remove('recording'); $('mic').setAttribute('aria-label', 'Start dictation');
  if (notify && activeRequest?.kind === 'asr') {
    asrFinalizing = true;
    send('asr.stop', activeRequest.id);
  }
  renderRuntime();
}

function startDictation(epoch = voiceEpoch) {
  if (state.busy) return;
  const id = crypto.randomUUID();
  const audioId = streamId++;
  activeRequest = {id, kind: 'asr', streamId: audioId, voiceEpoch: voiceMode ? epoch : null};
  sendAfterTranscription = false;
  asrFinalizing = false;
  state.busy = true; showNotice(); renderRuntime();
  try {
    send('asr.start', id, {
      stream_id: audioId,
      voice_mode: voiceMode,
      tts_options: voiceMode ? state.ttsSettings : undefined,
    });
  }
  catch (error) { showNotice(error.message); finishRequest(); }
}

function toggleDictation() {
  if (recording) return stopCapture();
  startDictation();
}

function scheduleVoiceListening(epoch = voiceEpoch) {
  clearTimeout(voiceRestartTimer);
  if (!voiceMode || epoch !== voiceEpoch) return;
  voiceRestartTimer = setTimeout(() => {
    if (voiceMode && epoch === voiceEpoch && !state.busy) startDictation(epoch);
  }, 250);
}

function stopVoiceMode(cancel = true) {
  const requestId = activeRequest?.id;
  voiceMode = false;
  voiceEpoch++;
  clearTimeout(voiceRestartTimer);
  if (recording) stopCapture(false);
  asrFinalizing = false;
  playback?.node.port.postMessage({type: 'reset'});
  for (const [stream, value] of ttsStreams) finishTts(value.button, stream);
  if (cancel && requestId) send('cancel', requestId);
  renderRuntime();
}

function toggleVoiceMode() {
  if (voiceMode) return stopVoiceMode();
  if (state.busy) return;
  voiceMode = true;
  voiceEpoch++;
  const epoch = voiceEpoch;
  showNotice();
  renderRuntime();
  ensurePlayback().catch(error => {
    showNotice(error.message);
    stopVoiceMode();
  });
  startDictation(epoch);
}

function startTts(text, button, voiceTurn = false, epoch = voiceEpoch) {
  if (state.busy) return;
  const id = crypto.randomUUID();
  const audioId = streamId++;
  activeRequest = {id, kind: 'tts', streamId: audioId, voiceTurn, voiceEpoch: voiceTurn ? epoch : null};
  ttsStreams.set(audioId, {button, completed: false, voiceTurn, voiceEpoch: voiceTurn ? epoch : null});
  playbackStreamId = audioId;
  setReadAloudState(button, true);
  state.busy = true; showNotice(); renderRuntime();
  ensurePlayback().catch(error => showNotice(error.message));
  try { send('tts.start', id, {stream_id: audioId, text, options: state.ttsSettings}); }
  catch (error) {
    finishTts(button, audioId);
    showNotice(error.message); finishRequest();
  }
}

function finishTts(button, stream) {
  const entry = ttsStreams.get(stream);
  const voiceTurn = entry?.voiceTurn;
  const epoch = entry?.voiceEpoch;
  setReadAloudState(button, false);
  ttsStreams.delete(stream);
  if (playbackStreamId === stream) playbackStreamId = null;
  if (voiceTurn && activeRequest?.id === entry?.requestId) finishRequest();
  renderRuntime();
  if (voiceTurn && voiceMode && epoch === voiceEpoch) scheduleVoiceListening(epoch);
}

function stopTts(stream) {
  const current = ttsStreams.get(stream);
  if (!current) return;
  playback?.node.port.postMessage({type: 'reset'});
  if (activeRequest && (activeRequest.streamId === stream || current.requestId === activeRequest.id))
    send('cancel', activeRequest.id);
  finishTts(current.button, stream);
}

async function sendMessage(event) {
  event?.preventDefault();
  if (activeRequest?.kind === 'asr') {
    sendAfterTranscription = true;
    stopCapture();
    renderRuntime();
    return;
  }
  const text = $('prompt').value.trim();
  if (!text || state.busy || !state.ready) return;
  let chat = activeChat();
  if (!chat) {
    chat = {id: crypto.randomUUID(), title: titleFor(text), updatedAt: Date.now(), messages: []};
    state.chats.unshift(chat); state.active = chat.id;
  }
  const requestId = crypto.randomUUID();
  const responseId = crypto.randomUUID();
  chat.messages.push(
    {id: crypto.randomUUID(), role: 'user', content: text},
    {id: responseId, role: 'assistant', content: '', pending: true},
  );
  chat.updatedAt = Date.now();
  $('prompt').value = ''; resizePrompt(); state.busy = true; showNotice(); render();
  try {
    const requestMessages = chat.messages.filter(message => !message.pending)
      .map(({role, content}) => ({role, content}));
    activeRequest = {
      id: requestId, kind: 'chat', messageId: responseId,
      voiceTurn: voiceMode, voiceEpoch: voiceMode ? voiceEpoch : null,
    };
    send('chat.start', requestId, {
      messages: requestMessages,
      speak: voiceMode,
      tts_options: voiceMode ? state.ttsSettings : undefined,
    });
  } catch (error) {
    chat.messages.pop();
    showNotice(error.message); finishRequest(); render();
  }
}

async function cancelGeneration() {
  try { if (activeRequest) send('cancel', activeRequest.id); }
  catch (error) { showNotice(error.message); }
}

function resizePrompt() {
  renderRuntime();
}

loadLocal();
$('new-chat').addEventListener('click', newChat);
$('composer').addEventListener('submit', sendMessage);
$('stop').addEventListener('click', cancelGeneration);
$('mic').addEventListener('click', toggleDictation);
$('voice-mode').addEventListener('click', toggleVoiceMode);
$('prompt').addEventListener('input', resizePrompt);
$('prompt').addEventListener('keydown', event => {
  if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); sendMessage(); }
});
$('menu').addEventListener('click', () => $('sidebar').classList.toggle('open'));
$('close-menu').addEventListener('click', closeSidebar);
$('sidebar-backdrop').addEventListener('click', closeSidebar);
$('open-config').addEventListener('click', () => { renderConfig(); $('config-dialog').showModal(); });
$('close-config').addEventListener('click', () => $('config-dialog').close());
$('apply-system-prompt').addEventListener('click', () => applySystemPrompt($('system-prompt').value));
$('reset-system-prompt').addEventListener('click', () =>
  applySystemPrompt(state.components.chat?.configured_system_prompt || ''));
$('config-dialog').addEventListener('click', event => {
  if (event.target === $('config-dialog')) $('config-dialog').close();
});
for (const id of ['tts-language', 'tts-steps', 'tts-pace', 'tts-gender', 'tts-age', 'tts-pitch', 'tts-style', 'tts-accent'])
  $(id).addEventListener('change', updateTtsSettings);
$('messages').addEventListener('scroll', () => {
  if (rebuildingMessages) return;
  const messages = $('messages');
  const distance = messages.scrollHeight - messages.clientHeight - messages.scrollTop;
  followOutput = distance <= 48;
});
$('chat-search').addEventListener('input', event => {
  historyQuery = event.target.value.trim().toLowerCase();
  renderHistory();
});
addEventListener('keydown', event => {
  if (event.key === 'Escape') closeSidebar();
});
for (const suggestion of document.querySelectorAll('[data-prompt]')) {
  suggestion.addEventListener('click', () => {
    $('prompt').value = suggestion.dataset.prompt; resizePrompt(); $('prompt').focus();
  });
}

try {
  const status = await api('/api/status');
  state.ready = status.ready; state.backend = status.backend || ''; state.model = status.model || '';
  state.config = status.config || ''; state.runtime = status.runtime || {}; state.components = status.components || {};
  initializeTtsSettings();
} catch (error) { showNotice(error.message); }
connectSocket();
render();
