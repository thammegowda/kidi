import {renderMarkdown, renderStreamingMarkdown} from './markdown.js';
import {backend} from './backend.js';
import {MAX_IMAGE_BYTES, MAX_IMAGES, readImage, removeUnusedImages, storeImage} from './images.js';
import {SpeechController} from './speech-controller.js';
import {renderSpeechHighlights} from './speech-highlights.js';

const $ = id => document.getElementById(id);
const STORAGE = 'kidi-app-v2';
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
  view: 'chat',
};
const welcome = $('welcome');
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
let runtimeLoadId;
let modelLoadProgress;
let cacheBusy = false;
let cachePath = [];
const speech = new SpeechController({
  send: (type, id, payload) => send(type, id, payload),
  nextWireStream: () => streamId++,
  onProgress: (current, payload) => updateTtsProgress(current.id, payload),
  onSegment: (current, segment) => {
    const stream = ttsStreams.get(current.streamId);
    if (!stream) return;
    stream.segments ||= new Map();
    stream.segments.set(segment.id, {...segment, status: 'ready'});
    renderSpeechHighlights(ttsStreams.values());
  },
  onPcm: (_current, frame) => handlePcm(frame),
  onCompleted: current => completeAudioStream(current.streamId),
  onCancelled: current => {
    const stream = ttsStreams.get(current.streamId);
    if (stream) finishTts(stream.button, current.streamId);
  },
  onError: (current, error) => audioFailure(current.streamId, error),
});
function formatCacheSize(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KiB`;
  return `${(bytes / 1024 ** 2).toFixed(1)} MiB`;
}
let settingsPage = 'llm';

function selectSettingsPage(page) {
  settingsPage = page;
  for (const button of document.querySelectorAll('[data-settings-page]')) {
    if (button.dataset.settingsPage === page) button.setAttribute('aria-current', 'page');
    else button.removeAttribute('aria-current');
  }
  for (const panel of document.querySelectorAll('[data-settings-panel]'))
    panel.hidden = panel.dataset.settingsPanel !== page;
  if (page === 'cache') refreshModelCache();
}

async function refreshModelCache(action) {
  if (cacheBusy) return;
  cacheBusy = true;
  $('refresh-model-cache').disabled = true;
  $('clear-model-cache').disabled = true;
  $('model-cache-files').replaceChildren();
  $('model-cache-status').textContent = 'Reading model cache...';
  try {
    if (action) await action();
    const files = await backend.listCachedFiles();
    const total = files.reduce((sum, file) => sum + file.bytes, 0);
    $('model-cache-status').textContent = `${files.length} files · ${formatCacheSize(total)}`;
    const root = {folders: new Map(), files: [], bytes: 0};
    for (const file of files) {
      const url = new URL(file.id);
      const parts = [url.origin, ...file.name.split('/').filter(Boolean)];
      const name = parts.pop();
      let folder = root;
      for (const part of parts) {
        if (!folder.folders.has(part))
          folder.folders.set(part, {folders: new Map(), files: [], bytes: 0, ids: []});
        folder = folder.folders.get(part);
        folder.bytes += file.bytes;
        folder.ids.push(file.id);
      }
      folder.files.push({...file, basename: name});
    }
    const renderFolder = (path) => {
      let folder = root;
      const resolved = [];
      for (const part of path) {
        if (!folder.folders.has(part)) break;
        folder = folder.folders.get(part);
        resolved.push(part);
      }
      cachePath = resolved;
      const parent = $('model-cache-files');
      parent.replaceChildren();
      const toolbar = document.createElement('nav');
      toolbar.className = 'cache-breadcrumbs';
      toolbar.setAttribute('aria-label', 'Cache folder path');
      const locationButton = (name, destination) => {
        const button = document.createElement('button');
        button.type = 'button';
        button.textContent = name;
        button.addEventListener('click', () => renderFolder(destination));
        toolbar.append(button);
        return button;
      };
      locationButton('Up', resolved.slice(0, -1)).disabled = !resolved.length;
      locationButton('Model cache', []);
      resolved.forEach((part, index) => locationButton(part, resolved.slice(0, index + 1)));
      parent.append(toolbar);
      const heading = document.createElement('div');
      heading.className = 'cache-explorer-heading';
      for (const title of ['Name', 'Size', 'Actions']) {
        const column = document.createElement('span');
        column.textContent = title;
        heading.append(column);
      }
      parent.append(heading);
      for (const [name, child] of [...folder.folders].sort(([a], [b]) => a.localeCompare(b))) {
        const row = document.createElement('div');
        row.className = 'cache-file cache-directory';
        const open = document.createElement('button');
        open.type = 'button';
        open.className = 'cache-open-directory';
        open.textContent = `${name}/`;
        open.setAttribute('aria-label', `Open folder ${name}`);
        open.addEventListener('click', () => renderFolder([...resolved, name]));
        const size = document.createElement('span');
        size.className = 'cache-folder-size';
        size.textContent = formatCacheSize(child.bytes);
        const remove = document.createElement('button');
        remove.type = 'button';
        remove.textContent = 'Delete';
        remove.setAttribute('aria-label', `Delete folder ${name}`);
        remove.addEventListener('click', () => {
          if (confirm(`Delete this cached folder recursively?\n${[...resolved, name].join('/')}\n${child.ids.length} files, ${formatCacheSize(child.bytes)}. All cached chunks beneath this folder will be removed.`))
            refreshModelCache(() => backend.deleteCachedFiles(child.ids));
        });
        row.append(open, size, remove);
        parent.append(row);
      }
      for (const file of folder.files.sort((a, b) => a.basename.localeCompare(b.basename))) {
      const row = document.createElement('div');
      row.className = 'cache-file';
      const label = document.createElement('span');
      label.textContent = file.basename;
      label.title = file.id;
      const size = document.createElement('span');
      size.className = 'cache-folder-size';
      size.textContent = formatCacheSize(file.bytes);
      const remove = document.createElement('button');
      remove.type = 'button';
      remove.textContent = 'Delete';
      remove.setAttribute('aria-label', `Delete cached ${file.name}`);
      remove.addEventListener('click', () => {
        if (confirm(`Delete this cached file and all its chunks?\n${file.name}`))
          refreshModelCache(() => backend.deleteCachedFile(file.id));
      });
      row.append(label, size, remove);
      parent.append(row);
      }
      if (!folder.folders.size && !folder.files.length) {
        const empty = document.createElement('p');
        empty.textContent = 'This folder is empty.';
        parent.append(empty);
      }
    };
    renderFolder(cachePath);
    if (!files.length) $('model-cache-status').textContent = 'No cached model files.';
  } catch (error) {
    $('model-cache-status').textContent = `Cache operation failed: ${error.message}`;
  } finally {
    cacheBusy = false;
    $('refresh-model-cache').disabled = false;
    $('clear-model-cache').disabled = false;
  }
}

function renderModelProgress() {
  const loading = Boolean(state.runtime.loading);
  $('model-loading').hidden = !loading;
  $('load-wasm-model').disabled = loading;
  const metrics = modelLoadProgress || {};
  const initializing = metrics.type === 'initializing';
  const total = metrics.totalBytes || 0;
  const loaded = metrics.loadedBytes || 0;
  const percent = total > 0 ? Math.min(100, Math.floor(loaded / total * 100)) : null;
  const stage = initializing ? 'Initializing model' :
    percent === null ? 'Preparing model' : `Loading model · ${percent}%`;
  $('model-loading-stage').textContent = stage;
  $('wasm-runtime-status').textContent = loading ? stage : state.ready ? 'Ready' : 'Offline';
  const bar = $('model-loading-progress');
  if (percent !== null && !initializing) { bar.max = total; bar.value = Math.min(loaded, total); }
  else bar.removeAttribute('value');
  const bytes = value => `${(value / (1024 * 1024)).toFixed(1)} MB`;
  $('model-loading-bytes').textContent = total > 0 ?
    `${bytes(loaded)} / ${bytes(total)} · Downloaded ${bytes(metrics.downloadedBytes || 0)} · Cached ${bytes(metrics.cachedBytes || 0)}` : '';
  $('model-loading-file').textContent = metrics.file || '';
}
let voiceMode = false;
let voiceEpoch = 0;
let asrFinalizing = false;
let voiceRestartTimer;
let draftImages = [];
const imageUrls = new Map();

function loadLocal() {
  try {
    let saved = JSON.parse(localStorage.getItem(STORAGE) || '{}');
    if (!Array.isArray(saved.chats)) {
      const desktop = JSON.parse(localStorage.getItem('kidi-desktop-v1') || '{}');
      const wasm = JSON.parse(localStorage.getItem('kidi-chats-v1') || '[]');
      const merged = new Map();
      for (const chat of [...(Array.isArray(desktop.chats) ? desktop.chats : []), ...(Array.isArray(wasm) ? wasm : [])])
        if (chat?.id && !merged.has(chat.id)) merged.set(chat.id, chat);
      saved = {
        chats: [...merged.values()],
        active: desktop.active || localStorage.getItem('kidi-active-chat-v1'),
        view: desktop.view,
        tts: desktop.tts,
      };
    }
    if (Array.isArray(saved.chats)) {
      state.chats = saved.chats.filter(chat => chat && typeof chat.id === 'string' &&
        typeof chat.title === 'string' && Array.isArray(chat.messages)).map(chat => ({
          ...chat,
          messages: chat.messages.filter(message => message && !message.pending &&
            ['user', 'assistant'].includes(message.role) && typeof message.content === 'string')
            .map(message => ({
              ...message,
              id: message.id || crypto.randomUUID(),
              images: Array.isArray(message.images) ? message.images : [],
              stats: message.stats && {
                ...message.stats,
                elapsed_ms: message.stats.elapsed_ms ?? message.stats.elapsedMs,
                decode_ms: message.stats.decode_ms ?? message.stats.decodeMs,
                decode_tokens: message.stats.decode_tokens ?? message.stats.decodeTokens ??
                  message.stats.tokenCount,
              },
            })),
        })).slice(0, 40);
    }
    state.active = typeof saved.active === 'string' ? saved.active : null;
    state.view = saved.view === 'reader' ? 'reader' : 'chat';
    if (saved.tts && typeof saved.tts === 'object') state.ttsSettings = saved.tts;
  } catch {}
  if (!state.chats.some(chat => chat.id === state.active)) state.active = state.chats[0]?.id || null;
}

function persist() {
  try {
    localStorage.setItem(STORAGE, JSON.stringify({
      chats: state.chats, active: state.active, view: state.view, tts: state.ttsSettings,
    }));
  }
  catch {}
}

async function imageUrl(image) {
  if (imageUrls.has(image.id)) return imageUrls.get(image.id);
  const url = URL.createObjectURL(await readImage(image));
  imageUrls.set(image.id, url);
  return url;
}

function renderImageList(target, images, removable = false) {
  target.replaceChildren();
  target.hidden = !images.length;
  for (const image of images) {
    const item = document.createElement('figure');
    item.className = 'attachment';
    const preview = document.createElement('img');
    preview.alt = image.name;
    imageUrl(image).then(url => { if (preview.isConnected) preview.src = url; })
      .catch(error => showNotice(error.message));
    item.append(preview);
    if (removable) {
      const remove = document.createElement('button');
      remove.type = 'button'; remove.textContent = '×';
      remove.setAttribute('aria-label', `Remove ${image.name}`);
      remove.addEventListener('click', () => {
        draftImages = draftImages.filter(candidate => candidate.id !== image.id);
        renderImageList($('attachments'), draftImages, true);
        renderRuntime();
      });
      item.append(remove);
    }
    target.append(item);
  }
}

async function attachImages(files) {
  const selected = [...files];
  const retained = [...draftImages, ...(activeChat()?.messages || []).flatMap(message => message.images || [])];
  if (retained.length + selected.length > MAX_IMAGES) {
    showNotice(`A conversation can contain at most ${MAX_IMAGES} images`);
    return;
  }
  try {
    let bytes = retained.reduce((sum, image) => sum + Number(image.size || 0), 0);
    for (const file of selected) {
      bytes += file.size;
      if (bytes > MAX_IMAGE_BYTES) throw new Error('Images exceed the 32 MiB conversation limit');
      draftImages.push(await storeImage(file, file.name));
    }
    renderImageList($('attachments'), draftImages, true);
    showNotice(); renderRuntime();
  } catch (error) {
    showNotice(error.message || 'Unable to attach image');
  }
}

function pruneImages() {
  const retained = [...draftImages, ...state.chats.flatMap(chat =>
    chat.messages.flatMap(message => message.images || []))];
  removeUnusedImages(retained).catch(error => showNotice(error.message));
}

function send(type, id, payload = {}) {
  backend.send(type, id, payload);
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

function connectBackend() {
  backend.connect({
    onOpen: () => { state.socketReady = true; renderRuntime(); },
    onEvent: handleEvent,
    onAudio: buffer => handleAudio(buffer).catch(error => audioFailure(playbackStreamId, error)),
    onPcm: frame => speech.handlePcm(frame) || handlePcm(frame),
    onClose: () => {
    speech.reset('The local speech transport disconnected');
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
    setTimeout(connectBackend, 750);
    },
    onError: error => showNotice(error.message || 'The local application connection failed'),
  });
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
    if (message.role === 'user' && !message.content) content.hidden = true;
    if (message.images?.length) {
      const images = document.createElement('div');
      images.className = 'message-images';
      renderImageList(images, message.images);
      body.append(images);
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
  renderSpeechHighlights(ttsStreams.values());
}

function renderRuntime() {
  const asrActive = activeRequest?.kind === 'asr';
  const playingVoiceReply = voiceMode && playbackStreamId !== null;
  const activeSpeech = ttsStreams.get(playbackStreamId);
  const pauseSeconds = recording ? (recording.silenceLimitMs / 1000).toFixed(1).replace(/\.0$/, '') : '';
  const listening = recording?.heardSpeech && voiceMode ?
    `Listening · pause ${pauseSeconds}s to send` : recording ? 'Listening for speech…' : '';
  const label = asrActive ? (recording ? listening :
    activeRequest.asrStatus || (asrFinalizing ? 'Finalizing dictation…' : 'Preparing speech recognition…')) :
    activeRequest?.kind === 'chat' && voiceMode ?
      (activeSpeech?.paused ? 'Speech paused' :
        activeSpeech?.bufferedFrames > 0 ? 'Speaking while writing…' : 'Kidi is thinking…') :
    activeRequest?.kind === 'tts' ? activeRequest.ttsStatus || 'Preparing speech…' :
    activeRequest?.kind === 'voice-tts' ? activeRequest.ttsStatus || 'Kidi is speaking…' :
    playingVoiceReply ? 'Kidi is speaking…' :
    !state.busy && activeSpeech?.paused ? 'Speech paused' :
    !state.busy && activeSpeech ? (activeSpeech.audioStarted ? 'Playing speech…' : 'Preparing speech…') :
    voiceMode ? 'Voice mode ready' :
    state.runtime.reloading ? 'Loading cached model…' :
    state.busy ? 'Generating locally…' : state.runtime.loading ?
      $('model-loading-stage').textContent : state.ready ?
        state.runtime.resident === false ? 'Model cached · loads on use' : executionLabel(state.backend) : 'No model loaded';
  $('runtime-label').textContent = label;
  $('runtime-dot').className = `runtime-dot${state.busy ? ' busy' : state.ready ? ' ready' : ''}`;
  $('send').disabled = (asrActive ? (!recording || sendAfterTranscription) :
    (state.busy || (!$('prompt').value.trim() && !draftImages.length))) || !state.ready || !state.socketReady;
  $('stop').hidden = !state.busy || asrActive || activeRequest?.kind === 'tts';
  $('send').hidden = state.busy && !asrActive;
  $('prompt').disabled = state.busy && !asrActive;
  $('prompt').readOnly = asrActive;
  $('mic').disabled = (state.busy && activeRequest?.kind !== 'asr') ||
    (asrActive && !recording) || (playbackStreamId !== null && !activeSpeech?.paused) ||
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

function selectView(view, updateRoute = true) {
  state.view = view === 'reader' ? 'reader' : 'chat';
  $('chat-view').hidden = state.view !== 'chat';
  $('reader-view').hidden = state.view !== 'reader';
  $('chat-navigation').hidden = state.view !== 'chat';
  $('sidebar').hidden = state.view !== 'chat';
  $('sidebar-backdrop').hidden = state.view !== 'chat';
  $('app-shell').classList.toggle('reader-active', state.view === 'reader');
  $('open-chat-view').classList.toggle('active', state.view === 'chat');
  $('open-reader-view').classList.toggle('active', state.view === 'reader');
  $('open-chat-view').setAttribute('aria-current', state.view === 'chat' ? 'page' : 'false');
  $('open-reader-view').setAttribute('aria-current', state.view === 'reader' ? 'page' : 'false');
  if (updateRoute) history.replaceState(null, '', state.view === 'reader' ? '#/reader' : '#/chat');
  persist();
  closeSidebar();
}

function restoreViewRoute() {
  selectView(location.hash.startsWith('#/reader') ? 'reader' : state.view, false);
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
  );
  $('asr-config-summary').replaceChildren(
    configSection('Speech recognition', [
      configRow('ASR', asr.enabled ? `${asr.accelerator} · ${asr.directory}` : 'Disabled', 'wide'),
      configRow('Language', asr.settings?.language || 'auto'),
      configRow('VAD threshold', asr.settings?.vad?.threshold ?? '—'),
      configRow('End silence', `${asr.settings?.vad?.silence_ms ?? '—'} ms`),
    ]),
  );
  $('tts-config-summary').replaceChildren(
    configSection('Speech synthesis', [
      configRow('TTS', tts.enabled ? `${tts.accelerator} · ${tts.directory}` : 'Disabled', 'wide'),
      configRow('CPU threads', tts.threads ?? '—'),
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
  $('tts-gpu-precision').value = settings.gpu_precision || 'w8a8';
  $('tts-availability').textContent = tts.enabled ? 'Available' : 'Model disabled';
  $('system-prompt').value = chat.system_prompt || '';
  const configured = chat.configured_system_prompt || '';
  $('system-prompt-status').textContent =
    (chat.system_prompt || '') === configured ? 'From configuration' : 'Modified for this session';
  const wasm = state.runtime.host === 'wasm';
  $('tts-gpu-precision-row').hidden = !wasm || state.runtime.requested_backend !== 'webgpu';
  document.querySelector('[data-settings-page="cache"]').hidden = !wasm;
  $('wasm-preferences').hidden = !wasm;
  selectSettingsPage(!wasm && settingsPage === 'cache' ? 'llm' : settingsPage);
  if (wasm) {
    $('wasm-model').value = state.runtime.model_id || $('wasm-model').value;
    $('wasm-backend').value = state.runtime.requested_backend || 'cpu';
    $('wasm-threads').value = String(state.runtime.threads || 1);
    $('wasm-tokens').value = String(state.runtime.maximum_tokens || 512);
    renderModelProgress();
  }
}

function initializeTtsSettings() {
  if (ttsSettingsInitialized) return;
  const defaults = state.components.tts?.settings || {};
  const saved = state.ttsSettings || {};
  const browserGpu = state.runtime.host === 'wasm' && state.runtime.requested_backend === 'webgpu';
  const savedPrecision = ['f32', 'mixed-f16', 'w8a8'].includes(saved.gpu_precision)
    ? saved.gpu_precision : defaults.gpu_precision || 'w8a8';
  const migratePrecision = browserGpu && saved.gpu_precision_version !== 1 && savedPrecision === 'f32';
  state.ttsSettings = {
    language: saved.language || defaults.language || 'English',
    steps: Number.isInteger(saved.steps) ? saved.steps : defaults.steps || 10,
    pace: Number.isFinite(saved.pace) ? saved.pace : defaults.pace || 1,
    gpu_precision: migratePrecision ? 'w8a8' : savedPrecision,
    gpu_precision_version: 1,
    voice: {...(defaults.voice || {}), ...(saved.voice || {})},
  };
  ttsSettingsInitialized = true;
  if (migratePrecision) queueMicrotask(persist);
}

function updateTtsSettings() {
  const steps = Number($('tts-steps').value);
  state.ttsSettings = {
    language: $('tts-language').value.trim() || 'English',
    steps: Number.isInteger(steps) && steps >= 1 && steps <= 128 ? steps : 10,
    pace: Number($('tts-pace').value),
    gpu_precision: $('tts-gpu-precision').value,
    gpu_precision_version: 1,
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

function ttsRequestOptions() {
  const options = {...state.ttsSettings};
  delete options.gpu_precision_version;
  if (state.runtime.host !== 'wasm') delete options.gpu_precision;
  return options;
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
  draftImages = []; renderImageList($('attachments'), draftImages, true);
  state.active = null; followOutput = true; selectView('chat'); persist(); showNotice(); render(); $('prompt').focus();
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
    renderSpeechHighlights(ttsStreams.values());
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

function setReadAloudState(button, state) {
  if (!button) return;
  const mode = state === 'playing' || state === 'paused' ? state : 'idle';
  button.classList.toggle('active', mode !== 'idle');
  button.classList.toggle('paused', mode === 'paused');
  const label = mode === 'playing' ? 'Pause reading response' :
    mode === 'paused' ? 'Resume reading response' : 'Read response aloud';
  button.setAttribute('aria-label', label);
  button.title = label;
}

function appendReadAloud(body, text) {
  const read = document.createElement('button');
  read.type = 'button'; read.className = 'message-action read-aloud';
  read.innerHTML = '<span class="read-aloud-icon" aria-hidden="true"></span>';
  setReadAloudState(read, 'idle');
  read.addEventListener('click', async () => {
    const stream = [...ttsStreams.entries()].find(([, value]) => value.button === read);
    if (stream) {
      if (stream[1].paused) resumeTts(stream[0]);
      else pauseTts(stream[0]);
    } else {
      for (const [id] of [...ttsStreams]) await stopTts(id);
      startTts(text, read);
    }
  });
  body.append(read);
}

function appendMessageActions(row, content, source) {
  const actions = document.createElement('div');
  actions.className = 'message-actions';
  if (state.components.tts?.enabled) appendReadAloud(actions, source);
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
  renderSpeechHighlights(ttsStreams.values());
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

function updateTtsProgress(id, payload) {
  if (activeRequest?.id !== id) return;
  if (ttsStreams.get(activeRequest.streamId)?.paused) {
    activeRequest.ttsStatus = 'Speech paused';
  } else if (payload.phase === 'loading' && payload.totalBytes > 0) {
    const percent = Math.min(100, Math.floor((payload.loadedBytes || 0) / payload.totalBytes * 100));
    activeRequest.ttsStatus = `Loading speech model · ${percent}%`;
  } else if (payload.phase === 'fallback') {
    activeRequest.ttsStatus = payload.message || 'Using threaded CPU speech…';
  } else if (payload.phase === 'synthesizing') {
    activeRequest.ttsStatus = payload.steps > 0 && payload.step > 0
      ? `Synthesizing speech · ${payload.step}/${payload.steps}`
      : 'Synthesizing speech…';
  }
  renderRuntime();
}

function handleEvent(event) {
  if (!event || event.v !== 1 || typeof event.type !== 'string') return;
  if (speech.handleEvent(event)) return;
  const payload = event.payload || {};
  if (event.type === 'session.ready') {
    state.components = payload.components || {};
    state.config = payload.config || state.config;
    state.runtime = payload.runtime || state.runtime;
    if (typeof payload.ready === 'boolean') state.ready = payload.ready;
    if (typeof payload.backend === 'string') state.backend = payload.backend;
    initializeTtsSettings();
    showNotice(); render();
    return;
  }
  if (event.type === 'runtime.loading') {
    runtimeLoadId = event.id;
    modelLoadProgress = null;
    state.ready = false;
    state.runtime = {...state.runtime, ...payload.runtime, loading: true};
    renderModelProgress(); renderRuntime();
    return;
  }
  if (event.type === 'runtime.availability') {
    state.runtime = {...state.runtime, resident: payload.resident, reloading: Boolean(payload.reloading)};
    if (payload.available === false) {
      state.ready = false;
      state.runtime.loading = false;
      showNotice(payload.message || 'Browser runtime is unavailable. Reload the model to recover.');
    }
    renderModelProgress(); renderRuntime();
    return;
  }
  if (event.type === 'asr.progress' && activeRequest?.id === event.id) {
    activeRequest.asrStatus = payload.state === 'ready' ? null :
      payload.message || payload.stage || 'Loading cached speech model…';
    renderRuntime();
    return;
  }
  if (event.type === 'runtime.progress' && event.id === runtimeLoadId) {
    state.runtime.loading = true;
    modelLoadProgress = payload;
    renderModelProgress(); renderRuntime();
    return;
  }
  if (event.type === 'runtime.ready' && event.id === runtimeLoadId) {
    runtimeLoadId = null;
    state.ready = true;
    state.backend = payload.backend || state.backend;
    state.runtime = {...state.runtime, ...(payload.runtime || {}), loading: false};
    state.components = payload.components || state.components;
    $('load-wasm-model').disabled = false;
    showNotice();
    renderConfig(); renderRuntime();
    return;
  }
  if (event.type === 'error' && event.id === runtimeLoadId) {
    runtimeLoadId = null;
    state.runtime.loading = false;
    renderModelProgress(); renderRuntime();
    $('load-wasm-model').disabled = false;
    $('wasm-runtime-status').textContent = 'Unable to load';
    showNotice(payload.message || 'Unable to load the browser model');
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
    const delta = payload.text || '';
    streamText(delta);
    if (activeRequest.voiceTurn) speech.append(event.id, delta);
    return;
  }
  if (event.type === 'chat.completed' && activeRequest?.id === event.id) {
    const speak = voiceMode && activeRequest.voiceTurn && activeRequest.voiceEpoch === voiceEpoch;
    const message = activeMessage();
    const streamed = message?.content || '';
    const completedText = payload.text || streamed;
    if (speak) {
      if (completedText.startsWith(streamed)) speech.append(event.id, completedText.slice(streamed.length), true);
      else if (completedText.trim() === streamed.trim()) speech.append(event.id, '', true);
      else speech.cancel(event.id, 'The streamed response changed before speech completed')
        .catch(error => showNotice(error.message));
    }
    if (message) {
      message.content = completedText;
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
        setReadAloudState(button, stream[1].paused ? 'paused' : 'playing');
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
  if (event.type === 'asr.progress' && activeRequest?.id === event.id) {
    if (payload.phase === 'fallback') showNotice(payload.message);
    return;
  }
  if (event.type === 'asr.partial' && activeRequest?.id === event.id) {
    if (asrFinalizing || !recording?.heardSpeech) return;
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
        }, voiceTurn ? 600 : 25);
      else if (voiceTurn) scheduleVoiceListening(voiceEpoch);
      else showNotice('No speech was recognized');
    }
    return;
  }
  if (event.type === 'tts.segment') {
    const stream = ttsStreams.get(payload.stream_id);
    if (stream) {
      stream.segments ||= new Map();
      stream.segments.set(payload.segment_id, {id: payload.segment_id, text: payload.text,
        start: payload.start, end: payload.end, status: 'ready'});
      renderSpeechHighlights(ttsStreams.values());
    }
    return;
  }
  if (event.type === 'tts.completed') {
    if (ttsStreams.has(payload.stream_id)) completeAudioStream(payload.stream_id);
    else if (activeRequest?.id === event.id) finishRequest();
    return;
  }
  if (event.type === 'tts.progress' && activeRequest?.id === event.id) {
    updateTtsProgress(event.id, payload);
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
  if (playback?.context.state === 'closed') {
    playback = null;
    playbackPromise = null;
  }
  if (playback) {
    if (playback.context.state === 'suspended') await playback.context.resume();
    return playback;
  }
  if (!playbackPromise) playbackPromise = (async () => {
    const context = sampleRate ? new AudioContext({sampleRate}) : new AudioContext();
    try {
      await Promise.all([context.resume(),
        context.audioWorklet.addModule(new URL('./playback-worklet.js', import.meta.url))]);
      const node = new AudioWorkletNode(context, 'kidi-playback', {outputChannelCount: [1]});
      node.connect(context.destination);
      node.port.onmessage = event => {
        const data = event.data;
        if (playback?.node !== node || data.streamId !== playbackStreamId) return;
        const stream = ttsStreams.get(data.streamId);
        if (!stream) return;
        if (data.type === 'buffered') {
          const wasPlaying = stream.bufferedFrames > 0;
          stream.playedFrames = data.playedFrames;
          stream.bufferedFrames = (stream.sentFrames || 0) - data.playedFrames;
          if (stream.bufferedFrames <= playerBufferLimit(context.sampleRate))
            for (const resolve of stream.capacityWaiters?.splice(0) || []) resolve();
          if (wasPlaying !== (stream.bufferedFrames > 0)) renderRuntime();
        } else if (data.type === 'segment-started' || data.type === 'segment-ended') {
          const segment = stream.segments?.get(data.segmentId);
          if (segment) segment.status = data.type === 'segment-started' ? 'playing' : 'played';
          renderSpeechHighlights(ttsStreams.values());
          renderRuntime();
        } else if (data.type === 'drained' && stream.completed) finishTts(stream.button, data.streamId);
      };
      playback = {context, node};
      const failed = error => {
        if (playback?.node !== node) return;
        const id = playbackStreamId;
        playback = null;
        playbackPromise = null;
        if (id !== null) audioFailure(id, error);
        if (context.state !== 'closed')
          context.close().catch(closeError => console.error('Unable to close failed playback context:', closeError));
      };
      node.onprocessorerror = () => failed(new Error('Speech audio processor failed'));
      context.onstatechange = () => {
        if (context.state === 'closed' || context.state === 'interrupted')
          failed(new Error('Speech audio playback was interrupted'));
      };
      return playback;
    } catch (error) {
      try { await context.close(); }
      catch (closeError) { console.error('Unable to close failed playback context:', closeError); }
      throw error;
    }
  })().catch(error => {
    playbackPromise = null;
    throw error;
  });
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
  if (length !== buffer.byteLength - 24 || length % 2) throw new Error('Invalid speech audio frame');
  const pcm = new Int16Array(buffer, 24, length / 2);
  const samples = new Float32Array(pcm.length);
  for (let index = 0; index < pcm.length; index++) samples[index] = pcm[index] / 32768;
  const frame = {streamId: id, sequence: view.getUint32(12, true), sampleRate, samples};
  if (speech.handlePcm(frame)) return;
  if (!ttsStreams.has(id)) return;
  return handlePcm(frame);
}

const playerBufferLimit = rate => rate * 6;

function handlePcm({streamId: id, sequence, sampleRate, samples, segmentId}) {
  const stream = ttsStreams.get(id);
  if (!stream) return;
  if (!(samples instanceof Float32Array) || !samples.length || samples.some(value => !Number.isFinite(value)) ||
      !Number.isInteger(sampleRate) || sampleRate <= 0 || !Number.isInteger(sequence) ||
      sequence !== (stream.audioSequence || 0) || stream.completed)
    throw new Error('Invalid or out-of-order speech PCM');
  stream.audioSequence = (sequence + 1) >>> 0;
  stream.delivery = (stream.delivery || Promise.resolve()).then(async () => {
    if (ttsStreams.get(id) !== stream) return;
    const player = await ensurePlayback(sampleRate);
    if (ttsStreams.get(id) !== stream) return;
    if (!stream.audioStarted) {
      player.node.port.postMessage({type: 'start', streamId: id});
      if (stream.paused) player.node.port.postMessage({type: 'pause', streamId: id});
      stream.audioStarted = true;
    }
    playbackStreamId = id;
    if (activeRequest?.streamId === id) {
      activeRequest.ttsStatus = activeRequest.voiceTurn ? 'Kidi is speaking…' : 'Playing speech…';
      renderRuntime();
    }
    let output = resample(samples, sampleRate, player.context.sampleRate);
    if (output.byteOffset || output.byteLength !== output.buffer.byteLength) output = output.slice();
    stream.sentFrames = (stream.sentFrames || 0) + output.length;
    stream.bufferedFrames = stream.sentFrames - (stream.playedFrames || 0);
    player.node.port.postMessage({type: 'chunk', streamId: id, segmentId, samples: output.buffer}, [output.buffer]);
    if (stream.bufferedFrames > playerBufferLimit(player.context.sampleRate))
      await new Promise(resolve => { (stream.capacityWaiters ||= []).push(resolve); });
  });
  return stream.delivery;
}

function completeAudioStream(id) {
  const stream = ttsStreams.get(id);
  if (!stream) return;
  stream.completed = true;
  Promise.resolve(stream.delivery).then(() => {
    if (ttsStreams.get(id) !== stream) return;
    if (!stream.audioStarted || !playback) throw new Error('Speech completed without audio');
    playback.node.port.postMessage({type: 'end', streamId: id});
  }).catch(error => audioFailure(id, error));
}

function audioFailure(id, error) {
  const stream = ttsStreams.get(id);
  if (stream?.requestId) speech.cancel(stream.requestId, error.message).catch(cancelError => showNotice(cancelError.message));
  if (stream?.voiceTurn) stopVoiceMode(false);
  else if (stream) {
    playback?.node.port.postMessage({type: 'reset'});
    finishTts(stream.button, id);
  }
  showNotice(error.message || 'Speech playback failed');
}

async function beginCapture(id) {
  const media = await navigator.mediaDevices.getUserMedia({audio: {
    channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true,
  }});
  const context = new AudioContext();
  await context.audioWorklet.addModule(new URL('./capture-worklet.js', import.meta.url));
  const source = context.createMediaStreamSource(media);
  const node = new AudioWorkletNode(context, 'kidi-capture');
  const silence = context.createGain(); silence.gain.value = 0;
  source.connect(node); node.connect(silence); silence.connect(context.destination);
  const vad = state.components.asr?.settings?.vad || {};
  const minimumSpeechMs = Number(vad.minimum_speech_ms || 250);
  recording = {
    id, sequence: 0, context, media, source, node,
    heardSpeech: false, speechMs: 0, silenceMs: 0,
    preRoll: [], preRollSamples: 0,
    threshold: Number(vad.threshold || .012),
    minimumSpeechMs,
    silenceLimitMs: Number(vad.silence_ms || 1500),
    preRollLimitSamples: Math.max(8000, Math.ceil(minimumSpeechMs * 16)),
  };
  const sendFrame = (current, values) => {
    if (backend.bufferedAmount > 1024 * 1024) {
      showNotice('Microphone stream cannot keep up'); stopCapture();
      return false;
    }
    const pcm = new Uint8Array(values.buffer, values.byteOffset, values.byteLength);
    const header = audioHeader(1, id, current.sequence++, 16000, pcm.byteLength);
    const frame = new Uint8Array(header.byteLength + pcm.byteLength);
    frame.set(header); frame.set(pcm, header.byteLength); backend.sendBinary(frame);
    return true;
  };
  node.port.onmessage = event => {
    const current = recording;
    if (!current) return;
    const values = new Int16Array(event.data);
    let energy = 0;
    for (const value of values) energy += (value / 32768) ** 2;
    const rms = Math.sqrt(energy / Math.max(1, values.length));
    const frameMs = values.length / 16;
    const active = rms >= current.threshold;
    if (!current.heardSpeech) {
      current.preRoll.push(values);
      current.preRollSamples += values.length;
      while (current.preRollSamples > current.preRollLimitSamples) {
        current.preRollSamples -= current.preRoll[0].length;
        current.preRoll.shift();
      }
      current.speechMs = active ? current.speechMs + frameMs : 0;
      if (current.speechMs < current.minimumSpeechMs) return;
      current.heardSpeech = true;
      renderRuntime();
      const buffered = current.preRoll;
      current.preRoll = [];
      current.preRollSamples = 0;
      for (const bufferedFrame of buffered)
        if (!sendFrame(current, bufferedFrame)) return;
    } else {
      if (active) {
        current.speechMs += frameMs;
        current.silenceMs = 0;
      } else {
        current.silenceMs += frameMs;
      }
      if (!sendFrame(current, values)) return;
    }
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
    if (current.heardSpeech) {
      asrFinalizing = true;
      send('asr.stop', activeRequest.id);
    } else {
      showNotice('No speech was detected');
      send('cancel', activeRequest.id);
    }
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
      tts_options: voiceMode ? ttsRequestOptions() : undefined,
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
  const speechId = speech.active?.id;
  voiceMode = false;
  voiceEpoch++;
  clearTimeout(voiceRestartTimer);
  if (recording) stopCapture(false);
  asrFinalizing = false;
  playback?.node.port.postMessage({type: 'reset'});
  for (const [stream, value] of ttsStreams) finishTts(value.button, stream);
  if (speechId) speech.cancel(speechId, 'Voice mode stopped').catch(error => showNotice(error.message));
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
  ttsStreams.set(audioId, {button, completed: false, voiceTurn, requestId: id, voiceEpoch: voiceTurn ? epoch : null,
    messageId: button?.closest('[data-message-id]')?.dataset.messageId, paused: false});
  playbackStreamId = audioId;
  setReadAloudState(button, 'playing');
  state.busy = true; showNotice(); renderRuntime();
  ensurePlayback().catch(error => showNotice(error.message));
  try {
    const options = ttsRequestOptions();
    speech.start({
      id, streamId: audioId, text, options, voice: voiceTurn, finished: true,
      cacheIdentity: JSON.stringify([state.runtime.host, state.backend, state.components.tts?.directory]),
    });
  }
  catch (error) {
    finishTts(button, audioId);
    showNotice(error.message); finishRequest();
  }
}

function finishTts(button, stream) {
  const entry = ttsStreams.get(stream);
  const voiceTurn = entry?.voiceTurn;
  const epoch = entry?.voiceEpoch;
  setReadAloudState(button, 'idle');
  for (const resolve of entry?.capacityWaiters?.splice(0) || []) resolve();
  ttsStreams.delete(stream);
  renderSpeechHighlights(ttsStreams.values());
  if (playbackStreamId === stream) playbackStreamId = null;
  if (activeRequest?.kind !== 'chat' &&
      (activeRequest?.streamId === stream || activeRequest?.id === entry?.requestId)) finishRequest();
  renderRuntime();
  if (voiceTurn && voiceMode && epoch === voiceEpoch) scheduleVoiceListening(epoch);
}

async function stopTts(stream) {
  const current = ttsStreams.get(stream);
  if (!current) return;
  playback?.node.port.postMessage({type: 'reset'});
  await speech.cancel(current.requestId, 'Speech playback replaced');
  if (ttsStreams.has(stream)) finishTts(current.button, stream);
}

function pauseTts(stream) {
  const current = ttsStreams.get(stream);
  if (!current || current.paused) return;
  current.paused = true;
  playback?.node.port.postMessage({type: 'pause', streamId: stream});
  setReadAloudState(current.button, 'paused');
  const foreground = activeRequest &&
    (activeRequest.streamId === stream || current.requestId === activeRequest.id);
  if (foreground)
    activeRequest.ttsStatus = 'Speech paused';
  if (foreground && !current.voiceTurn) {
    current.detached = true;
    finishRequest();
    return;
  }
  renderRuntime();
}

function resumeTts(stream) {
  const current = ttsStreams.get(stream);
  if (!current || !current.paused) return;
  current.paused = false;
  playback?.context.resume().catch(error => audioFailure(stream, error));
  playback?.node.port.postMessage({type: 'resume', streamId: stream});
  setReadAloudState(current.button, 'playing');
  if (activeRequest && (activeRequest.streamId === stream || current.requestId === activeRequest.id))
    activeRequest.ttsStatus = current.voiceTurn ? 'Kidi is speaking…' : 'Playing speech…';
  renderRuntime();
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
  if ((!text && !draftImages.length) || state.busy || !state.ready) return;
  let chat = activeChat();
  if (!chat) {
    chat = {
      id: crypto.randomUUID(),
      title: titleFor(text || draftImages[0]?.name || 'Image'),
      updatedAt: Date.now(),
      messages: [],
    };
    state.chats.unshift(chat); state.active = chat.id;
  }
  const requestId = crypto.randomUUID();
  const responseId = crypto.randomUUID();
  chat.messages.push(
    {id: crypto.randomUUID(), role: 'user', content: text, images: draftImages},
    {id: responseId, role: 'assistant', content: '', pending: true},
  );
  chat.updatedAt = Date.now();
  draftImages = [];
  renderImageList($('attachments'), draftImages, true);
  $('prompt').value = ''; resizePrompt(); state.busy = true; showNotice(); render();
  try {
    activeRequest = {
      id: requestId, kind: 'chat', messageId: responseId,
      voiceTurn: voiceMode, voiceEpoch: voiceMode ? voiceEpoch : null,
    };
    if (voiceMode) {
      const audioId = streamId++;
      activeRequest.streamId = audioId;
      ttsStreams.set(audioId, {
        button: null, completed: false, paused: false, voiceTurn: true,
        requestId, voiceEpoch, messageId: responseId,
      });
      playbackStreamId = audioId;
      const options = ttsRequestOptions();
      speech.start({
        id: requestId, streamId: audioId, text: '', options, voice: true, finished: false,
        cacheIdentity: JSON.stringify([state.runtime.host, state.backend, state.components.tts?.directory]),
      });
      ensurePlayback().catch(error => audioFailure(audioId, error));
    }
    const requestMessages = chat.messages.filter(message => !message.pending)
      .map(({role, content, images}) =>
        role === 'user' && images?.length ? {role, content, images} : {role, content});
    const preparedMessages = backend.prepareMessages
      ? await backend.prepareMessages(requestMessages, requestId)
      : requestMessages;
    send('chat.start', requestId, {
      messages: preparedMessages,
      speak: false,
    });
  } catch (error) {
    if (speech.active?.id === requestId) await speech.cancel(requestId, error.message);
    chat.messages = chat.messages.filter(message => message.id !== responseId);
    showNotice(error.message); finishRequest(); render();
  }
}

async function cancelGeneration() {
  try {
    if (!activeRequest) return;
    if (speech.active?.id === activeRequest.id)
      await speech.cancel(activeRequest.id, 'Speech generation cancelled');
    send('cancel', activeRequest.id);
  }
  catch (error) { showNotice(error.message); }
}

function resizePrompt() {
  renderRuntime();
}

loadLocal();
persist();
$('new-chat').addEventListener('click', newChat);
$('composer').addEventListener('submit', sendMessage);
$('stop').addEventListener('click', cancelGeneration);
$('mic').addEventListener('click', toggleDictation);
$('voice-mode').addEventListener('click', toggleVoiceMode);
$('prompt').addEventListener('input', resizePrompt);
$('attach').addEventListener('click', () => $('image-files').click());
$('image-files').addEventListener('change', event => {
  attachImages(event.target.files);
  event.target.value = '';
});
$('composer').addEventListener('paste', event => {
  const files = [...(event.clipboardData?.files || [])].filter(file => file.type.startsWith('image/'));
  if (files.length) { event.preventDefault(); attachImages(files); }
});
$('composer').addEventListener('dragover', event => event.preventDefault());
$('composer').addEventListener('drop', event => {
  event.preventDefault();
  const files = [...(event.dataTransfer?.files || [])].filter(file => file.type.startsWith('image/'));
  if (files.length) attachImages(files);
});
$('prompt').addEventListener('keydown', event => {
  if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); sendMessage(); }
});
$('menu').addEventListener('click', () => $('sidebar').classList.toggle('open'));
$('close-menu').addEventListener('click', closeSidebar);
$('sidebar-backdrop').addEventListener('click', closeSidebar);
$('open-chat-view').addEventListener('click', () => selectView('chat'));
$('open-reader-view').addEventListener('click', () => selectView('reader'));
$('open-config').addEventListener('click', () => { renderConfig(); $('config-dialog').showModal(); });
$('close-config').addEventListener('click', () => $('config-dialog').close());
$('apply-system-prompt').addEventListener('click', () => applySystemPrompt($('system-prompt').value));
$('reset-system-prompt').addEventListener('click', () =>
  applySystemPrompt(state.components.chat?.configured_system_prompt || ''));
$('load-wasm-model').addEventListener('click', () => {
  if (runtimeLoadId) return;
  runtimeLoadId = crypto.randomUUID();
  $('load-wasm-model').disabled = true;
  $('wasm-runtime-status').textContent = 'Loading';
  send('runtime.load', runtimeLoadId, {
    model: $('wasm-model').value.trim(),
    backend: $('wasm-backend').value,
    threads: Number($('wasm-threads').value),
    maximum_tokens: Number($('wasm-tokens').value),
  });
});
for (const button of document.querySelectorAll('[data-settings-page]'))
  button.addEventListener('click', () => selectSettingsPage(button.dataset.settingsPage));
$('refresh-model-cache').addEventListener('click', () => refreshModelCache());
$('clear-model-cache').addEventListener('click', () => {
  if (confirm('Clear all cached models? Chat, speech recognition, and TTS models will need to be fetched again. Conversations and images will not be deleted.'))
    refreshModelCache(() => backend.clearModelCache());
});
$('config-dialog').addEventListener('click', event => {
  if (event.target === $('config-dialog')) $('config-dialog').close();
});
for (const id of ['tts-language', 'tts-steps', 'tts-pace', 'tts-gender', 'tts-age', 'tts-pitch', 'tts-style', 'tts-accent',
  'tts-gpu-precision'])
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
addEventListener('hashchange', restoreViewRoute);
for (const suggestion of document.querySelectorAll('[data-prompt]')) {
  suggestion.addEventListener('click', () => {
    $('prompt').value = suggestion.dataset.prompt; resizePrompt(); $('prompt').focus();
  });
}

try {
  const status = await backend.status();
  state.ready = status.ready; state.backend = status.backend || ''; state.model = status.model || '';
  state.config = status.config || ''; state.runtime = status.runtime || {}; state.components = status.components || {};
  initializeTtsSettings();
} catch (error) { showNotice(error.message); }
connectBackend();
render();
restoreViewRoute();
pruneImages();
