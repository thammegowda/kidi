import {clearModelCache, deleteCachedModel, isModelCached, listCachedModels, MAX_WASM_MEMORY, webGpuSupport} from './model-cache.mjs';
import {renderMarkdown} from './markdown.mjs';
import {resampleAudio} from './speech.mjs';
import {storeImage, readImage, removeUnusedImages, MAX_IMAGES, MAX_IMAGE_BYTES} from './images.mjs';

const CHAT_STORAGE = 'kidi-chats-v1';
const ACTIVE_CHAT_STORAGE = 'kidi-active-chat-v1';
const MODEL_STORAGE = 'kidi-model-source-v1';
const SETTINGS_STORAGE = 'kidi-settings-v1';
const MODEL_ID = /^[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+$/;
const NEW_CHAT = '__new__';
const element = id => document.getElementById(id);
const megabytes = bytes => `${(bytes / 1e6).toFixed(1)} MB`;
const cacheSize = bytes => bytes < 1024 ? `${bytes} B` :
    bytes < 1024 ** 2 ? `${(bytes / 1024).toFixed(bytes < 10 * 1024 ? 1 : 0)} KB` : megabytes(bytes);
const settingsDialog = element('settings-dialog');
let worker;
let ready = false;
let busy = false;
let stopping = false;
let loading = false;
let reply;
let runtimeVersion = 0;
let currentStats;
let generationStarted;
let liveTimer;
let recording;
let recordingTimer;
let recordingDeadline;
let speechWorker;
let speechSession;
let speechReady = false;
let speechModel = '';
let speechThreads = 0;
let visionSupported = false;
let draftImages = [];
let preparingImages = false;
let encodingImages = '';
const imageUrls = new Map();
const preferences = {};

try {
    const saved = JSON.parse(localStorage.getItem(SETTINGS_STORAGE) || '{}');
    if (['cpu', 'webgpu'].includes(saved?.backend)) {
        preferences.backend = saved.backend;
        element('backend').value = saved.backend;
    }
    for (const id of ['threads', 'tokens', 'image-pixels']) {
        const input = element(id);
        const value = saved?.[id];
        if (Number.isInteger(value) && value >= Number(input.min) && value <= Number(input.max)) {
            preferences[id] = value;
            input.value = String(value);
        }
    }
    if (typeof saved?.speechModel === 'string' && MODEL_ID.test(saved.speechModel)) {
        preferences.speechModel = saved.speechModel;
        element('speech-model').value = saved.speechModel;
    }
} catch {}

function savePreference(id) {
    const input = element(id);
    const value = Number(input.value);
    if (!Number.isInteger(value) || value < Number(input.min) || value > Number(input.max)) return;
    preferences[id] = value;
    try { localStorage.setItem(SETTINGS_STORAGE, JSON.stringify(preferences)); } catch {}
}

try {
    const source = localStorage.getItem(MODEL_STORAGE);
    if (source) {
        const pinned = /^https:\/\/huggingface\.co\/([^/]+\/[^/]+)\/resolve\/[a-f0-9]{40}\/config\.json$/.exec(source);
        element('manifest').value = pinned?.[1] || source;
    }
} catch {}

function loadChats() {
    try {
        const stored = JSON.parse(localStorage.getItem(CHAT_STORAGE) || '[]');
        if (!Array.isArray(stored)) return [];
        return stored.filter(chat => chat && typeof chat.id === 'string' && typeof chat.title === 'string' &&
            Array.isArray(chat.messages)).map(chat => ({...chat, messages: chat.messages.filter(message =>
                message && ['user', 'assistant'].includes(message.role) && typeof message.content === 'string')}))
            .slice(0, 50);
    } catch { return []; }
}
let chats = loadChats();
let storedActiveChat = null;
try { storedActiveChat = localStorage.getItem(ACTIVE_CHAT_STORAGE); } catch {}
let currentChatId = storedActiveChat === NEW_CHAT ? null : storedActiveChat;
if (currentChatId && !chats.some(chat => chat.id === currentChatId)) currentChatId = chats[0]?.id || null;
if (storedActiveChat === null) currentChatId = chats[0]?.id || null;
let conversation = chats.find(chat => chat.id === currentChatId)?.messages || [];

function attachmentError(message = '') {
    element('attachment-error').textContent = message;
    element('attachment-error').hidden = !message;
}
async function pruneImages() {
    if (preparingImages) return;
    const retained = [...draftImages, ...chats.flatMap(chat => chat.messages.flatMap(message => message.images || []))];
    try {
        await removeUnusedImages(retained);
        const ids = new Set(retained.map(image => image.id));
        for (const [id, url] of imageUrls) if (!ids.has(id)) { URL.revokeObjectURL(url); imageUrls.delete(id); }
    } catch (error) { attachmentError(error.message); }
}
async function imageUrl(image) {
    if (imageUrls.has(image.id)) return imageUrls.get(image.id);
    const blob = await readImage(image);
    if (!imageUrls.has(image.id)) imageUrls.set(image.id, URL.createObjectURL(blob));
    return imageUrls.get(image.id);
}
function renderImages(container, images, removable = false) {
    container.replaceChildren();
    container.hidden = !images.length;
    for (const image of images) {
        const item = document.createElement('figure');item.className = 'attachment';
        const preview = document.createElement('button');preview.type = 'button';preview.className = 'attachment-preview';
        preview.title = `View ${image.name}`;
        const picture = document.createElement('img');picture.alt = image.name;
        preview.append(picture);item.append(preview);
        imageUrl(image).then(url => { if (picture.isConnected) picture.src = url; }).catch(error => {
            if (picture.isConnected) { picture.alt = 'Image unavailable'; preview.disabled = true; preview.title = error.message; }
        });
        preview.addEventListener('click', async () => {
            try { element('image-preview').src = await imageUrl(image);element('image-preview').alt = image.name;element('image-dialog').showModal(); }
            catch (error) { attachmentError(error.message); }
        });
        if (removable) {
            const remove = document.createElement('button');remove.type = 'button';remove.className = 'icon remove-image';
            remove.title = `Remove ${image.name}`;remove.setAttribute('aria-label', remove.title);
            const icon = document.createElement('img');icon.src = './icons/x.svg';icon.alt = '';remove.append(icon);
            remove.addEventListener('click', () => {
                if (busy || preparingImages) return;
                draftImages = draftImages.filter(candidate => candidate !== image);
                renderImages(element('attachments'), draftImages, true);attachmentError();pruneImages();controls();
            });
            item.append(remove);
        }
        container.append(item);
    }
}
async function attachImages(files) {
    if (busy || preparingImages || recording) return;
    if (!ready || !visionSupported) { attachmentError('Load a model with image support first');return; }
    const selected = [...files];
    const existing = [...draftImages, ...conversation.flatMap(message => message.images || [])];
    if (existing.length + selected.length > MAX_IMAGES) { attachmentError('At most eight images per conversation');return; }
    preparingImages = true;controls();attachmentError();
    const added = [];
    try {
        let bytes = existing.reduce((sum, image) => sum + image.size, 0);
        for (const file of selected) {
            const image = await storeImage(file, file.name);added.push(image);bytes += image.size;
            if (bytes > MAX_IMAGE_BYTES) throw new Error('Images exceed the 32 MiB conversation limit');
        }
        draftImages.push(...added);
        renderImages(element('attachments'), draftImages, true);
    } catch (error) { attachmentError(error.message || 'Unable to attach image'); }
    finally { preparingImages = false;controls();await pruneImages(); }
}

function currentChat() { return chats.find(chat => chat.id === currentChatId); }
function chatTitle(prompt) {
    const title = prompt.replace(/\s+/g, ' ').trim();
    return title.length > 46 ? `${title.slice(0, 43)}...` : title;
}
function persistChats() {
    chats.sort((left, right) => right.updatedAt - left.updatedAt);
    chats = chats.slice(0, 50);
    try {
        localStorage.setItem(CHAT_STORAGE, JSON.stringify(chats));
        localStorage.setItem(ACTIVE_CHAT_STORAGE, currentChatId || NEW_CHAT);
    } catch {}
}
function ensureChat(prompt) {
    let chat = currentChat();
    if (chat) return chat;
    const now = Date.now();
    chat = {id: crypto.randomUUID?.() || `${now}-${Math.random()}`, title: chatTitle(prompt), messages: [],
        createdAt: now, updatedAt: now};
    chats.unshift(chat);
    currentChatId = chat.id;
    conversation = chat.messages;
    return chat;
}
function saveCurrentChat() {
    const chat = currentChat();
    if (!chat) return;
    chat.updatedAt = Date.now();
    persistChats();
    renderHistory();
    element('chat-title').textContent = chat.title;
}
function settleInterruptedTurn() {
    const chat = currentChat();
    if (!chat || !reply) return;
    const partial = reply.text.trim();
    if (partial) conversation.push({role: 'assistant', content: partial, stats: currentStats});
    else if (conversation.at(-1)?.role === 'user') {
        const rejected = conversation.pop();
        if (rejected.images?.length) {
            draftImages = [...rejected.images];
            if (!element('prompt').value) element('prompt').value = rejected.content;
            renderImages(element('attachments'), draftImages, true);
        }
    }
    reply = null;
    if (conversation.length) saveCurrentChat();
    else {
        chats = chats.filter(candidate => candidate.id !== chat.id);
        currentChatId = null;
        conversation = [];
        persistChats();
        renderHistory();
    }
}
function emptyState() {
    const empty = document.createElement('div');
    empty.id = 'empty';
    const logo = document.createElement('div');
    logo.className = 'empty-logo';
    const logoImage = document.createElement('img');
    logoImage.src = './icons/kidi-logo.png';
    logoImage.alt = '';
    logo.append(logoImage);
    const heading = document.createElement('h2');
    heading.textContent = 'Start a new conversation';
    const detail = document.createElement('p');
    detail.textContent = ready ? 'Gemma is ready on this device' : 'Local model offline';
    empty.append(logo, heading, detail);
    return empty;
}
function showMessageStats(footer, stats) {
    const valid = stats && ['tokenCount', 'elapsedMs', 'decodeMs', 'decodeTokens']
        .every(key => Number.isFinite(stats[key]) && stats[key] >= 0);
    footer.hidden = !valid;
    if (!valid) return;
    const speed = stats.decodeMs > 0 && stats.decodeTokens > 0
        ? `${(stats.decodeTokens * 1000 / stats.decodeMs).toFixed(1)} tok/s` : '-- tok/s';
    footer.textContent = `${stats.tokenCount} tokens | ${(stats.elapsedMs / 1000).toFixed(1)} s | ${speed}`;
    footer.title = `Decode speed excludes prompt preparation.${Number.isFinite(stats.firstTokenMs)
        ? ` First token: ${(stats.firstTokenMs / 1000).toFixed(2)} s.` : ''}`;
}
function addMessage(role, content, stats, images = []) {
    element('empty')?.remove();
    const article = document.createElement('article');
    article.className = `message ${role}`;
    const inner = document.createElement('div');
    inner.className = 'message-inner';
    const name = document.createElement('p');
    name.className = 'role';
    name.textContent = role === 'user' ? 'You' : 'Gemma';
    const text = document.createElement('div');
    text.className = 'content';
    renderMarkdown(text, content);
    inner.append(name, text);
    if (images.length) {
        const attachments = document.createElement('div');attachments.className = 'message-images';
        inner.insertBefore(attachments, text);renderImages(attachments, images);
    }
    text.hidden = role === 'user' && !content;
    const footer = document.createElement('p');
    footer.className = 'message-stats';
    showMessageStats(footer, stats);
    if (role === 'assistant') inner.append(footer);
    article.append(inner);
    element('messages').append(article);
    return {element: text, text: content, footer};
}
function renderMessages() {
    element('messages').replaceChildren();
    if (!conversation.length) element('messages').append(emptyState());
    else for (const message of conversation) addMessage(message.role, message.content, message.stats, message.images);
    element('chat-title').textContent = currentChat()?.title || 'New chat';
    element('messages').scrollTop = element('messages').scrollHeight;
}
function selectChat(id) {
    if (busy || loading || preparingImages) return;
    draftImages = [];renderImages(element('attachments'), draftImages, true);attachmentError();
    currentChatId = id;
    conversation = currentChat()?.messages || [];
    persistChats();
    renderHistory();
    renderMessages();
    closeHistory();
    pruneImages();
}
function deleteChat(id) {
    if (busy || loading || preparingImages) return;
    chats = chats.filter(chat => chat.id !== id);
    if (currentChatId === id) {
        currentChatId = chats[0]?.id || null;
        conversation = currentChat()?.messages || [];
        renderMessages();
    }
    persistChats();
    renderHistory();
    pruneImages();
}
function renderHistory() {
    const container = element('chat-history');
    container.replaceChildren();
    if (!chats.length) {
        const empty = document.createElement('p');
        empty.className = 'history-empty';
        empty.textContent = 'No conversations yet';
        container.append(empty);
        return;
    }
    for (const chat of chats) {
        const row = document.createElement('div');
        row.className = `history-item${chat.id === currentChatId ? ' active' : ''}`;
        const open = document.createElement('button');
        open.className = 'history-open';
        open.type = 'button';
        open.disabled = busy || loading;
        open.setAttribute('aria-current', chat.id === currentChatId ? 'page' : 'false');
        const icon = document.createElement('img');
        icon.src = './icons/message-square.svg';
        icon.alt = '';
        const title = document.createElement('span');
        title.textContent = chat.title;
        open.append(icon, title);
        open.addEventListener('click', () => selectChat(chat.id));
        const remove = document.createElement('button');
        remove.className = 'history-delete';
        remove.type = 'button';
        remove.title = 'Delete chat';
        remove.setAttribute('aria-label', `Delete ${chat.title}`);
        const trash = document.createElement('img');
        trash.src = './icons/trash-2.svg';
        trash.alt = '';
        remove.append(trash);
        remove.addEventListener('click', () => deleteChat(chat.id));
        row.append(open, remove);
        container.append(row);
    }
}
function setRuntimeState(state, label, detail, badge) {
    for (const dot of document.querySelectorAll('.status-dot')) dot.className = `status-dot${state ? ` ${state}` : ''}`;
    element('runtime-label').textContent = label;
    element('runtime-detail').textContent = detail;
    element('runtime-badge').textContent = badge;
}
let heapLimit = MAX_WASM_MEMORY;
function updateMemory(bytes, limit = heapLimit) {
    const known = Number.isSafeInteger(bytes) && bytes >= 0;
    const gibibytes = value => `${(value / 2 ** 30).toFixed(2)} GiB`;
    element('heap-used').textContent = known ? gibibytes(bytes) : '--';
    element('heap-headroom').textContent = known ? gibibytes(Math.max(0, limit - bytes)) : '--';
    element('memory').textContent = known ? gibibytes(bytes) : '--';
    element('memory-stats').classList.toggle('tight', known && limit - bytes < 256 * 1024 ** 2);
}
let memoryDiagnostics = null;
const binaryBytes = value => !Number.isFinite(value) || value < 0 ? '--'
    : value >= 2 ** 30 ? `${(value / 2 ** 30).toFixed(2)} GiB` : `${(value / 2 ** 20).toFixed(1)} MiB`;
function renderMemoryDiagnostics(memory) {
    memoryDiagnostics = memory || null;
    element('copy-memory').disabled = !memory;
    const summary = element('memory-summary'), categories = element('memory-categories');
    summary.replaceChildren();
    categories.replaceChildren();
    const row = (label, value) => {
        const term = document.createElement('dt'), detail = document.createElement('dd');
        term.textContent = label;
        detail.textContent = value;
        summary.append(term, detail);
    };
    if (!memory) { row('Status', 'Load a model to collect statistics'); return; }
    row('Wasm heap', `${binaryBytes(memory.heap_bytes)} of ${binaryBytes(memory.heap_limit_bytes)}`);
    row('Allocated', binaryBytes(memory.malloc_in_use_bytes));
    row('Allocator free', binaryBytes(memory.malloc_free_bytes));
    row('Allocator peak', binaryBytes(memory.malloc_peak_footprint_bytes));
    row('Weights in heap', binaryBytes(memory.weights_heap_bytes));
    row('Weights outside heap', binaryBytes(memory.weights_external_bytes));
    if (memory.weights_gpu_bytes) row('Weights in GPU buffers', binaryBytes(memory.weights_gpu_bytes));
    row('KV cache', binaryBytes(memory.kv_cache_bytes));
    row('Image features', binaryBytes(memory.image_feature_bytes));
    if (memory.gpu_buffer_bytes !== undefined)
        row('GPU buffers', `${binaryBytes(memory.gpu_buffer_bytes)} (${binaryBytes(memory.gpu_pooled_bytes)} pooled)`);
    for (const entry of memory.categories || []) {
        const item = document.createElement('li'), name = document.createElement('span'), value = document.createElement('span');
        name.textContent = entry.category;
        name.title = `${entry.category}: ${entry.count} allocation${entry.count === 1 ? '' : 's'}`;
        value.textContent = `${entry.bytes < 0 ? '-' : ''}${binaryBytes(Math.abs(entry.bytes))} / ${entry.count}`;
        item.append(name, value);
        categories.append(item);
    }
}
function controls() {
    const isRecording = Boolean(recording);
    element('send').disabled = !ready || busy || isRecording || preparingImages;
    element('attach').disabled = !ready || !visionSupported || busy || isRecording || preparingImages;
    element('attach').title = ready && !visionSupported
        ? 'This model does not support images'
        : 'Attach image';
    for (const button of document.querySelectorAll('.remove-image')) button.disabled = busy || preparingImages;
    element('stop').hidden = !busy;
    element('stop').disabled = stopping;
    element('mic').disabled = busy || loading || preparingImages;
    element('mic').classList.toggle('recording', isRecording);
    element('mic').setAttribute('aria-pressed', String(isRecording));
    element('mic').title = isRecording ? 'Stop recording' : 'Record speech';
    element('mic').setAttribute('aria-label', element('mic').title);
    element('prompt').disabled = isRecording;
    element('load').disabled = busy || loading || isRecording || preparingImages;
    element('clear-cache').disabled = busy || loading || isRecording;
    element('new-chat').disabled = busy || loading || isRecording || preparingImages;
    for (const id of ['backend', 'threads', 'manifest', 'speech-model', 'image-pixels'])
        element(id).disabled = busy || loading || isRecording || preparingImages;
    element('threads').disabled ||= element('backend').value === 'webgpu';
    for (const button of document.querySelectorAll('.history-open, .history-delete')) button.disabled = busy || loading || preparingImages;
    for (const button of document.querySelectorAll('.cache-delete')) button.disabled = busy || loading || isRecording;
}
function updateLiveStats() {
    const speed = encodingImages || (currentStats?.decodeTokens > 0 && currentStats.decodeMs > 0
        ? `${(currentStats.decodeTokens * 1000 / currentStats.decodeMs).toFixed(1)} tok/s`
        : currentStats?.tokenCount ? 'Decoding' : 'Preparing prompt');
    element('live-speed').textContent = speed;
    element('live-detail').textContent = `${currentStats?.tokenCount || 0} tokens | ${((performance.now() - generationStarted) / 1000).toFixed(1)} s`;
}
function stopLiveStats() {
    clearInterval(liveTimer);
    element('live-generation').hidden = true;
}
function finish() {
    busy = false;
    stopping = false;
    encodingImages = '';
    stopLiveStats();
    controls();
}
function releaseRecording(session = recording) {
    if (!session) return;
    clearInterval(recordingTimer);
    clearTimeout(recordingDeadline);
    session.node.port.onmessage = null;
    try { session.source.disconnect(); session.node.disconnect(); session.sink.disconnect(); } catch {}
    for (const track of session.stream.getTracks()) track.stop();
    session.context.close().catch(() => {});
    if (recording === session) recording = null;
}
function restart() {
    releaseRecording();
    runtimeVersion++;
    worker?.terminate();
    worker = null;
    ready = false;
    visionSupported = false;
    busy = false;
    loading = false;
    stopping = false;
    stopLiveStats();
    updateMemory();
    renderMemoryDiagnostics(null);
    element('gpu-memory').textContent = '--';
    element('status').textContent = 'Not loaded';
    setRuntimeState('', 'Model offline', 'Gemma 4 E2B IT', 'Offline');
    renderMessages();
    controls();
}
function newChat() {
    if (busy || loading || recording || preparingImages) return;
    draftImages = [];renderImages(element('attachments'), draftImages, true);attachmentError();
    currentChatId = null;
    conversation = [];
    persistChats();
    renderHistory();
    renderMessages();
    element('generation-stats').textContent = '9,216-token context';
    closeHistory();
    element('prompt').focus();
    pruneImages();
}
function openHistory() { document.body.classList.add('history-visible'); }
function closeHistory() { document.body.classList.remove('history-visible'); }
function openSettings() {
    if (!settingsDialog.open) settingsDialog.showModal();
    refreshCachedModels();
}
function closeSettings() { if (settingsDialog.open) settingsDialog.close(); }
function outputTokenLimit() {
    const input = element('tokens');
    const value = Number(input.value);
    const minimum = Number(input.min);
    const maximum = Number(input.max);
    const valid = Number.isInteger(value) && value >= minimum && value <= maximum;
    input.setCustomValidity(valid ? '' : `Output tokens must be a whole number from ${minimum} to ${maximum}`);
    return valid ? value : null;
}

async function refreshCachedModels() {
    const container = element('cache-models');
    container.replaceChildren();
    const loading = document.createElement('p');
    loading.className = 'cache-empty';
    loading.textContent = 'Reading cache';
    container.append(loading);
    try {
        const models = await listCachedModels();
        container.replaceChildren();
        element('cache-total').textContent = megabytes(models.reduce((sum, model) => sum + model.bytes, 0));
        if (!models.length) {
            const empty = document.createElement('p');
            empty.className = 'cache-empty';
            empty.textContent = 'No cached models';
            container.append(empty);
            return;
        }
        for (const model of models.sort((left, right) => right.bytes - left.bytes)) {
            const article = document.createElement('article');
            article.className = 'cache-model';
            const header = document.createElement('div');
            header.className = 'cache-model-heading';
            const identity = document.createElement('div');
            const name = document.createElement('strong');
            name.textContent = model.name;
            const meta = document.createElement('small');
            const revision = /^[a-f0-9]{40}$/.test(model.revision) ? model.revision.slice(0, 8) : 'package';
            meta.textContent = `${model.kind} · ${megabytes(model.bytes)} · ${revision} · ${model.complete ? 'Ready' : 'Partial'}`;
            identity.append(name, meta);
            const remove = document.createElement('button');
            remove.type = 'button';
            remove.className = 'cache-delete';
            remove.title = `Delete ${model.name} from cache`;
            remove.setAttribute('aria-label', remove.title);
            const icon = document.createElement('img');
            icon.src = './icons/trash-2.svg';
            icon.alt = '';
            remove.append(icon, document.createTextNode('Delete'));
            remove.addEventListener('click', async () => {
                remove.disabled = true;
                try {
                    await deleteCachedModel(model.id);
                    element('warning').textContent = `Deleted cached files for ${model.name}`;
                    await refreshCachedModels();
                } catch (error) {
                    remove.disabled = false;
                    element('warning').textContent = error.message;
                }
            });
            header.append(identity, remove);
            const details = document.createElement('details');
            const summary = document.createElement('summary');
            summary.textContent = `${model.files.length} files`;
            const files = document.createElement('ul');
            for (const file of model.files) {
                const item = document.createElement('li');
                const fileName = document.createElement('span');
                fileName.textContent = file.name;
                const size = document.createElement('span');
                size.textContent = cacheSize(file.bytes);
                item.append(fileName, size);
                files.append(item);
            }
            details.append(summary, files);
            article.append(header, details);
            container.append(article);
        }
        controls();
    } catch (error) {
        container.replaceChildren();
        const failed = document.createElement('p');
        failed.className = 'cache-empty error';
        failed.textContent = error.message;
        container.append(failed);
        element('cache-total').textContent = '--';
    }
}

async function startRecording() {
    if (recording || busy || loading) return;
    if (!navigator.mediaDevices?.getUserMedia || !globalThis.AudioWorkletNode) {
        element('warning').textContent = 'This browser does not support microphone capture';
        return;
    }
    let session;
    try {
        const stream = await navigator.mediaDevices.getUserMedia({audio: {channelCount: 1, echoCancellation: true,
            noiseSuppression: true, autoGainControl: true}});
        const context = new AudioContext({sampleRate: 16000, latencyHint: 'interactive'});
        await context.audioWorklet.addModule(new URL('./audio-capture-worklet.mjs', location.href));
        const source = context.createMediaStreamSource(stream);
        const node = new AudioWorkletNode(context, 'kidi-audio-capture');
        const sink = context.createGain();
        sink.gain.value = 0;
        const modelId = element('speech-model').value.trim();
        if (!MODEL_ID.test(modelId)) throw new Error('Enter a Hugging Face speech model ID such as openai/whisper-small');
        session = {stream, context, source, node, sink, chunks: [], started: performance.now(), modelId,
            ready: false, transcribing: false, lastDraftRequest: 0, draftText: '', finalAudio: null};
        node.port.onmessage = ({data}) => {
            if (recording === session && data instanceof ArrayBuffer) session.chunks.push(new Float32Array(data));
        };
        source.connect(node).connect(sink).connect(context.destination);
        await context.resume();
        recording = session;
        startSpeechWorker(session);
        element('warning').textContent = '';
        const update = () => {
            const seconds = Math.min(30, (performance.now() - session.started) / 1000);
            if (session.ready && seconds >= 0.8 && performance.now() - session.lastDraftRequest >= 1200)
                requestSpeech(session, false);
            element('generation-stats').textContent = session.status || `Recording ${seconds.toFixed(1)} s`;
        };
        update();
        recordingTimer = setInterval(update, 200);
        recordingDeadline = setTimeout(() => finishRecording(), 30000);
        controls();
    } catch (error) {
        releaseRecording(session);
        element('warning').textContent = error.name === 'NotAllowedError' ? 'Microphone permission was denied' :
            `Unable to record speech: ${error.message}`;
    }
}

function updateDraft(text) {
    element('prompt').value = text.trim();
    element('prompt').dispatchEvent(new Event('input'));
}

function failSpeech(session, error) {
    disposeSpeechWorker();
    session.worker = null;
    if (recording === session) releaseRecording(session);
    element('warning').textContent = error;
    finish();
}

function requestSpeech(session, final) {
    if (!session.ready || session.transcribing) return;
    const audio = final ? session.finalAudio : resampleAudio(session.chunks, session.context.sampleRate);
    if (!audio || audio.length < 1600) return;
    session.transcribing = true;
    session.lastDraftRequest = performance.now();
    session.requestId = final ? 'final' : 'draft';
    session.status = final ? 'Refining transcript' : 'Updating draft transcript';
    session.worker.postMessage({type: 'transcribe', requestId: session.requestId, audio: audio.buffer,
        language: 'auto', maximumTokens: 128}, [audio.buffer]);
}

function disposeSpeechWorker() {
    speechWorker?.terminate();
    speechWorker = null;
    speechReady = false;
    if (speechSession) speechSession.worker = null;
    speechSession = null;
}

function preloadSpeech() {
    const modelId = element('speech-model').value.trim();
    const threads = crossOriginIsolated ? Number(element('threads').value) : 1;
    if (!MODEL_ID.test(modelId) || !Number.isInteger(threads) || threads < 1 || threads > 8) return null;
    if (speechWorker && speechModel === modelId && speechThreads === threads) return speechWorker;
    disposeSpeechWorker();
    speechModel = modelId;
    speechThreads = threads;
    const status = element('speech-status');
    status.textContent = 'Loading speech model';
    status.classList.remove('error');
    let worker;
    const failed = error => {
        const session = speechSession;
        disposeSpeechWorker();
        status.textContent = `Speech unavailable: ${error}`;
        status.classList.add('error');
        if (session) failSpeech(session, error);
    };
    try { worker = new Worker(new URL('./asr-worker.mjs', import.meta.url), {type: 'module'}); }
    catch (error) { failed(error.message);return null; }
    speechWorker = worker;
    worker.onerror = event => { if (speechWorker === worker) failed(event.message); };
    worker.onmessage = ({data}) => {
        if (speechWorker !== worker) return;
        document.dispatchEvent(new CustomEvent('kidi:speech', {detail: data}));
        const session = speechSession;
        if (data.type === 'progress') {
            const percent = data.totalBytes ? (data.loadedBytes / data.totalBytes * 100).toFixed(0) : '0';
            status.textContent = `Loading speech model ${percent}%`;
            if (session) session.status = status.textContent;
        } else if (data.type === 'cached') {
            // The worker finished replacing downloaded speech weights with its converted copy.
            refreshCachedModels();
        } else if (data.type === 'ready') {
            speechReady = true;
            status.textContent = 'Speech ready';
            if (session) {
                session.ready = true;
                session.status = recording === session ? 'Listening' : 'Preparing final transcript';
                requestSpeech(session, Boolean(session.finalAudio));
            }
        } else if (data.type === 'result' && session && session.worker === worker) {
            session.transcribing = false;
            const transcript = data.text.trim();
            if (data.requestId === 'draft' && !session.finalAudio) {
                session.draftText = transcript;
                updateDraft(transcript);
                session.status = `Draft transcript / ${data.language}`;
            } else if (session.finalAudio && data.requestId !== 'final') {
                requestSpeech(session, true);
            } else {
                updateDraft(transcript);
                element('generation-stats').textContent = transcript ?
                    `Transcript ready / ${data.language} / ${(data.encode_ms + data.decode_ms).toFixed(0)} ms` :
                    'No speech recognized';
                if (!transcript) element('warning').textContent = 'No speech was recognized';
                session.worker = null;
                speechSession = null;
                finish();
                element('prompt').focus();
            }
        } else if (data.type === 'error') {
            failed(data.error);
        }
    };
    worker.postMessage({type: 'load', source: modelId, threads});
    return worker;
}

function startSpeechWorker(session) {
    session.worker = preloadSpeech();
    if (!session.worker) throw new Error('Speech runtime is unavailable');
    speechSession = session;
    session.ready = speechReady;
    session.status = speechReady ? 'Listening' : element('speech-status').textContent;
}

function finishRecording() {
    const session = recording;
    if (!session) return;
    recording = null;
    const audio = resampleAudio(session.chunks, session.context.sampleRate);
    releaseRecording(session);
    if (audio.length < 1600) {
        session.worker = null;
        speechSession = null;
        if (session.transcribing) { disposeSpeechWorker();preloadSpeech(); }
        element('warning').textContent = 'Speech recording was too short';
        element('generation-stats').textContent = '9,216-token context';
        controls();
        return;
    }
    session.finalAudio = audio;
    session.status = 'Preparing final transcript';
    busy = true;
    controls();
    element('warning').textContent = '';
    element('generation-stats').textContent = session.status;
    if (session.ready && !session.transcribing) requestSpeech(session, true);
}

if (!crossOriginIsolated) element('threads').value = '1';
if (!crossOriginIsolated) element('threads').max = '1';
for (const id of ['threads', 'manifest']) element(id).addEventListener('change', restart);
element('threads').addEventListener('change', preloadSpeech);
element('speech-model').addEventListener('change', () => {
    const value = element('speech-model').value.trim();
    if (!MODEL_ID.test(value)) return;
    preferences.speechModel = value;
    try { localStorage.setItem(SETTINGS_STORAGE, JSON.stringify(preferences)); } catch {}
    preloadSpeech();
});
element('backend').addEventListener('change', () => {
    preferences.backend = element('backend').value;
    try { localStorage.setItem(SETTINGS_STORAGE, JSON.stringify(preferences)); } catch {}
    restart();
});
for (const id of ['threads', 'tokens', 'image-pixels']) element(id).addEventListener('input', () => savePreference(id));
for (const id of ['open-settings', 'header-settings']) element(id).addEventListener('click', openSettings);
element('close-settings').addEventListener('click', closeSettings);
element('open-history').addEventListener('click', openHistory);
for (const id of ['close-history', 'history-backdrop']) element(id).addEventListener('click', closeHistory);
element('new-chat').addEventListener('click', newChat);
settingsDialog.addEventListener('click', event => { if (event.target === settingsDialog) closeSettings(); });
element('tokens').addEventListener('input', outputTokenLimit);

async function loadRuntime(cacheOnly = false) {
    await backendSelection;
    if (loading || busy) return;
    const backend = element('backend').value;
    const threads = backend === 'webgpu' ? 1 : Number(element('threads').value);
    if (!Number.isInteger(threads) || threads < 1 || threads > Number(element('threads').max)) return;
    const manifest = element('manifest').value.trim();
    if (!MODEL_ID.test(manifest)) {
        element('warning').textContent = 'Enter a Hugging Face model ID such as google/gemma-4-E2B-it-qat-mobile-transformers';
        openSettings();
        return;
    }
    restart();
    const version = runtimeVersion;
    loading = true;
    controls();
    element('warning').textContent = '';
    element('status').textContent = 'Starting runtime';
    element('progress').hidden = false;
    setRuntimeState('loading', 'Starting runtime', 'Gemma 4 E2B IT', 'Loading');
    if (!cacheOnly) { try { await navigator.storage?.persist(); } catch {} }
    if (version !== runtimeVersion) return;
    const failed = event => {
        element('warning').textContent = event.message;
        element('status').textContent = 'Runtime failed';
        loading = false;
        ready = false;
        element('progress').hidden = true;
        setRuntimeState('error', 'Runtime failed', event.message, 'Error');
        openSettings();
        finish();
    };
    try { worker = new Worker(new URL('./inference-worker.mjs', import.meta.url), {type: 'module'}); }
    catch (error) { failed(error); return; }
    worker.onerror = failed;
    worker.onmessage = ({data}) => {
        document.dispatchEvent(new CustomEvent('kidi:runtime', {detail: data}));
        if (data.memory) {
            heapLimit = data.memory.heap_limit_bytes || heapLimit;
            renderMemoryDiagnostics(data.memory);
            updateMemory(data.memory.heap_bytes);
        } else if (data.heapBytes !== undefined) updateMemory(data.heapBytes);
        if (data.gpuStats) element('gpu-memory').textContent = megabytes(data.gpuStats.allocatedBytes);
        if (busy && data.stats) { currentStats = data.stats; updateLiveStats(); }
        if (data.type === 'progress') {
            const percent = (data.loadedBytes / data.totalBytes * 100).toFixed(0);
            element('progress').value = data.loadedBytes / data.totalBytes;
            element('download').textContent = megabytes(data.downloadedBytes);
            element('cached').textContent = megabytes(data.cachedBytes);
            element('status').textContent = `Loading ${percent}%`;
            element('warning').textContent = data.warning;
            setRuntimeState('loading', `Loading model ${percent}%`, data.file, `${percent}%`);
        } else if (data.type === 'initializing') {
            element('status').textContent = 'Initializing Gemma';
            setRuntimeState('loading', 'Initializing Gemma', 'Preparing operators', 'Loading');
        } else if (data.type === 'ready') {
            try { localStorage.setItem(MODEL_STORAGE, manifest); } catch {}
            ready = true;
            visionSupported = data.vision_supported === true;
            loading = false;
            const backend = data.backend === 'webgpu' ? 'WebGPU' : 'Wasm CPU';
            const detail = data.backend === 'webgpu' ? backend : `${backend} / ${data.threads} thread${data.threads === 1 ? '' : 's'}`;
            element('status').textContent = `${detail} / Ready`;
            element('load-time').textContent = `${(data.loadMs / 1000).toFixed(1)} s`;
            element('progress').hidden = true;
            setRuntimeState('ready', 'Model ready', detail, data.backend === 'webgpu' ? 'GPU' : `CPU ${data.threads}T`);
            renderMessages();
            closeSettings();
            refreshCachedModels();
            controls();
        } else if (data.type === 'encoding-images') {
            encodingImages = data.stage || 'Encoding images';
            element('live-speed').textContent = encodingImages;
            element('generation-stats').textContent = encodingImages;
        } else if (data.type === 'step') {
            encodingImages = '';
            const messages = element('messages');
            const follow = messages.scrollHeight - messages.scrollTop - messages.clientHeight < 100;
            for (const generationEvent of data.events) {
                if (generationEvent.text) reply.text += generationEvent.text;
                if (generationEvent.completed) {
                    renderMarkdown(reply.element, generationEvent.completed.text);
                    const result = generationEvent.completed;
                    const stats = {...currentStats, tokenCount: result.token_ids.length, elapsedMs: result.generation_ms,
                        decodeTokens: result.decode_tokens, decodeMs: result.decode_ms};
                    showMessageStats(reply.footer, stats);
                    conversation.push({role: 'assistant', content: result.text, stats});
                    reply = null;
                    saveCurrentChat();
                    element('generation-stats').textContent = `${result.token_ids.length} tokens / ${(result.generation_ms / 1000).toFixed(1)} s${result.decode_tokens ? ` / ${(result.decode_tokens * 1000 / result.decode_ms).toFixed(1)} tok/s decode` : ''}`;
                    finish();
                }
            }
            if (reply) renderMarkdown(reply.element, reply.text, {streaming: true});
            if (follow) messages.scrollTop = messages.scrollHeight;
        } else if (data.type === 'cancelled') {
            settleInterruptedTurn();
            renderMessages();
            element('generation-stats').textContent = 'Cancelled';
            finish();
        } else if (data.type === 'error') {
            if (busy) settleInterruptedTurn();
            element('warning').textContent = data.error;
            attachmentError(data.error);
            if (data.fatal) {
                ready = false;
                element('status').textContent = 'Runtime failed';
                setRuntimeState('error', 'Runtime failed', data.error, 'Error');
                openSettings();
            }
            loading = false;
            element('progress').hidden = true;
            renderMessages();
            finish();
        }
    };
    worker.postMessage({type: 'load', manifest, threads, backend, cacheOnly});
}
element('load').addEventListener('click', () => loadRuntime());
element('copy-memory').addEventListener('click', async () => {
    if (!memoryDiagnostics) return;
    try {
        await navigator.clipboard.writeText(JSON.stringify(memoryDiagnostics, null, 2));
        element('copy-memory').textContent = 'Copied';
    } catch { element('copy-memory').textContent = 'Copy failed'; }
    setTimeout(() => { element('copy-memory').textContent = 'Copy diagnostics'; }, 1500);
});

async function restoreCachedModel() {
    const version = runtimeVersion;
    const source = element('manifest').value;
    try {
        if (await isModelCached(source) && version === runtimeVersion &&
            source === element('manifest').value && !worker && !loading && !busy)
            await loadRuntime(true);
    } catch {}
}

const gpuUnavailable = webGpuSupport();
if (gpuUnavailable) {
    const option = element('backend').querySelector('option[value="webgpu"]');
    option.disabled = true;
    option.textContent = 'WebGPU (unavailable)';
    option.title = `WebGPU is unavailable: ${gpuUnavailable}`;
    if (element('backend').value === 'webgpu') element('backend').value = 'cpu';
}
async function selectDefaultBackend() {
    if (preferences.backend || gpuUnavailable) return;
    try {
        const adapter = await navigator.gpu.requestAdapter({powerPreference: 'high-performance'});
        if (adapter && !adapter.info.isFallbackAdapter && !preferences.backend) element('backend').value = 'webgpu';
    } catch {}
    controls();
}

element('compose').addEventListener('submit', event => {
    event.preventDefault();
    const prompt = element('prompt').value.trim();
    if ((!prompt && !draftImages.length) || !ready || busy || preparingImages) return;
    if (!visionSupported && (draftImages.length || conversation.some(message => message.images?.length))) {
        attachmentError('This model does not support images');return;
    }
    const maximumTokens = outputTokenLimit();
    if (maximumTokens === null) {
        const message = element('tokens').validationMessage;
        element('warning').textContent = message;
        element('generation-stats').textContent = message;
        openSettings();
        element('tokens').reportValidity();
        return;
    }
    const imageLimit = element('image-pixels');
    if (!imageLimit.checkValidity() || !Number.isInteger(imageLimit.valueAsNumber)) {
        element('warning').textContent = imageLimit.validationMessage || 'Enter an image resize limit';
        openSettings();imageLimit.reportValidity();return;
    }
    const imageMaxPixels = imageLimit.valueAsNumber;
    const images = draftImages;
    const chat = ensureChat(prompt || images[0].name);
    conversation.push({role: 'user', content: prompt, ...(images.length ? {images} : {})});
    draftImages = [];renderImages(element('attachments'), draftImages, true);attachmentError();
    chat.updatedAt = Date.now();
    saveCurrentChat();
    busy = true;
    currentStats = null;
    generationStarted = performance.now();
    element('live-generation').hidden = false;
    updateLiveStats();
    liveTimer = setInterval(updateLiveStats, 250);
    controls();
    element('warning').textContent = '';
    addMessage('user', prompt, undefined, images);
    reply = addMessage('assistant', '');
    element('messages').scrollTop = element('messages').scrollHeight;
    element('generation-stats').textContent = 'Generating';
    element('prompt').value = '';
    element('prompt').style.height = '';
    worker.postMessage({type: 'generate', messages: conversation, maximumTokens, imageMaxPixels});
});
element('prompt').addEventListener('input', event => {
    event.target.style.height = 'auto';
    event.target.style.height = `${Math.min(event.target.scrollHeight, 170)}px`;
});
element('attach').addEventListener('click', () => element('image-files').click());
element('image-files').addEventListener('change', event => {
    attachImages(event.target.files);event.target.value = '';
});
element('compose').addEventListener('paste', event => {
    const clipboard = event.clipboardData;
    let files = [...(clipboard?.items || [])]
        .filter(item => item.kind === 'file' && item.type.startsWith('image/'))
        .map(item => item.getAsFile()).filter(Boolean);
    if (!files.length) files = [...(clipboard?.files || [])].filter(file => file.type.startsWith('image/'));
    if (files.length) { event.preventDefault();attachImages(files); }
});
element('compose').addEventListener('dragover', event => {
    if (event.dataTransfer.types.includes('Files')) event.preventDefault();
});
element('compose').addEventListener('drop', event => {
    if (event.dataTransfer.files.length) { event.preventDefault();attachImages(event.dataTransfer.files); }
});
element('close-image').addEventListener('click', () => element('image-dialog').close());
element('image-dialog').addEventListener('click', event => { if (event.target === element('image-dialog')) event.target.close(); });
element('image-dialog').addEventListener('close', () => element('image-preview').removeAttribute('src'));
element('prompt').addEventListener('keydown', event => {
    if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
        event.preventDefault();
        element('compose').requestSubmit();
    }
});
element('mic').addEventListener('click', () => recording ? finishRecording() : startRecording());
element('stop').addEventListener('click', () => {
    if (!busy || stopping) return;
    if (speechSession) {
        disposeSpeechWorker();
        element('generation-stats').textContent = 'Transcription cancelled';
        finish();
        preloadSpeech();
        return;
    }
    if (!worker) return;
    stopping = true;
    element('generation-stats').textContent = 'Stopping after current model step';
    controls();
    worker.postMessage({type: 'cancel'});
});
element('clear-cache').addEventListener('click', async () => {
    runtimeVersion++;
    try {
        await clearModelCache();
        element('cached').textContent = '0 MB';
        element('warning').textContent = 'Model cache cleared';
        await refreshCachedModels();
    } catch (error) { element('warning').textContent = error.message; }
});
addEventListener('pagehide', () => {
    runtimeVersion++;
    releaseRecording();
    disposeSpeechWorker();
    stopLiveStats();
    if (busy) {
        worker?.postMessage({type: 'cancel'});
        settleInterruptedTurn();
    }
    worker?.terminate();
    worker = null;
});
addEventListener('pageshow', event => { if (event.persisted) { restart(); restoreCachedModel(); preloadSpeech(); } });

renderHistory();
renderMessages();
controls();
refreshCachedModels();
const backendSelection = selectDefaultBackend();
restoreCachedModel();
preloadSpeech();