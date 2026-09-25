import {clearModelCache, isModelCached} from './model-cache.mjs';
import {renderMarkdown} from './markdown.mjs';

const CHAT_STORAGE = 'kidi-chats-v1';
const ACTIVE_CHAT_STORAGE = 'kidi-active-chat-v1';
const MODEL_STORAGE = 'kidi-model-source-v1';
const SETTINGS_STORAGE = 'kidi-settings-v1';
const NEW_CHAT = '__new__';
const element = id => document.getElementById(id);
const megabytes = bytes => `${(bytes / 1e6).toFixed(1)} MB`;
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
const preferences = {};

try {
    const saved = JSON.parse(localStorage.getItem(SETTINGS_STORAGE) || '{}');
    if (['cpu', 'webgpu'].includes(saved?.backend)) {
        preferences.backend = saved.backend;
        element('backend').value = saved.backend;
    }
    for (const id of ['threads', 'tokens']) {
        const input = element(id);
        const value = saved?.[id];
        if (Number.isInteger(value) && value >= Number(input.min) && value <= Number(input.max)) {
            preferences[id] = value;
            input.value = String(value);
        }
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
    if (source) element('manifest').value = source;
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
    else if (conversation.at(-1)?.role === 'user') conversation.pop();
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
function addMessage(role, content, stats) {
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
    else for (const message of conversation) addMessage(message.role, message.content, message.stats);
    element('chat-title').textContent = currentChat()?.title || 'New chat';
    element('messages').scrollTop = element('messages').scrollHeight;
}
function selectChat(id) {
    if (busy || loading) return;
    currentChatId = id;
    conversation = currentChat()?.messages || [];
    persistChats();
    renderHistory();
    renderMessages();
    closeHistory();
}
function deleteChat(id) {
    if (busy || loading) return;
    chats = chats.filter(chat => chat.id !== id);
    if (currentChatId === id) {
        currentChatId = chats[0]?.id || null;
        conversation = currentChat()?.messages || [];
        renderMessages();
    }
    persistChats();
    renderHistory();
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
function updateMemory(bytes) {
    const known = Number.isSafeInteger(bytes) && bytes >= 0;
    const gibibytes = value => `${(value / 2 ** 30).toFixed(2)} GiB`;
    element('heap-used').textContent = known ? gibibytes(bytes) : '--';
    element('heap-headroom').textContent = known ? gibibytes(Math.max(0, 2 ** 32 - bytes)) : '--';
    element('memory').textContent = known ? gibibytes(bytes) : '--';
    element('memory-stats').classList.toggle('tight', known && bytes > 3.75 * 2 ** 30);
}
function controls() {
    element('send').disabled = !ready || busy;
    element('stop').hidden = !busy;
    element('stop').disabled = stopping;
    element('load').disabled = busy || loading;
    element('clear-cache').disabled = busy || loading;
    element('new-chat').disabled = busy || loading;
    for (const id of ['backend', 'threads', 'manifest']) element(id).disabled = busy || loading;
    element('threads').disabled ||= element('backend').value === 'webgpu';
    for (const button of document.querySelectorAll('.history-open, .history-delete')) button.disabled = busy || loading;
}
function updateLiveStats() {
    const speed = currentStats?.decodeTokens > 0 && currentStats.decodeMs > 0
        ? `${(currentStats.decodeTokens * 1000 / currentStats.decodeMs).toFixed(1)} tok/s`
        : currentStats?.tokenCount ? 'Decoding' : 'Preparing prompt';
    element('live-speed').textContent = speed;
    element('live-detail').textContent = `${currentStats?.tokenCount || 0} tokens | ${((performance.now() - generationStarted) / 1000).toFixed(1)} s`;
}
function stopLiveStats() {
    clearInterval(liveTimer);
    element('live-generation').hidden = true;
}
function finish() { busy = false; stopping = false; stopLiveStats(); controls(); }
function restart() {
    runtimeVersion++;
    worker?.terminate();
    worker = null;
    ready = false;
    busy = false;
    loading = false;
    stopping = false;
    stopLiveStats();
    updateMemory();
    element('gpu-memory').textContent = '--';
    element('status').textContent = 'Not loaded';
    setRuntimeState('', 'Model offline', 'Gemma 4 E2B IT', 'Offline');
    renderMessages();
    controls();
}
function newChat() {
    if (busy || loading) return;
    currentChatId = null;
    conversation = [];
    persistChats();
    renderHistory();
    renderMessages();
    element('generation-stats').textContent = '9,216-token context';
    closeHistory();
    element('prompt').focus();
}
function openHistory() { document.body.classList.add('history-visible'); }
function closeHistory() { document.body.classList.remove('history-visible'); }
function openSettings() { if (!settingsDialog.open) settingsDialog.showModal(); }
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

if (!crossOriginIsolated) element('threads').value = '1';
if (!crossOriginIsolated) element('threads').max = '1';
for (const id of ['threads', 'manifest']) element(id).addEventListener('change', restart);
element('backend').addEventListener('change', () => {
    preferences.backend = element('backend').value;
    try { localStorage.setItem(SETTINGS_STORAGE, JSON.stringify(preferences)); } catch {}
    restart();
});
for (const id of ['threads', 'tokens']) element(id).addEventListener('input', () => savePreference(id));
for (const id of ['open-settings', 'header-settings']) element(id).addEventListener('click', openSettings);
element('close-settings').addEventListener('click', closeSettings);
element('open-history').addEventListener('click', openHistory);
for (const id of ['close-history', 'history-backdrop']) element(id).addEventListener('click', closeHistory);
element('new-chat').addEventListener('click', newChat);
settingsDialog.addEventListener('click', event => { if (event.target === settingsDialog) closeSettings(); });
element('tokens').addEventListener('input', outputTokenLimit);

async function loadRuntime(cacheOnly = false) {
    if (loading || busy) return;
    const backend = element('backend').value;
    const threads = backend === 'webgpu' ? 1 : Number(element('threads').value);
    if (!Number.isInteger(threads) || threads < 1 || threads > Number(element('threads').max)) return;
    let manifest;
    try { manifest = new URL(element('manifest').value, location.href).href; }
    catch { element('warning').textContent = 'Enter a valid model source URL'; openSettings(); return; }
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
        if (data.heapBytes !== undefined) updateMemory(data.heapBytes);
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
            loading = false;
            const backend = data.backend === 'webgpu' ? 'WebGPU' : 'Wasm CPU';
            const detail = data.backend === 'webgpu' ? backend : `${backend} / ${data.threads} thread${data.threads === 1 ? '' : 's'}`;
            element('status').textContent = `${detail} / Ready`;
            element('load-time').textContent = `${(data.loadMs / 1000).toFixed(1)} s`;
            element('progress').hidden = true;
            setRuntimeState('ready', 'Model ready', detail, data.backend === 'webgpu' ? 'GPU' : `CPU ${data.threads}T`);
            renderMessages();
            closeSettings();
            controls();
        } else if (data.type === 'step') {
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

async function restoreCachedModel() {
    const version = runtimeVersion;
    const source = element('manifest').value;
    try {
        if (await isModelCached(new URL(source, location.href).href) && version === runtimeVersion &&
            source === element('manifest').value && !worker && !loading && !busy)
            await loadRuntime(true);
    } catch {}
}

element('compose').addEventListener('submit', event => {
    event.preventDefault();
    const prompt = element('prompt').value.trim();
    if (!prompt || !ready || busy) return;
    const maximumTokens = outputTokenLimit();
    if (maximumTokens === null) {
        const message = element('tokens').validationMessage;
        element('warning').textContent = message;
        element('generation-stats').textContent = message;
        openSettings();
        element('tokens').reportValidity();
        return;
    }
    const chat = ensureChat(prompt);
    conversation.push({role: 'user', content: prompt});
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
    addMessage('user', prompt);
    reply = addMessage('assistant', '');
    element('messages').scrollTop = element('messages').scrollHeight;
    element('generation-stats').textContent = 'Generating';
    element('prompt').value = '';
    element('prompt').style.height = '';
    worker.postMessage({type: 'generate', messages: conversation, maximumTokens});
});
element('prompt').addEventListener('input', event => {
    event.target.style.height = 'auto';
    event.target.style.height = `${Math.min(event.target.scrollHeight, 170)}px`;
});
element('prompt').addEventListener('keydown', event => {
    if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
        event.preventDefault();
        element('compose').requestSubmit();
    }
});
element('stop').addEventListener('click', () => {
    if (!worker || !busy || stopping) return;
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
    } catch (error) { element('warning').textContent = error.message; }
});
addEventListener('pagehide', () => {
    runtimeVersion++;
    stopLiveStats();
    if (busy) {
        worker?.postMessage({type: 'cancel'});
        settleInterruptedTurn();
    }
    worker?.terminate();
    worker = null;
});
addEventListener('pageshow', event => { if (event.persisted) { restart(); restoreCachedModel(); } });

renderHistory();
renderMessages();
controls();
restoreCachedModel();