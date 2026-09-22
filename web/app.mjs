import {clearModelCache} from './model-cache.mjs';

const CHAT_STORAGE = 'kidi-chats-v1';
const ACTIVE_CHAT_STORAGE = 'kidi-active-chat-v1';
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
const storedActiveChat = localStorage.getItem(ACTIVE_CHAT_STORAGE);
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
    const partial = reply.textContent.trim();
    if (partial) conversation.push({role: 'assistant', content: partial});
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
function addMessage(role, content) {
    element('empty')?.remove();
    const article = document.createElement('article');
    article.className = `message ${role}`;
    const inner = document.createElement('div');
    inner.className = 'message-inner';
    const avatar = document.createElement('span');
    avatar.className = 'message-avatar';
    avatar.textContent = role === 'user' ? 'Y' : 'K';
    const body = document.createElement('div');
    const name = document.createElement('p');
    name.className = 'role';
    name.textContent = role === 'user' ? 'You' : 'Gemma';
    const text = document.createElement('p');
    text.className = 'content';
    text.textContent = content;
    body.append(name, text);
    inner.append(avatar, body);
    article.append(inner);
    element('messages').append(article);
    return text;
}
function renderMessages() {
    element('messages').replaceChildren();
    if (!conversation.length) element('messages').append(emptyState());
    else for (const message of conversation) addMessage(message.role, message.content);
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
function controls() {
    element('send').disabled = !ready || busy;
    element('stop').hidden = !busy;
    element('stop').disabled = stopping;
    element('load').disabled = busy || loading;
    element('clear-cache').disabled = busy || loading;
    element('new-chat').disabled = busy || loading;
    for (const id of ['threads', 'manifest']) element(id).disabled = busy || loading;
    for (const button of document.querySelectorAll('.history-open, .history-delete')) button.disabled = busy || loading;
}
function finish() { busy = false; stopping = false; controls(); }
function restart() {
    worker?.terminate();
    worker = null;
    ready = false;
    busy = false;
    loading = false;
    stopping = false;
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
for (const id of ['open-settings', 'header-settings']) element(id).addEventListener('click', openSettings);
element('close-settings').addEventListener('click', closeSettings);
element('open-history').addEventListener('click', openHistory);
for (const id of ['close-history', 'history-backdrop']) element(id).addEventListener('click', closeHistory);
element('new-chat').addEventListener('click', newChat);
settingsDialog.addEventListener('click', event => { if (event.target === settingsDialog) closeSettings(); });
element('tokens').addEventListener('input', outputTokenLimit);

element('load').addEventListener('click', async () => {
    const threads = Number(element('threads').value);
    if (!Number.isInteger(threads) || threads < 1 || threads > Number(element('threads').max)) return;
    restart();
    loading = true;
    controls();
    element('warning').textContent = '';
    element('status').textContent = 'Starting runtime';
    element('progress').hidden = false;
    setRuntimeState('loading', 'Starting runtime', 'Gemma 4 E2B IT', 'Loading');
    try { await navigator.storage?.persist(); } catch {}
    worker = new Worker(new URL('./inference-worker.mjs', import.meta.url), {type: 'module'});
    worker.onerror = event => {
        element('warning').textContent = event.message;
        element('status').textContent = 'Runtime failed';
        loading = false;
        ready = false;
        setRuntimeState('error', 'Runtime failed', event.message, 'Error');
        openSettings();
        finish();
    };
    worker.onmessage = ({data}) => {
        document.dispatchEvent(new CustomEvent('kidi:runtime', {detail: data}));
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
            ready = true;
            loading = false;
            const backend = 'Wasm CPU';
            element('status').textContent = `${backend} / ${data.threads} thread${data.threads === 1 ? '' : 's'} / Ready`;
            element('load-time').textContent = `${(data.loadMs / 1000).toFixed(1)} s`;
            element('memory').textContent = `${(data.heapBytes / 2 ** 30).toFixed(2)} GiB`;
            element('progress').hidden = true;
            setRuntimeState('ready', 'Model ready', `${backend} / ${data.threads} thread${data.threads === 1 ? '' : 's'}`,
                `CPU ${data.threads}T`);
            renderMessages();
            closeSettings();
            controls();
        } else if (data.type === 'step') {
            for (const generationEvent of data.events) {
                if (generationEvent.text) reply.textContent += generationEvent.text;
                if (generationEvent.completed) {
                    reply.textContent = generationEvent.completed.text;
                    conversation.push({role: 'assistant', content: generationEvent.completed.text});
                    reply = null;
                    saveCurrentChat();
                    const result = generationEvent.completed;
                    element('generation-stats').textContent = `${result.token_ids.length} tokens / ${(result.generation_ms / 1000).toFixed(1)} s${result.decode_tokens ? ` / ${(result.decode_tokens * 1000 / result.decode_ms).toFixed(1)} tok/s decode` : ''}`;
                    finish();
                }
            }
            element('messages').scrollTop = element('messages').scrollHeight;
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
    worker.postMessage({type: 'load', manifest: new URL(element('manifest').value, location.href).href,
        threads});
});

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
    controls();
    element('warning').textContent = '';
    addMessage('user', prompt);
    reply = addMessage('assistant', '');
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
    try {
        await clearModelCache();
        element('cached').textContent = '0 MB';
        element('warning').textContent = 'Model cache cleared';
    } catch (error) { element('warning').textContent = error.message; }
});
addEventListener('pagehide', () => {
    if (busy) {
        worker?.postMessage({type: 'cancel'});
        settleInterruptedTurn();
    }
    worker?.terminate();
    worker = null;
});
addEventListener('pageshow', event => { if (event.persisted) restart(); });

renderHistory();
renderMessages();
controls();