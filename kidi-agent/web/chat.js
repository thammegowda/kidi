import {call, on} from './rpc.js';
import {endStreaming, renderMarkdown, renderStreamingMarkdown} from './liteapp/markdown/markdown.mjs';
import {openDiff, openFile, showDiff, unifiedDiff} from './editor.js';
import {hasImages, setupAttachments, takeImages} from './attachments.js';
import {readAloudButton} from './speech.js';

const messages = document.getElementById('messages');
const prompt = document.getElementById('prompt');
const sendButton = document.getElementById('send');
const stopButton = document.getElementById('stop');

let turn = null;     // the assistant block of the running turn
let segment = null;  // the reply being streamed inside it
const toolCards = new Map();
const approvals = new Map();
let busy = false;
let renderPending = new Set();

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function nearBottom() {
  return messages.scrollHeight - messages.scrollTop - messages.clientHeight < 80;
}

function append(node) {
  const follow = nearBottom();
  messages.append(node);
  if (follow) messages.scrollTop = messages.scrollHeight;
  return node;
}

function scrollIfFollowing(follow) {
  if (follow) messages.scrollTop = messages.scrollHeight;
}

// Coalesces Markdown re-rendering of streamed text to one pass per frame.
function scheduleRender(target) {
  renderPending.add(target);
  if (renderPending.size > 1) return;
  requestAnimationFrame(() => {
    const follow = nearBottom();
    for (const item of renderPending) {
      if (item.kind === 'answer') renderStreamingMarkdown(item.body, item.text);
      else item.body.textContent = item.text;
    }
    renderPending = new Set();
    scrollIfFollowing(follow);
  });
}

export function setBusy(value) {
  busy = value;
  sendButton.hidden = value;
  stopButton.hidden = !value;
  prompt.placeholder = value ? 'Kidi is working… (Stop to interrupt)' : 'Ask Kidi… (/ for commands)';
  document.body.classList.toggle('busy', value);
}

function userMessage(text, images = []) {
  const node = element('div', 'message user');
  const bubble = element('div', 'bubble', text);
  if (images.length) {
    const gallery = element('div', 'images');
    for (const url of images) {
      const image = document.createElement('img');
      image.src = url;
      image.alt = 'Attached image';
      gallery.append(image);
    }
    bubble.prepend(gallery);
  }
  node.append(bubble);
  append(node);
}

function startTurn() {
  turn = append(element('div', 'message assistant'));
  segment = null;
  return turn;
}

function ensureTurn() {
  return turn ?? startTurn();
}

function newSegment() {
  const container = ensureTurn();
  segment = {
    thinking: null,
    answer: null,
    node: container.appendChild(element('div', 'segment')),
  };
  return segment;
}

function thinkingBlock(target, text, {open = false} = {}) {
  const details = element('details', 'thinking');
  details.open = open;
  const summary = element('summary', '', 'Thinking');
  const body = element('div', 'thinking-body');
  details.append(summary, body);
  target.node.prepend(details);
  target.thinking = {kind: 'thinking', body, text, details, summary};
  body.textContent = text;
  return target.thinking;
}

function answerBlock(target, text) {
  const body = element('div', 'markdown');
  target.node.append(body, readAloudButton(body));
  target.answer = {kind: 'answer', body, text};
  renderMarkdown(body, text);
  return target.answer;
}

function onDelta({text, thinking}) {
  if (!segment) newSegment();
  if (thinking) {
    const block = segment.thinking ?? thinkingBlock(segment, '', {open: true});
    block.text += text;
    block.summary.textContent = 'Thinking…';
    scheduleRender(block);
  } else {
    if (segment.thinking) {
      segment.thinking.details.open = false;
      segment.thinking.summary.textContent = 'Thinking';
    }
    const block = segment.answer ?? answerBlock(segment, '');
    block.text += text;
    scheduleRender(block);
  }
}

function onReply(reply) {
  const target = segment ?? newSegment();
  if (!reply.streamed_reasoning && reply.reasoning_content) thinkingBlock(target, reply.reasoning_content);
  if (target.thinking) {
    target.thinking.details.open = false;
    target.thinking.summary.textContent = 'Thinking';
  }
  if (!reply.streamed_content && reply.content) answerBlock(target, reply.content);
  // The streamed text is complete: render it once more as a whole, including its last line.
  else if (target.answer) {
    renderPending.delete(target.answer);
    endStreaming(target.answer.body);
    renderMarkdown(target.answer.body, target.answer.text);
  }
  segment = null;
}

function describe(name, args) {
  if (!args || typeof args !== 'object') return String(args ?? '');
  if (typeof args.command === 'string') return '$ ' + args.command;
  if (typeof args.path === 'string') return args.path + (args.offset ? ` @${args.offset}` : '');
  return JSON.stringify(args);
}

function toolCard(id, name, args, {done = false, result = '', error = false} = {}) {
  const card = element('details', 'tool');
  const summary = element('summary');
  const status = element('span', 'tool-status running');
  const label = element('span', 'tool-name', name);
  const detail = element('span', 'tool-detail', describe(name, args).split('\n')[0]);
  const timer = element('span', 'tool-time');
  summary.append(status, label, detail, timer);
  const body = element('div', 'tool-body');
  if (typeof args?.command === 'string' && args.command.includes('\n')) body.append(element('pre', 'tool-args', args.command));
  if (typeof args?.content === 'string' && typeof args?.path === 'string') {
    const actions = element('div', 'tool-actions');
    const view = element('button', 'ghost small', 'View change');
    view.type = 'button';
    view.addEventListener('click', event => { event.preventDefault(); openDiff(args.path, 'session'); });
    const open = element('button', 'ghost small', 'Open file');
    open.type = 'button';
    open.addEventListener('click', event => { event.preventDefault(); openFile(args.path); });
    actions.append(view, open);
    body.append(actions);
  } else if (typeof args?.path === 'string' && name === 'read_file') {
    const open = element('button', 'ghost small', 'Open file');
    open.type = 'button';
    open.addEventListener('click', event => { event.preventDefault(); openFile(args.path); });
    body.append(open);
  }
  const output = element('pre', 'tool-output');
  body.append(output);
  card.append(summary, body);
  const entry = {card, status, output, timer, started: performance.now(), interval: null};
  toolCards.set(id, entry);
  if (done) finishTool(entry, result, error, false);
  else entry.interval = setInterval(() => {
    entry.timer.textContent = ((performance.now() - entry.started) / 1000).toFixed(1) + 's';
  }, 100);
  return card;
}

function formatResult(content) {
  try {
    const value = JSON.parse(content);
    if (value && typeof value === 'object' && 'output' in value) {
      const lines = [];
      if (value.output) lines.push(value.output.replace(/\n$/, ''));
      lines.push(`exit ${value.exit_code}${value.timed_out ? ' (timed out)' : ''}${value.truncated ? ' (truncated)' : ''}`);
      return lines.join('\n');
    }
    if (value && typeof value === 'object' && 'error' in value) return value.error;
    if (value && typeof value === 'object' && 'content' in value) return value.content;
    if (Array.isArray(value)) return value.join('\n');
  } catch {}
  return content;
}

function finishTool(entry, content, error, timed = true) {
  clearInterval(entry.interval);
  entry.interval = null;
  entry.status.className = 'tool-status ' + (error ? 'failed' : 'done');
  if (!timed) entry.timer.textContent = '';
  entry.output.textContent = formatResult(content);
  if (error) entry.card.open = true;
}

function onTool({id, name, arguments: args}) {
  segment = null;
  ensureTurn().append(toolCard(id, name, args));
  scrollIfFollowing(nearBottom());
}

function onToolResult({id, content, error}) {
  const entry = toolCards.get(id);
  if (entry) finishTool(entry, content, error);
}

function onApproval(payload) {
  const card = element('div', 'approval');
  card.append(element('div', 'approval-title', `Allow ${payload.tool}?`));
  const args = payload.arguments ?? {};
  if (payload.diff) {
    const header = element('div', 'approval-detail',
      `${payload.diff.exists ? 'Edit' : 'Create'} ${payload.diff.path}`);
    const preview = element('div', 'approval-diff');
    card.append(header, preview);
    requestAnimationFrame(() => unifiedDiff(preview, payload.diff.path, payload.diff.old, payload.diff.new));
    const full = element('button', 'ghost small', 'Open side by side');
    full.type = 'button';
    full.addEventListener('click', () => showDiff(payload.diff.path,
      {old: payload.diff.old, new: payload.diff.new, label: 'proposed change'}, {title: 'Proposed by the agent'}));
    card.append(full);
  } else {
    card.append(element('pre', 'approval-args', typeof args.command === 'string'
      ? '$ ' + args.command : JSON.stringify(args, null, 2)));
  }
  if (payload.unsandboxed)
    card.append(element('div', 'approval-warning',
      'Runs as your user, unsandboxed. "Always this session" allows any command.'));
  const buttons = element('div', 'approval-buttons');
  for (const [label, decision, style] of [['Allow', 'once', 'primary'], ['Always this session', 'session', ''],
    ['Deny', 'deny', 'danger']]) {
    const button = element('button', style, label);
    button.type = 'button';
    button.addEventListener('click', () => call('approval.respond', {id: payload.id, decision}).catch(() => {}));
    buttons.append(button);
  }
  card.append(buttons);
  approvals.set(payload.id, {card, payload});
  ensureTurn().append(card);
  messages.scrollTop = messages.scrollHeight;
}

// An answered approval shrinks to one line so the conversation stays compact.
function onApprovalDone({id, decision}) {
  const entry = approvals.get(id);
  if (!entry) return;
  approvals.delete(id);
  const {card, payload} = entry;
  const allowed = decision === 'allow';
  const label = allowed ? 'Allowed' : decision === 'deny' ? 'Denied' : 'Cancelled';
  card.className = `approval resolved ${allowed ? 'allowed' : 'refused'}`;
  card.replaceChildren(
    element('span', 'approval-mark', allowed ? '✓' : '✗'),
    element('span', 'approval-result', label),
    element('span', 'tool-name', payload.tool),
    element('span', 'tool-detail', payload.diff ? payload.diff.path : describe(payload.tool, payload.arguments)));
}

export function notice(text, kind = 'info') {
  const node = element('div', `notice ${kind}`, text);
  if (turn) turn.append(node);
  else append(node);
}

// A tool needs a service connected: Connect signs in through the browser and the tool call resumes on its own.
const setups = new Map();

function onSetup(payload) {
  const card = element('div', 'approval');
  card.append(element('div', 'approval-title', `Connect ${payload.service}?`),
    element('div', 'approval-detail', `${payload.tool} needs it: ${payload.reason}.`));
  const status = element('div', 'approval-detail');
  const buttons = element('div', 'approval-buttons');
  const connect = element('button', 'primary', `Connect ${payload.service}`);
  const later = element('button', '', 'Not now');
  for (const button of [connect, later]) button.type = 'button';
  const entry = {card, payload, status, tab: null};
  connect.addEventListener('click', () => {
    // A tab opened inside the click is allowed by popup blockers; it is pointed at the sign-in page once known.
    entry.tab = window.open('', '_blank');
    if (entry.tab) entry.tab.opener = null;
    buttons.remove();
    status.textContent = 'Opening the sign-in page…';
    call('setup.respond', {id: payload.id, decision: 'connect', origin: location.origin}).catch(error => {
      status.textContent = error.message;
      entry.tab?.close();
    });
  });
  later.addEventListener('click', () => call('setup.respond', {id: payload.id, decision: 'decline'}).catch(() => {}));
  buttons.append(connect, later);
  card.append(buttons, status);
  setups.set(payload.id, entry);
  ensureTurn().append(card);
  messages.scrollTop = messages.scrollHeight;
}

function onSetupOpen({id, url}) {
  const entry = setups.get(id);
  if (!entry) return;
  if (entry.tab && !entry.tab.closed) entry.tab.location = url;
  const link = element('a', '', 'sign-in page');
  link.href = url;
  link.target = '_blank';
  link.rel = 'noopener';
  entry.status.replaceChildren(`Finish signing in to ${entry.payload.service} in the `, link,
    ' that opened; this continues on its own.');
}

function onSetupDone({id, decision}) {
  const entry = setups.get(id);
  if (!entry) return;
  setups.delete(id);
  const signedIn = decision === 'returned';
  entry.card.className = `approval resolved ${signedIn ? 'allowed' : 'refused'}`;
  entry.card.replaceChildren(
    element('span', 'approval-mark', signedIn ? '✓' : '✗'),
    element('span', 'approval-result', signedIn ? 'Signed in to' : decision === 'declined' ? 'Not connected:' : 'Cancelled:'),
    element('span', 'tool-name', entry.payload.service));
}

function onDone(result) {
  for (const entry of toolCards.values()) if (entry.interval) finishTool(entry, '', true);
  // The session already reported the failure as a notice; only cancellations need saying.
  if (!result.ok && /^cancelled|approval cancelled/.test(result.error ?? '')) notice('Stopped.', 'warning');
  turn = null;
  segment = null;
  setBusy(false);
}

// Draws a saved conversation: user messages, replies with their reasoning and tool calls, and tool results.
export function renderHistory(history) {
  messages.replaceChildren();
  toolCards.clear();
  const results = new Map(history.filter(item => item.role === 'tool').map(item => [item.tool_call_id, item.content]));
  turn = null;
  for (const message of history) {
    if (message.role === 'user') {
      userMessage(message.content, message.images);
      turn = null;
    } else if (message.role === 'assistant') {
      const target = newSegment();
      if (message.reasoning_content) thinkingBlock(target, message.reasoning_content);
      if (message.content) answerBlock(target, message.content);
      for (const call of message.tool_calls ?? []) {
        const result = results.get(call.id);
        let failed = false;
        try { failed = 'error' in JSON.parse(result ?? '{}'); } catch {}
        turn.append(toolCard(call.id, call.name, call.arguments, {done: true, result: result ?? '', error: failed}));
      }
    }
  }
  turn = null;
  segment = null;
  messages.scrollTop = messages.scrollHeight;
}

export function clear() {
  messages.replaceChildren();
  turn = null;
  segment = null;
}

const HELP = `Commands: /new starts a conversation · /reset clears it · /compact summarizes older history ·
/context shows token use · /permissions lists session approvals (/permissions reset revokes them)`;

async function command(text) {
  const [name, ...rest] = text.trim().split(/\s+/);
  try {
    if (name === '/help') return notice(HELP);
    if (name === '/new') return document.getElementById('new-chat').click();
    if (name === '/reset') {
      await call('chat.reset');
      clear();
      return notice('Conversation cleared; session approvals unchanged.', 'context');
    }
    if (name === '/compact') {
      setBusy(true);
      return await call('chat.compact');
    }
    if (name === '/context') {
      const usage = await call('chat.context');
      return notice(`${usage.prompt_tokens} / ${usage.context_tokens} context tokens (including the system prompt and tools)`,
        'context');
    }
    if (name === '/permissions') {
      const result = await call(rest[0] === 'reset' ? 'permissions.reset' : 'permissions.get');
      return notice(`Permission mode: ${result.mode}. Session-approved tools: ${result.grants.join(', ') || 'none'}`,
        'context');
    }
    notice(`Unknown command ${name}. ${HELP}`, 'warning');
  } catch (error) {
    setBusy(false);
    notice(error.message, 'error');
  }
}

async function send() {
  let text = prompt.value;
  if ((!text.trim() && !hasImages()) || busy) return;
  if (text.trim().startsWith('/') && !hasImages()) {
    prompt.value = '';
    resize();
    return command(text);
  }
  if (!text.trim()) text = 'Describe this image.';
  prompt.value = '';
  resize();
  const images = takeImages();
  setBusy(true);
  try {
    await call('chat.send', images.length ? {text, images} : {text});
  } catch (error) {
    setBusy(false);
    notice(error.message, 'error');
  }
}

function resize() {
  prompt.style.height = 'auto';
  prompt.style.height = Math.min(prompt.scrollHeight, 200) + 'px';
}

prompt.addEventListener('input', resize);
prompt.addEventListener('keydown', event => {
  if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    send();
  }
});
document.getElementById('composer').addEventListener('submit', event => {
  event.preventDefault();
  send();
});
stopButton.addEventListener('click', () => call('chat.cancel').catch(() => {}));

setupAttachments(text => notice(text, 'warning'));

on('user', ({text, images}) => {
  userMessage(text, images);
  startTurn();
  setBusy(true);
});
on('delta', onDelta);
on('reply', onReply);
on('tool', onTool);
on('tool_result', onToolResult);
on('approval', onApproval);
on('approval_done', onApprovalDone);
on('setup', onSetup);
on('setup_open', onSetupOpen);
on('setup_done', onSetupDone);
on('notice', ({text, kind}) => notice(text, kind));
on('compact', ({started, detail}) => notice(started ? 'Compacting the conversation…' : detail, 'context'));
on('done', onDone);
