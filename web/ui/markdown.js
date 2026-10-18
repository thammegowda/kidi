import {marked} from './libs/marked/marked.esm.js';
import DOMPurify from './libs/dompurify/purify.es.mjs';
import hljs from './libs/highlightjs/highlight.min.js';

const streaming = new WeakMap();

function updateDisclosure(element) {
  const body = element.querySelector('.markdown-body');
  const toggle = element.querySelector('.message-expand');
  if (!body || !toggle) return;
  const long = body.scrollHeight > 320;
  toggle.hidden = !long;
  body.classList.toggle('collapsed', long && toggle.getAttribute('aria-expanded') !== 'true');
}

addEventListener('resize', () => {
  for (const element of document.querySelectorAll('.message-content')) updateDisclosure(element);
});

function openFence(source) {
  let offset = 0;
  let active = null;
  for (const line of source.split(/(?<=\n)/)) {
    const match = /^ {0,3}(`{3,}|~{3,})[^\n]*\n?$/.exec(line);
    if (match) {
      if (!active) {
        active = {delimiter: match[1][0], length: match[1].length, start: offset, content: offset + line.length};
      } else if (match[1][0] === active.delimiter && match[1].length >= active.length) {
        active = null;
      }
    }
    offset += line.length;
  }
  return active;
}

export function renderMarkdown(element, source, {streaming: partial = false} = {}) {
  const fragment = DOMPurify.sanitize(marked.parse(source, {gfm: true, breaks: true}), {
    ALLOWED_TAGS: [
      'p', 'br', 'strong', 'em', 'del', 'blockquote', 'ul', 'ol', 'li',
      'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'pre', 'code', 'hr', 'a',
      'table', 'thead', 'tbody', 'tr', 'th', 'td',
    ],
    ALLOWED_ATTR: ['href', 'title', 'start', 'align', 'class'],
    ALLOW_DATA_ATTR: false,
    ALLOW_ARIA_ATTR: false,
    RETURN_DOM_FRAGMENT: true,
  });
  const diagrams = [];
  for (const node of fragment.querySelectorAll('[class]')) {
    const language = node.tagName === 'CODE' && /^language-([\w+-]+)$/.exec(node.className)?.[1];
    node.removeAttribute('class');
    if (!language || node.parentElement?.tagName !== 'PRE') continue;
    if (language.toLowerCase() === 'mermaid') {
      if (!partial) diagrams.push(node);
    } else if (hljs.getLanguage(language)) {
      node.innerHTML = DOMPurify.sanitize(
        hljs.highlight(node.textContent, {language, ignoreIllegals: true}).value,
        {ALLOWED_TAGS: ['span'], ALLOWED_ATTR: ['class']},
      );
    }
  }
  for (const link of fragment.querySelectorAll('a')) {
    const href = link.getAttribute('href');
    if (!href || !/^(https?:|mailto:)/i.test(href)) {
      link.removeAttribute('href');
      continue;
    }
    link.target = '_blank';
    link.rel = 'noopener noreferrer';
  }
  for (const block of fragment.querySelectorAll('pre, table')) {
    const scroll = document.createElement('div');
    scroll.className = 'markdown-scroll';
    scroll.tabIndex = 0;
    scroll.setAttribute('role', 'region');
    scroll.setAttribute('aria-label', block.tagName === 'PRE' ? 'Code block' : 'Table');
    block.replaceWith(scroll);
    scroll.append(block);
  }
  const body = document.createElement('div');
  body.className = 'markdown-body';
  body.append(fragment);
  element.replaceChildren(body);
  if (!partial && source) {
    const toggle = document.createElement('button');
    toggle.type = 'button'; toggle.className = 'message-expand'; toggle.textContent = 'Show more';
    toggle.hidden = true; toggle.setAttribute('aria-expanded', 'false');
    body.id = `message-${crypto.randomUUID()}`;
    toggle.setAttribute('aria-controls', body.id);
    toggle.addEventListener('click', () => {
      const expanded = toggle.getAttribute('aria-expanded') !== 'true';
      toggle.setAttribute('aria-expanded', String(expanded));
      toggle.textContent = expanded ? 'Show less' : 'Show more';
      updateDisclosure(element);
    });
    body.addEventListener('focusin', () => {
      if (body.classList.contains('collapsed')) {
        toggle.setAttribute('aria-expanded', 'true');
        toggle.textContent = 'Show less';
        updateDisclosure(element);
      }
    });
    element.append(toggle);
    requestAnimationFrame(() => updateDisclosure(element));
  }
  for (const code of diagrams) {
    const scroll = code.closest('.markdown-scroll');
    const details = document.createElement('details');
    details.className = 'diagram-source';
    const summary = document.createElement('summary');
    summary.textContent = 'Diagram source';
    scroll.replaceWith(details); details.append(summary, scroll); details.open = true;
    details.addEventListener('toggle', () => updateDisclosure(element));
    import('./mermaid.js').then(module => module.renderDiagram(code.textContent)).then(frame => {
      if (!element.contains(details)) return;
      details.before(frame); details.open = false;
      frame.addEventListener('load', () => updateDisclosure(element));
      updateDisclosure(element);
    }).catch(() => { summary.textContent = 'Diagram unavailable - view source'; });
  }
}

export function renderStreamingMarkdown(element, source) {
  let state = streaming.get(element);
  if (!state) {
    const stable = document.createElement('div');
    stable.className = 'markdown-stream-stable';
    const tail = document.createElement('span');
    tail.className = 'markdown-stream-tail';
    element.replaceChildren(stable, tail);
    state = {stable, tail, rendered: null, mode: 'text'};
    streaming.set(element, state);
  }

  const completedLine = source.lastIndexOf('\n') + 1;
  let renderable = source.slice(0, completedLine);
  let tail = source.slice(completedLine);
  let mode = 'text';
  const fence = openFence(renderable);
  if (fence) {
    renderable = renderable.slice(0, fence.start);
    tail = source.slice(fence.content);
    mode = 'code';
  }

  if (state.rendered !== renderable) {
    renderMarkdown(state.stable, renderable, {streaming: true});
    state.rendered = renderable;
  }
  if (state.mode !== mode) {
    state.tail.replaceChildren();
    state.tail.className = mode === 'code' ? 'markdown-stream-code' : 'markdown-stream-tail';
    if (mode === 'code') {
      const pre = document.createElement('pre');
      pre.append(document.createElement('code'));
      state.tail.append(pre);
    }
    state.mode = mode;
  }
  const destination = mode === 'code' ? state.tail.querySelector('code') : state.tail;
  destination.textContent = tail;
}
