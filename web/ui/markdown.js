// Message rendering for the Kidi app: LiteApp's sanitized Markdown (code, math, tables) plus Mermaid diagrams and a
// "Show more" toggle for long messages.
import {
  renderMarkdown as renderBody, renderStreamingMarkdown as renderStream,
} from './liteapp/markdown/markdown.mjs';

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

function options(element) {
  return {
    breaks: true,
    onChange: () => updateDisclosure(element),
    renderDiagram: (language, source) => language !== 'mermaid' ? null
      : import('./mermaid.js').then(module => module.renderDiagram(source)).then(frame => {
        frame.addEventListener('load', () => updateDisclosure(element));
        return frame;
      }),
  };
}

function markdownBody(element) {
  const body = document.createElement('div');
  body.className = 'markdown-body';
  element.replaceChildren(body);
  return body;
}

export function renderMarkdown(element, source) {
  const body = markdownBody(element);
  renderBody(body, source, options(element));
  if (!source) return;
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

export function renderStreamingMarkdown(element, source) {
  const body = element.firstElementChild?.classList.contains('markdown-body') ? element.firstElementChild
    : markdownBody(element);
  renderStream(body, source, options(element));
}
