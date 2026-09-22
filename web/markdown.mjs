import {marked} from './libs/marked/marked.esm.js';
import DOMPurify from './libs/dompurify/purify.es.mjs';
import hljs from './libs/highlightjs/highlight.min.js';

function updateDisclosure(element) {
    const body = element.querySelector('.markdown-body');
    const toggle = element.querySelector('.message-expand');
    if (!body || !toggle) return;
    const long = body.scrollHeight > 320;
    toggle.hidden = !long;
    body.classList.toggle('collapsed', long && toggle.getAttribute('aria-expanded') !== 'true');
}

addEventListener('resize', () => {
    for (const element of document.querySelectorAll('.content')) updateDisclosure(element);
});

export function renderMarkdown(element, source, {streaming = false} = {}) {
    const fragment = DOMPurify.sanitize(marked.parse(source, {gfm: true, breaks: true}), {
        ALLOWED_TAGS: ['p', 'br', 'strong', 'em', 'del', 'blockquote', 'ul', 'ol', 'li',
            'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'pre', 'code', 'hr', 'a',
            'table', 'thead', 'tbody', 'tr', 'th', 'td'],
        ALLOWED_ATTR: ['href', 'title', 'start', 'align', 'class'],
        ALLOW_DATA_ATTR: false,
        ALLOW_ARIA_ATTR: false,
        RETURN_DOM_FRAGMENT: true
    });
    const diagrams = [];
    for (const node of fragment.querySelectorAll('[class]')) {
        const language = node.tagName === 'CODE' && /^language-([\w+-]+)$/.exec(node.className)?.[1];
        node.removeAttribute('class');
        if (!language || node.parentElement?.tagName !== 'PRE') continue;
        if (language.toLowerCase() === 'mermaid') {
            if (!streaming) diagrams.push(node);
        } else if (hljs.getLanguage(language)) {
            node.innerHTML = DOMPurify.sanitize(hljs.highlight(node.textContent, {language, ignoreIllegals: true}).value,
                {ALLOWED_TAGS: ['span'], ALLOWED_ATTR: ['class']});
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
    if (!streaming && source) {
        const toggle = document.createElement('button');
        toggle.type = 'button';
        toggle.className = 'message-expand';
        toggle.textContent = 'Show more';
        toggle.hidden = true;
        toggle.setAttribute('aria-expanded', 'false');
        body.id = `message-${crypto.randomUUID()}`;
        toggle.setAttribute('aria-controls', body.id);
        const expand = expanded => {
            toggle.setAttribute('aria-expanded', String(expanded));
            toggle.textContent = expanded ? 'Show less' : 'Show more';
            updateDisclosure(element);
        };
        toggle.addEventListener('click', () => {
            const expanded = toggle.getAttribute('aria-expanded') !== 'true';
            expand(expanded);
            if (!expanded && element.getBoundingClientRect().top < 0)
                element.scrollIntoView({block: 'start'});
        });
        body.addEventListener('focusin', () => { if (body.classList.contains('collapsed')) expand(true); });
        element.append(toggle);
        requestAnimationFrame(() => updateDisclosure(element));
    }
    for (const code of diagrams) {
        const scroll = code.closest('.markdown-scroll');
        const details = document.createElement('details');
        details.className = 'diagram-source';
        const summary = document.createElement('summary');
        summary.textContent = 'Diagram source';
        scroll.replaceWith(details);
        details.append(summary, scroll);
        details.open = true;
        details.addEventListener('toggle', () => updateDisclosure(element));
        import('./mermaid.mjs').then(module => module.renderDiagram(code.textContent)).then(frame => {
            if (!element.contains(details)) return;
            details.before(frame);
            details.open = false;
            frame.addEventListener('load', () => updateDisclosure(element));
            updateDisclosure(element);
        }).catch(() => {
            summary.textContent = 'Diagram unavailable - view source';
            updateDisclosure(element);
        });
    }
}