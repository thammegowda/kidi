(() => {
  if (globalThis.__kidiArticleBridge) return;
  globalThis.__kidiArticleBridge = true;
  const api = globalThis.browser || globalThis.chrome;
  const MAX_BYTES = 64 * 1024;
  const BLOCKS = 'p, h1, h2, h3, h4, h5, h6, li, blockquote, pre';
  const encoder = new TextEncoder();

  function bounded(text) {
    const bytes = encoder.encode(text.slice(0, MAX_BYTES + 2));
    if (bytes.length <= MAX_BYTES) return {text, truncated: false};
    let prefix = new TextDecoder().decode(bytes.subarray(0, MAX_BYTES), {stream: true});
    const paragraph = prefix.lastIndexOf('\n\n');
    if (paragraph > prefix.length / 2) prefix = prefix.slice(0, paragraph);
    else {
      const word = prefix.lastIndexOf(' ');
      if (word > prefix.length * .9) prefix = prefix.slice(0, word);
    }
    return {text: prefix.trimEnd(), truncated: true};
  }
  function capture() {
    const url = location.href;
    if (location.hostname === 'mail.google.com')
      throw new Error('Use the Gmail draft or reply actions instead of reading the whole mailbox page.');
    const selection = getSelection()?.toString()?.trim();
    let text, title = document.title, language = document.documentElement.lang || '', sourceKind;
    if (selection) {
      text = selection;
      sourceKind = 'selection';
    } else {
      if (typeof Readability !== 'function') throw new Error('The packaged article extractor is unavailable.');
      const originals = [...document.querySelectorAll('*')];
      if (originals.length > 50000)
        throw new Error('This page is too complex to extract safely. Select the passage you want instead.');
      const clone = document.cloneNode(true);
      const copies = [...clone.querySelectorAll('*')];
      for (let index = 0; index < originals.length; index++) {
        const element = originals[index];
        if (element === document.documentElement || element === document.body) continue;
        const style = getComputedStyle(element);
        if (element.matches('script, style, noscript, template, nav, aside, footer, form, input, textarea, ' +
            'select, button, [hidden], [aria-hidden="true"], [role="navigation"], [role="banner"], ' +
            '[role="complementary"]') || style.display === 'none' ||
            ['hidden', 'collapse'].includes(style.visibility) || style.opacity === '0')
          copies[index]?.remove();
      }
      const article = new Readability(clone, {maxElemsToParse: 50000}).parse();
      if (!article?.content)
        throw new Error('No readable article was found. Select a passage on the page and try again.');
      const parsed = new DOMParser().parseFromString(article.content, 'text/html');
      const blocks = [...parsed.body.querySelectorAll(BLOCKS)]
        .filter(element => !element.parentElement.closest(BLOCKS))
        .map(element => element.textContent.replace(/[ \t]+/gu, ' ').trim())
        .filter(Boolean);
      text = blocks.length ? blocks.join('\n\n') : article.textContent?.trim();
      title = article.title || title;
      language = article.lang || language;
      sourceKind = 'article';
    }
    if (!text?.trim()) throw new Error('There is no readable text in this context.');
    if (location.href !== url) throw new Error('The page navigated during capture. Try again.');
    const result = bounded(text);
    return {
      kind: 'article', title, language, sourceKind, sourceUrl: url,
      originalCharacters: characterCount(text), returnedCharacters: characterCount(result.text),
      ...result, operation: null,
      warning: result.truncated
        ? 'Only the first 64 KiB of readable text was captured. Results and narration cover the attached text only.' : '',
    };
  }
  function characterCount(text) {
    let count = 0;
    for (const _character of text) count++;
    return count;
  }
  api.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (sender.id !== api.runtime.id || message?.type !== 'kidi.page.capture') return false;
    try { sendResponse({ok: true, ...capture()}); }
    catch (error) { sendResponse({ok: false, error: error.message || String(error)}); }
    return false;
  });
})();
