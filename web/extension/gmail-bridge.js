(() => {
  if (globalThis.__kidiGmailBridge) return;
  globalThis.__kidiGmailBridge = true;
  const api = globalThis.browser || globalThis.chrome;
  const protectedContent = '.gmail_signature, [data-smartmail="gmail_signature"], .gmail_quote, blockquote[type="cite"]';
  const draftIds = new WeakMap();
  const captures = new Map();
  let lastEditor;
  let notificationTimer;

  function visible(element) {
    return element?.isConnected && !element.closest('[hidden], [aria-hidden="true"]') &&
      element.getClientRects().length > 0 && getComputedStyle(element).visibility !== 'hidden';
  }
  function editors() {
    return [...document.querySelectorAll(
      '[g_editable="true"][contenteditable="true"][role="textbox"], ' +
      '[role="dialog"] [contenteditable="true"][role="textbox"], ' +
      '[role="main"] .adn [contenteditable="true"][role="textbox"]',
    )].filter(element => visible(element) && !element.closest(protectedContent));
  }
  function draftId(editor) {
    if (!draftIds.has(editor)) draftIds.set(editor, crypto.randomUUID());
    return draftIds.get(editor);
  }
  function subject(editor) {
    const root = editor?.closest('[role="dialog"]') || editor?.closest('form') ||
      editor?.closest('.adn') || document.querySelector('[role="main"]');
    return root?.querySelector('input[name="subjectbox"]')?.value ||
      document.querySelector('[role="main"] h2')?.textContent?.trim() || 'Untitled email';
  }
  function emailBody() {
    const candidates = [...document.querySelectorAll('[role="main"] .adn .a3s, [role="main"] [data-message-id] .a3s')]
      .filter(element => visible(element) && !element.closest('[contenteditable="true"]'));
    const selection = getSelection();
    return candidates.find(element => selection?.anchorNode && element.contains(selection.anchorNode)) ||
      candidates.at(-1) || null;
  }
  function inspect() {
    const available = editors();
    return {
      drafts: available.map(editor => ({id: draftId(editor), title: subject(editor)})),
      activeDraftId: available.includes(lastEditor) ? draftId(lastEditor) :
        available.length === 1 ? draftId(available[0]) : null,
      emailAvailable: Boolean(emailBody()),
      emailTitle: document.querySelector('[role="main"] h2')?.textContent?.trim() || 'Opened email',
    };
  }
  function plainText(node) {
    let text = '';
    function visit(current) {
      if (current.nodeType === Node.TEXT_NODE) {
        text += current.textContent;
        return;
      }
      if (current.nodeType === Node.ELEMENT_NODE && current.matches(protectedContent)) return;
      if (current.nodeName === 'BR') {
        text += '\n';
        return;
      }
      const block = /^(DIV|P|LI|UL|OL|PRE|H[1-6])$/u.test(current.nodeName);
      if (block && text && !text.endsWith('\n')) text += '\n';
      for (const child of current.childNodes) visit(child);
      if (block && text && !text.endsWith('\n')) text += '\n';
    }
    visit(node);
    return text.replace(/^\n+|\n+$/gu, '');
  }
  function bodyRange(editor) {
    const range = document.createRange();
    range.selectNodeContents(editor);
    const firstProtected = editor.querySelector(protectedContent);
    if (firstProtected) range.setEndBefore(firstProtected);
    return range;
  }
  function editableRange(editor) {
    const selection = getSelection();
    if (!selection?.isCollapsed && selection?.rangeCount) {
      const range = selection.getRangeAt(0);
      if (editor.contains(range.startContainer) && editor.contains(range.endContainer)) {
        if ([...editor.querySelectorAll(protectedContent)].some(node => range.intersectsNode(node)))
          throw new Error('Select only the email body, excluding the signature and quoted history.');
        return {range: range.cloneRange(), selected: true};
      }
    }
    return {range: bodyRange(editor), selected: false};
  }
  function remember(capture) {
    for (const [id, value] of captures)
      if (Date.now() - value.createdAt > 30 * 60 * 1000) captures.delete(id);
    while (captures.size >= 16) captures.delete(captures.keys().next().value);
    const id = crypto.randomUUID();
    captures.set(id, {...capture, createdAt: Date.now(), url: location.href});
    return id;
  }
  function captureDraft(id) {
    const available = editors();
    const editor = available.find(element => draftId(element) === id) ||
      (!id && available.length === 1 ? available[0] : null);
    if (!editor) throw new Error('Choose an available Gmail draft first.');
    const {range, selected} = editableRange(editor);
    const text = plainText(range.cloneContents());
    const target = remember({kind: 'draft', editor, range, html: editor.innerHTML});
    return {
      target,
      kind: 'gmail-draft',
      title: subject(editor),
      text,
      operation: 'replace',
      operationLabel: selected ? 'Replace selection' : 'Replace draft',
    };
  }
  function replyEditors() {
    const main = document.querySelector('[role="main"]');
    return editors().filter(editor => main?.contains(editor) && !editor.closest('[role="dialog"]') &&
      !editor.closest('form')?.querySelector('input[name="subjectbox"]'));
  }
  function captureEmail() {
    const email = emailBody();
    if (!email) throw new Error('Open an email in Gmail first. Inbox snippets are not read.');
    const root = email.closest('.adn, [data-message-id]');
    const title = document.querySelector('[role="main"] h2')?.textContent?.trim() || 'Opened email';
    const from = root?.querySelector('[email]')?.getAttribute('email') || '';
    const text = plainText(email);
    const replies = replyEditors();
    const editor = replies.length === 1 ? replies[0] : null;
    const target = remember({
      kind: 'reply', email, emailHtml: email.innerHTML,
      editor, html: editor?.innerHTML,
    });
    return {
      target,
      kind: 'gmail-email',
      title,
      text: `Subject: ${title}${from ? `\nFrom: ${from}` : ''}\n\n${text}`,
      operation: 'reply',
      operationLabel: 'Insert into reply',
    };
  }
  function inputEvent(type, text, cancelable = false) {
    return new InputEvent(type, {
      bubbles: true, composed: true, cancelable, inputType: 'insertText', data: text,
    });
  }
  function insert(editor, range, expectedHtml, text) {
    if (!visible(editor) || editor.innerHTML !== expectedHtml ||
        !editor.contains(range.commonAncestorContainer))
      throw new Error('The draft changed. Capture it again before applying this result.');
    editor.focus();
    if (!editor.dispatchEvent(inputEvent('beforeinput', text, true)))
      throw new Error('Gmail rejected the insertion. Copy the result instead.');
    if (editor.innerHTML !== expectedHtml)
      throw new Error('The draft changed before insertion. Capture it again.');
    const selection = getSelection();
    const fragment = document.createDocumentFragment();
    for (const [index, line] of text.replace(/\r\n?/gu, '\n').split('\n').entries()) {
      if (index) fragment.append(document.createElement('br'));
      fragment.append(document.createTextNode(line));
    }
    const last = fragment.lastChild;
    // insertText can merge the following signature paragraph into the selection.
    range.deleteContents();
    range.insertNode(fragment);
    range.setStartAfter(last);
    range.collapse(true);
    selection.removeAllRanges();
    selection.addRange(range);
    editor.dispatchEvent(inputEvent('input', text));
  }
  function apply(target, text) {
    const capture = captures.get(target);
    if (!capture || Date.now() - capture.createdAt > 30 * 60 * 1000)
      throw new Error('This Gmail context expired. Capture it again.');
    if (capture.url !== location.href)
      throw new Error('Gmail navigated to another view. Capture the current email again.');
    if (typeof text !== 'string' || !text.trim() || new TextEncoder().encode(text).length > 64 * 1024)
      throw new Error('The result is empty or exceeds the insertion limit.');
    if (capture.kind === 'draft') {
      insert(capture.editor, capture.range, capture.html, text);
    } else {
      if (!visible(capture.email) || capture.email.innerHTML !== capture.emailHtml)
        throw new Error('The source email changed. Capture it again.');
      const replies = replyEditors();
      const editor = capture.editor || (replies.length === 1 ? replies[0] : null);
      if (!editor || !replies.includes(editor))
        throw new Error('Open a reply to this email in Gmail, then try Insert into reply again.');
      const range = bodyRange(editor);
      if (plainText(range.cloneContents()).trim())
        throw new Error('The reply already contains text. Use Polish draft, or copy this result.');
      insert(editor, range, capture.html ?? editor.innerHTML, text);
    }
    captures.delete(target);
    return {applied: true};
  }
  function notify() {
    clearTimeout(notificationTimer);
    notificationTimer = setTimeout(() => {
      api.runtime.sendMessage({type: 'kidi.gmail.changed'}).catch(error => {
        if (!String(error.message).includes('Extension context invalidated'))
          console.debug('Kidi Gmail connection unavailable:', error.message);
      });
    }, 250);
  }
  document.addEventListener('focusin', event => {
    const editor = event.target.closest?.('[contenteditable="true"][role="textbox"]');
    if (editor && editors().includes(editor)) {
      lastEditor = editor;
      notify();
    }
  });
  const observer = new MutationObserver(records => {
    if (records.some(record => record.type === 'childList' &&
        [...record.addedNodes, ...record.removedNodes].some(node => node.nodeType === Node.ELEMENT_NODE)))
      notify();
  });
  observer.observe(document.body, {childList: true, subtree: true});
  addEventListener('hashchange', notify);
  api.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (sender.id !== api.runtime.id || !['kidi.gmail.inspect', 'kidi.gmail.capture', 'kidi.gmail.apply']
      .includes(message?.type)) return false;
    try {
      let result;
      if (message.type === 'kidi.gmail.inspect') result = inspect();
      else if (message.type === 'kidi.gmail.capture')
        result = message.kind === 'email' ? captureEmail() : captureDraft(message.draftId);
      else result = apply(message.target, message.text);
      if (result.text && new TextEncoder().encode(result.text).length > 64 * 1024)
        throw new Error('This email exceeds the 64 KiB context limit. Select a shorter passage.');
      sendResponse({ok: true, ...result});
    } catch (error) {
      sendResponse({ok: false, error: error.message || String(error)});
    }
    return false;
  });
  notify();
})();
