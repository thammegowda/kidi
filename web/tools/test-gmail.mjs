import assert from 'node:assert/strict';
import {cp, mkdtemp, readFile, rm, writeFile} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';

import {chromium} from 'playwright';

const root = fileURLToPath(new URL('../../', import.meta.url));
const source = resolve(process.argv[2] || join(root, 'build-extension/chromium'));
const screenshot = process.argv[3];
const channel = process.env.KIDI_BROWSER_CHANNEL || 'chromium';
const temporary = await mkdtemp(join(tmpdir(), 'kidi-gmail-test-'));
const extension = join(temporary, 'extension');
let context;
try {
  await cp(source, extension, {recursive: true});
  const manifest = JSON.parse(await readFile(join(extension, 'manifest.json'), 'utf8'));
  // Only this synthetic test package pre-grants Gmail access; shipped packages remain opt-in.
  manifest.host_permissions = ['https://mail.google.com/*'];
  await writeFile(join(extension, 'manifest.json'), JSON.stringify(manifest));
  await cp(join(root, 'tests/browser/fixtures/extension-backend.mjs'), join(extension, 'backend.js'));
  const fixture = await readFile(join(root, 'tests/browser/fixtures/gmail.html'), 'utf8');
  context = await chromium.launchPersistentContext(join(temporary, 'profile'), {
    channel, headless: true, colorScheme: 'dark', viewport: {width: 390, height: 800},
    args: [`--disable-extensions-except=${extension}`, `--load-extension=${extension}`],
  });
  const unexpectedNetwork = [];
  await context.route('https://**/*', route => {
    if (new URL(route.request().url()).hostname === 'mail.google.com')
      return route.fulfill({status: 200, contentType: 'text/html', body: fixture});
    unexpectedNetwork.push(route.request().url());
    return route.abort();
  });
  const worker = context.serviceWorkers()[0] || await context.waitForEvent('serviceworker');
  const extensionId = new URL(worker.url()).host;
  const gmail = await context.newPage();
  await gmail.goto('https://mail.google.com/mail/u/0/#inbox/test-thread');
  await gmail.evaluate(() => {
    globalThis.sendClicks = 0;
    for (const button of document.querySelectorAll('[data-send]'))
      button.addEventListener('click', () => { globalThis.sendClicks++; });
  });
  const tabId = await worker.evaluate(async () => {
    const [tab] = await chrome.tabs.query({url: 'https://mail.google.com/*'});
    return tab.id;
  });
  const panel = await context.newPage();
  const errors = [];
  panel.on('pageerror', error => errors.push(error.message));
  await panel.goto(`chrome-extension://${extensionId}/editor.html?tab=${tabId}`);
  await panel.getByRole('button', {name: 'Polish draft', exact: true}).waitFor();
  assert.equal(await panel.locator('#task').count(), 0, 'chat does not contain the old action dropdown');
  assert.equal(await panel.locator('#conversation').getAttribute('aria-label'), 'Conversation');
  await panel.locator('#load-model').click();
  await panel.locator('#diagnostics').waitFor({state: 'visible'});
  assert.match(await panel.locator('#diagnostics').textContent(), /RAM128 MB/);
  const requests = () => panel.evaluate(async () => (await import('./backend.js')).backend.requests);
  const integration = (type, payload = {}) => panel.evaluate(
    ({type, payload, tabId}) => chrome.runtime.sendMessage({type, tabId, ...payload}),
    {type, payload, tabId});
  const ask = async text => {
    await panel.locator('#message').fill(text);
    await panel.locator('#send').click();
  };
  const waitForAnswers = count => panel.waitForFunction(expected =>
    document.querySelectorAll('.turn.assistant .result-actions').length === expected, count);
  const launchDraftId = (await integration('kidi.context.inspect')).drafts.find(draft => draft.title === 'Launch plan').id;
  await panel.locator('#gmail-draft').selectOption(launchDraftId);
  const signature = await gmail.locator('#draft-one .gmail_signature').innerHTML();
  const quotation = await gmail.locator('#draft-one .gmail_quote').innerHTML();
  const otherDraft = await gmail.locator('#draft-two').innerHTML();
  await panel.getByRole('button', {name: 'Polish draft', exact: true}).click();
  await waitForAnswers(1);
  let editorRequests = (await requests()).filter(request => request.type === 'editor.run');
  assert.equal(editorRequests.at(-1).payload.task, 'email_polish');
  assert.match(editorRequests.at(-1).payload.text, /rough launch draft/);
  assert.doesNotMatch(editorRequests.at(-1).payload.text, /My signature|quoted reply|other draft|unrelated private/);
  assert.match(await panel.locator('#context-label').textContent(), /Launch plan/);
  await ask('Make it warmer.');
  await waitForAnswers(2);
  editorRequests = (await requests()).filter(request => request.type === 'editor.run');
  assert.equal(editorRequests.at(-1).payload.history.length, 2);
  assert.equal(editorRequests.at(-1).payload.instruction, 'Make it warmer.');
  if (screenshot) await panel.screenshot({path: resolve(screenshot)});
  await panel.getByRole('button', {name: 'Replace draft', exact: true}).last().click();
  await panel.waitForFunction(() => document.querySelector('#notice').textContent.includes('has not been sent'));
  assert.match(await gmail.locator('#draft-one').innerText(), /The report will be ready on Monday/);
  assert.equal(await gmail.locator('#draft-one .gmail_signature').innerHTML(), signature);
  assert.equal(await gmail.locator('#draft-one .gmail_quote').innerHTML(), quotation);
  assert.equal(await gmail.locator('#draft-two').innerHTML(), otherDraft);

  await panel.locator('#message').fill('Ask to move the meeting to Monday.');
  await panel.getByRole('button', {name: 'Draft a reply', exact: true}).click();
  await waitForAnswers(1);
  editorRequests = (await requests()).filter(request => request.type === 'editor.run');
  assert.equal(editorRequests.at(-1).payload.task, 'email_reply');
  assert.match(editorRequests.at(-1).payload.text, /Could we meet on Friday/);
  assert.doesNotMatch(editorRequests.at(-1).payload.text, /unrelated private|Older quoted|launch draft/);
  await panel.getByRole('button', {name: 'Insert into reply', exact: true}).click();
  await panel.waitForFunction(() => document.querySelector('#notice').textContent.includes('Open a reply'));
  await gmail.locator('#inline-reply').evaluate(element => { element.hidden = false; });
  await panel.getByRole('button', {name: 'Insert into reply', exact: true}).click();
  await panel.waitForFunction(() => document.querySelector('#notice').textContent.includes('has not been sent'));
  assert.match(await gmail.locator('#reply-body').innerText(), /Monday instead/);
  assert.equal(await gmail.locator('#reply-body .gmail_signature').innerText(), 'My reply signature');
  assert.equal(await gmail.locator('#draft-two').innerHTML(), otherDraft);

  await panel.locator('#gmail-draft').selectOption(launchDraftId);
  await panel.getByRole('button', {name: 'Polish draft', exact: true}).click();
  await waitForAnswers(1);
  await gmail.locator('#draft-one').evaluate(element => {
    element.insertBefore(document.createTextNode('A newer edit. '), element.firstChild);
  });
  const changed = await gmail.locator('#draft-one').innerHTML();
  await panel.getByRole('button', {name: 'Replace draft', exact: true}).click();
  await panel.waitForFunction(() => document.querySelector('#notice').textContent.includes('draft changed'));
  assert.equal(await gmail.locator('#draft-one').innerHTML(), changed, 'stale generation must not overwrite edits');

  const details = await integration('kidi.context.inspect');
  const draftId = details.drafts.find(draft => draft.title === 'Launch plan').id;
  const captured = await integration('kidi.gmail.capture', {kind: 'draft', draftId});
  await gmail.locator('#draft-one').evaluate(element => {
    element.addEventListener('beforeinput', event => event.preventDefault(), {once: true});
  });
  const rejected = await integration('kidi.context.apply', {id: captured.task.id, text: 'Rejected edit'});
  assert.equal(rejected.ok, false);
  assert.match(rejected.error, /rejected the insertion/);
  assert.equal(await gmail.locator('#draft-one').innerHTML(), changed);
  const literal = 'Literal <img src=x onerror=alert(1)> text & punctuation';
  const applied = await integration('kidi.context.apply', {id: captured.task.id, text: literal});
  assert.equal(applied.ok, true);
  assert.equal(await gmail.locator('#draft-one img').count(), 0, 'generated text cannot inject HTML');
  assert.match(await gmail.locator('#draft-one').innerText(), /Literal <img/);
  assert.equal(await gmail.locator('#draft-one .gmail_signature').innerHTML(), signature);
  assert.equal(await gmail.locator('#draft-one .gmail_quote').innerHTML(), quotation);

  await gmail.locator('#draft-one').evaluate(element => {
    const walker = document.createTreeWalker(element, NodeFilter.SHOW_TEXT);
    let node;
    while ((node = walker.nextNode()) && !node.textContent.includes('Literal')) {}
    const range = document.createRange();
    range.setStart(node, 0);
    range.setEnd(node, 7);
    const selection = getSelection();
    selection.removeAllRanges();
    selection.addRange(range);
  });
  const passage = await integration('kidi.gmail.capture', {kind: 'draft', draftId});
  assert.equal(passage.task.text, 'Literal');
  assert.equal(passage.task.operationLabel, 'Replace selection');
  assert.equal((await integration('kidi.context.apply', {id: passage.task.id, text: 'Revised'})).ok, true);
  assert.match(await gmail.locator('#draft-one').innerText(), /Revised <img/);
  const beforeNavigation = await integration('kidi.gmail.capture', {kind: 'email'});
  await gmail.evaluate(() => { location.hash = '#inbox/other-thread'; });
  const stalePage = await integration('kidi.context.apply', {
    id: beforeNavigation.task.id, text: 'This must not enter another thread',
  });
  assert.equal(stalePage.ok, false);
  assert.match(stalePage.error, /source page changed/);
  assert.equal(await gmail.evaluate(() => globalThis.sendClicks), 0, 'no generated action sends an email');

  await panel.locator('#new-chat').click();
  await panel.waitForFunction(() => document.querySelectorAll('.turn').length === 0);
  await ask('Hello, Kidi.');
  await waitForAnswers(1);
  editorRequests = (await requests()).filter(request => request.type === 'editor.run');
  assert.equal(editorRequests.at(-1).payload.task, 'chat');
  assert.equal(editorRequests.at(-1).payload.text, '');
  assert.deepEqual(editorRequests.at(-1).payload.history, []);
  await ask('Show Markdown');
  await waitForAnswers(2);
  const markdown = panel.locator('.turn.assistant .turn-text').last();
  assert.equal(await markdown.locator('strong').textContent(), 'Bold result');
  assert.deepEqual(await markdown.locator('li').allTextContents(), ['First item', 'Second item']);
  assert.equal(await markdown.locator('script').count(), 0);
  assert.equal(await markdown.getByRole('link', {name: 'Safe link'}).getAttribute('target'), '_blank');
  await ask('Wait for cancellation');
  await panel.locator('#cancel').click();
  await panel.waitForFunction(() => document.querySelector('#notice').textContent.includes('Stopped'));
  assert.equal(await panel.locator('.turn.assistant .result-actions').count(), 2,
    'cancelled output is never presented as an applicable result');
  await panel.locator('#prompt-toggle').click();
  await panel.locator('#prompt-action').selectOption('email_reply');
  const defaultPrompt = await panel.locator('#system-prompt').inputValue();
  const customPrompt = 'Keep my email replies brief. Return only the reply body.';
  await panel.locator('#system-prompt').fill(customPrompt);
  await panel.reload();
  await panel.locator('#prompt-toggle').click();
  await panel.locator('#prompt-action').selectOption('email_reply');
  assert.equal(await panel.locator('#system-prompt').inputValue(), customPrompt);
  await panel.locator('#prompt-default').click();
  assert.equal(await panel.locator('#system-prompt').inputValue(), defaultPrompt);
  assert.deepEqual(unexpectedNetwork, [], 'tests never contact Hugging Face or a real mailbox');
  assert.deepEqual(errors, []);
  console.log('Gmail/chat integration passed: capture, follow-up, guarded apply, reply, prompts, cancellation.');
} finally {
  await context?.close();
  await rm(temporary, {recursive: true, force: true});
}
