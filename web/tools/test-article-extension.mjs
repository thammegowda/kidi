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
const temporary = await mkdtemp(join(tmpdir(), 'kidi-article-test-'));
const extension = join(temporary, 'extension');
let browser;
try {
  await cp(source, extension, {recursive: true});
  const manifest = JSON.parse(await readFile(join(extension, 'manifest.json'), 'utf8'));
  // This synthetic package alone receives the fixture origin; shipped packages use activeTab.
  manifest.host_permissions = ['https://example.org/*'];
  await writeFile(join(extension, 'manifest.json'), JSON.stringify(manifest));
  await cp(join(root, 'tests/browser/fixtures/extension-backend.mjs'), join(extension, 'backend.js'));
  const fixture = await readFile(join(root, 'tests/browser/fixtures/article.html'), 'utf8');
  browser = await chromium.launchPersistentContext(join(temporary, 'profile'), {
    channel, headless: true, colorScheme: 'dark', viewport: {width: 390, height: 800},
    args: [`--disable-extensions-except=${extension}`, `--load-extension=${extension}`],
  });
  const unexpectedNetwork = [];
  await browser.route('https://**/*', route => {
    if (new URL(route.request().url()).hostname === 'example.org')
      return route.fulfill({status: 200, contentType: 'text/html', body: fixture});
    unexpectedNetwork.push(route.request().url());
    return route.abort();
  });
  const worker = browser.serviceWorkers()[0] || await browser.waitForEvent('serviceworker');
  const extensionId = new URL(worker.url()).host;
  const article = await browser.newPage();
  await article.goto('https://example.org/article');
  const tabId = await worker.evaluate(async () => (await chrome.tabs.query({url: 'https://example.org/*'}))[0].id);
  const panel = await browser.newPage();
  const errors = [];
  panel.on('pageerror', error => errors.push(error.message));
  await panel.goto(`chrome-extension://${extensionId}/editor.html?tab=${tabId}`);
  await panel.waitForFunction(() => document.querySelector('#target-language').options.length > 50);
  const requests = () => panel.evaluate(async () => (await import('./backend.js')).backend.requests);
  const capture = () => panel.evaluate(tabId =>
    chrome.runtime.sendMessage({type: 'kidi.page.capture', tabId}), tabId);
  const answers = count => panel.waitForFunction(expected =>
    document.querySelectorAll('.turn.assistant .result-actions').length === expected, count);

  const captured = await capture();
  assert.equal(captured.ok, true);
  assert.equal(captured.task.kind, 'article');
  assert.equal(captured.task.language, 'en');
  assert.match(captured.task.text, /carefully controlled experiment/);
  assert.match(captured.task.text, /second stage measured/);
  assert.doesNotMatch(captured.task.text, /PRIVATE_MARKER/);
  assert.equal(captured.task.operation, null);
  assert.equal(captured.task.truncated, false);
  await article.locator('#second-paragraph').evaluate(element => {
    const range = document.createRange();
    range.selectNodeContents(element);
    getSelection().removeAllRanges();
    getSelection().addRange(range);
  });
  const selected = await capture();
  assert.equal(selected.task.sourceKind, 'selection');
  assert.match(selected.task.text, /second stage measured/);
  assert.doesNotMatch(selected.task.text, /carefully controlled experiment/);
  await article.evaluate(() => getSelection().removeAllRanges());

  await panel.getByRole('button', {name: 'Read article', exact: true}).click();
  await panel.locator('#speech-dialog').waitFor({state: 'visible'});
  assert.equal((await requests()).length, 0, 'reading does not load the chat or voice model before approval');
  await panel.locator('#speech-decline').click();
  assert.equal((await requests()).length, 0);
  await panel.getByRole('button', {name: 'Read article', exact: true}).click();
  await panel.locator('#speech-dialog').getByRole('button', {name: 'Read aloud', exact: true}).click();
  await panel.waitForFunction(() => document.querySelector('#speech-status').textContent.includes('50%'));
  await panel.locator('#speech-stop').click();
  assert.equal((await requests()).filter(request => request.type === 'runtime.load').length, 0,
    'original narration does not require LLM inference');
  assert.ok((await requests()).some(request => request.type === 'cancel' && request.payload.scope === 'speech'));
  await panel.getByRole('button', {name: 'Read article', exact: true}).click();
  await panel.waitForFunction(() => !document.querySelector('#speech-pause').disabled);
  await panel.locator('#speech-pause').click();
  assert.equal(await panel.locator('#speech-pause').textContent(), 'Resume');
  await panel.locator('#speech-pause').click();
  assert.equal(await panel.locator('#speech-pause').textContent(), 'Pause');
  await panel.locator('#speech-stop').click();
  await panel.locator('#speech-player').waitFor({state: 'hidden'});

  await panel.getByRole('button', {name: 'Summarize page', exact: true}).click();
  await answers(1);
  const summary = (await requests()).find(request => request.type === 'document.run');
  assert.equal(summary.payload.mode, 'summarize');
  assert.match(summary.payload.text, /carefully controlled experiment/);
  assert.doesNotMatch(summary.payload.text, /PRIVATE_MARKER/);
  assert.match(await panel.locator('#context-followup').textContent(), /uses this summary/);
  await panel.locator('#message').fill('What are its limitations?');
  await panel.locator('#send').click();
  await answers(2);
  const followup = (await requests()).filter(request => request.type === 'editor.run').at(-1);
  assert.match(followup.payload.text, /local-first approach/);
  assert.doesNotMatch(followup.payload.text, /carefully controlled experiment/);

  await panel.getByRole('button', {name: 'Translate page', exact: true}).click();
  await panel.locator('#translation-tools').waitFor({state: 'visible'});
  const before = (await requests()).filter(request => request.type === 'document.run').length;
  await panel.locator('#target-language').selectOption('fr');
  assert.equal((await requests()).filter(request => request.type === 'document.run').length, before,
    'choosing translation language does not start generation');
  await panel.locator('#send').click();
  await answers(3);
  const translation = (await requests()).filter(request => request.type === 'document.run').at(-1);
  assert.equal(translation.payload.mode, 'translate');
  assert.equal(translation.payload.target_language, 'French');
  assert.match(translation.payload.text, /carefully controlled experiment/);
  assert.deepEqual(
    await panel.locator('.turn.assistant .turn-text').last().locator('.markdown-body p').allTextContents(),
    ['Premier paragraphe traduit.', 'Deuxième paragraphe traduit.'],
  );
  assert.equal(await panel.getByRole('button', {name: 'Apply', exact: true}).count(), 0);
  await panel.getByRole('button', {name: 'Read aloud', exact: true}).last().click();
  await panel.waitForFunction(async () => (await import('./backend.js')).backend.requests
    .some(request => request.type === 'tts.start' && request.payload.options.language === 'French'));
  await panel.locator('#speech-stop').click();
  if (screenshot) await panel.screenshot({path: resolve(screenshot)});

  await panel.locator('#message').fill('Wait for cancellation');
  await panel.getByRole('button', {name: 'Summarize page', exact: true}).click();
  await panel.locator('#cancel').click();
  await panel.waitForFunction(() => document.querySelector('#notice').textContent.includes('Stopped'));
  assert.match(await panel.locator('.turn.assistant .turn-status').last().textContent(), /Stopped/);
  const sourceText = await article.locator('#first-paragraph').textContent();
  assert.match(sourceText, /carefully controlled experiment/, 'no result replaces webpage text');

  await article.evaluate(() => {
    document.documentElement.lang = 'und';
    const element = document.createElement('p');
    element.id = 'large-selection';
    element.textContent = '😀漢字 '.repeat(9000);
    document.body.append(element);
    const range = document.createRange();
    range.selectNodeContents(element);
    getSelection().removeAllRanges();
    getSelection().addRange(range);
  });
  const large = await capture();
  assert.equal(large.ok, true);
  assert.equal(large.task.truncated, true);
  assert.ok(new TextEncoder().encode(large.task.text).length <= 64 * 1024);
  assert.ok(!large.task.text.includes('\uFFFD'), 'byte limits never split Unicode codepoints');
  assert.match(large.task.warning, /attached text only/);
  await panel.locator('#new-chat').click();
  await panel.getByRole('button', {name: 'Read article', exact: true}).click();
  await panel.locator('#speech-dialog').waitFor({state: 'visible'});
  assert.equal(await panel.locator('#speech-language').inputValue(), '', 'unknown language does not become English');
  assert.match(await panel.locator('#context-warning').textContent(), /first 64 KiB/);
  await panel.locator('#speech-decline').click();
  const persisted = await panel.evaluate(() => JSON.stringify({...localStorage}));
  assert.doesNotMatch(persisted, /carefully controlled experiment|first approach|Readable source/);

  await article.evaluate(() => { document.body.replaceChildren(); getSelection().removeAllRanges(); });
  const empty = await capture();
  assert.equal(empty.ok, false);
  assert.match(empty.error, /No readable article/);
  await panel.reload();
  await panel.waitForFunction(() => document.querySelector('#target-language').options.length > 50);
  assert.equal(await panel.locator('#target-language').inputValue(), 'fr', 'translation preference is restored');
  assert.deepEqual(errors, []);
  assert.deepEqual(unexpectedNetwork, [], 'all executable assets and fixture inference remain local');
  console.log('Article extension passed: extraction, Unicode bounds, summary/follow-up, translation, consent, narration and stop.');
} finally {
  if (browser) await browser.close();
  await rm(temporary, {recursive: true, force: true});
}
