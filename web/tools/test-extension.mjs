import assert from 'node:assert/strict';
import {mkdtemp, rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {resolve, join} from 'node:path';

import {chromium} from 'playwright';

const extension = resolve(process.argv[2] || new URL('../../build-extension/chromium', import.meta.url).pathname);
const screenshot = process.argv[3] ? resolve(process.argv[3]) : null;
const channel = process.env.KIDI_BROWSER_CHANNEL || 'chromium';
const profile = await mkdtemp(join(tmpdir(), 'kidi-extension-profile-'));
const context = await chromium.launchPersistentContext(profile, {
    channel,
    colorScheme: 'dark',
    headless: true,
    viewport: {width: 390, height: 800},
    args: [
        `--disable-extensions-except=${extension}`,
        `--load-extension=${extension}`,
    ],
});
try {
    let worker = context.serviceWorkers()[0];
    worker ||= await context.waitForEvent('serviceworker', {timeout: 15000});
    const extensionId = new URL(worker.url()).host;
    assert.ok(extensionId, 'extension service worker has an ID');
    const page = await context.newPage();
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    page.on('console', message => {
        if (message.type() === 'error') errors.push(message.text());
    });
    await page.goto(`chrome-extension://${extensionId}/editor.html`);
    await page.locator('.app-bar').waitFor({state: 'visible'});
    await page.waitForFunction(() => document.querySelector('#system-prompt')?.value.length > 20);
    assert.equal(await page.locator('.app-bar > strong').count(), 0, 'browser chrome owns the only Kidi AI title');
    const header = await page.locator('.app-bar').boundingBox();
    assert.ok(header && header.height <= 32, `session controls are compact: ${JSON.stringify(header)}`);
    assert.equal(await page.locator('#task').count(), 0, 'chat replaces the action dropdown');
    for (const action of ['Summarize page', 'Translate page', 'Read article'])
        await page.getByRole('button', {name: action, exact: true}).waitFor();
    assert.equal(await page.locator('#prompt-action option').count(), 13);
    await page.locator('#prompt-toggle').click();
    await page.locator('#prompt-editor').waitFor({state: 'visible'});
    const defaultPrompt = await page.locator('#system-prompt').inputValue();
    const configuredSystem = await page.evaluate(async () =>
        (await (await import('./backend.js')).backend.status()).components.chat.configured_system_prompt);
    assert.equal(defaultPrompt, configuredSystem, 'extension chat uses the shared prompts.system default');
    assert.match(defaultPrompt, /untrusted reference material/);
    const customPrompt = 'Use the company writing style. Return only the revised text.';
    await page.locator('#system-prompt').fill(customPrompt);
    assert.match(await page.locator('#prompt-state').textContent(), /saved automatically/);
    await page.reload();
    await page.waitForFunction(() => document.querySelector('#system-prompt')?.value.length > 20);
    await page.locator('#prompt-toggle').click();
    assert.equal(await page.locator('#system-prompt').inputValue(), customPrompt);
    await page.locator('#prompt-default').click();
    assert.equal(await page.locator('#system-prompt').inputValue(), defaultPrompt);
    assert.equal(await page.locator('#prompt-state').textContent(), 'Configured default');
    if (!screenshot) await page.locator('#prompt-toggle').click();
    const loadButton = await page.locator('#load-model').boundingBox();
    assert.ok(loadButton && loadButton.width <= 80 && loadButton.height <= 34,
        `model load action is compact: ${JSON.stringify(loadButton)}`);
    const conversation = await page.locator('#conversation').boundingBox();
    assert.ok(conversation && conversation.height >= 150, 'conversation receives the primary panel space');
    const routes = await page.evaluate(async () => {
        const {createGpuWorker} = await import('./runtime-client.mjs');
        const results = [];
        for (const type of ['editor', 'document']) {
        const worker = createGpuWorker('chat');
        try {
            results.push(await new Promise((resolve, reject) => {
                const timer = setTimeout(() => reject(new Error('GPU editor route timed out')), 10000);
                worker.onmessage = ({data}) => { clearTimeout(timer); resolve(data); };
                worker.onerror = event => { clearTimeout(timer); reject(new Error(event.message)); };
                worker.postMessage({
                    type, request: {task: 'email_reply', mode: 'summarize', text: 'Can we meet?'}, maximumTokens: 8,
                });
            }));
        } finally { worker.terminate(); }
        }
        return results;
    });
    for (const routed of routes) {
        assert.equal(routed.type, 'error');
        assert.match(routed.error, /Load the model session before inference/);
        assert.doesNotMatch(routed.error, /Unknown service method/);
    }
    const runtime = await page.evaluate(() => ({
        isolated: crossOriginIsolated,
        sharedArrayBuffer: typeof SharedArrayBuffer,
        protocol: location.protocol,
    }));
    assert.deepEqual(runtime, {
        isolated: true,
        sharedArrayBuffer: 'function',
        protocol: 'chrome-extension:',
    });
    assert.deepEqual(errors, []);
    if (screenshot) await page.screenshot({path: screenshot, fullPage: true});
    console.log(`${channel} extension smoke test passed: ${extensionId}`);
} finally {
    await context.close();
    await rm(profile, {recursive: true, force: true});
}
