import assert from 'node:assert/strict';
import {mkdtemp, rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {resolve, join} from 'node:path';

import {chromium} from 'playwright';

const extension = resolve(process.argv[2] || new URL('../../build-extension/chromium', import.meta.url).pathname);
const screenshot = process.argv[3] ? resolve(process.argv[3]) : null;
const profile = await mkdtemp(join(tmpdir(), 'kidi-extension-profile-'));
const context = await chromium.launchPersistentContext(profile, {
    channel: 'chromium',
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
    await page.locator('header strong').waitFor({state: 'visible'});
    await page.waitForFunction(() => document.querySelector('#system-prompt')?.value.length > 20);
    assert.equal(await page.locator('header strong').textContent(), 'Kidi AI');
    const header = await page.locator('.app-bar').boundingBox();
    assert.ok(header && header.height <= 42, `application header is compact: ${JSON.stringify(header)}`);
    assert.equal(await page.locator('#task option').count(), 7);
    await page.locator('#prompt-toggle').click();
    await page.locator('#prompt-editor').waitFor({state: 'visible'});
    const defaultPrompt = await page.locator('#system-prompt').inputValue();
    assert.match(defaultPrompt, /untrusted data/);
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
    const source = await page.locator('#source').boundingBox();
    assert.ok(source && source.height >= 170, 'source editor receives the primary panel space');
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
    console.log(`Chromium extension smoke test passed: ${extensionId}`);
} finally {
    await context.close();
    await rm(profile, {recursive: true, force: true});
}
