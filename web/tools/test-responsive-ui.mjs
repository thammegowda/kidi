import assert from 'node:assert/strict';
import {devices} from 'playwright';

async function withinViewport(page, selector, {width, height}) {
    const box = await page.locator(selector).boundingBox();
    assert.ok(box && box.width > 0 && box.height > 0, `${selector} must be visible`);
    assert.ok(box.x >= 0 && box.x + box.width <= width + 1,
        `${selector} must fit horizontally in ${width}px: ${JSON.stringify(box)}`);
    assert.ok(box.y >= 0 && box.y + box.height <= height + 1,
        `${selector} must fit vertically in ${height}px: ${JSON.stringify(box)}`);
    return box;
}

export async function testResponsiveUi(browser, url) {
    const errors = [];
    const context = await browser.newContext({...devices['Pixel 5'], colorScheme: 'dark'});
    try {
        const page = await context.newPage();
        page.on('pageerror', error => errors.push(error.message));
        await page.goto(url);
        await page.waitForFunction(() => crossOriginIsolated && navigator.serviceWorker.controller);

        for (const viewport of [
            {width: 320, height: 640},
            {width: 360, height: 800},
            {width: 393, height: 851},
            {width: 412, height: 915},
            {width: 800, height: 360},
        ]) {
            await page.setViewportSize(viewport);
            const dimensions = await page.evaluate(() => ({
                width: innerWidth,
                scrollWidth: document.documentElement.scrollWidth,
            }));
            assert.equal(dimensions.width, viewport.width, 'mobile layout must not widen the viewport');
            assert.ok(dimensions.scrollWidth <= viewport.width, `no horizontal overflow at ${viewport.width}px`);
            const header = await withinViewport(page, '.app-header', viewport);
            assert.ok(header.height <= 56, `compact header must be at most 56px, got ${header.height}`);
            assert.ok(await page.locator('.topbar').evaluate(element => element.getBoundingClientRect().height <= 40),
                'conversation bar must stay small');
            assert.equal(await page.locator('#menu').evaluate(element =>
                element === document.querySelector('.app-header').firstElementChild), true,
            'panel toggle must be at the far left of the main header');
            for (const selector of ['#menu', '#open-config', '#open-chat-view', '#open-reader-view',
                '#language-controls > summary']) {
                const box = await withinViewport(page, selector, viewport);
                assert.ok(box.height >= 44, `${selector} must retain a 44px touch target`);
                assert.ok(await page.locator(selector).evaluate(element => {
                    const box = element.getBoundingClientRect();
                    return element.contains(document.elementFromPoint(box.x + box.width / 2, box.y + box.height / 2));
                }), `${selector} must be tappable, not covered`);
            }
            for (const selector of ['#onboarding-load-model', '#onboarding-load-asr', '#onboarding-load-tts']) {
                const box = await page.locator(selector).boundingBox();
                assert.ok(box && box.height >= 32 && box.height <= 48,
                    `${selector} must be a compact action, not a 116px-tall flex item`);
                assert.ok(box.width <= 80, `${selector} must be a compact load action`);
                assert.equal(await page.locator(selector).textContent(), 'Load');
                assert.match(await page.locator(selector).getAttribute('aria-label'), /^Load (LLM|ASR|TTS) model$/);
            }
            const composer = await withinViewport(page, '#composer', viewport);
            const disclaimer = await withinViewport(page, '.disclaimer', viewport);
            await withinViewport(page, '#prompt', viewport);
            assert.ok(disclaimer.y + disclaimer.height <= composer.y,
                'disclaimer must be above the text box');
            assert.ok(viewport.height - composer.y - composer.height <= 16,
                'text box must sit at the bottom, without a disclaimer row below');

            const initiallyOpen = viewport.width > 760;
            assert.equal(await page.locator('#menu').getAttribute('aria-expanded'), String(initiallyOpen));
            await page.locator('#menu').tap();
            assert.equal(await page.locator('#menu').getAttribute('aria-expanded'), String(!initiallyOpen));
            if (!initiallyOpen) await page.locator('#sidebar.open').waitFor({state: 'visible'});
            else assert.equal(await page.locator('#sidebar').isVisible(), false);
            await page.locator('#menu').tap();
            assert.equal(await page.locator('#menu').getAttribute('aria-expanded'), String(initiallyOpen));
            assert.equal(await page.locator('#sidebar').evaluate(element => element.inert), !initiallyOpen);
            if (!initiallyOpen) {
                await page.locator('#menu').tap();
                await page.locator('#chat-search').tap();
                await page.keyboard.press('Escape');
                assert.equal(await page.locator('#menu').getAttribute('aria-expanded'), 'false');
                assert.equal(await page.locator('#menu').evaluate(element => element === document.activeElement), true,
                    'closing navigation returns keyboard focus to its toggle');
            }

            assert.equal(await page.locator('#input-language').isVisible(), false);
            await page.locator('#language-controls > summary').tap();
            await withinViewport(page, '.language-switchers', viewport);
            await page.locator('#input-language').tap();
            await withinViewport(page, '#input-language-menu', viewport);
            await page.keyboard.press('Escape');
            assert.equal(await page.locator('#input-language').isVisible(), false);
            assert.equal(await page.locator('#language-controls > summary').evaluate(element =>
                element === document.activeElement), true, 'Escape returns focus to the language control');
            await page.getByRole('button', {name: 'Settings', exact: true}).tap();
            await page.locator('#wasm-preferences').waitFor({state: 'visible'});
            for (const selector of ['#load-wasm-model', '#load-asr-model', '#load-tts-model'])
                assert.equal(await page.locator(selector).textContent(), 'Load');
            await page.getByRole('button', {name: 'Close configuration'}).tap();
        }

        await page.setViewportSize({width: 360, height: 800});
        await page.locator('#language-controls > summary').tap();
        await page.getByRole('textbox', {name: 'Input language', exact: true}).fill('Kannada');
        await page.keyboard.press('Enter');
        assert.equal(await page.locator('#input-language').inputValue(), 'Kannada');
        await page.getByRole('textbox', {name: 'Output language', exact: true}).fill('English');
        await page.keyboard.press('Enter');
        assert.equal(await page.locator('#output-language').inputValue(), 'English');
        assert.equal(await page.locator('#language-summary').textContent(), 'Kannada → English');
        await page.getByRole('button', {name: 'Reader', exact: true}).tap();
        assert.equal(await page.locator('#input-language').isVisible(), false, 'outside taps close the panel');
        await withinViewport(page, '#open-config', {width: 360, height: 800});
        await page.getByRole('button', {name: 'Chat', exact: true}).tap();
        await page.reload();
        await page.waitForFunction(() => document.getElementById('language-summary')?.textContent === 'Kannada → English');
        assert.equal(await page.locator('#language-summary').textContent(), 'Kannada → English',
            'language choices survive reload');
        await page.locator('#language-controls > summary').focus();
        await page.keyboard.press('Enter');
        assert.equal(await page.locator('#input-language').isVisible(), true, 'keyboard opens the panel');
        await page.keyboard.press('Escape');

        await page.locator('#language-controls > summary').tap();
        await page.locator('#input-language').fill('English');
        await page.keyboard.press('Enter');
        assert.equal(await page.locator('#language-summary').textContent(), 'English',
            'matching explicit selections show the language once');
        await page.locator('#input-language').fill('Auto');
        await page.keyboard.press('Enter');
        assert.equal(await page.locator('#language-summary').textContent(), 'Auto → English');
        await page.keyboard.press('Escape');
        for (const width of [320, 360, 393]) {
            await page.setViewportSize({width, height: 800});
            await withinViewport(page, '#open-config', {width, height: 800});
            assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth),
                `a full language pair must also fit at ${width}px`);
        }
        await page.locator('#language-controls > summary').tap();
        await page.locator('#output-language').fill('Match input');
        await page.keyboard.press('Enter');
        assert.equal(await page.locator('#language-summary').textContent(), 'Auto',
            'automatic matching selections show Auto once');
        await page.keyboard.press('Escape');
        for (const width of [320, 360, 393]) {
            await page.setViewportSize({width, height: 800});
            await withinViewport(page, '#open-config', {width, height: 800});
            assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth),
                `a matching language summary must also fit at ${width}px`);
        }

        await page.evaluate(() => {
            const saved = JSON.parse(localStorage.getItem('kidi-app-v2'));
            const chat = {
                id: 'layout-chat', title: 'Layout example', updatedAt: Date.now(),
                messages: [{
                    id: 'layout-response', role: 'assistant',
                    content: `${'A paragraph from a saved conversation.\n\n'.repeat(12)}The response ends here.`,
                }],
            };
            localStorage.setItem('kidi-app-v2', JSON.stringify({
                ...saved, chats: [chat], active: chat.id, modelSetupDismissed: true,
            }));
        });
        await page.reload();
        await page.getByRole('button', {name: 'Show more', exact: true}).click();
        const lastParagraph = page.locator('[data-message-id="layout-response"] .message-content p').last();
        await lastParagraph.waitFor({state: 'visible'});
        for (const draft of ['A short draft.', 'A longer message draft.\n'.repeat(12)]) {
            await page.locator('#prompt').fill(draft);
            await page.locator('#messages').evaluate(element => { element.scrollTop = element.scrollHeight; });
            const response = await lastParagraph.boundingBox();
            const disclaimer = await withinViewport(page, '.disclaimer', {width: 393, height: 800});
            await withinViewport(page, '#composer', {width: 393, height: 800});
            assert.ok(response && response.y + response.height <= disclaimer.y,
                `model output must stay above the disclaimer: ${JSON.stringify({draftLength: draft.length, response, disclaimer})}`);
        }

        for (const viewport of [{width: 768, height: 800}, {width: 1024, height: 800}, {width: 1440, height: 900}]) {
            const desktop = await browser.newContext({viewport, colorScheme: 'light'});
            try {
                const desktopPage = await desktop.newPage();
                desktopPage.on('pageerror', error => errors.push(error.message));
                await desktopPage.goto(url);
                await desktopPage.waitForFunction(() => crossOriginIsolated && navigator.serviceWorker.controller);
                await withinViewport(desktopPage, '#open-config', viewport);
                assert.ok(await desktopPage.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
                assert.equal(await desktopPage.locator('#menu').getAttribute('aria-expanded'), 'true');
                await desktopPage.locator('#menu').click();
                assert.equal(await desktopPage.locator('#sidebar').isVisible(), false);
                assert.equal(await desktopPage.locator('#menu').getAttribute('aria-expanded'), 'false');
                await desktopPage.locator('#menu').click();
                assert.equal(await desktopPage.locator('#sidebar').isVisible(), true);
                await desktopPage.locator('#language-controls > summary').click();
                await withinViewport(desktopPage, '.language-switchers', viewport);
                await desktopPage.evaluate(() => {
                    document.getElementById('backend-activity').textContent = 'Loading TTS (text-to-speech) · 100%';
                    document.getElementById('backend-memory').textContent = 'GPU 3.8 GB · RAM 3.5 GB';
                    document.getElementById('language-summary').textContent = 'Chinese (Traditional) → English';
                    document.getElementById('conversation-title').textContent = 'A conversation with a very long title';
                    document.getElementById('runtime-label').textContent = 'Preparing speech recognition and text-to-speech';
                });
                await withinViewport(desktopPage, '#open-config', viewport);
                assert.ok(await desktopPage.evaluate(() => document.documentElement.scrollWidth <= innerWidth),
                    'busy telemetry and long language names must not push Settings off-screen');
            } finally {
                await desktop.close();
            }
        }
        assert.deepEqual(errors, [], 'responsive UI must not produce JavaScript errors');
    } finally {
        await context.close();
    }
}
