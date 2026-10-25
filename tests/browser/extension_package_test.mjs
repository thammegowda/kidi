import test from 'node:test';
import assert from 'node:assert/strict';
import {access, mkdtemp, mkdir, readFile, rm, writeFile} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';

import {packageExtensions} from '../../web/tools/package-extension.mjs';

test('packages Chromium and Firefox extensions without hosted-page isolation scripts', async t => {
    const directory = await mkdtemp(join(tmpdir(), 'kidi-extension-test-'));
    t.after(() => rm(directory, {recursive: true, force: true}));
    const runtimes = {};
    for (const name of ['threads', 'gpu', 'single']) {
        const build = join(directory, name);
        runtimes[name] = build;
        await mkdir(build);
        await writeFile(join(build, 'kidi.mjs'), 'export default () => HEAPU8[index >> 0];');
        await writeFile(join(build, 'kidi.wasm'), new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0]));
    }
    const packages = await packageExtensions(join(directory, 'extensions'), runtimes);
    for (const target of ['chromium', 'firefox']) {
        const output = packages[target].directory;
        const manifest = JSON.parse(await readFile(join(output, 'manifest.json'), 'utf8'));
        assert.equal(manifest.manifest_version, 3);
        assert.match(manifest.content_security_policy.extension_pages, /wasm-unsafe-eval/);
        assert.doesNotMatch(manifest.content_security_policy.extension_pages, /(?:^|\s)'unsafe-eval'/);
        for (const file of ['background.mjs', 'editor.html', 'editor.mjs', 'editor.css',
            'progress.mjs', 'selection-bridge.js', 'gmail-bridge.js', 'article-bridge.js',
            'read-aloud.js', 'audio-playback.js', 'speech-controller.js', 'playback-worklet.js',
            'libs/readability/Readability.js', 'libs/readability/LICENSE', 'single/kidi.mjs', 'single/kidi.wasm'])
            await access(join(output, file));
        assert.deepEqual(manifest.optional_host_permissions, [
            'https://mail.google.com/*', 'https://*/*', 'http://*/*',
        ]);
        assert.equal(manifest.host_permissions, undefined);
        await assert.rejects(access(join(output, 'coi-serviceworker.js')));
        assert.doesNotMatch(await readFile(join(output, 'index.html'), 'utf8'), /coi-serviceworker/);
    }
    const chromium = JSON.parse(
        await readFile(join(packages.chromium.directory, 'manifest.json'), 'utf8'));
    assert.equal(chromium.side_panel.default_path, 'editor.html');
    assert.equal(chromium.cross_origin_embedder_policy.value, 'require-corp');
    const firefox = JSON.parse(
        await readFile(join(packages.firefox.directory, 'manifest.json'), 'utf8'));
    assert.equal(firefox.sidebar_action.default_panel, 'editor.html');
    assert.equal(firefox.browser_specific_settings.gecko.id, 'kidi@gowda.ai');
});
