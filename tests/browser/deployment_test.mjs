import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';

test('shared entry assets and audio worklets resolve within repository subpaths', async () => {
    const html = await readFile(new URL('../../web/ui/index.html', import.meta.url), 'utf8');
    const css = await readFile(new URL('../../web/ui/style.css', import.meta.url), 'utf8');
    const app = await readFile(new URL('../../web/ui/app.js', import.meta.url), 'utf8');
    const worker = await readFile(new URL('../../web/hosts/wasm/tts-worker.mjs', import.meta.url), 'utf8');
    const assets = [...html.matchAll(/(?:src|href)="([^"]+)"/g)].map(match => match[1]);
    assets.push(...[...css.matchAll(/url\("([^"]+)"\)/g)].map(match => match[1]));
    const worklets = [...app.matchAll(/audioWorklet\.addModule\(new URL\('([^']+)', import\.meta\.url\)\)/g)];
    assert.equal(worklets.length, 2);
    assets.push(...worklets.map(match => match[1]));
    for (const asset of assets) {
        assert.ok(new URL(asset, 'https://example.test/kidi/').pathname.startsWith('/kidi/'), asset);
    }
    assert.match(app, /appendReadAloud\(actions, source\)/);
    assert.doesNotMatch(app, /startTts\(content\.textContent/);
    assert.match(worker, /useGpu \? '\.\/gpu\/kidi\.mjs'/);
});
