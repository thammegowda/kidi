import test from 'node:test';
import assert from 'node:assert/strict';
import {access, readFile} from 'node:fs/promises';

test('shared entry assets and audio worklets resolve within repository subpaths', async () => {
    const html = await readFile(new URL('../../web/ui/index.html', import.meta.url), 'utf8');
    const css = await readFile(new URL('../../web/ui/style.css', import.meta.url), 'utf8');
    const app = await readFile(new URL('../../web/ui/app.js', import.meta.url), 'utf8');
    const worker = await readFile(new URL('../../web/hosts/wasm/tts-worker.mjs', import.meta.url), 'utf8');
    const build = await readFile(new URL('../../web/tools/build.mjs', import.meta.url), 'utf8');
    const packager = await readFile(new URL('../../web/tools/package-site.mjs', import.meta.url), 'utf8');
    const desktop = await readFile(new URL('../../desktop/app.cpp', import.meta.url), 'utf8');
    const cmake = await readFile(new URL('../../CMakeLists.txt', import.meta.url), 'utf8');
    const manifest = await readFile(new URL('../../web/ui/assets.txt', import.meta.url), 'utf8');
    const desktopManifest = await readFile(new URL('../../desktop/assets.txt', import.meta.url), 'utf8');
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
    assert.match(worker, /'\.\/threads\/kidi\.mjs'/);
    assert.doesNotMatch(worker, /single\/kidi/);
    assert.doesNotMatch(build, /\['single'/);
    const shared = new Set(manifest.split(/\r?\n/).filter(Boolean));
    for (const match of app.matchAll(/from '\.\/([^']+)'/g))
        if (match[1] !== 'backend.js') assert.ok(shared.has(match[1]), `shared UI manifest omits ${match[1]}`);
    for (const asset of ['speech-highlights.js', 'pause.svg', 'play.svg']) {
        assert.match(manifest, new RegExp(asset.replace('.', '\\.')));
    }
    for (const consumer of [desktop, cmake, packager]) assert.match(consumer, /web\/ui\/assets\.txt/);
    assert.match(packager, /browser-epub\.js/);
    assert.match(cmake, /file\(STRINGS "desktop\/assets\.txt" KIDI_DESKTOP_ASSET_MAPPINGS\)/);
    assert.doesNotMatch(cmake, /web\/libs\/mermaid\/mermaid\.min\.js/);
    for (const line of desktopManifest.trim().split(/\r?\n/)) {
        const [destination, source] = line.split('\t');
        assert.ok(destination.startsWith('/') && source, `invalid desktop asset mapping: ${line}`);
        await access(new URL(`../../${source}`, import.meta.url));
    }
});
