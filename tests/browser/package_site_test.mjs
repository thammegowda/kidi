import test from 'node:test';
import assert from 'node:assert/strict';
import {access, mkdtemp, mkdir, readFile, readdir, rm, writeFile} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {pathToFileURL} from 'node:url';
import {parse} from '../../web/libs/acorn/acorn.mjs';
import {full} from '../../web/libs/acorn-walk/walk.mjs';
import {packageSite} from '../../web/tools/package-site.mjs';

async function verifyModuleAssets(directory) {
    for (const entry of await readdir(directory, {withFileTypes: true})) {
        const file = join(directory, entry.name);
        if (entry.isDirectory()) {
            if (entry.name !== 'libs') await verifyModuleAssets(file);
            continue;
        }
        if (!/\.(mjs|js)$/.test(entry.name)) continue;
        const tree = parse(await readFile(file, 'utf8'), {ecmaVersion: 'latest', sourceType: 'module'});
        const assets = new Set();
        full(tree, node => {
            if (['ImportDeclaration', 'ExportNamedDeclaration', 'ExportAllDeclaration', 'ImportExpression']
                .includes(node.type) && node.source?.type === 'Literal')
                assets.add(node.source.value);
            if (node.type === 'NewExpression' && node.callee.name === 'URL' &&
                node.arguments[0]?.type === 'Literal' &&
                node.arguments[1]?.object?.type === 'MetaProperty')
                assets.add(node.arguments[0].value);
        });
        for (const asset of assets) {
            if (!asset.startsWith('.')) continue;
            await assert.doesNotReject(access(new URL(asset, pathToFileURL(file))), `${file} requires ${asset}`);
        }
    }
}

test('site packaging includes transitive modules and relocates worker assets', async t => {
    const directory = await mkdtemp(join(tmpdir(), 'kidi-site-test-'));
    t.after(() => rm(directory, {recursive: true, force: true}));
    const runtimes = {};
    for (const name of ['threads', 'gpu']) {
        const build = join(directory, name);
        runtimes[name] = build;
        await mkdir(build);
        await writeFile(join(build, 'kidi.mjs'), 'export default () => HEAPU8[index >> 0];');
        await writeFile(join(build, 'kidi.wasm'), new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0]));
    }
    const destination = join(directory, 'site');
    await packageSite(destination, runtimes);
    await verifyModuleAssets(destination);
    const liteapp = await import(pathToFileURL(join(destination, 'liteapp/liteapp.mjs')));
    for (const name of ['RequestChannel', 'MessageTransport', 'WebSocketTransport',
        'createVariantClient', 'EventClient', 'ServiceRouter', 'ExtensionPortTransport', 'ServiceBroker'])
        assert.equal(typeof liteapp[name], 'function', `packaged LiteApp export: ${name}`);
    for (const name of ['threads', 'gpu']) {
        assert.match(await readFile(join(destination, name, 'kidi.mjs'), 'utf8'), /index >>> 0/);
        assert.deepEqual(await readFile(join(destination, name, 'kidi.wasm')),
            await readFile(join(runtimes[name], 'kidi.wasm')));
    }
    await access(join(destination, '.nojekyll'));
    assert.match(await readFile(join(destination, 'index.html'), 'utf8'), /src="\.\/coi-serviceworker.js"/);
    await t.test('chat images become image_url data URL parts', async () => {
        const {storeImage, readImage, chatMessages, hasImages, heapString, removeUnusedImages} =
            await import(pathToFileURL(join(destination, 'images.js')));
        const previous = globalThis.caches;
        const saved = new Map();
        globalThis.caches = {open: async () => ({
            put: async (key, response) => saved.set(key, response),
            match: async key => saved.get(key)?.clone(),
            keys: async () => [...saved.keys()].map(url => ({url})),
            delete: async key => saved.delete(typeof key === 'string' ? key : key.url)
        })};
        try {
            const image = await storeImage(new Blob(['image bytes'], {type: 'image/jpeg'}), 'photo.jpg');
            assert.equal(await (await readImage(image)).text(), 'image bytes');
            const original = new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10, 0, 255, 128, 42]);
            const png = await storeImage(new Blob([original], {type: 'image/png'}), 'original.png');
            const messages = await chatMessages([
                {role: 'system', content: 'Be brief'},
                {role: 'user', content: 'Describe', images: [image, png]}]);
            assert.deepEqual(messages[0], {role: 'system', content: 'Be brief'});
            const pngUrl = `data:image/png;base64,${Buffer.from(original).toString('base64')}`;
            assert.deepEqual(messages[1].content, [
                {type: 'image_url', image_url: {url: `data:image/jpeg;base64,${btoa('image bytes')}`}},
                {type: 'image_url', image_url: {url: pngUrl}},
                {type: 'text', text: 'Describe'}]);
            assert.equal(hasImages(messages), true);
            assert.equal(hasImages(messages.slice(0, 1)), false);
            await assert.rejects(chatMessages([{role: 'assistant', content: '', images: [image]}]), /user messages/);
            const tooMany = [{role: 'user', content: '', images: Array(9).fill(image)}];
            await assert.rejects(chatMessages(tooMany), /at most eight/);
            const heap = new Uint8Array(64);
            const pointer = heapString({HEAPU8: heap, _malloc: () => 8}, 'ಕಿ');
            assert.equal(new TextDecoder().decode(heap.subarray(pointer, heap.indexOf(0, pointer))), 'ಕಿ');
            assert.throws(() => heapString({HEAPU8: heap, _malloc: () => 0}, 'x'), /Cannot allocate/);
            await removeUnusedImages([image]);
            assert.equal(saved.size, 1);
            await removeUnusedImages([]);
            await assert.rejects(chatMessages([{role: 'user', content: '', images: [image]}]), /Image unavailable/);
            await assert.rejects(readImage({id: '../../model/model.yaml'}), /Invalid image/);
        } finally { globalThis.caches = previous; }
    });
    await t.test('a missing transitive re-export is detected', async () => {
        await rm(join(destination, 'liteapp/host/event-client.mjs'));
        await assert.rejects(verifyModuleAssets(destination), /event-client\.mjs/);
    });
});
