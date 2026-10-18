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
        'createVariantClient', 'EventClient', 'ServiceRouter'])
        assert.equal(typeof liteapp[name], 'function', `packaged LiteApp export: ${name}`);
    for (const name of ['threads', 'gpu']) {
        assert.match(await readFile(join(destination, name, 'kidi.mjs'), 'utf8'), /index >>> 0/);
        assert.deepEqual(await readFile(join(destination, name, 'kidi.wasm')),
            await readFile(join(runtimes[name], 'kidi.wasm')));
    }
    await access(join(destination, '.nojekyll'));
    assert.match(await readFile(join(destination, 'index.html'), 'utf8'), /src="\.\/coi-serviceworker.js"/);
    await t.test('a missing transitive re-export is detected', async () => {
        await rm(join(destination, 'liteapp/host/event-client.mjs'));
        await assert.rejects(verifyModuleAssets(destination), /event-client\.mjs/);
    });
});
