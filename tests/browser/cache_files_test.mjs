import test from 'node:test';
import assert from 'node:assert/strict';
import {listCachedFiles, deleteCachedFile, deleteCachedFiles, clearModelCache} from '../../web/hosts/wasm/model-cache.mjs';

test('file cache inventory and deletion are scoped to model storage', async () => {
    const previous = globalThis.caches;
    const entries = new Map([
        ['https://example.test/models/__kidi_chunks__/abc', new Response('chunk')],
        ['https://huggingface.co/org/model/resolve/rev/model.safetensors?kidi_range=0-7',
            new Response('12345678', {headers: {'X-Kidi-Size': '16'}})],
        ['https://huggingface.co/org/model/resolve/rev/model.safetensors?kidi_range=8-15',
            new Response('abcdefgh', {headers: {'X-Kidi-Size': '16'}})],
    ]);
    const opened = [];
    const deleted = [];
    globalThis.caches = {
        open: async name => {
            opened.push(name);
            return {
                keys: async () => [...entries.keys()].map(url => ({url})),
                match: async request => entries.get(request.url)?.clone(),
                delete: async id => entries.delete(id),
            };
        },
        delete: async name => { deleted.push(name); entries.clear(); return true; },
    };
    try {
        const files = await listCachedFiles();
        assert.deepEqual(files.map(file => file.bytes), [16, 5]);
        assert.equal(files[0].id, 'https://huggingface.co/org/model/resolve/rev/model.safetensors');
        await deleteCachedFile(files[0].id);
        assert.equal(entries.size, 1);
        await assert.rejects(deleteCachedFile(files[0].id), /no longer exists/);
        await clearModelCache();
        assert.equal(entries.size, 0);
        assert.ok(opened.every(name => name === 'kidi-model-v1'));
        assert.deepEqual(deleted, ['kidi-model-v1']);
    } finally { globalThis.caches = previous; }
});

test('recursive selection removes all selected files and ranges but preserves siblings', async () => {
    const previous = globalThis.caches;
    const root = 'https://example.test/models/';
    const entries = new Map([
        [root + 'tts/model.safetensors?kidi_range=0-3', new Response('abcd')],
        [root + 'tts/model.safetensors?kidi_range=4-7', new Response('efgh')],
        [root + 'tts/config/tokenizer.json', new Response('{}')],
        [root + 'other/model.safetensors', new Response('keep')],
    ]);
    globalThis.caches = {open: async () => ({
        keys: async () => [...entries.keys()].map(url => ({url})),
        match: async request => entries.get(request.url)?.clone(),
        delete: async id => entries.delete(id),
    })};
    try {
        const files = await listCachedFiles();
        const selected = files.filter(file => file.id.startsWith(root + 'tts/')).map(file => file.id);
        await assert.rejects(deleteCachedFiles([...selected, root + 'missing']), /no longer exists/);
        assert.equal(entries.size, 4);
        await deleteCachedFiles(selected);
        assert.deepEqual([...entries.keys()], [root + 'other/model.safetensors']);
    } finally { globalThis.caches = previous; }
});

test('package files combine all manifest chunks for one delete', async () => {
    const previous = globalThis.caches;
    const root = 'https://example.test/models/tts/';
    const entries = new Map([
        [root + 'manifest.json', new Response(JSON.stringify({
            version: 1,
            files: [{name: 'model.safetensors', chunks: [{sha256: 'aaa'}, {sha256: 'bbb'}]}],
        }))],
        [root + '__kidi_chunks__/aaa', new Response('first')],
        [root + '__kidi_chunks__/bbb', new Response('second')],
    ]);
    globalThis.caches = {open: async () => ({
        keys: async () => [...entries.keys()].map(url => ({url})),
        match: async request => entries.get(request.url)?.clone(),
        delete: async id => entries.delete(id),
    })};
    try {
        const files = await listCachedFiles();
        const weights = files.find(file => file.id === root + 'model.safetensors');
        assert.equal(weights.bytes, 11);
        await deleteCachedFile(weights.id);
        assert.deepEqual([...entries.keys()], [root + 'manifest.json']);
    } finally { globalThis.caches = previous; }
});
