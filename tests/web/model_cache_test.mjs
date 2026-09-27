import test from 'node:test';
import assert from 'node:assert/strict';
import {deleteCachedModel, isModelCached, listCachedModels, loadModel, resolveModelSource} from '../../web/model-cache.mjs';
import {unsignedHeapIndices} from '../../web/wasm-glue.mjs';

test('large-memory glue fixes direct and pthread heap indices without changing arithmetic shifts', () => {
    const source = 'HEAP32[ptr >> 2] = 1; (growMemViews(), HEAPU64)[addr >> 3] = 2n; const signed = value >> 3;';
    const fixed = unsignedHeapIndices(source);
    assert.equal(fixed, 'HEAP32[ptr >>> 2] = 1; (growMemViews(), HEAPU64)[addr >>> 3] = 2n; const signed = value >> 3;');
    assert.equal(unsignedHeapIndices(fixed), fixed);
});

test('model IDs resolve latest Hub revisions and retain an offline pinned mapping', async () => {
    const cache = new Map();
    const originalFetch = globalThis.fetch, originalCaches = globalThis.caches;
    globalThis.caches = {open: async () => ({
        match: async key => cache.get(String(key))?.clone(),
        put: async (key, response) => cache.set(String(key), response.clone())
    })};
    let revision = 'a'.repeat(40);
    let requests = 0;
    globalThis.fetch = async url => {
        requests++;
        assert.equal(String(url), 'https://huggingface.co/api/models/google/test/revision/main');
        return new Response(JSON.stringify({sha: revision}));
    };
    try {
        assert.equal(await resolveModelSource('google/test'),
            `https://huggingface.co/google/test/resolve/${revision}/config.json`);
        revision = 'b'.repeat(40);
        assert.equal(await resolveModelSource('google/test'),
            `https://huggingface.co/google/test/resolve/${revision}/config.json`);
        globalThis.fetch = async () => { throw new Error('offline'); };
        assert.equal(await resolveModelSource('google/test', {cacheOnly: true}),
            `https://huggingface.co/google/test/resolve/${revision}/config.json`);
        assert.equal(requests, 2);
        await assert.rejects(resolveModelSource('missing/model', {cacheOnly: true}), /no resolved revision/);
    } finally {
        globalThis.fetch = originalFetch;
        if (originalCaches === undefined) delete globalThis.caches;
        else globalThis.caches = originalCaches;
    }
});

test('cache inventory exposes partial model files and selectively deletes one model', async () => {
    const cache = new Map();
    const originalCaches = globalThis.caches;
    const revision = 'c'.repeat(40);
    const modelId = 'openai/whisper-tiny';
    const source = `https://huggingface.co/${modelId}/resolve/${revision}/config.json`;
    const config = JSON.stringify({model_type: 'whisper', architectures: ['WhisperForConditionalGeneration']});
    const store = async (key, body, headers = {}) => cache.set(key, new Response(body, {headers}));
    await store(`https://kidi.invalid/__model_refs__/${encodeURIComponent(modelId)}`,
        JSON.stringify({modelId, revision, url: source}));
    await store(source, config);
    await store(new URL('tokenizer.json', source).href, '{}', {'X-Kidi-Size': '2'});
    globalThis.caches = {open: async () => ({
        match: async key => cache.get(typeof key === 'string' ? key : key.url)?.clone(),
        keys: async () => [...cache.keys()].map(url => ({url})),
        delete: async key => cache.delete(typeof key === 'string' ? key : key.url)
    })};
    try {
        const models = await listCachedModels();
        assert.equal(models.length, 1);
        assert.equal(models[0].name, modelId);
        assert.equal(models[0].kind, 'Speech');
        assert.equal(models[0].complete, false);
        assert.deepEqual(models[0].files,
            [{name: 'config.json', bytes: config.length}, {name: 'tokenizer.json', bytes: 2}]);
        assert.equal(await deleteCachedModel(models[0].id), true);
        assert.equal((await listCachedModels()).length, 0);
        assert.equal(cache.size, 0);
    } finally {
        if (originalCaches === undefined) delete globalThis.caches;
        else globalThis.caches = originalCaches;
    }
});

test('Hub ranges normalize once in owned memory and preserve the upstream cache and checkpoint tensors', async () => {
    const source = `https://huggingface.co/google/test/resolve/${'a'.repeat(40)}/config.json`;
    const width = 8 * 1024 * 1024;
    const headerBytes = 2048;
    const header = {
        'model.language_model.test.weight': {dtype: 'U8', shape: [1, width], data_offsets: [0, width]},
        'lm_head.weight': {dtype: 'U8', shape: [1, 8], data_offsets: [width, width + 8]},
        'model.language_model.test_eight_bit.weight': {dtype: 'I8', shape: [1, 8], data_offsets: [width + 8, width + 16]},
        'model.language_model.test_float.weight': {dtype: 'BF16', shape: [1, 4], data_offsets: [width + 16, width + 24]}
    };
    const upstream = new Uint8Array(8 + headerBytes + width + 24);
    new DataView(upstream.buffer).setBigUint64(0, BigInt(headerBytes), true);
    upstream.fill(32, 8, 8 + headerBytes);
    upstream.set(new TextEncoder().encode(JSON.stringify(header)), 8);
    upstream.fill(0x12, 8 + headerBytes);
    const config = {model_type: 'gemma4', text_config: {enable_moe_block: false}, quantization_config: {
        quant_method: 'gemma', quantize_embeddings: true, num_bits: 4, modules_to_not_convert: [],
        module_quant_configs: {'^lm_head$': {num_bits: 2}, test_eight_bit: {num_bits: 8}}
    }};
    const cache = new Map();
    const requests = [];
    const originalFetch = globalThis.fetch, originalCaches = globalThis.caches;
    globalThis.caches = {open: async () => ({match: async key => cache.get(String(key))?.clone(),
        keys: async () => [...cache.keys()].map(url => ({url})),
        put: async (key, response) => cache.set(String(key), response.clone()),
        delete: async key => cache.delete(String(key))})};
    globalThis.fetch = async (url, options) => {
        requests.push(String(url));
        if (String(url).includes('model.safetensors')) {
            const [start, requestedEnd] = options.headers.Range.slice(6).split('-').map(Number);
            const end = Math.min(requestedEnd, upstream.length - 1);
            return new Response(upstream.slice(start, end + 1), {status: 206,
                headers: {'Content-Range': `bytes ${start}-${end}/${upstream.length}`}});
        }
        return new Response(JSON.stringify(String(url).endsWith('config.json') && !String(url).endsWith('tokenizer_config.json') ? config : {}));
    };
    const runtime = () => {
        const files = new Map();
        const heap = new Uint8Array(upstream.length + 4096);
        return {files, HEAPU8: heap, _malloc: () => 4096, FS: {
            mkdir: () => {}, createDataFile: (directory, name) => files.set(`${directory}/${name}`, {stream_ops: {}}),
            lookupPath: name => ({node: files.get(name)}), writeFile: (name, data) => files.set(name, data)
        }};
    };
    try {
        assert.equal(await isModelCached(source), false);
        await assert.rejects(loadModel(runtime(), source, () => {}, {cacheOnly: true}), /cache incomplete/);
        assert.equal(requests.length, 0);
        const first = runtime();
        const metrics = await loadModel(first, source, () => {});
        assert.equal(metrics.cachedBytes, 0);
        const mapped = first.files.get('/model/model.safetensors');
        assert.equal(mapped.contents.buffer, first.HEAPU8.buffer);
        assert.equal(mapped.usedBytes, upstream.length);
        assert.equal(mapped.stream_ops.mmap(null, upstream.length, 0, 1).allocated, false);
        const data = mapped.contents.subarray(8 + headerBytes);
        assert.equal(data[0], 0x12 ^ 0x88);
        assert.equal(data[width - 1], 0x12 ^ 0x88);
        assert.equal(data[width], 0x12 ^ 0xaa);
        assert.equal(data[width + 8], 0x12);
        assert.deepEqual(data.subarray(width + 16), upstream.subarray(upstream.length - 8));
        const normalized = JSON.parse(new TextDecoder().decode(mapped.contents.subarray(8, 8 + headerBytes)));
        assert.equal(normalized['model.language_model.test_eight_bit.weight'].dtype, 'U8');
        assert.equal(normalized['model.language_model.test_float.weight'].dtype, 'BF16');
        assert.deepEqual(Object.keys(normalized), Object.keys(header));
        const descriptor = JSON.parse(first.files.get('/model/model.yaml'));
        assert.equal(descriptor.model.packed_weights_signed, true);
        assert.deepEqual(descriptor.decode, {maximum_new_tokens: 1024, context_size: 9216});
        const rangeKey = [...cache.keys()].find(key => key.endsWith('kidi_range=0-8388607'));
        assert.deepEqual(new Uint8Array(await cache.get(rangeKey).clone().arrayBuffer()), upstream.subarray(0, width));
        const count = requests.length;
        assert.equal(await isModelCached(source), true);
        const lastRange = [...cache.keys()].find(key => key.includes(`kidi_range=${width}-`));
        const savedRange = cache.get(lastRange);
        cache.delete(lastRange);
        assert.equal(await isModelCached(source), false);
        await assert.rejects(loadModel(runtime(), source, () => {}, {cacheOnly: true}), /cache incomplete/);
        cache.set(lastRange, savedRange);
        const templateKey = new URL('chat_template.jinja', source).href;
        const savedTemplate = cache.get(templateKey);
        cache.delete(templateKey);
        assert.equal(await isModelCached(source), false);
        cache.set(templateKey, savedTemplate);
        assert.equal(await isModelCached(source), true);
        assert.equal(requests.length, count);
        const second = runtime();
        const reloaded = await loadModel(second, source, () => {}, {cacheOnly: true});
        assert.equal(requests.length, count);
        assert.equal(reloaded.downloadedBytes, 0);
        assert.deepEqual(second.files.get('/model/model.safetensors').contents, mapped.contents);
        const corrupted = new Uint8Array(await cache.get(rangeKey).clone().arrayBuffer());
        corrupted[4096] ^= 1;
        cache.set(rangeKey, new Response(corrupted, {headers: cache.get(rangeKey).headers}));
        await assert.rejects(loadModel(runtime(), source, () => {}, {cacheOnly: true}), /cache incomplete or damaged/);
        assert.equal(requests.length, count);
        assert.equal(await isModelCached(source), false);
        await loadModel(runtime(), source, () => {});
        assert.equal(requests.length, count + 1);
        await assert.rejects(loadModel(runtime(), source.replace('a'.repeat(40), 'main'), () => {}), /commit SHA/);
        const unavailable = runtime();
        unavailable._malloc = () => 0;
        await assert.rejects(loadModel(unavailable, source, () => {}), /Not enough WebAssembly memory/);
        cache.clear();
        const validFetch = globalThis.fetch;
        globalThis.fetch = (url, options) => String(url).includes('model.safetensors')
            ? Promise.resolve(new Response(upstream)) : validFetch(url, options);
        await assert.rejects(loadModel(runtime(), source, () => {}), /Invalid byte-range response/);
        globalThis.caches = {open: async () => { throw new Error('Storage unavailable'); }};
        assert.equal(await isModelCached(source), false);
    } finally {
        globalThis.fetch = originalFetch;
        if (originalCaches === undefined) delete globalThis.caches;
        else globalThis.caches = originalCaches;
    }
});