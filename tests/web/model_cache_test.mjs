import test from 'node:test';
import assert from 'node:assert/strict';
import {deleteCachedModel, interleavePacked, isModelCached, listCachedModels, loadModel, resolveModelSource,
    storeConvertedSpeech, weightLayout} from '../../web/model-cache.mjs';
import {unsignedHeapIndices} from '../../web/wasm-glue.mjs';

test('built wasm32 runtime addresses memory above 2 GiB within its 4 GiB limit', {skip: !process.env.KIDI_TEST_WASM}, async () => {
    const {pathToFileURL} = await import('node:url');
    const {resolve} = await import('node:path');
    const {default: create} = await import(pathToFileURL(resolve(process.env.KIDI_TEST_WASM)).href);
    const module = await create();
    const base = Number(module._malloc(2 ** 31 + 64));
    assert.ok(base > 0);
    try {
        const high = base + 2 ** 31;
        module.HEAPU8[high] = 137;
        module.HEAPU8[high + 31] = 251;
        assert.equal(module.HEAPU8[high], 137);
        assert.equal(module.HEAPU8[high + 31], 251);
        assert.ok(module.HEAPU8.byteLength > 2 ** 31);
        const result = JSON.parse(module.ccall('kidi_transcribe', 'string',
            ['pointer', 'number', 'string', 'number'], [high, 0, 'en', 1]));
        assert.match(result.error, /Speech must contain/);
        const memory = JSON.parse(module.ccall('kidi_memory_stats', 'string', [], []));
        // Emscripten keeps wasm32 memory one 64 KiB page below 4 GiB.
        assert.equal(memory.heap_limit_bytes, 4 * 1024 ** 3 - 65536);
        assert.ok(memory.malloc_in_use_bytes >= 2 ** 31);
    } finally { module._free(base); }
});

test('image attachments persist outside chat text and stage bounded temporary request files', async () => {
    const {storeImage, readImage, stageImages, removeUnusedImages} = await import('../../web/images.mjs');
    const previous = globalThis.caches;
    const saved = new Map(), files = new Map();
    globalThis.caches = {open: async () => ({
        put: async (key, response) => saved.set(key, response),
        match: async key => saved.get(key)?.clone(),
        keys: async () => [...saved.keys()].map(url => ({url})),
        delete: async key => saved.delete(typeof key === 'string' ? key : key.url)
    })};
    const module = {FS: {mkdirTree: () => {}, writeFile: (path, bytes) => files.set(path, bytes),
        unlink: path => files.delete(path)}};
    try {
        const image = await storeImage(new Blob(['image bytes'], {type: 'image/jpeg'}), 'photo.jpg');
        assert.equal(await (await readImage(image)).text(), 'image bytes');
        const staged = await stageImages(module, [{role: 'user', content: 'Describe', images: [image]}]);
        assert.deepEqual(staged.messages[0].images, ['/images/0.jpg']);
        assert.deepEqual(files.get('/images/0.jpg'), new TextEncoder().encode('image bytes'));
        assert.equal(files.size, 1);
        staged.dispose();
        assert.equal(files.size, 0);
        const original = new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10, 0, 255, 128, 42]);
        const png = await storeImage(new Blob([original], {type: 'image/png'}), 'original.png');
        const pngRequest = await stageImages(module, [{role: 'user', content: '', images: [png]}]);
        assert.deepEqual(files.get('/images/0.png'), original);
        pngRequest.dispose();
        assert.equal(files.size, 0);
        await assert.rejects(stageImages(module, [{role: 'assistant', content: '', images: [image]}]), /user messages/);
        await assert.rejects(stageImages(module, [{role: 'user', content: '', images: Array(9).fill(image)}]), /at most eight/);
        assert.equal(files.size, 0);
        await removeUnusedImages([image]);
        assert.equal(saved.size, 1);
        await removeUnusedImages([]);
        await assert.rejects(stageImages(module, [{role: 'user', content: '', images: [image]}]), /Image unavailable/);
        await assert.rejects(readImage({id: '../../model/model.yaml'}), /Invalid image/);
    } finally { globalThis.caches = previous; }
});
test('lowered wasm32 glue fixes direct and pthread heap indices without changing arithmetic shifts', () => {
    const source = 'HEAP32[ptr >> 2] = 1; (growMemViews(), HEAPU64)[addr >> 3] = 2n; const signed = value >> 3;';
    const fixed = unsignedHeapIndices(source);
    assert.equal(fixed, 'HEAP32[ptr >>> 2] = 1; (growMemViews(), HEAPU64)[addr >>> 3] = 2n; const signed = value >> 3;');
    assert.equal(unsignedHeapIndices(fixed), fixed);
});

test('Gemma heap layout drops audio, keeps per-layer embeddings outside, and joins gate/up projections', () => {
    const tensor = (dtype, shape, start, end) => ({dtype, shape, data_offsets: [start, end]});
    const header = {
        'model.audio_tower.weight': tensor('F32', [4], 0, 16),
        'model.language_model.embed_tokens_per_layer.embedding_quantized': tensor('U8', [4, 8], 16, 48),
        'model.language_model.layers.0.mlp.up_proj.weight': tensor('U8', [2, 3], 48, 54),
        'model.language_model.norm.weight': tensor('F32', [3], 54, 66),
        'model.language_model.layers.0.mlp.gate_proj.weight': tensor('U8', [2, 3], 66, 72),
    };
    const ranges = Object.entries(header).map(([name, info]) =>
        ({name, start: info.data_offsets[0], end: info.data_offsets[1], mask: 0}));
    const layout = weightLayout(header, ranges, true);
    assert.equal(layout.header['model.audio_tower.weight'], undefined);
    assert.equal(layout.header['model.language_model.embed_tokens_per_layer.embedding_quantized'], undefined);
    assert.deepEqual(layout.external.map(table => [table.name, table.rowBytes, table.shards[0].length]),
        [['model.language_model.embed_tokens_per_layer.embedding_quantized', 8, 32]]);
    const gate = layout.header['model.language_model.layers.0.mlp.gate_proj.weight'].data_offsets;
    const up = layout.header['model.language_model.layers.0.mlp.up_proj.weight'].data_offsets;
    assert.equal(gate[0] % 64, 0);
    assert.equal(up[0], gate[1]);
    assert.equal(layout.header['model.language_model.norm.weight'].data_offsets[0] % 64, 0);
    assert.equal((8 + layout.headerBytes) % 64, 0);
    assert.equal(layout.totalBytes, 8 + layout.headerBytes + layout.dataBytes);
});

test('packed weights interleave like the C++ WebGPU path', () => {
    // Reference rule from runtime/webgpu/operators.cpp: value `slot` moves to byte slot % 4 at bit bits * (slot / 4).
    const reference = (word, bits) => {
        let output = 0;
        for (let slot = 0; slot < 32 / bits; slot++)
            output |= ((word >>> (slot * bits)) & ((1 << bits) - 1)) << (8 * (slot % 4) + bits * Math.floor(slot / 4));
        return output >>> 0;
    };
    const words = new Uint32Array([0, 0xffffffff, 0x12345678, 0x89abcdef, 0x0f1e2d3c, 0xdeadbeef]);
    for (const bits of [2, 4]) {
        const bytes = new Uint8Array(words.slice().buffer);
        interleavePacked(bytes, bits);
        assert.deepEqual([...new Uint32Array(bytes.buffer)], [...words].map(word => reference(word, bits)));
    }
    const eight = new Uint8Array(words.slice().buffer);
    interleavePacked(eight, 8);
    assert.deepEqual([...new Uint32Array(eight.buffer)], [...words]);
});

test('WebGPU layout sends decoder projections to fused GPU buffers and keeps the rest in the heap', () => {
    const tensor = (dtype, shape, start, end) => ({dtype, shape, data_offsets: [start, end]});
    const header = {
        'model.language_model.layers.0.mlp.gate_proj.weight': tensor('U8', [4, 2], 0, 8),
        'model.language_model.layers.0.mlp.up_proj.weight': tensor('U8', [4, 2], 8, 16),
        'model.language_model.layers.0.self_attn.q_proj.weight': tensor('U8', [4, 2], 16, 24),
        'model.language_model.layers.0.mlp.gate_proj.weight_scale': tensor('F32', [4, 1], 24, 40),
        'model.language_model.embed_tokens.embedding_quantized': tensor('U8', [4, 2], 40, 48),
    };
    const ranges = Object.entries(header).map(([name, info]) => ({name, start: info.data_offsets[0],
        end: info.data_offsets[1], mask: 0, bits: name.endsWith('weight_scale') ? 0 : 4}));
    const layout = weightLayout(header, ranges, true, true);
    assert.deepEqual(layout.device.map(entry => [entry.name, entry.shape, entry.bytes, entry.bits]), [
        ['model.language_model.layers.0.mlp.gate_up_proj.weight', [8, 2], 16, 4],
        ['model.language_model.layers.0.self_attn.q_proj.weight', [4, 2], 8, 4]]);
    assert.deepEqual(Object.keys(layout.header).sort(), ['model.language_model.embed_tokens.embedding_quantized',
        'model.language_model.layers.0.mlp.gate_proj.weight_scale']);
    assert.equal(weightLayout(header, ranges.map(range => ({...range})), true).device.length, 0);
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

test('Hub ranges are cached sign-flipped once and reload into owned memory without rewriting', async () => {
    const source = `https://huggingface.co/google/test/resolve/${'a'.repeat(40)}/config.json`;
    const width = 8 * 1024 * 1024;
    const headerBytes = 2048;
    const header = {
        'model.language_model.test.weight': {dtype: 'U8', shape: [1, width], data_offsets: [0, width]},
        'lm_head.weight': {dtype: 'U8', shape: [1, 8], data_offsets: [width, width + 8]},
        'model.language_model.test_eight_bit.weight': {dtype: 'I8', shape: [1, 8], data_offsets: [width + 8, width + 16]},
        'model.language_model.test_float.weight': {dtype: 'BF16', shape: [1, 4], data_offsets: [width + 16, width + 24]},
        'model.language_model.embed_tokens_per_layer.embedding_quantized':
            {dtype: 'U8', shape: [2, 4], data_offsets: [width + 24, width + 32]},
        'model.audio_tower.test.weight': {dtype: 'F32', shape: [4], data_offsets: [width + 32, width + 48]}
    };
    const upstream = new Uint8Array(8 + headerBytes + width + 48);
    new DataView(upstream.buffer).setBigUint64(0, BigInt(headerBytes), true);
    upstream.fill(32, 8, 8 + headerBytes);
    upstream.set(new TextEncoder().encode(JSON.stringify(header)), 8);
    upstream.fill(0x12, 8 + headerBytes);
    const config = {model_type: 'gemma4', text_config: {enable_moe_block: false},
        vision_config: {hidden_size: 768}, image_token_id: 42, quantization_config: {
        quant_method: 'gemma', quantize_embeddings: true, num_bits: 4, modules_to_not_convert: [],
        module_quant_configs: {'^lm_head$': {num_bits: 2}, test_eight_bit: {num_bits: 8}}
    }};
    const cache = new Map();
    const requests = [];
    let failPuts = false;
    const originalFetch = globalThis.fetch, originalCaches = globalThis.caches;
    globalThis.caches = {open: async () => ({match: async key => cache.get(String(key))?.clone(),
        keys: async () => [...cache.keys()].map(url => ({url})),
        put: async (key, response) => {
            if (failPuts) throw new Error('QuotaExceededError');
            cache.set(String(key), response.clone());
        },
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
        assert.equal(mapped.usedBytes, metrics.heapWeightBytes);
        assert.equal(mapped.stream_ops.mmap(null, mapped.usedBytes, 0, 1).allocated, false);
        const contents = mapped.contents;
        const compactHeader = Number(new DataView(contents.buffer, contents.byteOffset, 8).getBigUint64(0, true));
        assert.equal((contents.byteOffset + 8 + compactHeader) % 64, 0);
        const normalized = JSON.parse(new TextDecoder().decode(contents.subarray(8, 8 + compactHeader)));
        const tensor = name => contents.subarray(8 + compactHeader + normalized[name].data_offsets[0],
            8 + compactHeader + normalized[name].data_offsets[1]);
        assert.equal(tensor('model.language_model.test.weight')[0], 0x12 ^ 0x88);
        assert.equal(tensor('model.language_model.test.weight').at(-1), 0x12 ^ 0x88);
        assert.equal(tensor('lm_head.weight')[0], 0x12 ^ 0xaa);
        assert.equal(tensor('model.language_model.test_eight_bit.weight')[0], 0x12);
        assert.deepEqual(tensor('model.language_model.test_float.weight'),
            upstream.subarray(8 + headerBytes + width + 16, 8 + headerBytes + width + 24));
        assert.equal(normalized['model.language_model.test_eight_bit.weight'].dtype, 'U8');
        assert.equal(normalized['model.language_model.test_float.weight'].dtype, 'BF16');
        assert.deepEqual(Object.keys(normalized).sort(), ['lm_head.weight', 'model.language_model.test.weight',
            'model.language_model.test_eight_bit.weight', 'model.language_model.test_float.weight']);
        assert.equal(metrics.externalWeightBytes, 8);
        const rows = 16, destination = 32;
        new Int32Array(first.HEAPU8.buffer, rows, 3).set([1, 0, 2]);
        first.kidiExternal.gather('model.language_model.embed_tokens_per_layer.embedding_quantized', rows, 2,
            destination, 4);
        assert.deepEqual([...first.HEAPU8.subarray(destination, destination + 8)], Array(8).fill(0x12 ^ 0x88));
        assert.throws(() => first.kidiExternal.gather('model.language_model.embed_tokens_per_layer.embedding_quantized',
            rows, 3, destination, 4), /outside/);
        const descriptor = JSON.parse(first.files.get('/model/model.yaml'));
        assert.deepEqual(descriptor.external_tensors, [{name: 'model.language_model.embed_tokens_per_layer.embedding_quantized',
            dtype: 'u8', shape: [2, 4]}]);
        assert.deepEqual(JSON.parse(new TextDecoder().decode(first.files.get('/model/config.json'))), config);
        assert.equal(descriptor.model.packed_weights_signed, true);
        assert.deepEqual(descriptor.decode, {maximum_new_tokens: 1024, context_size: 9216});
        const rangeKey = [...cache.keys()].find(key => key.endsWith('kidi_range=0-8388607'));
        const cachedFirst = new Uint8Array(await cache.get(rangeKey).clone().arrayBuffer());
        assert.equal(cache.get(rangeKey).headers.get('X-Kidi-Layout'), 'gemma-signed-v1');
        assert.deepEqual(cachedFirst.subarray(0, 8 + headerBytes), upstream.subarray(0, 8 + headerBytes));
        assert.ok(cachedFirst.subarray(8 + headerBytes).every(byte => byte === (0x12 ^ 0x88)));
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
        assert.deepEqual(second.files.get('/model/config.json'), first.files.get('/model/config.json'));
        // An entry cached by an earlier version holds upstream bytes: it is sign-flipped once and rewritten.
        const layoutHeaders = cache.get(rangeKey).headers;
        cache.set(rangeKey, new Response(upstream.slice(0, width), {headers: {'X-Kidi-Size': layoutHeaders.get('X-Kidi-Size')}}));
        const migrated = runtime();
        await loadModel(migrated, source, () => {}, {cacheOnly: true});
        assert.deepEqual(migrated.files.get('/model/model.safetensors').contents, mapped.contents);
        assert.equal(cache.get(rangeKey).headers.get('X-Kidi-Layout'), 'gemma-signed-v1');
        assert.equal(requests.length, count);
        // A rewrite that fails (for example over quota) leaves the old entry and still loads from the cache.
        const signedEntry = cache.get(rangeKey);
        cache.set(rangeKey, new Response(upstream.slice(0, width), {headers: {'X-Kidi-Size': layoutHeaders.get('X-Kidi-Size')}}));
        failPuts = true;
        const unconverted = runtime();
        await loadModel(unconverted, source, () => {}, {cacheOnly: true});
        failPuts = false;
        assert.deepEqual(unconverted.files.get('/model/model.safetensors').contents, mapped.contents);
        assert.equal(cache.get(rangeKey).headers.get('X-Kidi-Layout'), null);
        cache.set(rangeKey, signedEntry);
        // Contents are trusted; a truncated entry is still rejected.
        const truncated = new Uint8Array(await cache.get(rangeKey).clone().arrayBuffer()).slice(0, -1);
        cache.set(rangeKey, new Response(truncated, {headers: cache.get(rangeKey).headers}));
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
test('Whisper converts whisper.cpp Q8 weights once and reloads the cached INT8 checkpoint', async () => {
    const source = `https://huggingface.co/openai/whisper-small/resolve/${'d'.repeat(40)}/config.json`;
    const config = {model_type: 'whisper', architectures: ['WhisperForConditionalGeneration'], d_model: 768,
        vocab_size: 51865};
    const weights = Uint8Array.from({length: 8 * 1024 * 1024 + 5}, (_, index) => index % 251);
    const converted = Uint8Array.from({length: 8 * 1024 * 1024 + 3}, (_, index) => index % 13);
    const stale = `${new URL('model.safetensors', source).href}?kidi_range=0-8388607`;
    const cache = new Map([[stale, new Response('old')]]);
    const requests = [];
    const originalFetch = globalThis.fetch, originalCaches = globalThis.caches;
    globalThis.caches = {open: async () => ({match: async key => cache.get(String(key))?.clone(),
        keys: async () => [...cache.keys()].map(url => ({url})),
        put: async (key, response) => cache.set(String(key), response.clone()),
        delete: async key => cache.delete(typeof key === 'string' ? key : key.url)})};
    globalThis.fetch = async (url, options) => {
        requests.push(String(url));
        if (String(url).endsWith('/ggml-small-q8_0.bin')) {
            assert.match(String(url), /^https:\/\/huggingface\.co\/ggerganov\/whisper\.cpp\/resolve\/[a-f0-9]{40}\//);
            const [start, requestedEnd] = options.headers.Range.slice(6).split('-').map(Number);
            const end = Math.min(requestedEnd, weights.length - 1);
            return new Response(weights.slice(start, end + 1), {status: 206,
                headers: {'Content-Range': `bytes ${start}-${end}/${weights.length}`}});
        }
        assert.ok(!String(url).includes('safetensors'));
        return new Response(JSON.stringify(String(url).endsWith('/config.json') ? config : {}));
    };
    // A virtual file system with Emscripten's calls the loader uses.
    const runtime = () => {
        const files = new Map(), directories = new Set(['/']);
        const HEAPU8 = new Uint8Array(32 * 1024 * 1024);
        return {files, HEAPU8, _malloc: () => 4096, _free: () => {}, FS: {
            mkdir: path => directories.add(path), rmdir: path => directories.delete(path),
            writeFile: (name, data) => files.set(name, data), unlink: name => files.delete(name),
            analyzePath: path => ({exists: files.has(path) || directories.has(path)}),
            readdir: path => [...files.keys()].filter(name => name.startsWith(`${path}/`)).map(name => name.slice(path.length + 1)),
            createDataFile: (directory, name) => files.set(`${directory}/${name}`, {stream_ops: {}}),
            lookupPath: name => ({node: files.get(name)}),
        }};
    };
    try {
        const first = runtime();
        const metrics = await loadModel(first, source, () => {});
        assert.deepEqual(first.files.get('/model/ggml-model.bin'), weights);
        assert.deepEqual([...first.files.keys()].sort(), ['/model/config.json', '/model/generation_config.json',
            '/model/ggml-model.bin', '/model/preprocessor_config.json', '/model/tokenizer.json']);
        assert.equal(metrics.precision, 'INT8 from GGML Q8_0');
        assert.ok(!cache.has(stale));
        assert.ok(cache.has(`${new URL('ggml-model.bin', source).href}?kidi_range=8388608-8388612`));
        assert.equal(await isModelCached(source), true);
        // kidi_load_asr converts the GGML weights and writes the INT8 checkpoint beside them.
        const directory = '/model/ggml-model.bin.kidi-int8-v1';
        first.FS.mkdir(directory);
        first.files.set(`${directory}/model.safetensors`, {contents: converted, usedBytes: converted.length});
        first.files.set(`${directory}/quantization.json`, {});
        await storeConvertedSpeech(first, metrics);
        assert.ok(![...first.files.keys()].some(name => name.includes('ggml-model.bin')));
        assert.ok(![...cache.keys()].some(key => key.includes('ggml-model.bin')));
        assert.ok([...cache.keys()].some(key => key.startsWith(metrics.convertedUrl)));
        assert.equal(await isModelCached(source), true);
        const count = requests.length;
        const second = runtime();
        const reloaded = await loadModel(second, source, () => {}, {cacheOnly: true});
        assert.equal(reloaded.downloadedBytes, 0);
        assert.equal(reloaded.convertedUrl, undefined);
        assert.equal(requests.length, count);
        assert.ok(!second.files.has('/model/ggml-model.bin'));
        const mapped = second.files.get('/model/model.safetensors');
        assert.deepEqual(mapped.contents, converted);
        assert.equal(mapped.usedBytes, converted.length);
        config.d_model = 640;
        cache.clear();
        await assert.rejects(loadModel(runtime(), source, () => {}), /Tiny, Base, or Small/);
    } finally {
        globalThis.fetch = originalFetch;
        if (originalCaches === undefined) delete globalThis.caches;
        else globalThis.caches = originalCaches;
    }
});
