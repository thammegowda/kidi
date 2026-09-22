import test from 'node:test';
import assert from 'node:assert/strict';
import {loadModel} from '../../web/model-cache.mjs';
import {unsignedHeapIndices} from '../../web/wasm-glue.mjs';

test('large-memory glue fixes direct and pthread heap indices without changing arithmetic shifts', () => {
    const source = 'HEAP32[ptr >> 2] = 1; (growMemViews(), HEAPU64)[addr >> 3] = 2n; const signed = value >> 3;';
    const fixed = unsignedHeapIndices(source);
    assert.equal(fixed, 'HEAP32[ptr >>> 2] = 1; (growMemViews(), HEAPU64)[addr >>> 3] = 2n; const signed = value >> 3;');
    assert.equal(unsignedHeapIndices(fixed), fixed);
});

test('Hub ranges normalize once in owned memory and preserve the upstream cache and modality tensors', async () => {
    const source = `https://huggingface.co/google/test/resolve/${'a'.repeat(40)}/config.json`;
    const width = 8 * 1024 * 1024;
    const headerBytes = 2048;
    const header = {
        'model.language_model.test.weight': {dtype: 'U8', shape: [1, width], data_offsets: [0, width]},
        'lm_head.weight': {dtype: 'U8', shape: [1, 8], data_offsets: [width, width + 8]},
        'model.audio_tower.test.weight': {dtype: 'I8', shape: [1, 8], data_offsets: [width + 8, width + 16]},
        'model.vision_tower.test.weight': {dtype: 'BF16', shape: [1, 4], data_offsets: [width + 16, width + 24]}
    };
    const upstream = new Uint8Array(8 + headerBytes + width + 24);
    new DataView(upstream.buffer).setBigUint64(0, BigInt(headerBytes), true);
    upstream.fill(32, 8, 8 + headerBytes);
    upstream.set(new TextEncoder().encode(JSON.stringify(header)), 8);
    upstream.fill(0x12, 8 + headerBytes);
    const config = {model_type: 'gemma4', text_config: {enable_moe_block: false}, quantization_config: {
        quant_method: 'gemma', quantize_embeddings: true, num_bits: 4, modules_to_not_convert: [],
        module_quant_configs: {'^lm_head$': {num_bits: 2}, audio_tower: {num_bits: 8}}
    }};
    const cache = new Map();
    const requests = [];
    const originalFetch = globalThis.fetch, originalCaches = globalThis.caches;
    globalThis.caches = {open: async () => ({match: async key => cache.get(String(key))?.clone(),
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
        assert.equal(normalized['model.audio_tower.test.weight'].dtype, 'U8');
        assert.equal(normalized['model.vision_tower.test.weight'].dtype, 'BF16');
        assert.deepEqual(Object.keys(normalized), Object.keys(header));
        const descriptor = JSON.parse(first.files.get('/model/model.yaml'));
        assert.equal(descriptor.model.packed_weights_signed, true);
        assert.deepEqual(descriptor.decode, {maximum_new_tokens: 1024, context_size: 9216});
        const rangeKey = [...cache.keys()].find(key => key.endsWith('kidi_range=0-8388607'));
        assert.deepEqual(new Uint8Array(await cache.get(rangeKey).clone().arrayBuffer()), upstream.subarray(0, width));
        const count = requests.length;
        const second = runtime();
        const reloaded = await loadModel(second, source, () => {});
        assert.equal(requests.length, count);
        assert.equal(reloaded.downloadedBytes, 0);
        assert.deepEqual(second.files.get('/model/model.safetensors').contents, mapped.contents);
        const corrupted = new Uint8Array(await cache.get(rangeKey).clone().arrayBuffer());
        corrupted[4096] ^= 1;
        cache.set(rangeKey, new Response(corrupted, {headers: cache.get(rangeKey).headers}));
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
    } finally {
        globalThis.fetch = originalFetch;
        if (originalCaches === undefined) delete globalThis.caches;
        else globalThis.caches = originalCaches;
    }
});