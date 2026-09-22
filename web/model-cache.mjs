const CACHE_NAME = 'kidi-model-v1';
const CHUNK_BYTES = 8 * 1024 * 1024;
const hex = bytes => Array.from(new Uint8Array(bytes), byte => byte.toString(16).padStart(2, '0')).join('');
const digest = async data => hex(await crypto.subtle.digest('SHA-256', data));

export async function isModelCached(source) {
    try {
        const url = new URL(source);
        const cache = await caches.open(CACHE_NAME);
        const keys = new Set((await cache.keys()).map(request => request.url));
        if (!keys.has(url.href)) return false;
        if (url.pathname.endsWith('/config.json')) {
            if (url.origin !== 'https://huggingface.co' || url.search || url.hash ||
                !/^\/[^/]+\/[^/]+\/resolve\/[a-f0-9]{40}\/config\.json$/.test(url.pathname)) return false;
            for (const name of ['tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja'])
                if (!keys.has(new URL(name, url).href)) return false;
            const key = new URL('model.safetensors', url);
            key.searchParams.set('kidi_range', `0-${CHUNK_BYTES - 1}`);
            const first = await cache.match(key.href);
            const size = Number(first?.headers.get('X-Kidi-Size'));
            first?.body?.cancel().catch(() => {});
            if (!Number.isSafeInteger(size) || size <= 0 || size >= 2 ** 32) return false;
            for (let start = CHUNK_BYTES; start < size; start += CHUNK_BYTES) {
                key.searchParams.set('kidi_range', `${start}-${Math.min(size, start + CHUNK_BYTES) - 1}`);
                if (!keys.has(key.href)) return false;
            }
            return true;
        }
        const manifest = await (await cache.match(url.href)).json();
        if (manifest.version !== 1 || !Array.isArray(manifest.files) || !/^[a-f0-9]{64}$/.test(manifest.id)) return false;
        for (const name of ['model.yaml', 'model.safetensors', 'tokenizer.json', 'tokenizer_config.json'])
            if (!manifest.files.some(file => file.name === name)) return false;
        return manifest.files.every(file => Array.isArray(file.chunks) && file.chunks.length &&
            file.chunks.every(chunk => keys.has(new URL(`./__kidi_chunks__/${chunk.sha256}`, url).href)));
    } catch { return false; }
}

function allocateWeights(module, size) {
    if (!Number.isSafeInteger(size) || size <= 0 || size >= 2 ** 32)
        throw new Error('Checkpoint exceeds the Wasm32 address space');
    const pointer = Number(module._malloc(size));
    if (!Number.isSafeInteger(pointer) || pointer <= 0 || pointer + size > module.HEAPU8.byteLength)
        throw new Error('Not enough WebAssembly memory for model weights');
    return pointer;
}

function mountWeights(module, pointer, size) {
    module.FS.createDataFile('/model', 'model.safetensors', new Uint8Array(), true, false);
    const node = module.FS.lookupPath('/model/model.safetensors').node;
    node.usedBytes = size;
    Object.defineProperty(node, 'contents', {get: () => new Uint8Array(module.HEAPU8.buffer, pointer, size)});
    node.stream_ops = {...node.stream_ops, mmap: (_stream, length, position, protection) => {
        if (!Number.isSafeInteger(position) || !Number.isSafeInteger(length) || position < 0 || length < 0 ||
            length > size - position || (protection & 2)) throw new Error('Invalid model mapping');
        return {ptr: pointer + position, allocated: false};
    }};
}

function normalizationPlan(config, header, dataBytes) {
    const quantization = config.quantization_config;
    if (config.model_type !== 'gemma4' || quantization?.quant_method !== 'gemma' ||
        !quantization.quantize_embeddings || !config.text_config || config.text_config.enable_moe_block)
        throw new Error('Expected a dense Gemma 4 mobile-QAT checkpoint');
    const widths = {BOOL: 1, U8: 1, I8: 1, U16: 2, I16: 2, F16: 2, BF16: 2,
        U32: 4, I32: 4, F32: 4, U64: 8, I64: 8, F64: 8};
    const ranges = [];
    for (const [name, tensor] of Object.entries(header)) {
        if (name === '__metadata__') continue;
        if (!tensor || !Array.isArray(tensor.shape) || !Array.isArray(tensor.data_offsets) ||
            tensor.data_offsets.length !== 2 || !widths[tensor.dtype]) throw new Error(`Invalid tensor: ${name}`);
        const [start, end] = tensor.data_offsets;
        const elements = tensor.shape.reduce((count, dimension) => {
            if (!Number.isSafeInteger(dimension) || dimension < 0) throw new Error(`Invalid shape: ${name}`);
            return count * dimension;
        }, 1);
        if (![start, end, elements, elements * widths[tensor.dtype]].every(Number.isSafeInteger) ||
            start < 0 || end < start || end > dataBytes || end - start !== elements * widths[tensor.dtype] ||
            start % widths[tensor.dtype]) throw new Error(`Invalid tensor range: ${name}`);
        let mask = 0;
        if (/\.(weight|embedding_quantized)$/.test(name) && ['U8', 'I8'].includes(tensor.dtype)) {
            const module = name.replace(/\.(weight|embedding_quantized)$/, '');
            if (quantization.modules_to_not_convert.some(excluded => module.includes(excluded)))
                throw new Error(`Unexpected quantized excluded tensor: ${name}`);
            const override = Object.entries(quantization.module_quant_configs)
                .find(([pattern]) => new RegExp(pattern).test(module));
            const bits = override ? override[1].num_bits : quantization.num_bits;
            if (![2, 4, 8].includes(bits) || tensor.dtype !== (bits === 8 ? 'I8' : 'U8'))
                throw new Error(`Unsupported packed tensor: ${name}`);
            mask = bits === 2 ? 0xaa : bits === 4 ? 0x88 : 0;
            tensor.dtype = 'U8';
        }
        ranges.push({start, end, mask});
    }
    ranges.sort((left, right) => left.start - right.start);
    let end = 0;
    for (const range of ranges) {
        if (range.start !== end) throw new Error('Overlapping or incomplete checkpoint tensors');
        end = range.end;
    }
    if (end !== dataBytes) throw new Error('Checkpoint tensor sizes do not cover the payload');
    return ranges.filter(range => range.mask);
}

async function loadHubModel(module, source, progress, cache, warning, cacheOnly) {
    const configUrl = new URL(source);
    if (configUrl.origin !== 'https://huggingface.co' || configUrl.search || configUrl.hash ||
        !/^\/[^/]+\/[^/]+\/resolve\/[a-f0-9]{40}\/config\.json$/.test(configUrl.pathname))
        throw new Error('Use a public Hugging Face config.json URL pinned to a full commit SHA');
    const metrics = {totalBytes: 0, loadedBytes: 0, cachedBytes: 0, downloadedBytes: 0};
    const report = file => progress({...metrics, warning, file});
    const put = async (key, bytes, total) => {
        if (!cache) return;
        try {
            await cache.put(key, new Response(bytes, {headers: {
                'X-Kidi-SHA256': await digest(bytes), 'X-Kidi-Size': String(total)}}));
        } catch (error) { warning = `Model cache incomplete: ${error.message}`; cache = null; }
    };
    const get = async (url, start, end) => {
        const ranged = start !== undefined;
        const key = new URL(url);
        if (ranged) key.searchParams.set('kidi_range', `${start}-${end}`);
        const saved = cache && await cache.match(key.href);
        if (saved) {
            const bytes = await saved.arrayBuffer();
            const total = Number(saved.headers.get('X-Kidi-Size'));
            if (Number.isSafeInteger(total) && total > 0 &&
                bytes.byteLength === (ranged ? Math.min(end + 1, total) - start : total) &&
                await digest(bytes) === saved.headers.get('X-Kidi-SHA256')) {
                metrics.cachedBytes += bytes.byteLength;
                return {bytes, total};
            }
            await cache.delete(key.href);
        }
        if (cacheOnly) throw new Error('Model cache incomplete or damaged. Click Load model to download missing data.');
        const response = await fetch(key, {mode: 'cors', credentials: 'omit',
            headers: ranged ? {Range: `bytes=${start}-${end}`} : {}, signal: AbortSignal.timeout(120000)});
        let total;
        if (ranged) {
            const range = /^bytes (\d+)-(\d+)\/(\d+)$/.exec(response.headers.get('Content-Range') || '');
            if (response.status !== 206 || !range || Number(range[1]) !== start ||
                Number(range[2]) !== Math.min(end, Number(range[3]) - 1)) {
                await response.body?.cancel();
                throw new Error(`Invalid byte-range response (HTTP ${response.status})`);
            }
            total = Number(range[3]);
        } else if (!response.ok) {
            await response.body?.cancel();
            throw new Error(`Model metadata HTTP ${response.status}`);
        }
        const bytes = await response.arrayBuffer();
        total ??= bytes.byteLength;
        if (!Number.isSafeInteger(total) || total <= 0 || total >= 2 ** 32 ||
            (ranged && bytes.byteLength !== Math.min(end + 1, total) - start))
            throw new Error('Truncated or oversized model response');
        metrics.downloadedBytes += bytes.byteLength;
        await put(key.href, bytes, total);
        return {bytes, total};
    };
    const configData = await get(configUrl);
    const config = JSON.parse(new TextDecoder().decode(configData.bytes));
    const weightsUrl = new URL('model.safetensors', configUrl);
    const first = await get(weightsUrl, 0, CHUNK_BYTES - 1);
    const size = first.total;
    const headerBytes = Number(new DataView(first.bytes).getBigUint64(0, true));
    if (!Number.isSafeInteger(headerBytes) || headerBytes < 2 || headerBytes + 8 > first.bytes.byteLength)
        throw new Error('Safetensors header must fit in the first download chunk');
    const header = JSON.parse(new TextDecoder().decode(new Uint8Array(first.bytes, 8, headerBytes)));
    const ranges = normalizationPlan(config, header, size - 8 - headerBytes);
    const normalizedHeader = new TextEncoder().encode(JSON.stringify(header));
    if (normalizedHeader.byteLength > headerBytes) throw new Error('Normalized header exceeds reserved space');
    metrics.totalBytes = size + configData.bytes.byteLength;
    metrics.loadedBytes = configData.bytes.byteLength;
    const pointer = allocateWeights(module, size);
    const store = (data, start) => {
        const bytes = new Uint8Array(module.HEAPU8.buffer, pointer + start, data.byteLength);
        bytes.set(new Uint8Array(data));
        for (const range of ranges) {
            const begin = Math.max(0, 8 + headerBytes + range.start - start);
            const end = Math.min(bytes.length, 8 + headerBytes + range.end - start);
            for (let index = begin; index < end; index++) bytes[index] ^= range.mask;
        }
        metrics.loadedBytes += bytes.length;
        report('model.safetensors');
    };
    store(first.bytes, 0);
    for (let start = first.bytes.byteLength; start < size; start += CHUNK_BYTES) {
        const part = await get(weightsUrl, start, Math.min(size, start + CHUNK_BYTES) - 1);
        if (part.total !== size) throw new Error('Checkpoint size changed during download');
        store(part.bytes, start);
    }
    module.HEAPU8.fill(32, pointer + 8, pointer + 8 + headerBytes);
    module.HEAPU8.set(normalizedHeader, pointer + 8);
    module.FS.mkdir('/model');
    mountWeights(module, pointer, size);
    for (const name of ['tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja']) {
        const file = await get(new URL(name, configUrl));
        module.FS.writeFile(`/model/${name}`, new Uint8Array(file.bytes));
        metrics.totalBytes += file.bytes.byteLength;
        metrics.loadedBytes += file.bytes.byteLength;
        report(name);
    }
    module.FS.writeFile('/model/model.yaml', JSON.stringify({format_version: 1,
        weights_file: 'model.safetensors', tokenizer_file: 'tokenizer.json',
        model: {...config.text_config, type: 'gemma4_text', packed_weights_signed: true,
            quantization_config: config.quantization_config},
        decode: {maximum_new_tokens: 1024, context_size: 9216}}));
    return {...metrics, warning, model: 'Gemma 4 E2B IT', precision: 'QAT mixed 2/4/8-bit', id: configUrl.href};
}

export async function loadModel(module, manifestUrl, progress, {cacheOnly = false} = {}) {
    let cache;
    let warning = '';
    try { cache = await caches.open(CACHE_NAME); }
    catch (error) { warning = `Persistent cache unavailable: ${error.message}`; }
    if (new URL(manifestUrl).pathname.endsWith('/config.json'))
        return loadHubModel(module, manifestUrl, progress, cache, warning, cacheOnly);
    let response;
    if (cacheOnly) {
        response = cache && await cache.match(manifestUrl);
        if (!response) throw new Error('Model cache incomplete. Click Load model to download missing data.');
    } else try {
        response = await fetch(manifestUrl, {cache: 'no-cache'});
        if (!response.ok) throw new Error(`Manifest HTTP ${response.status}`);
        if (cache) await cache.put(manifestUrl, response.clone()).catch(error => { warning = error.message; });
    } catch (error) {
        response = cache && await cache.match(manifestUrl);
        if (!response) throw error;
    }
    const manifest = await response.json();
    if (manifest.version !== 1 || !Array.isArray(manifest.files) || !/^[a-f0-9]{64}$/.test(manifest.id))
        throw new Error('Invalid model manifest');
    const names = new Set();
    const allowed = new Set(['model.yaml', 'model.safetensors', 'tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja']);
    for (const file of manifest.files) {
        if (!allowed.has(file.name) || names.has(file.name) || !Number.isSafeInteger(file.size) || file.size <= 0 ||
            file.size >= (file.name === 'model.safetensors' ? 2 ** 32 : 64 * 1024 * 1024) ||
            !Array.isArray(file.chunks) || file.chunks.reduce((sum, chunk) => sum + chunk.size, 0) !== file.size)
            throw new Error('Invalid model file');
        names.add(file.name);
        for (const chunk of file.chunks) {
            if (!Number.isInteger(chunk.size) || chunk.size < 1 || chunk.size > 8 * 1024 * 1024 ||
                !/^[a-f0-9]{64}$/.test(chunk.sha256) || typeof chunk.url !== 'string' || /[:\\]|(^|\/)\.\.(\/|$)/.test(chunk.url))
                throw new Error('Invalid model chunk');
        }
    }
    for (const name of ['model.yaml', 'model.safetensors', 'tokenizer.json', 'tokenizer_config.json'])
        if (!names.has(name)) throw new Error(`Missing ${name}`);
    module.FS.mkdir('/model');
    const metrics = {totalBytes: manifest.files.reduce((sum, file) => sum + file.size, 0), loadedBytes: 0, cachedBytes: 0, downloadedBytes: 0};
    for (const file of [...manifest.files].sort((left, right) => right.size - left.size)) {
        const weights = file.name === 'model.safetensors';
        const pointer = weights ? allocateWeights(module, file.size) : 0;
        const smallFile = weights ? null : new Uint8Array(file.size);
        let offset = 0;
        for (const chunk of file.chunks) {
            const key = new URL(`./__kidi_chunks__/${chunk.sha256}`, manifestUrl).href;
            const verify = async data => data.byteLength === chunk.size && hex(await crypto.subtle.digest('SHA-256', data)) === chunk.sha256;
            let data;
            let cached = cache && await cache.match(key);
            if (cached) {
                data = await cached.arrayBuffer();
                if (!await verify(data)) { await cache.delete(key); cached = null; data = null; }
            }
            if (!cached) {
                if (cacheOnly) throw new Error('Model cache incomplete or damaged. Click Load model to download missing data.');
                const downloaded = await fetch(new URL(chunk.url, manifestUrl));
                if (!downloaded.ok) throw new Error(`Model download HTTP ${downloaded.status}`);
                data = await downloaded.arrayBuffer();
                if (!await verify(data)) throw new Error(`Model checksum mismatch: ${chunk.url}`);
                metrics.downloadedBytes += data.byteLength;
                if (cache) {
                    try { await cache.put(key, new Response(data)); }
                    catch (error) { warning = `Model cache incomplete: ${error.message}`; cache = null; }
                }
            } else metrics.cachedBytes += data.byteLength;
            if (weights) module.HEAPU8.set(new Uint8Array(data), pointer + offset);
            else smallFile.set(new Uint8Array(data), offset);
            offset += data.byteLength;
            metrics.loadedBytes += data.byteLength;
            progress({...metrics, warning, file: file.name});
        }
        if (weights) mountWeights(module, pointer, file.size);
        else module.FS.writeFile(`/model/${file.name}`, smallFile);
    }
    return {...metrics, warning, model: manifest.name, precision: manifest.precision, id: manifest.id};
}

export async function clearModelCache() { return caches.delete(CACHE_NAME); }