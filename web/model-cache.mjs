const CACHE_NAME = 'kidi-model-v1';
export const MAX_WASM_MEMORY = 4 * 1024 ** 3;
export const webGpuSupport = () => globalThis.navigator?.gpu ? '' : 'this browser has no WebGPU';
const CHUNK_BYTES = 8 * 1024 * 1024;
const MODEL_ID = /^[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+$/;
const hex = bytes => Array.from(new Uint8Array(bytes), byte => byte.toString(16).padStart(2, '0')).join('');
const digest = async data => hex(await crypto.subtle.digest('SHA-256', data));

const referenceKey = modelId => `https://kidi.invalid/__model_refs__/${encodeURIComponent(modelId)}`;
const configUrl = (modelId, revision) => `https://huggingface.co/${modelId}/resolve/${revision}/config.json`;

async function resolveReference(source, cache, cacheOnly) {
    const value = String(source).trim();
    if (!MODEL_ID.test(value)) return new URL(value).href;
    const key = referenceKey(value);
    const cached = cache && await cache.match(key);
    const cachedUrl = async () => {
        if (!cached) return null;
        const metadata = await cached.json();
        const expected = configUrl(value, metadata.revision);
        return /^[a-f0-9]{40}$/.test(metadata.revision) && metadata.url === expected ? expected : null;
    };
    if (cacheOnly) {
        const resolved = await cachedUrl();
        if (!resolved) throw new Error('Model cache has no resolved revision. Click Load model while online.');
        return resolved;
    }
    try {
        const path = value.split('/').map(encodeURIComponent).join('/');
        const response = await fetch(`https://huggingface.co/api/models/${path}/revision/main`,
            {mode: 'cors', credentials: 'omit', cache: 'no-cache', signal: AbortSignal.timeout(30000)});
        if (!response.ok) throw new Error(`Hugging Face model lookup HTTP ${response.status}`);
        const metadata = await response.json();
        if (!/^[a-f0-9]{40}$/.test(metadata.sha || '')) throw new Error('Hugging Face returned an invalid revision');
        const url = configUrl(value, metadata.sha);
        if (cache) await cache.put(key, new Response(JSON.stringify({modelId: value, revision: metadata.sha, url}),
            {headers: {'Content-Type': 'application/json'}}));
        return url;
    } catch (error) {
        const resolved = await cachedUrl();
        if (resolved) return resolved;
        throw error;
    }
}

export async function resolveModelSource(source, {cacheOnly = false} = {}) {
    const cache = await caches.open(CACHE_NAME);
    return resolveReference(source, cache, cacheOnly);
}

function hubFile(url) {
    const match = /^\/([^/]+)\/([^/]+)\/resolve\/([a-f0-9]{40})\/([^/]+)$/.exec(url.pathname);
    return match && {modelId: `${match[1]}/${match[2]}`, revision: match[3], name: match[4]};
}

async function cachedBytes(response, url) {
    const sizeHeader = response.headers.get('X-Kidi-Size');
    const total = sizeHeader === null ? NaN : Number(sizeHeader);
    const range = /^(\d+)-(\d+)$/.exec(url.searchParams.get('kidi_range') || '');
    if (range && Number.isSafeInteger(total) && total > 0) {
        const start = Number(range[1]), end = Number(range[2]);
        return Math.max(0, Math.min(end + 1, total) - start);
    }
    if (Number.isSafeInteger(total) && total >= 0) return total;
    const lengthHeader = response.headers.get('Content-Length');
    const length = lengthHeader === null ? NaN : Number(lengthHeader);
    if (Number.isSafeInteger(length) && length >= 0) return length;
    return (await response.arrayBuffer()).byteLength;
}

async function cacheInventory() {
    const cache = await caches.open(CACHE_NAME);
    const requests = await cache.keys();
    const references = new Map();
    const hubs = new Map();
    for (const request of requests) {
        const url = new URL(request.url);
        if (url.origin === 'https://kidi.invalid' && url.pathname.startsWith('/__model_refs__/')) {
            try {
                const metadata = await (await cache.match(request)).json();
                if (MODEL_ID.test(metadata.modelId) && /^[a-f0-9]{40}$/.test(metadata.revision) &&
                    metadata.url === configUrl(metadata.modelId, metadata.revision))
                    references.set(metadata.url, {modelId: metadata.modelId, key: request.url});
            } catch {}
            continue;
        }
        const file = url.origin === 'https://huggingface.co' && hubFile(url);
        if (file) {
            const source = configUrl(file.modelId, file.revision);
            let model = hubs.get(source);
            if (!model) {
                model = {id: source, source, modelId: file.modelId, revision: file.revision,
                    files: new Map(), cacheKeys: new Set(), config: null};
                hubs.set(source, model);
            }
            const response = await cache.match(request);
            const record = model.files.get(file.name) || {name: file.name, bytes: 0};
            record.bytes += await cachedBytes(response.clone(), url);
            model.files.set(file.name, record);
            model.cacheKeys.add(request.url);
            if (file.name === 'config.json' && !url.search) {
                try { model.config = await response.json(); } catch {}
            }
        }
    }
    const models = [];
    for (const model of hubs.values()) {
        const reference = references.get(model.source);
        if (reference) model.cacheKeys.add(reference.key);
        const files = [...model.files.values()].sort((left, right) => right.bytes - left.bytes);
        const kind = model.config?.model_type === 'whisper' ? 'Speech' :
            model.config?.model_type === 'gemma4' ? 'Chat' : 'Model';
        models.push({...model, name: reference?.modelId || model.modelId, kind, files,
            bytes: files.reduce((sum, file) => sum + file.bytes, 0), complete: await isModelCached(model.source)});
    }
    return {cache, models};
}

export async function listCachedModels() {
    const {models} = await cacheInventory();
    return models.map(({cacheKeys, ...model}) => model);
}

export async function deleteCachedModel(id) {
    const {cache, models} = await cacheInventory();
    const target = models.find(model => model.id === id);
    if (!target) return false;
    for (const key of target.cacheKeys) await cache.delete(key);
    return true;
}

export async function isModelCached(source) {
    try {
        const cache = await caches.open(CACHE_NAME);
        const url = new URL(await resolveReference(source, cache, true));
        const keys = new Set((await cache.keys()).map(request => request.url));
        if (!keys.has(url.href)) return false;
        if (url.pathname.endsWith('/config.json')) {
            if (url.origin !== 'https://huggingface.co' || url.search || url.hash ||
                !/^\/[^/]+\/[^/]+\/resolve\/[a-f0-9]{40}\/config\.json$/.test(url.pathname)) return false;
            const response = await cache.match(url.href);
            const config = await response.json();
            const required = config.model_type === 'whisper' &&
                config.architectures?.includes('WhisperForConditionalGeneration')
                ? ['tokenizer.json', 'preprocessor_config.json', 'generation_config.json']
                : ['tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja'];
            for (const name of required)
                if (!keys.has(new URL(name, url).href)) return false;
            const key = new URL('model.safetensors', url);
            key.searchParams.set('kidi_range', `0-${CHUNK_BYTES - 1}`);
            const first = await cache.match(key.href);
            const size = Number(first?.headers.get('X-Kidi-Size'));
            first?.body?.cancel().catch(() => {});
            if (!Number.isSafeInteger(size) || size <= 0 || size >= MAX_WASM_MEMORY) return false;
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
    if (!Number.isSafeInteger(size) || size <= 0 || size >= MAX_WASM_MEMORY)
        throw new Error('Checkpoint exceeds the WebAssembly memory budget');
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
    const gemma = config.model_type === 'gemma4' && quantization?.quant_method === 'gemma' &&
        quantization.quantize_embeddings && config.text_config && !config.text_config.enable_moe_block;
    const whisper = config.model_type === 'whisper' &&
        config.architectures?.includes('WhisperForConditionalGeneration');
    if (!gemma && !whisper) throw new Error('Expected a supported Gemma 4 or Whisper checkpoint');
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
        let mask = 0, bits = 0;
        if (gemma && /\.(weight|embedding_quantized)$/.test(name) && ['U8', 'I8'].includes(tensor.dtype)) {
            const module = name.replace(/\.(weight|embedding_quantized)$/, '');
            if (quantization.modules_to_not_convert.some(excluded => module.includes(excluded)))
                throw new Error(`Unexpected quantized excluded tensor: ${name}`);
            const override = Object.entries(quantization.module_quant_configs)
                .find(([pattern]) => new RegExp(pattern).test(module));
            bits = override ? override[1].num_bits : quantization.num_bits;
            if (![2, 4, 8].includes(bits) || tensor.dtype !== (bits === 8 ? 'I8' : 'U8'))
                throw new Error(`Unsupported packed tensor: ${name}`);
            mask = bits === 2 ? 0xaa : bits === 4 ? 0x88 : 0;
            tensor.dtype = 'U8';
        }
        ranges.push({name, start, end, mask, bits});
    }
    ranges.sort((left, right) => left.start - right.start);
    let end = 0;
    for (const range of ranges) {
        if (range.start !== end) throw new Error('Overlapping or incomplete checkpoint tensors');
        end = range.end;
    }
    if (end !== dataBytes) throw new Error('Checkpoint tensor sizes do not cover the payload');
    return ranges;
}

const ALIGNMENT = 64;
const SHARD_BYTES = 256 * 1024 * 1024;
const alignUp = (value, alignment) => Math.ceil(value / alignment) * alignment;
// Chat never runs the audio tower, and per-layer embeddings are read a few rows per token, so neither needs heap space.
const EXTERNAL_TENSORS = new Set(['model.language_model.embed_tokens_per_layer.embedding_quantized']);
// With WebGPU, decoder projections are only read by GPU kernels, so they go straight to GPU buffers.
const DEVICE_TENSOR = /^model\.language_model\.layers\.\d+\..+\.weight$/;
const placement = (range, gemma, gpu) => !gemma ? 'heap'
    : /^model\.(audio_tower|embed_audio)\./.test(range.name) ? 'skip'
    : EXTERNAL_TENSORS.has(range.name) ? 'external'
    : gpu && range.bits && DEVICE_TENSOR.test(range.name) ? 'gpu' : 'heap';

// GPU kernels read 2- and 4-bit weights interleaved, so one shift and mask yields the four bytes of a dot product:
// within each 32-bit word, value `slot` moves to byte `slot % 4` at bit `bits * floor(slot / 4)`.
const interleaveTables = new Map();
function interleaveTable(bits) {
    if (!interleaveTables.has(bits)) {
        const table = new Uint32Array(4 * 256), slots = 32 / bits, mask = (1 << bits) - 1;
        for (let position = 0; position < 4; position++)
            for (let value = 0; value < 256; value++) {
                const input = (value << (8 * position)) >>> 0;
                let output = 0;
                for (let slot = 0; slot < slots; slot++)
                    output |= ((input >>> (slot * bits)) & mask) << (8 * (slot % 4) + bits * Math.floor(slot / 4));
                table[position * 256 + value] = output >>> 0;
            }
        interleaveTables.set(bits, table);
    }
    return interleaveTables.get(bits);
}
export function interleavePacked(bytes, bits) {
    if (bits !== 2 && bits !== 4) return;
    if (bytes.byteOffset % 4 || bytes.length % 4) throw new Error('Packed weights must occupy whole 32-bit words');
    const table = interleaveTable(bits), words = new Uint32Array(bytes.buffer, bytes.byteOffset, bytes.length / 4);
    for (let index = 0; index < words.length; index++) {
        const word = words[index];
        words[index] = (table[word & 255] | table[256 + ((word >>> 8) & 255)] |
            table[512 + ((word >>> 16) & 255)] | table[768 + (word >>> 24)]) >>> 0;
    }
}

function applyMask(bytes, mask) {
    if (!mask) return;
    let index = 0;
    if (bytes.byteOffset % 4 === 0) {
        const words = new Uint32Array(bytes.buffer, bytes.byteOffset, bytes.length >>> 2);
        const wide = (mask * 0x01010101) >>> 0;
        for (let word = 0; word < words.length; word++) words[word] ^= wide;
        index = words.length * 4;
    }
    for (; index < bytes.length; index++) bytes[index] ^= mask;
}

/// Compacted heap layout: excluded tensors removed, 64-byte aligned starts, and each gate projection followed directly by
/// its up projection so the runtime can view the pair as one fused matrix without copying. With `gpu`, decoder
/// projections are staged for GPU buffers instead, gate and up fused into one `gate_up_proj` buffer.
export function weightLayout(header, ranges, gemma, gpu = false) {
    const byName = new Map(ranges.map(range => [range.name, range]));
    const metadata = header.__metadata__;
    const layout = {header: metadata ? {__metadata__: metadata} : {}, tensors: [], external: [], device: [],
        dataBytes: 0};
    let cursor = 0;
    const place = (range, adjacent) => {
        range.target = placement(range, gemma, gpu);
        if (range.target === 'skip') return;
        const info = header[range.name];
        if (range.target === 'gpu') {
            const bytes = range.end - range.start;
            if (adjacent) {
                const entry = layout.device.at(-1);
                entry.name = entry.name.replace('.mlp.gate_proj.', '.mlp.gate_up_proj.');
                entry.shape = [entry.shape[0] * 2, ...entry.shape.slice(1)];
                range.device = entry;
                range.offset = entry.bytes;
                entry.bytes += bytes;
                return;
            }
            range.device = {name: range.name, dtype: info.dtype, shape: [...info.shape], bits: range.bits, bytes};
            range.offset = 0;
            layout.device.push(range.device);
            return;
        }
        if (range.target === 'external') {
            const rowBytes = (range.end - range.start) / info.shape[0];
            const rowsPerShard = Math.max(1, Math.floor(SHARD_BYTES / rowBytes));
            const shardBytes = rowsPerShard * rowBytes;
            const shards = [];
            for (let start = 0; start < range.end - range.start; start += shardBytes)
                shards.push(new Uint8Array(Math.min(shardBytes, range.end - range.start - start)));
            range.table = {name: range.name, dtype: info.dtype, shape: info.shape, rowBytes, rowsPerShard, shardBytes, shards};
            layout.external.push(range.table);
            return;
        }
        range.offset = adjacent ? cursor : alignUp(cursor, ALIGNMENT);
        cursor = range.offset + range.end - range.start;
        layout.header[range.name] = {dtype: info.dtype, shape: info.shape, data_offsets: [range.offset, cursor]};
        layout.tensors.push(range);
    };
    for (const range of ranges) {
        if (range.name.includes('.mlp.up_proj.') && byName.has(range.name.replace('.mlp.up_proj.', '.mlp.gate_proj.')))
            continue;
        place(range, false);
        const partner = byName.get(range.name.replace('.mlp.gate_proj.', '.mlp.up_proj.'));
        if (partner && partner !== range) place(partner, range.target === 'heap' || range.target === 'gpu');
    }
    layout.dataBytes = cursor;
    const json = new TextEncoder().encode(JSON.stringify(layout.header));
    layout.headerBytes = alignUp(8 + json.byteLength, ALIGNMENT) - 8;
    layout.headerJson = json;
    layout.totalBytes = 8 + layout.headerBytes + layout.dataBytes;
    return layout;
}

function externalTables(module, tables, device = []) {
    const byName = new Map(tables.map(table => [table.name, table]));
    return {
        bytes: tables.reduce((sum, table) => sum + table.shards.reduce((total, shard) => total + shard.byteLength, 0), 0),
        deviceBytes: device.reduce((sum, entry) => sum + entry.bytes, 0),
        // GPU buffers the runtime adopts by name; ownership passes to Wasm when taken.
        gpu: new Map(device.map(entry => [entry.name, {handle: entry.handle, layout: entry.layout}])),
        gather(name, rowsAddress, count, destination, rowBytes) {
            const table = byName.get(name);
            if (!table || table.rowBytes !== rowBytes) throw new Error(`Unknown external table ${name}`);
            const heap = module.HEAPU8;
            const rows = new Int32Array(heap.buffer, rowsAddress, count);
            for (let index = 0; index < count; index++) {
                const row = rows[index];
                if (!Number.isInteger(row) || row < 0 || row >= table.shape[0]) throw new Error(`Row ${row} outside ${name}`);
                const shard = Math.floor(row / table.rowsPerShard), offset = (row - shard * table.rowsPerShard) * rowBytes;
                heap.set(table.shards[shard].subarray(offset, offset + rowBytes), destination + index * rowBytes);
            }
        },
    };
}

async function loadHubModel(module, source, progress, cache, warning, cacheOnly, gpu) {
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
        if (!Number.isSafeInteger(total) || total <= 0 || total >= MAX_WASM_MEMORY ||
            (ranged && bytes.byteLength !== Math.min(end + 1, total) - start))
            throw new Error('Truncated or oversized model response');
        metrics.downloadedBytes += bytes.byteLength;
        await put(key.href, bytes, total);
        return {bytes, total};
    };
    const configData = await get(configUrl);
    const config = JSON.parse(new TextDecoder().decode(configData.bytes));
    const whisper = config.model_type === 'whisper' &&
        config.architectures?.includes('WhisperForConditionalGeneration');
    const weightsUrl = new URL('model.safetensors', configUrl);
    const first = await get(weightsUrl, 0, CHUNK_BYTES - 1);
    const size = first.total;
    const headerBytes = Number(new DataView(first.bytes).getBigUint64(0, true));
    if (!Number.isSafeInteger(headerBytes) || headerBytes < 2 || headerBytes + 8 > first.bytes.byteLength)
        throw new Error('Safetensors header must fit in the first download chunk');
    const header = JSON.parse(new TextDecoder().decode(new Uint8Array(first.bytes, 8, headerBytes)));
    const ranges = normalizationPlan(config, header, size - 8 - headerBytes);
    const layout = weightLayout(header, ranges, !whisper, Boolean(gpu));
    metrics.totalBytes = size + configData.bytes.byteLength;
    metrics.loadedBytes = configData.bytes.byteLength;
    const pointer = allocateWeights(module, layout.totalBytes + ALIGNMENT);
    const base = alignUp(pointer, ALIGNMENT), data = base + 8 + layout.headerBytes, payload = 8 + headerBytes;
    let next = 0;
    const store = (chunk, fileStart) => {
        const bytes = new Uint8Array(chunk), fileEnd = fileStart + bytes.length;
        while (next < ranges.length && payload + ranges[next].end <= fileStart) next++;
        for (let index = next; index < ranges.length && payload + ranges[index].start < fileEnd; index++) {
            const range = ranges[index];
            const begin = Math.max(payload + range.start, fileStart), end = Math.min(payload + range.end, fileEnd);
            if (begin >= end || range.target === 'skip') continue;
            const piece = bytes.subarray(begin - fileStart, end - fileStart);
            let within = begin - payload - range.start;
            if (range.target === 'heap') {
                const destination = new Uint8Array(module.HEAPU8.buffer, data + range.offset + within, piece.length);
                destination.set(piece);
                applyMask(destination, range.mask);
                continue;
            }
            if (range.target === 'gpu') {
                const entry = range.device;
                entry.staging ??= new Uint8Array(entry.bytes);
                const destination = entry.staging.subarray(range.offset + within, range.offset + within + piece.length);
                destination.set(piece);
                applyMask(destination, range.mask);
                entry.received = (entry.received ?? 0) + piece.length;
                if (entry.received === entry.bytes) {
                    interleavePacked(entry.staging, entry.bits);
                    entry.handle = gpu.allocate(entry.bytes);
                    gpu.upload(entry.handle, 0, entry.staging);
                    entry.layout = entry.bits === 2 || entry.bits === 4 ? entry.bits : 0;
                    entry.staging = null;
                }
                continue;
            }
            for (let consumed = 0; consumed < piece.length;) {
                const {shardBytes, shards} = range.table;
                const shard = Math.floor(within / shardBytes), inner = within - shard * shardBytes;
                const count = Math.min(piece.length - consumed, shardBytes - inner);
                const destination = shards[shard].subarray(inner, inner + count);
                destination.set(piece.subarray(consumed, consumed + count));
                applyMask(destination, range.mask);
                consumed += count;
                within += count;
            }
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
    new DataView(module.HEAPU8.buffer, base, 8).setBigUint64(0, BigInt(layout.headerBytes), true);
    module.HEAPU8.fill(32, base + 8, data);
    module.HEAPU8.set(layout.headerJson, base + 8);
    module.FS.mkdir('/model');
    mountWeights(module, base, layout.totalBytes);
    if (layout.device.some(entry => !entry.handle)) throw new Error('Incomplete GPU weight upload');
    module.kidiExternal = externalTables(module, layout.external, layout.device);
    module.FS.writeFile('/model/config.json', new Uint8Array(configData.bytes));
    const metadata = whisper ? ['tokenizer.json', 'preprocessor_config.json', 'generation_config.json']
        : ['tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja'];
    for (const name of metadata) {
        const file = await get(new URL(name, configUrl));
        module.FS.writeFile(`/model/${name}`, new Uint8Array(file.bytes));
        metrics.totalBytes += file.bytes.byteLength;
        metrics.loadedBytes += file.bytes.byteLength;
        report(name);
    }
    if (!whisper) module.FS.writeFile('/model/model.yaml', JSON.stringify({format_version: 1,
        weights_file: 'model.safetensors', tokenizer_file: 'tokenizer.json',
        model: {...config.text_config, type: 'gemma4_text', packed_weights_signed: true,
            quantization_config: config.quantization_config},
        external_tensors: [...layout.external, ...layout.device].map(table => ({name: table.name,
            dtype: table.dtype.toLowerCase(), shape: table.shape})),
        decode: {maximum_new_tokens: 1024, context_size: 9216}}));
    const modelName = whisper ? (config._name_or_path?.split('/').at(-1) || 'Whisper') : 'Gemma 4 E2B IT';
    return {...metrics, warning, model: modelName,
        precision: whisper ? 'FP32' : 'QAT mixed 2/4/8-bit', id: configUrl.href,
        heapWeightBytes: layout.totalBytes, externalWeightBytes: module.kidiExternal.bytes,
        gpuWeightBytes: module.kidiExternal.deviceBytes};
}

/// Loads a model into `module`. With a WebGPU runtime `gpu`, decoder projection weights go straight to GPU buffers.
export async function loadModel(module, manifestUrl, progress, {cacheOnly = false, gpu = null} = {}) {
    let cache;
    let warning = '';
    try { cache = await caches.open(CACHE_NAME); }
    catch (error) { warning = `Persistent cache unavailable: ${error.message}`; }
    manifestUrl = await resolveReference(manifestUrl, cache, cacheOnly);
    if (new URL(manifestUrl).pathname.endsWith('/config.json'))
        return loadHubModel(module, manifestUrl, progress, cache, warning, cacheOnly, gpu);
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
            file.size >= (file.name === 'model.safetensors' ? MAX_WASM_MEMORY : 64 * 1024 * 1024) ||
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
    const heapWeightBytes = manifest.files.find(file => file.name === 'model.safetensors').size;
    return {...metrics, warning, model: manifest.name, precision: manifest.precision, id: manifest.id,
        heapWeightBytes, externalWeightBytes: 0};
}

export async function clearModelCache() { return caches.delete(CACHE_NAME); }