// Measures Kidi Wasm memory for Gemma 4 chat in Node: load, a long prompt, optional image, and decode.
// The real browser loader runs against a local Hugging Face snapshot, so heap layout matches the web app.
//
//   node benchmarks/web/heap_probe.mjs --snapshot ~/.cache/kidi/model-hub/models--google--gemma-4-E2B-it-qat-mobile-transformers/snapshots/<sha> \
//       [--glue build-web/single/kidi.mjs] [--threads 1] [--words 1600 | --prompt text] [--tokens 32] [--image photo.jpg]
//       [--json out.json]
import {readFile, open, writeFile} from 'node:fs/promises';
import {basename, dirname, join, resolve} from 'node:path';
import {parseArgs} from 'node:util';
import {pathToFileURL} from 'node:url';

const {values: options} = parseArgs({options: {
    snapshot: {type: 'string'}, glue: {type: 'string', default: 'build-web/single/kidi.mjs'},
    words: {type: 'string', default: '1600'}, tokens: {type: 'string', default: '32'},
    image: {type: 'string'}, json: {type: 'string'}, 'image-pixels': {type: 'string', default: '3000000'},
    threads: {type: 'string', default: '1'}, prompt: {type: 'string'},
}});
if (!options.snapshot) throw new Error('--snapshot is required');
const snapshot = resolve(options.snapshot);
const revision = basename(snapshot);
const repository = basename(dirname(dirname(snapshot))).replace(/^models--/, '').split('--').join('/');
const source = `https://huggingface.co/${repository}/resolve/${revision}/config.json`;

globalThis.caches = {open: async () => { throw new Error('Cache Storage is unavailable in the probe'); }};
globalThis.self ??= globalThis;
globalThis.postMessage ??= () => {};
globalThis.fetch = async (url, request = {}) => {
    const path = join(snapshot, new URL(url).pathname.split('/').at(-1));
    const range = /^bytes=(\d+)-(\d+)$/.exec(request.headers?.Range || '');
    if (!range) return new Response(await readFile(path));
    const handle = await open(path);
    try {
        const {size} = await handle.stat();
        const start = Number(range[1]), end = Math.min(Number(range[2]), size - 1);
        const bytes = Buffer.alloc(end - start + 1);
        await handle.read(bytes, 0, bytes.length, start);
        return new Response(bytes, {status: 206, headers: {'Content-Range': `bytes ${start}-${end}/${size}`}});
    } finally { await handle.close(); }
};

const {loadModel} = await import(new URL('../../web/model-cache.mjs', import.meta.url).href);
const {default: createKidi} = await import(pathToFileURL(resolve(options.glue)).href);
const module = await createKidi({printErr: text => console.error('[wasm]', text)});
const GiB = bytes => `${(bytes / 2 ** 30).toFixed(3)} GiB`;
const call = (name, types = [], args = []) => {
    const result = JSON.parse(module.ccall(name, 'string', types, args));
    if (result.error) throw new Error(`${name}: ${result.error}`);
    return result;
};
const phases = [];
const mark = label => {
    const stats = call('kidi_memory_stats');
    const phase = {label, heap_bytes: module.HEAPU8.byteLength, malloc_in_use_bytes: stats.malloc_in_use_bytes,
        kv_cache_bytes: stats.kv_cache_bytes, rss_bytes: process.memoryUsage().rss};
    phases.push(phase);
    console.log(`${label.padEnd(24)} heap ${GiB(phase.heap_bytes)}  malloc ${GiB(phase.malloc_in_use_bytes)}  ` +
        `rss ${GiB(phase.rss_bytes)}${phase.kv_cache_bytes ? `  kv ${GiB(phase.kv_cache_bytes)}` : ''}`);
    return stats;
};

mark('created');
call('kidi_configure', ['number'], [Number(options.threads)]);
const loaded = await loadModel(module, source, () => {});
mark('weights loaded');
let started = performance.now();
call('kidi_load', ['string'], ['/model']);
mark('kidi_load');
const loadSeconds = (performance.now() - started) / 1000;

const sentence = 'The quick brown fox studies distributed systems, compilers, and numerical linear algebra in the library. ';
const filler = Array.from({length: Math.ceil(Number(options.words) / 16)}, (_, index) => `${index + 1}. ${sentence}`).join('');
const images = [];
if (options.image) {
    module.FS.mkdirTree('/images');
    const path = `/images/0${options.image.toLowerCase().endsWith('.png') ? '.png' : '.jpg'}`;
    module.FS.writeFile(path, new Uint8Array(await readFile(options.image)));
    images.push(path);
}
const content = options.prompt ?? `${filler}\nSummarize the text above in one sentence.`;
const messages = [{role: 'user', content, images}];
started = performance.now();
call('kidi_enqueue', ['string', 'number', 'number'],
    [JSON.stringify(messages), Number(options.tokens), Number(options['image-pixels'])]);
mark('enqueued');
let steps = 0, completed, firstToken;
while (true) {
    const result = call('kidi_step');
    ++steps;
    firstToken ??= result.events.find(event => event.token !== undefined) && steps;
    completed ??= result.events.find(event => event.completed)?.completed;
    if (steps === 1 || steps === firstToken || steps % 32 === 0) mark(`step ${steps}`);
    if (!result.pending) break;
}
const final = mark('done');
const report = {glue: options.glue, threads: Number(options.threads), prompt_tokens: completed?.prompt_tokens, output_tokens: completed?.token_ids.length,
    token_ids: completed?.token_ids, load_seconds: loadSeconds, generation_ms: completed?.generation_ms,
    decode_ms: completed?.decode_ms, decode_tokens: completed?.decode_tokens,
    peak_heap_bytes: Math.max(...phases.map(phase => phase.heap_bytes)),
    weights_heap_bytes: loaded.heapWeightBytes, weights_external_bytes: loaded.externalWeightBytes,
    phases, categories: final.categories, text: completed?.text};
console.log('\nAllocation categories:');
for (const entry of final.categories.slice(0, 16))
    console.log(`  ${entry.category.padEnd(44)} ${GiB(entry.bytes)}  x${entry.count}`);
console.log(`\npeak heap ${GiB(report.peak_heap_bytes)}; prompt ${report.prompt_tokens} tokens; ` +
    `prefill ${((report.prompt_tokens - 1) * 1000 / (report.generation_ms - report.decode_ms)).toFixed(2)} tok/s; ` +
    `decode ${(report.decode_tokens * 1000 / report.decode_ms).toFixed(2)} tok/s; reply: ${report.text}`);
if (options.json) await writeFile(options.json, JSON.stringify(report, null, 2));
