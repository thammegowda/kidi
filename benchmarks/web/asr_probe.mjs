// Measures browser Whisper in Node through the web loader: a first load (download from the local Hugging Face cache,
// INT8 conversion, warm-up), a reload that reuses the converted checkpoint from an in-memory Cache Storage, then
// transcription of 16 kHz PCM WAV files.
//   node benchmarks/web/asr_probe.mjs --model openai/whisper-small --revision <sha> \
//       [--glue build-web/threads/kidi.mjs] [--threads 4] [--hub ~/.cache/huggingface/hub] speech.wav ...
// Files are read from <hub>/models--<owner>--<repo>/snapshots/<revision>/; the GGML weights need
// ggerganov/whisper.cpp at the loader's pinned revision.
import {open, readFile} from 'node:fs/promises';
import {homedir} from 'node:os';
import {join, resolve} from 'node:path';
import {pathToFileURL} from 'node:url';
import {parseArgs} from 'node:util';

const {values: options, positionals: wavs} = parseArgs({allowPositionals: true, options: {
    model: {type: 'string', default: 'openai/whisper-small'}, revision: {type: 'string'},
    glue: {type: 'string', default: 'build-web/threads/kidi.mjs'}, threads: {type: 'string', default: '4'},
    hub: {type: 'string', default: join(homedir(), '.cache/huggingface/hub')},
}});
if (!options.revision) throw new Error('--revision is required');

const stored = new Map();
globalThis.caches = {open: async () => ({
    match: async key => stored.get(String(key))?.clone(), keys: async () => [...stored.keys()].map(url => ({url})),
    put: async (key, response) => { stored.set(String(key), new Response(await response.arrayBuffer(), response)); },
    delete: async key => stored.delete(typeof key === 'string' ? key : key.url),
})};
globalThis.fetch = async (url, request = {}) => {
    const match = /^\/([^/]+)\/([^/]+)\/resolve\/([^/]+)\/([^/]+)$/.exec(new URL(url).pathname);
    const path = join(options.hub, `models--${match[1]}--${match[2]}`, 'snapshots', match[3], match[4]);
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

const pcm = bytes => {
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    for (let offset = 12; offset + 8 <= bytes.length;) {
        const id = String.fromCharCode(...bytes.subarray(offset, offset + 4)), size = view.getUint32(offset + 4, true);
        if (id === 'data') return Float32Array.from(new Int16Array(bytes.buffer.slice(bytes.byteOffset + offset + 8,
            bytes.byteOffset + offset + 8 + size)), value => value / 32768);
        offset += 8 + size + (size & 1);
    }
    throw new Error('WAV has no data chunk');
};

const {loadModel, storeConvertedSpeech} = await import(new URL('../../web/model-cache.mjs', import.meta.url).href);
const {default: createKidi} = await import(pathToFileURL(resolve(options.glue)).href);
const GiB = bytes => `${(bytes / 2 ** 30).toFixed(2)} GiB`;
let module, call;
// Loads like asr-worker.mjs: fetch or reuse weights, then convert (first load only) and warm up.
const load = async label => {
    module = await createKidi({printErr: text => console.error('[wasm]', text)});
    call = (name, types = [], args = []) => {
        const result = JSON.parse(module.ccall(name, 'string', types, args));
        if (result.error) throw new Error(`${name}: ${result.error}`);
        return result;
    };
    call('kidi_configure', ['number'], [Number(options.threads)]);
    let started = performance.now();
    const loaded = await loadModel(module,
        `https://huggingface.co/${options.model}/resolve/${options.revision}/config.json`, () => {});
    const fetched = performance.now() - started;
    started = performance.now();
    call('kidi_load_asr', ['string'], ['/model']);
    const runtime = performance.now() - started;
    await storeConvertedSpeech(module, loaded);
    console.log(`${label} ${options.model}: ${(loaded.totalBytes / 1e6).toFixed(0)} MB read in ${fetched.toFixed(0)} ms; ` +
        `${loaded.convertedUrl ? 'convert and ' : ''}warm up ${runtime.toFixed(0)} ms; heap ${GiB(module.HEAPU8.byteLength)}`);
};
await load('first load');
await load('reload');
for (const wav of wavs) {
    const audio = pcm(await readFile(wav));
    const pointer = Number(module._malloc(audio.byteLength));
    new Float32Array(module.HEAPU8.buffer, pointer, audio.length).set(audio);
    const started = performance.now();
    const result = call('kidi_transcribe', ['pointer', 'number', 'string', 'number'], [pointer, audio.length, 'auto', 128]);
    module._free(pointer);
    console.log(`${(audio.length / 16000).toFixed(1)} s audio in ${(performance.now() - started).toFixed(0)} ms ` +
        `(encode ${result.encode_ms.toFixed(0)}, decode ${result.decode_ms.toFixed(0)}) ${result.language}:${result.text}`);
}
console.log(`heap ${GiB(module.HEAPU8.byteLength)}`);
process.exit(0);
