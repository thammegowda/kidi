const CACHE_NAME = 'kidi-voice-prompts-v1';
const FORMAT_VERSION = 1;
const CACHE_ORIGIN = 'https://kidi.invalid/__voice_prompts__/';
const encoder = new TextEncoder();

const hex = bytes => Array.from(new Uint8Array(bytes), value => value.toString(16).padStart(2, '0')).join('');
const sha256 = bytes => crypto.subtle.digest('SHA-256', bytes).then(hex);
const canonical = value => Array.isArray(value) ? value.map(canonical) :
    value && typeof value === 'object' ? Object.fromEntries(
        Object.entries(value).sort(([left], [right]) => left.localeCompare(right))
            .map(([key, item]) => [key, canonical(item)])) : value;

function requireDescriptor(descriptor) {
    if (!descriptor || typeof descriptor.id !== 'string' || !/^[a-f0-9]{64}$/.test(descriptor.id) ||
        typeof descriptor.metadata !== 'string' || !descriptor.config)
        throw new Error('Invalid voice prompt descriptor');
    return descriptor;
}

function tokensFrom(value, config) {
    const source = value instanceof Int32Array || Array.isArray(value) ? value : null;
    const codebooks = Number(config.codebooks), frameRate = Number(config.frameRate);
    const vocabulary = Number(config.vocabulary), maximumSeconds = Number(config.maximumSeconds);
    if (!source || !Number.isSafeInteger(codebooks) || codebooks <= 0 ||
        !Number.isSafeInteger(frameRate) || frameRate <= 0 ||
        !Number.isSafeInteger(vocabulary) || vocabulary <= 0 ||
        !Number.isFinite(maximumSeconds) || maximumSeconds <= 0 ||
        !source.length || source.length % codebooks ||
        source.length / codebooks > frameRate * maximumSeconds ||
        source.some(token => !Number.isSafeInteger(token) || token < 0 || token >= vocabulary))
        throw new Error('Invalid cached voice prompt tokens');
    return value instanceof Int32Array ? value : Int32Array.from(value);
}

async function payloadDigest(text, tokens) {
    const transcript = encoder.encode(text);
    const bytes = new Uint8Array(4 + transcript.length + tokens.byteLength);
    new DataView(bytes.buffer).setUint32(0, transcript.length, true);
    bytes.set(transcript, 4);
    const view = new DataView(bytes.buffer, 4 + transcript.length);
    for (let index = 0; index < tokens.length; ++index) view.setInt32(index * 4, tokens[index], true);
    return sha256(bytes);
}

export async function describeVoicePrompt(config) {
    const normalized = canonical(config);
    const metadata = JSON.stringify(normalized);
    return {id: await sha256(encoder.encode(metadata)), metadata, config: normalized};
}

export async function loadVoicePrompt(descriptor) {
    requireDescriptor(descriptor);
    const response = await (await caches.open(CACHE_NAME)).match(CACHE_ORIGIN + descriptor.id);
    if (!response) return null;
    const payload = await response.json();
    if (payload?.format !== FORMAT_VERSION || payload.key !== descriptor.id ||
        payload.metadata !== descriptor.metadata || typeof payload.text !== 'string' || !payload.text.trim())
        throw new Error('Cached voice prompt metadata is invalid');
    const tokens = tokensFrom(payload.tokens, descriptor.config);
    if (payload.digest !== await payloadDigest(payload.text, tokens))
        throw new Error('Cached voice prompt checksum mismatch');
    return {text: payload.text, tokens};
}

export async function storeVoicePrompt(descriptor, prompt) {
    requireDescriptor(descriptor);
    if (!prompt || typeof prompt.text !== 'string' || !prompt.text.trim())
        throw new Error('Invalid voice prompt text');
    const tokens = tokensFrom(prompt.tokens, descriptor.config);
    const payload = {
        format: FORMAT_VERSION,
        key: descriptor.id,
        metadata: descriptor.metadata,
        text: prompt.text,
        tokens: [...tokens],
        digest: await payloadDigest(prompt.text, tokens),
    };
    await (await caches.open(CACHE_NAME)).put(CACHE_ORIGIN + descriptor.id,
        new Response(JSON.stringify(payload), {headers: {'Content-Type': 'application/json'}}));
}

export async function clearVoicePromptCache() {
    return caches.delete(CACHE_NAME);
}
