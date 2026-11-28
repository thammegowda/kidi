import {blobToDataUrl} from './liteapp/media/base64.mjs';

const CACHE = 'kidi-chat-images-v1';
const LEGACY_CACHES = ['kidi-chat-images-v2'];
export const MAX_IMAGES = 8;
export const MAX_IMAGE_BYTES = 32 * 1024 * 1024;
const imageKey = id => {
    if (!/^[a-f0-9]{64}$/.test(id)) throw new Error('Invalid image attachment');
    return new URL(`./__images__/${id}`, import.meta.url).href;
};

export async function storeImage(blob, name) {
    if (!blob.size || blob.size > MAX_IMAGE_BYTES || !['image/jpeg', 'image/png'].includes(blob.type))
        throw new Error('Choose a JPEG or PNG image smaller than 32 MiB');
    const digest = await crypto.subtle.digest('SHA-256', await blob.arrayBuffer());
    const id = Array.from(new Uint8Array(digest), byte => byte.toString(16).padStart(2, '0')).join('');
    const cache = await caches.open(CACHE);
    await cache.put(imageKey(id), new Response(blob));
    return {id, name: String(name || 'Image').slice(0, 256), type: blob.type, size: blob.size};
}

export async function readImage(image) {
    let response = await (await caches.open(CACHE)).match(imageKey(image.id));
    for (const name of LEGACY_CACHES) {
        if (response) break;
        response = await (await caches.open(name)).match(imageKey(image.id));
    }
    if (!response) throw new Error(`Image unavailable: ${image.name || 'attachment'}. Start a new chat and attach it again.`);
    const blob = await response.blob();
    if (!blob.size || blob.size > MAX_IMAGE_BYTES || !['image/jpeg', 'image/png'].includes(blob.type))
        throw new Error('Invalid cached image');
    return blob;
}

export async function removeUnusedImages(retained) {
    const keep = new Set(retained.map(image => imageKey(image.id)));
    for (const name of [CACHE, ...LEGACY_CACHES]) {
        const cache = await caches.open(name);
        for (const request of await cache.keys()) if (!keep.has(request.url)) await cache.delete(request);
    }
}

// OpenAI chat messages for kidi_enqueue: each user image becomes an image_url data URL part ahead of the text.
export async function chatMessages(messages) {
    let count = 0, total = 0;
    const result = [];
    for (const {role, content, images = []} of messages) {
        if (!images.length) {
            result.push({role, content});
            continue;
        }
        count += images.length;
        if (role !== 'user' || count > MAX_IMAGES)
            throw new Error('Images require user messages, at most eight per conversation');
        const parts = [];
        for (const image of images) {
            const blob = await readImage(image);
            total += blob.size;
            if (total > MAX_IMAGE_BYTES) throw new Error('Images exceed the 32 MiB conversation limit');
            parts.push({type: 'image_url', image_url: {url: await blobToDataUrl(blob)}});
        }
        parts.push({type: 'text', text: content});
        result.push({role, content: parts});
    }
    return result;
}

export const hasImages = messages => messages.some(message => Array.isArray(message.content));

// Copies `text` into the module heap as NUL-terminated UTF-8. ccall copies string arguments onto the 2 MiB Wasm
// stack, which image requests overflow. The caller frees the pointer.
export function heapString(module, text) {
    const bytes = new TextEncoder().encode(text);
    const pointer = Number(module._malloc(bytes.length + 1));
    if (!pointer || pointer + bytes.length + 1 > module.HEAPU8.byteLength) throw new Error('Cannot allocate request');
    module.HEAPU8.set(bytes, pointer);
    module.HEAPU8[pointer + bytes.length] = 0;
    return pointer;
}