const CACHE = 'kidi-chat-images-v1';
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
    const response = await (await caches.open(CACHE)).match(imageKey(image.id));
    if (!response) throw new Error(`Image unavailable: ${image.name || 'attachment'}. Start a new chat and attach it again.`);
    const blob = await response.blob();
    if (!blob.size || blob.size > MAX_IMAGE_BYTES || !['image/jpeg', 'image/png'].includes(blob.type))
        throw new Error('Invalid cached image');
    return blob;
}

export async function removeUnusedImages(retained) {
    const cache = await caches.open(CACHE);
    const keep = new Set(retained.map(image => imageKey(image.id)));
    for (const request of await cache.keys()) if (!keep.has(request.url)) await cache.delete(request);
}

export async function stageImages(module, messages) {
    const files = [];
    const dispose = () => { for (const path of files) module.FS.unlink(path); };
    let total = 0;
    try {
        const staged = [];
        for (const message of messages) {
            const images = [];
            for (const image of message.images || []) {
                if (message.role !== 'user' || files.length >= MAX_IMAGES)
                    throw new Error('Images require user messages, at most eight per conversation');
                const blob = await readImage(image);
                total += blob.size;
                if (total > MAX_IMAGE_BYTES) throw new Error('Images exceed the 32 MiB conversation limit');
                module.FS.mkdirTree('/images');
                const path = `/images/${files.length}.${blob.type === 'image/png' ? 'png' : 'jpg'}`;
                module.FS.writeFile(path, new Uint8Array(await blob.arrayBuffer()));
                files.push(path);
                images.push(path);
            }
            staged.push({role: message.role, content: message.content, images});
        }
        return {messages: staged, dispose};
    } catch (error) { dispose(); throw error; }
}