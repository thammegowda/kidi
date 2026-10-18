const CACHE = 'kidi-ebooks-v1';
export const MAX_EBOOK_BYTES = 64 * 1024 * 1024;

const ebookKey = id => {
  if (!/^[a-f0-9]{64}$/.test(id)) throw new Error('Invalid ebook identifier');
  return new URL(`./__ebooks__/${id}.epub`, import.meta.url).href;
};

export async function storeEbook(blob, name) {
  if (!(blob instanceof Blob) || !blob.size || blob.size > MAX_EBOOK_BYTES ||
      !String(name || '').toLowerCase().endsWith('.epub'))
    throw new Error('Choose an EPUB file no larger than 64 MiB');
  const digest = await crypto.subtle.digest('SHA-256', await blob.arrayBuffer());
  const id = Array.from(new Uint8Array(digest), byte => byte.toString(16).padStart(2, '0')).join('');
  await (await caches.open(CACHE)).put(ebookKey(id), new Response(blob, {
    headers: {'Content-Type': 'application/epub+zip'},
  }));
  return {id, name: String(name).slice(0, 256), size: blob.size};
}

export async function readEbook(book) {
  const response = await (await caches.open(CACHE)).match(ebookKey(book.id));
  if (!response) throw new Error(`Book unavailable: ${book.title || book.name || 'EPUB'}. Add it again.`);
  const blob = await response.blob();
  if (!blob.size || blob.size > MAX_EBOOK_BYTES) throw new Error('The cached EPUB is invalid');
  return blob;
}

export async function removeEbook(book) {
  await (await caches.open(CACHE)).delete(ebookKey(book.id));
}
