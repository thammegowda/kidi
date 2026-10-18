const MAX_ARCHIVE_BYTES = 64 * 1024 * 1024;
const MAX_FILES = 4096;
const MAX_FILE_BYTES = 4 * 1024 * 1024;
const MAX_TOTAL_BYTES = 128 * 1024 * 1024;
const MAX_CHAPTERS = 2048;
const decoder = new TextDecoder('utf-8', {fatal: true});

function fail(message) {
  throw new Error(`Invalid EPUB: ${message}`);
}

function u16(view, offset) {
  if (offset < 0 || offset + 2 > view.byteLength) fail('ZIP metadata is truncated');
  return view.getUint16(offset, true);
}

function u32(view, offset) {
  if (offset < 0 || offset + 4 > view.byteLength) fail('ZIP metadata is truncated');
  return view.getUint32(offset, true);
}

function normalizedEntryPath(value) {
  if (!value || value.startsWith('/') || value.includes('\\') || value.includes('\0')) return null;
  const directory = value.endsWith('/');
  const parts = value.split('/');
  if (directory) parts.pop();
  if (!parts.length || parts.some(part => !part || part === '.' || part === '..')) return null;
  return {path: parts.join('/'), directory};
}

function resolveEpubPath(base, reference) {
  const encoded = String(reference || '').split(/[?#]/u, 1)[0];
  let decoded;
  try {
    decoded = decodeURIComponent(encoded);
  } catch {
    return null;
  }
  if (!decoded || decoded.startsWith('/') || decoded.includes('\\') || decoded.includes('\0')) return null;
  const parts = base.split('/');
  parts.pop();
  for (const part of decoded.split('/')) {
    if (!part || part === '.') continue;
    if (part === '..') {
      if (!parts.length) return null;
      parts.pop();
    } else {
      parts.push(part);
    }
  }
  return parts.length ? parts.join('/') : null;
}

function xmlEntities(value) {
  return value.replace(/&(?:#(\d+)|#x([\da-f]+)|amp|lt|gt|quot|apos);/giu, entity => {
    if (entity === '&amp;') return '&';
    if (entity === '&lt;') return '<';
    if (entity === '&gt;') return '>';
    if (entity === '&quot;') return '"';
    if (entity === '&apos;') return "'";
    const decimal = entity.match(/^&#(\d+);$/u)?.[1];
    const hexadecimal = entity.match(/^&#x([\da-f]+);$/iu)?.[1];
    const codepoint = Number.parseInt(decimal || hexadecimal, decimal ? 10 : 16);
    if (!Number.isSafeInteger(codepoint) || codepoint < 0 || codepoint > 0x10ffff ||
        codepoint >= 0xd800 && codepoint <= 0xdfff)
      return entity;
    return String.fromCodePoint(codepoint);
  });
}

function normalizedText(value) {
  return xmlEntities(String(value || '').replace(/<[^>]*>/gu, ' ')).replace(/\s+/gu, ' ').trim();
}

function xmlTags(xml, name) {
  const expression = new RegExp(`<(?:[A-Za-z_][\\w.-]*:)?${name}\\b([^>]*)>`, 'giu');
  return [...xml.matchAll(expression)].map(match => match[1]);
}

function xmlAttribute(attributes, name) {
  const expression = new RegExp(`(?:^|\\s)(?:[A-Za-z_][\\w.-]*:)?${name}\\s*=\\s*(["'])(.*?)\\1`, 'iu');
  const match = attributes.match(expression);
  return match ? xmlEntities(match[2]) : null;
}

function xmlText(xml, name) {
  const expression = new RegExp(
    `<((?:[A-Za-z_][\\w.-]*:)?${name})\\b[^>]*>([\\s\\S]*?)<\\/\\1\\s*>`, 'iu');
  const match = xml.match(expression);
  const text = normalizedText(match?.[2]);
  return text || null;
}

function crcTable() {
  const table = new Uint32Array(256);
  for (let value = 0; value < table.length; ++value) {
    let crc = value;
    for (let bit = 0; bit < 8; ++bit) crc = crc >>> 1 ^ (crc & 1 ? 0xedb88320 : 0);
    table[value] = crc >>> 0;
  }
  return table;
}

const CRC_TABLE = crcTable();

function crc32(bytes) {
  let crc = 0xffffffff;
  for (const byte of bytes) crc = crc >>> 8 ^ CRC_TABLE[(crc ^ byte) & 0xff];
  return (crc ^ 0xffffffff) >>> 0;
}

async function inflateRaw(bytes) {
  if (typeof DecompressionStream !== 'function')
    throw new Error('This browser cannot decompress EPUB files');
  let stream;
  try {
    stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('deflate-raw'));
  } catch {
    throw new Error('This browser does not support EPUB deflate decompression');
  }
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

class ZipArchive {
  constructor(bytes, entries) {
    this.bytes = bytes;
    this.entries = entries;
  }

  static async open(blob) {
    if (!(blob instanceof Blob) || !blob.size || blob.size > MAX_ARCHIVE_BYTES)
      throw new Error('Choose an EPUB file no larger than 64 MiB');
    const bytes = new Uint8Array(await blob.arrayBuffer());
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const minimum = Math.max(0, bytes.length - 65557);
    let end = -1;
    for (let offset = bytes.length - 22; offset >= minimum; --offset) {
      if (u32(view, offset) === 0x06054b50) {
        end = offset;
        break;
      }
    }
    if (end < 0 || u16(view, end + 4) || u16(view, end + 6)) fail('multi-disk ZIP files are unsupported');
    const entriesCount = u16(view, end + 10);
    if (!entriesCount || entriesCount !== u16(view, end + 8) || entriesCount > MAX_FILES)
      fail('ZIP entry count is invalid');
    const centralSize = u32(view, end + 12);
    const centralOffset = u32(view, end + 16);
    if (centralOffset === 0xffffffff || centralSize === 0xffffffff ||
        centralOffset + centralSize > end)
      fail('ZIP64 or invalid central directories are unsupported');

    const entries = new Map();
    let total = 0;
    let cursor = centralOffset;
    for (let index = 0; index < entriesCount; ++index) {
      if (u32(view, cursor) !== 0x02014b50) fail('ZIP central directory is invalid');
      const flags = u16(view, cursor + 8);
      const method = u16(view, cursor + 10);
      const checksum = u32(view, cursor + 16);
      const compressedSize = u32(view, cursor + 20);
      const size = u32(view, cursor + 24);
      const nameLength = u16(view, cursor + 28);
      const extraLength = u16(view, cursor + 30);
      const commentLength = u16(view, cursor + 32);
      const localOffset = u32(view, cursor + 42);
      const next = cursor + 46 + nameLength + extraLength + commentLength;
      if (!nameLength || next > centralOffset + centralSize || localOffset === 0xffffffff)
        fail('ZIP entry metadata is invalid');
      let name;
      try {
        name = decoder.decode(bytes.subarray(cursor + 46, cursor + 46 + nameLength));
      } catch {
        fail('ZIP entry path is not UTF-8');
      }
      const normalized = normalizedEntryPath(name);
      if (!normalized) fail('ZIP contains an unsafe entry path');
      cursor = next;
      if (normalized.directory) continue;
      if (flags & 1 || ![0, 8].includes(method)) fail('ZIP contains an encrypted or unsupported entry');
      if (size > MAX_FILE_BYTES || compressedSize > MAX_ARCHIVE_BYTES || size > MAX_TOTAL_BYTES - total)
        fail('ZIP expanded data exceeds its safety limits');
      if (entries.has(normalized.path)) fail('ZIP contains duplicate entry paths');
      entries.set(normalized.path, {method, checksum, compressedSize, size, localOffset});
      total += size;
    }
    if (cursor !== centralOffset + centralSize) fail('ZIP central directory length is invalid');
    return new ZipArchive(bytes, entries);
  }

  async read(path) {
    const entry = this.entries.get(path);
    if (!entry) fail(`archive is missing ${path}`);
    const view = new DataView(this.bytes.buffer, this.bytes.byteOffset, this.bytes.byteLength);
    if (u32(view, entry.localOffset) !== 0x04034b50) fail('ZIP local header is invalid');
    const nameLength = u16(view, entry.localOffset + 26);
    const extraLength = u16(view, entry.localOffset + 28);
    const start = entry.localOffset + 30 + nameLength + extraLength;
    const end = start + entry.compressedSize;
    if (end > this.bytes.length) fail('ZIP entry data is truncated');
    const compressed = this.bytes.subarray(start, end);
    const contents = entry.method === 0 ? compressed.slice() : await inflateRaw(compressed);
    if (contents.length !== entry.size || crc32(contents) !== entry.checksum)
      fail('ZIP entry failed its size or checksum validation');
    return contents;
  }

  async text(path) {
    try {
      return decoder.decode(await this.read(path));
    } catch (error) {
      if (error instanceof TypeError) fail(`${path} is not UTF-8`);
      throw error;
    }
  }
}

function chapterTitle(html, index) {
  for (const name of ['h1', 'h2', 'h3', 'title']) {
    const title = xmlText(html, name);
    if (title) return title;
  }
  return `Chapter ${index + 1}`;
}

export async function openBrowserEpub(blob) {
  const archive = await ZipArchive.open(blob);
  const container = await archive.text('META-INF/container.xml');
  const rootfile = xmlTags(container, 'rootfile')[0];
  const packagePath = rootfile && xmlAttribute(rootfile, 'full-path');
  const normalizedPackage = packagePath && resolveEpubPath('container.xml', packagePath);
  if (!normalizedPackage) fail('container does not name a valid package document');
  const packageXml = await archive.text(normalizedPackage);

  const manifest = new Map();
  for (const item of xmlTags(packageXml, 'item')) {
    const id = xmlAttribute(item, 'id');
    const href = xmlAttribute(item, 'href');
    const mediaType = xmlAttribute(item, 'media-type');
    const path = href && resolveEpubPath(normalizedPackage, href);
    if (id && path && mediaType) manifest.set(id, {path, mediaType});
  }
  if (!manifest.size) fail('package manifest is empty');

  const chapters = [];
  for (const itemref of xmlTags(packageXml, 'itemref')) {
    const id = xmlAttribute(itemref, 'idref');
    const item = id && manifest.get(id);
    if (!item) fail('spine references a missing manifest item');
    if (!['application/xhtml+xml', 'text/html'].includes(item.mediaType)) continue;
    if (chapters.length >= MAX_CHAPTERS) fail('spine contains too many chapters');
    const html = await archive.text(item.path);
    chapters.push({id, title: chapterTitle(html, chapters.length), path: item.path});
  }
  if (!chapters.length) fail('spine does not contain readable chapters');

  const digest = await crypto.subtle.digest('SHA-256', archive.bytes);
  const bookId = Array.from(new Uint8Array(digest), byte => byte.toString(16).padStart(2, '0')).join('');
  const metadata = {
    book_id: bookId,
    title: xmlText(packageXml, 'title') || 'Untitled book',
    author: xmlText(packageXml, 'creator') || 'Unknown author',
    language: xmlText(packageXml, 'language') || 'und',
    chapters: chapters.map(({id, title}) => ({id, title})),
  };
  return {
    metadata,
    async chapter(index) {
      if (!Number.isSafeInteger(index) || index < 0 || index >= chapters.length)
        throw new Error('EPUB chapter index is out of range');
      return {
        book_id: bookId,
        chapter: index,
        title: chapters[index].title,
        html: await archive.text(chapters[index].path),
      };
    },
  };
}
