import test from 'node:test';
import assert from 'node:assert/strict';
import {deflateRawSync} from 'node:zlib';
import {openBrowserEpub} from '../../web/hosts/wasm/browser-epub.js';

const encoder = new TextEncoder();

function crc32(bytes) {
    const table = new Uint32Array(256);
    for (let value = 0; value < table.length; ++value) {
        let crc = value;
        for (let bit = 0; bit < 8; ++bit) crc = crc >>> 1 ^ (crc & 1 ? 0xedb88320 : 0);
        table[value] = crc >>> 0;
    }
    let crc = 0xffffffff;
    for (const byte of bytes) crc = crc >>> 8 ^ table[(crc ^ byte) & 0xff];
    return (crc ^ 0xffffffff) >>> 0;
}

function append16(output, value) {
    output.push(value & 0xff, value >>> 8 & 0xff);
}

function append32(output, value) {
    for (let shift = 0; shift < 32; shift += 8) output.push(value >>> shift & 0xff);
}

function zip(files) {
    const output = [];
    const central = [];
    for (const file of files) {
        const name = encoder.encode(file.name);
        const contents = encoder.encode(file.contents);
        const compressed = file.deflate ? new Uint8Array(deflateRawSync(contents)) : contents;
        const method = file.deflate ? 8 : 0;
        const checksum = crc32(contents);
        const offset = output.length;
        append32(output, 0x04034b50);
        append16(output, 20);
        append16(output, 0x800);
        append16(output, method);
        append16(output, 0);
        append16(output, 0);
        append32(output, checksum);
        append32(output, compressed.length);
        append32(output, contents.length);
        append16(output, name.length);
        append16(output, 0);
        output.push(...name, ...compressed);
        central.push({name, contents, compressed, method, checksum, offset});
    }
    const centralOffset = output.length;
    for (const file of central) {
        append32(output, 0x02014b50);
        append16(output, 20);
        append16(output, 20);
        append16(output, 0x800);
        append16(output, file.method);
        append16(output, 0);
        append16(output, 0);
        append32(output, file.checksum);
        append32(output, file.compressed.length);
        append32(output, file.contents.length);
        append16(output, file.name.length);
        append16(output, 0);
        append16(output, 0);
        append16(output, 0);
        append16(output, 0);
        append32(output, 0);
        append32(output, file.offset);
        output.push(...file.name);
    }
    const centralSize = output.length - centralOffset;
    append32(output, 0x06054b50);
    append16(output, 0);
    append16(output, 0);
    append16(output, central.length);
    append16(output, central.length);
    append32(output, centralSize);
    append32(output, centralOffset);
    append16(output, 0);
    return new Blob([Uint8Array.from(output)], {type: 'application/epub+zip'});
}

test('browser EPUB provider reads metadata and lazily decompresses spine chapters', async () => {
    const book = await openBrowserEpub(zip([
        {
            name: 'META-INF/container.xml',
            contents: '<container><rootfiles><rootfile full-path="OPS/content.opf"/></rootfiles></container>',
        },
        {
            name: 'OPS/content.opf',
            contents: `<package xmlns:dc="http://purl.org/dc/elements/1.1/">
                <metadata><dc:title>A &amp; B</dc:title><dc:creator>Test Author</dc:creator>
                <dc:language>en</dc:language></metadata>
                <manifest><item id="one" href="text/one.xhtml" media-type="application/xhtml+xml"/></manifest>
                <spine><itemref idref="one"/></spine></package>`,
        },
        {
            name: 'OPS/text/one.xhtml',
            contents: '<html><head><title>Fallback</title></head><body><h1>Opening</h1><p>Hello.</p></body></html>',
            deflate: true,
        },
    ]));
    assert.equal(book.metadata.title, 'A & B');
    assert.equal(book.metadata.author, 'Test Author');
    assert.equal(book.metadata.language, 'en');
    assert.deepEqual(book.metadata.chapters, [{id: 'one', title: 'Opening'}]);
    assert.match((await book.chapter(0)).html, /Hello\./);
    await assert.rejects(book.chapter(1), /out of range/);
});

test('browser EPUB provider rejects unsafe archive paths', async () => {
    await assert.rejects(openBrowserEpub(zip([
        {name: '../META-INF/container.xml', contents: '<container/>'},
    ])), /unsafe entry path/);
});
