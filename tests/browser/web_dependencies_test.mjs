import assert from 'node:assert/strict';
import test from 'node:test';
import {createHash} from 'node:crypto';
import {access, mkdtemp, readFile, readdir, rm, stat} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {fileURLToPath} from 'node:url';
import {prepareWebLibraries} from '../../web/tools/prepare-libs.mjs';

const root = fileURLToPath(new URL('../../', import.meta.url));
const vendored = ['acorn', 'acorn-walk', 'coi-serviceworker', 'dompurify', 'highlightjs', 'js-yaml',
    'lucide-static', 'marked', 'readability'];

async function vendorSnapshot() {
    const files = {};
    for (const directory of vendored) {
        for (const entry of await readdir(join(root, 'web/libs', directory), {recursive: true, withFileTypes: true})) {
            if (!entry.isFile()) continue;
            const path = join(entry.parentPath, entry.name);
            files[path] = {
                hash: createHash('sha256').update(await readFile(path)).digest('hex'),
                mtime: (await stat(path)).mtimeMs,
            };
        }
    }
    return files;
}

test('acquisition generates only Mermaid and PDF.js without changing vendored files', async t => {
    const destination = await mkdtemp(join(tmpdir(), 'kidi-web-dependencies-'));
    t.after(() => rm(destination, {recursive: true, force: true}));
    const vendors = await vendorSnapshot();
    await prepareWebLibraries();
    await prepareWebLibraries(destination);
    assert.deepEqual(await vendorSnapshot(), vendors);
    assert.deepEqual((await readdir(destination, {withFileTypes: true}))
        .filter(entry => entry.isDirectory()).map(entry => entry.name).sort(), ['mermaid', 'pdfjs']);
    const stamp = JSON.parse(await readFile(join(destination, '.stamp'), 'utf8'));
    for (const file of stamp.files) await access(join(destination, file));
    for (const [output, source] of [
        ['mermaid/mermaid.min.js', 'mermaid/dist/mermaid.min.js'],
        ['mermaid/LICENSE', 'mermaid/LICENSE'],
        ['pdfjs/pdf.min.mjs', 'pdfjs-dist/legacy/build/pdf.min.mjs'],
        ['pdfjs/pdf.worker.min.mjs', 'pdfjs-dist/legacy/build/pdf.worker.min.mjs'],
        ['pdfjs/cmaps/Adobe-Japan1-UCS2.bcmap', 'pdfjs-dist/cmaps/Adobe-Japan1-UCS2.bcmap'],
        ['pdfjs/standard_fonts/FoxitSerif.pfb', 'pdfjs-dist/standard_fonts/FoxitSerif.pfb'],
        ['pdfjs/standard_fonts/LICENSE_FOXIT', 'pdfjs-dist/standard_fonts/LICENSE_FOXIT'],
        ['pdfjs/standard_fonts/LICENSE_LIBERATION', 'pdfjs-dist/standard_fonts/LICENSE_LIBERATION'],
    ]) assert.deepEqual(await readFile(join(destination, output)), await readFile(join(root, 'web/node_modules', source)));
    for (const directory of ['mermaid', 'pdfjs'])
        await access(join(destination, directory, 'LICENSE'));
    for (const directory of ['cmaps', 'standard_fonts']) {
        const expected = join(root, 'web/node_modules/pdfjs-dist', directory);
        const actual = join(destination, 'pdfjs', directory);
        const files = await readdir(expected);
        assert.deepEqual((await readdir(actual)).sort(), files.sort());
        for (const file of files)
            assert.deepEqual(await readFile(join(actual, file)), await readFile(join(expected, file)));
    }

    const asset = join(destination, 'pdfjs/pdf.worker.min.mjs');
    const prepared = await stat(asset);
    await prepareWebLibraries(destination);
    assert.equal((await stat(asset)).mtimeMs, prepared.mtimeMs, 'unchanged acquisition must not restage assets');
    await rm(join(destination, 'pdfjs/cmaps/Adobe-Japan1-UCS2.bcmap'));
    await prepareWebLibraries(destination);
    await access(join(destination, 'pdfjs/cmaps/Adobe-Japan1-UCS2.bcmap'));
    assert.deepEqual(await vendorSnapshot(), vendors);
});

test('desktop and Android dependency mappings retain packaged licenses', async () => {
    for (const manifest of ['desktop/assets.txt', 'android/assets.txt']) {
        const source = await readFile(join(root, manifest), 'utf8');
        const mappings = new Map(source.trim().split('\n').map(line => line.split('\t')));
        for (const path of ['/icons/LICENSE.txt', '/libs/marked/LICENSE', '/libs/dompurify/LICENSE',
            '/libs/dompurify/LICENSE-MPL', '/libs/highlightjs/LICENSE', '/libs/mermaid/LICENSE'])
            assert.ok(mappings.has(path), `${manifest} requires ${path}`);
    }
});
