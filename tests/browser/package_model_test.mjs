import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp, mkdir, open, readFile, rm, stat} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {packageModel} from '../../web/tools/package-model.mjs';

test('model packaging splits weights at 256 MiB and retains small support files', async () => {
    const directory = await mkdtemp(join(tmpdir(), 'kidi-package-test-'));
    try {
        const source = join(directory, 'source');
        const destination = join(directory, 'package');
        await mkdir(source);
        const weights = await open(join(source, 'model.safetensors'), 'w');
        try { await weights.truncate(256 * 1024 * 1024 + 17); }
        finally { await weights.close(); }
        const config = await open(join(source, 'model.yaml'), 'w');
        try { await config.writeFile('model: omnivoice\n'); }
        finally { await config.close(); }
        const manifest = await packageModel(source, destination, ['model.safetensors', 'model.yaml']);
        assert.deepEqual(manifest.files[0].chunks.map(chunk => chunk.size), [256 * 1024 * 1024, 17]);
        assert.equal(manifest.files[1].chunks.length, 1);
        for (const file of manifest.files)
            for (const chunk of file.chunks)
                assert.equal((await stat(join(destination, chunk.url))).size, chunk.size);
        assert.deepEqual(JSON.parse(await readFile(join(destination, 'manifest.json'), 'utf8')), manifest);
    } finally {
        await rm(directory, {recursive: true, force: true});
    }
});
