import {createHash} from 'node:crypto';
import {mkdir, open, readFile, stat, writeFile} from 'node:fs/promises';
import {basename, join} from 'node:path';

const CHUNK_BYTES = 256 * 1024 * 1024;

async function packageFile(source, destination, name) {
    const information = await stat(source);
    const file = await open(source, 'r');
    const chunks = [];
    try {
        for (let offset = 0; offset < information.size; offset += CHUNK_BYTES) {
            const size = Math.min(CHUNK_BYTES, information.size - offset);
            const bytes = Buffer.allocUnsafe(size);
            const {bytesRead} = await file.read(bytes, 0, size, offset);
            if (bytesRead !== size) throw new Error(`Short read while packaging ${source}`);
            const sha256 = createHash('sha256').update(bytes).digest('hex');
            const name = `chunks/${sha256}.bin`;
            await writeFile(join(destination, name), bytes, {flag: 'wx'}).catch(error => {
                if (error.code !== 'EEXIST') throw error;
            });
            chunks.push({size, sha256, url: name});
        }
    } finally {
        await file.close();
    }
    return {name, size: information.size, chunks};
}

export async function packageModel(source, destination, files, metadata = {}) {
    await mkdir(join(destination, 'chunks'), {recursive: true});
    const packaged = [];
    for (const name of files) packaged.push(await packageFile(join(source, name), destination, name));
    const identity = createHash('sha256');
    for (const file of packaged) {
        identity.update(file.name);
        identity.update(String(file.size));
        for (const chunk of file.chunks) identity.update(chunk.sha256);
    }
    const manifest = {
        version: 1,
        id: identity.digest('hex'),
        name: metadata.name || basename(source),
        precision: metadata.precision || 'checkpoint',
        files: packaged,
    };
    await writeFile(join(destination, 'manifest.json'), `${JSON.stringify(manifest, null, 2)}\n`);
    return manifest;
}
