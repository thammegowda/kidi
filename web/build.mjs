import {spawnSync} from 'node:child_process';
import {mkdir, copyFile, readdir, readFile, writeFile} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
import {resolve, join} from 'node:path';
import {unsignedHeapIndices} from './wasm-glue.mjs';

const root = fileURLToPath(new URL('../', import.meta.url));
const libraries = join(root, 'src/web/libs');
const destination = resolve(process.argv[2] || join(root, 'build-web'));
for (const [name, threads, gpu] of [['single', 'OFF', 'OFF'], ['threads', 'ON', 'OFF'], ['gpu', 'OFF', 'ON']]) {
    const build = join(root, name === 'single' ? 'build-wasm' : name === 'gpu' ? 'build-webgpu' : 'build-wasm-threads');
    for (const [command, args] of [
        ['emcmake', ['cmake', '-S', root, '-B', build, '-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release',
            '-DBUILD_TESTING=OFF', '-DKIDI_BUILD_TESTS=OFF',
            '-DKIDI_BUILD_BENCHMARKS=OFF', '-DKIDI_BUILD_PYTHON=OFF',
            '-DKIDI_WASM_LARGE_MEMORY=ON', '-DCMAKE_EXE_LINKER_FLAGS=',
            `-DKIDI_WASM_THREADS=${threads}`, `-DKIDI_WASM_WEBGPU=${gpu}`]],
        ['cmake', ['--build', build, '--target', 'kidi_wasm', '-j8']]]) {
        const result = spawnSync(command, args, {stdio: 'inherit'});
        if (result.error || result.status !== 0) throw new Error(result.error?.message || `${command} failed`);
    }
    await mkdir(join(destination, name), {recursive: true});
    for (const file of await readdir(build))
        if (/^kidi.*\.(wasm|mjs|js)$/.test(file)) await copyFile(join(build, file), join(destination, name, file));
    const glue = join(destination, name, 'kidi.mjs');
    await writeFile(glue, unsignedHeapIndices(await readFile(glue, 'utf8')));
}
for (const file of ['index.html', 'app.mjs', 'style.css', 'inference-worker.mjs', 'model-cache.mjs', 'markdown.mjs', 'mermaid.mjs', 'webgpu.mjs', 'webgpu-kernels.mjs'])
    await copyFile(join(root, 'web', file), join(destination, file));
for (const name of ['marked', 'dompurify', 'highlightjs', 'mermaid']) {
    await mkdir(join(destination, 'libs', name), {recursive: true});
    for (const file of await readdir(join(libraries, name)))
        await copyFile(join(libraries, name, file), join(destination, 'libs', name, file));
}
await copyFile(join(libraries, 'coi-serviceworker/coi-serviceworker.js'),
    join(destination, 'coi-serviceworker.js'));
await copyFile(join(libraries, 'coi-serviceworker/LICENSE'),
    join(destination, 'coi-serviceworker.LICENSE.txt'));
await writeFile(join(destination, '.nojekyll'), '');
await mkdir(join(destination, 'icons'), {recursive: true});
await copyFile(join(libraries, 'lucide-static/LICENSE'), join(destination, 'icons/LICENSE.txt'));
for (const icon of ['send-horizontal', 'square', 'download', 'trash-2', 'plus', 'settings', 'x', 'panel-left',
    'message-square'])
    await copyFile(join(libraries, 'lucide-static/icons', `${icon}.svg`), join(destination, 'icons', `${icon}.svg`));
await copyFile(join(root, 'docs/kidi-logo-small.png'), join(destination, 'icons/kidi-logo.png'));
console.log(`Browser app: ${destination}`);