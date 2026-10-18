import {mkdir, cp, copyFile, readdir, readFile, rm, writeFile} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
import {join, dirname} from 'node:path';
import {unsignedHeapIndices} from './wasm-glue.mjs';

const root = fileURLToPath(new URL('../../', import.meta.url));
const libraries = join(root, 'web/libs');

export async function packageSite(destination, runtimes = {
    threads: join(root, 'build-wasm-threads'),
    gpu: join(root, 'build-webgpu'),
}) {
    await rm(destination, {recursive: true, force: true});
    for (const [name, build] of Object.entries(runtimes)) {
        await mkdir(join(destination, name), {recursive: true});
        for (const file of await readdir(build)) {
            if (!/^kidi.*\.(wasm|mjs|js)$/.test(file)) continue;
            const target = join(destination, name, file);
            // Lowered builds index heap views with signed shifts, which break above 2 GiB.
            if (!file.endsWith('.wasm'))
                await writeFile(target, unsignedHeapIndices(await readFile(join(build, file), 'utf8')));
            else await copyFile(join(build, file), target);
        }
    }
    for (const file of ['backend.js', 'inference-worker.mjs', 'asr-worker.mjs',
        'tts-worker.mjs', 'speech.mjs', 'model-cache.mjs', 'voice-prompt-cache.mjs',
        'webgpu.mjs', 'webgpu-kernels.mjs', 'scoped-model.mjs', 'runtime-scheduler.mjs'])
        await copyFile(join(root, 'web/hosts/wasm', file), join(destination, file));
    await writeFile(join(destination, 'app-config.mjs'),
        (await readFile(join(root, 'web/hosts/wasm/app-config.mjs'), 'utf8'))
            .replace('../../libs/js-yaml/', './libs/js-yaml/'));
    await copyFile(join(root, 'examples/kidi-app.yaml'), join(destination, 'app-config.yaml'));
    await writeFile(join(destination, 'model-residency.mjs'),
        (await readFile(join(root, 'web/hosts/wasm/model-residency.mjs'), 'utf8'))
            .replace('../../ui/speech-controller.js', './speech-controller.js'));
    for (const file of ['runtime-client.mjs', 'gpu-runtime-worker.mjs']) {
        const source = await readFile(join(root, 'web/hosts/wasm', file), 'utf8');
        await writeFile(join(destination, file),
            source.replaceAll('../../../third_party/liteapp/web/', './liteapp/'));
    }
    await cp(join(root, 'third_party/liteapp/web'), join(destination, 'liteapp'), {recursive: true});
    const sharedAssets = (await readFile(join(root, 'web/ui/assets.txt'), 'utf8')).split(/\r?\n/).filter(Boolean);
    for (const file of sharedAssets) {
        const target = join(destination, file);
        await mkdir(dirname(target), {recursive: true});
        await copyFile(join(root, 'web/ui', file), target);
    }
    for (const file of ['capture-worklet.js', 'playback-worklet.js'])
        await copyFile(join(root, 'web/hosts/desktop', file), join(destination, file));
    const sharedIndex = await readFile(join(root, 'web/ui/index.html'), 'utf8');
    await copyFile(join(root, 'benchmarks/browser/runtime_bench.mjs'), join(destination, 'runtime-bench.mjs'));
    const numericalChecks = await readFile(join(root, 'tests/browser/webgpu_numerical.mjs'), 'utf8');
    await writeFile(join(destination, 'webgpu-checks.mjs'),
        numericalChecks.replace('../../web/hosts/wasm/', './'));
    const integrationChecks = await readFile(join(root, 'tests/browser/gpu_runtime_integration.mjs'), 'utf8');
    await writeFile(join(destination, 'gpu-runtime-checks.mjs'),
        integrationChecks.replace('../../web/hosts/wasm/', './'));
    await writeFile(join(destination, 'index.html'),
        sharedIndex.replace('<script type="module" src="./app.js"></script>',
            '<script src="./coi-serviceworker.js"></script>\n  <script type="module" src="./app.js"></script>'));
    for (const name of ['marked', 'dompurify', 'highlightjs', 'mermaid', 'js-yaml']) {
        await mkdir(join(destination, 'libs', name), {recursive: true});
        for (const file of await readdir(join(libraries, name)))
            await copyFile(join(libraries, name, file), join(destination, 'libs', name, file));
    }
    await copyFile(join(libraries, 'coi-serviceworker/coi-serviceworker.js'),
        join(destination, 'coi-serviceworker.js'));
    await copyFile(join(libraries, 'coi-serviceworker/LICENSE'),
        join(destination, 'coi-serviceworker.LICENSE.txt'));
    await copyFile(join(root, 'src/kidi/checkpoint/ggml/LICENSE'), join(destination, 'ggml.LICENSE.txt'));
    await writeFile(join(destination, '.nojekyll'), '');
    await mkdir(join(destination, 'icons'), {recursive: true});
    await copyFile(join(libraries, 'lucide-static/LICENSE'), join(destination, 'icons/LICENSE.txt'));
    for (const icon of ['send-horizontal', 'square', 'download', 'trash-2', 'plus', 'settings', 'x', 'panel-left',
        'message-square', 'mic'])
        await copyFile(join(libraries, 'lucide-static/icons', `${icon}.svg`), join(destination, 'icons', `${icon}.svg`));
}
