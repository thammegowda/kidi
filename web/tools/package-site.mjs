import {mkdir, cp, copyFile, readdir, readFile, rm, writeFile} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
import {dirname, join, relative, sep} from 'node:path';
import {prepareWebLibraries} from './prepare-libs.mjs';
import {unsignedHeapIndices} from './wasm-glue.mjs';

const root = fileURLToPath(new URL('../../', import.meta.url));
const libraries = join(root, 'web/libs');

export async function packageSite(destination, runtimes = {
    threads: join(root, 'build-wasm-threads'),
    gpu: join(root, 'build-webgpu'),
}, {isolationServiceWorker = true} = {}) {
    await prepareWebLibraries();
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
        'webgpu.mjs', 'webgpu-kernels.mjs', 'scoped-model.mjs', 'runtime-scheduler.mjs', 'browser-epub.js'])
        await copyFile(join(root, 'web/hosts/wasm', file), join(destination, file));
    await writeFile(join(destination, 'app-config.mjs'),
        (await readFile(join(root, 'web/hosts/wasm/app-config.mjs'), 'utf8'))
            .replace('../../libs/js-yaml/', './libs/js-yaml/'));
    await copyFile(join(root, 'examples/kidi-app.yaml'), join(destination, 'app-config.yaml'));
    await copyFile(join(root, 'src/kidi/checkpoint/packages.jsonl'), join(destination, 'packages.jsonl'));
    await writeFile(join(destination, 'model-residency.mjs'),
        (await readFile(join(root, 'web/hosts/wasm/model-residency.mjs'), 'utf8'))
            .replace('../../ui/speech-controller.js', './speech-controller.js'));
    for (const file of ['runtime-client.mjs', 'gpu-runtime-worker.mjs']) {
        const source = await readFile(join(root, 'web/hosts/wasm', file), 'utf8');
        await writeFile(join(destination, file),
            source.replaceAll('../../../third_party/liteapp/web/', './liteapp/'));
    }
    // The browser app uses LiteApp's transports and Markdown renderer with its libraries; the IDE components, editor
    // libraries and the vendoring tool are not part of the site.
    const liteappWeb = join(root, 'third_party/liteapp/web');
    const unused = new Set(['ide', 'tools', 'libs/codemirror', 'libs/xterm']);
    await cp(liteappWeb, join(destination, 'liteapp'), {
        recursive: true,
        filter: source => {
            const parts = relative(liteappWeb, source).split(sep);
            return !unused.has(parts[0]) && !unused.has(parts.slice(0, 2).join('/'));
        },
    });
    const sharedAssets = (await readFile(join(root, 'web/ui/assets.txt'), 'utf8')).split(/\r?\n/).filter(Boolean);
    for (const file of sharedAssets) {
        const target = join(destination, file);
        await mkdir(dirname(target), {recursive: true});
        await copyFile(join(root, 'web/ui', file), target);
    }
    const sharedIndex = await readFile(join(root, 'web/ui/index.html'), 'utf8');
    await copyFile(join(root, 'benchmarks/browser/runtime_bench.mjs'), join(destination, 'runtime-bench.mjs'));
    const numericalChecks = await readFile(join(root, 'tests/browser/webgpu_numerical.mjs'), 'utf8');
    await writeFile(join(destination, 'webgpu-checks.mjs'),
        numericalChecks.replace('../../web/hosts/wasm/', './'));
    const integrationChecks = await readFile(join(root, 'tests/browser/gpu_runtime_integration.mjs'), 'utf8');
    await writeFile(join(destination, 'gpu-runtime-checks.mjs'),
        integrationChecks.replace('../../web/hosts/wasm/', './'));
    await writeFile(join(destination, 'index.html'), isolationServiceWorker
        ? sharedIndex.replace('<script type="module" src="./app.js"></script>',
            '<script src="./coi-serviceworker.js"></script>\n  <script type="module" src="./app.js"></script>')
        : sharedIndex);
    for (const name of ['mermaid', 'js-yaml']) {
        await mkdir(join(destination, 'libs', name), {recursive: true});
        for (const file of await readdir(join(libraries, name)))
            await copyFile(join(libraries, name, file), join(destination, 'libs', name, file));
    }
    if (isolationServiceWorker) {
        await copyFile(join(libraries, 'coi-serviceworker/coi-serviceworker.js'),
            join(destination, 'coi-serviceworker.js'));
        await copyFile(join(libraries, 'coi-serviceworker/LICENSE'),
            join(destination, 'coi-serviceworker.LICENSE.txt'));
    }
    await copyFile(join(root, 'src/kidi/checkpoint/ggml/LICENSE'), join(destination, 'ggml.LICENSE.txt'));
    await writeFile(join(destination, '.nojekyll'), '');
    await mkdir(join(destination, 'icons'), {recursive: true});
    await copyFile(join(libraries, 'lucide-static/LICENSE'), join(destination, 'icons/LICENSE.txt'));
    for (const icon of ['send-horizontal', 'square', 'download', 'trash-2', 'plus', 'settings', 'x', 'panel-left',
        'message-square', 'mic'])
        await copyFile(join(libraries, 'lucide-static/icons', `${icon}.svg`), join(destination, 'icons', `${icon}.svg`));
}
