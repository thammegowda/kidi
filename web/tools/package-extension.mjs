import {cp, copyFile, mkdir, rm} from 'node:fs/promises';
import {spawnSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';
import {join, resolve} from 'node:path';

import {packageSite} from './package-site.mjs';

const root = fileURLToPath(new URL('../../', import.meta.url));
const extensionAssets = [
    ['web/extension/background.mjs', 'background.mjs'],
    ['web/extension/editor.html', 'editor.html'],
    ['web/extension/editor.mjs', 'editor.mjs'],
    ['web/extension/editor.css', 'editor.css'],
    ['web/extension/progress.mjs', 'progress.mjs'],
    ['web/extension/gmail-bridge.js', 'gmail-bridge.js'],
    ['web/extension/article-bridge.js', 'article-bridge.js'],
    ['third_party/liteapp/web/extension/selection-bridge.js', 'selection-bridge.js'],
];

export async function packageExtensions(outputRoot, runtimes, {
    distribution,
    archive = Boolean(distribution),
} = {}) {
    const resolvedOutput = resolve(outputRoot);
    const base = join(resolvedOutput, 'base');
    await rm(resolvedOutput, {recursive: true, force: true});
    await mkdir(resolvedOutput, {recursive: true});
    await packageSite(base, runtimes, {isolationServiceWorker: false});
    for (const [source, destination] of extensionAssets)
        await copyFile(join(root, source), join(base, destination));
    await cp(join(root, 'web/libs/readability'), join(base, 'libs/readability'), {recursive: true});

    if (distribution) await mkdir(resolve(distribution), {recursive: true});
    const packages = {};
    for (const target of ['chromium', 'firefox']) {
        const destination = join(resolvedOutput, target);
        await cp(base, destination, {recursive: true});
        await copyFile(join(root, 'web/extension/manifests', `${target}.json`), join(destination, 'manifest.json'));
        let archivePath;
        if (archive) {
            archivePath = join(resolve(distribution), `kidi-${target}.zip`);
            await rm(archivePath, {force: true});
            const result = spawnSync('cmake', ['-E', 'tar', 'cf', archivePath, '--format=zip', '.'],
                {cwd: destination, stdio: 'inherit'});
            if (result.error || result.status !== 0)
                throw new Error(result.error?.message || `Unable to archive ${target} extension`);
        }
        packages[target] = {directory: destination, archive: archivePath};
    }
    await rm(base, {recursive: true, force: true});
    return packages;
}
