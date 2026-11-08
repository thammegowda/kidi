import {spawnSync} from 'node:child_process';
import {createHash} from 'node:crypto';
import {access, cp, mkdir, readFile, readdir, rename, rm, rmdir, writeFile} from 'node:fs/promises';
import {join, relative, resolve} from 'node:path';
import {setTimeout} from 'node:timers/promises';
import {fileURLToPath} from 'node:url';

const script = fileURLToPath(import.meta.url);
const web = fileURLToPath(new URL('../', import.meta.url));
const libraries = join(web, 'libs');
const modules = join(web, 'node_modules');
const lock = join(libraries, '.acquiring');
const assets = [
    ['mermaid', 'mermaid', {'dist/mermaid.min.js': 'mermaid.min.js', LICENSE: 'LICENSE'}],
    ['pdfjs', 'pdfjs-dist', {
        'legacy/build/pdf.min.mjs': 'pdf.min.mjs', 'legacy/build/pdf.worker.min.mjs': 'pdf.worker.min.mjs',
        cmaps: 'cmaps', standard_fonts: 'standard_fonts', LICENSE: 'LICENSE',
    }],
];

async function readOptional(path) {
    try { return await readFile(path, 'utf8'); }
    catch (error) {
        if (error.code !== 'ENOENT') throw error;
        return null;
    }
}

async function currentAssets(destination, fingerprint) {
    const source = await readOptional(join(destination, '.stamp'));
    if (!source) return false;
    const stamp = JSON.parse(source);
    if (stamp.fingerprint !== fingerprint) return false;
    try {
        await Promise.all(stamp.files.map(file => access(join(destination, file))));
        return true;
    } catch (error) {
        if (error.code !== 'ENOENT') throw error;
        return false;
    }
}

async function currentInstall(installed, fingerprint, pins) {
    if (await readOptional(installed) !== fingerprint) return false;
    for (const [name, version] of Object.entries({...pins.dependencies, ...pins.devDependencies})) {
        const source = await readOptional(join(modules, name, 'package.json'));
        if (!source || JSON.parse(source).version !== version) return false;
    }
    return true;
}

export async function prepareWebLibraries(destination = libraries) {
    destination = resolve(destination);
    const manifest = await readFile(join(web, 'package.json'));
    const lockfile = await readFile(join(web, 'package-lock.json'));
    const pins = JSON.parse(manifest);
    const installHash = createHash('sha256')
        .update(JSON.stringify([pins.dependencies, pins.devDependencies, pins.engines])).update(lockfile).digest('hex');
    const fingerprint = createHash('sha256').update(installHash).update(await readFile(script)).digest('hex');
    const installed = join(modules, '.kidi-lock');
    if (await currentInstall(installed, installHash, pins) && await currentAssets(destination, fingerprint)) return;
    await mkdir(libraries, {recursive: true});
    const deadline = Date.now() + 300000;
    while (true) {
        try { await mkdir(lock); break; }
        catch (error) {
            if (error.code !== 'EEXIST') throw error;
            if (Date.now() >= deadline)
                throw new Error(`Timed out waiting for web dependencies. Check for an interrupted build at ${lock}.`);
            await setTimeout(250);
        }
    }
    const stamp = join(destination, '.stamp');
    try {
        if (await currentInstall(installed, installHash, pins) && await currentAssets(destination, fingerprint)) return;
        if (!await currentInstall(installed, installHash, pins)) {
            const result = spawnSync(process.platform === 'win32' ? 'npm.cmd' : 'npm',
                ['ci', '--include=dev', '--ignore-scripts', '--no-audit', '--no-fund'],
                {cwd: web, stdio: 'inherit', shell: process.platform === 'win32'});
            if (result.error) throw result.error;
            if (result.status !== 0)
                throw new Error(`Locked web dependency acquisition failed (${result.signal || result.status}).`);
            await writeFile(installed, installHash);
        }
        const files = [];
        for (const [directory, packageName, mappings] of assets) {
            const source = join(modules, packageName);
            const metadata = JSON.parse(await readFile(join(source, 'package.json'), 'utf8'));
            const version = pins.dependencies[packageName] || pins.devDependencies[packageName];
            if (metadata.version !== version)
                throw new Error(`${packageName} does not match its ${version} pin. Run npm ci --prefix web.`);
            const target = join(destination, directory);
            await rm(target, {recursive: true, force: true});
            await mkdir(target, {recursive: true});
            for (const [input, output] of Object.entries(mappings))
                await cp(join(source, input), join(target, output), {recursive: true});
            for (const entry of await readdir(target, {recursive: true, withFileTypes: true}))
                if (entry.isFile()) files.push(relative(destination, join(entry.parentPath, entry.name)));
        }
        await writeFile(`${stamp}.tmp`, JSON.stringify({fingerprint, files}) + '\n');
        await rename(`${stamp}.tmp`, stamp);
    } finally {
        try { await rm(`${stamp}.tmp`, {force: true}); }
        finally { await rmdir(lock); }
    }
}

if (process.argv[1] && resolve(process.argv[1]) === script) await prepareWebLibraries(process.argv[2]);
