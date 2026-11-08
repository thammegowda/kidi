import {spawn} from 'node:child_process';
import {availableParallelism} from 'node:os';
import {fileURLToPath} from 'node:url';
import {resolve, join} from 'node:path';
import {packageSite} from './package-site.mjs';
import {prepareWebLibraries} from './prepare-libs.mjs';

const script = fileURLToPath(import.meta.url);
const root = fileURLToPath(new URL('../../', import.meta.url));

function run(command, args) {
    return new Promise((resolveRun, reject) => {
        const child = spawn(command, args, {stdio: 'inherit', env: {...process.env, CMAKE_GENERATOR: 'Ninja'}});
        child.once('error', reject);
        child.once('exit', (code, signal) => {
            if (code === 0) resolveRun();
            else reject(new Error(`${command} failed (${signal || code})`));
        });
    });
}

export async function buildWasmRuntimes(variants = [
    ['threads', 'ON', 'OFF'],
    ['gpu', 'OFF', 'ON'],
]) {
    await prepareWebLibraries();
    const jobs = String(Math.max(2, Math.floor(availableParallelism() / variants.length)));
    const builds = {};
    await Promise.all(variants.map(async ([name, threads, gpu]) => {
        const build = join(root, name === 'gpu' ? 'build-webgpu' : `build-wasm-${name}`);
        builds[name] = build;
        await run('emcmake', ['cmake', '-S', root, '-B', build, '-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release',
            '-DBUILD_TESTING=OFF', '-DKIDI_BUILD_TESTS=OFF',
            '-DKIDI_BUILD_BENCHMARKS=OFF', '-DKIDI_BUILD_PYTHON=OFF',
            '-DCMAKE_EXE_LINKER_FLAGS=',
            `-DKIDI_WASM_THREADS=${threads}`, `-DKIDI_WASM_WEBGPU=${gpu}`]);
        await run('cmake', ['--build', build, '--target', 'kidi_wasm', `-j${jobs}`]);
    }));
    return builds;
}

if (process.argv[1] && resolve(process.argv[1]) === script) {
    const destination = resolve(process.argv[2] || join(root, 'build-web'));
    await buildWasmRuntimes();
    await packageSite(destination);
    console.log(`Browser app: ${destination}`);
}
