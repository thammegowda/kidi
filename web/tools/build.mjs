import {spawnSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';
import {resolve, join} from 'node:path';
import {packageSite} from './package-site.mjs';

const root = fileURLToPath(new URL('../../', import.meta.url));
const destination = resolve(process.argv[2] || join(root, 'build-web'));
for (const [name, threads, gpu] of [
    ['threads', 'ON', 'OFF'],
    ['gpu', 'OFF', 'ON'],
]) {
    const build = join(root, name === 'gpu' ? 'build-webgpu' : 'build-wasm-threads');
    for (const [command, args] of [
        ['emcmake', ['cmake', '-S', root, '-B', build, '-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release',
            '-DBUILD_TESTING=OFF', '-DKIDI_BUILD_TESTS=OFF',
            '-DKIDI_BUILD_BENCHMARKS=OFF', '-DKIDI_BUILD_PYTHON=OFF',
            '-DCMAKE_EXE_LINKER_FLAGS=',
            `-DKIDI_WASM_THREADS=${threads}`, `-DKIDI_WASM_WEBGPU=${gpu}`]],
        ['cmake', ['--build', build, '--target', 'kidi_wasm', '-j8']]]) {
        const result = spawnSync(command, args, {stdio: 'inherit'});
        if (result.error || result.status !== 0) throw new Error(result.error?.message || `${command} failed`);
    }
}
await packageSite(destination);
console.log(`Browser app: ${destination}`);
