import {spawnSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';
import {join, resolve} from 'node:path';

import {packageExtensions} from './package-extension.mjs';

const root = fileURLToPath(new URL('../../', import.meta.url));
const outputRoot = resolve(process.argv[2] || join(root, 'build-extension'));
const distribution = resolve(process.argv[3] || join(root, 'dist'));
const runtimeBuilds = {
    threads: join(root, 'build-wasm-threads'),
    gpu: join(root, 'build-webgpu'),
    single: join(root, 'build-wasm-single'),
};

function run(command, args, options = {}) {
    const result = spawnSync(command, args, {stdio: 'inherit', ...options});
    if (result.error || result.status !== 0)
        throw new Error(result.error?.message || `${command} failed with status ${result.status}`);
}

for (const [name, threads, gpu] of [
    ['threads', 'ON', 'OFF'],
    ['gpu', 'OFF', 'ON'],
    ['single', 'OFF', 'OFF'],
]) {
    const build = runtimeBuilds[name];
    run('emcmake', ['cmake', '-S', root, '-B', build, '-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release',
        '-DBUILD_TESTING=OFF', '-DKIDI_BUILD_TESTS=OFF', '-DKIDI_BUILD_BENCHMARKS=OFF',
        '-DKIDI_BUILD_PYTHON=OFF', '-DCMAKE_EXE_LINKER_FLAGS=',
        `-DKIDI_WASM_THREADS=${threads}`, `-DKIDI_WASM_WEBGPU=${gpu}`]);
    run('cmake', ['--build', build, '--target', 'kidi_wasm', '-j8']);
}

const packages = await packageExtensions(outputRoot, runtimeBuilds, {distribution});
for (const [target, output] of Object.entries(packages)) {
    console.log(`${target} extension: ${output.directory}`);
    console.log(`${target} package: ${output.archive}`);
}
