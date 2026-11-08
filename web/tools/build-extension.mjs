import {fileURLToPath} from 'node:url';
import {join, resolve} from 'node:path';

import {buildWasmRuntimes} from './build.mjs';
import {packageExtensions} from './package-extension.mjs';

const root = fileURLToPath(new URL('../../', import.meta.url));
const outputRoot = resolve(process.argv[2] || join(root, 'build-extension'));
const distribution = resolve(process.argv[3] || join(root, 'dist'));
const runtimeBuilds = await buildWasmRuntimes([
    ['threads', 'ON', 'OFF'],
    ['gpu', 'OFF', 'ON'],
    ['single', 'OFF', 'OFF'],
]);
const packages = await packageExtensions(outputRoot, runtimeBuilds, {distribution});
for (const [target, output] of Object.entries(packages)) {
    console.log(`${target} extension: ${output.directory}`);
    console.log(`${target} package: ${output.archive}`);
}
