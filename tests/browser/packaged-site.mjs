import {rmSync} from 'node:fs';
import {mkdir, mkdtemp, writeFile} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {pathToFileURL} from 'node:url';
import {packageSite} from '../../web/tools/package-site.mjs';

// The Wasm UI imports LiteApp through site-relative paths (./liteapp/...), so tests import its modules from a
// packaged site with stub runtimes. The site is built once per test process.
let site;
export function packagedSite() {
    site ??= (async () => {
        const directory = await mkdtemp(join(tmpdir(), 'kidi-ui-site-'));
        process.on('exit', () => rmSync(directory, {recursive: true, force: true}));
        const runtimes = {};
        for (const name of ['threads', 'gpu']) {
            runtimes[name] = join(directory, name);
            await mkdir(runtimes[name]);
            await writeFile(join(runtimes[name], 'kidi.mjs'), 'export default () => {};');
            await writeFile(join(runtimes[name], 'kidi.wasm'), new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0]));
        }
        await packageSite(join(directory, 'site'), runtimes);
        return join(directory, 'site');
    })();
    return site;
}

export async function importSiteModule(file) {
    return import(pathToFileURL(join(await packagedSite(), file)));
}
