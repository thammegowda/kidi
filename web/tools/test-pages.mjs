import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {createServer} from 'node:http';
import {resolve, extname, sep} from 'node:path';
import {fileURLToPath} from 'node:url';
import {chromium} from 'playwright';

const directory = resolve(process.argv[2] || fileURLToPath(new URL('../../build-web', import.meta.url)));
const prefix = '/kidi/';
const types = {
    '.html': 'text/html',
    '.js': 'text/javascript',
    '.mjs': 'text/javascript',
    '.css': 'text/css',
    '.wasm': 'application/wasm',
    '.svg': 'image/svg+xml',
    '.png': 'image/png',
    '.yaml': 'text/yaml',
    '.tsv': 'text/tab-separated-values',
};

test('Pages artifact starts under a repository subpath without server isolation headers', {timeout: 120000}, async () => {
    const failures = new Set();
    const server = createServer(async (request, response) => {
        const pathname = decodeURIComponent(new URL(request.url, 'http://localhost').pathname);
        const file = resolve(directory, pathname.slice(prefix.length) || 'index.html');
        if (!pathname.startsWith(prefix) || !file.startsWith(`${directory}${sep}`)) {
            response.writeHead(404).end();
            return;
        }
        try {
            const content = await readFile(file);
            response.writeHead(200, {
                'Content-Type': types[extname(file)] || 'application/octet-stream',
                'Cache-Control': 'no-store',
            }).end(content);
        } catch (error) {
            if (error.code !== 'ENOENT') failures.add(`Static server: ${error.message}`);
            response.writeHead(error.code === 'ENOENT' ? 404 : 500).end();
        }
    });
    await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
    let browser;
    try {
        browser = await chromium.launch();
        const context = await browser.newContext();
        const page = await context.newPage();
        const origin = `http://127.0.0.1:${server.address().port}`;
        page.on('pageerror', error => failures.add(`JavaScript: ${error.message}`));
        page.on('console', message => {
            if (message.type() === 'error') failures.add(`Console: ${message.text()}`);
        });
        page.on('response', response => {
            if (response.status() >= 400) failures.add(`HTTP ${response.status()}: ${response.url()}`);
        });
        page.on('request', request => {
            if (new URL(request.url()).origin !== origin)
                failures.add(`Unexpected external request: ${request.url()}`);
        });
        page.on('requestfailed', request => {
            // The isolation helper deliberately reloads the first navigation.
            if (request.failure()?.errorText !== 'net::ERR_ABORTED')
                failures.add(`Request: ${request.url()}: ${request.failure()?.errorText}`);
        });
        await context.route('**/*', route => {
            if (new URL(route.request().url()).origin === origin) return route.continue();
            return route.abort();
        });
        const response = await page.goto(`${origin}${prefix}`);
        assert.equal(response.headers()['cross-origin-opener-policy'], undefined);
        await page.waitForFunction(() => crossOriginIsolated && navigator.serviceWorker.controller);
        await page.getByRole('button', {name: 'Settings', exact: true}).click();
        await page.locator('#wasm-preferences').waitFor({state: 'visible'});
        assert.ok(await page.locator('#wasm-model').inputValue());
        await page.getByRole('button', {name: 'Close configuration'}).click();
        await page.getByRole('button', {name: 'Reader', exact: true}).click();
        assert.equal(new URL(page.url()).hash, '#/reader');
        await page.getByRole('button', {name: 'Chat', exact: true}).click();
        assert.equal(new URL(page.url()).hash, '#/chat');

        const status = await page.evaluate(async () => {
            const {MessageTransport, RequestChannel} = await import(new URL('./liteapp/liteapp.mjs', location.href));
            const worker = new Worker(new URL('./gpu-runtime-worker.mjs', location.href), {type: 'module'});
            const channel = new RequestChannel(new MessageTransport(worker));
            worker.onerror = event => channel.dispose(event.message);
            try {
                return await channel.request('runtime.status', {}, {signal: AbortSignal.timeout(10000)});
            } finally {
                channel.dispose();
                worker.terminate();
            }
        });
        assert.equal(status.type, 'runtime-status');
        assert.equal(status.logicalSessions, 0);
        assert.deepEqual(status.memory, {gpu: 0, host: 0});
        assert.deepEqual(status.models, []);

        for (const variant of ['threads', 'gpu']) {
            const result = await page.evaluate(async variant => {
                const source = `
                    import create from ${JSON.stringify(new URL(`./${variant}/kidi.mjs`, location.href).href)};
                    try {
                        const module = await create();
                        const configured = JSON.parse(module.ccall('kidi_configure', 'string', ['number'],
                            [${variant === 'threads' ? 2 : 1}]));
                        const memory = JSON.parse(module.ccall('kidi_memory_stats', 'string', [], []));
                        self.postMessage({configured, memory});
                    } catch (error) {
                        self.postMessage({error: error.stack || error.message});
                    }`;
                const url = URL.createObjectURL(new Blob([source], {type: 'text/javascript'}));
                const worker = new Worker(url, {type: 'module'});
                try {
                    return await new Promise((resolve, reject) => {
                        const timeout = setTimeout(() => reject(new Error(`${variant} startup timed out`)), 30000);
                        worker.onmessage = ({data}) => { clearTimeout(timeout); resolve(data); };
                        worker.onerror = event => { clearTimeout(timeout); reject(new Error(event.message)); };
                    });
                } finally {
                    worker.terminate();
                    URL.revokeObjectURL(url);
                }
            }, variant);
            assert.equal(result.error, undefined, `${variant}: ${result.error}`);
            assert.equal(result.configured.configured, true, `${variant} configuration`);
            assert.equal(result.configured.threads, variant === 'threads' ? 2 : 1);
            assert.ok(result.memory.heap_bytes > 0, `${variant} native heap`);
            assert.equal(result.memory.error, undefined, `${variant} memory statistics`);
        }
        assert.deepEqual([...failures], [], 'The deployed site must not have JavaScript, network, or asset errors');
    } catch (error) {
        if (failures.size)
            throw new Error(`${error.message}\nArtifact errors:\n${[...failures].join('\n')}`, {cause: error});
        throw error;
    } finally {
        await browser?.close();
        await new Promise(resolve => server.close(resolve));
    }
});
