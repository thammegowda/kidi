import {createServer} from 'node:http';
import {createHash} from 'node:crypto';
import {open, readFile, readdir, realpath, stat} from 'node:fs/promises';
import {homedir} from 'node:os';
import {join, relative, resolve, sep} from 'node:path';
import {fileURLToPath, pathToFileURL} from 'node:url';

const SAFE_NAME = /^[A-Za-z0-9_.-]+$/u;
const REVISION = /^[a-f0-9]{40}$/u;
const headers = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'Range',
  'Access-Control-Allow-Methods': 'GET, HEAD, OPTIONS',
  'Access-Control-Expose-Headers': 'Accept-Ranges, Content-Length, Content-Range',
  'Cross-Origin-Resource-Policy': 'cross-origin',
  'Cache-Control': 'no-store',
};

const contentType = name => name.endsWith('.json') ? 'application/json' :
  /\.(?:jinja|md|txt|tsv)$/u.test(name) || name === 'NOTICE' ? 'text/plain; charset=utf-8' :
    'application/octet-stream';

function send(response, status, body = '', extra = {}) {
  response.writeHead(status, {...headers, ...extra});
  response.end(body);
}

function safeSegments(pathname) {
  const values = pathname.split('/').filter(Boolean).map(value => decodeURIComponent(value));
  return values.every(value => SAFE_NAME.test(value)) ? values : null;
}

async function inside(repository, path) {
  const root = await realpath(repository);
  const file = await realpath(path);
  const pathFromRoot = relative(root, file);
  return file === root || !pathFromRoot.startsWith(`..${sep}`) && pathFromRoot !== '..' ? file : null;
}

async function snapshotRevision(snapshot) {
  const entries = [];
  const walk = async (directory, prefix = '') => {
    for (const entry of (await readdir(directory, {withFileTypes: true}))
      .filter(value => value.name !== '.cache').sort((left, right) => left.name.localeCompare(right.name))) {
      const name = prefix ? `${prefix}/${entry.name}` : entry.name;
      const path = join(directory, entry.name);
      if (entry.isDirectory()) await walk(path, name);
      else {
        const details = await stat(path);
        if (details.isFile()) entries.push(`${name}\0${details.size}\0${details.mtimeMs}`);
      }
    }
  };
  await walk(snapshot);
  if (!entries.length) throw new Error('Local model snapshot is empty');
  return createHash('sha1').update(entries.join('\n')).digest('hex');
}

async function mainRevision(repository) {
  const reference = (await readFile(join(repository, 'refs/main'), 'utf8')).trim();
  if (REVISION.test(reference))
    return {revision: reference, snapshot: join(repository, 'snapshots', reference)};
  if (!SAFE_NAME.test(reference)) throw new Error('Invalid local model reference');
  const local = join(repository, 'snapshots', reference);
  try {
    const metadata = await readFile(join(local, '.cache/huggingface/download/config.json.metadata'), 'utf8');
    const revision = metadata.split(/\r?\n/u)[0].trim();
    if (REVISION.test(revision))
      return {revision, snapshot: join(repository, 'snapshots', revision)};
  } catch (error) {
    if (error?.code !== 'ENOENT') throw error;
  }
  return {revision: await snapshotRevision(local), snapshot: local};
}

export async function createModelCacheServer({
  root = join(homedir(), '.cache/kidi/model-hub'),
  host = '127.0.0.1',
  port = 0,
} = {}) {
  if (host !== '127.0.0.1') throw new Error('The local model bridge only binds to 127.0.0.1');
  const cache = await realpath(resolve(root));
  const server = createServer(async (request, response) => {
    try {
      if (request.method === 'OPTIONS') return send(response, 204);
      if (!['GET', 'HEAD'].includes(request.method || '')) return send(response, 405);
      const url = new URL(request.url, `http://${host}`);
      if (url.pathname === '/__kidi_health__')
        return send(response, 200, request.method === 'HEAD' ? '' : JSON.stringify({root: cache}),
          {'Content-Type': 'application/json'});
      const parts = safeSegments(url.pathname);
      if (!parts) return send(response, 400);
      let repository, path;
      if (parts.length === 6 && parts[0] === 'api' && parts[1] === 'models' &&
          parts[4] === 'revision' && parts[5] === 'main') {
        repository = join(cache, `models--${parts[2]}--${parts[3]}`);
        const {revision} = await mainRevision(repository);
        const body = JSON.stringify({sha: revision});
        return send(response, 200, request.method === 'HEAD' ? '' : body, {
          'Content-Type': 'application/json', 'Content-Length': String(Buffer.byteLength(body)),
        });
      }
      if (parts.length < 5 || parts[2] !== 'resolve' || !REVISION.test(parts[3]))
        return send(response, 404);
      repository = join(cache, `models--${parts[0]}--${parts[1]}`);
      let snapshot = join(repository, 'snapshots', parts[3]);
      try {
        if (!(await stat(snapshot)).isDirectory()) return send(response, 404);
      } catch (error) {
        if (error?.code !== 'ENOENT') throw error;
        const resolved = await mainRevision(repository);
        if (resolved.revision !== parts[3]) return send(response, 404);
        snapshot = resolved.snapshot;
      }
      path = join(snapshot, ...parts.slice(4));
      const file = await inside(cache, path);
      if (!file) return send(response, 403);
      const details = await stat(file);
      if (!details.isFile()) return send(response, 404);
      const range = request.headers.range;
      let start = 0, end = details.size - 1, status = 200;
      if (range) {
        const match = /^bytes=(\d+)-(\d*)$/u.exec(range);
        if (!match) return send(response, 416, '', {'Content-Range': `bytes */${details.size}`});
        start = Number(match[1]);
        end = match[2] ? Number(match[2]) : end;
        if (!Number.isSafeInteger(start) || !Number.isSafeInteger(end) || start > end || start >= details.size)
          return send(response, 416, '', {'Content-Range': `bytes */${details.size}`});
        end = Math.min(end, details.size - 1);
        status = 206;
      }
      const length = end - start + 1;
      const extra = {
        'Accept-Ranges': 'bytes',
        'Content-Length': String(length),
        'Content-Type': contentType(file),
        ...(status === 206 ? {'Content-Range': `bytes ${start}-${end}/${details.size}`} : {}),
      };
      if (request.method === 'HEAD') return send(response, status, '', extra);
      const handle = await open(file, 'r');
      try {
        const bytes = Buffer.allocUnsafe(length);
        const {bytesRead} = await handle.read(bytes, 0, length, start);
        if (bytesRead !== length) return send(response, 500);
        return send(response, status, bytes, extra);
      } finally {
        await handle.close();
      }
    } catch (error) {
      if (error?.code === 'ENOENT') return send(response, 404);
      console.error('Local model bridge request failed:', error);
      return send(response, 500);
    }
  });
  await new Promise((resolveListen, reject) => {
    server.once('error', reject);
    server.listen(port, host, resolveListen);
  });
  const address = server.address();
  return {
    server,
    root: cache,
    url: `http://${host}:${address.port}`,
    close: () => new Promise((resolveClose, reject) =>
      server.close(error => error ? reject(error) : resolveClose())),
  };
}

function argument(name, fallback) {
  const index = process.argv.indexOf(name);
  return index < 0 ? fallback : process.argv[index + 1];
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  const bridge = await createModelCacheServer({
    root: argument('--root', join(homedir(), '.cache/kidi/model-hub')),
    host: argument('--host', '127.0.0.1'),
    port: Number(argument('--port', '8765')),
  });
  console.log(JSON.stringify({url: bridge.url, root: bridge.root}));
  const close = async () => {
    await bridge.close();
    process.exit(0);
  };
  process.once('SIGINT', close);
  process.once('SIGTERM', close);
}
