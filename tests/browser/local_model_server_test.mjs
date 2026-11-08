import assert from 'node:assert/strict';
import test from 'node:test';
import {mkdir, mkdtemp, realpath, rm, symlink, writeFile} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';

import {createModelCacheServer} from '../../web/tools/serve-model-cache.mjs';

test('loopback model bridge exposes only cached Hub revisions with byte ranges', async t => {
  const root = await mkdtemp(join(tmpdir(), 'kidi-model-bridge-test-'));
  const repository = join(root, 'models--owner--model');
  const revision = 'a'.repeat(40);
  const snapshot = join(repository, 'snapshots', revision);
  const localMetadata = join(repository, 'snapshots/local/.cache/huggingface/download');
  const blob = join(root, 'blobs', 'weights');
  await mkdir(join(repository, 'refs'), {recursive: true});
  await mkdir(snapshot, {recursive: true});
  await mkdir(localMetadata, {recursive: true});
  await mkdir(join(root, 'blobs'), {recursive: true});
  await writeFile(join(repository, 'refs/main'), 'local\n');
  await writeFile(join(localMetadata, 'config.json.metadata'), `${revision}\nlocal-etag\n0\n`);
  await writeFile(join(snapshot, 'config.json'), '{"model_type":"gemma4"}');
  await writeFile(blob, Uint8Array.from({length: 32}, (_, index) => index));
  await symlink('../../../blobs/weights', join(snapshot, 'model.safetensors'));
  const bridge = await createModelCacheServer({root});
  t.after(async () => {
    await bridge.close();
    await rm(root, {recursive: true, force: true});
  });
  const health = await fetch(`${bridge.url}/__kidi_health__`);
  assert.equal(health.status, 200);
  assert.equal((await health.json()).root, await realpath(root));
  assert.equal(health.headers.get('cross-origin-resource-policy'), 'cross-origin');
  const resolved = await fetch(`${bridge.url}/api/models/owner/model/revision/main`);
  assert.deepEqual(await resolved.json(), {sha: revision});
  const config = await fetch(`${bridge.url}/owner/model/resolve/${revision}/config.json`);
  assert.equal(await config.text(), '{"model_type":"gemma4"}');
  const ranged = await fetch(`${bridge.url}/owner/model/resolve/${revision}/model.safetensors`,
    {headers: {Range: 'bytes=5-11'}});
  assert.equal(ranged.status, 206);
  assert.equal(ranged.headers.get('content-range'), 'bytes 5-11/32');
  assert.deepEqual([...new Uint8Array(await ranged.arrayBuffer())], [5, 6, 7, 8, 9, 10, 11]);
  assert.equal((await fetch(`${bridge.url}/owner/model/resolve/${revision}/model.safetensors`,
    {headers: {Range: 'bytes=32-40'}})).status, 416);
  assert.equal((await fetch(`${bridge.url}/owner/model/resolve/${revision}/missing.bin`)).status, 404);
  assert.equal((await fetch(`${bridge.url}/owner/model/resolve/${revision}/..%2Fsecret`)).status, 400);
  assert.equal((await fetch(`${bridge.url}/api/models/owner/model/revision/dev`)).status, 404);

  const localRepository = join(root, 'models--owner--local-model');
  const localSnapshot = join(localRepository, 'snapshots/local');
  await mkdir(join(localRepository, 'refs'), {recursive: true});
  await mkdir(localSnapshot, {recursive: true});
  await writeFile(join(localRepository, 'refs/main'), 'local\n');
  await writeFile(join(localSnapshot, 'config.json'), '{"model_type":"kidi_omnivoice"}');
  const localRevision = await (await fetch(`${bridge.url}/api/models/owner/local-model/revision/main`)).json();
  assert.match(localRevision.sha, /^[a-f0-9]{40}$/u);
  assert.equal(await (await fetch(
    `${bridge.url}/owner/local-model/resolve/${localRevision.sha}/config.json`)).text(),
  '{"model_type":"kidi_omnivoice"}');
});
