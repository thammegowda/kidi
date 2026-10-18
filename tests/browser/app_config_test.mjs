import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {load} from '../../web/libs/js-yaml/js-yaml.mjs';
import {loadAppConfig} from '../../web/hosts/wasm/app-config.mjs';

test('browser consumes the canonical desktop YAML configuration', async () => {
  const source = await readFile(new URL('../../examples/kidi-app.yaml', import.meta.url), 'utf8');
  const config = await loadAppConfig(load(source));
  assert.equal(config.model_hub.remote, 'https://huggingface.co');
  assert.equal(config.model_hub.local, '~/.cache/kidi/model-hub');
  assert.equal(config.models.chat.model_id, 'google/gemma-4-E2B-it-qat-mobile-transformers');
  assert.equal(config.models.asr.model_id, 'openai/whisper-small');
  assert.equal(config.models.tts.model_id, 'thammegowda/OmniVoice-q8');
  assert.equal(config.models.chat.system_prompt.includes('preserve natural code switching'), true);
  assert.equal(config.models.asr.vad.silence_ms, 800);
  assert.equal(config.models.tts.realtime_steps, 10);
  assert.equal(config.models.tts.steps, 16);
  assert.equal(config.models.tts.pace, 0.74);
  assert.equal(config.models.tts.gpu_precision, 'w8a8');
  assert.equal(config.models.tts.reference_text, 'Nice to meet you.');
});

test('browser rejects incomplete shared model configuration', async () => {
  await assert.rejects(loadAppConfig({version: 1, model_hub: {remote: 'http://example.com'}, models: {}}),
    /model_hub.remote|models/);
});

test('first-run browser UI requests model loading and exposes progress', async () => {
  const html = await readFile(new URL('../../web/ui/index.html', import.meta.url), 'utf8');
  const app = await readFile(new URL('../../web/ui/app.js', import.meta.url), 'utf8');
  for (const id of [
    'model-onboarding', 'onboarding-load-model', 'onboarding-model-loading',
    'onboarding-model-loading-progress', 'onboarding-model-loading-bytes', 'onboarding-load-asr',
    'onboarding-load-tts', 'dismiss-model-onboarding', 'load-asr-model', 'load-tts-model',
    'backend-activity', 'backend-memory', 'runtime-progress',
    'book-tts-controls', 'book-tts-steps', 'book-tts-pace', 'book-tts-gender', 'book-tts-age',
    'book-tts-pitch', 'book-tts-style', 'book-tts-accent', 'book-tts-gpu-precision', 'book-tts-more',
  ])
    assert.match(html, new RegExp(`id="${id}"`));
  assert.match(app, /function requestModelLoad\(\)/);
  assert.match(app, /container\.append\(modelOnboarding\);\s+modelOnboarding\.hidden = !needsOnboarding/);
  assert.match(app, /\$\('load-wasm-model'\)\.addEventListener\('click', requestModelLoad\)/);
  assert.match(app, /\$\('onboarding-load-model'\)\.addEventListener\('click', requestModelLoad\)/);
  assert.match(app, /\$\('onboarding-load-asr'\)\.addEventListener\('click'/);
  assert.match(app, /\$\('onboarding-load-tts'\)\.addEventListener\('click'/);
  assert.match(app, /\$\('dismiss-model-onboarding'\)\.addEventListener\('click', dismissModelSetup\)/);
  assert.match(app, /delete options\.gpu_precision_version;\s+delete options\.defaults_version;/);
  assert.match(app, /return state\.outputLanguage \? \[state\.outputLanguage\] : selectedInputLanguages\(\)/);
});

test('browser branding uses the square Kidi icon', async () => {
  const html = await readFile(new URL('../../web/ui/index.html', import.meta.url), 'utf8');
  const assets = await readFile(new URL('../../web/ui/assets.txt', import.meta.url), 'utf8');
  const icon = await readFile(new URL('../../web/ui/icons/kidi-logo.png', import.meta.url));
  assert.match(html, /rel="icon"[^>]+icons\/kidi-logo\.png/);
  assert.match(html, /class="brand-mark"[^>]+icons\/kidi-logo\.png/);
  assert.match(assets, /^icons\/kidi-logo\.png$/m);
  assert.deepEqual([...icon.subarray(0, 8)], [137, 80, 78, 71, 13, 10, 26, 10]);
  assert.equal(icon.readUInt32BE(16), 512);
  assert.equal(icon.readUInt32BE(20), 512);
});
