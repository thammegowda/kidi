import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {resolve} from 'node:path';
import {pathToFileURL} from 'node:url';

import {load} from '../libs/js-yaml/js-yaml.mjs';

const binary = resolve(process.argv[2] || 'build-extension/chromium/single/kidi.mjs');
const {default: create} = await import(pathToFileURL(binary));
const module = await create();
const call = (name, types = [], values = []) => {
  const result = JSON.parse(module.ccall(name, 'string', types, values));
  assert.equal(result.error, undefined, `${name}: ${result.error}`);
  return result;
};
const config = load(await readFile(new URL('../../tests/data/gemma4/model.yaml', import.meta.url), 'utf8'));
config.model.max_position_embeddings = 9216;
module.FS.mkdir('/fixture');
module.FS.writeFile('/fixture/model.safetensors',
  new Uint8Array(await readFile(new URL('../../tests/data/gemma4/model.safetensors', import.meta.url))));
module.FS.writeFile('/fixture/model.yaml', JSON.stringify({
  format_version: 1, model: config.model,
  weights_file: 'model.safetensors', tokenizer_file: 'tokenizer.json',
  decode: {maximum_new_tokens: 8, context_size: 9216},
}));
const tokens = ['<pad>', '<eos>', '<bos>', '<turn|>', '<|turn>', '<unk>',
  'alpha', 'beta', 'gamma', 'delta', 'theta', 'zeta', 'eta', 'iota', 'kappa', 'lambda'];
module.FS.writeFile('/fixture/tokenizer.json', JSON.stringify({
  version: '1.0', pre_tokenizer: {type: 'WhitespaceSplit'},
  decoder: {type: 'WordPiece', prefix: '##', cleanup: false},
  model: {type: 'WordLevel', unk_token: '<unk>', vocab: Object.fromEntries(tokens.map((token, id) => [token, id]))},
}));
module.FS.writeFile('/fixture/tokenizer_config.json', JSON.stringify({bos_token: '<bos>'}));
module.FS.writeFile('/fixture/chat_template.jinja',
  "{{ bos_token }} {% for message in messages %}{{ message.role }} {{ message.content }} <turn|> " +
  "{% endfor %}{% if add_generation_prompt %}<|turn> model {% endif %}");
call('kidi_configure', ['number'], [1]);
call('kidi_load', ['string'], ['/fixture']);
for (const task of ['chat', 'email_compose', 'email_polish', 'email_reply']) {
  const request = {task, text: task === 'chat' ? '' : 'alpha beta', instruction: 'gamma delta',
    system_prompt: 'alpha', history: [{role: 'user', content: 'alpha'}, {role: 'assistant', content: 'beta'}]};
  call('kidi_editor_enqueue', ['string', 'number'], [JSON.stringify(request), 3]);
  let completed;
  for (let count = 0; count < 100 && !completed; count++) {
    const step = call('kidi_step');
    completed = step.events.find(event => event.completed)?.completed;
  }
  assert.ok(completed, `${task} reaches C++ inference completion`);
  assert.ok(completed.prompt_tokens > 0);
  assert.ok(Array.isArray(completed.token_ids));
}
const invalid = JSON.parse(module.ccall('kidi_editor_enqueue', 'string', ['string', 'number'], [
  JSON.stringify({task: 'chat', instruction: 'hello', history: [{role: 'system', content: 'spoofed'}]}), 8,
]));
assert.match(invalid.error, /only accepts user and assistant roles/);

// Make the numerical fixture emit alpha, then EOS, independent of source content.
const controlled = new Uint8Array(await readFile(new URL('../../tests/data/gemma4/model.safetensors', import.meta.url)));
const view = new DataView(controlled.buffer, controlled.byteOffset, controlled.byteLength);
const headerBytes = Number(view.getBigUint64(0, true));
const header = JSON.parse(new TextDecoder().decode(controlled.subarray(8, 8 + headerBytes)));
const dataOffset = 8 + headerBytes;
for (const tensor of Object.values(header))
  if (tensor.shape?.length === 2)
    controlled.fill(0, dataOffset + tensor.data_offsets[0], dataOffset + tensor.data_offsets[1]);
const embedding = header['model.language_model.embed_tokens.weight'];
assert.equal(embedding.dtype, 'F32');
for (const [token, axis, value] of [[5, 0, 1], [6, 0, 2], [6, 1, 3], [1, 1, 5]])
  view.setFloat32(dataOffset + embedding.data_offsets[0] + (token * embedding.shape[1] + axis) * 4, value, true);
module.FS.writeFile('/fixture/model.safetensors', controlled);
call('kidi_load', ['string'], ['/fixture']);
function document(request, maximum = 3) {
  const {request_id: id} = call('kidi_document_enqueue', ['string', 'number'], [JSON.stringify(request), maximum]);
  const progress = [], replacements = [];
  let text = '';
  for (let count = 0; count < 300; count++) {
    const step = call('kidi_document_step');
    assert.ok(step.events.every(event => event.request_id === id), 'document events retain a stable public request ID');
    progress.push(step.document_progress);
    for (const event of step.events) if (event.text) {
      if (event.replace) {
        text = event.text;
        replacements.push(event.text);
      } else text += event.text;
    }
    const completed = step.events.find(event => event.completed)?.completed;
    if (completed) return {completed, progress, replacements, text};
  }
  assert.fail('document generation did not complete within its step budget');
}
const translated = document({
  mode: 'translate',
  text: Array.from({length: 3}, () => `${'alpha '.repeat(150).trim()}.`).join('\n\n'),
  target_language: 'French',
});
assert.equal(translated.completed.text, 'alpha\n\nalpha\n\nalpha');
assert.equal(translated.text, translated.completed.text);
assert.ok(translated.progress.some(progress => progress.part === 3));
assert.equal(translated.completed.output_limited, false);
const manyParagraphs = document({
  mode: 'translate',
  text: Array.from({length: 300}, () => 'alpha.').join('\n\n'),
  target_language: 'French',
});
assert.ok(Math.max(...manyParagraphs.progress.map(progress => progress.parts)) < 10,
  'small paragraphs are packed into model-sized translation segments');
const summarized = document({mode: 'summarize', text: 'alpha '.repeat(5000)});
assert.equal(summarized.completed.text, 'alpha');
assert.ok(summarized.progress.some(progress => progress.parts === 2));
assert.ok(summarized.progress.every(progress => progress.round === 1));
assert.deepEqual(summarized.replacements, ['alpha', 'alpha'],
  'each segment publishes a replacement cumulative summary');
assert.equal(summarized.completed.token_ids.length, 2, 'diagnostics aggregate both rolling summary requests');

call('kidi_document_enqueue', ['string', 'number'], [JSON.stringify({mode: 'summarize', text: 'alpha '.repeat(5000)}), 3]);
let advanced = false;
for (let count = 0; count < 100 && !advanced; count++)
  advanced = call('kidi_document_step').document_progress.part === 2;
assert.equal(advanced, true);
call('kidi_document_cancel');
assert.equal(call('kidi_step').pending, 0, 'cancel targets the current native request, not the initial one');

call('kidi_document_enqueue', ['string', 'number'], [
  JSON.stringify({mode: 'translate', text: 'alpha beta', target_language: 'French'}), 1,
]);
let retried = false, budgetError = '';
for (let count = 0; count < 200 && !budgetError; count++) {
  const step = JSON.parse(module.ccall('kidi_document_step', 'string', [], []));
  if (step.error) budgetError = step.error;
  else {
    retried ||= step.document_progress.parts > 1;
    assert.equal(step.events.some(event => event.text || event.completed), false,
      'truncated translation sections are never published as complete');
  }
}
assert.equal(retried, true);
assert.match(budgetError, /output budget/);
call('kidi_document_cancel');
assert.equal(document({mode: 'summarize', text: 'alpha beta'}).completed.text, 'alpha',
  'budget failure can be cleaned up and followed by another operation');
console.log('Real CPU WASM passed: editor/history, rolling summary replacement, ordered translation, adaptive budgets and cancellation.');
