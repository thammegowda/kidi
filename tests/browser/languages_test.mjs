import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {activeLanguages, parseLanguages, resolveLanguage, resolveLanguages} from '../../web/ui/languages.js';

const table = parseLanguages(await readFile(new URL('../../web/ui/languages.tsv', import.meta.url), 'utf8'));

test('language table resolves codes, names, ISO aliases, and native names', () => {
  assert.equal(resolveLanguage('Kannada', table.aliases), 'kn');
  assert.equal(resolveLanguage('ಕನ್ನಡ', table.aliases), 'kn');
  assert.deepEqual(resolveLanguages('English, Kannada, తెలుగు, hin', 'fr-FR', table.aliases),
    ['en', 'kn', 'te', 'hi']);
  assert.deepEqual(resolveLanguages('eng, kan, tel, हिन्दी', 'fr-FR', table.aliases),
    ['en', 'kn', 'te', 'hi']);
});

test('automatic language uses locale aliases with English fallback', () => {
  assert.deepEqual(resolveLanguages('auto', 'de-DE', table.aliases), ['de']);
  assert.deepEqual(resolveLanguages('auto', 'nb-NO', table.aliases), ['no']);
  assert.deepEqual(resolveLanguages('unknown', 'xx-ZZ', table.aliases), ['en']);
});

test('active language pins or preserves the configured code-switch allowlist', () => {
  const configured = ['en', 'kn', 'te', 'hi'];
  assert.deepEqual(activeLanguages(configured, 'kn'), ['kn']);
  assert.deepEqual(activeLanguages(configured, ''), configured);
  assert.deepEqual(activeLanguages(configured, 'fr'), configured);
});
