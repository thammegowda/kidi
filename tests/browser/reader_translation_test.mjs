import test from 'node:test';
import assert from 'node:assert/strict';
import {
  READER_TRANSLATION_LOOKAHEAD,
  readerTranslationAttentionLimit,
  readerTranslationAttentionWindow,
  splitReaderTranslationText,
} from '../../web/ui/reader-translation.js';

test('reader translation stops three paragraphs beyond the visible range', () => {
  const rows = Array.from({length: 12}, (_, index) => ({
    index,
    top: index * 100,
    bottom: index * 100 + 90,
  }));
  assert.equal(READER_TRANSLATION_LOOKAHEAD, 3);
  assert.equal(readerTranslationAttentionLimit(rows, 205, 495), 7);
  assert.deepEqual(readerTranslationAttentionWindow(rows, 205, 495), {start: 2, end: 7});
});

test('reader translation follows the nearest row when the viewport is between rows', () => {
  const rows = [
    {index: 0, top: 0, bottom: 80},
    {index: 1, top: 120, bottom: 200},
    {index: 2, top: 240, bottom: 320},
    {index: 3, top: 360, bottom: 440},
    {index: 4, top: 480, bottom: 560},
  ];
  assert.equal(readerTranslationAttentionLimit(rows, 205, 230, 1), 3);
  assert.equal(readerTranslationAttentionLimit(rows, 600, 700, 1), 4);
});

test('reader translation attention rejects invalid or empty geometry', () => {
  assert.equal(readerTranslationAttentionLimit([], 0, 100), -1);
  assert.equal(readerTranslationAttentionLimit([{index: 0, top: 0, bottom: 10}], 100, 0), -1);
});

test('reader translation keeps sentences whole within paragraph limits', () => {
  const text = `${'a'.repeat(180)}. ${'b'.repeat(180)}.`;
  const chunks = splitReaderTranslationText(text, {
    targetCharacters: 200,
    maximumCharacters: 400,
    maximumSentenceCharacters: 400,
  });
  assert.equal(chunks.length, 2);
  assert.match(chunks[0], /\.$/u);
  assert.match(chunks[1], /\.$/u);
});

test('reader translation force-splits long sentences at punctuation', () => {
  const text = `${'a'.repeat(260)}, ${'b'.repeat(260)}; ${'c'.repeat(260)}.`;
  const chunks = splitReaderTranslationText(text, {
    targetCharacters: 350,
    maximumCharacters: 400,
    maximumSentenceCharacters: 400,
  });
  assert.ok(chunks.length >= 3);
  assert.ok(chunks.every(chunk => [...chunk].length <= 400));
  assert.match(chunks[0], /,$/u);
});

test('reader translation hard-splits unbroken text', () => {
  const chunks = splitReaderTranslationText('x'.repeat(901), {
    targetCharacters: 300,
    maximumCharacters: 400,
    maximumSentenceCharacters: 400,
  });
  assert.deepEqual(chunks.map(chunk => [...chunk].length), [400, 400, 101]);
});
