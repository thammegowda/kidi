import test from 'node:test';
import assert from 'node:assert/strict';
import {
  READER_TRANSLATION_LOOKAHEAD,
  readerTranslationAttentionLimit,
  readerTranslationAttentionWindow,
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
