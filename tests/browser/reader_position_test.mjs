import test from 'node:test';
import assert from 'node:assert/strict';
import {
    READER_BLOCK_BREAK, normalizeReaderSpeechText, normalizeReaderText, readerProgressLabel, readerSpeechSlice,
}
    from '../../web/ui/reader-position.js';

test('reader narration positions normalize text and snap to word starts', () => {
    assert.equal(normalizeReaderText('  First\n\nsecond\tword. '), 'First second word.');
    assert.deepEqual(readerSpeechSlice('First second word.', 9), {
        text: 'First second word.',
        offset: 6,
        speech: 'second word.',
    });
    assert.deepEqual(readerSpeechSlice('First second word.', 6), {
        text: 'First second word.',
        offset: 6,
        speech: 'second word.',
    });
    assert.equal(readerProgressLabel(0, 100), 'Start');
    assert.equal(readerProgressLabel(46, 100), '46%');
    assert.equal(readerProgressLabel(100, 100), 'Finished');
});

test('reader speech text preserves semantic line breaks', () => {
    const text = normalizeReaderSpeechText('  Chapter One\n\nThe first sentence.\nSecond paragraph. ');
    assert.equal(text, `Chapter One${READER_BLOCK_BREAK}The first sentence.${READER_BLOCK_BREAK}Second paragraph.`);
    assert.deepEqual(readerSpeechSlice(text, 0), {text, offset: 0, speech: text});
});
