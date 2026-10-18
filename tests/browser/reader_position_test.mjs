import test from 'node:test';
import assert from 'node:assert/strict';
import {normalizeReaderText, readerProgressLabel, readerSpeechSlice}
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
