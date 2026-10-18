import test from 'node:test';
import assert from 'node:assert/strict';
import {MAX_SPEECH_SEGMENT_UNITS, SpeechSegments, speechUnits} from '../../web/ui/speech-controller.js';

test('speech starts at complete sentences before the input stream finishes', () => {
    const stream = new SpeechSegments();
    assert.deepEqual(stream.push('Hello'), []);
    assert.deepEqual(stream.push(' world. More'), [{id: 0, text: 'Hello world.', start: 0, end: 12}]);
    assert.deepEqual(stream.push(' text.', true), [{id: 1, text: 'More text.', start: 13, end: 23}]);
    assert.throws(() => stream.push('late'), /Invalid/);
});

test('long sentences are bounded without dropping or duplicating source text', () => {
    const clause = 'The small wooden bridge crosses the river near the old mill and follows a winding path through the trees where a bright bird sings softly while the sun rises over the valley';
    const text = `${clause} and ${clause.toLowerCase()} and ${clause.toLowerCase()}.`;
    const stream = new SpeechSegments(), output = [];
    for (let start = 0; start < text.length; start += 7) output.push(...stream.push(text.slice(start, start + 7)));
    output.push(...stream.push('', true));
    assert.ok(output.length >= 3);
    for (const segment of output) {
        assert.equal(segment.text, text.slice(segment.start, segment.end));
        assert.ok(speechUnits(segment.text) <= MAX_SPEECH_SEGMENT_UNITS);
    }
    assert.equal(output.map(segment => segment.text).join(' ').replace(/\s+/g, ' '), text);
});

test('batch and incremental input preserve abbreviations, Unicode, and hard bounds', () => {
    const text = 'Dr. Smith measured 3.14 meters. ' + '日本語の文章です。'.repeat(16) + 'x'.repeat(220);
    for (const size of [1, 11, text.length]) {
        const stream = new SpeechSegments(), output = [];
        for (let index = 0; index < text.length; index += size) output.push(...stream.push(text.slice(index, index + size)));
        output.push(...stream.push('', true));
        assert.ok(output.every(segment => speechUnits(segment.text) <= MAX_SPEECH_SEGMENT_UNITS));
        assert.equal(output.map(segment => segment.text.replace(/\s/g, '')).join(''), text.replace(/\s/g, ''));
        assert.ok(output.every(segment => !/^[\uDC00-\uDFFF]|[\uD800-\uDBFF]$/u.test(segment.text)));
        assert.notEqual(output[0].text, 'Dr.');
    }
});

test('abbreviations wait for their sentence and oversized graphemes fail explicitly', () => {
    const stream = new SpeechSegments();
    assert.deepEqual(stream.push('Meet Dr. '), []);
    assert.deepEqual(stream.push('Smith. More'), [{id: 0, text: 'Meet Dr. Smith.', start: 0, end: 15}]);
    assert.throws(() => new SpeechSegments().push('e' + '\u0301'.repeat(300), true), /grapheme/);
});

test('normal sentences remain whole instead of splitting a noun phrase', () => {
    const text = 'I can understand, generate, and debug code in many programming languages, including Python, JavaScript, Java, C++, and more.';
    for (const size of [1, 7, text.length]) {
        const stream = new SpeechSegments(), output = [];
        for (let index = 0; index < text.length; index += size) output.push(...stream.push(text.slice(index, index + size)));
        output.push(...stream.push('', true));
        assert.deepEqual(output.map(segment => segment.text), [text]);
    }
});
