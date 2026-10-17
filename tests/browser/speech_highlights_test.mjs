import test from 'node:test';
import assert from 'node:assert/strict';
import {speechMatchText} from '../../web/ui/speech-highlights.js';

test('speech highlights match visible Markdown without matching hidden URLs', () => {
    assert.equal(speechMatchText('A **bright** [star](https://example.com).'), 'abrightstar');
    assert.equal(speechMatchText('Hello, world!'), 'helloworld');
    assert.equal(speechMatchText('caf\u00e9'), speechMatchText('cafe\u0301'));
});
