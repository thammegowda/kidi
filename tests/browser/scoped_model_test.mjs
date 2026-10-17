import test from 'node:test';
import assert from 'node:assert/strict';
import {scopedModel} from '../../web/hosts/wasm/scoped-model.mjs';

test('model loader namespaces and host allocations belong to their session', () => {
    const calls = [];
    const module = {
        HEAPU8: new Uint8Array(8),
        FS: {
            mkdirTree: path => calls.push(['mkdirTree', path]),
            writeFile: (path, bytes) => calls.push(['writeFile', path, bytes]),
        },
        _malloc: size => size,
        _free: pointer => calls.push(['free', pointer]),
    };
    const first = scopedModel(module, '/models/chat-1');
    const second = scopedModel(module, '/models/tts-2');
    first.FS.writeFile('/model/model.yaml', 'chat');
    second.FS.writeFile('/model/model.yaml', 'speech');
    first.FS.writeFile('/model-other/unchanged', 'unchanged');
    assert.deepEqual(calls.slice(2), [
        ['writeFile', '/models/chat-1/model.yaml', 'chat'],
        ['writeFile', '/models/tts-2/model.yaml', 'speech'],
        ['writeFile', '/model-other/unchanged', 'unchanged'],
    ]);
    module.HEAPU8 = new Uint8Array(16);
    assert.equal(first.HEAPU8.byteLength, 16);
    first.kidiExternal = {name: 'chat'};
    assert.equal(second.kidiExternal, undefined);
    assert.throws(() => scopedModel(module, '/models/../escape'), /Invalid/);
});

test('disposing one model frees only its remaining owned allocations', () => {
    let pointer = 64;
    const freed = [];
    const module = {
        HEAPU8: new Uint8Array(256),
        FS: {mkdirTree() {}, analyzePath: () => ({exists: false})},
        _malloc(size) { const result = pointer; pointer += size; return result; },
        _free(value) { freed.push(value); },
    };
    const first = scopedModel(module, '/models/chat-1'), second = scopedModel(module, '/models/tts-2');
    const one = first._malloc(8), two = first._malloc(4), other = second._malloc(16);
    first.kidiExternal = {files: [new Uint8Array(10)]};
    assert.equal(first.allocatedBytes, 12);
    assert.equal(second.allocatedBytes, 16);
    first._free(one);
    assert.equal(first.allocatedBytes, 4);
    first.dispose();
    first.dispose();
    assert.deepEqual(freed, [one, two]);
    assert.equal(first.allocatedBytes, 0);
    assert.equal(first.kidiExternal, null);
    assert.equal(second.allocatedBytes, 16);
    second.dispose();
    assert.deepEqual(freed, [one, two, other]);
});
