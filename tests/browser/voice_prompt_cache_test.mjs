import test from 'node:test';
import assert from 'node:assert/strict';
import {clearVoicePromptCache, describeVoicePrompt, loadVoicePrompt, storeVoicePrompt}
    from '../../web/hosts/wasm/voice-prompt-cache.mjs';

function memoryCaches() {
    const stores = new Map();
    return {
        open: async name => {
            const entries = stores.get(name) || new Map();
            stores.set(name, entries);
            return {
                match: async key => entries.get(String(key))?.clone(),
                put: async (key, value) => entries.set(String(key), value.clone()),
            };
        },
        delete: async name => stores.delete(name),
        stores,
    };
}

const config = {
    model: 'model-hash',
    precision: 'w8a8',
    codebooks: 8,
    frameRate: 25,
    vocabulary: 1024,
    maximumSeconds: 20,
    voice: {gender: 'female', age: 'young adult'},
    reference: {language: 'English', text: 'Hello, I am Kidi.', seed: 0, steps: 10},
};

test('voice prompt keys are canonical and sensitive to identity settings', async () => {
    const reordered = {
        reference: config.reference, voice: config.voice, maximumSeconds: 20, vocabulary: 1024,
        frameRate: 25, codebooks: 8, precision: 'w8a8', model: 'model-hash',
    };
    assert.equal((await describeVoicePrompt(config)).id, (await describeVoicePrompt(reordered)).id);
    assert.notEqual((await describeVoicePrompt(config)).id,
        (await describeVoicePrompt({...config, reference: {...config.reference, seed: 1}})).id);
});

test('voice prompts round-trip through their dedicated cache', async () => {
    const previous = globalThis.caches;
    globalThis.caches = memoryCaches();
    try {
        const descriptor = await describeVoicePrompt(config);
        const tokens = Int32Array.from({length: 8 * 75}, (_, index) => index % 1024);
        assert.equal(await loadVoicePrompt(descriptor), null);
        await storeVoicePrompt(descriptor, {text: config.reference.text, tokens});
        const restored = await loadVoicePrompt(descriptor);
        assert.equal(restored.text, config.reference.text);
        assert.deepEqual(restored.tokens, tokens);
        assert.equal(await clearVoicePromptCache(), true);
        assert.equal(await loadVoicePrompt(descriptor), null);
    } finally {
        globalThis.caches = previous;
    }
});

test('voice prompt cache rejects corrupt and out-of-range data', async () => {
    const previous = globalThis.caches;
    const storage = memoryCaches();
    globalThis.caches = storage;
    try {
        const descriptor = await describeVoicePrompt(config);
        const tokens = new Int32Array(8 * 75).fill(1);
        await storeVoicePrompt(descriptor, {text: config.reference.text, tokens});
        const entries = storage.stores.get('kidi-voice-prompts-v1');
        const key = [...entries.keys()][0];
        const payload = await entries.get(key).json();
        payload.tokens[0] = 1024;
        entries.set(key, new Response(JSON.stringify(payload)));
        await assert.rejects(loadVoicePrompt(descriptor), /tokens/);
    } finally {
        globalThis.caches = previous;
    }
});
