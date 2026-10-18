import test from 'node:test';
import assert from 'node:assert/strict';

class MemoryCache {
  #entries = new Map();

  async put(key, response) {
    this.#entries.set(String(key), response.clone());
  }

  async match(key) {
    return this.#entries.get(String(key))?.clone();
  }

  async delete(key) {
    return this.#entries.delete(String(key));
  }

  async keys() {
    return [...this.#entries.keys()].map(url => ({url}));
  }
}

test('ebook translations are cached by source, chapter, and language', async () => {
  const previous = globalThis.caches;
  const stores = new Map();
  globalThis.caches = {
    async open(name) {
      if (!stores.has(name)) stores.set(name, new MemoryCache());
      return stores.get(name);
    },
  };
  try {
    const module = await import(`../../web/ui/ebooks.js?translation-test=${Date.now()}`);
    const book = {id: 'a'.repeat(64), title: 'Sample'};
    await module.storeEbookTranslation(book, 2, 'kn', 'Source paragraph', ['ಅನುವಾದ']);
    assert.deepEqual(
      await module.readEbookTranslation(book, 2, 'kn', 'Source paragraph'),
      ['ಅನುವಾದ']);
    assert.equal(await module.readEbookTranslation(book, 2, 'kn', 'Changed source'), null);
    assert.equal(await module.readEbookTranslation(book, 3, 'kn', 'Source paragraph'), null);
    assert.equal(await module.readEbookTranslation(book, 2, 'te', 'Source paragraph'), null);
    await assert.rejects(
      module.storeEbookTranslation(book, 2, 'kn', 'Source paragraph', ['']),
      /Invalid ebook translation/);
  } finally {
    if (previous === undefined) delete globalThis.caches;
    else globalThis.caches = previous;
  }
});
