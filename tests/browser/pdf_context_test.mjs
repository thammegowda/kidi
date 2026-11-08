import assert from 'node:assert/strict';
import test from 'node:test';
import {extractPdf} from '../../web/extension/pdf.mjs';

const url = 'https://example.org/document.pdf';
const maximumBytes = 32 * 1024 * 1024;

test('PDF download rejects non-web URLs before fetching', async t => {
  const fetch = t.mock.method(globalThis, 'fetch', async () => { throw new Error('Unexpected fetch'); });
  await assert.rejects(extractPdf('file:///private/document.pdf'), /Only HTTP and HTTPS/);
  assert.equal(fetch.mock.callCount(), 0);
});

test('PDF download surfaces HTTP failures without loading the parser', async t => {
  const fetch = t.mock.method(globalThis, 'fetch', async () => new Response(null, {status: 403}));
  await assert.rejects(extractPdf(url), /HTTP 403/);
  assert.equal(fetch.mock.calls[0].arguments[1].credentials, 'include');
  assert.ok(fetch.mock.calls[0].arguments[1].signal instanceof AbortSignal);
});

test('PDF download enforces the declared byte limit and cancels the body', async t => {
  let cancelled = false;
  t.mock.method(globalThis, 'fetch', async () => new Response(new ReadableStream({
    cancel() { cancelled = true; },
  }), {headers: {'content-length': String(maximumBytes + 1)}}));
  await assert.rejects(extractPdf(url), /32 MiB download limit/);
  assert.equal(cancelled, true);
});

test('PDF download enforces the streamed byte limit without a Content-Length header', async t => {
  let cancelled = false;
  t.mock.method(globalThis, 'fetch', async () => new Response(new ReadableStream({
    start(controller) {
      controller.enqueue(new Uint8Array(maximumBytes));
      controller.enqueue(new Uint8Array(1));
    },
    cancel() { cancelled = true; },
  })));
  await assert.rejects(extractPdf(url), /32 MiB download limit/);
  assert.equal(cancelled, true);
});

test('PDF download rejects HTML login pages and empty responses explicitly', async t => {
  const fetch = t.mock.method(globalThis, 'fetch', async () => new Response('<html>Sign in</html>'));
  await assert.rejects(extractPdf(url), /did not return a PDF/);
  fetch.mock.mockImplementation(async () => new Response(null));
  await assert.rejects(extractPdf(url), /returned no data/);
});
