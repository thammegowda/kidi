export const READER_BLOCK_BREAK = '\u2029';

export function normalizeReaderText(value) {
  return String(value ?? '').replace(/\s+/gu, ' ').trim();
}

export function normalizeReaderSpeechText(value) {
  return String(value ?? '')
    .split(/\u2029|\s*\n+\s*/gu)
    .map(normalizeReaderText)
    .filter(Boolean)
    .join(READER_BLOCK_BREAK);
}

export function readerSpeechSlice(value, requestedOffset = 0) {
  const text = normalizeReaderSpeechText(value);
  let offset = Number.isFinite(requestedOffset) ? Math.trunc(requestedOffset) : 0;
  offset = Math.max(0, Math.min(offset, text.length));
  if (offset === text.length) return {text, offset, speech: ''};
  while (offset > 0 && offset < text.length && !/\s/u.test(text[offset - 1]) && !/\s/u.test(text[offset]))
    --offset;
  while (offset < text.length && /\s/u.test(text[offset])) ++offset;
  return {text, offset, speech: text.slice(offset)};
}

export function readerProgressLabel(offset, length) {
  if (!length || offset <= 0) return 'Start';
  if (offset >= length) return 'Finished';
  return `${Math.round(offset / length * 100)}%`;
}
