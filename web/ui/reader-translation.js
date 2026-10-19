export const READER_TRANSLATION_LOOKAHEAD = 3;
const sentenceTerminator = /[.!?\u0589\u061f\u0964\u0965\u3002\uff01\uff1f]/u;
const clausePunctuation = /[,;:،؛、，：；—–-]/u;

const characters = text => [...text];

function splitLongSentence(text, maximumCharacters) {
  const result = [];
  let remaining = text.trim();
  while (characters(remaining).length > maximumCharacters) {
    const values = characters(remaining);
    const lower = Math.max(1, Math.floor(maximumCharacters / 2));
    let end = -1;
    for (let index = maximumCharacters - 1; index >= lower; --index)
      if (clausePunctuation.test(values[index])) {
        end = index + 1;
        break;
      }
    if (end < 0)
      for (let index = maximumCharacters - 1; index >= lower; --index)
        if (/\s/u.test(values[index])) {
          end = index;
          break;
        }
    if (end <= 0) end = maximumCharacters;
    result.push(values.slice(0, end).join('').trim());
    remaining = values.slice(end).join('').trim();
  }
  if (remaining) result.push(remaining);
  return result;
}

function naturalSentences(text) {
  const values = characters(String(text || '').trim());
  const result = [];
  let begin = 0;
  for (let index = 0; index < values.length; ++index) {
    if (!sentenceTerminator.test(values[index])) continue;
    let end = index + 1;
    while (end < values.length && /["'”’)\]]/u.test(values[end])) ++end;
    const sentence = values.slice(begin, end).join('').trim();
    if (sentence) result.push(sentence);
    begin = end;
    while (begin < values.length && /\s/u.test(values[begin])) ++begin;
    index = begin - 1;
  }
  const tail = values.slice(begin).join('').trim();
  if (tail) result.push(tail);
  return result;
}

export function splitReaderTranslationText(text, {
  targetCharacters = 800,
  maximumCharacters = 1200,
  maximumSentenceCharacters = 600,
} = {}) {
  if (![targetCharacters, maximumCharacters, maximumSentenceCharacters]
      .every(value => Number.isSafeInteger(value) && value > 0) ||
      targetCharacters > maximumCharacters || maximumSentenceCharacters > maximumCharacters)
    throw new Error('Invalid reader translation segmentation limits');
  const pieces = [];
  for (const segment of naturalSentences(text))
    pieces.push(...splitLongSentence(segment, maximumSentenceCharacters));
  const result = [];
  let current = '';
  for (const piece of pieces) {
    if (!piece) continue;
    const combined = current ? `${current} ${piece}` : piece;
    const combinedLength = characters(combined).length;
    if (current && (combinedLength > maximumCharacters ||
        (characters(current).length >= targetCharacters / 2 && combinedLength > targetCharacters))) {
      result.push(current);
      current = piece;
    } else {
      current = combined;
    }
  }
  if (current) result.push(current);
  return result;
}

export function readerTranslationAttentionWindow(rows, viewportTop, viewportBottom,
  lookahead = READER_TRANSLATION_LOOKAHEAD) {
  if (!Array.isArray(rows) || !rows.length || !Number.isFinite(viewportTop) ||
      !Number.isFinite(viewportBottom) || viewportBottom <= viewportTop ||
      !Number.isSafeInteger(lookahead) || lookahead < 0)
    return null;
  const normalized = rows
    .filter(row => Number.isSafeInteger(row?.index) && row.index >= 0 &&
      Number.isFinite(row.top) && Number.isFinite(row.bottom) && row.bottom >= row.top)
    .sort((left, right) => left.index - right.index);
  if (!normalized.length) return null;
  const visible = normalized.filter(row => row.bottom > viewportTop && row.top < viewportBottom);
  const attention = visible.length ? visible.map(row => row.index) :
    [normalized.find(row => row.bottom > viewportTop)?.index ?? normalized.at(-1).index];
  return {
    start: Math.min(...attention),
    end: Math.min(normalized.at(-1).index, Math.max(...attention) + lookahead),
  };
}

export function readerTranslationAttentionLimit(rows, viewportTop, viewportBottom,
  lookahead = READER_TRANSLATION_LOOKAHEAD) {
  return readerTranslationAttentionWindow(rows, viewportTop, viewportBottom, lookahead)?.end ?? -1;
}
