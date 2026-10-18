export const READER_TRANSLATION_LOOKAHEAD = 3;

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
