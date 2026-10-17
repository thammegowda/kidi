export function speechMatchText(text) {
  return text.replace(/!?\[([^\]]*)\]\([^)]*\)/g, '$1')
    .replace(/^```[^\n]*$/gm, '').normalize('NFKC').toLowerCase().replace(/[^\p{L}\p{N}]/gu, '');
}

export function speechRanges(root, segments) {
  const document = root.ownerDocument, positions = [];
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
    acceptNode: node => node.parentElement?.closest('button, script, style, [aria-hidden="true"]')
      ? NodeFilter.FILTER_REJECT : NodeFilter.FILTER_ACCEPT,
  });
  let text = '', node;
  while ((node = walker.nextNode())) {
    for (const {segment: character, index: offset} of graphemes.segment(node.data)) {
      const key = speechMatchText(character);
      for (const value of key) {
        text += value;
        for (let index = 0; index < value.length; ++index)
          positions.push({node, start: offset, end: offset + character.length});
      }
    }
  }
  const ranges = [];
  let cursor = 0;
  for (const segment of segments) {
    const needle = speechMatchText(segment.text);
    if (!needle) continue;
    const begin = text.indexOf(needle, cursor);
    if (begin < 0) continue;
    const first = positions[begin], last = positions[begin + needle.length - 1];
    if (!first || !last) continue;
    const range = document.createRange();
    range.setStart(first.node, first.start);
    range.setEnd(last.node, last.end);
    ranges.push({range, status: segment.status});
    cursor = begin + needle.length;
  }
  return ranges;
}

function clearMarks() {
  for (const mark of document.querySelectorAll('mark[data-kidi-speech]')) {
    const parent = mark.parentNode;
    mark.replaceWith(...mark.childNodes);
    parent.normalize();
  }
}

export function renderSpeechHighlights(streams) {
  clearMarks();
  const ready = [], playing = [];
  for (const stream of streams) {
    if (!stream.messageId || !stream.segments) continue;
    const root = document.querySelector(`[data-message-id="${CSS.escape(stream.messageId)}"] .message-content`);
    if (!root) continue;
    const segments = [...stream.segments.values()].sort((a, b) => a.id - b.id);
    for (const value of speechRanges(root, segments))
      (value.status === 'playing' ? playing : ready).push(value.range);
  }
  if (globalThis.Highlight && CSS.highlights) {
    CSS.highlights.set('kidi-speech-ready', new Highlight(...ready));
    CSS.highlights.set('kidi-speech-playing', new Highlight(...playing));
    return;
  }
  for (const [ranges, status] of [[ready, 'ready'], [playing, 'playing']]) {
    for (const range of ranges.reverse()) {
      const walker = document.createTreeWalker(range.commonAncestorContainer, NodeFilter.SHOW_TEXT);
      const nodes = range.commonAncestorContainer.nodeType === Node.TEXT_NODE ? [range.commonAncestorContainer] : [];
      let node;
      while ((node = walker.nextNode())) if (range.intersectsNode(node)) nodes.push(node);
      for (const text of nodes.reverse()) {
        const first = text === range.startContainer ? range.startOffset : 0;
        const last = text === range.endContainer ? range.endOffset : text.length;
        if (first >= last) continue;
        const part = document.createRange();
        part.setStart(text, first); part.setEnd(text, last);
        const mark = document.createElement('mark');
        mark.dataset.kidiSpeech = status;
        mark.className = `speech-segment-${status}`;
        part.surroundContents(mark);
      }
    }
  }
}
const graphemes = new Intl.Segmenter(undefined, {granularity: 'grapheme'});
