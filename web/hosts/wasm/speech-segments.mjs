export const MAX_SPEECH_SEGMENT_UNITS = 96;
const FIRST_SEGMENT_UNITS = 64;
const wideScript = /[\p{Script=Han}\p{Script=Hiragana}\p{Script=Katakana}\p{Script=Hangul}]/u;
const graphemes = new Intl.Segmenter(undefined, {granularity: 'grapheme'});
const sentences = new Intl.Segmenter(undefined, {granularity: 'sentence'});
const abbreviation = /\b(?:[A-Z]|Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|(?:[A-Za-z]\.)+[A-Za-z])\.$/u;

export function speechUnits(text) {
    let units = 0;
    for (const {segment} of graphemes.segment(text)) units += wideScript.test(segment) ? 3 : segment.length;
    return units;
}

export class SpeechSegments {
    #buffer = '';
    #offset = 0;
    #count = 0;
    #finished = false;

    push(text, finished = false) {
        if (this.#finished || typeof text !== 'string') throw new Error('Invalid speech text stream');
        this.#buffer += text;
        const result = [];
        for (;;) {
            const leading = this.#buffer.match(/^\s+/u)?.[0].length || 0;
            this.#offset += leading;
            this.#buffer = this.#buffer.slice(leading);
            if (!this.#buffer) break;
            const limit = this.#count ? MAX_SPEECH_SEGMENT_UNITS : FIRST_SEGMENT_UNITS;
            let bounded = 0, units = 0, word = 0, clause = 0;
            for (const {segment, index} of graphemes.segment(this.#buffer)) {
                const next = units + (wideScript.test(segment) ? 3 : segment.length);
                if (next > limit) break;
                units = next;
                bounded = index + segment.length;
                if (/\s/u.test(segment)) word = index;
                if (/[,;:，；：]/u.test(segment)) clause = bounded;
            }
            if (!bounded) throw new Error('Speech grapheme exceeds the segment budget');
            let end = 0;
            for (const {segment, index} of sentences.segment(this.#buffer)) {
                const candidate = this.#buffer.slice(0, index + segment.length).trimEnd();
                if (speechUnits(candidate) > limit) break;
                if (abbreviation.test(candidate)) continue;
                if ((finished || index + segment.length < this.#buffer.length || /[.!?。！？]["'”’)\]]*\s+$/u.test(segment)) &&
                    /[.!?。！？]["'”’)\]]*\s*$/u.test(segment)) end = candidate.length;
                break;
            }
            if (!end && bounded < this.#buffer.length) {
                end = clause >= bounded / 2 ? clause : word || bounded;
            }
            if (!end && finished) end = this.#buffer.length;
            if (!end) break;
            const raw = this.#buffer.slice(0, end), text = raw.trimEnd();
            if (text) result.push({id: this.#count++, text, start: this.#offset, end: this.#offset + text.length});
            this.#buffer = this.#buffer.slice(end);
            this.#offset += end;
        }
        this.#finished = finished;
        return result;
    }
}
