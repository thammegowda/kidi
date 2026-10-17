export class RecentAudio {
    #entries = new Map();
    #bytes = 0;

    constructor({maximumBytes = 16 * 1024 * 1024, maximumEntries = 48, lifetimeMs = 600000, now = Date.now} = {}) {
        if (![maximumBytes, maximumEntries, lifetimeMs].every(value => Number.isSafeInteger(value) && value > 0) ||
            typeof now !== 'function') throw new Error('Invalid recent audio limits');
        this.maximumBytes = maximumBytes;
        this.maximumEntries = maximumEntries;
        this.lifetimeMs = lifetimeMs;
        this.now = now;
    }

    get bytes() { return this.#bytes; }

    get(key) {
        const entry = this.#entries.get(key);
        if (!entry) return null;
        if (this.now() - entry.created >= this.lifetimeMs) {
            this.#remove(key);
            return null;
        }
        this.#entries.delete(key);
        this.#entries.set(key, entry);
        return {audio: entry.audio.slice(0), sample_rate: entry.sampleRate, cached: true};
    }

    put(key, audio, sampleRate) {
        if (!(audio instanceof ArrayBuffer) || !audio.byteLength || audio.byteLength % 4 ||
            !Number.isInteger(sampleRate) || sampleRate <= 0 || new Float32Array(audio).some(value => !Number.isFinite(value)))
            throw new Error('Invalid replay audio');
        this.#remove(key);
        if (audio.byteLength > this.maximumBytes) return;
        for (const [oldest, entry] of this.#entries)
            if (this.now() - entry.created >= this.lifetimeMs) this.#remove(oldest);
        while (this.#entries.size >= this.maximumEntries || this.#bytes + audio.byteLength > this.maximumBytes)
            this.#remove(this.#entries.keys().next().value);
        const copy = audio.slice(0);
        this.#entries.set(key, {audio: copy, sampleRate, created: this.now()});
        this.#bytes += copy.byteLength;
    }

    #remove(key) {
        const entry = this.#entries.get(key);
        if (!entry) return;
        this.#bytes -= entry.audio.byteLength;
        this.#entries.delete(key);
    }
}
