export class RuntimeScheduler {
    #queue = [];
    #running = false;

    run(priority, action, signal) {
        if (signal?.aborted) return Promise.reject(signal.reason || new DOMException('Request cancelled', 'AbortError'));
        return new Promise((resolve, reject) => {
            this.#queue.push({priority, action, signal, resolve, reject});
            this.#queue.sort((a, b) => b.priority - a.priority);
            queueMicrotask(() => this.#drain());
        });
    }

    async #drain() {
        if (this.#running) return;
        this.#running = true;
        try {
            while (this.#queue.length) {
                const entry = this.#queue.shift();
                if (entry.signal?.aborted) {
                    entry.reject(entry.signal.reason || new DOMException('Request cancelled', 'AbortError'));
                    continue;
                }
                try { entry.resolve(await entry.action()); }
                catch (error) { entry.reject(error); }
            }
        } finally { this.#running = false; }
    }
}

export function awaitWithSignal(promise, signal) {
    signal?.throwIfAborted();
    if (!signal) return promise;
    return new Promise((resolve, reject) => {
        const abort = () => reject(signal.reason);
        signal.addEventListener('abort', abort, {once: true});
        Promise.resolve(promise).then(value => {
            signal.removeEventListener('abort', abort);
            resolve(value);
        }, error => {
            signal.removeEventListener('abort', abort);
            reject(error);
        });
    });
}
