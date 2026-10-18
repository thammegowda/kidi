import {MessageTransport, RequestChannel} from '../../../third_party/liteapp/web/liteapp.mjs';

let runtime;

class RuntimeClient {
    constructor() {
        this.worker = new Worker(new URL('./gpu-runtime-worker.mjs', import.meta.url), {type: 'module'});
        this.channel = new RequestChannel(new MessageTransport(this.worker));
        this.sessions = new Set();
        this.failed = false;
        this.worker.onerror = event => {
            if (this.failed) return;
            this.failed = true;
            const sessions = [...this.sessions];
            for (const session of sessions) {
                session.closed = true;
                for (const request of session.requests) request.abort();
                session.requests.clear();
            }
            this.sessions.clear();
            this.channel.dispose(event.message || 'GPU runtime worker failed');
            this.worker.terminate();
            if (runtime === this) runtime = null;
            for (const session of sessions) session.onerror?.(event);
        };
        this.worker.addEventListener('message', event => {
            if (event.data?.type === 'runtime.error')
                this.worker.onerror({message: event.data.message});
            else if (event.data?.type === 'runtime.residency')
                for (const session of this.sessions)
                    if (session.id === event.data.session && !session.closed)
                        session.onmessage?.({data: {type: 'residency', state: event.data.state, kind: event.data.kind}});
        });
    }
}

export function createGpuWorker(kind) {
    runtime ||= new RuntimeClient();
    const client = runtime;
    const id = crypto.randomUUID();
    const session = {
        id,
        onmessage: null,
        onerror: null,
        requests: new Set(),
        closed: false,
        postMessage(data, transfer = []) {
            if (this.closed) throw new Error('GPU model session is closed');
            if (data.type === 'cancel') {
                for (const request of this.requests) request.abort();
                return;
            }
            const controller = new AbortController();
            this.requests.add(controller);
            client.channel.request(data.type, {session: id, kind, data}, {
                signal: controller.signal,
                transfer,
                onEvent: event => {
                    if (!this.closed && event.type === 'worker') this.onmessage?.({data: event.payload});
                },
            }).then(result => {
                if (!this.closed && result?.type) this.onmessage?.({data: result});
            }, error => {
                if (this.closed) return;
                this.onmessage?.({data: error.name === 'AbortError'
                    ? {type: 'cancelled', requestId: data.requestId}
                    : {type: 'error', requestId: data.requestId, error: error.message}});
            }).finally(() => this.requests.delete(controller));
        },
        terminate() {
            if (this.closed) return;
            this.closed = true;
            for (const request of this.requests) request.abort();
            this.requests.clear();
            client.sessions.delete(this);
            client.channel.request('release', {session: id}).catch(error => {
                console.error('Unable to release GPU model session:', error);
            });
        },
    };
    client.sessions.add(session);
    return session;
}
