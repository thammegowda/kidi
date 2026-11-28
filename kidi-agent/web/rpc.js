// The agent's one connection to its server (LiteApp's reconnecting WebSocket channel): `call` sends a request and
// `on` receives session-wide events (streamed replies, tool calls, approvals, file changes).
import {Connection} from './liteapp/host/connection.mjs';

const connection = new Connection();

export const connect = callbacks => connection.connect(callbacks);
export const call = (method, payload, options) => connection.call(method, payload, options);
export const on = (type, listener) => connection.on(type, listener);
export const subscribe = () => connection.subscribe();
