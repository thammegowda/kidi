import {load} from '../../libs/js-yaml/js-yaml.mjs';

const object = (value, name) => {
    if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error(`${name} must be a mapping`);
    return value;
};

const mapping = value => value && typeof value === 'object' && !Array.isArray(value);

function merge(base, override) {
    if (!mapping(base) || !mapping(override)) return structuredClone(override);
    const result = structuredClone(base);
    for (const [key, value] of Object.entries(override))
        result[key] = key in result ? merge(result[key], value) : structuredClone(value);
    return result;
}

export function resolveRuntimeConfig(document, runtimeName) {
    const root = object(document, 'application config');
    const runtime = object(root.runtime, 'runtime');
    const configured = runtime[runtimeName];
    if (configured !== undefined && !mapping(configured))
        throw new Error(`runtime.${runtimeName} must be a mapping`);
    const resolved = configured ? merge(root, configured) : structuredClone(root);
    if (mapping(resolved.runtime)) {
        delete resolved.runtime.native;
        delete resolved.runtime.wasm;
    }
    return resolved;
}

export async function loadAppConfig(source = globalThis.KIDI_APP_CONFIG ||
    new URL('./app-config.yaml', import.meta.url)) {
    const document = typeof source === 'object' && !(source instanceof URL)
        ? source
        : load(await (async () => {
            const response = await fetch(source);
            if (!response.ok) throw new Error(`Application config HTTP ${response.status}`);
            return response.text();
        })());
    const root = resolveRuntimeConfig(document, 'wasm');
    if (root.version !== 1) throw new Error('Application config version must be 1');
    const hub = object(root.model_hub, 'model_hub');
    const remote = new URL(hub.remote);
    if (remote.protocol !== 'https:' || remote.username || remote.password || remote.search || remote.hash)
        throw new Error('model_hub.remote must be a public HTTPS origin');
    const models = object(root.models, 'models');
    for (const name of ['chat', 'asr', 'tts']) {
        const model = object(models[name], `models.${name}`);
        if (model.enabled !== false && !/^[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+$/u.test(model.model_id || ''))
            throw new Error(`models.${name}.model_id must be a Hugging Face model ID`);
    }
    return root;
}
