import {MAX_SPEECH_SEGMENT_UNITS, speechUnits} from './speech-segments.mjs';

export const MODEL_IDLE_MS = 300000;
export const MAX_RESIDENT_MODELS = 3;
export const HOST_MODEL_BUDGET = 3 * 1024 * 1024 * 1024;

export function modelIdentity(kind, data) {
    const source = data?.manifest || data?.source;
    if (!['chat', 'tts', 'asr'].includes(kind) || typeof source !== 'string' || !source.trim())
        throw new Error('A GPU model requires a valid kind and source');
    const matrixPrecision = data.matrixPrecision ?? 'f32';
    if (!['f32', 'mixed-f16', 'w8a8'].includes(matrixPrecision))
        throw new Error('WebGPU matrix precision must be f32, mixed-f16, or w8a8');
    const w8a8Kernel = data.w8a8Kernel ?? 'float';
    if (!['packed', 'float'].includes(w8a8Kernel)) throw new Error('WebGPU W8A8 kernel must be packed or float');
    return JSON.stringify({kind, source, matrixPrecision, packedDot: data.packedDot !== false, w8a8Kernel});
}

export function evictionCandidate(models, {now, pressure = false, idleMs = MODEL_IDLE_MS, maximum = MAX_RESIDENT_MODELS}) {
    const entries = [...models];
    return entries.filter(model => !model.loading && !model.active && !model.closing && !model.pins)
        .sort((a, b) => a.lastUsed - b.lastUsed)
        .find(model => pressure || entries.length > maximum || now - model.lastUsed >= idleMs);
}

export function protectedRepresentationOwners(models, requested) {
    return new Set([...models].filter(model => model === requested || model.executing || model.admitted)
        .map(model => model.key));
}

export function synthesisGpuReserve(model, data, preparedBytes) {
    if (data.streamSegment && speechUnits(data.text) > MAX_SPEECH_SEGMENT_UNITS)
        throw new Error('Streaming speech segment exceeds its text budget');
    const signature = JSON.stringify([data.text, data.options || {}]);
    const same = model.lastSynthesis === signature && model.lastChunks === 1 &&
        data.options?.use_voice_prompt === false && data.options?.cache_voice_prompt === false;
    const workspace = !model.warmed && model.matrixPrecision === 'mixed-f16' ? 1536 : same ? 128 : 768;
    return {signature, bytes: workspace * 1024 * 1024 + Math.max(0, (model.requiredPreparedBytes || 0) - preparedBytes)};
}
