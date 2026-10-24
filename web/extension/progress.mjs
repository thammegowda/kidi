export function formatBytes(value) {
    const bytes = Number(value);
    if (!Number.isFinite(bytes) || bytes < 0) return '';
    if (bytes >= 1024 ** 3) return `${(bytes / 1024 ** 3).toFixed(bytes >= 10 * 1024 ** 3 ? 0 : 2)} GB`;
    if (bytes >= 1024 ** 2) return `${(bytes / 1024 ** 2).toFixed(bytes >= 100 * 1024 ** 2 ? 0 : 1)} MB`;
    if (bytes >= 1024) return `${(bytes / 1024).toFixed(0)} KB`;
    return `${bytes.toFixed(0)} B`;
}

function formatDuration(seconds) {
    if (!Number.isFinite(seconds) || seconds < 0) return '';
    if (seconds < 60) return `${Math.max(1, Math.round(seconds))}s`;
    const minutes = Math.floor(seconds / 60);
    const remainder = Math.round(seconds % 60);
    return `${minutes}m ${remainder}s`;
}

export function modelProgressDetails(payload = {}, elapsedMs = 0) {
    const total = Math.max(0, Number(payload.totalBytes) || 0);
    const loaded = Math.max(0, Number(payload.loadedBytes) || 0);
    const downloaded = Math.max(0, Number(payload.downloadedBytes) || 0);
    const cached = Math.max(0, Number(payload.cachedBytes) || 0);
    const initializing = payload.type === 'initializing';
    const percent = total > 0 ? Math.min(100, Math.floor(loaded / total * 100)) : null;
    const elapsedSeconds = Math.max(0, elapsedMs / 1000);
    const rate = elapsedSeconds > 0 && downloaded > 0 ? downloaded / elapsedSeconds : 0;
    const remaining = Math.max(0, total - loaded);
    const eta = rate > 0 && remaining > 0 ? formatDuration(remaining / rate) : '';
    const stage = initializing ? 'Initializing inference engine' :
        total > 0 && downloaded === 0 && cached > 0 ? 'Reading cached model' :
            total > 0 ? 'Downloading model' :
                payload.file ? 'Loading model metadata' : 'Connecting to model repository';
    const bytes = total > 0 ? `${formatBytes(Math.min(loaded, total))} of ${formatBytes(total)}` : '';
    const transfer = initializing
        ? `Download complete · ${formatDuration(elapsedSeconds)} elapsed`
        : [
            downloaded > 0 ? `${formatBytes(downloaded)} network` : '',
            cached > 0 ? `${formatBytes(cached)} cached` : '',
            rate > 0 ? `${formatBytes(rate)}/s` : '',
            eta ? `~${eta} remaining` : '',
        ].filter(Boolean).join(' · ');
    return {
        stage,
        percent: initializing ? 100 : percent,
        bytes,
        transfer,
        file: payload.file || '',
        warning: payload.warning || '',
        total,
        loaded: initializing && total > 0 ? total : Math.min(loaded, total),
        determinate: total > 0,
    };
}

export function runtimeDiagnostics(snapshot = {}, stats = {}) {
    const runtime = snapshot.runtime || {};
    const memory = runtime.memory || {};
    const requested = String(runtime.requested_backend || snapshot.backend || '');
    const backend = requested === 'webgpu' ? 'WebGPU' :
        requested === 'cpu' || requested === 'wasm-cpu' ? 'CPU' :
            requested ? requested.toUpperCase() : 'Ready';
    const hostBytes = Math.max(0, Number(memory.host_bytes) || 0);
    const gpuBytes = Math.max(0, Number(memory.gpu_bytes) || 0);
    const decodeTokens = Math.max(0, Number(stats.decode_tokens ?? stats.token_count) || 0);
    const decodeMs = Math.max(0, Number(stats.decode_ms) || 0);
    const firstTokenMs = Math.max(0, Number(stats.first_token_ms) || 0);
    return [
        {key: 'backend', label: 'Backend', value: backend},
        ...(hostBytes ? [{key: 'ram', label: 'RAM', value: formatBytes(hostBytes)}] : []),
        ...(gpuBytes ? [{key: 'gpu', label: 'GPU', value: formatBytes(gpuBytes)}] : []),
        ...(decodeTokens && decodeMs
            ? [{key: 'speed', label: 'Speed', value: `${(decodeTokens * 1000 / decodeMs).toFixed(1)} tok/s`}]
            : []),
        ...(decodeTokens ? [{key: 'tokens', label: 'Output', value: `${decodeTokens} tok`}] : []),
        ...(firstTokenMs ? [{key: 'latency', label: 'TTFT', value: `${(firstTokenMs / 1000).toFixed(2)} s`}] : []),
    ];
}
