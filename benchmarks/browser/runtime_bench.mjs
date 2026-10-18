const counters = ['dispatches', 'operatorCalls', 'bridgeCalls', 'regionCalls', 'submissions', 'readBytes', 'uploadBytes',
    'copyBytes',     'pipelineCreations', 'encodeMs', 'waitMs', 'preparedWeightHits', 'preparedWeightMisses', 'preparedWeightEvictions'];

export function measuredResult(kind, data, elapsedMs, before = {}) {
    const stats = data.gpuStats || {};
    const delta = Object.fromEntries(counters.map(key => [key, (stats[key] || 0) - (before[key] || 0)]));
    const audioSeconds = kind === 'tts' ? data.samples / data.sample_rate : data.input_audio_seconds;
    return {
        kind, elapsedMs,
        ...(audioSeconds > 0 ? {audioSeconds, rtf: elapsedMs / 1000 / audioSeconds} : {}),
        generationMs: data.generation_ms, decodeMs: data.decode_ms, preparationMs: data.preparation_ms,
        samples: data.samples, sampleRate: data.sample_rate, chunks: data.chunks,
        audioHash: data.audioHash, tokenHash: data.tokenHash,
        featureMs: data.feature_ms, encodeMs: data.encode_ms,
        attempts: data.attempts,
        tokenCount: data.token_ids?.length ?? data.stats?.tokenCount,
        text: data.text, stats: data.stats, gpu: delta,
        ...(stats.gpuMilliseconds ? {gpuMilliseconds: Object.fromEntries(
            Object.entries(stats.gpuMilliseconds).map(([label, value]) => [label, value - (before.gpuMilliseconds?.[label] || 0)]),
        )} : {}),
        gpuAllocatedBytes: stats.allocatedBytes, gpuPeakBytes: stats.peakAllocatedBytes,
        gpuPreparedWeightBytes: stats.preparedWeightBytes,
        heapBytes: data.heapBytes, memory: data.memory,
    };
}

/// Runs one loaded worker sequentially. A request array sweeps fresh inputs without reloading the model.
export async function benchmarkWorker({
    kind, workerUrl, workerFactory, load, request, trials = 3, timeoutMs = 180000, warmups = 1, onResult,
}) {
    const requests = Array.isArray(request) ? request : [request];
    if (!['tts', 'chat', 'asr'].includes(kind) || !Number.isInteger(trials) || trials < 1 || !requests.length ||
        !Number.isInteger(warmups) || warmups < 0)
        throw new Error('Invalid benchmark kind or trial count');
    const worker = workerFactory ? workerFactory() : new Worker(workerUrl, {type: 'module'});
    let pending;
    let timer;
    const logs = [];
    const wait = payload => new Promise((resolve, reject) => {
        pending = {resolve, reject, started: performance.now()};
        timer = setTimeout(() => {
            pending = null;
            reject(new Error(`Benchmark ${kind} timed out after ${timeoutMs} ms`));
        }, timeoutMs);
        worker.postMessage(payload);
    });
    worker.onmessage = ({data}) => {
        if (data.type === 'log') logs.push(data.text);
        if (data.type === 'step') {
            const completed = data.events?.find(event => event.completed)?.completed;
            if (!completed) return;
            data = {...data, ...completed, type: 'completed'};
        }
        if (!pending || !['ready', 'audio', 'result', 'completed', 'error'].includes(data.type)) return;
        const current = pending;
        pending = null;
        clearTimeout(timer);
        if (data.type === 'error') current.reject(new Error(data.error || 'Worker failed'));
        else current.resolve({data, elapsedMs: performance.now() - current.started});
    };
    worker.onerror = event => {
        clearTimeout(timer);
        pending?.reject(new Error(event.message || 'Benchmark worker failed'));
        pending = null;
    };
    try {
        const loaded = await wait({...load, type: 'load'});
        const results = [];
        let previous = {...loaded.data.gpuStats};
        for (let trial = 0; trial < warmups + trials; ++trial) {
            const sourceRequest = requests[trial % requests.length];
            const input = structuredClone(sourceRequest);
            const output = await wait({...input, requestId: `benchmark-${trial}`});
            const data = output.data;
            if (kind === 'asr') data.input_audio_seconds = sourceRequest.audio.byteLength / 4 / 16000;
            if (data.audio instanceof ArrayBuffer) {
                data.samples = data.audio.byteLength / 4;
                data.audioHash = [...new Uint8Array(await crypto.subtle.digest('SHA-256', data.audio))]
                    .map(byte => byte.toString(16).padStart(2, '0')).join('');
            }
            if (Array.isArray(data.token_ids))
                data.tokenHash = [...new Uint8Array(await crypto.subtle.digest('SHA-256',
                    new Int32Array(data.token_ids).buffer))].map(byte => byte.toString(16).padStart(2, '0')).join('');
            const result = measuredResult(kind, data, output.elapsedMs, previous);
            previous = {...data.gpuStats};
            results.push({...result, trial, inputIndex: trial % requests.length, warmup: trial < warmups});
            await onResult?.(data, {trial, warmup: trial < warmups, result});
        }
        const binaryHashes = await Promise.all(requests.map(async input => input.audio instanceof ArrayBuffer
            ? [...new Uint8Array(await crypto.subtle.digest('SHA-256', input.audio))]
                .map(byte => byte.toString(16).padStart(2, '0')).join('')
            : undefined));
        const binaryHash = Array.isArray(request) ? binaryHashes : binaryHashes[0];
        const identity = new TextEncoder().encode(JSON.stringify({kind, load, request, binaryHash}));
        const hash = await crypto.subtle.digest('SHA-256', identity);
        return {
            version: 1, kind, loadMs: loaded.elapsedMs,
            inputHash: [...new Uint8Array(hash)].map(byte => byte.toString(16).padStart(2, '0')).join(''),
            userAgent: navigator.userAgent, isolated: crossOriginIsolated,
            hardwareConcurrency: navigator.hardwareConcurrency,
            load: loaded.data, results, logs,
        };
    } finally {
        clearTimeout(timer);
        worker.terminate();
    }
}

export function transcriptionErrorRates(reference, hypothesis) {
    if (typeof reference !== 'string' || typeof hypothesis !== 'string')
        throw new Error('Transcription comparisons require text strings');
    const normalize = text => text.normalize('NFKC').toLowerCase()
        .replace(/[^\p{L}\p{N}\s]/gu, ' ').trim().replace(/\s+/g, ' ');
    const expected = normalize(reference), actual = normalize(hypothesis);
    if (!expected) throw new Error('Reference text must contain words');
    const distance = (left, right) => {
        let previous = Array.from({length: right.length + 1}, (_, index) => index);
        for (let row = 1; row <= left.length; ++row) {
            const next = [row];
            for (let column = 1; column <= right.length; ++column)
                next[column] = Math.min(next[column - 1] + 1, previous[column] + 1,
                    previous[column - 1] + Number(left[row - 1] !== right[column - 1]));
            previous = next;
        }
        return previous[right.length];
    };
    const words = expected.split(' '), heard = actual ? actual.split(' ') : [];
    const characters = Array.from(expected.replaceAll(' ', '')), heardCharacters = Array.from(actual.replaceAll(' ', ''));
    const wordErrors = distance(words, heard), characterErrors = distance(characters, heardCharacters);
    return {wordErrors, words: words.length, wordErrorRate: wordErrors / words.length,
        characterErrors, characters: characters.length, characterErrorRate: characterErrors / characters.length};
}

export async function evaluateTtsProfiles({profiles, fixtures, options, workerFactory, onAudio, onCase}) {
    if (!profiles?.length || !fixtures?.length) throw new Error('Speech evaluation requires profiles and fixtures');
    const results = [];
    for (const profile of profiles) {
        const cases = [];
        const measured = await benchmarkWorker({
            kind: 'tts', workerFactory: () => workerFactory('tts'),
            load: {source: '/models/omnivoice/manifest.json', backend: 'webgpu', threads: 1, cacheOnly: true, ...profile},
            request: fixtures.map(fixture => ({type: 'synthesize', text: fixture.text,
                options: {...options, language: fixture.language, use_voice_prompt: false,
                    cache_voice_prompt: false, return_tokens: true}})),
            warmups: 0, trials: fixtures.length,
            onResult: async (data, {trial, result: timing}) => {
                const fixture = fixtures[trial], pcm = new Float32Array(data.audio);
                if (!pcm.length || !Number.isInteger(data.sample_rate) || data.sample_rate <= 0)
                    throw new Error('Speech evaluation received invalid PCM');
                let energy = 0, peak = 0;
                for (const sample of pcm) {
                    if (!Number.isFinite(sample)) throw new Error('Speech evaluation received non-finite PCM');
                    energy += sample * sample;
                    peak = Math.max(peak, Math.abs(sample));
                }
                const rms = Math.sqrt(energy / pcm.length);
                if (rms < 1e-4) throw new Error('Speech evaluation received silent PCM');
                await onAudio?.(profile, fixture, data);
                const context = new OfflineAudioContext(1, Math.ceil(pcm.length * 16000 / data.sample_rate), 16000);
                const buffer = context.createBuffer(1, pcm.length, data.sample_rate);
                buffer.copyToChannel(pcm, 0);
                const source = context.createBufferSource();
                source.buffer = buffer;
                source.connect(context.destination);
                source.start();
                const audio = await context.startRendering();
                const recognition = await benchmarkWorker({
                    kind: 'asr', workerFactory: () => workerFactory('asr'),
                    load: {source: 'openai/whisper-tiny', backend: 'webgpu', threads: 1, cacheOnly: true},
                    request: {type: 'transcribe', audio: audio.getChannelData(0).slice().buffer,
                        language: fixture.asrLanguage, maximumTokens: 128},
                    warmups: 0, trials: 1,
                });
                const transcript = recognition.results[0].text || '';
                if (data.gpuStats?.totalBufferBytes > data.gpuStats?.memoryBudgetBytes)
                    throw new Error('Speech evaluation exceeded the GPU buffer budget');
                const measuredCase = {fixture, transcript, ...transcriptionErrorRates(fixture.text, transcript), rms, peak,
                    sharedGpuBufferBytes: data.gpuStats?.totalBufferBytes, elapsedMs: timing.elapsedMs, rtf: timing.rtf,
                    samples: timing.samples, tokenHash: timing.tokenHash, audioHash: timing.audioHash};
                cases.push(measuredCase);
                await onCase?.(profile, measuredCase);
            },
        });
        results.push({profile, model: measured.load, cases});
    }
    return {version: 1, options, results,
        measurement: 'Fresh-input wall times exclude ASR verification. ASR backtranscription is a quality proxy, not a listening score.'};
}
