import test from 'node:test';
import assert from 'node:assert/strict';
import {VoiceLatencyBenchmark} from '../../web/ui/speech-controller.js';

test('voice latency benchmark reports user-visible stages and percentiles', () => {
    let now = 1000;
    const logged = [];
    const benchmark = new VoiceLatencyBenchmark({now: () => now, logger: sample => logged.push(sample)});
    const turn = offset => {
        benchmark.begin(now - 700);
        benchmark.mark('asr_complete', now += 300 + offset);
        benchmark.mark('chat_request', now += 25);
        benchmark.mark('first_token', now += 400);
        benchmark.mark('tts_request', now += 100);
        benchmark.mark('first_pcm', now += 500);
        return benchmark.complete(now += 20);
    };
    assert.deepEqual(turn(0), {
        endpoint_ms: 700,
        transcription_ms: 300,
        dispatch_ms: 25,
        first_token_ms: 1425,
        text_to_tts_ms: 100,
        tts_first_pcm_ms: 500,
        playback_queue_ms: 20,
        end_to_end_ms: 2045,
    });
    turn(100);
    assert.equal(logged.length, 2);
    assert.deepEqual(benchmark.summary().end_to_end_ms, {p50: 2045, p95: 2145});
    benchmark.clear();
    assert.deepEqual(benchmark.summary(), {turns: 0});
});

test('cancelled voice turns do not produce samples', () => {
    const benchmark = new VoiceLatencyBenchmark({now: () => 10, logger() {}});
    benchmark.begin(5);
    benchmark.cancel();
    assert.equal(benchmark.complete(), null);
    assert.equal(benchmark.samples.length, 0);
});
