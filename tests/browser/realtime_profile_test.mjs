import test from 'node:test';
import assert from 'node:assert/strict';
import {selectRealtimeProfile} from '../../web/ui/realtime-profile.js';

test('native GPU uses the measured six-step low-latency profile', () => {
  assert.deepEqual(selectRealtimeProfile({
    runtime: {host: 'native', threads: 4, speech_backend: 'a_gpu'},
    backend: 'a_gpu',
    hardwareConcurrency: 12,
  }), {
    id: 'native-gpu',
    host: 'native',
    execution: 'gpu',
    cores: 12,
    endpointMs: 800,
    ttsSteps: 6,
    segmentation: {
      firstMaximumUnits: 128,
      firstTargetUnits: 48,
      nextTargetUnits: 160,
      maximumUnits: 192,
    },
  });
});

test('Wasm WebGPU dispatches independently of CPU thread count', () => {
  const profile = selectRealtimeProfile({
    runtime: {host: 'wasm', requested_backend: 'webgpu', speech_backend: 'webgpu', threads: 4},
    hardwareConcurrency: 4,
    deviceMemory: 4,
  });
  assert.equal(profile.id, 'wasm-webgpu');
  assert.equal(profile.execution, 'gpu');
  assert.equal(profile.ttsSteps, 4);
  assert.equal(profile.segmentation.firstMaximumUnits, 96);
});

test('higher-capacity WebGPU retains a five-step realtime profile', () => {
  const profile = selectRealtimeProfile({
    runtime: {host: 'wasm', requested_backend: 'webgpu', speech_backend: 'webgpu'},
    hardwareConcurrency: 8,
    deviceMemory: 8,
  });
  assert.equal(profile.id, 'wasm-webgpu');
  assert.equal(profile.ttsSteps, 5);
  assert.equal(profile.segmentation.firstMaximumUnits, 112);
});

test('constrained CPU gets smaller chunks and fewer diffusion steps', () => {
  const profile = selectRealtimeProfile({
    runtime: {host: 'wasm', requested_backend: 'cpu', speech_backend: 'cpu', threads: 4},
    components: {tts: {accelerator: 'wasm-threads'}},
    hardwareConcurrency: 4,
    deviceMemory: 4,
  });
  assert.equal(profile.id, 'wasm-cpu-constrained');
  assert.equal(profile.ttsSteps, 4);
  assert.equal(profile.endpointMs, 700);
  assert.equal(profile.segmentation.firstMaximumUnits, 96);
  assert.equal(profile.segmentation.maximumUnits, 160);
});

test('explicit native CPU overrides a GPU chat backend', () => {
  const profile = selectRealtimeProfile({
    runtime: {host: 'native', speech_backend: 'cpu', threads: 8},
    backend: 'a_gpu',
    components: {tts: {accelerator: 'cpu'}},
    hardwareConcurrency: 8,
    deviceMemory: 16,
  });
  assert.equal(profile.id, 'native-cpu-balanced');
  assert.equal(profile.execution, 'cpu');
  assert.equal(profile.ttsSteps, 5);
});

test('loaded Wasm CPU fallback overrides stale requested WebGPU', () => {
  const profile = selectRealtimeProfile({
    runtime: {host: 'wasm', requested_backend: 'webgpu', speech_backend: 'webgpu', threads: 4},
    components: {tts: {accelerator: 'wasm-threads'}},
    hardwareConcurrency: 4,
  });
  assert.equal(profile.id, 'wasm-cpu-constrained');
  assert.equal(profile.execution, 'cpu');
});
