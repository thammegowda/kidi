const finitePositive = (value, fallback) =>
  Number.isFinite(Number(value)) && Number(value) > 0 ? Number(value) : fallback;
// Round-trip tests become semantically unstable below this floor; eight steps can already repeat words.
export const MINIMUM_TTS_STEPS = 10;

function cpuTier(cores, memory) {
  if (cores <= 4 || memory <= 4) return 'constrained';
  if (cores <= 8 || memory <= 8) return 'balanced';
  return 'performance';
}

export function selectRealtimeProfile({
  runtime = {},
  backend = '',
  components = {},
  hardwareConcurrency = globalThis.navigator?.hardwareConcurrency,
  deviceMemory = globalThis.navigator?.deviceMemory,
} = {}) {
  const host = runtime.host === 'wasm' ? 'wasm' : 'native';
  const cores = Math.max(1, Math.floor(finitePositive(hardwareConcurrency, runtime.threads || 4)));
  const memory = finitePositive(deviceMemory, Number.POSITIVE_INFINITY);
  const configuredSpeech = String(components.tts?.accelerator || '').toLowerCase();
  const reportedSpeech = String(runtime.speech_backend || '').toLowerCase();
  const requested = String(runtime.requested_backend || backend || '').toLowerCase();
  const execution = configuredSpeech && configuredSpeech !== 'auto'
    ? configuredSpeech : reportedSpeech || requested;
  const gpu = /(?:web.?gpu|a_gpu|metal|mps|vulkan|cuda|npu)/u.test(execution) &&
    !/(?:cpu|wasm-threads)/u.test(execution);

  if (gpu) {
    const web = host === 'wasm';
    const constrainedWeb = web && (cores <= 4 || memory <= 4);
    return {
      id: web ? 'wasm-webgpu' : 'native-gpu',
      host,
      execution: 'gpu',
      cores,
      endpointMs: web ? (constrainedWeb ? 650 : 700) : 800,
      ttsSteps: MINIMUM_TTS_STEPS,
      segmentation: web ? {
        firstMaximumUnits: constrainedWeb ? 96 : 112,
        firstTargetUnits: constrainedWeb ? 36 : 40,
        nextTargetUnits: constrainedWeb ? 112 : 128,
        maximumUnits: constrainedWeb ? 160 : 176,
      } : {
        firstMaximumUnits: 128,
        firstTargetUnits: 48,
        nextTargetUnits: 160,
        maximumUnits: 192,
      },
    };
  }

  const tier = cpuTier(cores, memory);
  if (tier === 'constrained') {
    return {
      id: `${host}-cpu-constrained`,
      host,
      execution: 'cpu',
      cores,
      endpointMs: 700,
      ttsSteps: MINIMUM_TTS_STEPS,
      segmentation: {
        firstMaximumUnits: 96,
        firstTargetUnits: 36,
        nextTargetUnits: 112,
        maximumUnits: 160,
      },
    };
  }
  if (tier === 'balanced') {
    return {
      id: `${host}-cpu-balanced`,
      host,
      execution: 'cpu',
      cores,
      endpointMs: 750,
      ttsSteps: MINIMUM_TTS_STEPS,
      segmentation: {
        firstMaximumUnits: 112,
        firstTargetUnits: 40,
        nextTargetUnits: 128,
        maximumUnits: 176,
      },
    };
  }
  return {
    id: `${host}-cpu-performance`,
    host,
    execution: 'cpu',
    cores,
    endpointMs: 800,
    ttsSteps: MINIMUM_TTS_STEPS,
    segmentation: {
      firstMaximumUnits: 128,
      firstTargetUnits: 48,
      nextTargetUnits: 144,
      maximumUnits: 192,
    },
  };
}
