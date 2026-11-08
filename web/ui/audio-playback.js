export function resample(samples, sourceRate, targetRate) {
  if (sourceRate === targetRate) return samples;
  const output = new Float32Array(Math.max(1, Math.floor(samples.length * targetRate / sourceRate)));
  const scale = sourceRate / targetRate;
  for (let index = 0; index < output.length; index++) {
    const position = index * scale;
    const left = Math.floor(position);
    const fraction = position - left;
    output[index] = samples[left] * (1 - fraction) +
      samples[Math.min(left + 1, samples.length - 1)] * fraction;
  }
  return output;
}
