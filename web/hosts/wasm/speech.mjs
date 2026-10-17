const TARGET_RATE = 16000;
const MAXIMUM_SAMPLES = 30 * TARGET_RATE;

export function resampleAudio(chunks, sourceRate) {
    if (!Number.isFinite(sourceRate) || sourceRate <= 0 || !Array.isArray(chunks) ||
        chunks.some(chunk => !(chunk instanceof Float32Array))) throw new Error('Invalid captured audio');
    const length = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
    const input = new Float32Array(length);
    let offset = 0;
    for (const chunk of chunks) { input.set(chunk, offset); offset += chunk.length; }
    if (sourceRate === TARGET_RATE) return input.subarray(0, MAXIMUM_SAMPLES).slice();
    const outputLength = Math.min(MAXIMUM_SAMPLES, Math.floor(input.length * TARGET_RATE / sourceRate));
    const output = new Float32Array(outputLength);
    const scale = sourceRate / TARGET_RATE;
    for (let index = 0; index < output.length; index++) {
        const position = index * scale;
        const left = Math.floor(position);
        const fraction = position - left;
        output[index] = input[left] * (1 - fraction) + (input[Math.min(left + 1, input.length - 1)] || 0) * fraction;
    }
    return output;
}