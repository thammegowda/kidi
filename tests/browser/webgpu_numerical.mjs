import {createWebGpu} from '../../web/hosts/wasm/webgpu.mjs';

const roundEven = value => {
    const low = Math.floor(value), fraction = value - low;
    return low + Number(fraction > 0.5 || (fraction === 0.5 && low % 2 !== 0));
};

function affineActivations(input, rows, width) {
    const values = new Int8Array(input.length), scales = new Float32Array(rows), zeros = new Int32Array(rows);
    for (let row = 0; row < rows; ++row) {
        const source = input.subarray(row * width, (row + 1) * width);
        const low = Math.min(0, ...source), high = Math.max(0, ...source);
        scales[row] = high > low ? Math.fround(Math.fround(high - low) / 255) : 1;
        zeros[row] = high > low ? Math.max(-128, Math.min(127,
            roundEven(Math.fround(-128 - Math.fround(low / scales[row]))))) : 0;
        source.forEach((value, channel) => {
            values[row * width + channel] = Math.max(-128, Math.min(127,
                roundEven(Math.fround(value / scales[row])) + zeros[row]));
        });
    }
    return {values, scales, zeros};
}

export async function runNumericalChecks(options = {}) {
    const heap = new Uint8Array(2 * 1024 * 1024);
    const gpu = await createWebGpu({HEAPU8: heap}, options);
    if (options.portable) gpu.features.clear();
    const passed = [];
    const tensor = (shape, dtype) => ({shape, dtype});
    const run = async (spec, arrays, expected, tolerance = 1e-4, offsets = []) => {
        const handles = arrays.map((array, index) => gpu.allocate(array.byteLength + (offsets[index] || 0)));
        const output = gpu.allocate(expected.length * 4);
        const program = gpu.prepare(spec);
        try {
            arrays.forEach((array, index) => {
                let bytes = new Uint8Array(array.buffer, array.byteOffset, array.byteLength);
                if (offsets[index]) {
                    const padded = new Uint8Array(bytes.byteLength + offsets[index]);
                    padded.set(bytes, offsets[index]);
                    bytes = padded;
                }
                gpu.upload(handles[index], 0, bytes);
            });
            gpu.run(program, new Uint32Array([...handles, output].flatMap((handle, index) => [handle, offsets[index] || 0])));
            await gpu.read(output, 0, 0, expected.length * 4);
            const actual = expected instanceof Int32Array
                ? new Int32Array(heap.buffer, 0, expected.length)
                : new Float32Array(heap.buffer, 0, expected.length);
            for (let index = 0; index < expected.length; ++index) {
                if (!Number.isFinite(actual[index]) ||
                    Math.abs(actual[index] - expected[index]) > tolerance * Math.max(1, Math.abs(expected[index])))
                    throw new Error(`${spec.operation} mismatch at ${index}: ${actual[index]} versus ${expected[index]}`);
            }
            passed.push(`${spec.operation}/${spec.inputs[0].shape.join('x')}`);
        } finally {
            gpu.releaseProgram(program);
            for (const handle of handles) gpu.release(handle);
            gpu.release(output);
        }
    };
    try {
        for (const [rows, width] of [[1, 17], [1, 64], [8, 17], [8, 64], [65, 64], [8, 128]]) {
            const columns = rows >= 65 ? 71 : 19;
            const w8a8 = gpu.matrixPrecision === 'w8a8' && rows >= 4 && width % 4 === 0;
            const input = Float32Array.from({length: rows * width}, (_, index) => (index % 13 - 6) / 8);
            if (w8a8) for (let row = 0; row < rows; ++row) {
                for (let channel = 0; channel < width; ++channel) {
                    input[row * width + channel] = row % 4 === 0 ? 0 : row % 4 === 1
                        ? (channel === 1 ? 255 : channel % 17 / 4)
                        : row % 4 === 2 ? (channel === 0 ? -255 : -(channel % 17) / 4)
                        : channel === 0 ? -127 : channel === 1 ? 128 : channel % 17 / 4 - 2;
                }
            }
            const weight = Int8Array.from({length: width * columns}, (_, index) => index % 127 - 63);
            const scale = Float32Array.from({length: columns}, (_, index) => rows >= 65 ? 1 / 64 : (index + 1) / 1024);
            const bias = Float32Array.from({length: columns}, (_, index) => index / 100);
            const quantized = w8a8 ? affineActivations(input, rows, width) : null;
            const expected = Float32Array.from({length: rows * columns}, (_, index) => {
                const row = Math.floor(index / columns), column = index % columns;
                let sum = 0;
                for (let channel = 0; channel < width; ++channel)
                    sum += quantized ? (quantized.values[row * width + channel] - quantized.zeros[row]) *
                        weight[column * width + channel]
                        : input[row * width + channel] * weight[column * width + channel] * scale[column];
                return (quantized ? Math.fround(Math.fround(sum * quantized.scales[row]) * scale[column]) : sum) + bias[column];
            });
            const spec = {operation: 'quantized_linear', dtype: 11, epsilon: 0, attributes: [1],
                inputs: [tensor([1, rows, width], 11), tensor([columns, width], 2),
                    tensor([columns, 1], 11), tensor([columns], 11)]};
            await run(spec, [input, weight, scale, bias], expected);
            if (w8a8) await run(spec, [input, weight, scale, bias], expected, 1e-4, [4, 1, 4, 0]);
        }
        if (gpu.matrixPrecision === 'w8a8') {
            const rows = 4, width = 3072, columns = 3;
            const input = new Float32Array(rows * width).fill(255);
            const weight = Int8Array.from({length: columns * width}, (_, index) => index % 4 === 3 ? 126 : 127);
            await run({operation: 'quantized_linear', dtype: 11, epsilon: 0, attributes: [1],
                inputs: [tensor([1, rows, width], 11), tensor([columns, width], 2),
                    tensor([columns, 1], 11), tensor([columns], 11)]},
            [input, weight, new Float32Array(columns).fill(1 / 64), new Float32Array(columns)],
            new Float32Array(rows * columns).fill(255 * (width / 4) * 507 / 64), 1e-5);
        }
        if (gpu.matrixPrecision === 'w8a8' || (gpu.matrixPrecision === 'mixed-f16' && gpu.features.has('shader-f16'))) {
            const budget = gpu.preparedWeightBudget;
            const preparedBytes = gpu.matrixPrecision === 'w8a8' ? 20 : 640;
            gpu.preparedWeightBudget = preparedBytes;
            const arrays = [new Float32Array(8 * 64).fill(0.5), new Int8Array(5 * 64).fill(2),
                new Float32Array(5).fill(0.25), new Float32Array(5).fill(0.5), new Int8Array(5 * 64).fill(3)];
            const handles = arrays.map(array => gpu.allocate(array.byteLength));
            const output = gpu.allocate(8 * 5 * 4);
            const program = gpu.prepare({operation: 'quantized_linear', dtype: 11, epsilon: 0, attributes: [1],
                inputs: [tensor([1, 8, 64], 11), tensor([5, 64], 2), tensor([5, 1], 11), tensor([5], 11)]});
            const before = {...gpu.stats};
            try {
                arrays.forEach((array, index) => gpu.upload(handles[index], 0, new Uint8Array(array.buffer)));
                const check = async (weight, expected) => {
                    gpu.run(program, new Uint32Array([handles[0], weight, handles[2], handles[3], output]
                        .flatMap(handle => [handle, 0])));
                    await gpu.read(output, 0, 0, 8 * 5 * 4);
                    if (new Float32Array(heap.buffer, 0, 8 * 5).some(value => Math.abs(value - expected) > 1e-4))
                        throw new Error('Prepared matrix cache returned stale or incorrect weights');
                };
                await check(handles[1], 16.5);
                await check(handles[1], 16.5);
                arrays[1].fill(4);
                gpu.upload(handles[1], 0, new Uint8Array(arrays[1].buffer));
                await check(handles[1], 32.5);
                await check(handles[4], 24.5);
                await check(handles[1], 32.5);
                if (gpu.stats.preparedWeightHits - before.preparedWeightHits !== 1 ||
                    gpu.stats.preparedWeightMisses - before.preparedWeightMisses !== 4 ||
                    gpu.stats.preparedWeightEvictions - before.preparedWeightEvictions !== 2 ||
                    gpu.stats.preparedWeightBytes !== preparedBytes)
                    throw new Error('Prepared matrix cache did not respect versioning, LRU, or byte limits');
                passed.push(`${gpu.matrixPrecision}/cache-reuse-version-eviction`);
            } finally {
                gpu.releaseProgram(program);
                for (const handle of handles) gpu.release(handle);
                gpu.release(output);
                gpu.preparedWeightBudget = budget;
            }
            if (gpu.stats.preparedWeightBytes !== 0)
                throw new Error('Releasing model weights retained an FP16 representation');
        }
        const first = Float32Array.from({length: 53}, (_, index) => (index - 26) / 10);
        const second = Float32Array.from({length: 53}, (_, index) => index / 53);
        await run({operation: 'silu_multiply', dtype: 11, epsilon: 0, attributes: [],
            inputs: [tensor([53], 11), tensor([53], 11)]}, [first, second],
            Float32Array.from(first, (value, index) => value / (1 + Math.exp(-value)) * second[index]));

        const target = 3, books = 2, vocabulary = 17, mask = 16;
        const logits = Float32Array.from({length: target * books * vocabulary * 2},
            (_, index) => (index % 19 - 9) / 8);
        const expectedCandidates = new Float32Array(target * books * 2);
        for (let row = 0; row < target * books; ++row) {
            const values = Array.from({length: vocabulary}, (_, token) => {
                const index = row * vocabulary + token;
                return logits[index] + 2 * (logits[index] - logits[index + target * books * vocabulary]);
            });
            const maximum = Math.max(...values);
            const normalization = maximum + Math.log(values.reduce((sum, value) => sum + Math.exp(value - maximum), 0));
            let selected = 0;
            for (let token = 1; token < mask; ++token) if (values[token] > values[selected]) selected = token;
            expectedCandidates[row * 2] = selected;
            expectedCandidates[row * 2 + 1] = values[selected] - normalization;
        }
        await run({operation: 'diffusion_candidates', dtype: 11, epsilon: 2,
            attributes: [vocabulary, mask, books, target],
            inputs: [tensor([1, 2 * target, books * vocabulary], 11)]}, [logits], expectedCandidates);

        const tokens = new Int32Array([mask, mask, mask, 7, mask, mask]);
        const candidateValues = Float32Array.from({length: target * books * 2},
            (_, index) => index % 2 ? -Math.floor(index / 2) / 10 : Math.floor(index / 2) + 1);
        for (const temperature of [0, 1]) {
            const noise = new Float32Array([0, 0.1, 0.2, 0.3, 0.4]);
            const active = [];
            let draw = 0;
            for (let row = 0; row < target * books; ++row) {
                const index = (row % books) * target + Math.floor(row / books);
                if (tokens[index] !== mask) continue;
                active.push({row, index, score: candidateValues[row * 2 + 1] +
                    (temperature ? noise[draw] : 0)});
                draw++;
            }
            active.sort((a, b) => b.score - a.score || a.row - b.row);
            const expected = tokens.slice();
            for (const entry of active.slice(0, 2)) expected[entry.index] = candidateValues[entry.row * 2];
            const bits = value => new Uint32Array(new Float32Array([value]).buffer)[0];
            await run({operation: 'diffusion_update', dtype: 6, epsilon: 0,
                attributes: [2, mask, bits(0), bits(temperature)],
                inputs: [tensor([books, target], 6), tensor([target * books, 2], 11), tensor([5], 11)]},
            [tokens, candidateValues, noise], expected, 0);
        }

        const codes = new Int32Array([0, 1, 4, 2, 3, 4]);
        const weights = Int8Array.from({length: 10 * 4}, (_, index) => index % 13 - 6);
        const scales = Float32Array.from({length: 10}, (_, index) => (index + 1) / 16);
        const summed = Float32Array.from({length: 12}, (_, index) => {
            const position = Math.floor(index / 4), channel = index % 4;
            let sum = 0;
            for (let book = 0; book < 2; ++book) {
                const row = book * 5 + codes[book * 3 + position];
                sum += weights[row * 4 + channel] * scales[row];
            }
            return sum;
        });
        await run({operation: 'embedding_sum', dtype: 11, epsilon: 1, attributes: [4, 8, 1, 5],
            inputs: [tensor([2, 3], 6), tensor([10, 4], 2), tensor([10, 1], 11)]},
        [codes, weights, scales], summed);

        for (const dilation of [1, 2]) {
            const length = 7, channels = 2, outputs = 3, kernel = 3, stride = 2, padding = 1;
            const input = Float32Array.from({length: length * channels}, (_, index) => (index - 5) / 8);
            const weight = Int8Array.from({length: outputs * channels * kernel}, (_, index) => index % 7 - 3);
            const scales = new Float32Array([0.125, 0.25, 0.5]), bias = new Float32Array([0.25, -0.5, 0]);
            const rows = Math.floor((length + 2 * padding - dilation * (kernel - 1) - 1) / stride) + 1;
            const expected = Float32Array.from({length: rows * outputs}, (_, index) => {
                const row = Math.floor(index / outputs), column = index % outputs;
                let sum = bias[column];
                for (let channel = 0; channel < channels; ++channel)
                    for (let k = 0; k < kernel; ++k) {
                        const position = row * stride - padding + k * dilation;
                        if (position >= 0 && position < length)
                            sum += input[position * channels + channel] *
                                weight[(column * channels + channel) * kernel + k] * scales[column];
                    }
                return sum;
            });
            await run({operation: 'conv1d', dtype: 11, epsilon: 0, attributes: [kernel, stride, padding, dilation],
                inputs: [tensor([1, length, channels], 11), tensor([outputs, channels * kernel], 2),
                    tensor([outputs, 1], 11), tensor([outputs], 11)]}, [input, weight, scales, bias], expected);
        }

        const projected = Float32Array.from({length: 3 * 2 * 3}, (_, index) => index / 8);
        const offsets = new Float32Array([0.25, -0.5]);
        const folded = new Float32Array(6 * 2);
        for (let row = 0; row < 6; ++row) folded.set(offsets, row * 2);
        for (let source = 0; source < 3; ++source)
            for (let k = 0; k < 3; ++k) {
                const position = source * 2 - 1 + k;
                if (position >= 0 && position < 6)
                    for (let channel = 0; channel < 2; ++channel)
                        folded[position * 2 + channel] += projected[(source * 2 + channel) * 3 + k];
            }
        await run({operation: 'conv_transpose_fold', dtype: 11, epsilon: 0, attributes: [3, 2, 1, 1, 1],
            inputs: [tensor([1, 3, 6], 11), tensor([2], 11)]}, [projected, offsets], folded);
        const activations = new Float32Array([-2, -1, 1, 2, 0, 0.5, -0.5, 1.5]);
        const alpha = new Float32Array([0, 1e-10, 1.25, 2]);
        await run({operation: 'snake', dtype: 11, epsilon: 0, attributes: [],
            inputs: [tensor([1, 2, 4], 11), tensor([4], 11)]}, [activations, alpha],
        Float32Array.from(activations, (value, index) => Math.abs(alpha[index % 4]) < 1e-9
            ? value : value + Math.sin(alpha[index % 4] * value) ** 2 / (alpha[index % 4] + 1e-9)));
        await run({operation: 'gather', dtype: 11, epsilon: 0, attributes: [],
            inputs: [tensor([3, 4], 11), tensor([2], 6)]},
        [new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]), new Int32Array([2, 0])],
        new Float32Array([9, 10, 11, 12, 1, 2, 3, 4]));
        for (const [width, keys] of [[8, 7], [64, 1], [64, 17], [128, 17], [256, 9], [512, 9]])
        for (const bytes of [false, true]) {
            const batches = 2, queries = 5, heads = 4, keyHeads = 2, capacity = keys + 2;
            const keyBatches = bytes ? batches : 1;
            const query = Float32Array.from({length: batches * queries * heads * width},
                (_, index) => (index % 17 - 8) / 16);
            const key = bytes ? Int8Array.from({length: keyBatches * capacity * keyHeads * width}, (_, index) => index % 11 - 5)
                : Float32Array.from({length: capacity * keyHeads * width}, (_, index) => (index % 11 - 5) / 8);
            const value = bytes ? Int8Array.from({length: key.length}, (_, index) => index % 13 - 6)
                : Float32Array.from({length: key.length}, (_, index) => (index % 13 - 6) / 8);
            const mask = Float32Array.from({length: batches * queries * keys}, (_, index) => {
                const batch = Math.floor(index / (queries * keys)), row = Math.floor(index / keys) % queries;
                if (batch === 1 && row === queries - 1) return -Infinity;
                return index % keys > row + 1 ? -1e9 : 0;
            });
            const quantization = [
                {scales: Array.from({length: width / 4}, (_, index) => index % 2 ? 0.25 : 0.125),
                    zero_points: Array.from({length: width / 4}, (_, index) => index % 2 ? -2 : 1), block_size: 4},
                {scales: Array.from({length: width / 4}, (_, index) => index % 2 ? 0.125 : 0.25),
                    zero_points: Array.from({length: width / 4}, (_, index) => index % 2 ? 2 : -1), block_size: 4},
            ];
            const read = (array, index, channel, side) => bytes
                ? (array[index] - quantization[side].zero_points[Math.floor(channel / 4)]) *
                    quantization[side].scales[Math.floor(channel / 4)]
                : array[index];
            const expected = new Float32Array(query.length);
            for (let batch = 0; batch < batches; ++batch)
                for (let row = 0; row < queries; ++row)
                    for (let head = 0; head < heads; ++head) {
                        const scores = [];
                        const keyHead = Math.floor(head / (heads / keyHeads));
                        for (let position = 0; position < keys; ++position) {
                            let score = 0;
                            for (let channel = 0; channel < width; ++channel)
                                score += query[((batch * queries + row) * heads + head) * width + channel] *
                                    read(key, (((batch % keyBatches) * capacity + position + 1) * keyHeads + keyHead) *
                                        width + channel, channel, 0);
                            scores.push(score / Math.sqrt(width) + mask[(batch * queries + row) * keys + position]);
                        }
                        const maximum = Math.max(...scores);
                        if (!Number.isFinite(maximum)) continue;
                        const probabilities = scores.map(score => Math.exp(score - maximum));
                        const total = probabilities.reduce((sum, p) => sum + p, 0);
                        for (let channel = 0; channel < width; ++channel)
                            for (let position = 0; position < keys; ++position)
                                expected[((batch * queries + row) * heads + head) * width + channel] +=
                                    probabilities[position] / total *
                                    read(value, (((batch % keyBatches) * capacity + position + 1) * keyHeads + keyHead) *
                                        width + channel, channel, 1);
                    }
            await run({operation: 'attention', dtype: 11, epsilon: 1 / Math.sqrt(width), onlineAttention: true,
                attributes: [heads, keyHeads, 1, keys], quantization: bytes ? quantization : [],
                inputs: [tensor([batches, queries, heads * width], 11),
                    tensor([keyBatches, capacity, keyHeads * width], bytes ? 2 : 11),
                    tensor([keyBatches, capacity, keyHeads * width], bytes ? 2 : 11),
                    tensor([batches, 1, queries, keys], 11)]}, [query, key, value, mask], expected);
        }

        const left = gpu.allocate(16), right = gpu.allocate(16);
        const add = gpu.prepare({operation: 'add', inplace: true, dtype: 11, epsilon: 0, attributes: [],
            inputs: [tensor([4], 11), tensor([4], 11)]});
        try {
            gpu.upload(left, 0, new Uint8Array(new Float32Array([1, 2, 3, 4]).buffer));
            gpu.upload(right, 0, new Uint8Array(new Float32Array([5, 6, 7, 8]).buffer));
            const region = gpu.prepareRegion({key: 'rebind', commands: [
                {program: add, bindings: [{input: 0, offset: 0}, {input: 1, offset: 0}, {input: 0, offset: 0}]},
            ]});
            try {
                for (const inputs of [[left, 0, right, 0], [right, 0, left, 0]]) {
                    gpu.runRegion(region, new Uint32Array(inputs));
                    await gpu.synchronize();
                }
                await gpu.read(right, 0, 0, 16);
                const expected = [11, 14, 17, 20];
                const actual = new Float32Array(heap.buffer, 0, 4);
                if (actual.some((value, index) => value !== expected[index]))
                    throw new Error(`Rebound in-place region produced ${[...actual]}`);
                passed.push('region/rebind-and-inplace');
            } finally { gpu.releaseRegion(region); }
        } finally {
            gpu.releaseProgram(add);
            gpu.release(left);
            gpu.release(right);
        }
        return {passed, adapter: gpu.adapter, features: [...gpu.features], stats: {...gpu.stats}};
    } finally { gpu.dispose(); }
}
