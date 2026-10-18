import test from 'node:test';
import assert from 'node:assert/strict';
import {makeProgram} from '../../web/hosts/wasm/webgpu-kernels.mjs';

const f32 = shape => ({shape, dtype: 11});

test('quantized projections derive output shapes for scalar and tiled rows', () => {
    for (const rows of [1, 8]) {
        const plan = makeProgram({
            operation: 'quantized_linear',
            dtype: 11,
            epsilon: 0,
            attributes: [1],
            inputs: [
                f32([1, rows, 64]),
                {shape: [96, 64], dtype: 2},
                f32([96, 1]),
                f32([96]),
            ],
        }, new Set());
        assert.deepEqual(plan.shape, [1, rows, 96]);
        assert.equal(plan.dtype, 11);
        assert.ok(plan.groups > 0);
    }
});

test('W8A8 kernel variants share bounded activation scratch and binding layout', () => {
    const spec = {operation: 'quantized_linear', w8a8: true, dtype: 11, epsilon: 0, attributes: [1],
        inputs: [f32([1, 65, 64]), {shape: [71, 64], dtype: 2}, f32([71, 1]), f32([71])]};
    for (const plan of [
        makeProgram(spec, new Set(['packed-dot'])),
        makeProgram(spec, new Set()),
        makeProgram({...spec, w8a8Kernel: 'float'}, new Set(['shader-f16'])),
    ]) {
        assert.deepEqual(plan.shape, [1, 65, 71]);
        assert.equal(plan.dtype, 11);
        assert.equal(plan.scratch.bytes, 65 * (64 + 8));
        assert.equal(plan.bindings, 5);
    }
});

test('Snake projection quantizes the fused activation directly for W8A8', () => {
    const plan = makeProgram({
        operation: 'snake_quantized_linear',
        w8a8: true,
        w8a8Kernel: 'float',
        dtype: 11,
        epsilon: 0,
        attributes: [1],
        inputs: [
            f32([1, 65, 64]),
            f32([64]),
            {shape: [96, 64], dtype: 2},
            f32([96, 1]),
            f32([96]),
        ],
    }, new Set());
    assert.deepEqual(plan.shape, [1, 65, 96]);
    assert.deepEqual(plan.scratch.inputs, [0, 1]);
    assert.equal(plan.scratch.bytes, 65 * (64 + 8));
    assert.equal(plan.bindings, 5);
    assert.deepEqual(plan.finalInputs, [0, 2, 3, 4, 5]);
});

test('Snake convolution removes the standalone activation output', () => {
    const plan = makeProgram({
        operation: 'snake_conv1d',
        dtype: 11,
        epsilon: 0,
        attributes: [7, 1, 3, 1],
        inputs: [
            f32([1, 257, 64]),
            f32([64]),
            {shape: [32, 448], dtype: 2},
            f32([32, 1]),
            f32([32]),
        ],
    }, new Set());
    assert.deepEqual(plan.shape, [1, 257, 32]);
    assert.equal(plan.stages, undefined);
    assert.equal(plan.scratch, undefined);
});

test('in-place arithmetic binds only the right operand and writes the destination directly', () => {
    for (const operation of ['add', 'multiply']) {
        const plan = makeProgram({
            operation, inplace: true, dtype: 11, epsilon: 0, attributes: [],
            inputs: [f32([1, 4, 16]), f32([16])],
        }, new Set());
        assert.deepEqual(plan.shape, [1, 4, 16]);
        assert.equal(plan.bindings, 1);
        assert.deepEqual(plan.finalInputs, [1]);
    }
});

test('diffusion candidates reduce two full vocabulary arrays to compact token scores', () => {
    const plan = makeProgram({operation: 'diffusion_candidates', dtype: 11, epsilon: 2,
        attributes: [1025, 1024, 8, 43], inputs: [f32([1, 86, 8200])]}, new Set());
    assert.deepEqual(plan.shape, [344, 2]);
    assert.equal(plan.shape[0] * plan.shape[1] * 4, 2752);
});

test('online multi-query attention has no global score or probability scratch', () => {
    const plan = makeProgram({operation: 'attention', dtype: 11, epsilon: 0.125, onlineAttention: true,
        attributes: [8, 2, 0, 2048], inputs: [f32([1, 512, 512]), f32([1, 2048, 128]),
            f32([1, 2048, 128]), f32([1, 1, 512, 2048])]}, new Set());
    assert.deepEqual(plan.shape, [1, 512, 512]);
    assert.equal(plan.stages, undefined);
    assert.equal(plan.scratch, undefined);
    assert.ok(plan.groups > 0);
});
