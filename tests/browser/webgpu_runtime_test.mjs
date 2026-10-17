import test from 'node:test';
import assert from 'node:assert/strict';
import {setImmediate} from 'node:timers/promises';
import {createWebGpu} from '../../web/hosts/wasm/webgpu.mjs';

function deferred() {
    let resolve, reject;
    const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
    return {promise, resolve, reject};
}

async function withGpu(action, options = {}) {
    const handlers = new Map();
    const device = {
        features: new Set(), lost: new Promise(() => {}),
        limits: {minUniformBufferOffsetAlignment: 256, maxStorageBufferBindingSize: 1024 * 1024},
        queue: {
            writeBuffer(buffer, offset, data) {
                new Uint8Array(buffer.data, offset, data.byteLength)
                    .set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
            },
            submit() {},
            onSubmittedWorkDone: async () => {},
        },
        addEventListener(name, handler) { handlers.set(name, handler); },
        createBuffer({size}) {
            return {size, data: new ArrayBuffer(size), destroyed: false,
                destroy() { this.destroyed = true; },
                mapAsync: async () => {},
                getMappedRange() { return this.data; },
                unmap() {}};
        },
        createCommandEncoder() {
            return {
                clearBuffer(buffer) { new Uint8Array(buffer.data).fill(0); },
                copyBufferToBuffer(source, from, target, to, bytes) {
                    new Uint8Array(target.data, to, bytes).set(new Uint8Array(source.data, from, bytes));
                },
                finish() { return {}; },
            };
        },
        pushErrorScope() {},
        popErrorScope: async () => ({message: 'Invalid pipeline'}),
        createShaderModule() {
            return {getCompilationInfo: async () => ({
                messages: [{type: 'error', lineNum: 7, linePos: 2, message: 'Unknown shader identifier'}],
            })};
        },
        createBindGroupLayout() { return {}; },
        createPipelineLayout() { return {}; },
        createComputePipeline() { return {}; },
        destroy() { this.destroyed = true; },
    };
    const adapter = {info: {}, features: new Set(), limits: device.limits, requestDevice: async () => device};
    const globals = {
        navigator: {gpu: {requestAdapter: async () => adapter}},
        GPUBufferUsage: {STORAGE: 1, COPY_SRC: 2, COPY_DST: 4, UNIFORM: 8, MAP_READ: 16},
        GPUMapMode: {READ: 1},
        GPUShaderStage: {COMPUTE: 1},
    };
    const previous = new Map(Object.keys(globals).map(key => [key, Object.getOwnPropertyDescriptor(globalThis, key)]));
    let gpu;
    try {
        for (const [key, value] of Object.entries(globals))
            Object.defineProperty(globalThis, key, {configurable: true, value});
        const module = {HEAPU8: new Uint8Array(32)};
        gpu = await createWebGpu(module, options);
        await action(gpu, module, device, handlers);
    } finally {
        gpu?.dispose();
        for (const [key, descriptor] of previous)
            if (descriptor) Object.defineProperty(globalThis, key, descriptor);
            else delete globalThis[key];
    }
}

test('GPU buffer quotas include uniforms, reuse pooled storage, and track model ownership', async () => {
    await withGpu(async gpu => {
        const first = gpu.withOwner('first', () => gpu.allocate(16384));
        const second = gpu.withOwner('second', () => gpu.allocate(16384));
        assert.equal(gpu.stats.totalBufferBytes, 65536);
        assert.equal(gpu.ownedBytes('first'), 16384);
        assert.throws(() => gpu.allocate(4), /budget exceeded/);
        gpu.release(first);
        assert.equal(gpu.ownedBytes('first'), 0);
        gpu.forgetOwner('first');
        const reused = gpu.withOwner('second', () => gpu.allocate(16384));
        assert.equal(gpu.ownedBytes('second'), 32768);
        assert.equal(gpu.stats.totalBufferBytes, 65536);
        assert.throws(() => gpu.forgetOwner('second'), /retained/);
        gpu.release(second);
        gpu.release(reused);
        gpu.forgetOwner('second');
        gpu.trimBuffers(0);
        assert.equal(gpu.stats.totalBufferBytes, 32768);
        assert.throws(() => gpu.withOwner('temporary', () => { throw new Error('test'); }), /test/);
        assert.equal(gpu.owner, null);
        gpu.dispose();
        assert.equal(gpu.stats.totalBufferBytes, 0);
    }, {memoryBudgetBytes: 65536});
});

test('a failed read waits for sibling destinations before returning an error', async () => {
    await withGpu(async (gpu, module) => {
        const source = gpu.allocate(8);
        gpu.upload(source, 0, new Uint8Array([1, 2, 3, 4, 5, 6, 7, 8]));
        gpu.readLater(source, 0, 0, 4);
        gpu.readLater(source, 4, 4, 4);
        gpu.readbacks[0].staging.mapAsync = async () => { throw new Error('Read map failed'); };
        const second = deferred();
        gpu.readbacks[1].staging.mapAsync = () => second.promise;
        let settled = false;
        const completion = gpu.synchronize();
        completion.then(() => { settled = true; }, () => { settled = true; });
        await setImmediate();
        assert.equal(settled, false);
        second.resolve();
        await assert.rejects(completion, /Read map failed/);
        assert.deepEqual([...module.HEAPU8.subarray(4, 8)], [5, 6, 7, 8]);
        gpu.dispose();
        assert.equal(gpu.stats.readbackBytes, 0);
    });
});

test('prepared representations of admitted models survive pressure trimming', async () => {
    await withGpu(async gpu => {
        const handle = gpu.withOwner('tts', () => gpu.allocate(64));
        gpu.preparedWeights.set('voice', {handle, owner: 'tts', sources: [], bytes: 64});
        gpu.stats.preparedWeightBytes = 64;
        gpu.trimPreparedWeights(0, new Set(['tts']));
        assert.equal(gpu.stats.preparedWeightBytes, 64);
        assert.equal(gpu.ownedBytes('tts'), 64);
        gpu.trimPreparedWeights(0);
        assert.equal(gpu.stats.preparedWeightBytes, 0);
        assert.equal(gpu.ownedBytes('tts'), 0);
    });
});

test('disposing during readback cannot write into a former model destination', async () => {
    await withGpu(async (gpu, module) => {
        const source = gpu.allocate(4);
        gpu.upload(source, 0, new Uint8Array([1, 2, 3, 4]));
        gpu.readLater(source, 0, 0, 4);
        const mapping = deferred();
        gpu.readbacks[0].staging.mapAsync = () => mapping.promise;
        const completion = gpu.synchronize();
        gpu.dispose();
        mapping.resolve();
        await assert.rejects(completion, /disposed/);
        assert.deepEqual([...module.HEAPU8.subarray(0, 4)], [0, 0, 0, 0]);
        assert.equal(gpu.stats.totalBufferBytes, 0);
    });
});

test('direct JavaScript readback does not overwrite the Wasm heap', async () => {
    await withGpu(async (gpu, module) => {
        const source = gpu.allocate(4), destination = new Uint8Array(4);
        gpu.upload(source, 0, new Uint8Array([9, 8, 7, 6]));
        await gpu.read(source, 0, destination, 4);
        assert.deepEqual([...destination], [9, 8, 7, 6]);
        assert.deepEqual([...module.HEAPU8.subarray(0, 4)], [0, 0, 0, 0]);
        assert.throws(() => gpu.readLater(source, 0, new Float32Array(1), 4), /destination/);
        assert.throws(() => gpu.allocate(-1), /size/);
    });
});

test('shader diagnostics survive later generic command-buffer errors', async () => {
    await withGpu(async (gpu, _module, _device, handlers) => {
        gpu.pipeline('invalid WGSL', 0);
        handlers.get('uncapturederror')({error: {message: 'Invalid CommandBuffer'}});
        await assert.rejects(gpu.synchronize(), /7:2 Unknown shader identifier/);
    });
});
