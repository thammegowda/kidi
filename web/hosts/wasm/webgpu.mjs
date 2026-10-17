import {makeProgram} from './webgpu-kernels.mjs';

// Recorded work is submitted every UNIFORM_SLOTS dispatches, so the GPU starts on the first layers while the browser
// still records and translates the rest of the step (decode: 77 -> 85 tok/s on an Apple M5).
const UNIFORM_SLOTS = 128;
const MAX_POOLED_BYTES = 128 * 1024 * 1024;
const MAX_PREPARED_WEIGHT_BYTES = 1024 * 1024 * 1024;
const MAX_TRANSFER_BYTES = 64 * 1024 * 1024;
const MAX_RETAINED_READ_BYTES = 16 * 1024 * 1024;
export const DEFAULT_GPU_BUDGET = 3 * 1024 * 1024 * 1024;

export async function createWebGpu(module, options = {}) {
    if (options.matrixPrecision !== undefined && !['f32', 'mixed-f16', 'w8a8'].includes(options.matrixPrecision))
        throw new Error('WebGPU matrix precision must be f32, mixed-f16, or w8a8');
    if (options.w8a8Kernel !== undefined && !['packed', 'float'].includes(options.w8a8Kernel))
        throw new Error('WebGPU W8A8 kernel must be packed or float');
    if (options.preparedWeightBudget !== undefined &&
        (!Number.isSafeInteger(options.preparedWeightBudget) || options.preparedWeightBudget < 0))
        throw new Error('WebGPU prepared weight budget must be a non-negative integer');
    if (options.memoryBudgetBytes !== undefined &&
        (!Number.isSafeInteger(options.memoryBudgetBytes) || options.memoryBudgetBytes <= 0))
        throw new Error('WebGPU memory budget must be a positive integer');
    if (!navigator.gpu) throw new Error('This browser has no WebGPU');
    const adapter = await navigator.gpu.requestAdapter({powerPreference: 'high-performance'});
    if (!adapter || adapter.info.isFallbackAdapter) throw new Error('No hardware WebGPU adapter is available');
    const requiredFeatures = ['shader-f16', 'subgroups', 'timestamp-query'].filter(feature => adapter.features.has(feature));
    if (options.matrixPrecision === 'mixed-f16' && !requiredFeatures.includes('shader-f16'))
        throw new Error('Mixed FP16 matrices require shader-f16; select the portable F32 profile');
    const device = await adapter.requestDevice({requiredFeatures, requiredLimits: {
        maxBufferSize: adapter.limits.maxBufferSize,
        maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
        maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize,
        maxStorageBuffersPerShaderStage: adapter.limits.maxStorageBuffersPerShaderStage,
    }});
    let runtime;
    try { runtime = new WebGpu(module, device, options); }
    catch (error) { device.destroy(); throw error; }
    if (options.profile && device.features.has('timestamp-query')) {
        runtime.reserveBufferBytes(65536);
        runtime.queries=device.createQuerySet({type:'timestamp',count:4096});
        runtime.queryBuffer=device.createBuffer({size:32768,usage:GPUBufferUsage.QUERY_RESOLVE|GPUBufferUsage.COPY_SRC});
        runtime.queryRead=device.createBuffer({size:32768,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
        runtime.stats.queryBytes = 65536;
        runtime.stats.gpuMilliseconds={};
    }
    runtime.adapter = {vendor: adapter.info.vendor, architecture: adapter.info.architecture};
    module.kidiGpu = runtime;
    return runtime;
}

class WebGpu {
    constructor(module, device, options = {}) {
        this.module = module;
        this.device = device;
        this.buffers = new Map();
        this.free = new Map();
        this.nextHandle = 1;
        this.epoch = 0;
        this.encoder = null;
        this.staging = [];
        this.readbacks = [];
        this.readPool = [];
        this.readBuffers = new Set();
        this.uploadBuffers = new Set();
        this.owner = null;
        this.owners = new Map();
        this.memoryBudgetBytes = options.memoryBudgetBytes ?? DEFAULT_GPU_BUDGET;
        this.disposed = false;
        this.failure = null;
        this.programs = new Map();
        this.regions = new Map();
        this.nextRegion = 1;
        this.inRegion = false;
        this.lowerGraphs = options.lowerGraphs !== false;
        this.onlineAttention = options.onlineAttention ?? 'auto';
        this.matrixPrecision = options.matrixPrecision ?? 'f32';
        this.w8a8Kernel = options.w8a8Kernel ?? 'float';
        this.preparedWeightBudget = options.preparedWeightBudget ?? MAX_PREPARED_WEIGHT_BYTES;
        this.preparedWeights = new Map();
        this.pipelines = new Map();
        this.pipelineChecks = [];
        this.nextProgram = 1;
        this.nextPipeline = 1;
        this.nextRecord = 1;
        this.currentPass = null;
        this.bindGroups = new Map();
        this.uniformStride = Math.max(256, device.limits.minUniformBufferOffsetAlignment);
        this.uniformWords = this.uniformStride / 4;
        this.uniformData = new Uint32Array(this.uniformWords * UNIFORM_SLOTS);
        this.uniformBuffer = device.createBuffer({size: this.uniformStride * UNIFORM_SLOTS,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST});
        this.uniformIndex = 0;
        this.queryLabels = [];
        this.validationBuffer = null;
        this.validationPending = false;
        this.measure = options.measure === true;
        this.stats = {
            allocatedBytes: 0, peakAllocatedBytes: 0, pooledBytes: 0, submissions: 0,
            readBytes: 0, uploadBytes: 0, copyBytes: 0, dispatches: 0, operatorCalls: 0,
            pipelineCreations: 0, regionCalls: 0, bridgeCalls: 0, encodeMs: 0, waitMs: 0,
            regionOriginalBytes: 0, regionPlannedBytes: 0,
            preparedWeightBytes: 0, preparedWeightHits: 0, preparedWeightMisses: 0, preparedWeightEvictions: 0,
            stagingBytes: 0, readbackBytes: 0, queryBytes: 0,
            memoryBudgetBytes: this.memoryBudgetBytes,
        };
        Object.defineProperty(this.stats, 'totalBufferBytes', {enumerable: true, get: () =>
            this.stats.allocatedBytes + this.stats.stagingBytes + this.stats.readbackBytes +
            this.stats.queryBytes + (this.disposed ? 0 : this.uniformBuffer.size)});
        if (this.stats.totalBufferBytes > this.memoryBudgetBytes)
            throw new Error('WebGPU memory budget is smaller than the uniform buffer');
        // Kernels test optional device features and WGSL language extensions through one set.
        this.packedDotAvailable = Boolean(navigator.gpu.wgslLanguageFeatures?.has('packed_4x8_integer_dot_product'));
        this.features = new Set([...device.features,
            ...(this.packedDotAvailable && options.packedDot !== false ? ['packed-dot'] : [])]);
        device.addEventListener('uncapturederror', event => { this.failure ??= event.error.message; });
        device.lost.then(info => { this.failure = `WebGPU device lost: ${info.message || info.reason}`; });
    }
    check() {
        if (this.failure || this.disposed) throw new Error(this.failure || 'WebGPU runtime is disposed');
    }
    withOwner(owner, action) {
        const previous = this.owner;
        this.owner = owner;
        try { return action(); }
        finally { this.owner = previous; }
    }
    ownedBytes(owner) { return this.owners.get(owner) || 0; }
    preparedBytes(owner) {
        return [...this.preparedWeights.values()].reduce((bytes, entry) =>
            bytes + (entry.owner === owner ? entry.bytes : 0), 0);
    }
    forgetOwner(owner) {
        if (this.ownedBytes(owner)) throw new Error('GPU model disposal retained owned buffers');
        this.owners.delete(owner);
    }
    reserveBufferBytes(bytes) {
        if (!Number.isSafeInteger(bytes) || bytes < 0) throw new Error('Invalid WebGPU buffer size');
        if (this.stats.totalBufferBytes + bytes > this.memoryBudgetBytes) {
            this.submitPending();
            this.trimBuffers(0);
        }
        if (this.stats.totalBufferBytes + bytes > this.memoryBudgetBytes)
            throw new Error(`WebGPU buffer budget exceeded (${this.stats.totalBufferBytes + bytes} > ${this.memoryBudgetBytes} bytes; ` +
                `requested ${bytes}, prepared ${this.stats.preparedWeightBytes}, owner ${this.owner || 'shared'})`);
    }
    commands() { this.check(); return this.encoder ??= this.device.createCommandEncoder(); }
    // Consecutive dispatches share one pass; WebGPU orders them and makes each write visible to the next.
    computePass() { return this.currentPass ??= this.commands().beginComputePass(); }
    endPass() { this.currentPass?.end(); this.currentPass = null; }
    transferCommands() { this.endPass(); return this.commands(); }
    // Wasm never waits on the GPU, so pipelines are created synchronously; compile errors surface on the next check.
    pipeline(code, inputs, validation = false) {
        if (!this.pipelines.has(code)) {
            this.device.pushErrorScope('validation');
            const shader = this.device.createShaderModule({code});
            const entries=Array.from({length:inputs+2},(_,binding)=>({binding,visibility:GPUShaderStage.COMPUTE,
                buffer:binding<inputs?{type:'read-only-storage'}:binding===inputs?{type:'storage'}
                    :{type:'uniform',hasDynamicOffset:true,minBindingSize:256}}));
                    if(validation)entries.push({binding:inputs+2,visibility:GPUShaderStage.COMPUTE,buffer:{type:'storage'}});
            const group=this.device.createBindGroupLayout({entries});
            const layout=this.device.createPipelineLayout({bindGroupLayouts:[group]});
            const id=this.nextPipeline++;
            const pipeline=this.device.createComputePipeline({layout, compute:{module:shader, entryPoint:'main'}});
            this.pipelineChecks.push(this.device.popErrorScope().then(async error => {
                if (!error) return;
                const messages=(await shader.getCompilationInfo()).messages.filter(message=>message.type==='error');
                this.failure = messages.map(message=>`${message.lineNum}:${message.linePos} ${message.message}`).join('\n') ||
                    this.failure || error.message;
            }));
            this.pipelines.set(code, {pipeline,group,id,validation});
            if (this.pipelines.size > 256) this.pipelines.delete(this.pipelines.keys().next().value);
            ++this.stats.pipelineCreations;
        }
        return this.pipelines.get(code);
    }
    prepare(spec) {
        this.check();
        const scoreBytes = spec.operation === 'attention'
            ? 8 * spec.inputs[0].shape[0] * spec.attributes[0] * spec.inputs[0].shape[1] *
                (spec.attributes[3] ?? spec.inputs[1].shape[1]) : 0;
        const onlineAttention = spec.onlineAttention ?? (this.onlineAttention === true ||
            (this.onlineAttention === 'auto' && scoreBytes > 32 * 1024 * 1024));
        const quantizedMatrix = spec.operation === 'quantized_linear' &&
            spec.attributes[0] !== 0 && spec.inputs[1].dtype === 2 && spec.inputs[0].shape.at(-1) % 4 === 0 &&
            spec.inputs[0].shape.slice(0, -1).reduce((count, size) => count * size, 1) >= 4;
        const mixedF16 = quantizedMatrix && this.matrixPrecision === 'mixed-f16' && this.features.has('shader-f16') &&
            spec.inputs[1].shape[0] * spec.inputs[1].shape[1] * 2 <=
                Math.min(this.preparedWeightBudget, this.device.limits.maxStorageBufferBindingSize);
        const w8a8 = quantizedMatrix && this.matrixPrecision === 'w8a8' && spec.inputs[0].shape.at(-1) <= 65536 &&
            spec.inputs[1].shape[0] * 4 <= this.preparedWeightBudget;
        const w8a8Kernel = this.w8a8Kernel;
        const plan = makeProgram({...spec, onlineAttention, mixedF16, w8a8, w8a8Kernel,
            inputs: mixedF16 ? spec.inputs.map((input, index) => index === 1 ? {...input, dtype: 9} : input) : spec.inputs},
        this.features);
        if (mixedF16 || w8a8) {
            const indices = w8a8 ? [1] : [1, 2];
            const preparation = makeProgram({operation: w8a8 ? 'w8a8_weight_sums' : 'expand_quantized_f16',
                dtype: w8a8 ? 6 : 9, attributes: [], inputs: indices.map(index => spec.inputs[index])}, this.features);
            preparation.pipeline = this.pipeline(preparation.code, indices.length, preparation.validatesIndices);
            preparation.bytes = spec.inputs[1].shape[0] * (w8a8 ? 4 : spec.inputs[1].shape[1] * 2);
            preparation.width = spec.inputs[1].shape[1];
            preparation.inputs = indices;
            preparation.kind = w8a8 ? 'w8a8' : 'mixed-f16';
            plan.weightPreparation = preparation;
            plan.w8a8 = w8a8;
        }
        plan.label=`${spec.operation}/${spec.attributes[0]??''}/${spec.inputs.map(input=>input.shape.join('x')).join(';')}`;
        plan.pipeline = this.pipeline(plan.code, plan.bindings ?? spec.inputs.length, plan.validatesIndices);
        if (plan.scratch) {
            plan.scratch.pipeline = this.pipeline(plan.scratch.code,1,plan.scratch.validatesIndices);
        }
        for (const stage of plan.stages ?? []) {
            stage.pipeline = this.pipeline(stage.code,stage.inputs.length);
        }
        const id = this.nextProgram++;
        this.programs.set(id, plan);
        return id;
    }
    releaseProgram(id) {
        this.programs.delete(id);
    }
    prepareRegion(description) {
        if (!this.lowerGraphs) return 0;
        for (const command of description.commands) {
            if (!this.programs.has(command.program)) throw new Error('Region references an unknown GPU program');
            command.values = new Uint32Array(command.bindings.length * 2);
        }
        const id = this.nextRegion++;
        this.regions.set(id, description);
        this.stats.regionOriginalBytes += description.originalBytes || 0;
        this.stats.regionPlannedBytes += description.plannedBytes || 0;
        return id;
    }
    releaseRegion(id) {
        const region = this.regions.get(id);
        if (region) {
            this.stats.regionOriginalBytes -= region.originalBytes || 0;
            this.stats.regionPlannedBytes -= region.plannedBytes || 0;
        }
        this.regions.delete(id);
    }
    runRegion(id, inputs) {
        const region = this.regions.get(id);
        if (!region) throw new Error('Unknown WebGPU execution region');
        ++this.stats.regionCalls;
        ++this.stats.bridgeCalls;
        const binding = source => source.input === undefined ? source :
            {handle: inputs[source.input * 2], offset: inputs[source.input * 2 + 1] + source.offset};
        this.inRegion = true;
        try {
            for (const command of region.commands) {
                command.bindings.forEach((source, index) => {
                    const value = binding(source);
                    command.values[index * 2] = value.handle;
                    command.values[index * 2 + 1] = value.offset;
                });
                this.run(command.program, command.values);
                const output = command.values.subarray(command.values.length - 2);
                if (command.copy) {
                    const target = binding(command.copy.destination);
                    this.copy(output[0], output[1], target.handle, target.offset, command.copy.bytes);
                }
                if (command.read)
                    this.readLater(output[0], output[1], command.read.destination, command.read.bytes);
            }
        } finally { this.inRegion = false; }
    }
    dispatch(program, inputs, output, uniforms, groups, label) {
        if (this.uniformIndex >= UNIFORM_SLOTS) this.submitPending();
        const slot = this.uniformIndex++;
        inputs.forEach((input, index) => { uniforms[index] = input.offset; });
        uniforms[8] = output.offset;
        const gridY=Math.ceil(groups/65535),gridX=Math.ceil(groups/gridY);
        uniforms[63]=gridX;
        this.uniformData.set(uniforms, slot * this.uniformWords);
        const records = [...inputs,output].map(binding=>this.buffer(binding.handle));
        if(program.validation){
            if (!this.validationBuffer) {
                this.validationBuffer = this.withOwner(null, () => this.allocate(4));
                this.clear(this.validationBuffer);
            }
            this.buffer(this.validationBuffer).epoch=this.epoch;
            this.validationPending=true;
        }
        let key = String(program.id);
        for (const record of records) key += ':' + record.id;
        let group = this.bindGroups.get(key);
        if (!group) {
            const entries = records.map((record,binding)=>({binding,resource:{buffer:record.buffer}}));
            entries.push({binding:records.length,resource:{buffer:this.uniformBuffer,offset:0,size:256}});
            if(program.validation)entries.push({binding:records.length+1,
                resource:{buffer:this.buffer(this.validationBuffer).buffer}});
            group = this.device.createBindGroup({layout:program.group,entries});
            if (this.bindGroups.size >= 4096) this.bindGroups.clear();
            this.bindGroups.set(key, group);
        }
        let pass;
        if (this.queries) {
            this.endPass();
            const query=this.queryLabels.length*2;
            const timestampWrites=query+1<4096?{querySet:this.queries,beginningOfPassWriteIndex:query,endOfPassWriteIndex:query+1}:undefined;
            if(timestampWrites)this.queryLabels.push(label);
            else this.stats.unprofiledDispatches=(this.stats.unprofiledDispatches||0)+1;
            pass=this.commands().beginComputePass(timestampWrites?{timestampWrites}:{});
        } else pass = this.computePass();
        pass.setPipeline(program.pipeline);
        pass.setBindGroup(0, group, [slot * this.uniformStride]);
        pass.dispatchWorkgroups(gridX,gridY);
        if (this.queries) pass.end();
        for (const record of records) record.epoch=this.epoch;
        ++records.at(-1).version;
        ++this.stats.dispatches;
    }
    dropPreparedWeight(key, evicted = false) {
        const entry = this.preparedWeights.get(key);
        if (!entry) return;
        this.preparedWeights.delete(key);
        this.stats.preparedWeightBytes -= entry.bytes;
        if (evicted) ++this.stats.preparedWeightEvictions;
        this.release(entry.handle);
    }
    trimPreparedWeights(bytes = 0, protectedOwners = new Set()) {
        if (!Number.isSafeInteger(bytes) || bytes < 0) throw new Error('Invalid prepared weight budget');
        for (const [key, entry] of this.preparedWeights) {
            if (this.stats.preparedWeightBytes <= bytes) break;
            if (!protectedOwners.has(entry.owner)) this.dropPreparedWeight(key, true);
        }
    }
    preparedWeight(plan, operands) {
        const preparation = plan.weightPreparation;
        const inputs = preparation.inputs.map(index => operands[index]);
        const records = inputs.map(input => this.buffer(input.handle));
        const key = `${preparation.kind}:${inputs.map(input => `${input.handle}:${input.offset}`).join(':')}:${preparation.width}:${preparation.bytes}`;
        const cached = this.preparedWeights.get(key);
        if (cached && records.every((record, index) => record.version === cached.versions[index])) {
            this.preparedWeights.delete(key);
            this.preparedWeights.set(key, cached);
            ++this.stats.preparedWeightHits;
            return {handle: cached.handle, offset: 0};
        }
        if (cached) this.dropPreparedWeight(key);
        while (this.preparedWeights.size >= 256 || this.stats.preparedWeightBytes + preparation.bytes > this.preparedWeightBudget)
            this.dropPreparedWeight(this.preparedWeights.keys().next().value, true);
        const handle = this.allocate(preparation.bytes);
        try {
            this.dispatch(preparation.pipeline, inputs, {handle, offset: 0},
                preparation.uniforms, preparation.groups, `prepare-${preparation.kind}/${plan.label}`);
        } catch (error) {
            this.release(handle);
            throw error;
        }
        this.preparedWeights.set(key, {handle, sources: inputs.map(input => input.handle),
            versions: records.map(record => record.version), bytes: preparation.bytes, owner: records[0].owner});
        this.stats.preparedWeightBytes += preparation.bytes;
        ++this.stats.preparedWeightMisses;
        return {handle, offset: 0};
    }
    run(id, bindings) {
        const started = this.measure ? performance.now() : 0;
        ++this.stats.operatorCalls;
        if (!this.inRegion) ++this.stats.bridgeCalls;
        const plan=this.programs.get(id), operands=[];
        for(let index=0;index<bindings.length;index+=2)operands.push({handle:bindings[index],offset:bindings[index+1]});
        const output=operands.pop();
        const temporaries=[];
        try {
            if (plan.weightPreparation) {
                const prepared = this.preparedWeight(plan, operands);
                if (plan.w8a8) operands.push(prepared);
                else operands[1] = prepared;
            }
            if(plan.scratch){
                const scratch={handle:this.allocate(plan.scratch.bytes),offset:0};
                temporaries.push(scratch.handle);
                this.dispatch(plan.scratch.pipeline,[operands[0]],scratch,plan.scratch.uniforms,plan.scratch.groups,`quantize/${plan.label}`);
                operands[0]=scratch;
            }
            let previous;
            for (const stage of plan.stages ?? []) {
                const result={handle:this.allocate(stage.bytes),offset:0};
                temporaries.push(result.handle);
                this.dispatch(stage.pipeline,stage.inputs.map(index=>index==='previous'?previous:operands[index]),result,stage.uniforms,stage.groups,`stage/${plan.label}`);
                previous=result;
            }
            if(plan.finalInputs)operands.splice(0,operands.length,...plan.finalInputs.map(index=>index==='previous'?previous:operands[index]));
            else if(plan.operation==='greedy_token')operands.splice(0,operands.length,previous);
            else if(plan.stages&&plan.operation==='attention')operands.splice(0,operands.length,previous,operands[2]);
            this.dispatch(plan.pipeline,operands,output,plan.uniforms,plan.groups,plan.label);
        } finally {
            for (const handle of temporaries) this.release(handle);
            if (this.measure) this.stats.encodeMs += performance.now() - started;
        }
    }
    allocate(bytes) {
        this.check();
        if (!Number.isSafeInteger(bytes) || bytes < 0) throw new Error('Invalid WebGPU buffer size');
        const size = Math.max(4, Math.ceil(bytes / 4) * 4);
        if (!Number.isSafeInteger(size) || size > this.device.limits.maxStorageBufferBindingSize)
            throw new Error(`WebGPU tensor exceeds buffer limits: ${size} bytes`);
        const available = this.free.get(size);
        let record = available?.pop();
        if (available?.length === 0) this.free.delete(size);
        if (record) this.stats.pooledBytes -= size;
        else {
            this.reserveBufferBytes(size);
            record = {buffer: this.device.createBuffer({size, usage: GPUBufferUsage.STORAGE |
                GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST}), size, epoch: -1, id: this.nextRecord++, version: 0};
            this.stats.allocatedBytes += size;
            this.stats.peakAllocatedBytes = Math.max(this.stats.peakAllocatedBytes, this.stats.allocatedBytes);
        }
        const handle = this.nextHandle++;
        record.owner = this.owner;
        if (record.owner !== null) this.owners.set(record.owner, this.ownedBytes(record.owner) + size);
        this.buffers.set(handle, record);
        return handle;
    }
    release(handle) {
        for (const [key, entry] of this.preparedWeights)
            if (entry.sources.includes(handle)) this.dropPreparedWeight(key);
        const record = this.buffers.get(handle);
        if (!record) return;
        this.buffers.delete(handle);
        if (record.owner !== null) this.owners.set(record.owner, this.ownedBytes(record.owner) - record.size);
        record.owner = null;
        if (!this.free.has(record.size)) this.free.set(record.size, []);
        this.free.get(record.size).push(record);
        this.stats.pooledBytes += record.size;
        this.trimBuffers();
    }
    trimBuffers(limit = MAX_POOLED_BYTES) {
        if (this.stats.pooledBytes <= limit) return;
        for (const [size, records] of this.free) {
            while (this.stats.pooledBytes > limit && records.length && records.at(-1).epoch !== this.epoch) {
                records.pop().buffer.destroy();
                this.stats.allocatedBytes -= size;
                this.stats.pooledBytes -= size;
                this.bindGroups.clear();
            }
            if (!records.length) this.free.delete(size);
            if (this.stats.pooledBytes <= limit) break;
        }
    }
    buffer(handle) {
        const record = this.buffers.get(handle);
        if (!record) throw new Error(`Invalid WebGPU buffer ${handle}`);
        return record;
    }
    clear(handle) {
        const record=this.buffer(handle);
        this.transferCommands().clearBuffer(record.buffer);
        record.epoch=this.epoch;
        ++record.version;
    }
    upload(handle, offset, bytes) {
        this.check();
        if (!bytes.byteLength) return;
        const record = this.buffer(handle);
        if (offset % 4 || offset + bytes.byteLength > record.size) throw new Error('Unaligned WebGPU upload');
        const padded = bytes.byteLength % 4 ? new Uint8Array(Math.ceil(bytes.byteLength / 4) * 4) : bytes;
        if (padded !== bytes) padded.set(bytes);
        if (record.epoch !== this.epoch) this.device.queue.writeBuffer(record.buffer, offset, padded);
        else {
            if (this.stats.stagingBytes + padded.byteLength > MAX_TRANSFER_BYTES)
                throw new Error('WebGPU upload staging exceeds 64 MiB');
            this.reserveBufferBytes(padded.byteLength);
            const staging = this.device.createBuffer({size: padded.byteLength, mappedAtCreation: true,
                usage: GPUBufferUsage.COPY_SRC});
            new Uint8Array(staging.getMappedRange()).set(padded);
            staging.unmap();
            this.transferCommands().copyBufferToBuffer(staging, 0, record.buffer, offset, padded.byteLength);
            this.staging.push(staging);
            this.uploadBuffers.add(staging);
            this.stats.stagingBytes += staging.size;
        }
        this.stats.uploadBytes += bytes.byteLength;
        ++record.version;
    }
    copy(source, sourceOffset, destination, destinationOffset, bytes) {
        if (!bytes) return;
        if ((sourceOffset | destinationOffset | bytes) % 4) throw new Error('Unaligned WebGPU copy');
        const input = this.buffer(source), output = this.buffer(destination);
        if (sourceOffset + bytes > input.size || destinationOffset + bytes > output.size) throw new Error('WebGPU copy outside buffer');
        if (input === output) {
            const temporary = this.allocate(bytes);
            this.copy(source, sourceOffset, temporary, 0, bytes);
            this.copy(temporary, 0, destination, destinationOffset, bytes);
            this.release(temporary);
            return;
        }
        this.transferCommands().copyBufferToBuffer(input.buffer, sourceOffset, output.buffer, destinationOffset, bytes);
        this.stats.copyBytes += bytes;
        input.epoch = output.epoch = this.epoch;
        ++output.version;
    }
    readLater(handle, offset, destination, bytes) {
        if (!bytes) return;
        if (destination !== null && !(destination instanceof Uint8Array) &&
            (!Number.isSafeInteger(destination) || destination < 0))
            throw new Error('Invalid WebGPU read destination');
        if (destination instanceof Uint8Array && destination.byteLength < bytes)
            throw new Error('WebGPU read destination is too small');
        const aligned = Math.floor(offset / 4) * 4, extra = offset - aligned;
        const size = Math.ceil((bytes + extra) / 4) * 4;
        let staging = this.readPool.pop();
        if (staging && staging.size < size) { this.destroyReadBuffer(staging); staging = null; }
        if (!staging) {
            if (this.stats.readbackBytes + size > MAX_TRANSFER_BYTES)
                for (const buffer of this.readPool.splice(0)) this.destroyReadBuffer(buffer);
            if (this.stats.readbackBytes + size > MAX_TRANSFER_BYTES)
                throw new Error('WebGPU pending readback exceeds 64 MiB');
            this.reserveBufferBytes(size);
            staging = this.device.createBuffer({size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST});
            this.readBuffers.add(staging);
            this.stats.readbackBytes += staging.size;
        }
        const input = this.buffer(handle);
        this.transferCommands().copyBufferToBuffer(input.buffer, aligned, staging, 0, size);
        input.epoch = this.epoch;
        this.readbacks.push({staging, destination, bytes, extra, size});
        this.stats.readBytes += bytes;
    }
    destroyReadBuffer(buffer) {
        if (!this.readBuffers.delete(buffer)) return;
        buffer.destroy();
        this.stats.readbackBytes -= buffer.size;
    }
    async read(handle, offset, destination, bytes) {
        this.readLater(handle, offset, destination, bytes);
        await this.synchronize();
    }
    /// Called from Wasm at model synchronization points: submits recorded work without waiting. Results queued with
    /// `readLater` reach the heap when JavaScript awaits `synchronize()` after the Wasm call returns.
    flush() {
        this.check();
        if (this.validationPending) {
            this.readLater(this.validationBuffer, 0, null, 4);
            this.validationPending = false;
        }
        this.submitPending();
    }
    // Timestamp labels accumulate across submissions and resolve once in synchronize().
    submitPending() {
        this.endPass();
        if (!this.encoder) { this.uniformIndex = 0; return; }
        if (this.uniformIndex)
            this.device.queue.writeBuffer(this.uniformBuffer, 0, this.uniformData, 0, this.uniformIndex * this.uniformWords);
        this.device.queue.submit([this.encoder.finish()]);
        this.encoder = null;
        this.uniformIndex = 0;
        ++this.epoch;
        ++this.stats.submissions;
    }
    async synchronize() {
        const started = this.measure ? performance.now() : 0;
        if (this.pipelineChecks.length) await Promise.all(this.pipelineChecks.splice(0));
        this.flush();
        const labels = this.queryLabels.splice(0);
        if (labels.length) {
            const encoder = this.device.createCommandEncoder();
            encoder.resolveQuerySet(this.queries, 0, labels.length * 2, this.queryBuffer, 0);
            encoder.copyBufferToBuffer(this.queryBuffer, 0, this.queryRead, 0, labels.length * 16);
            this.device.queue.submit([encoder.finish()]);
        }
        if (!this.readbacks.length && !this.staging.length && !labels.length) {
            await this.device.queue.onSubmittedWorkDone();
            this.trimBuffers();
            this.check();
            if (this.measure) this.stats.waitMs += performance.now() - started;
            return;
        }
        const pending = this.readbacks.splice(0), staging = this.staging.splice(0);
        const reads=pending.map(async item => {
            let mapped = false;
            try {
                await item.staging.mapAsync(GPUMapMode.READ, 0, item.size);
                mapped = true;
                if (this.disposed) throw new Error('WebGPU runtime was disposed during readback');
                const data=item.staging.getMappedRange(0,item.size);
                if(item.destination===null){
                    const flags = new Uint32Array(data)[0];
                    if (flags & 1) this.failure = 'WebGPU scatter index outside cache';
                    else if (flags & 2) this.failure = 'WebGPU gather index outside tensor';
                    else if (flags & 4) this.failure = 'WebGPU FP16 weight expansion produced non-finite values';
                    else if (flags & 8) this.failure = 'WebGPU W8A8 activation range is non-finite or underflows';
                }else if (item.destination instanceof Uint8Array)
                    item.destination.set(new Uint8Array(data,item.extra,item.bytes));
                else this.module.HEAPU8.set(new Uint8Array(data,item.extra,item.bytes),item.destination);
            } finally {
                if (mapped) item.staging.unmap();
                if (mapped && !this.disposed && !this.failure && this.readPool.length < 4 &&
                    this.readPool.reduce((bytes, buffer) => bytes + buffer.size, item.staging.size) <= MAX_RETAINED_READ_BYTES)
                    this.readPool.push(item.staging);
                else this.destroyReadBuffer(item.staging);
            }
        });
        if(labels.length)reads.push((async()=>{
            await this.queryRead.mapAsync(GPUMapMode.READ,0,labels.length*16);
            const timestamps=new BigUint64Array(this.queryRead.getMappedRange(0,labels.length*16));
            labels.forEach((label,index)=>{this.stats.gpuMilliseconds[label]=(this.stats.gpuMilliseconds[label]||0)+Number(timestamps[index*2+1]-timestamps[index*2])/1e6;});
            this.queryRead.unmap();
        })());
        if (!reads.length) reads.push(this.device.queue.onSubmittedWorkDone());
        const results = await Promise.allSettled(reads);
        for (const buffer of staging) {
            if (this.uploadBuffers.delete(buffer)) {
                buffer.destroy();
                this.stats.stagingBytes -= buffer.size;
            }
        }
        const rejected = results.find(result => result.status === 'rejected');
        if (rejected) this.failure ??= rejected.reason?.message || String(rejected.reason);
        this.trimBuffers();
        this.check();
        if (this.measure) this.stats.waitMs += performance.now() - started;
    }
    dispose() {
        if (this.disposed) return;
        this.disposed = true;
        this.device.destroy();
        for (const buffer of this.readBuffers) buffer.destroy();
        for (const buffer of this.uploadBuffers) buffer.destroy();
        this.uniformBuffer.destroy();
        this.queryBuffer?.destroy();
        this.queryRead?.destroy();
        this.queries?.destroy();
        this.buffers.clear();
        this.free.clear();
        this.bindGroups.clear();
        this.programs.clear();
        this.regions.clear();
        this.pipelines.clear();
        this.owners.clear();
        this.readBuffers.clear();
        this.uploadBuffers.clear();
        this.staging.length = this.readbacks.length = this.readPool.length = 0;
        this.preparedWeights.clear();
        this.stats.preparedWeightBytes = 0;
        this.stats.allocatedBytes = this.stats.pooledBytes = this.stats.stagingBytes =
            this.stats.readbackBytes = this.stats.queryBytes = 0;
    }
}