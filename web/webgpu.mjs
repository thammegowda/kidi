import {makeProgram} from './webgpu-kernels.mjs';

const UNIFORM_SLOTS = 8192;
const MAX_POOLED_BYTES = 128 * 1024 * 1024;

export async function createWebGpu(module, options = {}) {
    if (!navigator.gpu || !WebAssembly.Suspending || !WebAssembly.promising)
        throw new Error('WebGPU requires a browser with WebGPU and WebAssembly JSPI support');
    const adapter = await navigator.gpu.requestAdapter({powerPreference: 'high-performance'});
    if (!adapter || adapter.info.isFallbackAdapter) throw new Error('No hardware WebGPU adapter is available');
    const requiredFeatures = ['shader-f16', 'subgroups', 'timestamp-query'].filter(feature => adapter.features.has(feature));
    const device = await adapter.requestDevice({requiredFeatures, requiredLimits: {
        maxBufferSize: adapter.limits.maxBufferSize,
        maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
        maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize,
        maxStorageBuffersPerShaderStage: adapter.limits.maxStorageBuffersPerShaderStage,
    }});
    const runtime = new WebGpu(module, device);
    if (options.profile && device.features.has('timestamp-query')) {
        runtime.queries=device.createQuerySet({type:'timestamp',count:4096});
        runtime.queryBuffer=device.createBuffer({size:32768,usage:GPUBufferUsage.QUERY_RESOLVE|GPUBufferUsage.COPY_SRC});
        runtime.queryRead=device.createBuffer({size:32768,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
        runtime.stats.gpuMilliseconds={};
    }
    runtime.adapter = {vendor: adapter.info.vendor, architecture: adapter.info.architecture};
    module.kidiGpu = runtime;
    return runtime;
}

class WebGpu {
    constructor(module, device) {
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
        this.failure = null;
        this.programs = new Map();
        this.pipelines = new Map();
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
        this.stats = {allocatedBytes: 0, pooledBytes: 0, submissions: 0, readBytes: 0, uploadBytes: 0, dispatches: 0};
        device.addEventListener('uncapturederror', event => { this.failure = event.error.message; });
        device.lost.then(info => { this.failure = `WebGPU device lost: ${info.message || info.reason}`; });
    }
    check() { if (this.failure) throw new Error(this.failure); }
    commands() { this.check(); return this.encoder ??= this.device.createCommandEncoder(); }
    // Consecutive dispatches share one pass; WebGPU orders them and makes each write visible to the next.
    computePass() { return this.currentPass ??= this.commands().beginComputePass(); }
    endPass() { this.currentPass?.end(); this.currentPass = null; }
    transferCommands() { this.endPass(); return this.commands(); }
    async pipeline(code, inputs) {
        if (!this.pipelines.has(code)) {
            const shader = this.device.createShaderModule({code});
            const entries=Array.from({length:inputs+2},(_,binding)=>({binding,visibility:GPUShaderStage.COMPUTE,
                buffer:binding<inputs?{type:'read-only-storage'}:binding===inputs?{type:'storage'}
                    :{type:'uniform',hasDynamicOffset:true,minBindingSize:256}}));
            const group=this.device.createBindGroupLayout({entries});
            const layout=this.device.createPipelineLayout({bindGroupLayouts:[group]});
            const id=this.nextPipeline++;
            this.pipelines.set(code, this.device.createComputePipelineAsync({layout, compute:{module:shader, entryPoint:'main'}})
                .then(pipeline=>({pipeline,group,id})).catch(async error => {
                const messages=(await shader.getCompilationInfo()).messages.filter(message=>message.type==='error');
                throw new Error(messages.map(message=>`${message.lineNum}:${message.linePos} ${message.message}`).join('\n') || error.message);
            }));
        }
        return await this.pipelines.get(code);
    }
    async prepare(spec) {
        const plan = makeProgram(spec, this.device.features);
        plan.label=`${spec.operation}/${spec.attributes[0]??''}/${spec.inputs.map(input=>input.shape.join('x')).join(';')}`;
        plan.pipeline = await this.pipeline(plan.code, plan.bindings ?? spec.inputs.length);
        if (plan.scratch) {
            plan.scratch.pipeline = await this.pipeline(plan.scratch.code,1);
        }
        for (const stage of plan.stages ?? []) {
            stage.pipeline = await this.pipeline(stage.code,stage.inputs.length);
        }
        const id = this.nextProgram++;
        this.programs.set(id, plan);
        return id;
    }
    releaseProgram(id) {
        this.programs.delete(id);
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
        let key = String(program.id);
        for (const record of records) key += ':' + record.id;
        let group = this.bindGroups.get(key);
        if (!group) {
            const entries = records.map((record,binding)=>({binding,resource:{buffer:record.buffer}}));
            entries.push({binding:records.length,resource:{buffer:this.uniformBuffer,offset:0,size:256}});
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
        ++this.stats.dispatches;
    }
    run(id, bindings) {
        const plan=this.programs.get(id), operands=[];
        for(let index=0;index<bindings.length;index+=2)operands.push({handle:bindings[index],offset:bindings[index+1]});
        const output=operands.pop();
        const temporaries=[];
        try {
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
        }
    }
    allocate(bytes) {
        this.check();
        const size = Math.max(4, Math.ceil(bytes / 4) * 4);
        if (!Number.isSafeInteger(size) || size > this.device.limits.maxStorageBufferBindingSize)
            throw new Error(`WebGPU tensor exceeds buffer limits: ${size} bytes`);
        const available = this.free.get(size);
        let record = available?.pop();
        if (available?.length === 0) this.free.delete(size);
        if (record) this.stats.pooledBytes -= size;
        else {
            record = {buffer: this.device.createBuffer({size, usage: GPUBufferUsage.STORAGE |
                GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST}), size, epoch: -1, id: this.nextRecord++};
            this.stats.allocatedBytes += size;
        }
        const handle = this.nextHandle++;
        this.buffers.set(handle, record);
        return handle;
    }
    release(handle) {
        const record = this.buffers.get(handle);
        if (!record) return;
        this.buffers.delete(handle);
        if (!this.free.has(record.size)) this.free.set(record.size, []);
        this.free.get(record.size).push(record);
        this.stats.pooledBytes += record.size;
        this.trimBuffers();
    }
    trimBuffers() {
        if (this.stats.pooledBytes <= MAX_POOLED_BYTES) return;
        for (const [size, records] of this.free) {
            while (this.stats.pooledBytes > MAX_POOLED_BYTES && records.length && records.at(-1).epoch !== this.epoch) {
                records.pop().buffer.destroy();
                this.stats.allocatedBytes -= size;
                this.stats.pooledBytes -= size;
                this.bindGroups.clear();
            }
            if (!records.length) this.free.delete(size);
            if (this.stats.pooledBytes <= MAX_POOLED_BYTES) break;
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
            const staging = this.device.createBuffer({size: padded.byteLength, mappedAtCreation: true,
                usage: GPUBufferUsage.COPY_SRC});
            new Uint8Array(staging.getMappedRange()).set(padded);
            staging.unmap();
            this.transferCommands().copyBufferToBuffer(staging, 0, record.buffer, offset, padded.byteLength);
            this.staging.push(staging);
        }
        this.stats.uploadBytes += bytes.byteLength;
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
        input.epoch = output.epoch = this.epoch;
    }
    readLater(handle, offset, destination, bytes) {
        if (!bytes) return;
        const aligned = Math.floor(offset / 4) * 4, extra = offset - aligned;
        const size = Math.ceil((bytes + extra) / 4) * 4;
        let staging = this.readPool.pop();
        if (staging && staging.size < size) { staging.destroy(); staging = null; }
        staging ??= this.device.createBuffer({size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST});
        const input = this.buffer(handle);
        this.transferCommands().copyBufferToBuffer(input.buffer, aligned, staging, 0, size);
        input.epoch = this.epoch;
        this.readbacks.push({staging, destination, bytes, extra, size});
        this.stats.readBytes += bytes;
    }
    async read(handle, offset, destination, bytes) {
        this.readLater(handle, offset, destination, bytes);
        await this.synchronize();
    }
    submitPending() {
        this.endPass();
        if (!this.encoder) { this.uniformIndex = 0; return null; }
        const labels = this.queryLabels.splice(0);
        if (labels.length) {
            this.encoder.resolveQuerySet(this.queries,0,labels.length*2,this.queryBuffer,0);
            this.encoder.copyBufferToBuffer(this.queryBuffer,0,this.queryRead,0,labels.length*16);
        }
        if (this.uniformIndex)
            this.device.queue.writeBuffer(this.uniformBuffer, 0, this.uniformData, 0, this.uniformIndex * this.uniformWords);
        this.device.queue.submit([this.encoder.finish()]);
        this.encoder = null;
        this.uniformIndex = 0;
        ++this.epoch;
        ++this.stats.submissions;
        return labels;
    }
    async synchronize() {
        this.check();
        const labels = this.submitPending();
        if (!labels) { this.trimBuffers(); return; }
        const pending = this.readbacks.splice(0), staging = this.staging.splice(0);
        const reads=pending.map(async item => {
            await item.staging.mapAsync(GPUMapMode.READ, 0, item.size);
            this.module.HEAPU8.set(new Uint8Array(item.staging.getMappedRange(0, item.size), item.extra, item.bytes), item.destination);
            item.staging.unmap();
            if (this.readPool.length < 4) this.readPool.push(item.staging); else item.staging.destroy();
        });
        if(labels.length)reads.push((async()=>{
            await this.queryRead.mapAsync(GPUMapMode.READ,0,labels.length*16);
            const timestamps=new BigUint64Array(this.queryRead.getMappedRange(0,labels.length*16));
            labels.forEach((label,index)=>{this.stats.gpuMilliseconds[label]=(this.stats.gpuMilliseconds[label]||0)+Number(timestamps[index*2+1]-timestamps[index*2])/1e6;});
            this.queryRead.unmap();
        })());
        if(reads.length)await Promise.all(reads);else await this.device.queue.onSubmittedWorkDone();
        for (const buffer of staging) buffer.destroy();
        this.trimBuffers();
        this.check();
    }
    dispose() {
        this.device.destroy();
        this.buffers.clear();
        this.free.clear();
        this.bindGroups.clear();
    }
}