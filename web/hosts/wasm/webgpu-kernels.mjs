const product = shape => shape.reduce((size, extent) => size * extent, 1);
const floatBits = value => new Uint32Array(new Float32Array([value]).buffer)[0];
const strides = shape => shape.map((_, axis) => product(shape.slice(axis + 1)));
const broadcast = (left, right) => {
    const rank = Math.max(left.length, right.length);
    return Array.from({length: rank}, (_, axis) => {
        const first = left[axis - rank + left.length] ?? 1, second = right[axis - rank + right.length] ?? 1;
        if (first !== second && first !== 1 && second !== 1) throw new Error('Invalid WebGPU broadcast');
        return Math.max(first, second);
    });
};

// `dtypes` fixes operand types at compile time so loads carry no runtime type branches (otherwise read from uniforms).
function header(count, extra = '', dtypes) {
    return `${extra}
${Array.from({length: count}, (_, index) => `@group(0) @binding(${index}) var<storage,read> input${index}:array<u32>;`).join('\n')}
@group(0) @binding(${count}) var<storage,read_write> output:array<u32>;
struct Parameters { values:array<vec4<u32>,16> };
@group(0) @binding(${count + 1}) var<uniform> parameters:Parameters;
fn param(index:u32)->u32 {return parameters.values[index/4u][index%4u];}
fn scalar(index:u32)->f32 {return bitcast<f32>(param(index));}
fn round_even(value:f32)->f32 {let low=floor(value);let part=value-low;return low+select(0.0,1.0,part>0.5 || (part==0.5 && (i32(low)&1)!=0));}
fn calibrated(value:f32,scale:f32)->f32 {if(scale==0.0){return value;}return round_even(clamp(value/scale,-128.0,127.0))*scale;}
fn stable_tanh(value:f32)->f32 {if(value>10.0){return 1.0;}if(value< -10.0){return -1.0;}return tanh(value);}
fn bf16(value:f32)->u32 {let bits=bitcast<u32>(value);return (bits+32767u+((bits>>16u)&1u))>>16u;}
${Array.from({length: count}, (_, index) => `
fn load${index}(index:u32)->f32 {
    let dtype=${dtypes ? `${dtypes[index]}u` : `param(${9 + index}u)`};
    if(dtype==2u){let address=param(${index}u)+index;return f32(bitcast<i32>(input${index}[address/4u]<<((3u-address%4u)*8u))>>24u);}
    if(dtype==10u || dtype==9u){let address=param(${index}u)/2u+index;let half=(input${index}[address/2u]>>((address%2u)*16u))&65535u;
        if(dtype==10u){return bitcast<f32>(half<<16u);}return unpack2x16float(half).x;}
    let value=input${index}[param(${index}u)/4u+index];
    if(dtype==6u){return f32(bitcast<i32>(value));}return bitcast<f32>(value);
}`).join('\n')}
fn store(index:u32,value:f32){output[param(8u)/4u+index]=bitcast<u32>(value);}
fn group_index(group:vec3<u32>)->u32{return group.x+group.y*param(63u);}
fn invocation_index(id:vec3<u32>)->u32{return id.x+id.y*param(63u)*128u;}
`;
}
// Register-blocked products shared by multi-row projections and prefill attention. 256 threads form a 16x16 grid;
// each accumulates TILE.m x TILE.n outputs, so a workgroup covers 16*m rows by 16*n columns. Shared tiles hold
// `entries` four-channel entries per row plus one padding entry against bank conflicts, within 16 KiB.
// Larger blocks were no faster on Apple M5 (benchmarks/gemma4/JOURNAL.md, "Register Block Sweep").
const TILE = {m: 4, n: 4}, TILE_ROWS = 16 * TILE.m, TILE_COLUMNS = 16 * TILE.n;
// `prelude` defines a_entry(row,e), b_entry(column,e) (entry e covers depth 4e..4e+3) and put_out(row,column,value);
// `setup` defines row_start, column_start, rows, columns and depth_entries from the workgroup index `tile`.
function tiledProduct({prelude, setup, type, accumulator, zero, mac, entries}) {
    const {m, n} = TILE, bytes = type === 'u32' ? 4 : 16, stride = entries + 1;
    if (16 * (m + n) * stride * bytes > 16384) throw new Error('Tiled product exceeds 16 KiB of workgroup memory');
    const load = (tileName, count, start, limit, entry) => `
    for(var index=lane;index<${count}u;index+=256u){
        let r=index/${entries}u;let w=index%${entries}u;let e=base+w;var value=${zero};
        if(${start}+r<${limit}&&e<depth_entries){value=${entry}(${start}+r,e);}
        ${tileName}[r*${stride}u+w]=value;
    }`;
    // Fully unrolled with one named register per accumulator and operand: indexed private arrays may spill to memory.
    const each = (count, body) => Array.from({length: count}, (_, index) => body(index)).join('');
    const pairs = body => each(m, i => each(n, j => body(i, j)));
    return `${prelude}
var<workgroup> a_tile:array<${type},${16 * m * stride}>;var<workgroup> b_tile:array<${type},${16 * n * stride}>;
@compute @workgroup_size(256) fn main(@builtin(workgroup_id) group:vec3<u32>,@builtin(local_invocation_index) lane:u32){
let tile=group_index(group);
${setup}
let tx=lane%16u;let ty=lane/16u;
${pairs((i, j) => `var acc_${i}_${j}=${accumulator}();`)}
for(var base=0u;base<depth_entries;base+=${entries}u){${load('a_tile', 16 * m * entries, 'row_start', 'rows', 'a_entry')}${load('b_tile', 16 * n * entries, 'column_start', 'columns', 'b_entry')}
    workgroupBarrier();
    for(var inner=0u;inner<${entries}u;inner++){
        ${each(m, i => `let a_${i}=a_tile[(ty+${16 * i}u)*${stride}u+inner];`)}
        ${each(n, j => `let b_${j}=b_tile[(tx+${16 * j}u)*${stride}u+inner];`)}
        ${pairs((i, j) => `acc_${i}_${j}+=${mac(`a_${i}`, `b_${j}`)};`)}
    }
    workgroupBarrier();
}
${pairs((i, j) => `if(row_start+ty+${16 * i}u<rows&&column_start+tx+${16 * j}u<columns){put_out(row_start+ty+${16 * i}u,column_start+tx+${16 * j}u,acc_${i}_${j});}\n`)}}`;
}
// Calibrated low-bit projections of one to three rows (decode). GPUs without an int8 dot instruction (Apple) emulate
// dot4I8Packed, so weights are decoded into floats with one mask each: flipping every field's sign bit makes it the
// unsigned value w + 2^(bits-1), and masking a field in place yields that value times 2^shift. Activations are staged
// in shared memory pre-scaled by 2^-shift (quantized on the way unless already int8), and a per-word activation sum
// removes the 2^(bits-1) bias. Products and each lane's per-chunk sums are integers below 2^24, so converting chunk
// sums to i32 reproduces the integer dot product exactly. Eight lanes share a column; each loads up to four coalesced
// words per step so several loads are in flight. Returns null for widths of a partial step (8 lanes, one word each).
function decodeProduct({bits, width, columns, rows, ffnGate, prequantized, dtypes, uniforms, gatedByte}) {
    const perWord = 32 / bits, parts = 8 / bits, lanes = 8;
    const unroll = [4, 2, 1].find(count => width % (lanes * count * perWord) === 0);
    if (!unroll || (ffnGate && columns % 32)) return null;
    const span = lanes * unroll * perWord;
    let chunk = span;
    for (let size = span; size <= 2048; size += span) if (width % size === 0) chunk = size;
    const threads = !ffnGate && columns <= 512 ? 64 : 256, columnsPerGroup = threads / lanes;
    const words = chunk / perWord, steps = words / (lanes * unroll);
    uniforms[24] = Math.ceil(columns / columnsPerGroup);
    const each = (count, body) => Array.from({length: count}, (_, index) => body(index)).join('');
    const list = (count, item) => Array.from({length: count}, (_, index) => item(index)).join(',');
    // Channel 4p+k of a word sits at bit 8k+p*bits.
    const shift = (part, k) => 8 * k + part * bits;
    const signs = {2: '0xaaaaaaaau', 4: '0x88888888u', 8: '0x80808080u'}[bits];
    const weight = (word, part) => `vec4<f32>(vec4<u32>(${word})&vec4<u32>(${list(4, k => `${((1 << bits) - 1) * 2 ** shift(part, k)}u`)}))`;
    const scale = part => `bitcast<vec4<f32>>(vec4<u32>(${list(4, k => `${(127 - shift(part, k)) << 23}u`)}))`;
    const quad = index => prequantized
        ? `{let packed=input0[param(0u)/4u+(row*param(22u)+base)/4u+${index}];
value=vec4<f32>(vec4<i32>(bitcast<i32>(packed<<24u),bitcast<i32>(packed<<16u),bitcast<i32>(packed<<8u),bitcast<i32>(packed))>>vec4<u32>(24u));}`
        : `{let channel=row*width+base+(${index})*4u;
value=vec4<f32>(quantized_input(channel),quantized_input(channel+1u),quantized_input(channel+2u),quantized_input(channel+3u));}`;
    const mapping = ffnGate ? '(logical_column/32u)*16u+logical_column%16u+select(0u,columns/2u,(logical_column%32u)>=16u)' : 'logical_column';
    const code = header(3, '', dtypes) + `
var<workgroup> tile:array<vec4<f32>,${chunk / 4}>;var<workgroup> offsets:array<f32,${words}>;var<workgroup> partial:array<i32,${threads}>;
${gatedByte}
fn quantized_input(index:u32)->f32{return round_even(clamp(load0(index)/scalar(20u),-128.0,127.0));}
@compute @workgroup_size(${threads}) fn main(@builtin(workgroup_id) group:vec3<u32>,@builtin(local_invocation_index) thread:u32){
let width=${width}u;let columns=param(17u);let block=group_index(group);let per_row=param(24u);
let row=block/per_row;let logical_column=(block%per_row)*${columnsPerGroup}u+thread/${lanes}u;
let column=${mapping};let lane=thread%${lanes}u;
let weights=param(1u)/4u+column*${width / perWord}u+lane;
var total=0i;
for(var base=0u;base<width;base+=${chunk}u){
    for(var word=thread;word<${words}u;word+=${threads}u){
        var value:vec4<f32>;var activations=0.0;
        ${each(parts, part => `${quad(`word*${parts}u+${part}u`)}
        activations+=value.x+value.y+value.z+value.w;tile[${part * words}u+word]=value*${scale(part)};`)}
        offsets[word]=${-(2 ** (bits - 1))}.0*activations;
    }
    workgroupBarrier();
    if(column<columns){
        var sum=vec4<f32>();var offset=0.0;
        for(var step=0u;step<${steps}u;step++){
            let local=step*${lanes * unroll}u+lane;let first=weights+base/${perWord}u+step*${lanes * unroll}u;
            ${each(unroll, u => `let w${u}=input1[first+${u * lanes}u]^${signs};`)}
            offset+=${list(unroll, u => `offsets[local+${u * lanes}u]`).replaceAll(',', '+')};
            ${each(unroll, u => each(parts, p => `sum+=tile[local+${u * lanes + p * words}u]*${weight(`w${u}`, p)};`))}
        }
        total+=i32(sum.x+sum.y+sum.z+sum.w)+i32(offset);
    }
    workgroupBarrier();
}
partial[thread]=total;workgroupBarrier();
for(var step=${lanes / 2}u;step>0u;step/=2u){if(lane<step){partial[thread]+=partial[thread+step];}workgroupBarrier();}
${ffnGate ? `if(thread<4u){var packed=0u;let first=(block%per_row)*16u;
for(var component=0u;component<4u;component++){let channel=thread*4u+component;
let gate=calibrated(f32(partial[channel*8u])*scalar(20u)*load2(first+channel),scalar(21u));
let up=calibrated(f32(partial[(channel+16u)*8u])*scalar(20u)*load2(columns/2u+first+channel),scalar(21u));
packed|=gated_byte(gate,up)<<(component*8u);}
output[param(8u)/4u+row*(columns/8u)+first/4u+thread]=packed;}`
        : `if(lane==0u&&column<columns){store(row*columns+column,calibrated(f32(partial[thread])*scalar(20u)*load2(column),scalar(21u)));}`}
}`;
    return {code, groups: rows * uniforms[24]};
}
const matrixSetup = `let row_start=(tile/param(23u))*${TILE_ROWS}u;let column_start=(tile%param(23u))*${TILE_COLUMNS}u;
let rows=param(18u);let columns=param(17u);let depth_entries=(param(16u)+3u)/4u;`;
const vectorEntry = (name, bound, element) => `fn ${name}(index:u32,e:u32)->vec4<f32>{var v=vec4<f32>();
for(var k=0u;k<4u;k++){let d=e*4u+k;if(d<${bound}){v[k]=${element};}}return v;}`;
const reduction = `var<workgroup> partial:array<f32,128>;
fn reduce_sum(value:f32,lane:u32)->f32 {partial[lane]=value;workgroupBarrier();for(var step=64u;step>0u;step/=2u){if(lane<step){partial[lane]+=partial[lane+step];}workgroupBarrier();}return partial[0];}`;

// Signed 4x8-bit dot product for WGSL implementations without packed_4x8_integer_dot_product.
const DOT4_I8 = `
fn dot4_i8(a:u32,b:u32)->i32{
let x=vec4<i32>(bitcast<i32>(a<<24u)>>24u,bitcast<i32>(a<<16u)>>24u,bitcast<i32>(a<<8u)>>24u,bitcast<i32>(a)>>24u);
let y=vec4<i32>(bitcast<i32>(b<<24u)>>24u,bitcast<i32>(b<<16u)>>24u,bitcast<i32>(b<<8u)>>24u,bitcast<i32>(b)>>24u);
return dot(x,y);}`;
const portable = (code, features) => !code || features.has('packed-dot') || !code.includes('dot4I8Packed(') ? code
    : code.replaceAll('requires packed_4x8_integer_dot_product;', '').replaceAll('dot4I8Packed(', 'dot4_i8(') + DOT4_I8;

/// WGSL program for one operator. `features` holds device features and 'packed-dot' when the WGSL
/// packed_4x8_integer_dot_product extension is available.
export function makeProgram(spec, features) {
    const plan = buildProgram(spec, features);
    plan.code = portable(plan.code, features);
    if (plan.scratch) plan.scratch.code = portable(plan.scratch.code, features);
    for (const stage of plan.stages ?? []) stage.code = portable(stage.code, features);
    return plan;
}

function buildProgram(spec, features) {
    const operands = spec.inputs, inputShape = operands[0].shape, attributes = spec.attributes;
    const uniforms = new Uint32Array(64);
    operands.forEach((operand, index) => { uniforms[9 + index] = operand.dtype; });
    let shape = [...inputShape], dtype = spec.dtype, code, groups, scratch, stages, bindings, finalInputs, validatesIndices;
    const operation = spec.operation;
    const elementwise = ['add','multiply','gelu','tanh','static_round','gelu_multiply'];
    if (elementwise.includes(operation)) {
        if(dtype!==11)throw new Error('WebGPU arithmetic requires FP32 output');
        if (operation === 'add' || operation === 'multiply' || operation === 'gelu_multiply') shape = broadcast(inputShape, operands[1].shape);
        uniforms[16] = product(shape); uniforms[17] = shape.length; uniforms.set(shape,18);
        for (let operand = 0; operand < Math.min(2, operands.length); ++operand) {
            const own = operands[operand].shape, ownStrides = strides(own);
            uniforms.set(shape.map((_, axis) => { const index = axis - shape.length + own.length;
                return index < 0 || own[index] === 1 ? 0 : ownStrides[index]; }),26 + operand*8);
        }
        uniforms[42] = floatBits(spec.epsilon);
        const expression = operation === 'add' ? 'first+load1(right)' : operation === 'multiply' ? 'first*load1(right)' :
            operation === 'static_round' ? 'calibrated(first,scalar(42u))' : operation === 'tanh' ? 'stable_tanh(first)' :
            operation === 'gelu_multiply' ? 'gelu(first)*load1(right)' : 'gelu(first)';
        code = header(operands.length) + `fn gelu(value:f32)->f32{return 0.5*value*(1.0+stable_tanh(0.7978845608028654*(value+0.044715*value*value*value)));}
    @compute @workgroup_size(128) fn main(@builtin(global_invocation_id) id:vec3<u32>){let index=invocation_index(id);if(index>=param(16u)){return;}
    var remaining=index;var left=0u;var right=0u;for(var axis=i32(param(17u))-1;axis>=0;axis--){let coordinate=remaining%param(18u+u32(axis));remaining/=param(18u+u32(axis));left+=coordinate*param(26u+u32(axis));right+=coordinate*param(34u+u32(axis));}
    let first=load0(left);store(index,${expression});}`;
        groups = Math.ceil(uniforms[16]/128);
    } else if (['rms_norm','rms_norm_residual','rms_rotary','layer_norm'].includes(operation)) {
        const width=inputShape.at(-1), rows=product(inputShape)/width;
        uniforms[16]=width;uniforms[17]=rows;uniforms[18]=floatBits(spec.epsilon);
        const rotary=operation==='rms_rotary', residual=operation==='rms_norm_residual', layerNorm=operation==='layer_norm';
        if(rotary){uniforms[19]=inputShape.at(-2);uniforms[20]=inputShape.at(-3);}
        const value = residual ? `var value=normalized+load2(index);${operands.length===4?'value*=load3(0u);':''}` : 'let value=normalized;';
        // Each thread keeps its row values in registers: one read of the row, all loads issued together.
        const count=Math.ceil(width/128), inRow=width%128?'if(channel<width)':'';
        const each=body=>`for(var k=0u;k<${count}u;k++){let channel=lane+k*128u;${inRow}{${body}}}`;
        code=header(operands.length,'',operands.map(operand=>operand.dtype))+reduction+`
@compute @workgroup_size(128) fn main(@builtin(workgroup_id) group:vec3<u32>,@builtin(local_invocation_index) lane:u32){
let row=group_index(group);if(row>=param(17u)){return;}let width=${width}u;var sum=0.0;var mean=0.0;var x:array<f32,${count}>;
${each('x[k]=load0(row*width+channel);')}
${layerNorm?`${each('sum+=x[k];')}mean=reduce_sum(sum,lane)/f32(width);workgroupBarrier();sum=0.0;`:''}
${each('let value=x[k]-mean;sum+=value*value;')}
let inverse=inverseSqrt(reduce_sum(sum,lane)/f32(width)+scalar(18u));
${each(`let index=row*width+channel;
${rotary?`let half=width/2u;let partner=(channel+half)%width;let position=(row/param(19u))%param(20u);
let first=x[k]*inverse*load1(channel);let other=load0(row*width+partner)*inverse*load1(partner);
let normalized=first*load2(position*half+channel%half)+select(-other,other,channel>=half)*load3(position*half+channel%half);`:
`let normalized=(x[k]-mean)*inverse*load1(channel)${layerNorm?'+load2(channel)':''};`}
${value}store(index,value);`)}}`;
        groups=rows;
    } else if (operation==='cast') {
        const count=product(shape);uniforms[16]=count;
        const packed=dtype===10||dtype===9;
        const bytes=dtype===2;
        uniforms[20]=floatBits(spec.epsilon);
        if(!packed&&!bytes&&dtype!==11)throw new Error('Unsupported WebGPU cast');
        code=header(1)+`
    @compute @workgroup_size(128) fn main(@builtin(global_invocation_id) id:vec3<u32>){let index=invocation_index(id);if(index>=${bytes?'(param(16u)+3u)/4u':packed?'(param(16u)+1u)/2u':'param(16u)'}){return;}
    ${bytes?`var packed=0u;for(var component=0u;component<4u;component++){if(index*4u+component<param(16u)){let value=i32(${spec.epsilon>0?'round_even(clamp(load0(index*4u+component)/scalar(20u),-128.0,127.0))':'clamp(load0(index*4u+component),-128.0,127.0)'});packed|=(u32(value)&255u)<<(component*8u);}}output[param(8u)/4u+index]=packed;`
:packed?`let first=load0(index*2u);var second=0.0;if(index*2u+1u<param(16u)){second=load0(index*2u+1u);}output[index]=${dtype===10?'bf16(first)|(bf16(second)<<16u)':'pack2x16float(vec2<f32>(first,second))'};`:'store(index,load0(index));'}}`;
        groups=Math.ceil(count/(bytes?512:packed?256:128));
    } else if(operation==='gated_feed_forward') {
        const [bits,width,intermediate,gateOutput,downInput,downOutput]=attributes;
        const asFloat=value=>new Float32Array(new Uint32Array([value]).buffer)[0];
        const downScale=asFloat(downInput);
        if(![2,4,8].includes(bits)||width%128||intermediate%32||width<=0||intermediate<=0||
            ![spec.epsilon,asFloat(gateOutput),downScale,asFloat(downOutput)].every(value=>Number.isFinite(value)&&value>0))
            throw new Error('WebGPU fused FFN requires aligned calibrated projections');
        const gate=makeProgram({operation:'packed_linear',dtype:11,epsilon:spec.epsilon,
            attributes:[bits,width,gateOutput],inputs:operands.slice(0,3),ffnGate:downInput},features);
        const hiddenShape=[...inputShape];hiddenShape[hiddenShape.length-1]=intermediate;
        const down=makeProgram({operation:'packed_linear',dtype:11,epsilon:downScale,
            attributes:[bits,intermediate,downOutput],inputs:[{shape:hiddenShape,dtype:2},operands[3],operands[4]],
            prequantized:true},features);
        return {...down,operation,scratch:gate.scratch,bindings:3,
            stages:[{code:gate.code,groups:gate.groups,uniforms:gate.uniforms,bytes:product(hiddenShape),inputs:[0,1,2]}],
            finalInputs:['previous',3,4]};
    } else if (operation==='packed_linear') {
        const width=inputShape.at(-1), rows=product(inputShape)/width, columns=operands[1].shape[0];
        const [bits, group]=attributes;
        shape[shape.length-1]=columns;dtype=11;
        uniforms[16]=width;uniforms[17]=columns;uniforms[18]=rows;uniforms[19]=group;
        uniforms[20]=floatBits(spec.epsilon);uniforms[21]=attributes[2]>>>0;
        const quantized=spec.epsilon>0;
        const ffnGate=spec.ffnGate!==undefined;
        if(ffnGate)uniforms[25]=spec.ffnGate>>>0;
        if(quantized&&group!==width)throw new Error('WebGPU calibrated projections require one scale per output channel');
        const paddedWidth=Math.ceil(width/4)*4;uniforms[22]=paddedWidth;
        // Weights reach the GPU interleaved, so one shift and mask yields the four bytes of a dot product.
        const laneMask = bits===8 ? '0xffffffffu' : bits===4 ? '0x0f0f0f0fu' : '0x03030303u';
        const signExtension = bits===8 ? 'expanded' : bits===4 ? 'expanded|((expanded&0x08080808u)*30u)' : 'expanded|((expanded&0x02020202u)*126u)';
        const packFour = width%4 ?
            `var packed=0u;for(var component=0u;component<4u;component++){if(channel+component<width){packed|=(u32(weight(column*width+channel+component))&255u)<<(component*8u);}}` :
            `let offset=column*width+channel;let expanded=(input1[param(1u)/4u+offset/${32/bits}u]>>(${bits}u*((offset%${32/bits}u)/4u)))&${laneMask};let packed=${signExtension};`;
        const gatedByte=ffnGate?`fn gated_byte(gate:f32,up:f32)->u32{
    let value=0.5*gate*(1.0+stable_tanh(0.7978845608028654*(gate+0.044715*gate*gate*gate)))*up;
    return u32(i32(round_even(clamp(value/scalar(25u),-128.0,127.0))))&255u;}`:'';
        const decode=quantized&&rows<4?decodeProduct({bits,width,columns,rows,ffnGate,prequantized:spec.prequantized,
            dtypes:operands.map(operand=>operand.dtype),uniforms,gatedByte}):null;
        // The decode kernel quantizes its own input; the tiled kernels read int8 activations from a scratch pass.
        if(quantized&&!spec.prequantized&&!decode){
            const quantUniforms=new Uint32Array(uniforms);
            scratch={bytes:rows*paddedWidth,uniforms:quantUniforms,groups:Math.ceil(rows*paddedWidth/512),code:header(1)+`
@compute @workgroup_size(128) fn main(@builtin(global_invocation_id) id:vec3<u32>){let index=invocation_index(id);let row=index/(param(22u)/4u);let first=(index%(param(22u)/4u))*4u;if(row>=param(18u)){return;}var packed=0u;
for(var component=0u;component<4u;component++){var value=0i;if(first+component<param(16u)){value=i32(round_even(clamp(load0(row*param(16u)+first+component)/scalar(20u),-128.0,127.0)));}packed|=(u32(value)&255u)<<(component*8u);}output[index]=packed;}`};
        }
        if(decode){
            ({code,groups}=decode);
        } else if(quantized && (rows<32 || ffnGate)){
            // Few rows, and decode shapes the decode kernel does not cover: 8x32 output tiles of packed dots.
            uniforms[23]=Math.ceil(columns/32);
            code=header(3,'requires packed_4x8_integer_dot_product;')+gatedByte+`
var<workgroup> activation_tile:array<u32,256>;var<workgroup> weight_tile:array<u32,1024>;
${ffnGate?'var<workgroup> ffn_values:array<f32,256>;':''}
fn weight(index:u32)->i32 {let word=input1[param(1u)/4u+index/${32/bits}u];let slot=index%${32/bits}u;return i32(((word>>(8u*(slot%4u)+${bits}u*(slot/4u)))&${(1<<bits)-1}u)<<${32-bits}u)>>${32-bits}u;}
@compute @workgroup_size(256) fn main(@builtin(workgroup_id) group:vec3<u32>,@builtin(local_invocation_index) lane:u32){
let tile=group_index(group);let row_start=(tile/param(23u))*8u;let column_start=(tile%param(23u))*32u;
let local_row=lane/32u;let local_column=lane%32u;let width=param(16u);var sum=0i;
for(var base=0u;base<width;base+=128u){
    let input_row=row_start+local_row;let input_channel=base+local_column*4u;
    var input_word=0u;if(input_row<param(18u)&&input_channel<width){input_word=input0[param(0u)/4u+(input_row*param(22u)+input_channel)/4u];}activation_tile[lane]=input_word;
    for(var index=lane;index<1024u;index+=256u){let logical_column=column_start+index/32u;
        let column=${ffnGate?'(logical_column/32u)*16u+logical_column%16u+select(0u,param(17u)/2u,(logical_column%32u)>=16u)':'logical_column'};let channel=base+(index%32u)*4u;var word=0u;
        if(column<param(17u)&&channel<width){${packFour}word=packed;}weight_tile[index]=word;
    }
    workgroupBarrier();
    for(var inner=0u;inner<32u;inner++){sum+=dot4I8Packed(activation_tile[local_row*32u+inner],weight_tile[local_column*32u+inner]);}
    workgroupBarrier();
}
let row=row_start+local_row;let logical_column=column_start+local_column;
let column=${ffnGate?'(logical_column/32u)*16u+logical_column%16u+select(0u,param(17u)/2u,(logical_column%32u)>=16u)':'logical_column'};
${ffnGate?`ffn_values[lane]=calibrated(f32(sum)*scalar(20u)*load2(column),scalar(21u));workgroupBarrier();
if(row<param(18u)&&local_column<4u){var packed=0u;for(var component=0u;component<4u;component++){
let index=local_row*32u+local_column*4u+component;packed|=gated_byte(ffn_values[index],ffn_values[index+16u])<<(component*8u);}
output[param(8u)/4u+row*(param(17u)/8u)+column_start/8u+local_column]=packed;}`:
`if(row<param(18u)&&column<param(17u)){store(row*param(17u)+column,calibrated(f32(sum)*scalar(20u)*load2(column),scalar(21u)));}`} }`;
            groups=Math.ceil(rows/8)*Math.ceil(columns/32);
        } else if(quantized){
            // Prefill: register-blocked tiles so every shared word feeds several packed dots.
            uniforms[23]=Math.ceil(columns/TILE_COLUMNS);
            code=tiledProduct({type:'u32',accumulator:'i32',zero:'0u',entries:16,mac:(a,b)=>`dot4I8Packed(${a},${b})`,
                setup:matrixSetup,prelude:header(3,'requires packed_4x8_integer_dot_product;')+`
fn weight(index:u32)->i32 {let word=input1[param(1u)/4u+index/${32/bits}u];let slot=index%${32/bits}u;return i32(((word>>(8u*(slot%4u)+${bits}u*(slot/4u)))&${(1<<bits)-1}u)<<${32-bits}u)>>${32-bits}u;}
fn a_entry(row:u32,e:u32)->u32{return input0[param(0u)/4u+row*(param(22u)/4u)+e];}
fn b_entry(column:u32,e:u32)->u32{let width=param(16u);let channel=e*4u;${packFour}return packed;}
fn put_out(row:u32,column:u32,value:i32){store(row*param(17u)+column,calibrated(f32(value)*scalar(20u)*load2(column),scalar(21u)));}`});
            groups=Math.ceil(rows/TILE_ROWS)*Math.ceil(columns/TILE_COLUMNS);
        } else {
            // Uncalibrated projections: FP32 activations times dequantized weights with per-group scales.
            const subgroups=features.has('subgroups');
            const perLane=32/bits, parts=8/bits, lanes=8, groupThreads=256, columnsPerGroup=groupThreads/lanes;
            const chunk=Math.min(width,2048);
            const staged=width%chunk===0&&chunk%(lanes*perLane)===0&&group%perLane===0;
            const singleScale=group>=width;
            uniforms[24]=Math.ceil(columns/(staged?columnsPerGroup:4));
            // Columns in a workgroup share one activation row, so stage it once and read it as vectors.
            const stagedLoop=`
for(var base=0u;base<width;base+=${chunk}u){
for(var index=thread;index<${chunk/4}u;index+=${groupThreads}u){let source=param(0u)/4u+row*width+base+index*4u;tile[index]=bitcast<vec4<f32>>(vec4<u32>(input0[source],input0[source+1u],input0[source+2u],input0[source+3u]));}
workgroupBarrier();
if(column<columns){
for(var offset=lane*${perLane}u;offset<${chunk}u;offset+=${lanes*perLane}u){
let channel=base+offset;
let packed_word=input1[param(1u)/4u+(column*width+channel)/${32/bits}u];
var partial_sum=0.0;
${Array.from({length:parts},(_,part)=>`{
let raw=(vec4<u32>(packed_word)>>vec4<u32>(${part*bits}u,${8+part*bits}u,${16+part*bits}u,${24+part*bits}u))&vec4<u32>(${(1<<bits)-1}u);
partial_sum+=dot(tile[offset/4u+${part}u],vec4<f32>(vec4<i32>(raw^vec4<u32>(${1<<(bits-1)}u))-vec4<i32>(${1<<(bits-1)}i)));}`).join('\n')}
sum+=partial_sum${singleScale?'':'*load2(column*(width/param(19u))+channel/param(19u))'};
}
}
workgroupBarrier();
}`;
            const fallback=`if(row<param(18u)&&column<columns){for(var channel=lane;channel<width;channel+=32u){sum+=load0(row*width+channel)*f32(weight(column*width+channel))*load2(column*(width/param(19u))+channel/param(19u));}}`;
            const threads=staged?groupThreads:128, laneCount=staged?lanes:32;
            code=header(3,subgroups&&!staged?'enable subgroups;':'')+`var<workgroup> partial:array<f32,${threads}>;${staged?`var<workgroup> tile:array<vec4<f32>,${chunk/4}>;`:''}
fn weight(index:u32)->i32 {let word=input1[param(1u)/4u+index/${32/bits}u];let slot=index%${32/bits}u;return i32(((word>>(8u*(slot%4u)+${bits}u*(slot/4u)))&${(1<<bits)-1}u)<<${32-bits}u)>>${32-bits}u;}
@compute @workgroup_size(${threads}) fn main(@builtin(workgroup_id) group:vec3<u32>,@builtin(local_invocation_index) thread:u32${subgroups&&!staged?',@builtin(subgroup_size) subgroup_width:u32':''}){
let width=param(16u);let columns=param(17u);let block=group_index(group);let per_row=param(24u);
let row=block/per_row;let column=(block%per_row)*${staged?columnsPerGroup:4}u+thread/${laneCount}u;let lane=thread%${laneCount}u;
var sum=0.0;
${staged?stagedLoop:fallback}
${subgroups&&!staged?'if(subgroup_width==32u){sum=subgroupAdd(sum);}else{':''}
partial[thread]=sum;workgroupBarrier();for(var step=${laneCount/2}u;step>0u;step/=2u){if(lane<step){partial[thread]+=partial[thread+step];}workgroupBarrier();}sum=partial[thread];${subgroups&&!staged?'}':''}
if(lane==0u&&column<columns){store(row*columns+column,calibrated(${staged&&singleScale?'sum*load2(column)':'sum'},scalar(21u)));} }`;
            groups=rows*uniforms[24];
        }
    } else if (operation==='linear' && product(inputShape)/inputShape.at(-1)>=4) {
        // Multi-row FP32 projections use register-blocked tiles; four channels per shared entry.
        const width=inputShape.at(-1),rows=product(inputShape)/width,transpose=attributes[0]!==0;
        const columns=operands[1].shape[transpose?0:1];shape[shape.length-1]=columns;dtype=11;
        uniforms[16]=width;uniforms[17]=columns;uniforms[18]=rows;uniforms[23]=Math.ceil(columns/TILE_COLUMNS);
        code=tiledProduct({type:'vec4<f32>',accumulator:'f32',zero:'vec4<f32>()',entries:4,mac:(a,b)=>`dot(${a},${b})`,
            setup:matrixSetup,prelude:header(operands.length)+`
${vectorEntry('a_entry','param(16u)','load0(index*param(16u)+d)')}
${vectorEntry('b_entry','param(16u)',transpose?'load1(index*param(16u)+d)':'load1(d*param(17u)+index)')}
fn put_out(row:u32,column:u32,value:f32){store(row*param(17u)+column,value${operands.length===3?'+load2(column)':''});}`});
        groups=Math.ceil(rows/TILE_ROWS)*Math.ceil(columns/TILE_COLUMNS);
    } else if (operation==='linear') {
        const width=inputShape.at(-1),rows=product(inputShape)/width,transpose=attributes[0]!==0;
        const columns=operands[1].shape[transpose?0:1];shape[shape.length-1]=columns;dtype=11;
        uniforms[16]=width;uniforms[17]=columns;uniforms[18]=rows;
        code=header(operands.length)+reduction+`
@compute @workgroup_size(128) fn main(@builtin(workgroup_id) group:vec3<u32>,@builtin(local_invocation_index) lane:u32){let index=group_index(group);let columns=param(17u);let width=param(16u);let row=index/columns;let column=index%columns;if(row>=param(18u)){return;}var sum=0.0;
for(var channel=lane;channel<width;channel+=128u){sum+=load0(row*width+channel)*load1(${transpose?'column*width+channel':'channel*columns+column'});}let value=reduce_sum(sum,lane);if(lane==0u){store(index,value${operands.length===3?'+load2(column)':''});}}`;
        groups=rows*columns;
    } else if (operation==='slice' || operation==='transpose') {
        const rank=inputShape.length, inputStrides=strides(inputShape);
        if(rank>8)throw new Error('WebGPU views support at most eight dimensions');
        if(operation==='slice'){
            const axis=(attributes[0]+rank)%rank;
            shape[axis]=attributes[2];uniforms[43]=attributes[1]*inputStrides[axis];
            uniforms.set(inputStrides,26);
        } else {
            shape=attributes.map(axis=>inputShape[axis]);uniforms.set(attributes.map(axis=>inputStrides[axis]),26);
        }
        uniforms[16]=product(shape);uniforms[17]=rank;uniforms.set(shape,18);
        if(dtype!==11&&dtype!==6)throw new Error('WebGPU copied views currently require FP32 or I32');
        code=header(1)+`@compute @workgroup_size(128) fn main(@builtin(global_invocation_id) id:vec3<u32>){let index=invocation_index(id);if(index>=param(16u)){return;}var remaining=index;var source=param(43u);
for(var axis=i32(param(17u))-1;axis>=0;axis--){source+=(remaining%param(18u+u32(axis)))*param(26u+u32(axis));remaining/=param(18u+u32(axis));}output[param(8u)/4u+index]=input0[param(0u)/4u+source];}`;
        groups=Math.ceil(product(shape)/128);
    } else if (operation==='rotary') {
        const width=inputShape.at(-1),heads=inputShape.at(-2),length=inputShape.at(-3);
        uniforms[16]=product(shape);uniforms[17]=width;uniforms[18]=heads;uniforms[19]=length;
        code=header(3)+`@compute @workgroup_size(128) fn main(@builtin(global_invocation_id) id:vec3<u32>){let index=invocation_index(id);if(index>=param(16u)){return;}let width=param(17u);let half=width/2u;let channel=index%width;let position=(index/width/param(18u))%param(19u);
let other=load0(index-channel+(channel+half)%width);store(index,load0(index)*load1(position*half+channel%half)+select(-other,other,channel>=half)*load2(position*half+channel%half));}`;
        groups=Math.ceil(product(shape)/128);
    } else if(operation==='embedding') {
        const [width,bits,scaleGroups]=attributes;
        const rows=product(inputShape);shape=[1,rows,width];dtype=11;
        uniforms[16]=rows*width;uniforms[17]=width;uniforms[18]=scaleGroups;uniforms[19]=floatBits(spec.epsilon);
        code=header(operands.length)+`@compute @workgroup_size(128) fn main(@builtin(global_invocation_id) id:vec3<u32>){let index=invocation_index(id);if(index>=param(16u)){return;}let width=param(17u);let token=input0[param(0u)/4u+index/width];let channel=index%width;let offset=token*width+channel;
${bits?`let raw=(input1[param(1u)/4u+offset/${32/bits}u]>>((offset%${32/bits}u)*${bits}u))&${(1<<bits)-1}u;let value=f32(i32(raw<<${32-bits}u)>>${32-bits}u)*load2(token*param(18u)+channel/(width/param(18u)));`:'let value=load1(offset);'}store(index,value*scalar(19u));}`;
        groups=Math.ceil(rows*width/128);
    } else if(operation==='greedy_token') {
        const width=inputShape.at(-1),rows=product(inputShape)/width,blocks=Math.ceil(width/1024);
        shape=inputShape.slice(0,-1);dtype=6;
        const firstUniforms=new Uint32Array(uniforms);firstUniforms[16]=width;firstUniforms[17]=blocks;firstUniforms[18]=rows;
        const greedy = final => header(1)+`var<workgroup> maxima:array<f32,128>;var<workgroup> indices:array<i32,128>;var<workgroup> invalid:array<u32,128>;
@compute @workgroup_size(128) fn main(@builtin(workgroup_id) group:vec3<u32>,@builtin(local_invocation_index) lane:u32){let block=group_index(group);let width=param(16u);let count=param(17u);
var best=-3.402823466e38;var selected=2147483647i;var bad=0u;
${final?`for(var column=lane;column<count;column+=128u){let offset=(block*count+column)*3u;let value=bitcast<f32>(input0[offset]);let index=bitcast<i32>(input0[offset+1u]);bad|=input0[offset+2u];if(value>best||(value==best&&index<selected)){best=value;selected=index;}}`:
`let row=block/count;let start=(block%count)*1024u;for(var column=start+lane;column<min(start+1024u,width);column+=128u){let value=load0(row*width+column);let bits=bitcast<u32>(value);if((bits&0x7fffffffu)>0x7f800000u||bits==0x7f800000u){bad=1u;}if(value>best||(value==best&&i32(column)<selected)){best=value;selected=i32(column);}}`}
maxima[lane]=best;indices[lane]=selected;invalid[lane]=bad;workgroupBarrier();for(var step=64u;step>0u;step/=2u){if(lane<step){let other=maxima[lane+step];let index=indices[lane+step];if(other>maxima[lane]||(other==maxima[lane]&&index<indices[lane])){maxima[lane]=other;indices[lane]=index;}invalid[lane]|=invalid[lane+step];}workgroupBarrier();}
if(lane==0u){${final?'output[param(8u)/4u+block]=bitcast<u32>(select(indices[0],-1i,invalid[0]!=0u||indices[0]==2147483647i));':'output[block*3u]=bitcast<u32>(maxima[0]);output[block*3u+1u]=bitcast<u32>(indices[0]);output[block*3u+2u]=invalid[0];'}}}`;
        stages=[{code:greedy(false),uniforms:firstUniforms,groups:rows*blocks,inputs:[0],bytes:rows*blocks*12}];
        uniforms[16]=width;uniforms[17]=blocks;uniforms[18]=rows;
        bindings=1;
        code=greedy(true);groups=rows;
    } else if(operation==='softmax'||operation==='log_softmax') {
        const width=inputShape.at(-1),rows=product(inputShape)/width;
        uniforms[16]=width;uniforms[17]=rows;
        code=softmax(operands.length,operation==='log_softmax');groups=rows;
    } else if(operation==='attention') {
        const heads=attributes[0],keyHeads=attributes[1]??heads,width=inputShape[2]/heads;
        const queries=inputShape[1],batch=inputShape[0],keyLength=attributes[3]??operands[1].shape[1];
        const keyStart=attributes[2]??0,cacheLength=operands[1].shape[1],rows=batch*heads*queries;
        uniforms.set([width,heads,keyHeads,queries,keyLength,cacheLength,keyStart,batch],16);
        uniforms[24]=floatBits(spec.epsilon||1/Math.sqrt(width));
        uniforms[25]=operands[1].shape[0];
        const maskShape=operands[3]?.shape??[];
        uniforms[26]=maskShape.at(-2)??1;uniforms[27]=maskShape.at(-1)??1;
        uniforms[28]=maskShape.length>=3?maskShape.at(-3):1;uniforms[29]=maskShape.length>=4?maskShape.at(-4):1;
        if(width%4)throw new Error('WebGPU attention head width must divide four');
        // Four cached channels share one word, so read it once rather than through four dtype-dispatching loads.
        const byteKeys=operands[1].dtype===2;
        const quantizationFunction=(name,metadata) => {
            if(!metadata?.scales?.length)return `fn ${name}_scale(channel:u32)->f32{return 1.0;}fn ${name}_zero(channel:u32)->f32{return 0.0;}`;
            if(metadata.block_size<=0||metadata.scales.length!==metadata.zero_points.length)
                throw new Error(`Invalid ${name} blockwise quantization`);
            const scaleCases=metadata.scales.map((value,index)=>`case ${index}u:{return ${Number(value).toPrecision(9)};}`).join('');
            const zeroCases=metadata.zero_points.map((value,index)=>`case ${index}u:{return ${Number(value).toFixed(1)};}`).join('');
            return `fn ${name}_scale(channel:u32)->f32{switch(channel/${metadata.block_size}u){${scaleCases}default:{return ${Number(metadata.scales.at(-1)).toPrecision(9)};}}}
fn ${name}_zero(channel:u32)->f32{switch(channel/${metadata.block_size}u){${zeroCases}default:{return ${Number(metadata.zero_points.at(-1)).toFixed(1)};}}}`;
        };
        const quantizationCode=quantizationFunction('key',spec.quantization?.[0])+quantizationFunction('value',spec.quantization?.[1]);
        const keyBase='(((batch%param(25u))*param(21u)+KEY+param(22u))*param(18u)+keyHead)*width';
        if(queries===1){
            // Decode keeps scores in workgroup memory and rescales an online softmax, so one dispatch covers a head.
            const mask=operands.length===4?'+load3((((batch%param(29u))*param(28u)+head%param(28u))*param(26u))*param(27u)+key%param(27u))':'';
            // One workgroup per head leaves most of the device idle, so long key ranges are split and recombined.
            const splits=Math.max(1,Math.min(64,Math.ceil(keyLength/64)));
            const slot=width+2;
            // Blocks of 64 keys: four lanes score each key from a query pre-multiplied by the key scales (the zero points
            // become one per-lane constant), then each thread accumulates one four-channel quad of values over a share
            // of the block's keys; the value scales and zero points are applied once at the end.
            const quads=width/4, valueGroups=Math.floor(256/quads), byteValues=operands[2].dtype===2;
            if(quads>256)throw new Error('WebGPU decode attention supports head widths up to 1024');
            const unpack=word=>`vec4<f32>(vec4<i32>(bitcast<i32>(${word}<<24u),bitcast<i32>(${word}<<16u),bitcast<i32>(${word}<<8u),bitcast<i32>(${word}))>>vec4<u32>(24u))`;
            // Each lane takes every fourth quad; a fixed trip count compiles to faster code when four divides the quads.
            const laneQuads=quads%4?`for(var quad=lane;quad<${quads}u;quad+=4u){`:`for(var i=0u;i<${quads/4}u;i++){let quad=lane+i*4u;`;
            const quadOf=(binding,name)=>`vec4<f32>(${name}(c),${name}(c+1u),${name}(c+2u),${name}(c+3u))`;
            const scan=epilogue=>`
var<workgroup> qtile:array<vec4<f32>,${quads}>;var<workgroup> zero_terms:array<f32,${quads}>;
var<workgroup> partials:array<f32,256>;var<workgroup> scores:array<f32,64>;var<workgroup> weights:array<f32,64>;
var<workgroup> sums:array<vec4<f32>,256>;
@compute @workgroup_size(256) fn main(@builtin(workgroup_id) group:vec3<u32>,@builtin(local_invocation_index) thread:u32){
let block=group_index(group);let heads=param(17u);let width=${width}u;let keys=param(20u);
let owner=block/${splits}u;let piece=block%${splits}u;
let head=owner%heads;let batch=owner/heads;let keyHead=head/(heads/param(18u));
let queryBase=(batch*heads+head)*width;
let stride=(keys+${splits}u-1u)/${splits}u;
let rangeStart=min(keys,piece*stride);let rangeEnd=min(keys,rangeStart+stride);
for(var quad=thread;quad<${quads}u;quad+=256u){let c=quad*4u;
let query=vec4<f32>(load0(queryBase+c),load0(queryBase+c+1u),load0(queryBase+c+2u),load0(queryBase+c+3u));
${byteKeys?`let scaled=query*${quadOf(1,'key_scale')};qtile[quad]=scaled;zero_terms[quad]=dot(scaled,${quadOf(1,'key_zero')});`:'qtile[quad]=query;'}}
workgroupBarrier();
let lane=thread%4u;let slot_key=thread/4u;let value_quad=thread%${quads}u;let value_group=thread/${quads}u;
var lane_offset=0.0;
${byteKeys?`${laneQuads}lane_offset+=zero_terms[quad];}`:''}
var maximum=-3.402823466e38;var total=0.0;var acc=vec4<f32>();
for(var first=rangeStart;first<rangeEnd;first+=64u){
let span=min(64u,rangeEnd-first);
var products=0.0;
if(slot_key<span){let source=${keyBase.replace('KEY','first+slot_key')};
${laneQuads}
${byteKeys?`let word=input1[(param(1u)+source)/4u+quad];products+=dot(qtile[quad],${unpack('word')});`
:`let c=source+quad*4u;products+=dot(qtile[quad],vec4<f32>(load1(c),load1(c+1u),load1(c+2u),load1(c+3u)));`}}
products-=lane_offset;}
partials[thread]=products;workgroupBarrier();
if(thread<64u){var score=-3.402823466e38;
if(thread<span){let key=first+thread;score=(partials[thread*4u]+partials[thread*4u+1u]+partials[thread*4u+2u]+partials[thread*4u+3u])*scalar(24u)${mask};}
scores[thread]=score;}
workgroupBarrier();
var highest=-3.402823466e38;for(var j=0u;j<64u;j++){highest=max(highest,scores[j]);}
let updated=max(maximum,highest);let rescale=exp(maximum-updated);
if(thread<64u){weights[thread]=select(0.0,exp(scores[thread]-updated),thread<span);}
workgroupBarrier();
var block_total=0.0;for(var j=0u;j<64u;j++){block_total+=weights[j];}
total=total*rescale+block_total;maximum=updated;acc*=rescale;
for(var j=value_group;j<span;j+=${valueGroups}u){let source=${keyBase.replace('KEY','first+j')};
${byteValues?`let word=input2[(param(2u)+source)/4u+value_quad];acc+=weights[j]*${unpack('word')};`
:`let c=source+value_quad*4u;acc+=weights[j]*vec4<f32>(load2(c),load2(c+1u),load2(c+2u),load2(c+3u));`}}
workgroupBarrier();}
sums[thread]=acc;workgroupBarrier();
if(thread<${quads}u){var sum=sums[thread];for(var g=1u;g<${valueGroups}u;g++){sum+=sums[thread+g*${quads}u];}
let c=thread*4u;
let result=${byteValues?`${quadOf(2,'value_scale')}*(sum-${quadOf(2,'value_zero')}*total)`:'sum'};
${epilogue}}}`;
            if(splits===1){
                bindings=operands.length;
                code=header(operands.length)+quantizationCode+scan('for(var k=0u;k<4u;k++){store(queryBase+c+k,result[k]/total);}');
                groups=batch*heads;
            } else {
                // Each piece stores its unnormalised sum with the running maximum and total it was scaled by.
                stages=[{code:header(operands.length)+quantizationCode+scan(`let store_base=block*${slot}u;for(var k=0u;k<4u;k++){store(store_base+c+k,result[k]);}
if(thread==0u){store(store_base+width,maximum);store(store_base+width+1u,total);}`),
                    uniforms:new Uint32Array(uniforms),groups:batch*heads*splits,
                    inputs:operands.map((_,index)=>index),bytes:batch*heads*splits*slot*4}];
                bindings=1;
                finalInputs=['previous'];
                code=header(1)+`
@compute @workgroup_size(256) fn main(@builtin(workgroup_id) group:vec3<u32>,@builtin(local_invocation_index) thread:u32){
let block=group_index(group);let width=param(16u);
var maximum=-3.402823466e38;
for(var piece=0u;piece<${splits}u;piece++){
let head_slot=(block*${splits}u+piece)*${slot}u;
if(load0(head_slot+width+1u)>0.0){maximum=max(maximum,load0(head_slot+width));}}
var total=0.0;
for(var piece=0u;piece<${splits}u;piece++){
let head_slot=(block*${splits}u+piece)*${slot}u;
let weight=load0(head_slot+width+1u);
if(weight>0.0){total+=weight*exp(load0(head_slot+width)-maximum);}}
for(var channel=thread;channel<width;channel+=256u){
var sum=0.0;
for(var piece=0u;piece<${splits}u;piece++){
let head_slot=(block*${splits}u+piece)*${slot}u;
if(load0(head_slot+width+1u)>0.0){sum+=load0(head_slot+channel)*exp(load0(head_slot+width)-maximum);}}
store(block*width+channel,sum/total);}}`;
                groups=batch*heads;
            }
        } else {
            // Prefill attention as two register-blocked products per (batch, head): scores = Q K^T and output = P V,
            // with the row softmax between them.
            const elementAt=(binding,address,name)=>`(f32(bitcast<i32>(input${binding}[(${address})/4u]<<((3u-(${address})%4u)*8u))>>24u)-${name}_zero(c))*${name}_scale(c)`;
            const keyAt=byteKeys?elementAt(1,'param(1u)+source+c','key'):'load1(source+c)';
            const valueAt=operands[2].dtype===2?elementAt(1,'param(1u)+source+c','value'):'load1(source+c)';
            const maskPrelude=operands.length===4?'let batch=bh/param(17u);let head=bh%param(17u);':'';
            const maskTerm=operands.length===4?'+load3((((batch%param(29u))*param(28u)+head%param(28u))*param(26u)+query%param(26u))*param(27u)+key%param(27u))':'';
            // Element offset of a (batch, key head) slice of the cache, before the key index.
            const cacheBase=`fn b_base(bh:u32)->u32{let batch=bh/param(17u);let keyHead=(bh%param(17u))/(param(17u)/param(18u));
return (((batch%param(25u))*param(21u)+param(22u))*param(18u)+keyHead)*param(16u);}`;
            const offsets=`fn a_base(bh:u32)->u32{let heads=param(17u);return (bh/heads*param(19u)*heads+bh%heads)*param(16u);}
${cacheBase}`;
            const tiled=(rows,columns,depth)=>{
                const values=new Uint32Array(uniforms);
                const columnTiles=Math.ceil(columns/TILE_COLUMNS);
                values.set([rows,columns,depth,Math.ceil(rows/TILE_ROWS)*columnTiles,columnTiles],44);
                return {uniforms:values,groups:batch*heads*values[47]};
            };
            // Each workgroup serves one (batch, head) entry; its offsets are computed once.
            const batched=(inputs,prelude)=>tiledProduct({type:'vec4<f32>',accumulator:'f32',zero:'vec4<f32>()',entries:4,
                mac:(a,b)=>`dot(${a},${b})`,prelude:header(inputs)+quantizationCode+prelude+`
var<private> a_offset:u32;var<private> b_offset:u32;var<private> o_offset:u32;var<private> entry_index:u32;
${vectorEntry('a_entry','param(46u)','a_at(a_offset,index,d)')}
${vectorEntry('b_entry','param(46u)','b_at(b_offset,index,d)')}
fn put_out(row:u32,column:u32,value:f32){put(o_offset,entry_index,row,column,value);}`,
                setup:`let bh=tile/param(47u);let local_tile=tile%param(47u);
let row_start=(local_tile/param(48u))*${TILE_ROWS}u;let column_start=(local_tile%param(48u))*${TILE_COLUMNS}u;
let rows=param(44u);let columns=param(45u);let depth_entries=(param(46u)+3u)/4u;
a_offset=a_base(bh);b_offset=b_base(bh);o_offset=o_base(bh);entry_index=bh;`});
            const scores=tiled(queries,keyLength,width);
            stages=[{code:batched(operands.length,`
${offsets}
fn a_at(a:u32,query:u32,c:u32)->f32{return load0(a+query*param(17u)*param(16u)+c);}
fn b_at(b:u32,key:u32,c:u32)->f32{let source=b+key*param(18u)*param(16u);return ${keyAt};}
fn o_base(bh:u32)->u32{return bh*param(19u)*param(20u);}
fn put(o:u32,bh:u32,query:u32,key:u32,value:f32){${maskPrelude}store(o+query*param(20u)+key,value*scalar(24u)${maskTerm});}`),
                ...scores,inputs:operands.map((_,index)=>index),bytes:rows*keyLength*4}];
            const softUniforms=new Uint32Array(64);softUniforms[9]=11;softUniforms[16]=keyLength;softUniforms[17]=rows;
            stages.push({code:softmax(1,false),uniforms:softUniforms,groups:rows,inputs:['previous'],bytes:rows*keyLength*4});
            const values=tiled(queries,width,keyLength);
            uniforms.set(values.uniforms);uniforms[9]=11;uniforms[10]=operands[2].dtype;
            bindings=2;
            code=batched(2,`
fn a_base(bh:u32)->u32{return bh*param(19u)*param(20u);}
fn a_at(a:u32,query:u32,key:u32)->f32{return load0(a+query*param(20u)+key);}
${cacheBase}
fn b_at(b:u32,c:u32,key:u32)->f32{let source=b+key*param(18u)*param(16u);return ${valueAt};}
fn o_base(bh:u32)->u32{let heads=param(17u);return (bh/heads*param(19u)*heads+bh%heads)*param(16u);}
fn put(o:u32,bh:u32,query:u32,c:u32,value:f32){store(o+query*param(17u)*param(16u)+c,value);}`);
            groups=values.groups;
        }
    } else if(operation==='scatter') {
        const bytes=dtype===2||dtype===1?1:dtype===9||dtype===10?2:dtype===11||dtype===6?4:0;
        if(!bytes)throw new Error('Unsupported WebGPU scatter dtype');
        const [batches,capacity,width]=shape, updates=operands[1].shape[1];
        uniforms[16]=capacity;uniforms[17]=width;uniforms[18]=updates;
        validatesIndices=true;
        const validation=count=>`@group(0) @binding(${count+2}) var<storage,read_write> failure:atomic<u32>;
    fn valid_indices()->bool{for(var row=0u;row<param(18u);row++){if(input${count-1}[param(${count-1}u)/4u+row]>=param(16u)){atomicStore(&failure,1u);return false;}}return true;}`;
        if(spec.inplace&&width*bytes%4===0){
            bindings=2;finalInputs=[1,2];
            uniforms[19]=width*bytes/4;uniforms[20]=batches*updates*uniforms[19];
            code=header(2)+validation(2)+`@compute @workgroup_size(128) fn main(@builtin(global_invocation_id) id:vec3<u32>){
let index=invocation_index(id);if(index>=param(20u)||!valid_indices()){return;}let word=index%param(19u);let row=(index/param(19u))%param(18u);let batch=index/param(19u)/param(18u);
let position=input1[param(1u)/4u+row];if(position>=param(16u)){return;}
for(var later=row+1u;later<param(18u);later++){if(input1[param(1u)/4u+later]==position){return;}}
output[param(8u)/4u+(batch*param(16u)+position)*param(19u)+word]=input0[param(0u)/4u+index];}`;
            groups=Math.ceil(uniforms[20]/128);
        }else{
            if(spec.inplace){bindings=2;finalInputs=[1,2];}
            const count=spec.inplace?2:3, update=spec.inplace?0:1, indices=count-1;
            uniforms[19]=product(shape);uniforms[20]=Math.ceil(product(shape)*bytes/4);
            code=header(count)+validation(count)+`@compute @workgroup_size(128) fn main(@builtin(global_invocation_id) id:vec3<u32>){
let word=invocation_index(id);if(word>=param(20u)||!valid_indices()){return;}var packed=${spec.inplace?'output[param(8u)/4u+word]':'0u'};
for(var component=0u;component<${4/bytes}u;component++){let index=word*${4/bytes}u+component;if(index>=param(19u)){break;}
let channel=index%param(17u);let position=(index/param(17u))%param(16u);let batch=index/param(17u)/param(16u);
let original=param(${spec.inplace?8:0}u)+index*${bytes}u;var value=${spec.inplace?'output':'input0'}[original/4u]>>((original%4u)*8u);
for(var row=0u;row<param(18u);row++){if(input${indices}[param(${indices}u)/4u+row]==position){let source=param(${update}u)+((batch*param(18u)+row)*param(17u)+channel)*${bytes}u;value=input${update}[source/4u]>>((source%4u)*8u);}}
let mask=${bytes===4?'0xffffffffu':bytes===2?'65535u':'255u'};let shift=component*${bytes*8}u;
packed=(packed&~(mask<<shift))|((value&mask)<<shift);}
output[param(8u)/4u+word]=packed;}`;
            groups=Math.ceil(uniforms[20]/128);
        }
    } else if(operation==='concat') {
        if(dtype!==11)throw new Error('WebGPU concatenation requires FP32 output');
        const axis=(attributes[0]+shape.length)%shape.length;
        shape[axis]=operands.reduce((sum,operand)=>sum+operand.shape[axis],0);
        uniforms[16]=product(shape);uniforms[17]=product(shape.slice(axis+1));uniforms[18]=shape[axis];
        operands.forEach((operand,index)=>{uniforms[20+index]=operand.shape[axis];});
        code=header(operands.length)+`@compute @workgroup_size(128) fn main(@builtin(global_invocation_id) id:vec3<u32>){let index=invocation_index(id);if(index>=param(16u)){return;}let inner=param(17u);let outer=index/inner/param(18u);let tail=index%inner;var coordinate=(index/inner)%param(18u);
    ${operands.map((_,index)=>`if(coordinate<param(${20+index}u)){store(index,load${index}((outer*param(${20+index}u)+coordinate)*inner+tail));return;}
    coordinate-=param(${20+index}u);`).join('\n')}}`;
        groups=Math.ceil(product(shape)/128);
    } else throw new Error(`WebGPU operator ${operation} is not implemented`);
    return {shape,dtype,code,groups,uniforms,scratch,stages,operation,bindings,finalInputs,validatesIndices};
}

function softmax(count, logarithmic) {
    return header(count)+reduction+`@compute @workgroup_size(128) fn main(@builtin(workgroup_id) group:vec3<u32>,@builtin(local_invocation_index) lane:u32){let row=group_index(group);if(row>=param(17u)){return;}let width=param(16u);var maximum=-3.402823466e38;
for(var column=lane;column<width;column+=128u){maximum=max(maximum,load0(row*width+column));}partial[lane]=maximum;workgroupBarrier();for(var step=64u;step>0u;step/=2u){if(lane<step){partial[lane]=max(partial[lane],partial[lane+step]);}workgroupBarrier();}maximum=partial[0];workgroupBarrier();var sum=0.0;
for(var column=lane;column<width;column+=128u){sum+=exp(load0(row*width+column)-maximum);}let denominator=reduce_sum(sum,lane);workgroupBarrier();for(var column=lane;column<width;column+=128u){let value=load0(row*width+column)-maximum;store(row*width+column,${logarithmic?'value-log(denominator)':'exp(value)/denominator'});}}`;
}