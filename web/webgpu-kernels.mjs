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

function header(count, extra = '') {
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
    let dtype=param(${9 + index}u);
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
const reduction = `var<workgroup> partial:array<f32,128>;
fn reduce_sum(value:f32,lane:u32)->f32 {partial[lane]=value;workgroupBarrier();for(var step=64u;step>0u;step/=2u){if(lane<step){partial[lane]+=partial[lane+step];}workgroupBarrier();}return partial[0];}`;

export function makeProgram(spec, features) {
    const operands = spec.inputs, inputShape = operands[0].shape, attributes = spec.attributes;
    const uniforms = new Uint32Array(64);
    operands.forEach((operand, index) => { uniforms[9 + index] = operand.dtype; });
    let shape = [...inputShape], dtype = spec.dtype, code, groups, scratch, stages, bindings, finalInputs;
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
        code=header(operands.length)+reduction+`
@compute @workgroup_size(128) fn main(@builtin(workgroup_id) group:vec3<u32>,@builtin(local_invocation_index) lane:u32){
let row=group_index(group);if(row>=param(17u)){return;}let width=param(16u);var sum=0.0;var mean=0.0;
${layerNorm?'for(var channel=lane;channel<width;channel+=128u){sum+=load0(row*width+channel);}mean=reduce_sum(sum,lane)/f32(width);workgroupBarrier();sum=0.0;':''}
for(var channel=lane;channel<width;channel+=128u){let value=load0(row*width+channel)-mean;sum+=value*value;}
let inverse=inverseSqrt(reduce_sum(sum,lane)/f32(width)+scalar(18u));workgroupBarrier();
for(var channel=lane;channel<width;channel+=128u){let index=row*width+channel;
${rotary?`let half=width/2u;let partner=(channel+half)%width;let position=(row/param(19u))%param(20u);
let first=load0(index)*inverse*load1(channel);let other=load0(row*width+partner)*inverse*load1(partner);
let normalized=first*load2(position*half+channel%half)+select(-other,other,channel>=half)*load3(position*half+channel%half);`:
`let normalized=(load0(index)-mean)*inverse*load1(channel)${layerNorm?'+load2(channel)':''};`}
${value}store(index,value);}}`;
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
    } else if (operation==='packed_linear') {
        const width=inputShape.at(-1), rows=product(inputShape)/width, columns=operands[1].shape[0];
        const [bits, group]=attributes;
        shape[shape.length-1]=columns;dtype=11;
        uniforms[16]=width;uniforms[17]=columns;uniforms[18]=rows;uniforms[19]=group;
        uniforms[20]=floatBits(spec.epsilon);uniforms[21]=attributes[2]>>>0;
        const quantized=spec.epsilon>0;
        if(quantized&&group!==width)throw new Error('WebGPU calibrated projections require one scale per output channel');
        const paddedWidth=Math.ceil(width/4)*4;uniforms[22]=paddedWidth;
        if(quantized){
            const quantUniforms=new Uint32Array(uniforms);
            scratch={bytes:rows*paddedWidth,uniforms:quantUniforms,groups:Math.ceil(rows*paddedWidth/512),code:header(1)+`
@compute @workgroup_size(128) fn main(@builtin(global_invocation_id) id:vec3<u32>){let index=invocation_index(id);let row=index/(param(22u)/4u);let first=(index%(param(22u)/4u))*4u;if(row>=param(18u)){return;}var packed=0u;
for(var component=0u;component<4u;component++){var value=0i;if(first+component<param(16u)){value=i32(round_even(clamp(load0(row*param(16u)+first+component)/scalar(20u),-128.0,127.0)));}packed|=(u32(value)&255u)<<(component*8u);}output[index]=packed;}`};
        }
        const subgroups=features.has('subgroups');
        // Weights reach the GPU interleaved, so one shift and mask yields the four bytes of a dot product.
        const laneMask = bits===8 ? '0xffffffffu' : bits===4 ? '0x0f0f0f0fu' : '0x03030303u';
        const signExtension = bits===8 ? 'expanded' : bits===4 ? 'expanded|((expanded&0x08080808u)*30u)' : 'expanded|((expanded&0x02020202u)*126u)';
        const packFour = width%4 ?
            `var packed=0u;for(var component=0u;component<4u;component++){if(channel+component<width){packed|=(u32(weight(column*width+channel+component))&255u)<<(component*8u);}}` :
            `let offset=column*width+channel;let expanded=(input1[param(1u)/4u+offset/${32/bits}u]>>(${bits}u*((offset%${32/bits}u)/4u)))&${laneMask};let packed=${signExtension};`;
        const perLane=32/bits, parts=8/bits, lanes=8, groupThreads=256, columnsPerGroup=groupThreads/lanes;
        const chunk=Math.min(width,2048);
        const staged=width%chunk===0&&chunk%(lanes*perLane)===0&&(quantized||group%perLane===0);
        const singleScale=!quantized&&group>=width;
        const tileEntries=chunk/4;
        uniforms[24]=Math.ceil(columns/(staged?columnsPerGroup:4));
        // Columns in a workgroup share one activation row, so stage it once and read it as vectors.
        const stagedLoop=`
for(var base=0u;base<width;base+=${chunk}u){
for(var index=thread;index<${tileEntries}u;index+=${groupThreads}u){${quantized
?'tile[index]=input0[param(0u)/4u+(row*param(22u)+base)/4u+index];'
:'let source=param(0u)/4u+row*width+base+index*4u;tile[index]=bitcast<vec4<f32>>(vec4<u32>(input0[source],input0[source+1u],input0[source+2u],input0[source+3u]));'}}
workgroupBarrier();
if(column<columns){
for(var offset=lane*${perLane}u;offset<${chunk}u;offset+=${lanes*perLane}u){
let channel=base+offset;
let packed_word=input1[param(1u)/4u+(column*width+channel)/${32/bits}u];
${quantized?Array.from({length:parts},(_,part)=>`{let expanded=(packed_word>>${part*bits}u)&${laneMask};let packed=${signExtension};sum+=dot4I8Packed(tile[offset/4u+${part}u],packed);}`).join('\n')
:`var partial_sum=0.0;
${Array.from({length:parts},(_,part)=>`{
let raw=(vec4<u32>(packed_word)>>vec4<u32>(${part*bits}u,${8+part*bits}u,${16+part*bits}u,${24+part*bits}u))&vec4<u32>(${(1<<bits)-1}u);
partial_sum+=dot(tile[offset/4u+${part}u],vec4<f32>(vec4<i32>(raw^vec4<u32>(${1<<(bits-1)}u))-vec4<i32>(${1<<(bits-1)}i)));}`).join('\n')}
sum+=partial_sum${singleScale?'':'*load2(column*(width/param(19u))+channel/param(19u))'};`}
}
}
workgroupBarrier();
}`;
        const fallback=`if(row<param(18u)&&column<columns){${quantized
            ?`for(var channel=lane*4u;channel<width;channel+=128u){${packFour}sum+=dot4I8Packed(input0[param(0u)/4u+(row*param(22u)+channel)/4u],packed);}`
            :`for(var channel=lane;channel<width;channel+=32u){sum+=load0(row*width+channel)*f32(weight(column*width+channel))*load2(column*(width/param(19u))+channel/param(19u));}`}}`;
        const threads=staged?groupThreads:128, laneCount=staged?lanes:32;
        const declarations=`${quantized?`var<workgroup> partial:array<i32,${threads}>;`:`var<workgroup> partial:array<f32,${threads}>;`}${staged?`var<workgroup> tile:array<${quantized?'u32':'vec4<f32>'},${tileEntries}>;`:''}`;
        code=header(3,`${quantized?'requires packed_4x8_integer_dot_product;':''}${subgroups&&!staged?'enable subgroups;':''}`)+declarations+`
fn weight(index:u32)->i32 {let word=input1[param(1u)/4u+index/${32/bits}u];let slot=index%${32/bits}u;return i32(((word>>(8u*(slot%4u)+${bits}u*(slot/4u)))&${(1<<bits)-1}u)<<${32-bits}u)>>${32-bits}u;}
@compute @workgroup_size(${threads}) fn main(@builtin(workgroup_id) group:vec3<u32>,@builtin(local_invocation_index) thread:u32${subgroups&&!staged?',@builtin(subgroup_size) subgroup_width:u32':''}){
let width=param(16u);let columns=param(17u);let block=group_index(group);let per_row=param(24u);
let row=block/per_row;let column=(block%per_row)*${staged?columnsPerGroup:4}u+thread/${laneCount}u;let lane=thread%${laneCount}u;
var sum=${quantized?'0i':'0.0'};
${staged?stagedLoop:fallback}
${subgroups&&!staged?'if(subgroup_width==32u){sum=subgroupAdd(sum);}else{':''}
partial[thread]=sum;workgroupBarrier();for(var step=${laneCount/2}u;step>0u;step/=2u){if(lane<step){partial[thread]+=partial[thread+step];}workgroupBarrier();}sum=partial[thread];${subgroups&&!staged?'}':''}
if(lane==0u&&column<columns){let value=${quantized?'f32(sum)*scalar(20u)*load2(column)':staged&&singleScale?'sum*load2(column)':'sum'};store(row*columns+column,calibrated(value,scalar(21u)));}}`;
        groups=rows*uniforms[24];
        if(quantized && rows>=4){
            uniforms[23]=Math.ceil(columns/32);
            code=header(3,'requires packed_4x8_integer_dot_product;')+`
var<workgroup> activation_tile:array<u32,256>;var<workgroup> weight_tile:array<u32,1024>;
fn weight(index:u32)->i32 {let word=input1[param(1u)/4u+index/${32/bits}u];let slot=index%${32/bits}u;return i32(((word>>(8u*(slot%4u)+${bits}u*(slot/4u)))&${(1<<bits)-1}u)<<${32-bits}u)>>${32-bits}u;}
@compute @workgroup_size(256) fn main(@builtin(workgroup_id) group:vec3<u32>,@builtin(local_invocation_index) lane:u32){
let tile=group_index(group);let row_start=(tile/param(23u))*8u;let column_start=(tile%param(23u))*32u;
let local_row=lane/32u;let local_column=lane%32u;let width=param(16u);var sum=0i;
for(var base=0u;base<width;base+=128u){
    let input_row=row_start+local_row;let input_channel=base+local_column*4u;
    var input_word=0u;if(input_row<param(18u)&&input_channel<width){input_word=input0[param(0u)/4u+(input_row*param(22u)+input_channel)/4u];}activation_tile[lane]=input_word;
    for(var index=lane;index<1024u;index+=256u){let column=column_start+index/32u;let channel=base+(index%32u)*4u;var word=0u;
        if(column<param(17u)&&channel<width){${packFour}word=packed;}weight_tile[index]=word;
    }
    workgroupBarrier();
    for(var inner=0u;inner<32u;inner++){sum+=dot4I8Packed(activation_tile[local_row*32u+inner],weight_tile[local_column*32u+inner]);}
    workgroupBarrier();
}
let row=row_start+local_row;let column=column_start+local_column;
if(row<param(18u)&&column<param(17u)){store(row*param(17u)+column,calibrated(f32(sum)*scalar(20u)*load2(column),scalar(21u)));}}`;
            groups=Math.ceil(rows/8)*Math.ceil(columns/32);
        }
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
        // An INT8 cache folds its key scale into the softmax scale, so only the value scale is left to apply.
        const valueScale=attributes[4]?`*bitcast<f32>(${attributes[4]>>>0}u)`:'';
        // Four cached channels share one word, so read it once rather than through four dtype-dispatching loads.
        const byteKeys=operands[1].dtype===2;
        const keyQuad=byteKeys
            ?`let word=input1[(param(1u)+source+c)/4u];
let second=vec4<f32>(f32(bitcast<i32>(word<<24u)>>24u),f32(bitcast<i32>(word<<16u)>>24u),f32(bitcast<i32>(word<<8u)>>24u),f32(bitcast<i32>(word)>>24u));`
            :`let second=vec4<f32>(load1(source+c),load1(source+c+1u),load1(source+c+2u),load1(source+c+3u));`;
        const valueAt=operands[2].dtype===2
            ?'f32(bitcast<i32>(input2[(param(2u)+source+channel)/4u]<<((3u-(param(2u)+source+channel)%4u)*8u))>>24u)'
            :'load2(source+channel)';
        const keyBase='(((batch%param(25u))*param(21u)+KEY+param(22u))*param(18u)+keyHead)*width';
        if(queries===1){
            // Decode keeps scores in workgroup memory and rescales an online softmax, so one dispatch covers a head.
            const perThread=Math.ceil(width/256);
            const mask=operands.length===4?'+load3((((batch%param(29u))*param(28u)+head%param(28u))*param(26u))*param(27u)+key%param(27u))':'';
            // One workgroup per head leaves most of the device idle, so long key ranges are split and recombined.
            const splits=Math.max(1,Math.min(32,Math.floor(keyLength/256)));
            const slot=width+2;
            const scan=`
var<workgroup> qtile:array<f32,${width}>;
var<workgroup> weights:array<f32,256>;
var<workgroup> reduction:array<f32,256>;
@compute @workgroup_size(256) fn main(@builtin(workgroup_id) group:vec3<u32>,@builtin(local_invocation_index) thread:u32){
let block=group_index(group);let heads=param(17u);let width=param(16u);let keys=param(20u);
let owner=block/${splits}u;let piece=block%${splits}u;
let head=owner%heads;let batch=owner/heads;let keyHead=head/(heads/param(18u));
let queryBase=(batch*heads+head)*width;
let stride=(keys+${splits}u-1u)/${splits}u;
let rangeStart=min(keys,piece*stride);
let rangeEnd=min(keys,rangeStart+stride);
for(var c=thread;c<width;c+=256u){qtile[c]=load0(queryBase+c);}
var maximum=-3.402823466e38;var total=0.0;
var acc:array<f32,${perThread}>;
for(var p=0u;p<${perThread}u;p++){acc[p]=0.0;}
workgroupBarrier();
for(var base=rangeStart;base<rangeEnd;base+=256u){
let span=min(256u,rangeEnd-base);
let key=base+thread;
var score=-3.402823466e38;
if(key<rangeEnd){
let source=${keyBase.replace('KEY','key')};
var products=0.0;
for(var c=0u;c<width;c+=4u){
let first=vec4<f32>(qtile[c],qtile[c+1u],qtile[c+2u],qtile[c+3u]);
${keyQuad}
products+=dot(first,second);}
score=products*scalar(24u)${mask};}
reduction[thread]=score;workgroupBarrier();
for(var step=128u;step>0u;step/=2u){if(thread<step){reduction[thread]=max(reduction[thread],reduction[thread+step]);}workgroupBarrier();}
let highest=reduction[0];workgroupBarrier();
let updated=max(maximum,highest);
let rescale=exp(maximum-updated);
let weight=select(0.0,exp(score-updated),thread<span);
weights[thread]=weight;reduction[thread]=weight;workgroupBarrier();
for(var step=128u;step>0u;step/=2u){if(thread<step){reduction[thread]+=reduction[thread+step];}workgroupBarrier();}
total=total*rescale+reduction[0];
maximum=updated;
for(var p=0u;p<${perThread}u;p++){
let channel=thread+p*256u;
var partial=0.0;
if(channel<width){for(var j=0u;j<span;j++){let source=${keyBase.replace('KEY','base+j')};partial+=weights[j]*${valueAt};}}
acc[p]=acc[p]*rescale+partial;}
workgroupBarrier();}
`;
            if(splits===1){
                bindings=operands.length;
                code=header(operands.length)+scan+
`for(var p=0u;p<${perThread}u;p++){let channel=thread+p*256u;if(channel<width){store(queryBase+channel,acc[p]/total${valueScale});}}}`;
                groups=batch*heads;
            } else {
                // Each piece stores its unnormalised sum with the running maximum and total it was scaled by.
                stages=[{code:header(operands.length)+scan+
`let store_base=block*${slot}u;
for(var p=0u;p<${perThread}u;p++){let channel=thread+p*256u;if(channel<width){store(store_base+channel,acc[p]);}}
if(thread==0u){store(store_base+width,maximum);store(store_base+width+1u,total);}}`,
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
for(var p=0u;p<${perThread}u;p++){
let channel=thread+p*256u;
if(channel<width){
var sum=0.0;
for(var piece=0u;piece<${splits}u;piece++){
let head_slot=(block*${splits}u+piece)*${slot}u;
if(load0(head_slot+width+1u)>0.0){sum+=load0(head_slot+channel)*exp(load0(head_slot+width)-maximum);}}
store(block*width+channel,sum/total${valueScale});}}}`;
                groups=batch*heads;
            }
        } else {
            const scoreCode=header(operands.length)+`@compute @workgroup_size(128) fn main(@builtin(global_invocation_id) id:vec3<u32>){let index=invocation_index(id);let keys=param(20u);let queries=param(19u);let heads=param(17u);let width=param(16u);if(index>=param(23u)*heads*queries*keys){return;}
let key=index%keys;let query=(index/keys)%queries;let head=(index/keys/queries)%heads;let batch=index/keys/queries/heads;let keyHead=head/(heads/param(18u));
let queryBase=((batch*queries+query)*heads+head)*width;let keyBase=${keyBase.replace('KEY','key')};var sum=0.0;
for(var channel=0u;channel<width;channel+=4u){let first=vec4<f32>(load0(queryBase+channel),load0(queryBase+channel+1u),load0(queryBase+channel+2u),load0(queryBase+channel+3u));let second=vec4<f32>(load1(keyBase+channel),load1(keyBase+channel+1u),load1(keyBase+channel+2u),load1(keyBase+channel+3u));sum+=dot(first,second);}
store(index,sum*scalar(24u)${operands.length===4?'+load3((((batch%param(29u))*param(28u)+head%param(28u))*param(26u)+query%param(26u))*param(27u)+key%param(27u))':''});}`;
            const softUniforms=new Uint32Array(64);softUniforms[9]=11;softUniforms[16]=keyLength;softUniforms[17]=rows;
            stages=[{code:scoreCode,uniforms:new Uint32Array(uniforms),groups:Math.ceil(rows*keyLength/128),inputs:operands.map((_,index)=>index),bytes:rows*keyLength*4},
                {code:softmax(1,false),uniforms:softUniforms,groups:rows,inputs:['previous'],bytes:rows*keyLength*4}];
            uniforms[9]=11;uniforms[10]=operands[2].dtype;
            bindings=2;
            code=header(2)+`@compute @workgroup_size(128) fn main(@builtin(global_invocation_id) id:vec3<u32>){let index=invocation_index(id);let width=param(16u);let heads=param(17u);let queries=param(19u);let keys=param(20u);if(index>=param(23u)*queries*heads*width){return;}
let channel=index%width;let head=(index/width)%heads;let query=(index/width/heads)%queries;let batch=index/width/heads/queries;let scores=((batch*heads+head)*queries+query)*keys;let keyHead=head/(heads/param(18u));var sum=0.0;
for(var key=0u;key<keys;key++){let base=${keyBase.replace('KEY','key')};sum+=load0(scores+key)*load1(base+channel);}store(index,sum${valueScale});}`;
            groups=Math.ceil(product(shape)/128);
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
    return {shape,dtype,code,groups,uniforms,scratch,stages,operation,bindings,finalInputs};
}

function softmax(count, logarithmic) {
    return header(count)+reduction+`@compute @workgroup_size(128) fn main(@builtin(workgroup_id) group:vec3<u32>,@builtin(local_invocation_index) lane:u32){let row=group_index(group);if(row>=param(17u)){return;}let width=param(16u);var maximum=-3.402823466e38;
for(var column=lane;column<width;column+=128u){maximum=max(maximum,load0(row*width+column));}partial[lane]=maximum;workgroupBarrier();for(var step=64u;step>0u;step/=2u){if(lane<step){partial[lane]=max(partial[lane],partial[lane+step]);}workgroupBarrier();}maximum=partial[0];workgroupBarrier();var sum=0.0;
for(var column=lane;column<width;column+=128u){sum+=exp(load0(row*width+column)-maximum);}let denominator=reduce_sum(sum,lane);workgroupBarrier();for(var column=lane;column<width;column+=128u){let value=load0(row*width+column)-maximum;store(row*width+column,${logarithmic?'value-log(denominator)':'exp(value)/denominator'});}}`;
}