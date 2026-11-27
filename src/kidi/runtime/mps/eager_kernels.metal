#include <metal_stdlib>
using namespace metal;
struct Geometry { uint count, width, other, heads, mode; float epsilon, gate_cap, up_cap; };
struct Quantization { uint count; float scale; };
struct Candidate { float value; int token; };
struct Selection { uint width, parts, final; };
struct AudioGeometry {
    uint count, input_length, input_channels, output_length, output_channels;
    int kernel_size, stride, padding, dilation;
    uint snake;
};
inline float snake_value(float value, float alpha) {
    if (abs(alpha) < 1e-9f) return value;
    float sine = precise::sin(alpha * value);
    return value + precise::divide(sine * sine, alpha + 1e-9f);
}
kernel void greedy_token(device const float* input [[buffer(0)]], device Candidate* candidates [[buffer(1)]],
                         device int* output [[buffer(2)]], constant Selection& geometry [[buffer(3)]],
                         uint2 group [[threadgroup_position_in_grid]], uint thread_index [[thread_index_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]], uint simd_group [[simdgroup_index_in_threadgroup]]) {
    threadgroup float values[8];
    threadgroup int tokens[8];
    threadgroup uint invalids[8];
    float best = -INFINITY;
    int token = INT_MAX;
    bool invalid = false;
    uint width = geometry.final ? geometry.parts : geometry.width;
    uint begin = geometry.final ? 0 : group.x * 1024;
    uint end = geometry.final ? width : min(width, begin + 1024);
    for (uint column = begin + thread_index; column < end; column += 256) {
        Candidate candidate = geometry.final ? candidates[group.y * geometry.parts + column]
                                            : Candidate{input[group.y * geometry.width + column], int(column)};
        invalid |= candidate.token < 0 || isnan(candidate.value) || candidate.value == INFINITY;
        if (candidate.value > best || (candidate.value == best && candidate.token < token)) {
            best = candidate.value;
            token = candidate.token;
        }
    }
    float maximum = simd_max(best);
    token = simd_min(best == maximum ? token : INT_MAX);
    invalid = simd_any(invalid);
    if (lane == 0) { values[simd_group] = maximum; tokens[simd_group] = token; invalids[simd_group] = invalid; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group == 0) {
        best = lane < 8 ? values[lane] : -INFINITY;
        maximum = simd_max(best);
        token = simd_min(lane < 8 && best == maximum ? tokens[lane] : INT_MAX);
        invalid = simd_any(lane < 8 && invalids[lane] != 0);
        if (lane == 0) {
            if (geometry.final) output[group.y] = invalid || !isfinite(maximum) ? -1 : token;
            else candidates[group.y * geometry.parts + group.x] = Candidate{maximum, invalid ? -1 : token};
        }
    }
}
kernel void round_to_half(device const float* input [[buffer(0)]], device half* output [[buffer(3)]],
                          constant Geometry& geometry [[buffer(4)]], uint index [[thread_position_in_grid]]) {
    if (index < geometry.count)
        output[index] = half(clamp(rint(input[index] / geometry.epsilon), -128.0f, 127.0f) * geometry.epsilon);
}
kernel void quantize_int8(device const float* input [[buffer(0)]], device char* output [[buffer(1)]],
                          constant Quantization& geometry [[buffer(2)]], uint index [[thread_position_in_grid]]) {
    if (index < geometry.count)
        output[index] = char(clamp(rint(input[index] / geometry.scale), -128.0f, 127.0f));
}
kernel void rms(device const float* input [[buffer(0)]], device const float* scale [[buffer(1)]],
                device const float* residual [[buffer(2)]], device const float* output_scale [[buffer(5)]],
                device float* output [[buffer(3)]], constant Geometry& geometry [[buffer(4)]],
                uint row [[threadgroup_position_in_grid]], uint thread_index [[thread_index_in_threadgroup]]) {
    threadgroup float partial[8];
    float square = 0;
    for (uint channel = thread_index; channel < geometry.width; channel += 256) {
        float value = input[row * geometry.width + channel];
        square += value * value;
    }
    square = simd_sum(square);
    if (thread_index % 32 == 0) partial[thread_index / 32] = square;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (thread_index == 0) {
        float sum = 0;
        for (uint index = 0; index < 8; ++index) sum += partial[index];
        float mean = precise::divide(sum, float(geometry.width)) + geometry.epsilon;
        partial[0] = precise::divide(1.0f, precise::sqrt(mean));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint channel = thread_index; channel < geometry.width; channel += 256) {
        float value = input[row * geometry.width + channel] * partial[0] * scale[channel];
        if (geometry.mode == 5 || geometry.mode == 6) value = residual[row * geometry.width + channel] + value;
        if (geometry.mode == 6) value *= output_scale[0];
        if (geometry.mode == 8) {
            uint half_width = geometry.width / 2;
            uint other_channel = channel < half_width ? channel + half_width : channel - half_width;
            float rotated = input[row * geometry.width + other_channel] * partial[0] * scale[other_channel];
            if (channel < half_width) rotated = -rotated;
            uint angle = ((row / geometry.heads) % geometry.other) * half_width + channel % half_width;
            value = value * residual[angle] + rotated * output_scale[angle];
        }
        if (geometry.mode == 10) {
            uint axis_width = geometry.width / 2;
            uint half_axis = axis_width / 2;
            uint axis = channel / axis_width;
            uint axis_channel = channel % axis_width;
            uint other_channel = axis * axis_width +
                (axis_channel < half_axis ? axis_channel + half_axis : axis_channel - half_axis);
            float rotated = input[row * geometry.width + other_channel] * partial[0] * scale[other_channel];
            if (axis_channel < half_axis) rotated = -rotated;
            uint position = (row / geometry.heads) % geometry.other;
            uint angle = (axis * geometry.other + position) * half_axis + axis_channel % half_axis;
            value = value * residual[angle] + rotated * output_scale[angle];
        }
        output[row * geometry.width + channel] = value;
    }
}
kernel void pointwise(device const float* input [[buffer(0)]], device const float* other [[buffer(1)]],
                      device const float* sine [[buffer(2)]], device float* output [[buffer(3)]],
                      constant Geometry& geometry [[buffer(4)]], uint index [[thread_position_in_grid]]) {
    if (index >= geometry.count) return;
    float value = input[index];
    if (geometry.mode == 0) output[index] = value + other[index % geometry.other];
    else if (geometry.mode == 1) output[index] = value * other[index % geometry.other];
    else if (geometry.mode == 2 || geometry.mode == 9) {
        float activated;
        if (value > 10.0f) activated = value;
        else if (value < -10.0f) activated = 0.0f;
        else activated = 0.5f * value * (1.0f + tanh(clamp(0.7978845608028654f * (value + 0.044715f * value * value * value), -10.0f, 10.0f)));
        output[index] = geometry.mode == 9 ? activated * other[index] : activated;
    }
    else if (geometry.mode == 3) output[index] = tanh(clamp(value, -10.0f, 10.0f));
    else if (geometry.mode == 11) output[index] = precise::divide(value, 1.0f + precise::exp(-value)) * other[index];
    else if (geometry.mode == 12) {
        uint half_width = geometry.width / 2;
        uint row = index / half_width;
        uint channel = index % half_width;
        float gate = input[row * geometry.width + channel];
        float up = input[row * geometry.width + half_width + channel];
        output[index] = precise::divide(gate, 1.0f + precise::exp(-gate)) * up;
    }
    else if (geometry.mode == 13) {
        // Softcapped SiLU gate: softcap(gate) * sigmoid(gate) * softcap(up), optionally scaled per row.
        uint half_width = geometry.width / 2;
        uint row = index / half_width;
        float gate = input[row * geometry.width + index % half_width];
        float up = input[row * geometry.width + half_width + index % half_width];
        float value = precise::tanh(gate / geometry.gate_cap) * geometry.gate_cap / (1.0f + precise::exp(-gate)) *
                      precise::tanh(up / geometry.up_cap) * geometry.up_cap;
        output[index] = geometry.heads ? value * other[row] : value;
    }
    else if (geometry.mode == 7) output[index] = clamp(rint(value / geometry.epsilon), -128.0f, 127.0f) * geometry.epsilon;
    else {
        uint half_width = geometry.width / 2;
        uint channel = index % geometry.width;
        uint position = (index / (geometry.width * geometry.heads)) % geometry.other;
        uint angle = position * half_width + channel % half_width;
        float rotated = channel < half_width ? -input[index + half_width] : input[index - half_width];
        output[index] = value * other[angle] + rotated * sine[angle];
    }
}
kernel void audio_unfold(device const float* input [[buffer(0)]], device const float* alpha [[buffer(1)]],
                         device float* output [[buffer(2)]], constant AudioGeometry& geometry [[buffer(3)]],
                         uint index [[thread_position_in_grid]]) {
    if (index >= geometry.count) return;
    uint kernel_index = index % uint(geometry.kernel_size);
    uint row = index / uint(geometry.kernel_size);
    uint channel = row % geometry.input_channels;
    uint output_position = row / geometry.input_channels;
    int input_position = int(output_position) * geometry.stride - geometry.padding +
                         int(kernel_index) * geometry.dilation;
    float value = input_position >= 0 && input_position < int(geometry.input_length)
        ? input[(uint(input_position) * geometry.input_channels) + channel] : 0.0f;
    output[index] = geometry.snake ? snake_value(value, alpha[channel]) : value;
}
kernel void snake_audio(device const float* input [[buffer(0)]], device const float* alpha [[buffer(1)]],
                        device float* output [[buffer(2)]], constant AudioGeometry& geometry [[buffer(3)]],
                        uint index [[thread_position_in_grid]]) {
    if (index < geometry.count)
        output[index] = snake_value(input[index], alpha[index % geometry.input_channels]);
}
kernel void conv_transpose_fold(device const float* projected [[buffer(0)]], device const float* bias [[buffer(1)]],
                                device float* output [[buffer(2)]], constant AudioGeometry& geometry [[buffer(3)]],
                                uint index [[thread_position_in_grid]]) {
    if (index >= geometry.count) return;
    uint channel = index % geometry.output_channels;
    int output_position = int(index / geometry.output_channels);
    float value = bias[channel];
    for (int kernel_index = 0; kernel_index < geometry.kernel_size; ++kernel_index) {
        int numerator = output_position + geometry.padding - kernel_index * geometry.dilation;
        if (numerator < 0 || numerator % geometry.stride) continue;
        uint input_position = uint(numerator / geometry.stride);
        if (input_position >= geometry.input_length) continue;
        uint projected_index =
            (input_position * geometry.output_channels + channel) * uint(geometry.kernel_size) + uint(kernel_index);
        value += projected[projected_index];
    }
    output[index] = value;
}

// Inkling-style fused operators. Every operand is a contiguous F32 tensor unless noted.
struct RouteGeometry { uint experts, top; float scale; };
// One simdgroup per token: sigmoid scores, biased top-k (lower index wins ties), weights normalized over the
// selected scores. Output [tokens, top, (expert, weight)]; at most 256 experts and 16 slots.
kernel void route_experts(device const float* logits [[buffer(0)]], device const float* bias [[buffer(1)]],
                          device float* output [[buffer(2)]], constant RouteGeometry& geometry [[buffer(3)]],
                          uint token [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    float scores[8], biased[8], chosen[16];
    for (uint i = 0; i < 8; ++i) {
        uint expert = lane + 32 * i;
        scores[i] = expert < geometry.experts ? 1.0f / (1.0f + precise::exp(-logits[token * geometry.experts + expert]))
                                              : 0.0f;
        biased[i] = expert < geometry.experts ? scores[i] + bias[expert] : -INFINITY;
    }
    float total = 0;
    for (uint slot = 0; slot < geometry.top; ++slot) {
        float best = -INFINITY;
        uint index = UINT_MAX;
        for (uint i = 0; i < 8; ++i)
            if (biased[i] > best) best = biased[i], index = lane + 32 * i;
        float peak = simd_max(best);
        uint winner = simd_min(best == peak ? index : UINT_MAX);
        uint row = min(winner / 32, 7u);
        float score = simd_shuffle(scores[row], ushort(winner % 32));
        if (winner % 32 == lane) biased[row] = -INFINITY;
        chosen[slot] = score;
        total += score;
        if (lane == 0) output[(token * geometry.top + slot) * 2] = float(winner);
    }
    if (lane == 0)
        for (uint slot = 0; slot < geometry.top; ++slot)
            output[(token * geometry.top + slot) * 2 + 1] = chosen[slot] / (total + 1e-20f) * geometry.scale;
}

struct MixGeometry { uint tokens, width, sources, queries; float epsilon; };
inline const device float* mix_source(uint index, device const float* s0, device const float* s1,
                                      device const float* s2, device const float* s3, device const float* s4,
                                      device const float* s5, device const float* s6) {
    switch (index) {
        case 0: return s0;
        case 1: return s1;
        case 2: return s2;
        case 3: return s3;
        case 4: return s4;
        case 5: return s5;
        default: return s6;
    }
}
// One threadgroup per token: RMS-normalize each source, softmax the scaled query logits over sources, then mix.
// Output [queries, tokens, width]; up to 7 sources and 4 queries. Uses float4 vectorized loads and avoids
// thread-local arrays so everything stays in registers.
kernel void residual_mix(device const float* s0 [[buffer(0)]], device const float* s1 [[buffer(1)]],
                         device const float* s2 [[buffer(2)]], device const float* s3 [[buffer(3)]],
                         device const float* s4 [[buffer(4)]], device const float* s5 [[buffer(5)]],
                         device const float* s6 [[buffer(6)]], device const float* queries [[buffer(7)]],
                         device float* output [[buffer(8)]], constant MixGeometry& geometry [[buffer(9)]],
                         uint token [[threadgroup_position_in_grid]], uint thread_index [[thread_index_in_threadgroup]],
                         uint threads [[threads_per_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                         uint simd [[simdgroup_index_in_threadgroup]]) {
    constexpr uint SOURCES = 7, QUERIES = 4, SUMS = SOURCES * (QUERIES + 1);
    threadgroup float reduction[8][SUMS];
    threadgroup float inverse[SOURCES], weights[QUERIES][SOURCES];
    const uint width = geometry.width, n_sources = geometry.sources, n_queries = geometry.queries;
    const uint groups = (threads + 31) / 32;
    const bool vec4 = (width & 3u) == 0u;
    const uint width4 = width >> 2;
    for (uint source = 0; source < n_sources; ++source) {
        const device float* src = mix_source(source, s0, s1, s2, s3, s4, s5, s6) + token * width;
        float sq = 0, d0 = 0, d1 = 0, d2 = 0, d3 = 0;
        if (vec4) {
            const device float4* src4 = (const device float4*)src;
            const device float4* q0 = (const device float4*)queries;
            const device float4* q1 = (const device float4*)(queries + width);
            const device float4* q2 = (const device float4*)(queries + 2 * width);
            const device float4* q3 = (const device float4*)(queries + 3 * width);
            for (uint c = thread_index; c < width4; c += threads) {
                const float4 v = src4[c];
                sq += dot(v, v);
                d0 += dot(q0[c], v);
                if (n_queries > 1) d1 += dot(q1[c], v);
                if (n_queries > 2) d2 += dot(q2[c], v);
                if (n_queries > 3) d3 += dot(q3[c], v);
            }
        } else {
            for (uint column = thread_index; column < width; column += threads) {
                const float value = src[column];
                sq += value * value;
                d0 += queries[column] * value;
                if (n_queries > 1) d1 += queries[width + column] * value;
                if (n_queries > 2) d2 += queries[2 * width + column] * value;
                if (n_queries > 3) d3 += queries[3 * width + column] * value;
            }
        }
        sq = simd_sum(sq);
        d0 = simd_sum(d0);
        if (n_queries > 1) d1 = simd_sum(d1);
        if (n_queries > 2) d2 = simd_sum(d2);
        if (n_queries > 3) d3 = simd_sum(d3);
        if (lane == 0) {
            reduction[simd][source] = sq;
            reduction[simd][SOURCES + source] = d0;
            if (n_queries > 1) reduction[simd][2 * SOURCES + source] = d1;
            if (n_queries > 2) reduction[simd][3 * SOURCES + source] = d2;
            if (n_queries > 3) reduction[simd][4 * SOURCES + source] = d3;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd == 0) {
        float inv_src = 0.0f;
        if (lane < n_sources) {
            float total = 0;
            for (uint group = 0; group < groups; ++group) total += reduction[group][lane];
            inv_src = precise::rsqrt(total / float(width) + geometry.epsilon);
            inverse[lane] = inv_src;
        }
        const float inv_sqrt_w = precise::rsqrt(float(width));
        for (uint query = 0; query < n_queries; ++query) {
            float w = -INFINITY;
            if (lane < n_sources) {
                float dot_val = 0;
                const uint idx = SOURCES * (query + 1) + lane;
                for (uint group = 0; group < groups; ++group) dot_val += reduction[group][idx];
                w = dot_val * inv_src * inv_sqrt_w;
            }
            const float peak = simd_max(w);
            const float ex = lane < n_sources ? precise::exp(w - peak) : 0.0f;
            const float denom = simd_sum(ex);
            if (lane < n_sources) weights[query][lane] = ex / denom * inv_src;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (vec4) {
        device float4* out0 = (device float4*)(output + token * width);
        device float4* out1 = (device float4*)(output + (geometry.tokens + token) * width);
        device float4* out2 = (device float4*)(output + (2 * geometry.tokens + token) * width);
        device float4* out3 = (device float4*)(output + (3 * geometry.tokens + token) * width);
        for (uint c = thread_index; c < width4; c += threads) {
            float4 o0 = 0, o1 = 0, o2 = 0, o3 = 0;
            for (uint source = 0; source < n_sources; ++source) {
                const float4 v = ((const device float4*)(mix_source(source, s0, s1, s2, s3, s4, s5, s6) + token * width))[c];
                o0 += weights[0][source] * v;
                if (n_queries > 1) o1 += weights[1][source] * v;
                if (n_queries > 2) o2 += weights[2][source] * v;
                if (n_queries > 3) o3 += weights[3][source] * v;
            }
            out0[c] = o0;
            if (n_queries > 1) out1[c] = o1;
            if (n_queries > 2) out2[c] = o2;
            if (n_queries > 3) out3[c] = o3;
        }
    } else {
        for (uint column = thread_index; column < width; column += threads) {
            float o0 = 0, o1 = 0, o2 = 0, o3 = 0;
            for (uint source = 0; source < n_sources; ++source) {
                const float v = mix_source(source, s0, s1, s2, s3, s4, s5, s6)[token * width + column];
                o0 += weights[0][source] * v;
                if (n_queries > 1) o1 += weights[1][source] * v;
                if (n_queries > 2) o2 += weights[2][source] * v;
                if (n_queries > 3) o3 += weights[3][source] * v;
            }
            output[token * width + column] = o0;
            if (n_queries > 1) output[(geometry.tokens + token) * width + column] = o1;
            if (n_queries > 2) output[(2 * geometry.tokens + token) * width + column] = o2;
            if (n_queries > 3) output[(3 * geometry.tokens + token) * width + column] = o3;
        }
    }
}

struct ConvolutionGeometry { uint count, channels, width, ring, groups, flags; float scale; };
// Causal depthwise convolution with lag-0-first taps. Ring form reads earlier rows from `state [ring, channels]`
// at slot-relative positions; the padded form (flag 8) reads the previous width - 1 rows from `input` itself.
// Flags: 1 dynamic taps, 2 residual, 4 write the current input into the ring slot (one token only).
kernel void short_convolution(device const float* input [[buffer(0)]], device float* state [[buffer(1)]],
                              device const float* weight [[buffer(2)]], device const int* slot [[buffer(3)]],
                              device const float* taps [[buffer(4)]], device const float* residual [[buffer(5)]],
                              device float* output [[buffer(6)]], constant ConvolutionGeometry& geometry [[buffer(7)]],
                              uint index [[thread_position_in_grid]]) {
    if (index >= geometry.count) return;
    const uint channels = geometry.channels, width = geometry.width;
    const uint token = index / channels, channel = index % channels;
    const uint group = channel / (channels / geometry.groups);
    const bool padded = geometry.flags & 8;
    const int ring = int(geometry.ring), base = padded ? 0 : slot[0];
    const float current = input[(padded ? token + width - 1 : token) * channels + channel];
    float sum = current;
    for (uint lag = 0; lag < width; ++lag) {
        float tap = weight[lag * channels + channel];
        if (geometry.flags & 1) tap += taps[(token * geometry.groups + group) * width + lag];
        float value;
        if (padded) value = input[(token + width - 1 - lag) * channels + channel];
        else if (int(token) - int(lag) >= 0) value = input[(token - lag) * channels + channel];
        else value = state[uint(((base + int(token) - int(lag)) % ring + ring) % ring) * channels + channel];
        sum += tap * value;
    }
    float result = sum * geometry.scale;
    if (geometry.flags & 2) result += residual[index];
    if (geometry.flags & 4) state[uint((base % ring + ring) % ring) * channels + channel] = current;
    output[index] = result;
}

struct ClipGeometry { uint rows, width, scales; float threshold; };
// One simdgroup per row: optionally scale by `scale [S, width]` (row % S), then shrink rows whose L2 norm exceeds
// the threshold.
kernel void l2_clip(device const float* input [[buffer(0)]], device const float* scale [[buffer(1)]],
                    device float* output [[buffer(2)]], constant ClipGeometry& geometry [[buffer(3)]],
                    uint2 group [[threadgroup_position_in_grid]], uint simd [[simdgroup_index_in_threadgroup]],
                    uint lane [[thread_index_in_simdgroup]]) {
    const uint row = group.x * 4 + simd;
    if (row >= geometry.rows) return;
    const uint width = geometry.width;
    const uint offset = geometry.scales ? (row * width) % geometry.scales : 0;
    float squares = 0;
    for (uint column = lane; column < width; column += 32) {
        float value = input[row * width + column] * (geometry.scales ? scale[offset + column] : 1.0f);
        squares += value * value;
    }
    const float factor = min(1.0f, geometry.threshold / (precise::sqrt(simd_sum(squares)) + 1e-6f));
    for (uint column = lane; column < width; column += 32)
        output[row * width + column] =
            input[row * width + column] * (geometry.scales ? scale[offset + column] : 1.0f) * factor;
}

struct RelativeAttentionGeometry {
    uint heads, key_heads, start, window, extent, width, key_stride, capacity, has_positions;
    float log_alpha, log_floor;
    uint kv_dtype, scale_stride, scale_group, ring_mode;
};

inline float read_kv_val(device const uchar* buf, device const float* scales, uint slot, uint col, uint stride,
                         uint kv_dtype, uint scale_stride, uint scale_group) {
    const ulong idx = ulong(slot) * stride + col;
    if (kv_dtype == 0u) return ((device const float*)buf)[idx];
    if (kv_dtype == 1u) return as_type<float>(uint(((device const ushort*)buf)[idx]) << 16);
    const float s = scales[ulong(slot) * scale_stride + col / scale_group];
    return float(((device const char*)buf)[idx]) * s;
}

// One threadgroup per (token, head), one simdgroup per key stride, online softmax. Supports F32 (kv_dtype=0),
// BF16 (kv_dtype=1), and group-32 INT8 (kv_dtype=2) caches, plus SWA ring modulo indexing (ring_mode=1).
kernel void relative_attention(device const float* query [[buffer(0)]], device const uchar* key [[buffer(1)]],
                               device const uchar* value [[buffer(2)]], device const float* relative [[buffer(3)]],
                               device const float* sink [[buffer(4)]], device const float* gate [[buffer(5)]],
                               device const int* positions [[buffer(6)]], device float* output [[buffer(7)]],
                               constant RelativeAttentionGeometry& geometry [[buffer(8)]],
                               device const float* key_scales [[buffer(9)]],
                               device const float* value_scales [[buffer(10)]],
                               uint2 group [[threadgroup_position_in_grid]],
                               uint thread_index [[thread_index_in_threadgroup]],
                               uint2 group_size [[threads_per_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                               uint simd [[simdgroup_index_in_threadgroup]]) {
    constexpr uint GROUPS = 8, VALUES = 8;
    const uint threads = group_size.x;
    threadgroup float peaks[GROUPS], totals[GROUPS], partial[GROUPS][VALUES * 32];
    const uint token = group.x, head = group.y, width = geometry.width, values = width / 32;
    const uint key_head = head / (geometry.heads / geometry.key_heads), groups = threads / 32;
    const int raw_pos = geometry.has_positions ? positions[token] : int(geometry.start + token);
    const int position = geometry.ring_mode ? raw_pos : min(raw_pos, int(geometry.capacity) - 1);
    const int first = geometry.window ? max(0, position - int(geometry.window) + 1) : 0;
    const float tau = geometry.log_floor > 0
                          ? 1.0f + geometry.log_alpha *
                                       precise::log(max(1.0f, float(position + 1) / geometry.log_floor))
                          : 1.0f;
    const float scale = precise::rsqrt(float(width));
    const uint row = token * geometry.heads + head;
    float peak = -1e30f, total = 0;
    if (width == 128) {
        const uint qbase = row * 128 + lane;
        const float q0 = query[qbase + 0] * scale, q1 = query[qbase + 32] * scale;
        const float q2 = query[qbase + 64] * scale, q3 = query[qbase + 96] * scale;
        float a0 = 0, a1 = 0, a2 = 0, a3 = 0;
        const uint key_col = key_head * 128 + lane;
        for (int k = first + int(simd); k <= position; k += int(groups)) {
            const uint slot = geometry.ring_mode ? (uint(k) % geometry.capacity) : uint(k);
            const float k0 = read_kv_val(key, key_scales, slot, key_col + 0, geometry.key_stride, geometry.kv_dtype,
                                         geometry.scale_stride, geometry.scale_group);
            const float k1 = read_kv_val(key, key_scales, slot, key_col + 32, geometry.key_stride, geometry.kv_dtype,
                                         geometry.scale_stride, geometry.scale_group);
            const float k2 = read_kv_val(key, key_scales, slot, key_col + 64, geometry.key_stride, geometry.kv_dtype,
                                         geometry.scale_stride, geometry.scale_group);
            const float k3 = read_kv_val(key, key_scales, slot, key_col + 96, geometry.key_stride, geometry.kv_dtype,
                                         geometry.scale_stride, geometry.scale_group);
            const float dot = q0 * k0 + q1 * k1 + q2 * k2 + q3 * k3;
            float logit = simd_sum(dot);
            const uint distance = uint(position - k);
            if (distance < geometry.extent) logit += relative[row * geometry.extent + distance];
            logit *= tau;
            const float next = max(peak, logit), decay = precise::exp(peak - next), weight = precise::exp(logit - next);
            total = total * decay + weight;
            const float v0 = read_kv_val(value, value_scales, slot, key_col + 0, geometry.key_stride, geometry.kv_dtype,
                                         geometry.scale_stride, geometry.scale_group);
            const float v1 = read_kv_val(value, value_scales, slot, key_col + 32, geometry.key_stride, geometry.kv_dtype,
                                         geometry.scale_stride, geometry.scale_group);
            const float v2 = read_kv_val(value, value_scales, slot, key_col + 64, geometry.key_stride, geometry.kv_dtype,
                                         geometry.scale_stride, geometry.scale_group);
            const float v3 = read_kv_val(value, value_scales, slot, key_col + 96, geometry.key_stride, geometry.kv_dtype,
                                         geometry.scale_stride, geometry.scale_group);
            a0 = a0 * decay + weight * v0;
            a1 = a1 * decay + weight * v1;
            a2 = a2 * decay + weight * v2;
            a3 = a3 * decay + weight * v3;
            peak = next;
        }
        if (lane == 0) peaks[simd] = peak, totals[simd] = total;
        partial[simd][0 + lane] = a0;
        partial[simd][32 + lane] = a1;
        partial[simd][64 + lane] = a2;
        partial[simd][96 + lane] = a3;
    } else {
        float q[VALUES], accumulator[VALUES];
        for (uint i = 0; i < values; ++i) {
            q[i] = query[row * width + i * 32 + lane] * scale;
            accumulator[i] = 0;
        }
        for (int k = first + int(simd); k <= position; k += int(groups)) {
            const uint slot = geometry.ring_mode ? (uint(k) % geometry.capacity) : uint(k);
            float dot = 0;
            for (uint i = 0; i < values; ++i) {
                const uint col = key_head * width + i * 32 + lane;
                dot += q[i] * read_kv_val(key, key_scales, slot, col, geometry.key_stride, geometry.kv_dtype,
                                          geometry.scale_stride, geometry.scale_group);
            }
            float logit = simd_sum(dot);
            const uint distance = uint(position - k);
            if (distance < geometry.extent) logit += relative[row * geometry.extent + distance];
            logit *= tau;
            const float next = max(peak, logit), decay = precise::exp(peak - next), weight = precise::exp(logit - next);
            total = total * decay + weight;
            for (uint i = 0; i < values; ++i) {
                const uint col = key_head * width + i * 32 + lane;
                accumulator[i] = accumulator[i] * decay +
                                 weight * read_kv_val(value, value_scales, slot, col, geometry.key_stride,
                                                      geometry.kv_dtype, geometry.scale_stride, geometry.scale_group);
            }
            peak = next;
        }
        if (lane == 0) peaks[simd] = peak, totals[simd] = total;
        for (uint i = 0; i < values; ++i) partial[simd][i * 32 + lane] = accumulator[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float sink_logit = sink[row];
    float combined = sink_logit;
    for (uint i = 0; i < groups; ++i) combined = max(combined, peaks[i]);
    float denominator = precise::exp(sink_logit - combined);
    for (uint i = 0; i < groups; ++i) denominator += totals[i] * precise::exp(peaks[i] - combined);
    const float inv_denom = 1.0f / denominator;
    for (uint column = thread_index; column < width; column += threads) {
        float sum = 0;
        for (uint i = 0; i < groups; ++i) sum += partial[i][column] * precise::exp(peaks[i] - combined);
        const uint index = row * width + column;
        output[index] = sum * inv_denom / (1.0f + precise::exp(-gate[index]));
    }
}

struct SliceGeometry { uint count, outer, src_extent, dst_extent, start, inner; };
kernel void slice_f32(device const float* input [[buffer(0)]], device float* output [[buffer(1)]],
                      constant SliceGeometry& geometry [[buffer(2)]], uint index [[thread_position_in_grid]]) {
    if (index >= geometry.count) return;
    const uint inner = geometry.inner, dst_stride = geometry.dst_extent * inner;
    const uint outer = index / dst_stride, rem = index % dst_stride;
    const uint axis = rem / inner, col = rem % inner;
    output[index] = input[(outer * geometry.src_extent + geometry.start + axis) * inner + col];
}

struct ConcatGeometry { uint count, inputs, inner, out_extent; uint extents[4]; uint offsets[4]; };
kernel void concat_f32(device const float* i0 [[buffer(0)]], device const float* i1 [[buffer(1)]],
                       device const float* i2 [[buffer(2)]], device const float* i3 [[buffer(3)]],
                       device float* output [[buffer(4)]], constant ConcatGeometry& geometry [[buffer(5)]],
                       uint index [[thread_position_in_grid]]) {
    if (index >= geometry.count) return;
    const uint inner = geometry.inner, out_stride = geometry.out_extent * inner;
    const uint outer = index / out_stride, rem = index % out_stride;
    const uint axis = rem / inner, col = rem % inner;
    const device float* srcs[4] = {i0, i1, i2, i3};
    uint src = 0;
    for (uint s = 1; s < geometry.inputs; ++s)
        if (axis >= geometry.offsets[s]) src = s;
    const uint local_axis = axis - geometry.offsets[src];
    output[index] = srcs[src][(outer * geometry.extents[src] + local_axis) * inner + col];
}

struct CastGeometry { uint count, mode; };
// mode 0: BF16->F32, 1: F32->BF16, 2: F16->F32, 3: F32->F16
kernel void cast_values(device const uchar* input [[buffer(0)]], device uchar* output [[buffer(1)]],
                        constant CastGeometry& geometry [[buffer(2)]], uint index [[thread_position_in_grid]]) {
    if (index >= geometry.count) return;
    if (geometry.mode == 0u) {
        const uint bits = uint(((device const ushort*)input)[index]) << 16;
        ((device float*)output)[index] = as_type<float>(bits);
    } else if (geometry.mode == 1u) {
        const uint bits = as_type<uint>(((device const float*)input)[index]);
        const uint rounded = bits + 0x7FFFu + ((bits >> 16) & 1u);
        ((device ushort*)output)[index] = ushort(rounded >> 16);
    } else if (geometry.mode == 2u) {
        ((device float*)output)[index] = float(((device const half*)input)[index]);
    } else {
        ((device half*)output)[index] = half(((device const float*)input)[index]);
    }
}

struct BlendGeometry { uint width; float epsilon; };
inline float threadgroup_sum(float value, threadgroup float* shared, uint lane, uint simd, uint threads) {
    value = simd_sum(value);
    if (lane == 0) shared[simd] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0;
    for (uint i = 0; i < (threads + 31) / 32; ++i) total += shared[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return total;
}
// Unit-RMS-normalizes `destination` and `message` rows into the concatenated [rows, 2 * width] output.
kernel void blend_prologue(device const float* destination [[buffer(0)]], device const float* message [[buffer(1)]],
                           device float* output [[buffer(4)]], constant BlendGeometry& geometry [[buffer(5)]],
                           uint row [[threadgroup_position_in_grid]], uint thread_index [[thread_index_in_threadgroup]],
                           uint threads [[threads_per_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                           uint simd [[simdgroup_index_in_threadgroup]]) {
    threadgroup float shared[32];
    const uint width = geometry.width;
    float squares_d = 0, squares_m = 0;
    for (uint column = thread_index; column < width; column += threads) {
        float d = destination[row * width + column], m = message[row * width + column];
        squares_d += d * d;
        squares_m += m * m;
    }
    const float inverse_d =
        precise::rsqrt(threadgroup_sum(squares_d, shared, lane, simd, threads) / float(width) + geometry.epsilon);
    const float inverse_m =
        precise::rsqrt(threadgroup_sum(squares_m, shared, lane, simd, threads) / float(width) + geometry.epsilon);
    for (uint column = thread_index; column < width; column += threads) {
        output[row * 2 * width + column] = destination[row * width + column] * inverse_d;
        output[row * 2 * width + width + column] = message[row * width + column] * inverse_m;
    }
}
// mixed = destination * (1 + tanh(forget)) + message * tanh(gain), then RMS-normalized and scaled by `norm`.
kernel void blend_epilogue(device const float* destination [[buffer(0)]], device const float* message [[buffer(1)]],
                           device const float* gain_forget [[buffer(2)]], device const float* norm [[buffer(3)]],
                           device float* output [[buffer(4)]], constant BlendGeometry& geometry [[buffer(5)]],
                           uint row [[threadgroup_position_in_grid]], uint thread_index [[thread_index_in_threadgroup]],
                           uint threads [[threads_per_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                           uint simd [[simdgroup_index_in_threadgroup]]) {
    threadgroup float shared[32];
    const uint width = geometry.width;
    float squares = 0;
    for (uint column = thread_index; column < width; column += threads) {
        const float gain = gain_forget[row * 2 * width + column], forget = gain_forget[row * 2 * width + width + column];
        const float mixed = destination[row * width + column] * (1.0f + precise::tanh(forget)) +
                            message[row * width + column] * precise::tanh(gain);
        output[row * width + column] = mixed;
        squares += mixed * mixed;
    }
    const float inverse =
        precise::rsqrt(threadgroup_sum(squares, shared, lane, simd, threads) / float(width) + geometry.epsilon);
    for (uint column = thread_index; column < width; column += threads)
        output[row * width + column] *= inverse * norm[column];
}

struct QuantizeCacheGeometry {
    uint tokens, width, groups, capacity;
};
// Group-32 dynamic symmetric INT8 KV cache writer: 1 simdgroup (32 lanes) per (token, group).
kernel void quantize_cache_i8(device const float* input [[buffer(0)]], device const int* positions [[buffer(1)]],
                              device char* cache [[buffer(2)]], device float* scales [[buffer(3)]],
                              constant QuantizeCacheGeometry& geom [[buffer(4)]],
                              uint2 group [[threadgroup_position_in_grid]],
                              uint lane [[thread_index_in_simdgroup]]) {
    const uint g = group.x, t = group.y;
    if (g >= geom.groups || t >= geom.tokens) return;
    const int pos = positions[t];
    if (pos < 0 || uint(pos) >= geom.capacity) return;
    const uint col = g * 32 + lane;
    const float v = input[ulong(t) * geom.width + col];
    const float peak = simd_max(abs(v));
    const float scale = peak > 0.0f ? (peak / 127.0f) : 1.0f;
    const float q = clamp(rint(v / scale), -127.0f, 127.0f);
    cache[ulong(pos) * geom.width + col] = char(int(q));
    if (lane == 0) scales[ulong(pos) * geom.groups + g] = scale;
}
