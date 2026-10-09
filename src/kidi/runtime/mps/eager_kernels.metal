#include <metal_stdlib>
using namespace metal;
struct Geometry { uint count, width, other, heads, mode; float epsilon; };
struct Quantization { uint count; float scale; };
struct Candidate { float value; int token; };
struct Selection { uint width, parts, final; };
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
    if (thread_index < 8) {
        float square = 0;
        for (uint channel = thread_index; channel < geometry.width; channel += 8) {
            float value = input[row * geometry.width + channel];
            square += value * value;
        }
        partial[thread_index] = square;
    }
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
