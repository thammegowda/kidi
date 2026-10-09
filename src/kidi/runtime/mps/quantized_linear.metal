#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
struct Geometry { uint rows; uint width; uint columns; };
kernel void quantize_rows(device const float* input [[buffer(0)]],
                          device char* quantized [[buffer(1)]],
                          device float* scales [[buffer(2)]],
                          device int* zero_points [[buffer(3)]],
                          constant Geometry& geometry [[buffer(4)]],
                          uint row [[threadgroup_position_in_grid]],
                          uint lane [[thread_index_in_threadgroup]]) {
    threadgroup float minima[256];
    threadgroup float maxima[256];
    float minimum = 0.0f, maximum = 0.0f;
    for (uint channel = lane; channel < geometry.width; channel += 256) {
        float value = input[row * geometry.width + channel];
        minimum = min(minimum, value);
        maximum = max(maximum, value);
    }
    minima[lane] = minimum;
    maxima[lane] = maximum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride > 0; stride /= 2) {
        if (lane < stride) {
            minima[lane] = min(minima[lane], minima[lane + stride]);
            maxima[lane] = max(maxima[lane], maxima[lane + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float range = maxima[0] - minima[0];
    float scale = range == 0.0f ? 1.0f : range / 255.0f;
    float unsigned_zero = range == 0.0f ? 0.0f : clamp(-minima[0] * (255.0f / range), 0.0f, 255.0f);
    int zero = int(unsigned_zero + 0.5f) - 128;
    if (lane == 0) { scales[row] = scale; zero_points[row] = zero; }
    float inverse_scale = 1.0f / scale;
    for (uint channel = lane; channel < geometry.width; channel += 256) {
        float value = rint(input[row * geometry.width + channel] * inverse_scale) + float(zero);
        quantized[row * geometry.width + channel] = char(clamp(value, -128.0f, 127.0f));
    }
}
kernel void linear_a8w8_packed(device const char* input [[buffer(0)]],
                        device const char* weights [[buffer(1)]],
                        device const float* input_scales [[buffer(2)]],
                        device const int* zero_points [[buffer(3)]],
                        device const float* weight_scales [[buffer(4)]],
                        device const float* bias [[buffer(5)]],
                        device float* output [[buffer(6)]],
                        constant Geometry& geometry [[buffer(7)]],
                        uint2 group [[threadgroup_position_in_grid]],
                        uint2 lane [[thread_position_in_threadgroup]]) {
    uint column = group.x * 4 + lane.y;
    uint row = group.y;
    uint stride = (geometry.width + 3) & ~3u;
    int sum = 0;
    if (column < geometry.columns) {
        int zero = zero_points[row];
        for (uint channel = lane.x * 4; channel < geometry.width; channel += 128) {
            if (channel + 3 < geometry.width && geometry.width % 4 == 0) {
                int4 activation = int4(*(device const char4*)(input + row * geometry.width + channel)) - zero;
                int4 weight = int4(*(device const char4*)(weights + column * stride + channel));
                int4 product = activation * weight;
                sum += product.x + product.y + product.z + product.w;
            } else {
                for (uint offset = 0; offset < 4 && channel + offset < geometry.width; ++offset)
                    sum += (int(input[row * geometry.width + channel + offset]) - zero) *
                           int(weights[column * stride + channel + offset]);
            }
        }
    }
    sum = simd_sum(sum);
    if (lane.x == 0 && column < geometry.columns)
        output[row * geometry.columns + column] = float(sum) * (input_scales[row] * weight_scales[column]) + bias[column];
}
kernel void linear_a8w8_tiled(device const char* input [[buffer(0)]],
                        device const char* weights [[buffer(1)]],
                        device const float* input_scales [[buffer(2)]],
                        device const int* zero_points [[buffer(3)]],
                        device const float* weight_scales [[buffer(4)]],
                        device const float* bias [[buffer(5)]],
                        device float* output [[buffer(6)]],
                        constant Geometry& geometry [[buffer(7)]],
                        uint2 group [[threadgroup_position_in_grid]],
                        uint thread_index [[thread_index_in_threadgroup]],
                        uint simd_group [[simdgroup_index_in_threadgroup]]) {
    threadgroup half activations[32 * 32];
    threadgroup half matrix[32 * 32];
    threadgroup float partials[32 * 32];
    int totals[8];
    for (uint index = 0; index < 8; ++index) totals[index] = 0;
    uint row_base = group.y * 32;
    uint column_base = group.x * 32;
    uint local_row = (simd_group / 2) * 16;
    uint local_column = (simd_group % 2) * 16;
    for (uint chunk = 0; chunk < geometry.width; chunk += 256) {
        simdgroup_float8x8 acc00 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        simdgroup_float8x8 acc01 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        simdgroup_float8x8 acc10 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        simdgroup_float8x8 acc11 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        for (uint base = chunk; base < min(chunk + 256, geometry.width); base += 32) {
            for (uint index = thread_index; index < 1024; index += 128) {
                uint row = row_base + index / 32;
                uint channel = base + index % 32;
                activations[index] = row < geometry.rows && channel < geometry.width ?
                    half(int(input[row * geometry.width + channel]) - zero_points[row]) : half(0);
                channel = base + index / 32;
                uint column = column_base + index % 32;
                matrix[index] = channel < geometry.width && column < geometry.columns ?
                    half(weights[channel * geometry.columns + column]) : half(0);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint offset = 0; offset < 32; offset += 8) {
                simdgroup_half8x8 left0, left1, right0, right1;
                simdgroup_load(left0, activations + local_row * 32 + offset, 32);
                simdgroup_load(left1, activations + (local_row + 8) * 32 + offset, 32);
                simdgroup_load(right0, matrix + offset * 32 + local_column, 32);
                simdgroup_load(right1, matrix + offset * 32 + local_column + 8, 32);
                simdgroup_multiply_accumulate(acc00, left0, right0, acc00);
                simdgroup_multiply_accumulate(acc01, left0, right1, acc01);
                simdgroup_multiply_accumulate(acc10, left1, right0, acc10);
                simdgroup_multiply_accumulate(acc11, left1, right1, acc11);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        simdgroup_store(acc00, partials + local_row * 32 + local_column, 32);
        simdgroup_store(acc01, partials + local_row * 32 + local_column + 8, 32);
        simdgroup_store(acc10, partials + (local_row + 8) * 32 + local_column, 32);
        simdgroup_store(acc11, partials + (local_row + 8) * 32 + local_column + 8, 32);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint index = 0; index < 8; ++index) totals[index] += int(partials[thread_index + index * 128]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint index = 0; index < 8; ++index) {
        uint position = thread_index + index * 128;
        uint row = row_base + position / 32;
        uint column = column_base + position % 32;
        if (row < geometry.rows && column < geometry.columns)
            output[row * geometry.columns + column] = float(totals[index]) *
                (input_scales[row] * weight_scales[column]) + bias[column];
    }
}
struct PackedGeometry { uint rows; uint width; uint columns; float input_scale; float output_scale; };
struct GateUpGeometry {
    uint rows;
    uint width;
    uint columns;
    float input_scale;
    float output_scale;
    float hidden_scale;
};
constant bool CALIBRATE_OUTPUT [[function_constant(2)]];
inline float calibrate(float value, float scale) {
    return scale > 0 ? clamp(rint(value / scale), -128.0f, 127.0f) * scale : value;
}
inline float gelu_approx(float value) {
    if (value > 10.0f) return value;
    if (value < -10.0f) return 0.0f;
    return 0.5f * value *
           (1.0f + tanh(clamp(0.7978845608028654f * (value + 0.044715f * value * value * value), -10.0f, 10.0f)));
}
constant uint WEIGHT_BITS [[function_constant(0)]];
constant uint WEIGHT_GROUP [[function_constant(1)]];
inline float unpack_weight(device const uchar* weights, device const float* scales,
                           uint column, uint channel, constant PackedGeometry& geometry) {
    uint per_byte = 8 / WEIGHT_BITS;
    uint raw = (weights[column * (geometry.width / per_byte) + channel / per_byte] >>
                ((channel % per_byte) * WEIGHT_BITS)) & ((1u << WEIGHT_BITS) - 1);
    uint sign = 1u << (WEIGHT_BITS - 1);
    int value = int(raw ^ sign) - int(sign);
    return float(value) * scales[column * (geometry.width / WEIGHT_GROUP) + channel / WEIGHT_GROUP];
}
kernel void expand_packed_weight(device const uchar* weights [[buffer(0)]], device const float* scales [[buffer(1)]],
                                  device half* output [[buffer(2)]], constant PackedGeometry& geometry [[buffer(3)]],
                                  uint index [[thread_position_in_grid]]) {
    if (index < geometry.width * geometry.columns)
        output[index] = half(unpack_weight(weights, scales, index / geometry.width, index % geometry.width, geometry));
}
kernel void packed_gemv(device const float* input [[buffer(0)]],
                        device const uchar* weights [[buffer(1)]],
                        device const float* scales [[buffer(2)]],
                        device float* output [[buffer(3)]],
                        constant PackedGeometry& geometry [[buffer(4)]],
                        uint2 group [[threadgroup_position_in_grid]],
                        uint2 lane [[thread_position_in_threadgroup]]) {
    uint column = group.x * 4 + lane.y;
    float sum = 0;
    if (column < geometry.columns) {
        if (WEIGHT_BITS == 2 && WEIGHT_GROUP % 16 == 0) {
            device const uint* packed = (device const uint*)(weights + column * (geometry.width / 4));
            for (uint channel = lane.x * 16; channel < geometry.width; channel += 512) {
                uint word = packed[channel / 16];
                float partial = 0;
                for (uint part = 0; part < 4; ++part) {
                    int4 weight = int4(((uint4(word) >> (uint4(0, 2, 4, 6) + part * 8)) & 3u) ^ 2u) - 2;
                    float4 activation = *(device const float4*)(input + group.y * geometry.width + channel + part * 4);
                    partial += dot(activation, float4(weight));
                }
                sum += partial * scales[column * (geometry.width / WEIGHT_GROUP) + channel / WEIGHT_GROUP];
            }
        } else if (WEIGHT_BITS == 4 && WEIGHT_GROUP % 8 == 0) {
            device const uint* packed = (device const uint*)(weights + column * (geometry.width / 2));
            for (uint channel = lane.x * 8; channel < geometry.width; channel += 256) {
                uint word = packed[channel / 8];
                int4 lower = int4(((uint4(word) >> uint4(0, 4, 8, 12)) & 15u) ^ 8u) - 8;
                int4 upper = int4(((uint4(word) >> uint4(16, 20, 24, 28)) & 15u) ^ 8u) - 8;
                float4 first = *(device const float4*)(input + group.y * geometry.width + channel);
                float4 second = *(device const float4*)(input + group.y * geometry.width + channel + 4);
                float scale = scales[column * (geometry.width / WEIGHT_GROUP) + channel / WEIGHT_GROUP];
                sum += (dot(first, float4(lower)) + dot(second, float4(upper))) * scale;
            }
        } else if (WEIGHT_BITS == 8 && WEIGHT_GROUP % 4 == 0) {
            device const char4* packed = (device const char4*)(weights + column * geometry.width);
            for (uint channel = lane.x * 4; channel < geometry.width; channel += 128) {
                float4 activation = *(device const float4*)(input + group.y * geometry.width + channel);
                float scale = scales[column * (geometry.width / WEIGHT_GROUP) + channel / WEIGHT_GROUP];
                sum += dot(activation, float4(packed[channel / 4])) * scale;
            }
        } else {
            for (uint channel = lane.x; channel < geometry.width; channel += 32) {
                float activation = input[group.y * geometry.width + channel];
                sum += activation * unpack_weight(weights, scales, column, channel, geometry);
            }
        }
    }
    sum = simd_sum(sum);
    if (lane.x == 0 && column < geometry.columns) output[group.y * geometry.columns + column] = CALIBRATE_OUTPUT ? calibrate(sum, geometry.output_scale) : sum;
}
kernel void packed_gemm(device const float* input [[buffer(0)]],
                        device const uchar* weights [[buffer(1)]],
                        device const float* scales [[buffer(2)]],
                        device float* output [[buffer(3)]],
                        constant PackedGeometry& geometry [[buffer(4)]],
                        uint2 group [[threadgroup_position_in_grid]],
                        uint thread_index [[thread_index_in_threadgroup]],
                        uint simd_group [[simdgroup_index_in_threadgroup]]) {
    threadgroup half activations[32 * 32];
    threadgroup half matrix[32 * 32];
    threadgroup float result[32 * 32];
    uint row_base = group.y * 32, column_base = group.x * 32;
    uint local_row = (simd_group / 2) * 16, local_column = (simd_group % 2) * 16;
    simdgroup_float8x8 acc00 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 acc01 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 acc10 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 acc11 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (uint base = 0; base < geometry.width; base += 32) {
        for (uint index = thread_index; index < 1024; index += 128) {
            uint row = row_base + index / 32, channel = base + index % 32;
            float activation = row < geometry.rows && channel < geometry.width ? input[row * geometry.width + channel] : 0.0f;
            activations[index] = half(activation);
            channel = base + index / 32;
            uint column = column_base + index % 32;
            matrix[index] = column < geometry.columns && channel < geometry.width ?
                half(unpack_weight(weights, scales, column, channel, geometry)) : half(0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint offset = 0; offset < 32; offset += 8) {
            simdgroup_half8x8 left0, left1, right0, right1;
            simdgroup_load(left0, activations + local_row * 32 + offset, 32);
            simdgroup_load(left1, activations + (local_row + 8) * 32 + offset, 32);
            simdgroup_load(right0, matrix + offset * 32 + local_column, 32);
            simdgroup_load(right1, matrix + offset * 32 + local_column + 8, 32);
            simdgroup_multiply_accumulate(acc00, left0, right0, acc00);
            simdgroup_multiply_accumulate(acc01, left0, right1, acc01);
            simdgroup_multiply_accumulate(acc10, left1, right0, acc10);
            simdgroup_multiply_accumulate(acc11, left1, right1, acc11);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    simdgroup_store(acc00, result + local_row * 32 + local_column, 32);
    simdgroup_store(acc01, result + local_row * 32 + local_column + 8, 32);
    simdgroup_store(acc10, result + (local_row + 8) * 32 + local_column, 32);
    simdgroup_store(acc11, result + (local_row + 8) * 32 + local_column + 8, 32);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint index = thread_index; index < 1024; index += 128) {
        uint row = row_base + index / 32, column = column_base + index % 32;
        if (row < geometry.rows && column < geometry.columns) output[row * geometry.columns + column] =
            CALIBRATE_OUTPUT ? calibrate(result[index], geometry.output_scale) : result[index];
    }
}
kernel void packed_gemm_i8w8(device const char* input [[buffer(0)]], device const char* weights [[buffer(1)]],
                             device const float* scales [[buffer(2)]], device float* output [[buffer(3)]],
                             constant PackedGeometry& geometry [[buffer(4)]],
                             uint2 group [[threadgroup_position_in_grid]],
                             uint thread_index [[thread_index_in_threadgroup]],
                             uint simd_group [[simdgroup_index_in_threadgroup]]) {
    threadgroup half activations[64 * 64];
    threadgroup half matrix[64 * 64];
    threadgroup float result[64 * 64];
    uint row_base = group.y * 64, column_base = group.x * 64;
    uint local_row = (simd_group / 4) * 16, local_column = (simd_group % 4) * 16;
    simdgroup_float8x8 acc00 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 acc01 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 acc10 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 acc11 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (uint base = 0; base < geometry.width; base += 64) {
        for (uint index = thread_index; index < 4096; index += 512) {
            uint row = row_base + index / 64, channel = base + index % 64;
            activations[index] =
                row < geometry.rows && channel < geometry.width ? half(input[row * geometry.width + channel]) : half(0);
            channel = base + index / 64;
            uint column = column_base + index % 64;
            matrix[index] = column < geometry.columns && channel < geometry.width
                                ? half(weights[column * geometry.width + channel])
                                : half(0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint offset = 0; offset < 64; offset += 8) {
            simdgroup_half8x8 left0, left1, right0, right1;
            simdgroup_load(left0, activations + local_row * 64 + offset, 64);
            simdgroup_load(left1, activations + (local_row + 8) * 64 + offset, 64);
            simdgroup_load(right0, matrix + offset * 64 + local_column, 64);
            simdgroup_load(right1, matrix + offset * 64 + local_column + 8, 64);
            simdgroup_multiply_accumulate(acc00, left0, right0, acc00);
            simdgroup_multiply_accumulate(acc01, left0, right1, acc01);
            simdgroup_multiply_accumulate(acc10, left1, right0, acc10);
            simdgroup_multiply_accumulate(acc11, left1, right1, acc11);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    simdgroup_store(acc00, result + local_row * 64 + local_column, 64);
    simdgroup_store(acc01, result + local_row * 64 + local_column + 8, 64);
    simdgroup_store(acc10, result + (local_row + 8) * 64 + local_column, 64);
    simdgroup_store(acc11, result + (local_row + 8) * 64 + local_column + 8, 64);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint index = thread_index; index < 4096; index += 512) {
        uint row = row_base + index / 64, column = column_base + index % 64;
        if (row < geometry.rows && column < geometry.columns) {
            float value = (result[index] * scales[column]) * geometry.input_scale;
            output[row * geometry.columns + column] = calibrate(value, geometry.output_scale);
        }
    }
}
kernel void packed_gate_up_i8w8(device const char* input [[buffer(0)]],
                                device const char* weights [[buffer(1)]],
                                device const float* scales [[buffer(2)]],
                                device char* output [[buffer(3)]],
                                constant GateUpGeometry& geometry [[buffer(4)]],
                                uint2 group [[threadgroup_position_in_grid]],
                                uint thread_index [[thread_index_in_threadgroup]],
                                uint simd_group [[simdgroup_index_in_threadgroup]]) {
    threadgroup uint storage[8 * 1024];
                                threadgroup uchar* bytes = (threadgroup uchar*)storage;
                                threadgroup half* activations = (threadgroup half*)bytes;
                                threadgroup half* gate_matrix = (threadgroup half*)(bytes + 8 * 1024);
                                threadgroup half* up_matrix = (threadgroup half*)(bytes + 16 * 1024);
    uint row_base = group.y * 64, column_base = group.x * 64;
    uint local_row = (simd_group / 4) * 16, local_column = (simd_group % 4) * 16;
    simdgroup_float8x8 gate00 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 gate01 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 gate10 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 gate11 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 up00 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 up01 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 up10 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 up11 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (uint base = 0; base < geometry.width; base += 64) {
        for (uint index = thread_index; index < 4096; index += 512) {
            uint row = row_base + index / 64, channel = base + index % 64;
            activations[index] =
                row < geometry.rows && channel < geometry.width ? half(input[row * geometry.width + channel]) : half(0);
        }
        for (uint index = thread_index; index < 4096; index += 512) {
            uint channel = base + index / 64, column = column_base + index % 64;
            bool valid = column < geometry.columns && channel < geometry.width;
            gate_matrix[index] = valid ? half(weights[column * geometry.width + channel]) : half(0);
            up_matrix[index] =
                valid ? half(weights[(geometry.columns + column) * geometry.width + channel]) : half(0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint offset = 0; offset < 64; offset += 8) {
            simdgroup_half8x8 left0, left1, gate_right0, gate_right1, up_right0, up_right1;
            simdgroup_load(left0, activations + local_row * 64 + offset, 64);
            simdgroup_load(left1, activations + (local_row + 8) * 64 + offset, 64);
            simdgroup_load(gate_right0, gate_matrix + offset * 64 + local_column, 64);
            simdgroup_load(gate_right1, gate_matrix + offset * 64 + local_column + 8, 64);
            simdgroup_load(up_right0, up_matrix + offset * 64 + local_column, 64);
            simdgroup_load(up_right1, up_matrix + offset * 64 + local_column + 8, 64);
            simdgroup_multiply_accumulate(gate00, left0, gate_right0, gate00);
            simdgroup_multiply_accumulate(gate01, left0, gate_right1, gate01);
            simdgroup_multiply_accumulate(gate10, left1, gate_right0, gate10);
            simdgroup_multiply_accumulate(gate11, left1, gate_right1, gate11);
            simdgroup_multiply_accumulate(up00, left0, up_right0, up00);
            simdgroup_multiply_accumulate(up01, left0, up_right1, up01);
            simdgroup_multiply_accumulate(up10, left1, up_right0, up10);
            simdgroup_multiply_accumulate(up11, left1, up_right1, up11);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // Input and weight tiles are dead after the K loop, so their 32 KiB becomes the FP32 result tiles.
    threadgroup float* gate_result = (threadgroup float*)bytes;
    threadgroup float* up_result = (threadgroup float*)(bytes + 16 * 1024);
    simdgroup_store(gate00, gate_result + local_row * 64 + local_column, 64);
    simdgroup_store(gate01, gate_result + local_row * 64 + local_column + 8, 64);
    simdgroup_store(gate10, gate_result + (local_row + 8) * 64 + local_column, 64);
    simdgroup_store(gate11, gate_result + (local_row + 8) * 64 + local_column + 8, 64);
    simdgroup_store(up00, up_result + local_row * 64 + local_column, 64);
    simdgroup_store(up01, up_result + local_row * 64 + local_column + 8, 64);
    simdgroup_store(up10, up_result + (local_row + 8) * 64 + local_column, 64);
    simdgroup_store(up11, up_result + (local_row + 8) * 64 + local_column + 8, 64);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint index = thread_index; index < 4096; index += 512) {
        uint row = row_base + index / 64, column = column_base + index % 64;
        if (row < geometry.rows && column < geometry.columns) {
            float gate = (gate_result[index] * scales[column]) * geometry.input_scale;
            float up = (up_result[index] * scales[geometry.columns + column]) * geometry.input_scale;
            gate = calibrate(gate, geometry.output_scale);
            up = calibrate(up, geometry.output_scale);
            float hidden = gelu_approx(gate) * up;
            output[row * geometry.columns + column] =
                char(clamp(rint(hidden / geometry.hidden_scale), -128.0f, 127.0f));
        }
    }
}
