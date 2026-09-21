#include "kidi/runtime/mps/quantized_linear.h"
#include "kidi/tensor/metal.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include <algorithm>
#include <limits>
#include <mutex>
#include <map>
#include <tuple>
#include <string>
#include <utility>

namespace kidi::runtime::mps {
namespace {
using tensor::Device;
using tensor::DType;
using tensor::Tensor;

constexpr const char* SOURCE = R"metal(
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
struct PackedGeometry { uint rows; uint width; uint columns; uint bits; uint group_size; float input_scale; float output_scale; };
constant bool CALIBRATE_INPUT [[function_constant(2)]];
constant bool CALIBRATE_OUTPUT [[function_constant(3)]];
inline float calibrate(float value, float scale) {
    return scale > 0 ? clamp(rint(value / scale), -128.0f, 127.0f) * scale : value;
}
inline float4 calibrate(float4 value, float scale) {
    return scale > 0 ? clamp(rint(value / scale), -128.0f, 127.0f) * scale : value;
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
                    if (CALIBRATE_INPUT) activation = calibrate(activation, geometry.input_scale);
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
                if (CALIBRATE_INPUT) { first = calibrate(first, geometry.input_scale); second = calibrate(second, geometry.input_scale); }
                float scale = scales[column * (geometry.width / WEIGHT_GROUP) + channel / WEIGHT_GROUP];
                sum += (dot(first, float4(lower)) + dot(second, float4(upper))) * scale;
            }
        } else if (WEIGHT_BITS == 8 && WEIGHT_GROUP % 4 == 0) {
            device const char4* packed = (device const char4*)(weights + column * geometry.width);
            for (uint channel = lane.x * 4; channel < geometry.width; channel += 128) {
                float4 activation = *(device const float4*)(input + group.y * geometry.width + channel);
                if (CALIBRATE_INPUT) activation = calibrate(activation, geometry.input_scale);
                float scale = scales[column * (geometry.width / WEIGHT_GROUP) + channel / WEIGHT_GROUP];
                sum += dot(activation, float4(packed[channel / 4])) * scale;
            }
        } else {
            for (uint channel = lane.x; channel < geometry.width; channel += 32) {
                float activation = input[group.y * geometry.width + channel];
                if (CALIBRATE_INPUT) activation = calibrate(activation, geometry.input_scale);
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
            activations[index] = half(CALIBRATE_INPUT ? calibrate(activation, geometry.input_scale) : activation);
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
        if (row < geometry.rows && column < geometry.columns) output[row * geometry.columns + column] = CALIBRATE_OUTPUT ? calibrate(result[index], geometry.output_scale) : result[index];
    }
}
)metal";

struct Pipelines {
    id<MTLDevice> device;
    id<MTLComputePipelineState> quantize;
    id<MTLComputePipelineState> linear_packed;
    id<MTLComputePipelineState> linear_tiled;
    id<MTLLibrary> library;
};

auto pipelines() -> Result<std::shared_ptr<Pipelines>> {
    static std::mutex mutex;
    static std::shared_ptr<Pipelines> cached;
    std::scoped_lock lock(mutex);
    if (cached) return cached;
    @autoreleasepool {
        auto result = std::make_shared<Pipelines>();
        result->device = MTLCreateSystemDefaultDevice();
        if (!result->device) return std::unexpected(Error{ErrorCode::UNSUPPORTED, "no Metal device"});
        MTLCompileOptions* options = [MTLCompileOptions new];
        options.mathMode = MTLMathModeSafe;
        NSError* error = nil;
        id<MTLLibrary> library = [result->device newLibraryWithSource:[NSString stringWithUTF8String:SOURCE]
                                                              options:options
                                                                error:&error];
        if (!library)
            return std::unexpected(Error{ErrorCode::RUNTIME, "compile INT8 Metal kernels: " +
                                                                 std::string(error.localizedDescription.UTF8String)});
        result->quantize =
            [result->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"quantize_rows"]
                                                          error:&error];
        result->linear_tiled =
            [result->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"linear_a8w8_tiled"]
                                                          error:&error];
        result->linear_packed =
            [result->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"linear_a8w8_packed"]
                                                          error:&error];
        result->library = library;
        if (!result->quantize || !result->linear_tiled || !result->linear_packed)
            return std::unexpected(Error{ErrorCode::RUNTIME, "create INT8 Metal pipelines"});
        cached = result;
        return result;
    }
}
struct PackedPipelines {
    id<MTLComputePipelineState> gemv, gemm;
};
auto packed_pipelines(std::uint32_t bits, std::uint32_t group_size, bool input_scale, bool output_scale)
    -> Result<std::shared_ptr<PackedPipelines>> {
    static std::mutex mutex;
    static std::map<std::tuple<std::uint32_t, std::uint32_t, bool, bool>, std::shared_ptr<PackedPipelines>> cache;
    std::scoped_lock lock(mutex);
    const auto key = std::tuple{bits, group_size, input_scale, output_scale};
    if (const auto found = cache.find(key); found != cache.end()) return found->second;
    auto shared = pipelines();
    if (!shared) return std::unexpected(std::move(shared.error()));
    MTLFunctionConstantValues* constants = [MTLFunctionConstantValues new];
    [constants setConstantValue:&bits type:MTLDataTypeUInt atIndex:0];
    [constants setConstantValue:&group_size type:MTLDataTypeUInt atIndex:1];
    [constants setConstantValue:&input_scale type:MTLDataTypeBool atIndex:2];
    [constants setConstantValue:&output_scale type:MTLDataTypeBool atIndex:3];
    NSError* error = nil;
    auto result = std::make_shared<PackedPipelines>();
    id<MTLFunction> gemv = [(*shared)->library newFunctionWithName:@"packed_gemv"
                                                    constantValues:constants
                                                             error:&error];
    id<MTLFunction> gemm = [(*shared)->library newFunctionWithName:@"packed_gemm"
                                                    constantValues:constants
                                                             error:&error];
    if (!gemv || !gemm) return std::unexpected(Error{ErrorCode::RUNTIME, "specialize packed Metal functions"});
    result->gemv = [(*shared)->device newComputePipelineStateWithFunction:gemv error:&error];
    result->gemm = [(*shared)->device newComputePipelineStateWithFunction:gemm error:&error];
    if (!result->gemv || !result->gemm)
        return std::unexpected(Error{ErrorCode::RUNTIME, "compile packed Metal pipelines"});
    cache.emplace(key, result);
    return result;
}
auto bind(id<MTLComputeCommandEncoder> encoder, const Tensor& tensor, NSUInteger index) -> Result<void> {
    auto buffer = tensor::metal_buffer(tensor);
    if (!buffer) return std::unexpected(std::move(buffer.error()));
    [encoder setBuffer:(__bridge id<MTLBuffer>)buffer->handle offset:buffer->offset_bytes atIndex:index];
    return {};
}
} // namespace

auto encode_expand_packed_weight(CommandBatch& batch, const Tensor& weight, const Tensor& scales, Tensor& output,
                                 std::int32_t bits, std::int32_t group_size) -> Result<void> {
    if (output.dtype() != DType::F16 || output.dimensions() != 2 || output.numel() > UINT32_MAX ||
        (bits != 2 && bits != 4 && bits != 8) || group_size <= 0 || output.size(1) % group_size ||
        weight.dtype() != DType::U8 || weight.dimensions() != 2 || scales.dtype() != DType::F32 ||
        scales.dimensions() != 2 || weight.size(0) != output.size(0) || weight.size(1) * (8 / bits) != output.size(1) ||
        scales.size(0) != output.size(0) || scales.size(1) != output.size(1) / group_size)
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "invalid streamed packed weight geometry"});
    static std::mutex mutex;
    static std::map<std::pair<std::uint32_t, std::uint32_t>, id<MTLComputePipelineState>> cached;
    id<MTLComputePipelineState> pipeline;
    {
        std::scoped_lock lock(mutex);
        const auto key = std::pair<std::uint32_t, std::uint32_t>{bits, group_size};
        if (const auto found = cached.find(key); found != cached.end())
            pipeline = found->second;
        else {
            auto shared = pipelines();
            if (!shared) return std::unexpected(std::move(shared.error()));
            MTLFunctionConstantValues* constants = [MTLFunctionConstantValues new];
            [constants setConstantValue:&key.first type:MTLDataTypeUInt atIndex:0];
            [constants setConstantValue:&key.second type:MTLDataTypeUInt atIndex:1];
            NSError* error = nil;
            auto function = [(*shared)->library newFunctionWithName:@"expand_packed_weight"
                                                     constantValues:constants
                                                              error:&error];
            pipeline = function ? [(*shared)->device newComputePipelineStateWithFunction:function error:&error] : nil;
            if (!pipeline)
                return std::unexpected(Error{ErrorCode::RUNTIME, "compile streamed packed weight expansion"});
            cached.emplace(key, pipeline);
        }
    }
    const struct {
        std::uint32_t rows, width, columns, bits, group_size;
        float input_scale, output_scale;
    } geometry{1,
               static_cast<std::uint32_t>(output.size(1)),
               static_cast<std::uint32_t>(output.size(0)),
               static_cast<std::uint32_t>(bits),
               static_cast<std::uint32_t>(group_size),
               0,
               0};
    MPSCommandBuffer* buffer = (__bridge MPSCommandBuffer*)batch.native_handle();
    auto encoder = [buffer computeCommandEncoder];
    if (!encoder) return std::unexpected(Error{ErrorCode::RUNTIME, "create streamed packed weight encoder"});
    [encoder setComputePipelineState:pipeline];
    auto result = bind(encoder, weight, 0);
    if (result) result = bind(encoder, scales, 1);
    if (result) result = bind(encoder, output, 2);
    [encoder setBytes:&geometry length:sizeof(geometry) atIndex:3];
    if (result)
        [encoder dispatchThreads:MTLSizeMake(output.numel(), 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [encoder endEncoding];
    return result;
}

auto encode_packed_linear(CommandBatch& batch, const Tensor& input, const Tensor& weight, const Tensor& scales,
                          Tensor& output, std::int32_t bits, std::int32_t group_size, float input_scale,
                          float output_scale, bool vector_projection) -> Result<void> {
    auto shared = packed_pipelines(bits, group_size, input_scale > 0, output_scale > 0);
    if (!shared) return std::unexpected(std::move(shared.error()));
    if (input.numel() > UINT32_MAX || output.numel() > UINT32_MAX || weight.numel() > UINT32_MAX)
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "packed Metal linear exceeds indexing bounds"});
    const struct {
        std::uint32_t rows, width, columns, bits, group_size;
        float input_scale, output_scale;
    } geometry{static_cast<std::uint32_t>(input.numel() / input.size(-1)),
               static_cast<std::uint32_t>(input.size(-1)),
               static_cast<std::uint32_t>(weight.size(0)),
               static_cast<std::uint32_t>(bits),
               static_cast<std::uint32_t>(group_size),
               input_scale,
               output_scale};
    MPSCommandBuffer* buffer = (__bridge MPSCommandBuffer*)batch.native_handle();
    id<MTLComputeCommandEncoder> encoder = [buffer computeCommandEncoder];
    if (!encoder) return std::unexpected(Error{ErrorCode::RUNTIME, "create packed projection encoder"});
    const bool tiled = geometry.rows >= 4 && !vector_projection;
    const auto columns_per_group = tiled ? 32u : 4u;
    [encoder setComputePipelineState:tiled ? (*shared)->gemm : (*shared)->gemv];
    auto status = bind(encoder, input, 0);
    if (status) status = bind(encoder, weight, 1);
    if (status) status = bind(encoder, scales, 2);
    if (status) status = bind(encoder, output, 3);
    [encoder setBytes:&geometry length:sizeof(geometry) atIndex:4];
    if (status)
        [encoder dispatchThreadgroups:MTLSizeMake((geometry.columns + columns_per_group - 1) / columns_per_group,
                                                  tiled ? (geometry.rows + 31) / 32 : geometry.rows, 1)
                threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
    [encoder endEncoding];
    return status;
}

struct QuantizedLinear::Impl {
    std::shared_ptr<Pipelines> pipelines;
    Tensor weights, packed_weights, scales, bias, quantized, activation_scales, zeros;
    std::size_t width, columns, rows = 0;
};
QuantizedLinear::QuantizedLinear(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
QuantizedLinear::QuantizedLinear(QuantizedLinear&&) noexcept = default;
auto QuantizedLinear::operator=(QuantizedLinear&&) noexcept -> QuantizedLinear& = default;
QuantizedLinear::~QuantizedLinear() = default;
auto QuantizedLinear::create(const Tensor& weights, const Tensor& scales, const Tensor& bias)
    -> Result<QuantizedLinear> {
    if (!weights.defined() || weights.dtype() != DType::I8 || weights.dimensions() != 2 || weights.size(0) == 0 ||
        weights.size(0) > 65536 || weights.size(1) == 0 || weights.numel() > UINT32_MAX || !weights.is_contiguous() ||
        !scales.defined() || scales.dtype() != DType::F32 || scales.dimensions() != 2 ||
        scales.size(0) != weights.size(1) || scales.size(1) != 1 || !scales.is_contiguous() || !bias.defined() ||
        bias.dtype() != DType::F32 || bias.dimensions() != 1 || bias.size(0) != weights.size(1) ||
        !bias.is_contiguous())
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "invalid Metal quantized linear parameters"});
    auto shared = pipelines();
    if (!shared) return std::unexpected(std::move(shared.error()));
    auto impl = std::make_unique<Impl>();
    impl->pipelines = *shared;
    impl->width = weights.size(0);
    impl->columns = weights.size(1);
    auto device_weights = weights.to(Device::apple_gpu());
    auto device_scales = scales.to(Device::apple_gpu());
    auto device_bias = bias.to(Device::apple_gpu());
    if (!device_weights) return std::unexpected(std::move(device_weights.error()));
    if (!device_scales) return std::unexpected(std::move(device_scales.error()));
    if (!device_bias) return std::unexpected(std::move(device_bias.error()));
    impl->weights = std::move(*device_weights);
    impl->scales = std::move(*device_scales);
    impl->bias = std::move(*device_bias);
    return QuantizedLinear(std::move(impl));
}
auto QuantizedLinear::prepare(const Tensor& input) -> Result<void> {
    if (!input.defined() || input.device() != Device::apple_gpu() || input.dtype() != DType::F32 ||
        input.dimensions() < 2 || input.size(-1) != impl_->width || !input.is_contiguous() || input.numel() == 0 ||
        input.numel() > UINT32_MAX)
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "invalid Metal quantized linear input"});
    auto rows = input.numel() / impl_->width;
    if (rows > UINT32_MAX / impl_->columns)
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "Metal quantized linear output too large"});
    if (impl_->rows == rows) return {};
    if (rows < 4 && !impl_->packed_weights.defined()) {
        const auto stride = (impl_->width + 3) & ~std::size_t{3};
        if (impl_->columns > UINT32_MAX / stride)
            return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "packed Metal weights exceed indexing bounds"});
        auto packed = Tensor::zeros({static_cast<std::int64_t>(impl_->columns), static_cast<std::int64_t>(stride)},
                                    DType::I8, Device::apple_gpu());
        if (!packed) return std::unexpected(std::move(packed.error()));
        auto destination = packed->data<std::int8_t>();
        auto source = std::as_const(impl_->weights).data<std::int8_t>();
        if (!source) return std::unexpected(std::move(source.error()));
        if (!destination) return std::unexpected(std::move(destination.error()));
        for (std::size_t column_base = 0; column_base < impl_->columns; column_base += 32)
            for (std::size_t channel_base = 0; channel_base < impl_->width; channel_base += 32)
                for (std::size_t column = column_base; column < std::min(column_base + 32, impl_->columns); ++column)
                    for (std::size_t channel = channel_base; channel < std::min(channel_base + 32, impl_->width);
                         ++channel)
                        (*destination)[column * stride + channel] = (*source)[channel * impl_->columns + column];
        impl_->packed_weights = std::move(*packed);
    }
    auto quantized = Tensor::empty({static_cast<std::int64_t>(rows), static_cast<std::int64_t>(impl_->width)},
                                   DType::I8, Device::apple_gpu());
    auto scales = Tensor::empty({static_cast<std::int64_t>(rows)}, DType::F32, Device::apple_gpu());
    auto zeros = Tensor::empty({static_cast<std::int64_t>(rows)}, DType::I32, Device::apple_gpu());
    if (!quantized) return std::unexpected(std::move(quantized.error()));
    if (!scales) return std::unexpected(std::move(scales.error()));
    if (!zeros) return std::unexpected(std::move(zeros.error()));
    impl_->quantized = std::move(*quantized);
    impl_->activation_scales = std::move(*scales);
    impl_->zeros = std::move(*zeros);
    impl_->rows = rows;
    return {};
}
auto QuantizedLinear::encode(CommandBatch& batch, const Tensor& input, Tensor& output) -> Result<void> {
    auto status = prepare(input);
    if (!status) return status;
    auto expected_shape = std::vector<std::int64_t>(input.shape().begin(), input.shape().end());
    expected_shape.back() = static_cast<std::int64_t>(impl_->columns);
    if (!output.defined() || output.device() != Device::apple_gpu() || output.dtype() != DType::F32 ||
        !output.is_contiguous() || !std::ranges::equal(output.shape(), expected_shape))
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "invalid Metal quantized linear output"});
    MPSCommandBuffer* buffer = (__bridge MPSCommandBuffer*)batch.native_handle();
    const struct {
        std::uint32_t rows, width, columns;
    } geometry = {static_cast<std::uint32_t>(impl_->rows), static_cast<std::uint32_t>(impl_->width),
                  static_cast<std::uint32_t>(impl_->columns)};
    id<MTLComputeCommandEncoder> quantize = [buffer computeCommandEncoder];
    if (!quantize) return std::unexpected(Error{ErrorCode::RUNTIME, "create quantization encoder"});
    [quantize setComputePipelineState:impl_->pipelines->quantize];
    status = bind(quantize, input, 0);
    if (status) status = bind(quantize, impl_->quantized, 1);
    if (status) status = bind(quantize, impl_->activation_scales, 2);
    if (status) status = bind(quantize, impl_->zeros, 3);
    [quantize setBytes:&geometry length:sizeof(geometry) atIndex:4];
    if (status)
        [quantize dispatchThreadgroups:MTLSizeMake(impl_->rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    if (!status) {
        [quantize endEncoding];
        return status;
    }
    [quantize memoryBarrierWithScope:MTLBarrierScopeBuffers];
    id<MTLComputeCommandEncoder> linear = quantize;
    const bool tiled = impl_->rows >= 4;
    [linear setComputePipelineState:tiled ? impl_->pipelines->linear_tiled : impl_->pipelines->linear_packed];
    status = bind(linear, impl_->quantized, 0);
    if (status) status = bind(linear, tiled ? impl_->weights : impl_->packed_weights, 1);
    if (status) status = bind(linear, impl_->activation_scales, 2);
    if (status) status = bind(linear, impl_->zeros, 3);
    if (status) status = bind(linear, impl_->scales, 4);
    if (status) status = bind(linear, impl_->bias, 5);
    if (status) status = bind(linear, output, 6);
    [linear setBytes:&geometry length:sizeof(geometry) atIndex:7];
    if (status)
        [linear dispatchThreadgroups:MTLSizeMake(tiled ? (impl_->columns + 31) / 32 : (impl_->columns + 3) / 4,
                                                 tiled ? (impl_->rows + 31) / 32 : impl_->rows, 1)
               threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
    [linear endEncoding];
    return status;
}
auto QuantizedLinear::run(const Tensor& input, Tensor& output) -> Result<void> {
    auto batch = CommandBatch::create();
    if (!batch) return std::unexpected(std::move(batch.error()));
    auto status = encode(*batch, input, output);
    if (!status) return status;
    return batch->finish();
}
} // namespace kidi::runtime::mps