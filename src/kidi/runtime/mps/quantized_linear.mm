#include "kidi/runtime/mps/quantized_linear.h"
#include "kidi/tensor/metal.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include <algorithm>
#include <limits>
#include <mutex>
#include <string>

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
)metal";

struct Pipelines {
    id<MTLDevice> device;
    id<MTLComputePipelineState> quantize;
    id<MTLComputePipelineState> linear_packed;
    id<MTLComputePipelineState> linear_tiled;
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
        if (!result->quantize || !result->linear_tiled || !result->linear_packed)
            return std::unexpected(Error{ErrorCode::RUNTIME, "create INT8 Metal pipelines"});
        cached = result;
        return result;
    }
}
auto bind(id<MTLComputeCommandEncoder> encoder, const Tensor& tensor, NSUInteger index) -> Result<void> {
    auto buffer = tensor::metal_buffer(tensor);
    if (!buffer) return std::unexpected(std::move(buffer.error()));
    [encoder setBuffer:(__bridge id<MTLBuffer>)buffer->handle offset:buffer->offset_bytes atIndex:index];
    return {};
}
} // namespace

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