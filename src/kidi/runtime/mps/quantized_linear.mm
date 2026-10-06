#include "kidi/runtime/mps/quantized_linear.h"
#include "kidi/tensor/metal.h"
#include "quantized_linear_metal_source.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include <algorithm>
#include <cmath>
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

struct Pipelines {
    id<MTLDevice> device;
    id<MTLComputePipelineState> quantize;
    id<MTLComputePipelineState> linear_packed;
    id<MTLComputePipelineState> linear_tiled;
    id<MTLComputePipelineState> packed_i8w8;
    id<MTLComputePipelineState> packed_gate_up;
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
        id<MTLLibrary> library =
            [result->device newLibraryWithSource:[NSString stringWithUTF8String:KIDI_QUANTIZED_LINEAR_METAL_SOURCE]
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
        result->packed_i8w8 =
            [result->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"packed_gemm_i8w8"]
                                                          error:&error];
        result->packed_gate_up =
            [result->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"packed_gate_up_i8w8"]
                                                          error:&error];
        result->library = library;
        if (!result->quantize || !result->linear_tiled || !result->linear_packed || !result->packed_i8w8 ||
            !result->packed_gate_up)
            return std::unexpected(Error{ErrorCode::RUNTIME, "create INT8 Metal pipelines"});
        cached = result;
        return result;
    }
}
struct PackedPipelines {
    id<MTLComputePipelineState> gemv, gemm;
};
auto packed_pipelines(std::uint32_t bits, std::uint32_t group_size, bool output_scale)
    -> Result<std::shared_ptr<PackedPipelines>> {
    static std::mutex mutex;
    static std::map<std::tuple<std::uint32_t, std::uint32_t, bool>, std::shared_ptr<PackedPipelines>> cache;
    std::scoped_lock lock(mutex);
    const auto key = std::tuple{bits, group_size, output_scale};
    if (const auto found = cache.find(key); found != cache.end()) return found->second;
    auto shared = pipelines();
    if (!shared) return std::unexpected(std::move(shared.error()));
    MTLFunctionConstantValues* constants = [MTLFunctionConstantValues new];
    [constants setConstantValue:&bits type:MTLDataTypeUInt atIndex:0];
    [constants setConstantValue:&group_size type:MTLDataTypeUInt atIndex:1];
    [constants setConstantValue:&output_scale type:MTLDataTypeBool atIndex:2];
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
        std::uint32_t rows, width, columns;
        float input_scale, output_scale;
    } geometry{1,
               static_cast<std::uint32_t>(output.size(1)),
               static_cast<std::uint32_t>(output.size(0)),
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
    const bool quantized_input = input.dtype() == DType::I8;
    if (!input.defined() || input.dimensions() < 2 || !input.size(-1) ||
        (input.dtype() != DType::F32 && !quantized_input) || (!quantized_input && input_scale != 0) ||
        (quantized_input && (bits != 8 || group_size != input.size(-1) || input_scale <= 0 || vector_projection ||
                             input.numel() / input.size(-1) < 4)))
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "invalid packed Metal linear input"});
    if (input.numel() > UINT32_MAX || output.numel() > UINT32_MAX || weight.numel() > UINT32_MAX)
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "packed Metal linear exceeds indexing bounds"});
    id<MTLComputePipelineState> gemv = nil, gemm = nil;
    if (quantized_input) {
        auto shared = pipelines();
        if (!shared) return std::unexpected(std::move(shared.error()));
        gemm = (*shared)->packed_i8w8;
    } else {
        auto shared = packed_pipelines(bits, group_size, output_scale > 0);
        if (!shared) return std::unexpected(std::move(shared.error()));
        gemv = (*shared)->gemv;
        gemm = (*shared)->gemm;
    }
    const struct {
        std::uint32_t rows, width, columns;
        float input_scale, output_scale;
    } geometry{static_cast<std::uint32_t>(input.numel() / input.size(-1)),
               static_cast<std::uint32_t>(input.size(-1)),
               static_cast<std::uint32_t>(weight.size(0)),
               input_scale,
               output_scale};
    MPSCommandBuffer* buffer = (__bridge MPSCommandBuffer*)batch.native_handle();
    id<MTLComputeCommandEncoder> encoder = [buffer computeCommandEncoder];
    if (!encoder) return std::unexpected(Error{ErrorCode::RUNTIME, "create packed projection encoder"});
    const bool tiled = geometry.rows >= 4 && !vector_projection;
    const bool wide_tiled = tiled && quantized_input;
    const auto columns_per_group = wide_tiled ? 64u : tiled ? 32u : 4u;
    [encoder setComputePipelineState:tiled ? gemm : gemv];
    auto status = bind(encoder, input, 0);
    if (status) status = bind(encoder, weight, 1);
    if (status) status = bind(encoder, scales, 2);
    if (status) status = bind(encoder, output, 3);
    [encoder setBytes:&geometry length:sizeof(geometry) atIndex:4];
    if (status)
        [encoder dispatchThreadgroups:MTLSizeMake((geometry.columns + columns_per_group - 1) / columns_per_group,
                                                  wide_tiled ? (geometry.rows + 63) / 64
                                                  : tiled    ? (geometry.rows + 31) / 32
                                                             : geometry.rows,
                                                  1)
                threadsPerThreadgroup:MTLSizeMake(32, wide_tiled ? 16 : 4, 1)];
    [encoder endEncoding];
    return status;
}

auto encode_packed_gate_up(CommandBatch& batch, const Tensor& input, const Tensor& weight, const Tensor& scales,
                           Tensor& output, float input_scale, float output_scale, float hidden_scale) -> Result<void> {
    if (!input.defined() || input.device() != Device::apple_gpu() || input.dtype() != DType::I8 ||
        input.dimensions() < 2 || !input.size(-1) || !input.numel() || input.numel() > UINT32_MAX ||
        !std::isfinite(input_scale) || input_scale <= 0 || !output.defined() ||
        output.device() != Device::apple_gpu() || output.dtype() != DType::I8 || output.numel() > UINT32_MAX ||
        !std::isfinite(hidden_scale) || hidden_scale <= 0)
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "invalid packed Metal gate/up activation"});
    if (!weight.defined() || weight.dtype() != DType::U8 || weight.dimensions() != 2 || !weight.size(0) ||
        weight.size(0) % 2 || scales.dtype() != DType::F32 || scales.dimensions() != 2 ||
        scales.size(0) != weight.size(0) || scales.size(1) != 1 || !std::isfinite(output_scale) || output_scale <= 0)
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "invalid packed Metal gate/up parameters"});
    const auto rows = input.numel() / input.size(-1);
    const auto width = input.size(-1);
    const auto columns = weight.size(0) / 2;
    auto expected_shape = std::vector<std::int64_t>(input.shape().begin(), input.shape().end());
    expected_shape.back() = columns;
    if (!std::ranges::equal(output.shape(), expected_shape))
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "invalid packed Metal gate/up output"});
    if (weight.size(1) != width)
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "invalid packed Metal gate/up parameters"});
    auto shared = pipelines();
    if (!shared) return std::unexpected(std::move(shared.error()));
    const struct {
        std::uint32_t rows, width, columns;
        float input_scale, output_scale, hidden_scale;
    } geometry{static_cast<std::uint32_t>(rows),
               static_cast<std::uint32_t>(width),
               static_cast<std::uint32_t>(columns),
               input_scale,
               output_scale,
               hidden_scale};
    MPSCommandBuffer* buffer = (__bridge MPSCommandBuffer*)batch.native_handle();
    id<MTLComputeCommandEncoder> encoder = [buffer computeCommandEncoder];
    if (!encoder) return std::unexpected(Error{ErrorCode::RUNTIME, "create packed gate/up encoder"});
    [encoder setComputePipelineState:(*shared)->packed_gate_up];
    auto status = bind(encoder, input, 0);
    if (status) status = bind(encoder, weight, 1);
    if (status) status = bind(encoder, scales, 2);
    if (status) status = bind(encoder, output, 3);
    [encoder setBytes:&geometry length:sizeof(geometry) atIndex:4];
    if (status)
        [encoder dispatchThreadgroups:MTLSizeMake((columns + 63) / 64, (rows + 63) / 64, 1)
                threadsPerThreadgroup:MTLSizeMake(32, 16, 1)];
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