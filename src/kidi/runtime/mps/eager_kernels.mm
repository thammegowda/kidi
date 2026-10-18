#include "kidi/runtime/mps/eager_kernels.h"
#include "kidi/tensor/metal.h"
#include "eager_kernels_metal_source.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include <mutex>

namespace kidi::runtime::mps {
namespace {
struct Pipelines {
    id<MTLComputePipelineState> rms, pointwise, round_to_half, greedy_token, quantize_int8;
    id<MTLComputePipelineState> audio_unfold, snake_audio, conv_transpose_fold;
};
auto pipelines() -> Result<std::shared_ptr<Pipelines>> {
    static std::mutex mutex;
    static std::shared_ptr<Pipelines> cached;
    std::scoped_lock lock(mutex);
    if (cached) return cached;
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        MTLCompileOptions* options = [MTLCompileOptions new];
        options.mathMode = MTLMathModeSafe;
        NSError* error = nil;
        id<MTLLibrary> library =
            [device newLibraryWithSource:[NSString stringWithUTF8String:KIDI_EAGER_KERNELS_METAL_SOURCE]
                                 options:options
                                   error:&error];
        if (!library)
            return std::unexpected(Error{ErrorCode::RUNTIME, "compile eager Metal kernels: " +
                                                                 std::string(error.localizedDescription.UTF8String)});
        auto result = std::make_shared<Pipelines>();
        result->rms = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"rms"] error:&error];
        result->pointwise = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"pointwise"]
                                                                  error:&error];
        result->round_to_half =
            [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"round_to_half"] error:&error];
        result->greedy_token = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"greedy_token"]
                                                                     error:&error];
        result->quantize_int8 =
            [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"quantize_int8"] error:&error];
        result->audio_unfold =
            [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"audio_unfold"] error:&error];
        result->snake_audio =
            [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"snake_audio"] error:&error];
        result->conv_transpose_fold = [device
            newComputePipelineStateWithFunction:[library newFunctionWithName:@"conv_transpose_fold"] error:&error];
        if (!result->rms || !result->pointwise || !result->round_to_half || !result->greedy_token ||
            !result->quantize_int8 || !result->audio_unfold || !result->snake_audio ||
            !result->conv_transpose_fold)
            return std::unexpected(Error{ErrorCode::RUNTIME, "create eager Metal pipelines"});
        cached = result;
        return result;
    }
}

struct AudioGeometry {
    std::uint32_t count, input_length, input_channels, output_length, output_channels;
    std::int32_t kernel, stride, padding, dilation;
    std::uint32_t snake;
};

auto bind_audio(id<MTLComputeCommandEncoder> encoder, const tensor::Tensor& tensor, NSUInteger index) -> Result<void> {
    auto view = tensor::metal_buffer(tensor);
    if (!view) return std::unexpected(std::move(view.error()));
    [encoder setBuffer:(__bridge id<MTLBuffer>)view->handle offset:view->offset_bytes atIndex:index];
    return {};
}

auto encode_audio(CommandBatch& batch, id<MTLComputePipelineState> pipeline, const tensor::Tensor& first,
                  const tensor::Tensor& second, tensor::Tensor& output, const AudioGeometry& geometry) -> Result<void> {
    MPSCommandBuffer* buffer = (__bridge MPSCommandBuffer*)batch.native_handle();
    id<MTLComputeCommandEncoder> encoder = [buffer computeCommandEncoder];
    if (!encoder) return std::unexpected(Error{ErrorCode::RUNTIME, "create audio Metal encoder"});
    [encoder setComputePipelineState:pipeline];
    auto status = bind_audio(encoder, first, 0);
    if (status) status = bind_audio(encoder, second, 1);
    if (status) status = bind_audio(encoder, output, 2);
    [encoder setBytes:&geometry length:sizeof(geometry) atIndex:3];
    if (status)
        [encoder dispatchThreads:MTLSizeMake(geometry.count, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(std::min<std::uint32_t>(256, geometry.count), 1, 1)];
    [encoder endEncoding];
    return status;
}
} // namespace
auto encode_greedy_token(CommandBatch& batch, const tensor::Tensor& input, tensor::Tensor& scratch,
                         tensor::Tensor& output) -> Result<void> {
    auto shared = pipelines();
    if (!shared) return std::unexpected(std::move(shared.error()));
    auto source = tensor::metal_buffer(input), temporary = tensor::metal_buffer(scratch),
         target = tensor::metal_buffer(output);
    if (!source) return std::unexpected(std::move(source.error()));
    if (!temporary) return std::unexpected(std::move(temporary.error()));
    if (!target) return std::unexpected(std::move(target.error()));
    struct {
        std::uint32_t width, parts, final;
    } geometry{static_cast<std::uint32_t>(input.size(-1)), static_cast<std::uint32_t>((input.size(-1) + 1023) / 1024),
               0};
    MPSCommandBuffer* buffer = (__bridge MPSCommandBuffer*)batch.native_handle();
    for (auto stage : {0u, 1u}) {
        auto encoder = [buffer computeCommandEncoder];
        if (!encoder) return std::unexpected(Error{ErrorCode::RUNTIME, "create greedy selection encoder"});
        geometry.final = stage;
        [encoder setComputePipelineState:(*shared)->greedy_token];
        [encoder setBuffer:(__bridge id<MTLBuffer>)source->handle offset:source->offset_bytes atIndex:0];
        [encoder setBuffer:(__bridge id<MTLBuffer>)temporary->handle offset:temporary->offset_bytes atIndex:1];
        [encoder setBuffer:(__bridge id<MTLBuffer>)target->handle offset:target->offset_bytes atIndex:2];
        [encoder setBytes:&geometry length:sizeof(geometry) atIndex:3];
        [encoder dispatchThreadgroups:MTLSizeMake(stage ? 1 : geometry.parts, output.numel(), 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [encoder endEncoding];
    }
    return {};
}
auto encode_quantize_int8(CommandBatch& batch, float scale, const tensor::Tensor& input, tensor::Tensor& output)
    -> Result<void> {
    auto shared = pipelines();
    if (!shared) return std::unexpected(std::move(shared.error()));
    auto source = tensor::metal_buffer(input), target = tensor::metal_buffer(output);
    if (!source) return std::unexpected(std::move(source.error()));
    if (!target) return std::unexpected(std::move(target.error()));
    const struct {
        std::uint32_t count;
        float scale;
    } geometry{static_cast<std::uint32_t>(input.numel()), scale};
    MPSCommandBuffer* buffer = (__bridge MPSCommandBuffer*)batch.native_handle();
    id<MTLComputeCommandEncoder> encoder = [buffer computeCommandEncoder];
    if (!encoder) return std::unexpected(Error{ErrorCode::RUNTIME, "create INT8 quantization encoder"});
    [encoder setComputePipelineState:(*shared)->quantize_int8];
    [encoder setBuffer:(__bridge id<MTLBuffer>)source->handle offset:source->offset_bytes atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)target->handle offset:target->offset_bytes atIndex:1];
    [encoder setBytes:&geometry length:sizeof(geometry) atIndex:2];
    [encoder dispatchThreads:MTLSizeMake(input.numel(), 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [encoder endEncoding];
    return {};
}
auto encode_eager(CommandBatch& batch, Operation operation, float epsilon, TensorInputs inputs, tensor::Tensor& output)
    -> Result<void> {
    auto shared = pipelines();
    if (!shared) return std::unexpected(std::move(shared.error()));
    if (output.numel() > UINT32_MAX)
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "eager Metal output exceeds indexing bounds"});
    const bool rms = operation == Operation::RMS_NORM || operation == Operation::RMS_NORM_RESIDUAL ||
                     operation == Operation::RMS_ROTARY;
    const bool rotary = operation == Operation::ROTARY || operation == Operation::RMS_ROTARY;
    const bool axial = operation == Operation::RMS_ROTARY && inputs[2].size(0) == 2;
    const struct {
        std::uint32_t count, width, other, heads, mode;
        float epsilon;
    } geometry{static_cast<std::uint32_t>(output.numel()),
               static_cast<std::uint32_t>(inputs[0].dimensions() ? inputs[0].size(-1) : 1),
               static_cast<std::uint32_t>(rotary              ? inputs[0].size(1)
                                          : inputs.size() > 1 ? inputs[1].numel()
                                                              : 1),
               static_cast<std::uint32_t>(rotary ? inputs[0].size(2) : 1),
               axial                                       ? 10u
               : operation == Operation::SILU_GATE         ? 12u
               : operation == Operation::SILU_MULTIPLY     ? 11u
               : operation == Operation::GELU_MULTIPLY     ? 9u
               : operation == Operation::RMS_ROTARY        ? 8u
               : operation == Operation::STATIC_ROUND      ? 7u
               : operation == Operation::RMS_NORM_RESIDUAL ? (inputs.size() == 4 ? 6u : 5u)
               : operation == Operation::ADD               ? 0u
               : operation == Operation::MULTIPLY          ? 1u
               : operation == Operation::GELU              ? 2u
               : operation == Operation::TANH              ? 3u
                                                           : 4u,
               epsilon};
    MPSCommandBuffer* buffer = (__bridge MPSCommandBuffer*)batch.native_handle();
    id<MTLComputeCommandEncoder> encoder = [buffer computeCommandEncoder];
    if (!encoder) return std::unexpected(Error{ErrorCode::RUNTIME, "create eager kernel encoder"});
    const auto bind = [&](const tensor::Tensor& tensor, NSUInteger index) -> Result<void> {
        auto view = tensor::metal_buffer(tensor);
        if (!view) return std::unexpected(std::move(view.error()));
        [encoder setBuffer:(__bridge id<MTLBuffer>)view->handle offset:view->offset_bytes atIndex:index];
        return {};
    };
    [encoder setComputePipelineState:rms                                    ? (*shared)->rms
                                     : output.dtype() == tensor::DType::F16 ? (*shared)->round_to_half
                                                                            : (*shared)->pointwise];
    Result<void> status;
    for (std::size_t index = 0; index < 3 && status; ++index)
        status = bind(inputs[index < inputs.size() ? index : 0], index);
    if (status) status = bind(output, 3);
    if (status && rms) status = bind(inputs[inputs.size() == 4 ? 3 : 0], 5);
    [encoder setBytes:&geometry length:sizeof(geometry) atIndex:4];
    if (status) {
        if (rms)
            [encoder dispatchThreadgroups:MTLSizeMake(geometry.count / geometry.width, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        else
            [encoder dispatchThreads:MTLSizeMake(geometry.count, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }
    [encoder endEncoding];
    return status;
}

auto encode_audio_unfold(CommandBatch& batch, const tensor::Tensor& input, const tensor::Tensor* alpha,
                         tensor::Tensor& output, std::int32_t kernel, std::int32_t stride, std::int32_t padding,
                         std::int32_t dilation) -> Result<void> {
    if (input.dtype() != tensor::DType::F32 || input.dimensions() != 3 || input.size(0) != 1 ||
        output.dtype() != tensor::DType::F32 || output.dimensions() != 3 || output.size(0) != 1 ||
        kernel <= 0 || stride <= 0 || padding < 0 || dilation <= 0 ||
        output.size(2) != input.size(2) * static_cast<std::size_t>(kernel) ||
        (alpha && (alpha->dtype() != tensor::DType::F32 || alpha->numel() != input.size(2))))
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "invalid Metal audio unfold"});
    auto shared = pipelines();
    if (!shared) return std::unexpected(std::move(shared.error()));
    const AudioGeometry geometry{
        static_cast<std::uint32_t>(output.numel()), static_cast<std::uint32_t>(input.size(1)),
        static_cast<std::uint32_t>(input.size(2)), static_cast<std::uint32_t>(output.size(1)), 0,
        kernel, stride, padding, dilation, alpha ? 1U : 0U,
    };
    return encode_audio(batch, (*shared)->audio_unfold, input, alpha ? *alpha : input, output, geometry);
}

auto encode_snake(CommandBatch& batch, const tensor::Tensor& input, const tensor::Tensor& alpha,
                  tensor::Tensor& output) -> Result<void> {
    if (input.dtype() != tensor::DType::F32 || input.dimensions() != 3 || alpha.dtype() != tensor::DType::F32 ||
        alpha.numel() != input.size(2) || output.dtype() != tensor::DType::F32 ||
        !std::ranges::equal(input.shape(), output.shape()))
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "invalid Metal Snake activation"});
    auto shared = pipelines();
    if (!shared) return std::unexpected(std::move(shared.error()));
    const AudioGeometry geometry{
        static_cast<std::uint32_t>(output.numel()), static_cast<std::uint32_t>(input.size(1)),
        static_cast<std::uint32_t>(input.size(2)), static_cast<std::uint32_t>(output.size(1)),
        static_cast<std::uint32_t>(output.size(2)), 0, 0, 0, 0, 1,
    };
    return encode_audio(batch, (*shared)->snake_audio, input, alpha, output, geometry);
}

auto encode_conv_transpose_fold(CommandBatch& batch, const tensor::Tensor& projected, const tensor::Tensor& bias,
                                tensor::Tensor& output, std::int32_t kernel, std::int32_t stride,
                                std::int32_t padding, std::int32_t dilation) -> Result<void> {
    if (projected.dtype() != tensor::DType::F32 || projected.dimensions() != 3 || projected.size(0) != 1 ||
        bias.dtype() != tensor::DType::F32 || bias.numel() != output.size(2) || output.dtype() != tensor::DType::F32 ||
        output.dimensions() != 3 || output.size(0) != 1 || projected.size(2) != output.size(2) * kernel ||
        kernel <= 0 || stride <= 0 || padding < 0 || dilation <= 0)
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "invalid Metal transposed convolution fold"});
    auto shared = pipelines();
    if (!shared) return std::unexpected(std::move(shared.error()));
    const AudioGeometry geometry{
        static_cast<std::uint32_t>(output.numel()), static_cast<std::uint32_t>(projected.size(1)),
        static_cast<std::uint32_t>(projected.size(2)), static_cast<std::uint32_t>(output.size(1)),
        static_cast<std::uint32_t>(output.size(2)), kernel, stride, padding, dilation, 0,
    };
    return encode_audio(batch, (*shared)->conv_transpose_fold, projected, bias, output, geometry);
}
} // namespace kidi::runtime::mps