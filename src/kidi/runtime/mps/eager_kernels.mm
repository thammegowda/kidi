#include "kidi/runtime/mps/eager_kernels.h"
#include "kidi/runtime/mps/quantized_linear.h"
#include "kidi/tensor/metal.h"
#include "eager_kernels_metal_source.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include <bit>
#include <mutex>

namespace kidi::runtime::mps {
namespace {
struct Pipelines {
    id<MTLComputePipelineState> rms, pointwise, round_to_half, greedy_token, quantize_int8;
    id<MTLComputePipelineState> audio_unfold, snake_audio, conv_transpose_fold;
    id<MTLComputePipelineState> route_experts, residual_mix, short_convolution, l2_clip, relative_attention;
    id<MTLComputePipelineState> blend_prologue, blend_epilogue, slice_f32, concat_f32, cast_values;
    id<MTLComputePipelineState> quantize_cache_i8;
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
        result->audio_unfold = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"audio_unfold"]
                                                                     error:&error];
        result->snake_audio = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"snake_audio"]
                                                                    error:&error];
        result->conv_transpose_fold =
            [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"conv_transpose_fold"]
                                                  error:&error];
        const auto create = [&](NSString* name) {
            return [device newComputePipelineStateWithFunction:[library newFunctionWithName:name] error:&error];
        };
        result->route_experts = create(@"route_experts");
        result->residual_mix = create(@"residual_mix");
        result->short_convolution = create(@"short_convolution");
        result->l2_clip = create(@"l2_clip");
        result->relative_attention = create(@"relative_attention");
        result->blend_prologue = create(@"blend_prologue");
        result->blend_epilogue = create(@"blend_epilogue");
        result->slice_f32 = create(@"slice_f32");
        result->concat_f32 = create(@"concat_f32");
        result->cast_values = create(@"cast_values");
        result->quantize_cache_i8 = create(@"quantize_cache_i8");
        if (!result->rms || !result->pointwise || !result->round_to_half || !result->greedy_token ||
            !result->quantize_int8 || !result->audio_unfold || !result->snake_audio || !result->conv_transpose_fold ||
            !result->route_experts || !result->residual_mix || !result->short_convolution || !result->l2_clip ||
            !result->relative_attention || !result->blend_prologue || !result->blend_epilogue || !result->slice_f32 ||
            !result->concat_f32 || !result->cast_values || !result->quantize_cache_i8)
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
    auto handle = batch.compute_encoder();
    if (!handle) return std::unexpected(std::move(handle.error()));
    id<MTLComputeCommandEncoder> encoder = (__bridge id<MTLComputeCommandEncoder>)*handle;
    [encoder setComputePipelineState:pipeline];
    auto status = bind_audio(encoder, first, 0);
    if (status) status = bind_audio(encoder, second, 1);
    if (status) status = bind_audio(encoder, output, 2);
    [encoder setBytes:&geometry length:sizeof(geometry) atIndex:3];
    if (status)
        [encoder dispatchThreads:MTLSizeMake(geometry.count, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(std::min<std::uint32_t>(256, geometry.count), 1, 1)];
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
    auto handle = batch.compute_encoder();
    if (!handle) return std::unexpected(std::move(handle.error()));
    id<MTLComputeCommandEncoder> encoder = (__bridge id<MTLComputeCommandEncoder>)*handle;
    for (auto stage : {0u, 1u}) {
        geometry.final = stage;
        [encoder setComputePipelineState:(*shared)->greedy_token];
        [encoder setBuffer:(__bridge id<MTLBuffer>)source->handle offset:source->offset_bytes atIndex:0];
        [encoder setBuffer:(__bridge id<MTLBuffer>)temporary->handle offset:temporary->offset_bytes atIndex:1];
        [encoder setBuffer:(__bridge id<MTLBuffer>)target->handle offset:target->offset_bytes atIndex:2];
        [encoder setBytes:&geometry length:sizeof(geometry) atIndex:3];
        [encoder dispatchThreadgroups:MTLSizeMake(stage ? 1 : geometry.parts, output.numel(), 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
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
    auto handle = batch.compute_encoder();
    if (!handle) return std::unexpected(std::move(handle.error()));
    id<MTLComputeCommandEncoder> encoder = (__bridge id<MTLComputeCommandEncoder>)*handle;
    [encoder setComputePipelineState:(*shared)->quantize_int8];
    [encoder setBuffer:(__bridge id<MTLBuffer>)source->handle offset:source->offset_bytes atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)target->handle offset:target->offset_bytes atIndex:1];
    [encoder setBytes:&geometry length:sizeof(geometry) atIndex:2];
    [encoder dispatchThreads:MTLSizeMake(input.numel(), 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    return {};
}
auto encode_eager(CommandBatch& batch, Operation operation, float epsilon, TensorInputs inputs, tensor::Tensor& output,
                  std::span<const std::int64_t> attributes) -> Result<void> {
    auto shared = pipelines();
    if (!shared) return std::unexpected(std::move(shared.error()));
    if (output.numel() > UINT32_MAX)
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "eager Metal output exceeds indexing bounds"});
    auto handle = batch.compute_encoder();
    if (!handle) return std::unexpected(std::move(handle.error()));
    id<MTLComputeCommandEncoder> encoder = (__bridge id<MTLComputeCommandEncoder>)*handle;
    const auto bind_to = [&](id<MTLComputeCommandEncoder> enc, const tensor::Tensor& tensor,
                             NSUInteger index) -> Result<void> {
        auto view = tensor::metal_buffer(tensor);
        if (!view) return std::unexpected(std::move(view.error()));
        [enc setBuffer:(__bridge id<MTLBuffer>)view->handle offset:view->offset_bytes atIndex:index];
        return {};
    };
    if (operation == Operation::SLICE) {
        const auto axis = static_cast<std::size_t>(attributes[0]);
        const auto start = static_cast<std::uint32_t>(attributes[1]);
        const auto length = static_cast<std::uint32_t>(attributes[2]);
        std::uint32_t outer = 1, inner = 1;
        for (std::size_t d = 0; d < axis; ++d) outer *= static_cast<std::uint32_t>(inputs[0].size(d));
        for (std::size_t d = axis + 1; d < inputs[0].dimensions(); ++d)
            inner *= static_cast<std::uint32_t>(inputs[0].size(d));
        const struct {
            std::uint32_t count, outer, src_extent, dst_extent, start, inner;
        } geometry{static_cast<std::uint32_t>(output.numel()),
                   outer,
                   static_cast<std::uint32_t>(inputs[0].size(axis)),
                   length,
                   start,
                   inner};
        [encoder setComputePipelineState:(*shared)->slice_f32];
        auto status = bind_to(encoder, inputs[0], 0);
        if (status) status = bind_to(encoder, output, 1);
        [encoder setBytes:&geometry length:sizeof(geometry) atIndex:2];
        if (status)
            [encoder dispatchThreads:MTLSizeMake(geometry.count, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        return status;
    }
    if (operation == Operation::CONCAT) {
        const auto axis = static_cast<std::size_t>(attributes[0]);
        std::uint32_t outer = 1, inner = 1;
        for (std::size_t d = 0; d < axis; ++d) outer *= static_cast<std::uint32_t>(inputs[0].size(d));
        for (std::size_t d = axis + 1; d < inputs[0].dimensions(); ++d)
            inner *= static_cast<std::uint32_t>(inputs[0].size(d));
        struct {
            std::uint32_t count, inputs, inner, out_extent;
            std::uint32_t extents[4];
            std::uint32_t offsets[4];
        } geometry{static_cast<std::uint32_t>(output.numel()),
                   static_cast<std::uint32_t>(inputs.size()),
                   inner,
                   static_cast<std::uint32_t>(output.size(axis)),
                   {},
                   {}};
        std::uint32_t offset = 0;
        for (std::size_t i = 0; i < inputs.size(); ++i) {
            geometry.extents[i] = static_cast<std::uint32_t>(inputs[i].size(axis));
            geometry.offsets[i] = offset;
            offset += geometry.extents[i];
        }
        [encoder setComputePipelineState:(*shared)->concat_f32];
        Result<void> status;
        for (std::size_t i = 0; i < 4 && status; ++i) status = bind_to(encoder, inputs[i < inputs.size() ? i : 0], i);
        if (status) status = bind_to(encoder, output, 4);
        [encoder setBytes:&geometry length:sizeof(geometry) atIndex:5];
        if (status)
            [encoder dispatchThreads:MTLSizeMake(geometry.count, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        return status;
    }
    if (operation == Operation::CAST) {
        using tensor::DType;
        const std::uint32_t mode = inputs[0].dtype() == DType::BF16 && output.dtype() == DType::F32   ? 0u
                                   : inputs[0].dtype() == DType::F32 && output.dtype() == DType::BF16 ? 1u
                                   : inputs[0].dtype() == DType::F16 && output.dtype() == DType::F32  ? 2u
                                                                                                      : 3u;
        const struct {
            std::uint32_t count, mode;
        } geometry{static_cast<std::uint32_t>(output.numel()), mode};
        [encoder setComputePipelineState:(*shared)->cast_values];
        auto status = bind_to(encoder, inputs[0], 0);
        if (status) status = bind_to(encoder, output, 1);
        [encoder setBytes:&geometry length:sizeof(geometry) atIndex:2];
        if (status)
            [encoder dispatchThreads:MTLSizeMake(geometry.count, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        return status;
    }
    const bool rms = operation == Operation::RMS_NORM || operation == Operation::RMS_NORM_RESIDUAL ||
                     operation == Operation::RMS_ROTARY;
    const bool rotary = operation == Operation::ROTARY || operation == Operation::RMS_ROTARY;
    const bool axial = operation == Operation::RMS_ROTARY && inputs[2].size(0) == 2;
    const bool capped = operation == Operation::SILU_GATE && attributes.size() == 2;
    const auto attribute_float = [&](std::size_t index) {
        return capped ? std::bit_cast<float>(static_cast<std::int32_t>(attributes[index])) : 0.F;
    };
    const struct {
        std::uint32_t count, width, other, heads, mode;
        float epsilon, gate_cap, up_cap;
    } geometry{static_cast<std::uint32_t>(output.numel()),
               static_cast<std::uint32_t>(inputs[0].dimensions() ? inputs[0].size(-1) : 1),
               static_cast<std::uint32_t>(rotary              ? inputs[0].size(1)
                                          : inputs.size() > 1 ? inputs[1].numel()
                                                              : 1),
               static_cast<std::uint32_t>(rotary   ? inputs[0].size(2)
                                          : capped ? inputs.size() > 1
                                                   : 1),
               capped                                      ? 13u
               : axial                                     ? 10u
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
               epsilon,
               attribute_float(0),
               attribute_float(1)};
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
    return status;
}

auto encode_fused(CommandBatch& batch, Operation operation, std::span<const std::int64_t> attributes, float epsilon,
                  TensorInputs inputs, tensor::Tensor& output) -> Result<void> {
    using tensor::DType;
    const auto invalid = [](const char* message) {
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, message});
    };
    for (std::size_t index = 0; index < inputs.size(); ++index) {
        const bool kv_cache = operation == Operation::RELATIVE_ATTENTION && (index == 1 || index == 2);
        const auto dt = inputs[index].dtype();
        const bool ok = dt == DType::F32 || dt == DType::I32 || (kv_cache && (dt == DType::BF16 || dt == DType::I8));
        if (!ok) return invalid("fused Metal operands are F32");
    }
    if (output.dtype() != DType::F32 || output.numel() > UINT32_MAX) return invalid("fused Metal output is F32");
    auto shared = pipelines();
    if (!shared) return std::unexpected(std::move(shared.error()));
    const auto as_float = [&](std::size_t index) {
        return std::bit_cast<float>(static_cast<std::int32_t>(attributes[index]));
    };
    const auto u32 = [](auto value) { return static_cast<std::uint32_t>(value); };
    auto handle = batch.compute_encoder();
    if (!handle) return std::unexpected(std::move(handle.error()));
    id<MTLComputeCommandEncoder> encoder = (__bridge id<MTLComputeCommandEncoder>)*handle;
    Result<void> status;
    const auto bind = [&](const tensor::Tensor& tensor, NSUInteger index) {
        if (!status) return;
        auto view = tensor::metal_buffer(tensor);
        if (!view) {
            status = std::unexpected(std::move(view.error()));
            return;
        }
        [encoder setBuffer:(__bridge id<MTLBuffer>)view->handle offset:view->offset_bytes atIndex:index];
    };
    const auto groups = [&](std::size_t x, std::size_t y, std::size_t threads) {
        if (status)
            [encoder dispatchThreadgroups:MTLSizeMake(x, y, 1) threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    };
    switch (operation) {
        case Operation::ROUTE_EXPERTS: {
            const auto experts = inputs[0].size(-1), top = static_cast<std::size_t>(attributes[0]);
            if (experts > 256 || top > 16 || top > experts || inputs[1].numel() != experts)
                return invalid("Metal expert routing supports up to 256 experts and 16 slots");
            const struct {
                std::uint32_t experts, top;
                float scale;
            } geometry{u32(experts), u32(top), as_float(1)};
            [encoder setComputePipelineState:(*shared)->route_experts];
            bind(inputs[0], 0);
            bind(inputs[1], 1);
            bind(output, 2);
            [encoder setBytes:&geometry length:sizeof(geometry) atIndex:3];
            groups(inputs[0].numel() / experts, 1, 32);
            break;
        }
        case Operation::RESIDUAL_MIX: {
            const auto sources = inputs.size() - 1;
            const auto& queries = inputs[sources];
            if (sources > 7 || queries.size(0) > 4)
                return invalid("Metal residual mix supports up to 7 sources and 4 queries");
            const struct {
                std::uint32_t tokens, width, sources, queries;
                float epsilon;
            } geometry{u32(inputs[0].size(0)), u32(queries.size(1)), u32(sources), u32(queries.size(0)), epsilon};
            [encoder setComputePipelineState:(*shared)->residual_mix];
            for (std::size_t index = 0; index < 7; ++index) bind(inputs[index < sources ? index : 0], index);
            bind(queries, 7);
            bind(output, 8);
            [encoder setBytes:&geometry length:sizeof(geometry) atIndex:9];
            groups(geometry.tokens, 1, 256);
            break;
        }
        case Operation::SHORT_CONVOLUTION: {
            // Padded form: {width}; ring form: {width, ring[, flags]} with input, state, weight, slot[, taps][,
            // residual].
            const bool padded = attributes.size() == 1;
            const auto width = attributes[0];
            const auto channels = output.size(1);
            std::uint32_t flags = padded ? 8U : 0U;
            const tensor::Tensor* taps = nullptr;
            const tensor::Tensor* residual = nullptr;
            if (padded) {
                if (inputs.size() == 3) taps = &inputs[2];
            } else if (attributes.size() == 3) {
                flags |= static_cast<std::uint32_t>(attributes[2]) & 7U;
                if (flags & 1U) taps = &inputs[4];
                if (flags & 2U) residual = &inputs[inputs.size() - 1];
            } else if (inputs.size() == 5) {
                taps = &inputs[4];
            }
            if (taps) flags |= 1U;
            const struct {
                std::uint32_t count, channels, width, ring, groups, flags;
                float scale;
            } geometry{u32(output.numel()),
                       u32(channels),
                       u32(width),
                       u32(padded ? 1 : attributes[1]),
                       u32(taps ? taps->size(1) / width : 1),
                       flags,
                       epsilon};
            [encoder setComputePipelineState:(*shared)->short_convolution];
            bind(inputs[0], 0);
            bind(padded ? inputs[0] : inputs[1], 1);
            bind(padded ? inputs[1] : inputs[2], 2);
            bind(padded ? inputs[0] : inputs[3], 3);
            bind(taps ? *taps : inputs[0], 4);
            bind(residual ? *residual : inputs[0], 5);
            bind(output, 6);
            [encoder setBytes:&geometry length:sizeof(geometry) atIndex:7];
            if (status)
                [encoder dispatchThreads:MTLSizeMake(geometry.count, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            break;
        }
        case Operation::L2_CLIP: {
            const auto width = inputs[0].size(-1);
            const struct {
                std::uint32_t rows, width, scales;
                float threshold;
            } geometry{u32(inputs[0].numel() / width), u32(width), u32(inputs.size() > 1 ? inputs[1].numel() : 0),
                       epsilon};
            [encoder setComputePipelineState:(*shared)->l2_clip];
            bind(inputs[0], 0);
            bind(inputs[inputs.size() > 1 ? 1 : 0], 1);
            bind(output, 2);
            [encoder setBytes:&geometry length:sizeof(geometry) atIndex:3];
            groups((geometry.rows + 3) / 4, 1, 128);
            break;
        }
        case Operation::RELATIVE_ATTENTION: {
            const auto heads = attributes[0];
            const auto width = inputs[0].size(1) / heads;
            if (width % 32 || width > 256)
                return invalid("Metal relative attention head width must be a multiple of 32 up to 256");
            auto accelerated = encode_relative_attention_mpp(batch, attributes, inputs, output);
            if (!accelerated) return std::unexpected(std::move(accelerated.error()));
            if (*accelerated) return {};
            const bool has_positions = inputs.size() == 7 || inputs.size() == 9;
            const bool has_scales = inputs.size() == 9;
            const std::uint32_t kv_dtype =
                inputs[1].dtype() == tensor::DType::F32 ? 0u : (inputs[1].dtype() == tensor::DType::BF16 ? 1u : 2u);
            const std::uint32_t scale_stride = has_scales ? u32(inputs[7].size(1)) : 1u;
            const std::uint32_t scale_group = has_scales ? u32(inputs[1].size(1) / inputs[7].size(1)) : 32u;
            const std::uint32_t ring_mode = attributes.size() >= 8 && attributes[7] != 0 ? 1u : 0u;
            const struct {
                std::uint32_t heads, key_heads, start, window, extent, width, key_stride, capacity, has_positions;
                float log_alpha, log_floor;
                std::uint32_t kv_dtype, scale_stride, scale_group, ring_mode;
            } geometry{u32(heads),
                       u32(attributes[1]),
                       u32(attributes[2]),
                       u32(attributes[3]),
                       u32(attributes[4]),
                       u32(width),
                       u32(inputs[1].size(1)),
                       u32(inputs[1].size(0)),
                       u32(has_positions),
                       as_float(5),
                       as_float(6),
                       kv_dtype,
                       scale_stride,
                       scale_group,
                       ring_mode};
            [encoder setComputePipelineState:(*shared)->relative_attention];
            for (std::size_t index = 0; index < 6; ++index) bind(inputs[index], index);
            bind(inputs[has_positions ? 6 : 0], 6);
            bind(output, 7);
            [encoder setBytes:&geometry length:sizeof(geometry) atIndex:8];
            bind(inputs[has_scales ? 7 : 0], 9);
            bind(inputs[has_scales ? 8 : 0], 10);
            groups(inputs[0].size(0), u32(heads), 256);
            break;
        }
        case Operation::BLEND_PROLOGUE:
        case Operation::BLEND_EPILOGUE: {
            const bool prologue = operation == Operation::BLEND_PROLOGUE;
            const struct {
                std::uint32_t width;
                float epsilon;
            } geometry{u32(inputs[0].size(1)), epsilon};
            [encoder setComputePipelineState:prologue ? (*shared)->blend_prologue : (*shared)->blend_epilogue];
            for (std::size_t index = 0; index < inputs.size(); ++index) bind(inputs[index], index);
            bind(output, 4);
            [encoder setBytes:&geometry length:sizeof(geometry) atIndex:5];
            groups(inputs[0].size(0), 1, 256);
            break;
        }
        default:
            return std::unexpected(Error{ErrorCode::UNSUPPORTED, "unsupported fused Metal operation"});
    }
    return status;
}

auto encode_audio_unfold(CommandBatch& batch, const tensor::Tensor& input, const tensor::Tensor* alpha,
                         tensor::Tensor& output, std::int32_t kernel, std::int32_t stride, std::int32_t padding,
                         std::int32_t dilation) -> Result<void> {
    if (input.dtype() != tensor::DType::F32 || input.dimensions() != 3 || input.size(0) != 1 ||
        output.dtype() != tensor::DType::F32 || output.dimensions() != 3 || output.size(0) != 1 || kernel <= 0 ||
        stride <= 0 || padding < 0 || dilation <= 0 ||
        output.size(2) != input.size(2) * static_cast<std::size_t>(kernel) ||
        (alpha && (alpha->dtype() != tensor::DType::F32 || alpha->numel() != input.size(2))))
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "invalid Metal audio unfold"});
    auto shared = pipelines();
    if (!shared) return std::unexpected(std::move(shared.error()));
    const AudioGeometry geometry{
        static_cast<std::uint32_t>(output.numel()),
        static_cast<std::uint32_t>(input.size(1)),
        static_cast<std::uint32_t>(input.size(2)),
        static_cast<std::uint32_t>(output.size(1)),
        0,
        kernel,
        stride,
        padding,
        dilation,
        alpha ? 1U : 0U,
    };
    return encode_audio(batch, (*shared)->audio_unfold, input, alpha ? *alpha : input, output, geometry);
}

auto encode_snake(CommandBatch& batch, const tensor::Tensor& input, const tensor::Tensor& alpha, tensor::Tensor& output)
    -> Result<void> {
    if (input.dtype() != tensor::DType::F32 || input.dimensions() != 3 || alpha.dtype() != tensor::DType::F32 ||
        alpha.numel() != input.size(2) || output.dtype() != tensor::DType::F32 ||
        !std::ranges::equal(input.shape(), output.shape()))
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "invalid Metal Snake activation"});
    auto shared = pipelines();
    if (!shared) return std::unexpected(std::move(shared.error()));
    const AudioGeometry geometry{
        static_cast<std::uint32_t>(output.numel()),
        static_cast<std::uint32_t>(input.size(1)),
        static_cast<std::uint32_t>(input.size(2)),
        static_cast<std::uint32_t>(output.size(1)),
        static_cast<std::uint32_t>(output.size(2)),
        0,
        0,
        0,
        0,
        1,
    };
    return encode_audio(batch, (*shared)->snake_audio, input, alpha, output, geometry);
}

auto encode_conv_transpose_fold(CommandBatch& batch, const tensor::Tensor& projected, const tensor::Tensor& bias,
                                tensor::Tensor& output, std::int32_t kernel, std::int32_t stride, std::int32_t padding,
                                std::int32_t dilation) -> Result<void> {
    if (projected.dtype() != tensor::DType::F32 || projected.dimensions() != 3 || projected.size(0) != 1 ||
        bias.dtype() != tensor::DType::F32 || bias.numel() != output.size(2) || output.dtype() != tensor::DType::F32 ||
        output.dimensions() != 3 || output.size(0) != 1 || projected.size(2) != output.size(2) * kernel ||
        kernel <= 0 || stride <= 0 || padding < 0 || dilation <= 0)
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "invalid Metal transposed convolution fold"});
    auto shared = pipelines();
    if (!shared) return std::unexpected(std::move(shared.error()));
    const AudioGeometry geometry{
        static_cast<std::uint32_t>(output.numel()),
        static_cast<std::uint32_t>(projected.size(1)),
        static_cast<std::uint32_t>(projected.size(2)),
        static_cast<std::uint32_t>(output.size(1)),
        static_cast<std::uint32_t>(output.size(2)),
        kernel,
        stride,
        padding,
        dilation,
        0,
    };
    return encode_audio(batch, (*shared)->conv_transpose_fold, projected, bias, output, geometry);
}

auto encode_quantize_cache_i8(CommandBatch& batch, const tensor::Tensor& input, const tensor::Tensor& positions,
                              tensor::Tensor& cache, tensor::Tensor& scales) -> Result<void> {
    if (input.dtype() != tensor::DType::F32 || positions.dtype() != tensor::DType::I32 ||
        cache.dtype() != tensor::DType::I8 || scales.dtype() != tensor::DType::F32 || cache.dimensions() != 3 ||
        cache.size(0) != 1 || scales.dimensions() != 2 || scales.size(0) != cache.size(1) ||
        cache.size(2) != scales.size(1) * 32)
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "invalid Metal INT8 KV cache update"});
    auto shared = pipelines();
    if (!shared) return std::unexpected(std::move(shared.error()));
    auto handle = batch.compute_encoder();
    if (!handle) return std::unexpected(std::move(handle.error()));
    id<MTLComputeCommandEncoder> encoder = (__bridge id<MTLComputeCommandEncoder>)*handle;
    const struct {
        std::uint32_t tokens, width, groups, capacity;
    } geom{static_cast<std::uint32_t>(positions.size(0)), static_cast<std::uint32_t>(cache.size(2)),
           static_cast<std::uint32_t>(scales.size(1)), static_cast<std::uint32_t>(cache.size(1))};
    [encoder setComputePipelineState:(*shared)->quantize_cache_i8];
    auto status = bind_audio(encoder, input, 0);
    if (status) status = bind_audio(encoder, positions, 1);
    if (status) status = bind_audio(encoder, cache, 2);
    if (status) status = bind_audio(encoder, scales, 3);
    [encoder setBytes:&geom length:sizeof(geom) atIndex:4];
    if (status)
        [encoder dispatchThreadgroups:MTLSizeMake(geom.groups, geom.tokens, 1)
                threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    return status;
}
} // namespace kidi::runtime::mps