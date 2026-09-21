#include "kidi/runtime/mps/eager_kernels.h"
#include "kidi/tensor/metal.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include <mutex>

namespace kidi::runtime::mps {
namespace {
constexpr const char* SOURCE = R"metal(
#include <metal_stdlib>
using namespace metal;
struct Geometry { uint count, width, other, heads, mode; float epsilon; };
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
kernel void rms(device const float* input [[buffer(0)]], device const float* scale [[buffer(1)]],
                device const float* residual [[buffer(2)]], device const float* output_scale [[buffer(5)]],
                device float* output [[buffer(3)]], constant Geometry& geometry [[buffer(4)]],
                uint row [[threadgroup_position_in_grid]], uint thread_index [[thread_index_in_threadgroup]],
                uint lane [[thread_index_in_simdgroup]], uint simd_group [[simdgroup_index_in_threadgroup]]) {
    threadgroup float partial[8];
    float square = 0;
    for (uint channel = thread_index; channel < geometry.width; channel += 256) {
        float value = input[row * geometry.width + channel];
        square += value * value;
    }
    square = simd_sum(square);
    if (lane == 0) partial[simd_group] = square;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group == 0) {
        float sum = simd_sum(lane < 8 ? partial[lane] : 0.0f);
        if (lane == 0) partial[0] = rsqrt(sum / float(geometry.width) + geometry.epsilon);
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
)metal";
struct Pipelines {
    id<MTLComputePipelineState> rms, pointwise, round_to_half, greedy_token;
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
        id<MTLLibrary> library = [device newLibraryWithSource:[NSString stringWithUTF8String:SOURCE]
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
        if (!result->rms || !result->pointwise || !result->round_to_half || !result->greedy_token)
            return std::unexpected(Error{ErrorCode::RUNTIME, "create eager Metal pipelines"});
        cached = result;
        return result;
    }
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
auto encode_eager(CommandBatch& batch, Operation operation, float epsilon, TensorInputs inputs, tensor::Tensor& output)
    -> Result<void> {
    auto shared = pipelines();
    if (!shared) return std::unexpected(std::move(shared.error()));
    if (output.numel() > UINT32_MAX)
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "eager Metal output exceeds indexing bounds"});
    const bool rms = operation == Operation::RMS_NORM || operation == Operation::RMS_NORM_RESIDUAL ||
                     operation == Operation::RMS_ROTARY;
    const bool rotary = operation == Operation::ROTARY || operation == Operation::RMS_ROTARY;
    const struct {
        std::uint32_t count, width, other, heads, mode;
        float epsilon;
    } geometry{static_cast<std::uint32_t>(output.numel()),
               static_cast<std::uint32_t>(inputs[0].dimensions() ? inputs[0].size(-1) : 1),
               static_cast<std::uint32_t>(rotary              ? inputs[0].size(1)
                                          : inputs.size() > 1 ? inputs[1].numel()
                                                              : 1),
               static_cast<std::uint32_t>(rotary ? inputs[0].size(2) : 1),
               operation == Operation::GELU_MULTIPLY       ? 9u
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
} // namespace kidi::runtime::mps