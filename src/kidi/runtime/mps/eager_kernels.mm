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
        if (geometry.mode >= 5) value = residual[row * geometry.width + channel] + value;
        if (geometry.mode == 6) value *= output_scale[0];
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
    else if (geometry.mode == 2) {
        if (value > 10.0f) output[index] = value;
        else if (value < -10.0f) output[index] = 0.0f;
        else output[index] = 0.5f * value * (1.0f + tanh(clamp(0.7978845608028654f * (value + 0.044715f * value * value * value), -10.0f, 10.0f)));
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
    id<MTLComputePipelineState> rms, pointwise;
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
        if (!result->rms || !result->pointwise)
            return std::unexpected(Error{ErrorCode::RUNTIME, "create eager Metal pipelines"});
        cached = result;
        return result;
    }
}
} // namespace
auto encode_eager(CommandBatch& batch, Operation operation, float epsilon, TensorInputs inputs, tensor::Tensor& output)
    -> Result<void> {
    auto shared = pipelines();
    if (!shared) return std::unexpected(std::move(shared.error()));
    if (output.numel() > UINT32_MAX)
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "eager Metal output exceeds indexing bounds"});
    const bool rms = operation == Operation::RMS_NORM || operation == Operation::RMS_NORM_RESIDUAL;
    const bool rotary = operation == Operation::ROTARY;
    const struct {
        std::uint32_t count, width, other, heads, mode;
        float epsilon;
    } geometry{static_cast<std::uint32_t>(output.numel()),
               static_cast<std::uint32_t>(inputs[0].dimensions() ? inputs[0].size(-1) : 1),
               static_cast<std::uint32_t>(rotary              ? inputs[0].size(1)
                                          : inputs.size() > 1 ? inputs[1].numel()
                                                              : 1),
               static_cast<std::uint32_t>(rotary ? inputs[0].size(2) : 1),
               operation == Operation::STATIC_ROUND        ? 7u
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
    [encoder setComputePipelineState:rms ? (*shared)->rms : (*shared)->pointwise];
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