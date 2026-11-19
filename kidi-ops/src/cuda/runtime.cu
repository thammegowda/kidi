#include "cuda/runtime.h"

#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>

namespace kidi::cuda {
namespace {
void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(std::string("CUDA: ") + cudaGetErrorString(status));
}
std::mutex allocations_mutex;
struct Allocation {
    std::size_t bytes;
    int device;
};
std::unordered_map<void*, Allocation> allocations;
std::atomic<std::size_t> live_bytes{0};
std::size_t peak_bytes = 0;
std::unordered_map<int, std::size_t> device_bytes, device_peaks;
thread_local cudaStream_t execution_stream = nullptr;
thread_local const TensorBinding* execution_bindings = nullptr;
thread_local std::size_t execution_binding_count = 0;

class DeviceScope {
public:
    explicit DeviceScope(int device) {
        check(cudaGetDevice(&previous_));
        changed_ = device != previous_;
        if (changed_) check(cudaSetDevice(device));
    }
    ~DeviceScope() {
        if (changed_) cudaSetDevice(previous_);
    }
    DeviceScope(const DeviceScope&) = delete;
    DeviceScope& operator=(const DeviceScope&) = delete;

private:
    int previous_ = 0;
    bool changed_ = false;
};

struct BlasHandle {
    cublasHandle_t handle = nullptr;
    cudaStream_t stream = nullptr;
    int device;
    explicit BlasHandle(int selected_device, cublasMath_t math = CUBLAS_PEDANTIC_MATH) : device(selected_device) {
        DeviceScope scope(device);
        if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("CUDA: cuBLAS initialization failed");
        if (cublasSetMathMode(handle, math) != CUBLAS_STATUS_SUCCESS) {
            cublasDestroy(handle);
            throw std::runtime_error("CUDA: cuBLAS math configuration failed");
        }
    }
    void bind_stream() {
        if (stream == execution_stream) return;
        if (cublasSetStream(handle, execution_stream) != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("CUDA: cuBLAS stream configuration failed");
        stream = execution_stream;
    }
    ~BlasHandle() {
        int previous = 0;
        if (cudaGetDevice(&previous) == cudaSuccess && cudaSetDevice(device) == cudaSuccess) {
            cublasDestroy(handle);
            cudaSetDevice(previous);
        }
    }
};
thread_local std::unordered_map<int, std::unique_ptr<BlasHandle>> blas_handles;
thread_local std::unordered_map<int, std::unique_ptr<BlasHandle>> tensor_blas_handles;

struct View {
    void* data;
    int type;
    int rank;
    std::int64_t shape[8];
    void* const* binding = nullptr;
    std::size_t binding_offset = 0;
};
struct Launch {
    int operation;
    View inputs[8];
    View output;
    int count;
    int attribute_count;
    std::int64_t attributes[32];
    float epsilon;
    std::size_t elements;
};

View view(const Tensor& tensor) {
    View result{};
    result.data = tensor.data;
    result.type = static_cast<int>(tensor.type);
    result.rank = tensor.rank;
    std::copy(tensor.shape.begin(), tensor.shape.end(), result.shape);
    return result;
}

void bind(View& tensor) {
    const auto address = reinterpret_cast<std::uintptr_t>(tensor.data);
    for (std::size_t index = 0; index < execution_binding_count; ++index) {
        const auto& candidate = execution_bindings[index];
        const auto begin = reinterpret_cast<std::uintptr_t>(candidate.data);
        if (address >= begin && address - begin < candidate.bytes) {
            tensor.binding = candidate.slot;
            tensor.binding_offset = address - begin;
            return;
        }
    }
}

__device__ void* address(const View& tensor) {
    return tensor.binding ? static_cast<unsigned char*>(*tensor.binding) + tensor.binding_offset : tensor.data;
}

__device__ float read(const View& tensor, std::size_t index) {
    const auto* data = address(tensor);
    switch (static_cast<Type>(tensor.type)) {
        case Type::BOOL:
        case Type::U8:
            return static_cast<const unsigned char*>(data)[index];
        case Type::I8:
            return static_cast<const signed char*>(data)[index];
        case Type::I32:
            return static_cast<const int*>(data)[index];
        case Type::I64:
            return static_cast<float>(static_cast<const long long*>(data)[index]);
        case Type::F16:
            return __half2float(static_cast<const __half*>(data)[index]);
        case Type::BF16:
            return __bfloat162float(static_cast<const __nv_bfloat16*>(data)[index]);
        case Type::F32:
            return static_cast<const float*>(data)[index];
        case Type::F64:
            return static_cast<float>(static_cast<const double*>(data)[index]);
        default:
            return nanf("");
    }
}
__device__ void write(const View& tensor, std::size_t index, float value) {
    auto* data = address(tensor);
    switch (static_cast<Type>(tensor.type)) {
        case Type::BOOL:
            static_cast<unsigned char*>(data)[index] = value != 0;
            break;
        case Type::U8:
            static_cast<unsigned char*>(data)[index] =
                static_cast<unsigned char>(fminf(255, fmaxf(0, nearbyintf(value))));
            break;
        case Type::I8:
            static_cast<signed char*>(data)[index] =
                static_cast<signed char>(fminf(127, fmaxf(-128, nearbyintf(value))));
            break;
        case Type::I32:
            static_cast<int*>(data)[index] = static_cast<int>(value);
            break;
        case Type::I64:
            static_cast<long long*>(data)[index] = static_cast<long long>(value);
            break;
        case Type::F16:
            static_cast<__half*>(data)[index] = __float2half(value);
            break;
        case Type::BF16:
            static_cast<__nv_bfloat16*>(data)[index] = __float2bfloat16(value);
            break;
        case Type::F32:
            static_cast<float*>(data)[index] = value;
            break;
        case Type::F64:
            static_cast<double*>(data)[index] = value;
            break;
        default:
            break;
    }
}
__device__ std::size_t broadcast_index(std::size_t index, const View& output, const View& input) {
    std::size_t offset = 0, stride = 1;
    for (int axis = output.rank - 1; axis >= 0; --axis) {
        const auto coordinate = index % output.shape[axis];
        index /= output.shape[axis];
        const int input_axis = axis - output.rank + input.rank;
        if (input_axis >= 0) {
            if (input.shape[input_axis] != 1) offset += coordinate * stride;
            stride *= input.shape[input_axis];
        }
    }
    return offset;
}

__global__ void pointwise(Launch plan) {
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= plan.elements) return;
    const float left = read(plan.inputs[0], broadcast_index(index, plan.output, plan.inputs[0]));
    float result = left;
    if (plan.operation == static_cast<int>(Op::ADD))
        result += read(plan.inputs[1], broadcast_index(index, plan.output, plan.inputs[1]));
    if (plan.operation == static_cast<int>(Op::MULTIPLY))
        result *= read(plan.inputs[1], broadcast_index(index, plan.output, plan.inputs[1]));
    if (plan.operation == static_cast<int>(Op::GELU)) {
        const float probability = plan.attribute_count && plan.attributes[0]
                                      ? tanhf(0.7978845608028654f * (left + 0.044715f * left * left * left))
                                      : erff(left * 0.7071067811865475f);
        result = 0.5f * left * (1 + probability);
    }
    if (plan.operation == static_cast<int>(Op::TANH)) result = tanhf(left);
    if (plan.operation == static_cast<int>(Op::STATIC_ROUND))
        result = fminf(127, fmaxf(-128, nearbyintf(left / plan.epsilon))) * plan.epsilon;
    if (plan.operation == static_cast<int>(Op::CAST) && plan.epsilon > 0) result = nearbyintf(left / plan.epsilon);
    if (plan.operation == static_cast<int>(Op::GELU_MULTIPLY))
        result = 0.5f * left * (1 + tanhf(0.7978845608028654f * (left + 0.044715f * left * left * left))) *
                 read(plan.inputs[1], index);
    if (plan.operation == static_cast<int>(Op::SILU_MULTIPLY))
        result = left / (1 + expf(-left)) * read(plan.inputs[1], index);
    if (plan.operation == static_cast<int>(Op::SILU_GATE)) {
        const auto width = plan.output.shape[plan.output.rank - 1];
        const auto offset = index / width * 2 * width + index % width;
        const float gate = read(plan.inputs[0], offset);
        result = gate / (1 + expf(-gate)) * read(plan.inputs[0], offset + width);
    }
    if (plan.operation == static_cast<int>(Op::ROTARY)) {
        const auto width = plan.output.shape[3], half = width / 2;
        const auto column = index % width;
        const auto sequence = index / width / plan.output.shape[2] % plan.output.shape[1];
        const auto angle = sequence * half + column % half;
        const float paired = read(plan.inputs[0], index - column + (column + half) % width);
        result = left * read(plan.inputs[1], angle) + (column < half ? -paired : paired) * read(plan.inputs[2], angle);
    }
    if (plan.operation == static_cast<int>(Op::SNAKE)) {
        const float alpha = read(plan.inputs[1], index % plan.output.shape[2]);
        const float sine = sinf(alpha * left);
        result = fabsf(alpha) < 1e-9f ? left : left + sine * sine / (alpha + 1e-9f);
    }
    write(plan.output, index, result);
}

__global__ void gated_activation(Launch plan) {
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= plan.elements) return;
    const auto width = plan.output.shape[plan.output.rank - 1];
    const auto offset = index / width * 2 * width + index % width;
    const float gate = read(plan.inputs[0], offset), up = read(plan.inputs[0], offset + width);
    const float result =
        plan.operation == static_cast<int>(Op::GELU_GATE)
            ? 0.5F * gate * (1 + tanhf(0.7978845608028654F * (gate + 0.044715F * gate * gate * gate))) * up
            : gate / (1 + expf(-gate)) * up;
    write(plan.output, index, result);
}

__global__ void unfold_convolution(Launch plan, float* columns, std::size_t elements) {
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    const auto kernel = plan.attributes[0], stride = plan.attributes[1], padding = plan.attributes[2],
               dilation = plan.attributes[3];
    const auto channels = plan.inputs[0].shape[2];
    const auto width = channels * kernel;
    const auto position = static_cast<std::int64_t>(index / width) * stride - padding +
                          static_cast<std::int64_t>(index % kernel) * dilation;
    const auto channel = index % width / kernel;
    float value =
        position < 0 || position >= plan.inputs[0].shape[1] ? 0 : read(plan.inputs[0], position * channels + channel);
    if (plan.operation == static_cast<int>(Op::SNAKE_CONV1D) && position >= 0 && position < plan.inputs[0].shape[1]) {
        const float alpha = read(plan.inputs[1], channel), sine = sinf(alpha * value);
        if (fabsf(alpha) >= 1e-9f) value += sine * sine / (alpha + 1e-9f);
    }
    columns[index] = value;
}

__global__ void fold_convolution(Launch plan) {
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= plan.elements) return;
    const auto kernel = plan.attributes[0], stride = plan.attributes[1], padding = plan.attributes[2],
               dilation = plan.attributes[4];
    const auto channels = plan.output.shape[2], position = static_cast<std::int64_t>(index / channels),
               channel = static_cast<std::int64_t>(index % channels);
    float value = read(plan.inputs[1], channel);
    for (std::int64_t slot = 0; slot < kernel; ++slot) {
        const auto input = position + padding - slot * dilation;
        if (input < 0 || input % stride || input / stride >= plan.inputs[0].shape[1]) continue;
        value += read(plan.inputs[0], (input / stride * channels + channel) * kernel + slot);
    }
    write(plan.output, index, value);
}

__device__ float reduction(float value, float* shared, bool maximum) {
    shared[threadIdx.x] = value;
    __syncthreads();
    for (int step = blockDim.x / 2; step; step /= 2) {
        if (threadIdx.x < step)
            shared[threadIdx.x] = maximum ? fmaxf(shared[threadIdx.x], shared[threadIdx.x + step])
                                          : shared[threadIdx.x] + shared[threadIdx.x + step];
        __syncthreads();
    }
    const float result = shared[0];
    __syncthreads();
    return result;
}

template <bool Bfloat>
__global__ void normalize(Launch plan) {
    __shared__ float shared[256];
    const auto row = blockIdx.x;
    const auto width = plan.output.shape[plan.output.rank - 1];
    const auto operation = static_cast<Op>(plan.operation);
    const bool softmax = operation == Op::SOFTMAX || operation == Op::LOG_SOFTMAX;
    const bool residual = operation == Op::RESIDUAL_NORM;
    const bool centered = residual || operation == Op::LAYER_NORM;
    auto source = plan.inputs[0];
    std::size_t base = row * width;
    if (operation == Op::RMS_ROTARY && plan.attribute_count == 5) {
        const auto head = row % plan.output.shape[2];
        const auto sequence = row / plan.output.shape[2] % plan.output.shape[1];
        const auto batch = row / (plan.output.shape[2] * plan.output.shape[1]);
        base = (batch * source.shape[1] + sequence + plan.attributes[3]) * source.shape[2] + plan.attributes[2] +
               head * width;
    }
    float aggregate = softmax ? -INFINITY : 0;
    for (std::int64_t column = threadIdx.x; column < width; column += blockDim.x) {
        float value = read(source, base + column);
        if (residual) value += read(plan.inputs[1], base + column);
        aggregate = softmax ? fmaxf(aggregate, value) : aggregate + (centered ? value : value * value);
    }
    const float first = reduction(aggregate, shared, softmax);
    const float mean = centered ? first / width : 0;
    float second = 0;
    if (softmax || centered) {
        for (std::int64_t column = threadIdx.x; column < width; column += blockDim.x) {
            float value = read(source, base + column);
            if (residual) value += read(plan.inputs[1], base + column);
            second += softmax ? expf(value - first) : (value - mean) * (value - mean);
        }
        second = reduction(second, shared, false);
    }
    const float inverse = softmax ? 1 / second : rsqrtf((centered ? second : first) / width + plan.epsilon);
    for (std::int64_t column = threadIdx.x; column < width; column += blockDim.x) {
        float value = read(source, base + column);
        if (residual) value += read(plan.inputs[1], base + column);
        if (softmax)
            value = operation == Op::LOG_SOFTMAX ? value - first - logf(second) : expf(value - first) * inverse;
        else {
            const int scale_index = residual ? 2 : 1;
            value = (value - mean) * inverse * read(plan.inputs[scale_index], column);
            if (centered) value += read(plan.inputs[scale_index + 1], column);
            if (operation == Op::RMS_NORM_RESIDUAL) {
                if constexpr (Bfloat) value = __bfloat162float(__float2bfloat16(value));
                value += read(plan.inputs[2], row * width + column);
                if (plan.count == 4) {
                    if constexpr (Bfloat) value = __bfloat162float(__float2bfloat16(value));
                    value *= read(plan.inputs[3], 0);
                }
            }
            if (operation == Op::RMS_ROTARY) {
                const auto half = width / 2;
                const auto partner = (column + half) % width;
                float paired = read(source, base + partner) * inverse * read(plan.inputs[1], partner);
                if constexpr (Bfloat) {
                    value = __bfloat162float(__float2bfloat16(value));
                    paired = __bfloat162float(__float2bfloat16(paired));
                }
                const auto sequence = row / plan.output.shape[2] % plan.output.shape[1];
                const auto angle = sequence * half + column % half;
                value = value * read(plan.inputs[2], angle) +
                        (column < half ? -paired : paired) * read(plan.inputs[3], angle);
            }
        }
        write(plan.output, row * width + column, value);
    }
}

template <Op Operation>
__global__ void bfloat_normalize(Launch plan) {
    __shared__ float partial[8];
    const auto width = plan.output.shape[plan.output.rank - 1], row = static_cast<std::int64_t>(blockIdx.x);
    const auto* input = static_cast<const __nv_bfloat16*>(plan.inputs[0].data) + row * width;
    const auto* scale = static_cast<const float*>(plan.inputs[1].data);
    float values[8]{}, sum = 0;
#pragma unroll
    for (int chunk = 0; chunk < 8; ++chunk) {
        const auto column = threadIdx.x + chunk * 256;
        if (column < width) {
            values[chunk] = __bfloat162float(input[column]);
            sum = fmaf(values[chunk], values[chunk], sum);
        }
    }
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    for (int delta = 16; delta > 0; delta /= 2) sum += __shfl_down_sync(0xffffffff, sum, delta);
    if (lane == 0) partial[warp] = sum;
    __syncthreads();
    if (warp == 0) {
        sum = lane < 8 ? partial[lane] : 0;
        for (int delta = 16; delta > 0; delta /= 2) sum += __shfl_down_sync(0xffffffff, sum, delta);
        if (lane == 0) partial[0] = rsqrtf(sum / static_cast<float>(width) + plan.epsilon);
    }
    __syncthreads();
    const float inverse = partial[0];
#pragma unroll
    for (int chunk = 0; chunk < 8; ++chunk) {
        const auto column = threadIdx.x + chunk * 256;
        if (column >= width) continue;
        float result = values[chunk] * inverse * scale[column];
        if constexpr (Operation == Op::RMS_NORM_RESIDUAL) {
            result = __bfloat162float(__float2bfloat16(result)) +
                     __bfloat162float(static_cast<const __nv_bfloat16*>(plan.inputs[2].data)[row * width + column]);
            if (plan.count == 4)
                result = __bfloat162float(__float2bfloat16(result)) * static_cast<const float*>(plan.inputs[3].data)[0];
        }
        if constexpr (Operation == Op::RMS_ROTARY) {
            const auto half = width / 2, partner = (column + half) % width;
            const float paired =
                __bfloat162float(__float2bfloat16(__bfloat162float(input[partner]) * inverse * scale[partner]));
            const auto sequence = row / plan.output.shape[2] % plan.output.shape[1];
            const auto angle = sequence * half + column % half;
            result =
                __bfloat162float(__float2bfloat16(result)) * static_cast<const float*>(plan.inputs[2].data)[angle] +
                (column < half ? -paired : paired) * static_cast<const float*>(plan.inputs[3].data)[angle];
        }
        static_cast<__nv_bfloat16*>(plan.output.data)[row * width + column] = __float2bfloat16(result);
    }
}

struct TokenCandidate {
    float value;
    int index;
};

__device__ TokenCandidate reduce_candidate(TokenCandidate candidate, TokenCandidate* shared) {
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    for (int delta = 16; delta > 0; delta /= 2) {
        const float other_value = __shfl_down_sync(0xffffffff, candidate.value, delta);
        const int other_index = __shfl_down_sync(0xffffffff, candidate.index, delta);
        if (other_value > candidate.value || (other_value == candidate.value && other_index < candidate.index))
            candidate = {other_value, other_index};
    }
    if (lane == 0) shared[warp] = candidate;
    __syncthreads();
    if (warp == 0) {
        candidate = lane < 8 ? shared[lane] : TokenCandidate{-INFINITY, INT32_MAX};
        for (int delta = 16; delta > 0; delta /= 2) {
            const float other_value = __shfl_down_sync(0xffffffff, candidate.value, delta);
            const int other_index = __shfl_down_sync(0xffffffff, candidate.index, delta);
            if (other_value > candidate.value || (other_value == candidate.value && other_index < candidate.index))
                candidate = {other_value, other_index};
        }
    }
    return candidate;
}

__global__ void greedy_partitions(Launch plan, TokenCandidate* candidates, std::int64_t partitions) {
    __shared__ TokenCandidate partial[8];
    const auto width = plan.inputs[0].shape[plan.inputs[0].rank - 1];
    const auto row = blockIdx.x / partitions, partition = blockIdx.x % partitions;
    const auto begin = partition * 2048, end = min(begin + 2048, width);
    TokenCandidate candidate{-INFINITY, INT32_MAX};
    for (std::int64_t column = begin + threadIdx.x; column < end; column += blockDim.x) {
        const float value = read(plan.inputs[0], row * width + column);
        if (isnan(value) || value == INFINITY)
            candidate = {INFINITY, -1};
        else if (value > candidate.value || (value == candidate.value && column < candidate.index))
            candidate = {value, static_cast<int>(column)};
    }
    candidate = reduce_candidate(candidate, partial);
    if (threadIdx.x == 0) candidates[blockIdx.x] = candidate;
}

__global__ void greedy_merge(Launch plan, const TokenCandidate* candidates, std::int64_t partitions) {
    __shared__ TokenCandidate partial[8];
    TokenCandidate candidate{-INFINITY, INT32_MAX};
    for (std::int64_t partition = threadIdx.x; partition < partitions; partition += blockDim.x) {
        const auto other = candidates[blockIdx.x * partitions + partition];
        if (other.value > candidate.value || (other.value == candidate.value && other.index < candidate.index))
            candidate = other;
    }
    candidate = reduce_candidate(candidate, partial);
    if (threadIdx.x == 0)
        static_cast<int*>(plan.output.data)[blockIdx.x] = isfinite(candidate.value) ? candidate.index : -1;
}

__global__ void quantize_rows(Launch plan, signed char* quantized, float* scales, int* zeros) {
    __shared__ float shared[256];
    const auto width = plan.inputs[0].shape[plan.inputs[0].rank - 1];
    const auto row = blockIdx.x;
    float minimum = 0, maximum = 0;
    for (std::int64_t column = threadIdx.x; column < width; column += blockDim.x) {
        const float value = read(plan.inputs[0], row * width + column);
        minimum = fminf(minimum, value);
        maximum = fmaxf(maximum, value);
    }
    minimum = -reduction(-minimum, shared, true);
    maximum = reduction(maximum, shared, true);
    const float difference = maximum - minimum;
    const bool calibrated = plan.operation == static_cast<int>(Op::PACKED_LINEAR) && plan.epsilon > 0;
    const float scale = calibrated ? plan.epsilon : difference == 0 ? 1 : difference / 255;
    const int zero = calibrated ? 0
                     : difference == 0
                         ? -128
                         : static_cast<int>(fminf(255, fmaxf(0, -minimum * (255 / difference))) + 0.5f) - 128;
    if (threadIdx.x == 0) {
        scales[row] = scale;
        zeros[row] = zero;
    }
    for (std::int64_t column = threadIdx.x; column < width; column += blockDim.x)
        quantized[row * width + column] = static_cast<signed char>(
            fminf(127, fmaxf(-128, nearbyintf(read(plan.inputs[0], row * width + column) / scale) + zero)));
}

__global__ void quantized_matrix(Launch plan, const signed char* quantized, const float* scales, const int* zeros) {
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= plan.elements) return;
    const auto columns = plan.output.shape[plan.output.rank - 1];
    const auto width = plan.inputs[0].shape[plan.inputs[0].rank - 1];
    const auto row = index / columns, column = index % columns;
    const bool packed = plan.operation == static_cast<int>(Op::PACKED_LINEAR);
    const auto bits = packed ? plan.attributes[0] : 8;
    const auto group = packed ? plan.attributes[1] : width;
    const auto groups = width / group;
    float result = 0;
    for (std::int64_t block = 0; block < groups; ++block) {
        int sum = 0;
        for (std::int64_t slot = 0; slot < group; ++slot) {
            const auto contraction = block * group + slot;
            int weight;
            if (packed) {
                const auto position = column * width + contraction;
                const int byte = static_cast<const unsigned char*>(plan.inputs[1].data)[position / (8 / bits)];
                weight = (byte >> (position % (8 / bits) * bits)) & ((1 << bits) - 1);
                if (weight & (1 << (bits - 1))) weight -= 1 << bits;
            } else
                weight = static_cast<const signed char*>(
                    plan.inputs[1]
                        .data)[plan.attributes[0] ? column * width + contraction : contraction * columns + column];
            sum += (static_cast<int>(quantized[row * width + contraction]) - zeros[row]) * weight;
        }
        result += static_cast<float>(sum) * scales[row] * read(plan.inputs[2], column * groups + block);
    }
    if (!packed) result += read(plan.inputs[3], column);
    if (packed && plan.attribute_count > 2) {
        const float output_scale = __int_as_float(static_cast<int>(plan.attributes[2]));
        if (output_scale > 0) result = fminf(127, fmaxf(-128, nearbyintf(result / output_scale))) * output_scale;
    }
    write(plan.output, index, result);
}

template <int Bits>
__global__ void prepare_projection_weights(Launch plan, signed char* expanded, int* sums) {
    __shared__ int partial[256];
    const auto width = plan.inputs[0].shape[plan.inputs[0].rank - 1];
    const auto* weights = static_cast<const signed char*>(plan.inputs[1].data);
    int sum = 0;
    for (std::int64_t contraction = threadIdx.x; contraction < width; contraction += blockDim.x) {
        const auto position = blockIdx.x * width + contraction;
        int value;
        if constexpr (Bits == 8)
            value = weights[position];
        else {
            const auto byte = static_cast<const unsigned char*>(plan.inputs[1].data)[position / (8 / Bits)];
            value = (byte >> (position % (8 / Bits) * Bits)) & ((1 << Bits) - 1);
            if (value & (1 << (Bits - 1))) value -= 1 << Bits;
            expanded[position] = static_cast<signed char>(value);
        }
        sum += value;
    }
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (int step = blockDim.x / 2; step; step /= 2) {
        if (threadIdx.x < step) partial[threadIdx.x] += partial[threadIdx.x + step];
        __syncthreads();
    }
    if (threadIdx.x == 0) sums[blockIdx.x] = partial[0];
}

__device__ void write_projection(Launch plan, std::size_t row, std::size_t column, int corrected, const float* scales) {
    const auto columns = plan.output.shape[plan.output.rank - 1];
    float result = static_cast<float>(corrected) * scales[row] * read(plan.inputs[2], column);
    if (plan.operation == static_cast<int>(Op::QUANTIZED_LINEAR))
        result += read(plan.inputs[3], column);
    else if (plan.attribute_count > 2) {
        const float output_scale = __int_as_float(static_cast<int>(plan.attributes[2]));
        if (output_scale > 0) result = fminf(127, fmaxf(-128, nearbyintf(result / output_scale))) * output_scale;
    }
    write(plan.output, row * columns + column, result);
}

__global__ void projection_epilogue(Launch plan, const int* accumulators, const int* sums, const float* scales,
                                    const int* zeros) {
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= plan.elements) return;
    const auto columns = plan.output.shape[plan.output.rank - 1];
    const auto row = index / columns, column = index % columns;
    write_projection(plan, row, column, accumulators[index] - zeros[row] * sums[column], scales);
}

template <int Bits>
__global__ void quantized_vector(Launch plan, const signed char* quantized, const float* scales, const int* zeros) {
    constexpr int slots = 32 / Bits;
    const int lane = threadIdx.x % 32;
    const auto column = static_cast<std::size_t>(blockIdx.x) * (blockDim.x / 32) + threadIdx.x / 32;
    const auto row = static_cast<std::size_t>(blockIdx.y);
    const auto columns = plan.output.shape[plan.output.rank - 1];
    const auto width = plan.inputs[0].shape[plan.inputs[0].rank - 1];
    if (column >= columns) return;
    const auto words = width / slots;
    const auto* weight = static_cast<const unsigned int*>(plan.inputs[1].data) + column * words;
    int sum = 0, weight_sum = 0;
    for (std::int64_t word = lane; word < words; word += 32) {
        const auto packed = weight[word];
        for (int chunk = 0; chunk < slots / 4; ++chunk) {
            unsigned int values = packed;
            if constexpr (Bits != 8) {
                values = 0;
                for (int element = 0; element < 4; ++element) {
                    int value = (packed >> ((chunk * 4 + element) * Bits)) & ((1 << Bits) - 1);
                    if (value & (1 << (Bits - 1))) value -= 1 << Bits;
                    values |= static_cast<unsigned int>(static_cast<unsigned char>(value)) << (8 * element);
                }
            }
            const int activation = *reinterpret_cast<const int*>(quantized + row * width + word * slots + chunk * 4);
            sum = __dp4a(activation, static_cast<int>(values), sum);
            weight_sum = __dp4a(0x01010101, static_cast<int>(values), weight_sum);
        }
    }
    for (int offset = 16; offset; offset /= 2) {
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
        weight_sum += __shfl_down_sync(0xffffffffu, weight_sum, offset);
    }
    if (lane == 0) write_projection(plan, row, column, sum - zeros[row] * weight_sum, scales);
}

__device__ void copy_element(const View& output, std::size_t destination, const View& input, std::size_t source) {
    const auto type = static_cast<Type>(output.type);
    if (type == Type::F32 || type == Type::I32)
        static_cast<std::uint32_t*>(address(output))[destination] =
            static_cast<const std::uint32_t*>(address(input))[source];
    else if (type == Type::F16 || type == Type::BF16)
        static_cast<std::uint16_t*>(address(output))[destination] =
            static_cast<const std::uint16_t*>(address(input))[source];
    else
        static_cast<unsigned char*>(address(output))[destination] =
            static_cast<const unsigned char*>(address(input))[source];
}

__global__ void layout(Launch plan) {
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= plan.elements) return;
    const auto operation = static_cast<Op>(plan.operation);
    auto source = plan.inputs[0];
    std::size_t source_index = 0, remaining = index;
    std::int64_t coordinates[8]{};
    for (int axis = plan.output.rank - 1; axis >= 0; --axis) {
        coordinates[axis] = remaining % plan.output.shape[axis];
        remaining /= plan.output.shape[axis];
    }
    if (operation == Op::TRANSPOSE) {
        std::int64_t input_coordinates[8]{};
        for (int axis = 0; axis < source.rank; ++axis) input_coordinates[plan.attributes[axis]] = coordinates[axis];
        for (int axis = 0; axis < source.rank; ++axis)
            source_index = source_index * source.shape[axis] + input_coordinates[axis];
    } else if (operation == Op::SLICE || operation == Op::GATHER) {
        const int axis =
            static_cast<int>(plan.attributes[0] < 0 ? plan.attributes[0] + source.rank : plan.attributes[0]);
        coordinates[axis] = operation == Op::SLICE ? coordinates[axis] + plan.attributes[1]
                                                   : static_cast<std::int64_t>(read(plan.inputs[1], coordinates[axis]));
        if (coordinates[axis] < 0) coordinates[axis] += source.shape[axis];
        if (coordinates[axis] < 0 || coordinates[axis] >= source.shape[axis]) {
            write(plan.output, index, 0);
            return;
        }
        for (int dimension = 0; dimension < source.rank; ++dimension)
            source_index = source_index * source.shape[dimension] + coordinates[dimension];
    } else if (operation == Op::CONCAT) {
        const int axis =
            static_cast<int>(plan.attributes[0] < 0 ? plan.attributes[0] + source.rank : plan.attributes[0]);
        int selected = 0;
        while (selected + 1 < plan.count && coordinates[axis] >= plan.inputs[selected].shape[axis])
            coordinates[axis] -= plan.inputs[selected++].shape[axis];
        source = plan.inputs[selected];
        for (int dimension = 0; dimension < source.rank; ++dimension)
            source_index = source_index * source.shape[dimension] + coordinates[dimension];
    } else if (operation == Op::SCATTER) {
        source_index = index;
        const auto rows = plan.inputs[1].shape[1];
        const auto width = source.shape[2];
        const auto output_row = index / width % source.shape[1];
        for (std::int64_t row = 0; row < rows; ++row) {
            if (static_cast<std::int64_t>(read(plan.inputs[2], row)) != output_row) continue;
            source = plan.inputs[1];
            source_index = (index / (width * plan.output.shape[1]) * rows + row) * width + index % width;
        }
    }
    copy_element(plan.output, index, source, source_index);
}

__global__ void scatter_updates(Launch plan) {
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= plan.elements) return;
    const auto width = plan.inputs[1].shape[2], rows = plan.inputs[1].shape[1];
    const auto row = index / width % rows, batch = index / (width * rows);
    const auto* indices = static_cast<const int*>(address(plan.inputs[2]));
    const auto target = indices[row];
    if (target < 0 || target >= plan.output.shape[1]) return;
    for (std::int64_t later = row + 1; later < rows; ++later)
        if (indices[later] == target) return;
    const auto destination = (batch * plan.output.shape[1] + target) * width + index % width;
    const auto type = static_cast<Type>(plan.output.type);
    const int bytes = type == Type::F16 || type == Type::BF16 ? 2 : type == Type::F32 || type == Type::I32 ? 4 : 1;
    auto* output = static_cast<unsigned char*>(address(plan.output));
    const auto* source = static_cast<const unsigned char*>(address(plan.inputs[1]));
    for (int byte = 0; byte < bytes; ++byte) output[destination * bytes + byte] = source[index * bytes + byte];
}

__global__ void embedding(Launch plan) {
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= plan.elements) return;
    const auto width = plan.attributes[0], bits = plan.attributes[1];
    const auto token = static_cast<std::int64_t>(read(plan.inputs[0], index / width));
    const auto column = index % width;
    if (token < 0 || token >= plan.inputs[1].shape[0]) {
        write(plan.output, index, 0);
        return;
    }
    float value;
    if (bits) {
        const auto position = token * width + column;
        const auto byte = static_cast<const unsigned char*>(plan.inputs[1].data)[position / (8 / bits)];
        int weight = (byte >> (position % (8 / bits) * bits)) & ((1 << bits) - 1);
        if (weight & (1 << (bits - 1))) weight -= 1 << bits;
        const auto groups = plan.attributes[2];
        value = weight * read(plan.inputs[2], token * groups + column / (width / groups));
    } else
        value = read(plan.inputs[1], token * width + column);
    write(plan.output, index, value * plan.epsilon);
}

__global__ void attention_scores(Launch plan, float* scores, std::int64_t heads, std::int64_t key_heads,
                                 std::int64_t keys) {
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const auto queries = plan.inputs[0].shape[1];
    const auto batches = plan.inputs[0].shape[0];
    if (index >= batches * heads * queries * keys) return;
    const auto key = index % keys, query = index / keys % queries;
    const auto head = index / (keys * queries) % heads, batch = index / (keys * queries * heads);
    const auto width = plan.inputs[0].shape[2] / heads;
    float value = 0;
    for (std::int64_t column = 0; column < width; ++column)
        value += read(plan.inputs[0], (batch * queries + query) * heads * width + head * width + column) *
                 read(plan.inputs[1], ((plan.inputs[1].shape[0] == 1 ? 0 : batch) * plan.inputs[1].shape[1] + key +
                                       (plan.attribute_count >= 3 ? plan.attributes[2] : 0)) *
                                              key_heads * width +
                                          (head / (heads / key_heads)) * width + column);
    value *= plan.epsilon > 0 ? plan.epsilon : rsqrtf(static_cast<float>(width));
    if (plan.count == 4) {
        View score_view{};
        score_view.rank = 4;
        score_view.shape[0] = batches;
        score_view.shape[1] = heads;
        score_view.shape[2] = queries;
        score_view.shape[3] = keys;
        value += read(plan.inputs[3], broadcast_index(index, score_view, plan.inputs[3]));
    }
    scores[index] = value;
}

__global__ void attention_mask(Launch plan, float* scores, std::int64_t heads, std::int64_t keys) {
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= plan.inputs[0].shape[0] * heads * plan.inputs[0].shape[1] * keys) return;
    View score_view{};
    score_view.rank = 4;
    score_view.shape[0] = plan.inputs[0].shape[0];
    score_view.shape[1] = heads;
    score_view.shape[2] = plan.inputs[0].shape[1];
    score_view.shape[3] = keys;
    scores[index] += read(plan.inputs[3], broadcast_index(index, score_view, plan.inputs[3]));
}

void execute_attention_scores(Launch plan, float* scores, int device, std::int64_t heads, std::int64_t key_heads,
                              std::int64_t keys, bool reference) {
    const auto queries = plan.inputs[0].shape[1], width = plan.inputs[0].shape[2] / heads;
    const auto score_count = plan.inputs[0].shape[0] * heads * queries * keys;
    if (reference || queries <= 1 || plan.inputs[0].type != static_cast<int>(Type::BF16) ||
        plan.inputs[1].type != static_cast<int>(Type::BF16) || keys > INT32_MAX || queries > INT32_MAX ||
        heads * width > INT32_MAX) {
        attention_scores<<<static_cast<unsigned int>((score_count + 255) / 256), 256, 0, execution_stream>>>(
            plan, scores, heads, key_heads, keys);
        return;
    }
    auto& owner = tensor_blas_handles[device];
    if (!owner) owner = std::make_unique<BlasHandle>(device, CUBLAS_DEFAULT_MATH);
    owner->bind_stream();
    const auto group = heads / key_heads;
    const auto offset = plan.attribute_count >= 3 ? plan.attributes[2] : 0;
    const float alpha = plan.epsilon > 0 ? plan.epsilon : 1.F / std::sqrt(static_cast<float>(width)), beta = 0;
    for (std::int64_t batch = 0; batch < plan.inputs[0].shape[0]; ++batch)
        for (std::int64_t key_head = 0; key_head < key_heads; ++key_head) {
            const auto* query = static_cast<const __nv_bfloat16*>(plan.inputs[0].data) +
                                batch * queries * heads * width + key_head * group * width;
            const auto* key =
                static_cast<const __nv_bfloat16*>(plan.inputs[1].data) +
                ((plan.inputs[1].shape[0] == 1 ? 0 : batch) * plan.inputs[1].shape[1] + offset) * key_heads * width +
                key_head * width;
            auto* destination = scores + (batch * heads + key_head * group) * queries * keys;
            const auto status = cublasGemmStridedBatchedEx(
                owner->handle, CUBLAS_OP_T, CUBLAS_OP_N, static_cast<int>(keys), static_cast<int>(queries),
                static_cast<int>(width), &alpha, key, CUDA_R_16BF, static_cast<int>(key_heads * width), 0, query,
                CUDA_R_16BF, static_cast<int>(heads * width), width, &beta, destination, CUDA_R_32F,
                static_cast<int>(keys), queries * keys, static_cast<int>(group), CUBLAS_COMPUTE_32F,
                CUBLAS_GEMM_DEFAULT);
            if (status != CUBLAS_STATUS_SUCCESS)
                throw std::runtime_error("CUDA: attention score GEMM failed with status " + std::to_string(status));
        }
    if (plan.count == 4)
        attention_mask<<<static_cast<unsigned int>((score_count + 255) / 256), 256, 0, execution_stream>>>(plan, scores,
                                                                                                           heads, keys);
}

__global__ void attention_values(Launch plan, const float* scores, std::int64_t heads, std::int64_t key_heads,
                                 std::int64_t keys) {
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= plan.elements) return;
    const auto width = plan.output.shape[2] / heads, queries = plan.output.shape[1];
    const auto column = index % width, head = index / width % heads;
    const auto query = index / (width * heads) % queries, batch = index / (width * heads * queries);
    float value = 0;
    for (std::int64_t key = 0; key < keys; ++key)
        value += scores[((batch * heads + head) * queries + query) * keys + key] *
                 read(plan.inputs[2], ((plan.inputs[2].shape[0] == 1 ? 0 : batch) * plan.inputs[2].shape[1] + key +
                                       (plan.attribute_count >= 3 ? plan.attributes[2] : 0)) *
                                              key_heads * width +
                                          head / (heads / key_heads) * width + column);
    write(plan.output, index, value);
}

__global__ void bfloat_probabilities(const float* source, __nv_bfloat16* output, std::size_t elements) {
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < elements) output[index] = __float2bfloat16(source[index]);
}

void execute_attention_values(Launch plan, const __nv_bfloat16* probabilities, int device, std::int64_t heads,
                              std::int64_t key_heads, std::int64_t keys) {
    const auto queries = plan.inputs[0].shape[1], width = plan.inputs[0].shape[2] / heads;
    const auto group = heads / key_heads;
    const auto offset = plan.attribute_count >= 3 ? plan.attributes[2] : 0;
    auto& owner = tensor_blas_handles[device];
    if (!owner) owner = std::make_unique<BlasHandle>(device, CUBLAS_DEFAULT_MATH);
    owner->bind_stream();
    const float alpha = 1, beta = 0;
    for (std::int64_t batch = 0; batch < plan.inputs[0].shape[0]; ++batch)
        for (std::int64_t key_head = 0; key_head < key_heads; ++key_head) {
            const auto* value =
                static_cast<const __nv_bfloat16*>(plan.inputs[2].data) +
                ((plan.inputs[2].shape[0] == 1 ? 0 : batch) * plan.inputs[2].shape[1] + offset) * key_heads * width +
                key_head * width;
            const auto* probability = probabilities + (batch * heads + key_head * group) * queries * keys;
            auto* output = static_cast<__nv_bfloat16*>(plan.output.data) + batch * queries * heads * width +
                           key_head * group * width;
            const auto status = cublasGemmStridedBatchedEx(
                owner->handle, CUBLAS_OP_N, CUBLAS_OP_N, static_cast<int>(width), static_cast<int>(queries),
                static_cast<int>(keys), &alpha, value, CUDA_R_16BF, static_cast<int>(key_heads * width), 0, probability,
                CUDA_R_16BF, static_cast<int>(keys), queries * keys, &beta, output, CUDA_R_16BF,
                static_cast<int>(heads * width), width, static_cast<int>(group), CUBLAS_COMPUTE_32F,
                CUBLAS_GEMM_DEFAULT);
            if (status != CUBLAS_STATUS_SUCCESS)
                throw std::runtime_error("CUDA: attention value GEMM failed with status " + std::to_string(status));
        }
}

template <int Vector>
__device__ void load_bfloat_vector(const __nv_bfloat16* source, float (&output)[Vector]) {
#pragma unroll
    for (int block = 0; block < Vector / 8; ++block) {
        union Packed {
            uint4 words;
            __nv_bfloat162 pairs[4];
        } value;
        value.words = *reinterpret_cast<const uint4*>(source + block * 8);
#pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            const float2 decoded = __bfloat1622float2(value.pairs[pair]);
            output[block * 8 + pair * 2] = decoded.x;
            output[block * 8 + pair * 2 + 1] = decoded.y;
        }
    }
}

template <int Width, int Group>
__global__ void bfloat_decode_partitions(Launch plan, float* partials, std::int64_t heads, std::int64_t key_heads,
                                         std::int64_t keys, std::int64_t partitions) {
    constexpr int Vector = Width / 32, Warps = 4, Partition = 64;
    __shared__ float maxima[Warps][Group], denominators[Warps][Group], outputs[Warps][Group][Width];
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const auto batch = static_cast<std::int64_t>(blockIdx.x), head = static_cast<std::int64_t>(blockIdx.y) * Group;
    const auto key_head = head / (heads / key_heads);
    const auto offset = plan.attribute_count >= 3 ? plan.attributes[2] : 0;
    const auto key_batch = plan.inputs[1].shape[0] == 1 ? 0 : batch;
    const auto* query = static_cast<const __nv_bfloat16*>(plan.inputs[0].data) + (batch * heads + head) * Width;
    const auto* key = static_cast<const __nv_bfloat16*>(address(plan.inputs[1])) +
                      (key_batch * plan.inputs[1].shape[1] + offset) * key_heads * Width + key_head * Width;
    const auto* value = static_cast<const __nv_bfloat16*>(address(plan.inputs[2])) +
                        (key_batch * plan.inputs[2].shape[1] + offset) * key_heads * Width + key_head * Width;
    float activation[Group][Vector], accumulated[Group][Vector]{}, maximum[Group], denominator[Group]{};
#pragma unroll
    for (int group = 0; group < Group; ++group) {
        load_bfloat_vector(query + group * Width + lane * Vector, activation[group]);
        maximum[group] = -INFINITY;
    }
    const float scale = plan.epsilon > 0 ? plan.epsilon : rsqrtf(static_cast<float>(Width));
    View score_view{};
    score_view.rank = 4;
    score_view.shape[0] = plan.inputs[0].shape[0];
    score_view.shape[1] = heads;
    score_view.shape[2] = 1;
    score_view.shape[3] = keys;
    const auto begin = static_cast<std::int64_t>(blockIdx.z) * Partition;
    const auto end = min(begin + Partition, keys);
    for (std::int64_t position = begin + warp; position < end; position += Warps) {
        float key_values[Vector], value_values[Vector];
        load_bfloat_vector(key + position * key_heads * Width + lane * Vector, key_values);
        load_bfloat_vector(value + position * key_heads * Width + lane * Vector, value_values);
#pragma unroll
        for (int group = 0; group < Group; ++group) {
            float score = 0;
#pragma unroll
            for (int column = 0; column < Vector; ++column)
                score = fmaf(activation[group][column], key_values[column], score);
            for (int delta = 16; delta > 0; delta /= 2) score += __shfl_xor_sync(0xffffffff, score, delta);
            score *= scale;
            if (plan.count == 4)
                score += read(plan.inputs[3], broadcast_index((batch * heads + head + group) * keys + position,
                                                              score_view, plan.inputs[3]));
            const float updated = fmaxf(maximum[group], score);
            const float previous = maximum[group] == -INFINITY ? 0.F : __expf(maximum[group] - updated);
            const float probability = score == -INFINITY ? 0.F : __expf(score - updated);
            denominator[group] = denominator[group] * previous + probability;
#pragma unroll
            for (int column = 0; column < Vector; ++column)
                accumulated[group][column] = accumulated[group][column] * previous + probability * value_values[column];
            maximum[group] = updated;
        }
    }
#pragma unroll
    for (int group = 0; group < Group; ++group) {
        if (lane == 0) {
            maxima[warp][group] = maximum[group];
            denominators[warp][group] = denominator[group];
        }
#pragma unroll
        for (int column = 0; column < Vector; ++column)
            outputs[warp][group][lane * Vector + column] = accumulated[group][column];
    }
    __syncthreads();
    if (warp != 0) return;
#pragma unroll
    for (int group = 0; group < Group; ++group) {
        float merged_maximum = -INFINITY, merged_denominator = 0;
#pragma unroll
        for (int source = 0; source < Warps; ++source) merged_maximum = fmaxf(merged_maximum, maxima[source][group]);
        float merged[Vector]{};
#pragma unroll
        for (int source = 0; source < Warps; ++source) {
            const float factor =
                maxima[source][group] == -INFINITY ? 0.F : __expf(maxima[source][group] - merged_maximum);
            merged_denominator += denominators[source][group] * factor;
#pragma unroll
            for (int column = 0; column < Vector; ++column)
                merged[column] += outputs[source][group][lane * Vector + column] * factor;
        }
        auto* destination = partials + ((batch * heads + head + group) * partitions + blockIdx.z) * (Width + 2);
#pragma unroll
        for (int column = 0; column < Vector; ++column) destination[lane * Vector + column] = merged[column];
        if (lane == 0) {
            destination[Width] = merged_maximum;
            destination[Width + 1] = merged_denominator;
        }
    }
}

template <int Width>
__global__ void bfloat_decode_merge(Launch plan, const float* partials, std::int64_t partitions) {
    const auto* source = partials + static_cast<std::int64_t>(blockIdx.x) * partitions * (Width + 2);
    float maximum = -INFINITY, denominator = 0;
    for (std::int64_t partition = 0; partition < partitions; ++partition)
        maximum = fmaxf(maximum, source[partition * (Width + 2) + Width]);
    float result[Width / 256]{};
    for (std::int64_t partition = 0; partition < partitions; ++partition) {
        const auto* input = source + partition * (Width + 2);
        const float factor = input[Width] == -INFINITY ? 0.F : __expf(input[Width] - maximum);
        denominator += input[Width + 1] * factor;
#pragma unroll
        for (int column = 0; column < Width / 256; ++column)
            result[column] += input[threadIdx.x + column * 256] * factor;
    }
#pragma unroll
    for (int column = 0; column < Width / 256; ++column)
        static_cast<__nv_bfloat16*>(
            plan.output.data)[static_cast<std::int64_t>(blockIdx.x) * Width + threadIdx.x + column * 256] =
            __float2bfloat16(result[column] / denominator);
}

template <int Width>
__global__ void bfloat_decode_attention(Launch plan, std::int64_t heads, std::int64_t key_heads, std::int64_t keys) {
    extern __shared__ float probabilities[];
    __shared__ float partial[8];
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const auto head = blockIdx.x % heads, batch = blockIdx.x / heads;
    const auto key_head = head / (heads / key_heads);
    const auto offset = plan.attribute_count >= 3 ? plan.attributes[2] : 0;
    const auto key_batch = plan.inputs[1].shape[0] == 1 ? 0 : batch;
    const auto* query = static_cast<const __nv_bfloat16*>(plan.inputs[0].data) + (batch * heads + head) * Width;
    const auto* key = static_cast<const __nv_bfloat16*>(address(plan.inputs[1])) +
                      (key_batch * plan.inputs[1].shape[1] + offset) * key_heads * Width + key_head * Width;
    const auto* value = static_cast<const __nv_bfloat16*>(address(plan.inputs[2])) +
                        (key_batch * plan.inputs[2].shape[1] + offset) * key_heads * Width + key_head * Width;
    float activation[Width / 32];
#pragma unroll
    for (int chunk = 0; chunk < Width / 32; ++chunk) activation[chunk] = __bfloat162float(query[lane + chunk * 32]);
    const float scale = plan.epsilon > 0 ? plan.epsilon : rsqrtf(static_cast<float>(Width));
    View score_view{};
    score_view.rank = 4;
    score_view.shape[0] = plan.inputs[0].shape[0];
    score_view.shape[1] = heads;
    score_view.shape[2] = 1;
    score_view.shape[3] = keys;
    for (std::int64_t position = warp; position < keys; position += 8) {
        float score = 0;
#pragma unroll
        for (int chunk = 0; chunk < Width / 32; ++chunk)
            score =
                fmaf(activation[chunk], __bfloat162float(key[position * key_heads * Width + lane + chunk * 32]), score);
        for (int delta = 16; delta > 0; delta /= 2) score += __shfl_down_sync(0xffffffff, score, delta);
        if (lane == 0) {
            score *= scale;
            if (plan.count == 4)
                score += read(plan.inputs[3],
                              broadcast_index((batch * heads + head) * keys + position, score_view, plan.inputs[3]));
            probabilities[position] = score;
        }
    }
    __syncthreads();
    float maximum = -INFINITY;
    for (std::int64_t position = threadIdx.x; position < keys; position += blockDim.x)
        maximum = fmaxf(maximum, probabilities[position]);
    for (int delta = 16; delta > 0; delta /= 2) maximum = fmaxf(maximum, __shfl_down_sync(0xffffffff, maximum, delta));
    if (lane == 0) partial[warp] = maximum;
    __syncthreads();
    maximum = lane < 8 ? partial[lane] : -INFINITY;
    for (int delta = 16; delta > 0; delta /= 2) maximum = fmaxf(maximum, __shfl_down_sync(0xffffffff, maximum, delta));
    maximum = __shfl_sync(0xffffffff, maximum, 0);
    float denominator = 0;
    for (std::int64_t position = threadIdx.x; position < keys; position += blockDim.x) {
        const float probability = expf(probabilities[position] - maximum);
        probabilities[position] = probability;
        denominator += probability;
    }
    for (int delta = 16; delta > 0; delta /= 2) denominator += __shfl_down_sync(0xffffffff, denominator, delta);
    __syncthreads();
    if (lane == 0) partial[warp] = denominator;
    __syncthreads();
    denominator = lane < 8 ? partial[lane] : 0;
    for (int delta = 16; delta > 0; delta /= 2) denominator += __shfl_down_sync(0xffffffff, denominator, delta);
    denominator = __shfl_sync(0xffffffff, denominator, 0);
    for (int column = threadIdx.x; column < Width; column += blockDim.x) {
        float result = 0;
        for (std::int64_t position = 0; position < keys; ++position)
            result =
                fmaf(probabilities[position], __bfloat162float(value[position * key_heads * Width + column]), result);
        static_cast<__nv_bfloat16*>(plan.output.data)[(batch * heads + head) * Width + column] =
            __float2bfloat16(result / denominator);
    }
}

__global__ void matrix(Launch plan) {
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= plan.elements) return;
    const View left = plan.inputs[0], right = plan.inputs[1];
    const bool transposed = plan.attribute_count && plan.attributes[0];
    const auto columns = plan.output.shape[plan.output.rank - 1];
    const auto rows = left.shape[left.rank - 2];
    const auto width = left.shape[left.rank - 1];
    const auto row = index / columns % rows;
    const auto column = index % columns;
    auto batch = index / (rows * columns);
    std::size_t left_batch = 0, right_batch = 0, left_stride = 1, right_stride = 1;
    for (int axis = plan.output.rank - 3; axis >= 0; --axis) {
        const auto coordinate = batch % plan.output.shape[axis];
        batch /= plan.output.shape[axis];
        const int left_axis = axis - plan.output.rank + left.rank;
        const int right_axis = axis - plan.output.rank + right.rank;
        if (left_axis >= 0) {
            if (left.shape[left_axis] != 1) left_batch += coordinate * left_stride;
            left_stride *= left.shape[left_axis];
        }
        if (right_axis >= 0) {
            if (right.shape[right_axis] != 1) right_batch += coordinate * right_stride;
            right_stride *= right.shape[right_axis];
        }
    }
    float sum = 0;
    for (std::int64_t contraction = 0; contraction < width; ++contraction) {
        const auto right_index = right_batch * width * columns +
                                 (transposed ? column * width + contraction : contraction * columns + column);
        sum += read(left, left_batch * rows * width + row * width + contraction) * read(right, right_index);
    }
    if (plan.count == 3) sum += read(plan.inputs[2], column);
    write(plan.output, index, sum);
}

__global__ void bfloat_vector(Launch plan, std::int64_t width, std::int64_t columns) {
    const int lane = threadIdx.x % 32;
    const auto column = static_cast<std::int64_t>(blockIdx.x) * 4 + threadIdx.x / 32;
    if (column >= columns) return;
    const auto* input = static_cast<const __nv_bfloat16*>(plan.inputs[0].data);
    const auto* weight = static_cast<const __nv_bfloat16*>(plan.inputs[1].data) + column * width;
    float sums[8]{};
    for (std::int64_t contraction = lane * 8; contraction < width; contraction += 32 * 8) {
        union Vector {
            uint4 packed;
            __nv_bfloat162 pairs[4];
        };
        Vector left, right;
        left.packed = *reinterpret_cast<const uint4*>(input + contraction);
        right.packed = *reinterpret_cast<const uint4*>(weight + contraction);
#pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            const float2 activation = __bfloat1622float2(left.pairs[pair]);
            const float2 parameter = __bfloat1622float2(right.pairs[pair]);
            sums[pair * 2] = fmaf(activation.x, parameter.x, sums[pair * 2]);
            sums[pair * 2 + 1] = fmaf(activation.y, parameter.y, sums[pair * 2 + 1]);
        }
    }
    float sum = 0;
#pragma unroll
    for (int index = 0; index < 8; ++index) sum += sums[index];
    for (int delta = 16; delta > 0; delta /= 2) sum += __shfl_down_sync(0xffffffff, sum, delta);
    if (lane == 0) {
        if (plan.count == 3) sum += read(plan.inputs[2], column);
        static_cast<__nv_bfloat16*>(plan.output.data)[column] = __float2bfloat16(sum);
    }
}

__global__ void bfloat_vector_block(Launch plan, std::int64_t width) {
    __shared__ float partial[4];
    const auto column = static_cast<std::int64_t>(blockIdx.x);
    const auto* input = static_cast<const __nv_bfloat16*>(plan.inputs[0].data);
    const auto* weight = static_cast<const __nv_bfloat16*>(plan.inputs[1].data) + column * width;
    float sums[4]{};
    for (std::int64_t contraction = threadIdx.x * 4; contraction < width; contraction += 128 * 4) {
        union Vector {
            uint2 packed;
            __nv_bfloat162 pairs[2];
        };
        Vector left, right;
        left.packed = *reinterpret_cast<const uint2*>(input + contraction);
        right.packed = *reinterpret_cast<const uint2*>(weight + contraction);
#pragma unroll
        for (int pair = 0; pair < 2; ++pair) {
            const float2 activation = __bfloat1622float2(left.pairs[pair]);
            const float2 parameter = __bfloat1622float2(right.pairs[pair]);
            sums[pair * 2] = fmaf(activation.x, parameter.x, sums[pair * 2]);
            sums[pair * 2 + 1] = fmaf(activation.y, parameter.y, sums[pair * 2 + 1]);
        }
    }
    float sum = sums[0] + sums[1] + sums[2] + sums[3];
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    for (int delta = 16; delta > 0; delta /= 2) sum += __shfl_down_sync(0xffffffff, sum, delta);
    if (lane == 0) partial[warp] = sum;
    __syncthreads();
    if (warp == 0) {
        sum = lane < 4 ? partial[lane] : 0;
        for (int delta = 16; delta > 0; delta /= 2) sum += __shfl_down_sync(0xffffffff, sum, delta);
        if (lane == 0) {
            if (plan.count == 3) sum += read(plan.inputs[2], column);
            static_cast<__nv_bfloat16*>(plan.output.data)[column] = __float2bfloat16(sum);
        }
    }
}

void execute_matrix(Launch plan, int device) {
    const auto blocks = static_cast<unsigned int>((plan.elements + 255) / 256);
    const auto& left = plan.inputs[0];
    const auto& right = plan.inputs[1];
    const auto rows = left.shape[left.rank - 2], width = left.shape[left.rank - 1];
    const auto columns = plan.output.shape[plan.output.rank - 1];
    const bool bfloat = left.type == static_cast<int>(Type::BF16) && right.type == static_cast<int>(Type::BF16) &&
                        plan.output.type == static_cast<int>(Type::BF16);
    if (bfloat && plan.count == 2 && right.rank == 2 && plan.attribute_count && plan.attributes[0] &&
        plan.elements == static_cast<std::size_t>(columns) && width > 0 && width <= 16384 && columns <= 4096 &&
        width % 256 == 0 && reinterpret_cast<std::uintptr_t>(left.data) % 16 == 0 &&
        reinterpret_cast<std::uintptr_t>(right.data) % 16 == 0) {
        if (width >= 1024)
            bfloat_vector_block<<<static_cast<unsigned int>(columns), 128, 0, execution_stream>>>(plan, width);
        else
            bfloat_vector<<<static_cast<unsigned int>((columns + 3) / 4), 128, 0, execution_stream>>>(plan, width,
                                                                                                      columns);
        return;
    }
    if ((!bfloat && (left.type != static_cast<int>(Type::F32) || right.type != static_cast<int>(Type::F32) ||
                     plan.output.type != static_cast<int>(Type::F32))) ||
        !width) {
        matrix<<<blocks, 256, 0, execution_stream>>>(plan);
        return;
    }
    if (rows > INT32_MAX || width > INT32_MAX || columns > INT32_MAX)
        throw std::runtime_error("CUDA: matrix exceeds cuBLAS indexing limits");
    auto& owner = bfloat ? tensor_blas_handles[device] : blas_handles[device];
    if (!owner) owner = std::make_unique<BlasHandle>(device, bfloat ? CUBLAS_DEFAULT_MATH : CUBLAS_PEDANTIC_MATH);
    owner->bind_stream();
    const bool transpose = plan.attribute_count && plan.attributes[0];
    const float alpha = 1, beta = 0;
    const auto batches = plan.elements / (rows * columns);
    for (std::size_t batch_index = 0; batch_index < batches; ++batch_index) {
        auto batch = batch_index;
        std::size_t left_batch = 0, right_batch = 0, left_stride = 1, right_stride = 1;
        for (int axis = plan.output.rank - 3; axis >= 0; --axis) {
            const auto coordinate = batch % plan.output.shape[axis];
            batch /= plan.output.shape[axis];
            const int left_axis = axis - plan.output.rank + left.rank,
                      right_axis = axis - plan.output.rank + right.rank;
            if (left_axis >= 0) {
                if (left.shape[left_axis] != 1) left_batch += coordinate * left_stride;
                left_stride *= left.shape[left_axis];
            }
            if (right_axis >= 0) {
                if (right.shape[right_axis] != 1) right_batch += coordinate * right_stride;
                right_stride *= right.shape[right_axis];
            }
        }
        const auto status =
            bfloat
                ? cublasGemmEx(owner->handle, transpose ? CUBLAS_OP_T : CUBLAS_OP_N, CUBLAS_OP_N,
                               static_cast<int>(columns), static_cast<int>(rows), static_cast<int>(width), &alpha,
                               static_cast<const __nv_bfloat16*>(right.data) + right_batch * width * columns,
                               CUDA_R_16BF, static_cast<int>(transpose ? width : columns),
                               static_cast<const __nv_bfloat16*>(left.data) + left_batch * rows * width, CUDA_R_16BF,
                               static_cast<int>(width), &beta,
                               static_cast<__nv_bfloat16*>(plan.output.data) + batch_index * rows * columns,
                               CUDA_R_16BF, static_cast<int>(columns), CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT)
                : cublasSgemm(owner->handle, transpose ? CUBLAS_OP_T : CUBLAS_OP_N, CUBLAS_OP_N,
                              static_cast<int>(columns), static_cast<int>(rows), static_cast<int>(width), &alpha,
                              static_cast<const float*>(right.data) + right_batch * width * columns,
                              static_cast<int>(transpose ? width : columns),
                              static_cast<const float*>(left.data) + left_batch * rows * width, static_cast<int>(width),
                              &beta, static_cast<float*>(plan.output.data) + batch_index * rows * columns,
                              static_cast<int>(columns));
        if (status != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("CUDA: cuBLAS GEMM failed with status " + std::to_string(status));
    }
    if (plan.count == 3) {
        Launch bias = plan;
        bias.operation = static_cast<int>(Op::ADD);
        bias.inputs[0] = plan.output;
        bias.inputs[1] = plan.inputs[2];
        pointwise<<<blocks, 256, 0, execution_stream>>>(bias);
    }
}

__global__ void embedding_sum(Launch plan) {
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= plan.elements) return;
    const auto width = plan.attributes[0], bits = plan.attributes[1], groups = plan.attributes[2],
               vocabulary = plan.attributes[3];
    const auto length = plan.inputs[0].shape[1];
    const auto position = index / width, column = index % width;
    float sum = 0;
    for (std::int64_t codebook = 0; codebook < plan.inputs[0].shape[0]; ++codebook) {
        const auto token =
            static_cast<std::int64_t>(read(plan.inputs[0], codebook * length + position)) + codebook * vocabulary;
        if (token < 0 || token >= plan.inputs[1].shape[0]) continue;
        if (bits) {
            const auto offset = token * width + column;
            const int byte = static_cast<const unsigned char*>(plan.inputs[1].data)[offset / (8 / bits)];
            int weight = (byte >> (offset % (8 / bits) * bits)) & ((1 << bits) - 1);
            if (weight & (1 << (bits - 1))) weight -= 1 << bits;
            sum += weight * read(plan.inputs[2], token * groups + column / (width / groups)) * plan.epsilon;
        } else
            sum += read(plan.inputs[1], token * width + column) * plan.epsilon;
    }
    write(plan.output, index, sum);
}

__device__ float guided_logit(Launch plan, std::int64_t position, std::int64_t codebook, std::int64_t token) {
    const auto vocabulary = plan.attributes[0], codebooks = plan.attributes[2], target = plan.attributes[3];
    const auto offset = (position * codebooks + codebook) * vocabulary + token;
    const float conditioned = read(plan.inputs[0], offset);
    const float unconditional = plan.inputs[0].shape[1] == 2 * target
                                    ? read(plan.inputs[0], offset + target * codebooks * vocabulary)
                                    : conditioned;
    return conditioned + plan.epsilon * (conditioned - unconditional);
}

__global__ void diffusion_candidates(Launch plan) {
    __shared__ float shared[256];
    const auto vocabulary = plan.attributes[0], mask = plan.attributes[1], target = plan.attributes[3];
    const auto codebook = blockIdx.x / target, position = blockIdx.x % target;
    float maximum = -INFINITY, best = -INFINITY;
    for (std::int64_t token = threadIdx.x; token < vocabulary; token += blockDim.x) {
        const float value = guided_logit(plan, position, codebook, token);
        maximum = fmaxf(maximum, value);
        if (token != mask) best = fmaxf(best, value);
    }
    maximum = reduction(maximum, shared, true);
    best = reduction(best, shared, true);
    float sum = 0;
    for (std::int64_t token = threadIdx.x; token < vocabulary; token += blockDim.x)
        sum += expf(guided_logit(plan, position, codebook, token) - maximum);
    sum = reduction(sum, shared, false);
    if (threadIdx.x == 0) {
        int selected = 0;
        for (std::int64_t token = 0; token < vocabulary; ++token)
            if (token != mask && guided_logit(plan, position, codebook, token) == best) {
                selected = static_cast<int>(token);
                break;
            }
        write(plan.output, blockIdx.x * 2, static_cast<float>(selected));
        write(plan.output, blockIdx.x * 2 + 1, best - maximum - logf(sum));
    }
}

__global__ void diffusion_scores(Launch plan, float* scores) {
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= plan.elements) return;
    const auto length = plan.output.shape[1], codebooks = plan.output.shape[0];
    if (static_cast<int>(read(plan.inputs[0], index)) != plan.attributes[1]) {
        scores[index] = -INFINITY;
        return;
    }
    const auto position = index % length, codebook = index / length, order = position * codebooks + codebook;
    std::size_t noise_index = 0;
    for (std::int64_t previous = 0; previous < order; ++previous)
        if (static_cast<int>(read(plan.inputs[0], previous % codebooks * length + previous / codebooks)) ==
            plan.attributes[1])
            ++noise_index;
    const float penalty = __int_as_float(static_cast<int>(plan.attributes[2]));
    const float temperature = __int_as_float(static_cast<int>(plan.attributes[3]));
    float score = read(plan.inputs[1], index * 2 + 1) - codebook * penalty;
    if (temperature > 0) score = score / temperature + read(plan.inputs[2], noise_index);
    scores[index] = score;
}

__global__ void diffusion_update(Launch plan, const float* scores) {
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= plan.elements) return;
    int token = static_cast<int>(read(plan.inputs[0], index));
    if (token == plan.attributes[1]) {
        std::size_t rank = 0;
        for (std::size_t other = 0; other < plan.elements; ++other) {
            if (static_cast<int>(read(plan.inputs[0], other)) != plan.attributes[1]) continue;
            if (scores[other] > scores[index] || (scores[other] == scores[index] && other < index)) ++rank;
        }
        if (rank < static_cast<std::size_t>(plan.attributes[0]))
            token = static_cast<int>(read(plan.inputs[1], index * 2));
    }
    static_cast<int*>(plan.output.data)[index] = token;
}

__global__ void greedy(Launch plan) {
    __shared__ TokenCandidate partial[8];
    const auto width = plan.inputs[0].shape[plan.inputs[0].rank - 1];
    TokenCandidate candidate{-INFINITY, INT32_MAX};
    for (std::int64_t column = threadIdx.x; column < width; column += blockDim.x) {
        const float value = read(plan.inputs[0], blockIdx.x * width + column);
        if (isnan(value) || value == INFINITY)
            candidate = {INFINITY, -1};
        else if (value > candidate.value || (value == candidate.value && column < candidate.index))
            candidate = {value, static_cast<int>(column)};
    }
    candidate = reduce_candidate(candidate, partial);
    if (threadIdx.x == 0)
        static_cast<int*>(plan.output.data)[blockIdx.x] = isfinite(candidate.value) ? candidate.index : -1;
}

struct PreparedProjection {
    std::unique_ptr<void, decltype(&release)> expanded{nullptr, release};
    std::unique_ptr<void, decltype(&release)> sums{nullptr, release};
    cudaEvent_t ready{};
    int device;
    PreparedProjection(Launch launch, int bits, int selected_device) : device(selected_device) {
        const auto width = launch.inputs[0].shape[launch.inputs[0].rank - 1];
        const auto columns = launch.output.shape[launch.output.rank - 1];
        sums.reset(allocate(columns * sizeof(int), device));
        if (bits != 8) expanded.reset(allocate(columns * width, device));
        check(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming));
        prepare(launch, bits);
    }
    ~PreparedProjection() {
        int previous = 0;
        if (ready && cudaGetDevice(&previous) == cudaSuccess && cudaSetDevice(device) == cudaSuccess) {
            cudaEventDestroy(ready);
            cudaSetDevice(previous);
        }
    }
    void prepare(Launch launch, int bits) {
        const auto columns = static_cast<unsigned int>(launch.output.shape[launch.output.rank - 1]);
        if (bits == 2)
            prepare_projection_weights<2><<<columns, 256, 0, execution_stream>>>(
                launch, static_cast<signed char*>(expanded.get()), static_cast<int*>(sums.get()));
        else if (bits == 4)
            prepare_projection_weights<4><<<columns, 256, 0, execution_stream>>>(
                launch, static_cast<signed char*>(expanded.get()), static_cast<int*>(sums.get()));
        else
            prepare_projection_weights<8>
                <<<columns, 256, 0, execution_stream>>>(launch, nullptr, static_cast<int*>(sums.get()));
        check(cudaGetLastError());
        check(cudaEventRecord(ready, execution_stream));
    }
};

struct PreparedMatrix {
    std::unique_ptr<void, decltype(&release)> values{nullptr, release};
    cudaEvent_t ready{};
    int device;
    PreparedMatrix(const Tensor& weight, int selected_device) : device(selected_device) {
        values.reset(allocate(weight.elements() * sizeof(float), device));
        check(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming));
        prepare(weight);
    }
    ~PreparedMatrix() {
        int previous = 0;
        if (ready && cudaGetDevice(&previous) == cudaSuccess && cudaSetDevice(device) == cudaSuccess) {
            cudaEventDestroy(ready);
            cudaSetDevice(previous);
        }
    }
    void prepare(const Tensor& weight) {
        Launch launch{};
        launch.operation = static_cast<int>(Op::CAST);
        launch.inputs[0] = view(weight);
        launch.output = launch.inputs[0];
        launch.output.type = static_cast<int>(Type::F32);
        launch.output.data = values.get();
        launch.count = 1;
        launch.elements = weight.elements();
        if (launch.elements)
            pointwise<<<static_cast<unsigned int>((launch.elements + 255) / 256), 256, 0, execution_stream>>>(launch);
        check(cudaGetLastError());
        check(cudaEventRecord(ready, execution_stream));
    }
};

int minimum_inputs(const Descriptor& descriptor) {
    const auto operation = descriptor.operation;
    switch (operation) {
        case Op::ADD:
        case Op::MULTIPLY:
        case Op::MATMUL:
        case Op::LINEAR:
        case Op::GATHER:
        case Op::RMS_NORM:
        case Op::GELU_MULTIPLY:
        case Op::SILU_MULTIPLY:
        case Op::SNAKE:
        case Op::CONV_TRANSPOSE_FOLD:
            return 2;
        case Op::LAYER_NORM:
        case Op::RMS_NORM_RESIDUAL:
        case Op::ROTARY:
        case Op::PACKED_LINEAR:
        case Op::ATTENTION:
        case Op::SCATTER:
        case Op::DIFFUSION_UPDATE:
            return 3;
        case Op::RESIDUAL_NORM:
        case Op::RMS_ROTARY:
        case Op::QUANTIZED_LINEAR:
            return 4;
        case Op::EMBEDDING:
        case Op::EMBEDDING_SUM:
            // Quantized tables carry a scale operand.
            return descriptor.attribute_count >= 2 && descriptor.attributes[1] ? 3 : 2;
        case Op::CONV1D:
        case Op::SNAKE_CONV1D:
        case Op::SNAKE_QUANTIZED_LINEAR: {
            const int weight = operation == Op::CONV1D ? 1 : 2;
            if (descriptor.input_count <= weight) return weight + 1;
            return weight + (descriptor.inputs[weight].type == Type::I8 ? 3 : 2);
        }
        default:
            return 1;
    }
}

void validate_attention(const Descriptor& descriptor) {
    const auto& query = descriptor.inputs[0];
    const auto& key = descriptor.inputs[1];
    const auto& value = descriptor.inputs[2];
    const auto count = descriptor.attribute_count;
    const std::int64_t heads = count >= 1 ? descriptor.attributes[0] : 0;
    const std::int64_t key_heads = count >= 2 ? descriptor.attributes[1] : heads;
    const std::int64_t offset = count >= 3 ? descriptor.attributes[2] : 0;
    if (descriptor.input_count > 4 || query.rank != 3 || key.rank != 3 || value.rank != 3 || heads <= 0 ||
        key_heads <= 0 || heads % key_heads || query.shape[2] % heads ||
        key.shape[2] != key_heads * (query.shape[2] / heads) ||
        !std::equal(key.shape.begin(), key.shape.begin() + 3, value.shape.begin()) ||
        (key.shape[0] != 1 && key.shape[0] != query.shape[0]))
        throw std::runtime_error("CUDA: invalid attention geometry");
    const std::int64_t keys = count >= 4 ? descriptor.attributes[3] : key.shape[1];
    if (offset < 0 || keys < 0 || offset > key.shape[1] || keys > key.shape[1] - offset)
        throw std::runtime_error("CUDA: attention keys exceed the cache");
}

void infer(Descriptor& descriptor) {
    if (descriptor.input_count < 1 || descriptor.input_count > 8) throw std::runtime_error("CUDA: invalid input count");
    if (descriptor.attribute_count < 0 || descriptor.attribute_count > 32)
        throw std::runtime_error("CUDA: invalid attribute count");
    const auto required = minimum_inputs(descriptor);
    if (descriptor.input_count < required)
        throw std::runtime_error("CUDA: operation " + std::to_string(static_cast<int>(descriptor.operation)) +
                                 " requires at least " + std::to_string(required) + " inputs");
    for (int index = 0; index < descriptor.input_count; ++index) {
        descriptor.inputs[index].elements();
        switch (descriptor.inputs[index].type) {
            case Type::BOOL:
            case Type::U8:
            case Type::I8:
            case Type::I32:
            case Type::F16:
            case Type::BF16:
            case Type::F32:
                break;
            default:
                throw std::runtime_error("CUDA: unsupported input dtype");
        }
    }
    auto& output = descriptor.output;
    const auto requested_type = output.type;
    const auto& input = descriptor.inputs[0];
    output.rank = input.rank;
    output.shape = input.shape;
    if (descriptor.operation != Op::CAST) output.type = input.type;
    switch (descriptor.operation) {
        case Op::ADD:
        case Op::MULTIPLY: {
            const auto& other = descriptor.inputs[1];
            output.rank = std::max(input.rank, other.rank);
            for (int axis = 0; axis < output.rank; ++axis) {
                const int left_axis = axis - output.rank + input.rank, right_axis = axis - output.rank + other.rank;
                const auto left = left_axis < 0 ? 1 : input.shape[left_axis];
                const auto right = right_axis < 0 ? 1 : other.shape[right_axis];
                if (left != right && left != 1 && right != 1) throw std::runtime_error("CUDA: invalid broadcast");
                output.shape[axis] = left == 1 ? right : left;
            }
            if (requested_type == Type::BF16) output.type = Type::BF16;
            break;
        }
        case Op::CAST:
        case Op::GELU:
        case Op::TANH:
        case Op::STATIC_ROUND:
        case Op::RMS_NORM:
        case Op::RMS_NORM_RESIDUAL:
        case Op::LAYER_NORM:
        case Op::RESIDUAL_NORM:
        case Op::SOFTMAX:
        case Op::LOG_SOFTMAX:
        case Op::ROTARY:
        case Op::GELU_MULTIPLY:
        case Op::SILU_MULTIPLY:
        case Op::SNAKE:
            break;
        case Op::ATTENTION:
            validate_attention(descriptor);
            break;
        case Op::SCATTER:
            if (descriptor.input_count != 3 || input.rank != 3 || descriptor.inputs[1].rank != 3 ||
                descriptor.inputs[2].rank != 1 || descriptor.inputs[2].type != Type::I32 ||
                descriptor.inputs[1].type != input.type || descriptor.inputs[1].shape[0] != input.shape[0] ||
                descriptor.inputs[1].shape[2] != input.shape[2] ||
                descriptor.inputs[1].shape[1] != descriptor.inputs[2].shape[0])
                throw std::runtime_error("CUDA: invalid scatter geometry");
            break;
        case Op::GELU_GATE:
        case Op::SILU_GATE:
            output.shape[input.rank - 1] /= 2;
            break;
        case Op::RMS_ROTARY:
            if (descriptor.attribute_count == 5) {
                output.rank = 4;
                output.shape = {input.shape[0], descriptor.attributes[4], descriptor.attributes[0],
                                descriptor.attributes[1]};
            }
            break;
        case Op::PACKED_LINEAR:
        case Op::QUANTIZED_LINEAR:
            output.shape[input.rank - 1] = descriptor.inputs[1].shape[0];
            if (descriptor.operation == Op::QUANTIZED_LINEAR && !descriptor.attributes[0])
                output.shape[input.rank - 1] = descriptor.inputs[1].shape[1];
            output.type = Type::F32;
            break;
        case Op::SNAKE_QUANTIZED_LINEAR:
            output.shape[input.rank - 1] = descriptor.inputs[2].shape[descriptor.attributes[0] ? 0 : 1];
            output.type = Type::F32;
            break;
        case Op::CONV1D:
        case Op::SNAKE_CONV1D: {
            const auto kernel = descriptor.attributes[0], stride = descriptor.attributes[1],
                       padding = descriptor.attributes[2], dilation = descriptor.attributes[3];
            output.shape[1] = (input.shape[1] + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1;
            output.shape[2] = descriptor.inputs[descriptor.operation == Op::SNAKE_CONV1D ? 2 : 1].shape[0];
            output.type = Type::F32;
            break;
        }
        case Op::CONV_TRANSPOSE_FOLD:
            output.shape[1] = (input.shape[1] - 1) * descriptor.attributes[1] - 2 * descriptor.attributes[2] +
                              descriptor.attributes[4] * (descriptor.attributes[0] - 1) + descriptor.attributes[3] + 1;
            output.shape[2] = input.shape[2] / descriptor.attributes[0];
            output.type = Type::F32;
            break;
        case Op::TRANSPOSE:
            for (int axis = 0; axis < input.rank; ++axis) output.shape[axis] = input.shape[descriptor.attributes[axis]];
            break;
        case Op::SLICE:
        case Op::GATHER: {
            const int axis = static_cast<int>(descriptor.attributes[0] < 0 ? descriptor.attributes[0] + input.rank
                                                                           : descriptor.attributes[0]);
            if (axis < 0 || axis >= input.rank) throw std::runtime_error("CUDA: invalid layout axis");
            output.shape[axis] =
                descriptor.operation == Op::SLICE ? descriptor.attributes[2] : descriptor.inputs[1].elements();
            break;
        }
        case Op::CONCAT: {
            const int axis = static_cast<int>(descriptor.attributes[0] < 0 ? descriptor.attributes[0] + input.rank
                                                                           : descriptor.attributes[0]);
            output.shape[axis] = 0;
            for (int index = 0; index < descriptor.input_count; ++index)
                output.shape[axis] += descriptor.inputs[index].shape[axis];
            break;
        }
        case Op::EMBEDDING:
            output.rank = 3;
            output.shape = {1, static_cast<std::int64_t>(input.elements()), descriptor.attributes[0]};
            output.type = requested_type == Type::BF16 ? Type::BF16 : Type::F32;
            break;
        case Op::GREEDY_TOKEN:
            output.rank = input.rank - 1;
            output.type = Type::I32;
            break;
        case Op::EMBEDDING_SUM:
            output.rank = 3;
            output.shape = {1, input.shape[1], descriptor.attributes[0]};
            output.type = Type::F32;
            break;
        case Op::DIFFUSION_CANDIDATES:
            output.rank = 3;
            output.shape = {descriptor.attributes[2], descriptor.attributes[3], 2};
            output.type = Type::F32;
            break;
        case Op::DIFFUSION_UPDATE:
            output.type = Type::I32;
            break;
        case Op::MATMUL:
        case Op::LINEAR: {
            const auto& right = descriptor.inputs[1];
            if (input.rank < 2 || right.rank < 2) throw std::runtime_error("CUDA: matrix rank must be at least two");
            const bool transpose = descriptor.attribute_count && descriptor.attributes[0];
            output.rank = std::max(input.rank, right.rank);
            for (int axis = 0; axis < output.rank - 2; ++axis) {
                const int left_axis = axis - output.rank + input.rank, right_axis = axis - output.rank + right.rank;
                const auto left = left_axis < 0 ? 1 : input.shape[left_axis];
                const auto other = right_axis < 0 ? 1 : right.shape[right_axis];
                if (left != other && left != 1 && other != 1)
                    throw std::runtime_error("CUDA: invalid matrix batch broadcast");
                output.shape[axis] = left == 1 ? other : left;
            }
            output.shape[output.rank - 2] = input.shape[input.rank - 2];
            output.shape[output.rank - 1] = right.shape[right.rank - (transpose ? 2 : 1)];
            if (input.shape[input.rank - 1] != right.shape[right.rank - (transpose ? 1 : 2)])
                throw std::runtime_error("CUDA: matrix contraction mismatch");
            output.type = requested_type == Type::BF16 ? Type::BF16 : Type::F32;
            break;
        }
        default:
            throw std::runtime_error("CUDA: unsupported operation " +
                                     std::to_string(static_cast<int>(descriptor.operation)));
    }
    switch (output.type) {
        case Type::BOOL:
        case Type::U8:
        case Type::I8:
        case Type::I32:
        case Type::F16:
        case Type::BF16:
        case Type::F32:
            break;
        default:
            throw std::runtime_error("CUDA: unsupported output dtype");
    }
}
} // namespace

StreamScope::StreamScope(void* stream) noexcept : previous_(execution_stream) {
    execution_stream = static_cast<cudaStream_t>(stream);
}
StreamScope::~StreamScope() { execution_stream = static_cast<cudaStream_t>(previous_); }

BindingScope::BindingScope(const TensorBinding* bindings, std::size_t count) noexcept
    : previous_(execution_bindings), previous_count_(execution_binding_count) {
    execution_bindings = bindings;
    execution_binding_count = count;
}
BindingScope::~BindingScope() {
    execution_bindings = previous_;
    execution_binding_count = previous_count_;
}

class ProjectionCache {
public:
    std::shared_ptr<PreparedMatrix> matrix(const Tensor& weight, int device) {
        const auto key =
            std::tuple{device, reinterpret_cast<std::uintptr_t>(weight.data), weight.type, weight.elements()};
        std::lock_guard<std::mutex> lock(mutex_);
        auto& cached = matrices_[key];
        if (auto result = cached.lock()) return result;
        auto result = std::make_shared<PreparedMatrix>(weight, device);
        cached = result;
        return result;
    }
    std::shared_ptr<PreparedProjection> get(Launch launch, int bits, int device) {
        const auto key = std::tuple{device, reinterpret_cast<std::uintptr_t>(launch.inputs[1].data),
                                    launch.inputs[0].shape[launch.inputs[0].rank - 1],
                                    launch.output.shape[launch.output.rank - 1], bits};
        std::lock_guard<std::mutex> lock(mutex_);
        auto& cached = weights_[key];
        if (auto result = cached.lock()) return result;
        auto result = std::make_shared<PreparedProjection>(launch, bits, device);
        cached = result;
        return result;
    }

private:
    std::mutex mutex_;
    std::map<std::tuple<int, std::uintptr_t, Type, std::size_t>, std::weak_ptr<PreparedMatrix>> matrices_;
    std::map<std::tuple<int, std::uintptr_t, std::int64_t, std::int64_t, int>, std::weak_ptr<PreparedProjection>>
        weights_;
};
std::shared_ptr<ProjectionCache> make_projection_cache() { return std::make_shared<ProjectionCache>(); }

std::size_t Tensor::elements() const {
    if (rank < 0 || rank > 8) throw std::runtime_error("CUDA: unsupported tensor rank");
    std::size_t count = 1;
    for (int axis = 0; axis < rank; ++axis) {
        if (shape[axis] < 0) throw std::runtime_error("CUDA: negative tensor extent");
        if (shape[axis] && count > std::numeric_limits<std::size_t>::max() / static_cast<std::size_t>(shape[axis]))
            throw std::runtime_error("CUDA: tensor size overflow");
        count *= static_cast<std::size_t>(shape[axis]);
    }
    return count;
}
int device_count() noexcept {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess ? count : 0;
}
int current_device() {
    int device = 0;
    check(cudaGetDevice(&device));
    return device;
}
bool available(int device) noexcept { return device >= 0 && device < device_count(); }
std::string device_name(int device) {
    cudaDeviceProp properties{};
    check(cudaGetDeviceProperties(&properties, device));
    return properties.name;
}
void* allocate(std::size_t bytes, int device) {
    DeviceScope scope(device);
    void* result = nullptr;
    check(cudaMalloc(&result, std::max<std::size_t>(bytes, 1)));
    std::lock_guard<std::mutex> lock(allocations_mutex);
    allocations.emplace(result, Allocation{bytes, device});
    live_bytes += bytes;
    device_bytes[device] += bytes;
    device_peaks[device] = std::max(device_peaks[device], device_bytes[device]);
    peak_bytes = std::max(peak_bytes, live_bytes.load());
    return result;
}
void release(void* pointer) noexcept {
    if (!pointer) return;
    std::lock_guard<std::mutex> lock(allocations_mutex);
    const auto found = allocations.find(pointer);
    if (found == allocations.end()) return;
    int previous = 0;
    if (cudaGetDevice(&previous) != cudaSuccess || cudaSetDevice(found->second.device) != cudaSuccess) return;
    const auto status = cudaFree(pointer);
    cudaSetDevice(previous);
    if (status == cudaSuccess) {
        live_bytes -= found->second.bytes;
        device_bytes[found->second.device] -= found->second.bytes;
        allocations.erase(found);
    }
}
void copy_to_device(void* destination, const void* source, std::size_t bytes, int device) {
    DeviceScope scope(device);
    if (bytes) {
        check(cudaMemcpyAsync(destination, source, bytes, cudaMemcpyHostToDevice, execution_stream));
        check(cudaStreamSynchronize(execution_stream));
    }
}
void copy_to_host(void* destination, const void* source, std::size_t bytes, int device) {
    DeviceScope scope(device);
    if (bytes) {
        check(cudaMemcpyAsync(destination, source, bytes, cudaMemcpyDeviceToHost, execution_stream));
        check(cudaStreamSynchronize(execution_stream));
    }
}
void copy_device(void* destination, const void* source, std::size_t bytes, int device) {
    DeviceScope scope(device);
    if (bytes) check(cudaMemcpyAsync(destination, source, bytes, cudaMemcpyDeviceToDevice, execution_stream));
}
void copy_device_batch(void* const* destinations, const void* const* sources, const std::size_t* bytes,
                       std::size_t count, int device) {
    if (!count) return;
    DeviceScope scope(device);
    if (!execution_stream || std::any_of(bytes, bytes + count, [](std::size_t size) { return !size; })) {
        for (std::size_t index = 0; index < count; ++index)
            if (bytes[index])
                check(cudaMemcpyAsync(destinations[index], sources[index], bytes[index], cudaMemcpyDeviceToDevice,
                                      execution_stream));
        return;
    }
    cudaMemcpyAttributes attributes{};
    attributes.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
    std::size_t first = 0;
    check(cudaMemcpyBatchAsync(destinations, sources, bytes, count, &attributes, &first, 1, execution_stream));
}
void clear(void* pointer, std::size_t bytes, int device) {
    DeviceScope scope(device);
    if (bytes) check(cudaMemsetAsync(pointer, 0, bytes, execution_stream));
}
void synchronize(int device) {
    DeviceScope scope(device);
    check(cudaDeviceSynchronize());
}
void start_profiler() { check(cudaProfilerStart()); }
void stop_profiler() { check(cudaProfilerStop()); }
std::size_t allocated_bytes(int device) noexcept {
    if (device < 0) return live_bytes;
    std::lock_guard<std::mutex> lock(allocations_mutex);
    std::size_t bytes = 0;
    for (const auto& allocation : allocations)
        if (allocation.second.device == device) bytes += allocation.second.bytes;
    return bytes;
}
std::size_t peak_allocated_bytes(int device) noexcept {
    std::lock_guard<std::mutex> lock(allocations_mutex);
    if (device < 0) return peak_bytes;
    const auto found = device_peaks.find(device);
    return found == device_peaks.end() ? 0 : found->second;
}

struct Program::Impl {
    Descriptor descriptor;
    int device;
    std::unique_ptr<void, decltype(&release)> scratch{nullptr, release};
    std::unique_ptr<void, decltype(&release)> scales{nullptr, release};
    std::unique_ptr<void, decltype(&release)> zeros{nullptr, release};
    std::unique_ptr<void, decltype(&release)> columns{nullptr, release};
    std::unique_ptr<void, decltype(&release)> accumulators{nullptr, release};
    std::unique_ptr<void, decltype(&release)> probabilities{nullptr, release};
    std::unique_ptr<void, decltype(&release)> attention_partials{nullptr, release};
    std::unique_ptr<void, decltype(&release)> token_candidates{nullptr, release};
    std::shared_ptr<PreparedProjection> prepared;
    std::shared_ptr<PreparedMatrix> matrix_weight;
    std::unique_ptr<void, decltype(&release)> matrix_input{nullptr, release};
    const void* matrix_source = nullptr;
    const void* prepared_weight = nullptr;
    bool tensor_projection = false;
    bool vector_projection = false;
    explicit Impl(Descriptor value, int selected_device) : descriptor(value), device(selected_device) {
        DeviceScope scope(device);
        infer(descriptor);
        descriptor.output.device = device;
        const auto operation = descriptor.operation;
        const auto& input = descriptor.inputs[0];
        if (!descriptor.reference_projection && descriptor.output.type != Type::BF16 &&
            (operation == Op::LINEAR || operation == Op::MATMUL) &&
            (input.type == Type::BF16 || input.type == Type::F16))
            matrix_input.reset(allocate(input.elements() * sizeof(float), device));
        const bool convolution = operation == Op::CONV1D || operation == Op::SNAKE_CONV1D;
        const bool snake_linear = operation == Op::SNAKE_QUANTIZED_LINEAR;
        const int weight_index = operation == Op::SNAKE_CONV1D || snake_linear ? 2 : 1;
        const auto last_width = input.rank ? input.shape[input.rank - 1] : 0;
        const auto rows = convolution  ? descriptor.output.shape[1]
                          : last_width ? static_cast<std::int64_t>(input.elements() / last_width)
                                       : 0;
        const auto width = convolution ? input.shape[2] * descriptor.attributes[0] : last_width;
        if (convolution || snake_linear) columns.reset(allocate(rows * width * sizeof(float), device));
        if (operation == Op::PACKED_LINEAR || operation == Op::QUANTIZED_LINEAR ||
            ((convolution || snake_linear) && descriptor.inputs[weight_index].type == Type::I8)) {
            scratch.reset(allocate(rows * width, device));
            scales.reset(allocate(rows * sizeof(float), device));
            zeros.reset(allocate(rows * sizeof(int), device));
            const auto output_columns = descriptor.output.shape[descriptor.output.rank - 1];
            const bool transposed =
                convolution || snake_linear || (operation == Op::QUANTIZED_LINEAR && descriptor.attributes[0]);
            const bool supported_weights =
                operation == Op::QUANTIZED_LINEAR || convolution || snake_linear ||
                (operation == Op::PACKED_LINEAR &&
                 (descriptor.attributes[0] == 2 || descriptor.attributes[0] == 4 || descriptor.attributes[0] == 8) &&
                 descriptor.attributes[1] == width);
            cudaDeviceProp properties{};
            check(cudaGetDeviceProperties(&properties, device));
            const auto bits = operation == Op::PACKED_LINEAR ? descriptor.attributes[0] : 8;
            vector_projection = !descriptor.reference_projection && properties.major >= 7 &&
                                (transposed || operation == Op::PACKED_LINEAR) &&
                                (bits == 2 || bits == 4 || bits == 8) &&
                                (operation != Op::PACKED_LINEAR || descriptor.attributes[1] == width) && rows > 0 &&
                                rows <= 4 && width > 0 && width <= 65536 && width % (32 / bits) == 0 &&
                                output_columns > 0 && output_columns <= INT32_MAX;
            tensor_projection = !descriptor.reference_projection && properties.major >= 7 &&
                                (transposed || operation == Op::PACKED_LINEAR) && supported_weights && width > 0 &&
                                width <= 65536 && width % 4 == 0 && output_columns > 0 && output_columns <= INT32_MAX &&
                                output_columns % 4 == 0 && rows > 0 && rows <= INT32_MAX;
            if (tensor_projection && !vector_projection) {
                accumulators.reset(allocate(descriptor.output.elements() * sizeof(int), device));
            }
        } else if (operation == Op::DIFFUSION_UPDATE) {
            scratch.reset(allocate(input.elements() * sizeof(float), device));
        } else if (operation == Op::GREEDY_TOKEN && input.shape[input.rank - 1] > 4096) {
            token_candidates.reset(allocate(
                descriptor.output.elements() * ((input.shape[input.rank - 1] + 2047) / 2048) * sizeof(TokenCandidate),
                device));
        } else if (operation == Op::ATTENTION) {
            if ((descriptor.inputs[1].type != Type::F32 && descriptor.inputs[1].type != Type::BF16) ||
                descriptor.inputs[2].type != descriptor.inputs[1].type)
                throw std::runtime_error("CUDA: quantized KV attention not yet supported");
            const auto keys =
                descriptor.attribute_count >= 4 ? descriptor.attributes[3] : descriptor.inputs[1].shape[1];
            scratch.reset(
                allocate(input.shape[0] * descriptor.attributes[0] * input.shape[1] * keys * sizeof(float), device));
            if (!descriptor.reference_attention && input.type == Type::BF16 &&
                descriptor.inputs[1].type == Type::BF16 && descriptor.output.type == Type::BF16 && input.shape[1] > 1)
                probabilities.reset(allocate(
                    input.shape[0] * descriptor.attributes[0] * input.shape[1] * keys * sizeof(__nv_bfloat16), device));
            const auto head_width = input.shape[2] / descriptor.attributes[0];
            if (!descriptor.reference_attention && input.type == Type::BF16 &&
                descriptor.inputs[1].type == Type::BF16 && input.shape[1] == 1 &&
                (head_width == 256 || head_width == 512))
                attention_partials.reset(allocate(
                    input.shape[0] * descriptor.attributes[0] * ((keys + 63) / 64) * (head_width + 2) * sizeof(float),
                    device));
        }
    }
    void floating(Launch launch, const Tensor& weight) {
        if (launch.output.type == static_cast<int>(Type::BF16)) {
            if (descriptor.reference_projection || launch.count == 3)
                matrix<<<static_cast<unsigned int>((launch.elements + 255) / 256), 256, 0, execution_stream>>>(launch);
            else
                execute_matrix(launch, device);
            return;
        }
        if (matrix_input) {
            Launch converted{};
            converted.operation = static_cast<int>(Op::CAST);
            converted.inputs[0] = launch.inputs[0];
            converted.output = launch.inputs[0];
            converted.output.type = static_cast<int>(Type::F32);
            converted.output.data = matrix_input.get();
            converted.count = 1;
            converted.elements = descriptor.inputs[0].elements();
            if (converted.elements)
                pointwise<<<static_cast<unsigned int>((converted.elements + 255) / 256), 256, 0, execution_stream>>>(
                    converted);
            launch.inputs[0] = converted.output;
        }
        if (!descriptor.reference_projection && launch.inputs[0].type == static_cast<int>(Type::F32) &&
            (weight.type == Type::BF16 || weight.type == Type::F16)) {
            if (!descriptor.immutable_weights || matrix_source != weight.data) {
                if (descriptor.immutable_weights && descriptor.projection_cache)
                    matrix_weight = descriptor.projection_cache->matrix(weight, device);
                else if (!matrix_weight || matrix_source != weight.data)
                    matrix_weight = std::make_shared<PreparedMatrix>(weight, device);
                else
                    matrix_weight->prepare(weight);
                check(cudaStreamWaitEvent(execution_stream, matrix_weight->ready));
                matrix_source = weight.data;
            }
            launch.inputs[1].type = static_cast<int>(Type::F32);
            launch.inputs[1].data = matrix_weight->values.get();
        }
        execute_matrix(launch, device);
    }
    void project(Launch launch, unsigned int blocks) {
        const auto output_columns = launch.output.shape[launch.output.rank - 1];
        const auto rows = launch.elements / output_columns;
        if (vector_projection && reinterpret_cast<std::uintptr_t>(launch.inputs[1].data) % 4 == 0) {
            const dim3 grid(static_cast<unsigned int>((output_columns + 3) / 4), static_cast<unsigned int>(rows));
            const auto bits = launch.operation == static_cast<int>(Op::PACKED_LINEAR) ? launch.attributes[0] : 8;
            if (bits == 2)
                quantized_vector<2><<<grid, 128, 0, execution_stream>>>(
                    launch, static_cast<signed char*>(scratch.get()), static_cast<float*>(scales.get()),
                    static_cast<int*>(zeros.get()));
            else if (bits == 4)
                quantized_vector<4><<<grid, 128, 0, execution_stream>>>(
                    launch, static_cast<signed char*>(scratch.get()), static_cast<float*>(scales.get()),
                    static_cast<int*>(zeros.get()));
            else
                quantized_vector<8><<<grid, 128, 0, execution_stream>>>(
                    launch, static_cast<signed char*>(scratch.get()), static_cast<float*>(scales.get()),
                    static_cast<int*>(zeros.get()));
            return;
        }
        if (!tensor_projection || vector_projection || reinterpret_cast<std::uintptr_t>(launch.inputs[1].data) % 4) {
            quantized_matrix<<<blocks, 256, 0, execution_stream>>>(launch, static_cast<signed char*>(scratch.get()),
                                                                   static_cast<float*>(scales.get()),
                                                                   static_cast<int*>(zeros.get()));
            return;
        }
        const auto width = launch.inputs[0].shape[launch.inputs[0].rank - 1];
        const auto bits = launch.operation == static_cast<int>(Op::PACKED_LINEAR) ? launch.attributes[0] : 8;
        if (!descriptor.immutable_weights || prepared_weight != launch.inputs[1].data) {
            if (descriptor.immutable_weights && descriptor.projection_cache)
                prepared = descriptor.projection_cache->get(launch, bits, device);
            else if (!prepared || prepared_weight != launch.inputs[1].data)
                prepared = std::make_shared<PreparedProjection>(launch, bits, device);
            else
                prepared->prepare(launch, bits);
            check(cudaStreamWaitEvent(execution_stream, prepared->ready));
            prepared_weight = launch.inputs[1].data;
        }
        auto& owner = tensor_blas_handles[device];
        if (!owner) owner = std::make_unique<BlasHandle>(device, CUBLAS_DEFAULT_MATH);
        owner->bind_stream();
        const int alpha = 1, beta = 0;
        const auto status = cublasGemmEx(
            owner->handle, CUBLAS_OP_T, CUBLAS_OP_N, static_cast<int>(output_columns), static_cast<int>(rows),
            static_cast<int>(width), &alpha, prepared->expanded ? prepared->expanded.get() : launch.inputs[1].data,
            CUDA_R_8I, static_cast<int>(width), scratch.get(), CUDA_R_8I, static_cast<int>(width), &beta,
            accumulators.get(), CUDA_R_32I, static_cast<int>(output_columns), CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
        if (status != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("CUDA: integer cuBLAS GEMM failed with status " + std::to_string(status));
        projection_epilogue<<<blocks, 256, 0, execution_stream>>>(
            launch, static_cast<int*>(accumulators.get()), static_cast<int*>(prepared->sums.get()),
            static_cast<float*>(scales.get()), static_cast<int*>(zeros.get()));
    }
};
Program::Program(Descriptor descriptor, int device) : impl_(std::make_unique<Impl>(descriptor, device)) {}
Program::~Program() = default;
const Tensor& Program::output() const { return impl_->descriptor.output; }
void Program::run(const Tensor* inputs, int count, void* output) {
    DeviceScope scope(impl_->device);
    const auto& descriptor = impl_->descriptor;
    if (count != descriptor.input_count) throw std::runtime_error("CUDA: input count changed");
    Launch launch{};
    launch.operation = static_cast<int>(descriptor.operation);
    launch.count = count;
    launch.attribute_count = descriptor.attribute_count;
    launch.epsilon = descriptor.epsilon;
    std::copy(descriptor.attributes.begin(), descriptor.attributes.end(), launch.attributes);
    for (int index = 0; index < count; ++index) {
        if (inputs[index].device != impl_->device)
            throw std::runtime_error("CUDA: operands belong to different devices");
        const auto& expected = descriptor.inputs[index];
        if (inputs[index].rank != expected.rank || inputs[index].type != expected.type ||
            !std::equal(inputs[index].shape.begin(), inputs[index].shape.begin() + inputs[index].rank,
                        expected.shape.begin()))
            throw std::runtime_error("CUDA: operand shape or dtype changed after preparation");
        if (!inputs[index].data && inputs[index].elements()) throw std::runtime_error("CUDA: null input pointer");
        launch.inputs[index] = view(inputs[index]);
        bind(launch.inputs[index]);
    }
    launch.output = view(descriptor.output);
    launch.output.data = output;
    bind(launch.output);
    launch.elements = descriptor.output.elements();
    if (!launch.elements) return;
    if (!output) throw std::runtime_error("CUDA: null output pointer");
    const auto operation = descriptor.operation;
    const bool relocated =
        launch.output.binding ||
        std::any_of(launch.inputs, launch.inputs + count, [](const View& tensor) { return tensor.binding != nullptr; });
    if (relocated && operation != Op::SCATTER && !(operation == Op::ATTENTION && descriptor.inputs[0].shape[1] == 1))
        throw std::runtime_error("CUDA: relocated operands are only supported by scatter and single-query attention");
    const auto blocks = static_cast<unsigned int>((launch.elements + 255) / 256);
    if (operation == Op::CONV1D || operation == Op::SNAKE_CONV1D || operation == Op::SNAKE_QUANTIZED_LINEAR) {
        const bool convolution = operation != Op::SNAKE_QUANTIZED_LINEAR;
        const bool snake = operation != Op::CONV1D;
        const auto rows = convolution ? launch.output.shape[1]
                                      : static_cast<std::int64_t>(descriptor.inputs[0].elements() /
                                                                  launch.inputs[0].shape[launch.inputs[0].rank - 1]);
        const auto width = convolution ? launch.inputs[0].shape[2] * launch.attributes[0]
                                       : launch.inputs[0].shape[launch.inputs[0].rank - 1];
        if (convolution)
            unfold_convolution<<<static_cast<unsigned int>((rows * width + 255) / 256), 256, 0, execution_stream>>>(
                launch, static_cast<float*>(impl_->columns.get()), rows * width);
        else {
            Launch activated = launch;
            activated.operation = static_cast<int>(Op::SNAKE);
            activated.elements = descriptor.inputs[0].elements();
            activated.output = launch.inputs[0];
            activated.output.data = impl_->columns.get();
            pointwise<<<static_cast<unsigned int>((activated.elements + 255) / 256), 256, 0, execution_stream>>>(
                activated);
        }
        const int weight = snake ? 2 : 1;
        const bool quantized = descriptor.inputs[weight].type == Type::I8;
        Launch projected = launch;
        projected.inputs[0] = {impl_->columns.get(), static_cast<int>(Type::F32), 3, {1, rows, width}};
        projected.inputs[1] = launch.inputs[weight];
        projected.count = quantized ? 4 : 3;
        projected.inputs[2] = launch.inputs[weight + 1];
        if (quantized) projected.inputs[3] = launch.inputs[weight + 2];
        projected.attribute_count = 1;
        projected.attributes[0] = 1;
        projected.operation = static_cast<int>(quantized ? Op::QUANTIZED_LINEAR : Op::LINEAR);
        if (quantized) {
            quantize_rows<<<static_cast<unsigned int>(rows), 256, 0, execution_stream>>>(
                projected, static_cast<signed char*>(impl_->scratch.get()), static_cast<float*>(impl_->scales.get()),
                static_cast<int*>(impl_->zeros.get()));
            impl_->project(projected, blocks);
        } else
            execute_matrix(projected, impl_->device);
    } else if (operation == Op::CONV_TRANSPOSE_FOLD)
        fold_convolution<<<blocks, 256, 0, execution_stream>>>(launch);
    else if (operation == Op::MATMUL || operation == Op::LINEAR)
        impl_->floating(launch, inputs[1]);
    else if (operation == Op::PACKED_LINEAR || operation == Op::QUANTIZED_LINEAR) {
        const auto rows = descriptor.inputs[0].elements() / descriptor.inputs[0].shape[descriptor.inputs[0].rank - 1];
        quantize_rows<<<static_cast<unsigned int>(rows), 256, 0, execution_stream>>>(
            launch, static_cast<signed char*>(impl_->scratch.get()), static_cast<float*>(impl_->scales.get()),
            static_cast<int*>(impl_->zeros.get()));
        impl_->project(launch, blocks);
    } else if (operation == Op::RMS_NORM || operation == Op::RMS_NORM_RESIDUAL || operation == Op::RMS_ROTARY ||
               operation == Op::LAYER_NORM || operation == Op::RESIDUAL_NORM || operation == Op::SOFTMAX ||
               operation == Op::LOG_SOFTMAX) {
        const auto rows = static_cast<unsigned int>(launch.elements / launch.output.shape[launch.output.rank - 1]);
        const bool bfloat = launch.inputs[0].type == static_cast<int>(Type::BF16) &&
                            launch.output.type == static_cast<int>(Type::BF16) &&
                            launch.output.shape[launch.output.rank - 1] <= 2048 && !launch.attribute_count;
        if (bfloat && operation == Op::RMS_NORM)
            bfloat_normalize<Op::RMS_NORM><<<rows, 256, 0, execution_stream>>>(launch);
        else if (bfloat && operation == Op::RMS_NORM_RESIDUAL)
            bfloat_normalize<Op::RMS_NORM_RESIDUAL><<<rows, 256, 0, execution_stream>>>(launch);
        else if (bfloat && operation == Op::RMS_ROTARY)
            bfloat_normalize<Op::RMS_ROTARY><<<rows, 256, 0, execution_stream>>>(launch);
        else if (launch.output.type == static_cast<int>(Type::BF16))
            normalize<true><<<rows, 256, 0, execution_stream>>>(launch);
        else
            normalize<false><<<rows, 256, 0, execution_stream>>>(launch);
    } else if (operation == Op::SCATTER && output == inputs[0].data) {
        const auto bytes = [](Type type) -> std::size_t {
            return type == Type::F32 || type == Type::I32 ? 4 : type == Type::F16 || type == Type::BF16 ? 2 : 1;
        };
        const auto begin = reinterpret_cast<std::uintptr_t>(output);
        const auto end = begin + descriptor.output.elements() * bytes(descriptor.output.type);
        for (int index = 1; index < 3; ++index) {
            const auto source = reinterpret_cast<std::uintptr_t>(inputs[index].data);
            if (source < end && begin < source + inputs[index].elements() * bytes(inputs[index].type))
                throw std::runtime_error("CUDA: in-place scatter requires nonoverlapping updates and indices");
        }
        launch.elements = inputs[1].elements();
        if (launch.elements)
            scatter_updates<<<static_cast<unsigned int>((launch.elements + 255) / 256), 256, 0, execution_stream>>>(
                launch);
    } else if (operation == Op::SLICE) {
        const auto& source = inputs[0];
        const auto axis =
            descriptor.attributes[0] < 0 ? descriptor.attributes[0] + source.rank : descriptor.attributes[0];
        std::size_t inner = 1, outer = 1;
        for (int dimension = 0; dimension < axis; ++dimension) outer *= source.shape[dimension];
        for (int dimension = static_cast<int>(axis) + 1; dimension < source.rank; ++dimension)
            inner *= source.shape[dimension];
        const auto type = source.type;
        const std::size_t bytes = type == Type::F64 || type == Type::I64 || type == Type::U64   ? 8
                                  : type == Type::F32 || type == Type::I32 || type == Type::U32 ? 4
                                  : type == Type::F16 || type == Type::BF16 || type == Type::I16 || type == Type::U16
                                      ? 2
                                      : 1;
        const auto pitch = source.shape[axis] * inner * bytes;
        const auto width = descriptor.output.shape[axis] * inner * bytes;
        const auto* source_bytes =
            static_cast<const unsigned char*>(source.data) + descriptor.attributes[1] * inner * bytes;
        check(cudaMemcpy2DAsync(output, width, source_bytes, pitch, width, outer, cudaMemcpyDeviceToDevice,
                                execution_stream));
    } else if (operation == Op::TRANSPOSE || operation == Op::GATHER || operation == Op::CONCAT ||
               operation == Op::SCATTER)
        layout<<<blocks, 256, 0, execution_stream>>>(launch);
    else if (operation == Op::GELU_GATE || operation == Op::SILU_GATE)
        gated_activation<<<blocks, 256, 0, execution_stream>>>(launch);
    else if (operation == Op::EMBEDDING)
        embedding<<<blocks, 256, 0, execution_stream>>>(launch);
    else if (operation == Op::GREEDY_TOKEN) {
        if (impl_->token_candidates) {
            const auto partitions = (launch.inputs[0].shape[launch.inputs[0].rank - 1] + 2047) / 2048;
            greedy_partitions<<<static_cast<unsigned int>(launch.elements * partitions), 256, 0, execution_stream>>>(
                launch, static_cast<TokenCandidate*>(impl_->token_candidates.get()), partitions);
            greedy_merge<<<static_cast<unsigned int>(launch.elements), 256, 0, execution_stream>>>(
                launch, static_cast<const TokenCandidate*>(impl_->token_candidates.get()), partitions);
        } else {
            greedy<<<static_cast<unsigned int>(launch.elements), 256, 0, execution_stream>>>(launch);
        }
    } else if (operation == Op::EMBEDDING_SUM)
        embedding_sum<<<blocks, 256, 0, execution_stream>>>(launch);
    else if (operation == Op::DIFFUSION_CANDIDATES)
        diffusion_candidates<<<static_cast<unsigned int>(launch.elements / 2), 256, 0, execution_stream>>>(launch);
    else if (operation == Op::DIFFUSION_UPDATE) {
        diffusion_scores<<<blocks, 256, 0, execution_stream>>>(launch, static_cast<float*>(impl_->scratch.get()));
        diffusion_update<<<blocks, 256, 0, execution_stream>>>(launch, static_cast<float*>(impl_->scratch.get()));
    } else if (operation == Op::ATTENTION) {
        const auto heads = descriptor.attributes[0],
                   key_heads = descriptor.attribute_count >= 2 ? descriptor.attributes[1] : heads;
        const auto keys = descriptor.attribute_count >= 4 ? descriptor.attributes[3] : descriptor.inputs[1].shape[1];
        const auto width = descriptor.inputs[0].shape[2] / heads;
        if (impl_->attention_partials) {
            const auto partitions = (keys + 63) / 64;
            const int group = (heads / key_heads) % 2 == 0 ? 2 : 1;
            const dim3 grid(static_cast<unsigned int>(descriptor.inputs[0].shape[0]),
                            static_cast<unsigned int>(heads / group), static_cast<unsigned int>(partitions));
            auto* partials = static_cast<float*>(impl_->attention_partials.get());
            if (width == 256) {
                if (group == 2)
                    bfloat_decode_partitions<256, 2>
                        <<<grid, 128, 0, execution_stream>>>(launch, partials, heads, key_heads, keys, partitions);
                else
                    bfloat_decode_partitions<256, 1>
                        <<<grid, 128, 0, execution_stream>>>(launch, partials, heads, key_heads, keys, partitions);
                bfloat_decode_merge<256>
                    <<<static_cast<unsigned int>(descriptor.inputs[0].shape[0] * heads), 256, 0, execution_stream>>>(
                        launch, partials, partitions);
            } else {
                if (group == 2)
                    bfloat_decode_partitions<512, 2>
                        <<<grid, 128, 0, execution_stream>>>(launch, partials, heads, key_heads, keys, partitions);
                else
                    bfloat_decode_partitions<512, 1>
                        <<<grid, 128, 0, execution_stream>>>(launch, partials, heads, key_heads, keys, partitions);
                bfloat_decode_merge<512>
                    <<<static_cast<unsigned int>(descriptor.inputs[0].shape[0] * heads), 256, 0, execution_stream>>>(
                        launch, partials, partitions);
            }
            check(cudaGetLastError());
            return;
        }
        if (!descriptor.reference_attention && descriptor.inputs[0].type == Type::BF16 &&
            descriptor.inputs[1].type == Type::BF16 && descriptor.inputs[0].shape[1] == 1 && keys > 0 && keys <= 8192 &&
            (width == 32 || width == 64 || width == 128 || width == 256 || width == 512)) {
            const auto rows = static_cast<unsigned int>(descriptor.inputs[0].shape[0] * heads);
            if (width == 32)
                bfloat_decode_attention<32>
                    <<<rows, 256, keys * sizeof(float), execution_stream>>>(launch, heads, key_heads, keys);
            if (width == 64)
                bfloat_decode_attention<64>
                    <<<rows, 256, keys * sizeof(float), execution_stream>>>(launch, heads, key_heads, keys);
            if (width == 128)
                bfloat_decode_attention<128>
                    <<<rows, 256, keys * sizeof(float), execution_stream>>>(launch, heads, key_heads, keys);
            if (width == 256)
                bfloat_decode_attention<256>
                    <<<rows, 256, keys * sizeof(float), execution_stream>>>(launch, heads, key_heads, keys);
            if (width == 512)
                bfloat_decode_attention<512>
                    <<<rows, 256, keys * sizeof(float), execution_stream>>>(launch, heads, key_heads, keys);
            check(cudaGetLastError());
            return;
        }
        const auto score_count = descriptor.inputs[0].shape[0] * heads * descriptor.inputs[0].shape[1] * keys;
        execute_attention_scores(launch, static_cast<float*>(impl_->scratch.get()), impl_->device, heads, key_heads,
                                 keys, descriptor.reference_attention);
        Launch softmax{};
        softmax.operation = static_cast<int>(Op::SOFTMAX);
        softmax.inputs[0] =
            softmax.output = {impl_->scratch.get(),
                              static_cast<int>(Type::F32),
                              4,
                              {descriptor.inputs[0].shape[0], heads, descriptor.inputs[0].shape[1], keys}};
        normalize<false><<<static_cast<unsigned int>(score_count / keys), 256, 0, execution_stream>>>(softmax);
        if (impl_->probabilities) {
            bfloat_probabilities<<<static_cast<unsigned int>((score_count + 255) / 256), 256, 0, execution_stream>>>(
                static_cast<float*>(impl_->scratch.get()), static_cast<__nv_bfloat16*>(impl_->probabilities.get()),
                score_count);
            execute_attention_values(launch, static_cast<const __nv_bfloat16*>(impl_->probabilities.get()),
                                     impl_->device, heads, key_heads, keys);
        } else {
            attention_values<<<blocks, 256, 0, execution_stream>>>(launch, static_cast<float*>(impl_->scratch.get()),
                                                                   heads, key_heads, keys);
        }
    } else
        pointwise<<<blocks, 256, 0, execution_stream>>>(launch);
    check(cudaGetLastError());
}
} // namespace kidi::cuda