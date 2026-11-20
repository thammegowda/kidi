#include "cuda/elementwise.h"

#include <cmath>

namespace kidi::cuda {
namespace {
namespace kernel {

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

constexpr int rms_values_per_thread = 10;

template <typename Element, Op Operation>
__global__ void dense_rms_normalize(Launch plan) {
    __shared__ float partial[8];
    const auto width = plan.output.shape[plan.output.rank - 1], row = static_cast<std::int64_t>(blockIdx.x);
    const auto* input = static_cast<const Element*>(plan.inputs[0].data) + row * width;
    const auto* scale = static_cast<const float*>(plan.inputs[1].data);
    float values[rms_values_per_thread]{}, sum = 0;
#pragma unroll
    for (int chunk = 0; chunk < rms_values_per_thread; ++chunk) {
        const auto column = threadIdx.x + chunk * 256;
        if (column < width) {
            values[chunk] = static_cast<float>(input[column]);
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
    for (int chunk = 0; chunk < rms_values_per_thread; ++chunk) {
        const auto column = threadIdx.x + chunk * 256;
        if (column >= width) continue;
        float result = values[chunk] * inverse * scale[column];
        if constexpr (Operation == Op::RMS_NORM_RESIDUAL) {
            result = static_cast<float>(static_cast<Element>(result)) +
                     static_cast<float>(static_cast<const Element*>(plan.inputs[2].data)[row * width + column]);
            if (plan.count == 4)
                result = static_cast<float>(static_cast<Element>(result)) *
                         static_cast<const float*>(plan.inputs[3].data)[0];
        }
        if constexpr (Operation == Op::RMS_ROTARY) {
            const auto half = width / 2, partner = (column + half) % width;
            const float paired =
                static_cast<float>(static_cast<Element>(static_cast<float>(input[partner]) * inverse * scale[partner]));
            const auto sequence = row / plan.output.shape[2] % plan.output.shape[1];
            const auto angle = sequence * half + column % half;
            result = static_cast<float>(static_cast<Element>(result)) *
                         static_cast<const float*>(plan.inputs[2].data)[angle] +
                     (column < half ? -paired : paired) * static_cast<const float*>(plan.inputs[3].data)[angle];
        }
        static_cast<Element*>(plan.output.data)[row * width + column] = static_cast<Element>(result);
    }
}

template <typename Element>
bool dense_rms(const Launch& plan, Op operation, unsigned int rows, cudaStream_t stream) {
    const auto floating = static_cast<int>(Type::F32);
    if (plan.inputs[0].type != plan.output.type || plan.attribute_count || plan.inputs[1].type != floating ||
        plan.output.shape[plan.output.rank - 1] > 256 * rms_values_per_thread)
        return false;
    if (operation == Op::RMS_NORM) {
        dense_rms_normalize<Element, Op::RMS_NORM><<<rows, 256, 0, stream>>>(plan);
    } else if (operation == Op::RMS_NORM_RESIDUAL && plan.inputs[2].type == plan.output.type &&
               (plan.count == 3 || plan.inputs[3].type == floating)) {
        dense_rms_normalize<Element, Op::RMS_NORM_RESIDUAL><<<rows, 256, 0, stream>>>(plan);
    } else if (operation == Op::RMS_ROTARY && plan.inputs[2].type == floating && plan.inputs[3].type == floating) {
        dense_rms_normalize<Element, Op::RMS_ROTARY><<<rows, 256, 0, stream>>>(plan);
    } else {
        return false;
    }
    return true;
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

} // namespace kernel
} // namespace

auto token_candidate_bytes() -> std::size_t { return sizeof(kernel::TokenCandidate); }

auto pointwise(const Launch& plan, cudaStream_t stream) -> void {
    const auto blocks = static_cast<unsigned int>((plan.elements + 255) / 256);
    kernel::pointwise<<<blocks, 256, 0, stream>>>(plan);
}

auto gated_activation(const Launch& plan, cudaStream_t stream) -> void {
    const auto blocks = static_cast<unsigned int>((plan.elements + 255) / 256);
    kernel::gated_activation<<<blocks, 256, 0, stream>>>(plan);
}

auto unfold_convolution(const Launch& plan, float* columns, std::size_t elements, cudaStream_t stream) -> void {
    kernel::unfold_convolution<<<static_cast<unsigned int>((elements + 255) / 256), 256, 0, stream>>>(plan, columns,
                                                                                                      elements);
}

auto fold_convolution(const Launch& plan, cudaStream_t stream) -> void {
    const auto blocks = static_cast<unsigned int>((plan.elements + 255) / 256);
    kernel::fold_convolution<<<blocks, 256, 0, stream>>>(plan);
}

auto normalize(const Launch& plan, Op operation, cudaStream_t stream) -> void {
    const auto rows = static_cast<unsigned int>(plan.elements / plan.output.shape[plan.output.rank - 1]);
    if (plan.output.type == static_cast<int>(Type::BF16)) {
        if (!kernel::dense_rms<__nv_bfloat16>(plan, operation, rows, stream))
            kernel::normalize<true><<<rows, 256, 0, stream>>>(plan);
    } else if (plan.output.type != static_cast<int>(Type::F32) ||
               !kernel::dense_rms<float>(plan, operation, rows, stream)) {
        kernel::normalize<false><<<rows, 256, 0, stream>>>(plan);
    }
}

auto greedy_token(const Launch& plan, void* candidates, cudaStream_t stream) -> void {
    if (candidates) {
        const auto partitions = (plan.inputs[0].shape[plan.inputs[0].rank - 1] + 2047) / 2048;
        kernel::greedy_partitions<<<static_cast<unsigned int>(plan.elements * partitions), 256, 0, stream>>>(
            plan, static_cast<kernel::TokenCandidate*>(candidates), partitions);
        kernel::greedy_merge<<<static_cast<unsigned int>(plan.elements), 256, 0, stream>>>(
            plan, static_cast<const kernel::TokenCandidate*>(candidates), partitions);
    } else {
        kernel::greedy<<<static_cast<unsigned int>(plan.elements), 256, 0, stream>>>(plan);
    }
}

auto layout(const Launch& plan, cudaStream_t stream) -> void {
    const auto blocks = static_cast<unsigned int>((plan.elements + 255) / 256);
    kernel::layout<<<blocks, 256, 0, stream>>>(plan);
}

auto scatter_updates(const Launch& plan, cudaStream_t stream) -> void {
    if (plan.elements)
        kernel::scatter_updates<<<static_cast<unsigned int>((plan.elements + 255) / 256), 256, 0, stream>>>(plan);
}

auto embedding(const Launch& plan, cudaStream_t stream) -> void {
    const auto blocks = static_cast<unsigned int>((plan.elements + 255) / 256);
    kernel::embedding<<<blocks, 256, 0, stream>>>(plan);
}

auto embedding_sum(const Launch& plan, cudaStream_t stream) -> void {
    const auto blocks = static_cast<unsigned int>((plan.elements + 255) / 256);
    kernel::embedding_sum<<<blocks, 256, 0, stream>>>(plan);
}

auto diffusion_candidates(const Launch& plan, cudaStream_t stream) -> void {
    kernel::diffusion_candidates<<<static_cast<unsigned int>(plan.elements / 2), 256, 0, stream>>>(plan);
}

auto diffusion_update(const Launch& plan, float* scratch, cudaStream_t stream) -> void {
    const auto blocks = static_cast<unsigned int>((plan.elements + 255) / 256);
    kernel::diffusion_scores<<<blocks, 256, 0, stream>>>(plan, scratch);
    kernel::diffusion_update<<<blocks, 256, 0, stream>>>(plan, scratch);
}

} // namespace kidi::cuda
