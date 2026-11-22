#include "cuda/fused_ops.h"

#include <cmath>

namespace kidi::cuda {
namespace {
namespace kernel {

constexpr int MAX_SOURCES = 7, MAX_QUERIES = 4, MAX_WARPS = 8, MAX_LANE_VALUES = 8;

__device__ float warp_sum(float value) {
    for (int offset = 16; offset; offset >>= 1) value += __shfl_xor_sync(0xffffffffu, value, offset);
    return value;
}

/// Sums `value` over the block; every thread receives the total.
__device__ float block_sum(float value, float* shared) {
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32, warps = (blockDim.x + 31) / 32;
    value = warp_sum(value);
    __syncthreads();
    if (lane == 0) shared[warp] = value;
    __syncthreads();
    float total = 0;
    for (int index = 0; index < warps; ++index) total += shared[index];
    return total;
}

__device__ float attribute_float(const Launch& plan, int index) {
    return __int_as_float(static_cast<int>(plan.attributes[index]));
}

/// One block per token: RMS-normalize each source, softmax pseudo-query logits over sources, mix.
__global__ void residual_mix(Launch plan) {
    __shared__ float reduction[32];
    __shared__ float inverse[MAX_SOURCES], weights[MAX_QUERIES][MAX_SOURCES];
    const int sources = plan.count - 1;
    const auto& queries = plan.inputs[sources];
    const auto width = queries.shape[1], queries_count = queries.shape[0], tokens = plan.inputs[0].shape[0];
    const auto token = blockIdx.x;
    for (int source = 0; source < sources; ++source) {
        float squares = 0;
        for (auto column = threadIdx.x; column < width; column += blockDim.x) {
            const float value = read(plan.inputs[source], token * width + column);
            squares += value * value;
        }
        squares = block_sum(squares, reduction);
        if (threadIdx.x == 0) inverse[source] = rsqrtf(squares / width + plan.epsilon);
    }
    __syncthreads();
    for (int query = 0; query < queries_count; ++query)
        for (int source = 0; source < sources; ++source) {
            float dot = 0;
            for (auto column = threadIdx.x; column < width; column += blockDim.x)
                dot += read(queries, query * width + column) * read(plan.inputs[source], token * width + column);
            dot = block_sum(dot, reduction);
            if (threadIdx.x == 0) weights[query][source] = dot * inverse[source] * rsqrtf(static_cast<float>(width));
        }
    __syncthreads();
    if (threadIdx.x < queries_count) {
        auto* row = weights[threadIdx.x];
        float peak = row[0];
        for (int source = 1; source < sources; ++source) peak = fmaxf(peak, row[source]);
        float total = 0;
        for (int source = 0; source < sources; ++source) total += row[source] = expf(row[source] - peak);
        for (int source = 0; source < sources; ++source) row[source] /= total;
    }
    __syncthreads();
    for (auto column = threadIdx.x; column < width; column += blockDim.x) {
        float normalized[MAX_SOURCES];
        for (int source = 0; source < sources; ++source)
            normalized[source] = read(plan.inputs[source], token * width + column) * inverse[source];
        for (int query = 0; query < queries_count; ++query) {
            float value = 0;
            for (int source = 0; source < sources; ++source) value += weights[query][source] * normalized[source];
            write(plan.output, (query * tokens + token) * width + column, value);
        }
    }
}

/// Causal depthwise convolution over `padded` rows (the previous W - 1 inputs first), lag-0-first taps.
__global__ void short_convolution(Launch plan) {
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= plan.elements) return;
    const auto width = plan.attributes[0], channels = plan.output.shape[1];
    const auto token = static_cast<std::int64_t>(index / channels), channel = static_cast<std::int64_t>(index % channels);
    const bool dynamic = plan.count == 3;
    const auto groups = dynamic ? plan.inputs[2].shape[1] / width : 1;
    const auto group = channel / (channels / groups);
    const auto current = (token + width - 1) * channels + channel;
    float sum = read(plan.inputs[0], current);
    for (std::int64_t lag = 0; lag < width; ++lag) {
        float tap = read(plan.inputs[1], lag * channels + channel);
        if (dynamic) tap += read(plan.inputs[2], (token * groups + group) * width + lag);
        sum += tap * read(plan.inputs[0], current - lag * channels);
    }
    write(plan.output, index, sum * plan.epsilon);
}

/// One warp per row: scale, then shrink rows whose L2 norm exceeds the threshold.
__global__ void l2_clip(Launch plan) {
    const auto width = plan.inputs[0].shape[plan.inputs[0].rank - 1];
    const std::size_t row = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane = threadIdx.x % 32;
    if (row * width >= plan.elements) return;
    const auto scales = plan.count == 2 ? plan.inputs[1].shape[0] * plan.inputs[1].shape[1] : 0;
    const auto scaled = [&](std::int64_t column) {
        const auto index = row * width + column;
        const float value = read(plan.inputs[0], index);
        return scales ? value * read(plan.inputs[1], index % scales) : value;
    };
    float squares = 0;
    for (auto column = lane; column < width; column += 32) {
        const float value = scaled(column);
        squares += value * value;
    }
    const float factor = fminf(1.F, plan.epsilon / (sqrtf(warp_sum(squares)) + 1e-6F));
    for (auto column = lane; column < width; column += 32) write(plan.output, row * width + column, scaled(column) * factor);
}

/// One block per (query, head), one warp per key stride. The sink is a zero-valued key with logit `sink`;
/// keys closer than `extent` add the per-query relative logit, and a sigmoid gate scales the output.
__global__ void relative_attention(Launch plan) {
    __shared__ float peaks[MAX_WARPS], totals[MAX_WARPS], partial[MAX_WARPS][MAX_LANE_VALUES * 32];
    const auto heads = plan.attributes[0], key_heads = plan.attributes[1], start = plan.attributes[2],
               window = plan.attributes[3], extent = plan.attributes[4];
    const auto token = static_cast<std::int64_t>(blockIdx.x), head = static_cast<std::int64_t>(blockIdx.y);
    const auto width = plan.inputs[0].shape[1] / heads, key_head = head / (heads / key_heads);
    const auto key_stride = plan.inputs[1].shape[1];
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32, warps = blockDim.x / 32;
    const int values = static_cast<int>(width / 32);
    const auto position = start + token;
    const auto first = window ? max(static_cast<std::int64_t>(0), position - window + 1) : 0;
    const float scale = rsqrtf(static_cast<float>(width));
    float query[MAX_LANE_VALUES], accumulator[MAX_LANE_VALUES];
    for (int value = 0; value < values; ++value) {
        query[value] = read(plan.inputs[0], (token * heads + head) * width + value * 32 + lane) * scale;
        accumulator[value] = 0;
    }
    float peak = -1e30F, total = 0;
    for (auto key = first + warp; key <= position; key += warps) {
        const auto base = key * key_stride + key_head * width;
        float dot = 0;
        for (int value = 0; value < values; ++value) dot += query[value] * read(plan.inputs[1], base + value * 32 + lane);
        float logit = warp_sum(dot);
        const auto distance = position - key;
        if (distance < extent) logit += read(plan.inputs[3], (token * heads + head) * extent + distance);
        const float next = fmaxf(peak, logit), decay = expf(peak - next), weight = expf(logit - next);
        total = total * decay + weight;
        for (int value = 0; value < values; ++value)
            accumulator[value] = accumulator[value] * decay + weight * read(plan.inputs[2], base + value * 32 + lane);
        peak = next;
    }
    if (lane == 0) peaks[warp] = peak, totals[warp] = total;
    for (int value = 0; value < values; ++value) partial[warp][value * 32 + lane] = accumulator[value];
    __syncthreads();
    float combined = read(plan.inputs[4], token * heads + head);
    for (int index = 0; index < warps; ++index) combined = fmaxf(combined, peaks[index]);
    float denominator = expf(read(plan.inputs[4], token * heads + head) - combined);
    for (int index = 0; index < warps; ++index) denominator += totals[index] * expf(peaks[index] - combined);
    for (auto column = threadIdx.x; column < width; column += blockDim.x) {
        float sum = 0;
        for (int index = 0; index < warps; ++index) sum += partial[index][column] * expf(peaks[index] - combined);
        const auto output = (token * heads + head) * width + column;
        const float gate = read(plan.inputs[5], output);
        write(plan.output, output, sum / denominator / (1 + expf(-gate)));
    }
}

/// softcap(gate) * sigmoid(gate) * softcap(up) * row weight, for [rows, gate | up] inputs.
__global__ void softcapped_silu_gate(Launch plan) {
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= plan.elements) return;
    const auto width = plan.output.shape[plan.output.rank - 1];
    const auto row = index / width, offset = row * 2 * width + index % width;
    const float gate = read(plan.inputs[0], offset), up = read(plan.inputs[0], offset + width);
    const float gate_cap = attribute_float(plan, 0), up_cap = attribute_float(plan, 1);
    float result = tanhf(gate / gate_cap) * gate_cap / (1 + expf(-gate)) * tanhf(up / up_cap) * up_cap;
    if (plan.count == 2) result *= read(plan.inputs[1], row);
    write(plan.output, index, result);
}

} // namespace kernel

auto blocks(std::size_t elements, int threads) -> unsigned { return static_cast<unsigned>((elements + threads - 1) / threads); }
} // namespace

auto fused_operation(const Launch& plan, Op operation, cudaStream_t stream) -> void {
    switch (operation) {
        case Op::RESIDUAL_MIX:
            kernel::residual_mix<<<static_cast<unsigned>(plan.inputs[0].shape[0]), 256, 0, stream>>>(plan);
            break;
        case Op::SHORT_CONVOLUTION:
            kernel::short_convolution<<<blocks(plan.elements, 256), 256, 0, stream>>>(plan);
            break;
        case Op::L2_CLIP: {
            const auto width = plan.inputs[0].shape[plan.inputs[0].rank - 1];
            kernel::l2_clip<<<blocks(plan.elements / width * 32, 256), 256, 0, stream>>>(plan);
            break;
        }
        case Op::RELATIVE_ATTENTION: {
            const dim3 grid(static_cast<unsigned>(plan.inputs[0].shape[0]), static_cast<unsigned>(plan.attributes[0]));
            kernel::relative_attention<<<grid, 256, 0, stream>>>(plan);
            break;
        }
        case Op::SILU_GATE:
            kernel::softcapped_silu_gate<<<blocks(plan.elements, 256), 256, 0, stream>>>(plan);
            break;
        default:
            throw std::runtime_error("CUDA: unsupported fused operation");
    }
    check(cudaGetLastError());
}

} // namespace kidi::cuda
