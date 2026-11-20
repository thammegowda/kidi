#include "cuda/attention.h"
#include "cuda/elementwise.h"
#if defined(KIDI_CUDA_USE_CUTLASS_INT8)
#include "cuda/flash_attention.h"
#endif

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <type_traits>

namespace kidi::cuda {
namespace {

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

template <typename Element, int Vector>
__device__ void load_decode_vector(const Element* source, float (&output)[Vector]) {
    if constexpr (std::is_same_v<Element, __nv_bfloat16>) {
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
    } else {
#pragma unroll
        for (int block = 0; block < Vector / 4; ++block) {
            const float4 value = *reinterpret_cast<const float4*>(source + block * 4);
            output[block * 4] = value.x;
            output[block * 4 + 1] = value.y;
            output[block * 4 + 2] = value.z;
            output[block * 4 + 3] = value.w;
        }
    }
}

template <typename Element, int Width, int Group>
__global__ void decode_partitions(Launch plan, float* partials, std::int64_t heads, std::int64_t key_heads,
                                  std::int64_t keys, std::int64_t partitions) {
    constexpr int Vector = Width / 32, Warps = 4, Partition = 64;
    __shared__ float maxima[Warps][Group], denominators[Warps][Group], outputs[Warps][Group][Width];
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const auto batch = static_cast<std::int64_t>(blockIdx.x), head = static_cast<std::int64_t>(blockIdx.y) * Group;
    const auto key_head = head / (heads / key_heads);
    const auto offset = plan.attribute_count >= 3 ? plan.attributes[2] : 0;
    const auto key_batch = plan.inputs[1].shape[0] == 1 ? 0 : batch;
    const auto* query = static_cast<const Element*>(plan.inputs[0].data) + (batch * heads + head) * Width;
    const auto* key = static_cast<const Element*>(address(plan.inputs[1])) +
                      (key_batch * plan.inputs[1].shape[1] + offset) * key_heads * Width + key_head * Width;
    const auto* value = static_cast<const Element*>(address(plan.inputs[2])) +
                        (key_batch * plan.inputs[2].shape[1] + offset) * key_heads * Width + key_head * Width;
    float activation[Group][Vector], accumulated[Group][Vector]{}, maximum[Group], denominator[Group]{};
#pragma unroll
    for (int group = 0; group < Group; ++group) {
        load_decode_vector<Element, Vector>(query + group * Width + lane * Vector, activation[group]);
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
        load_decode_vector<Element, Vector>(key + position * key_heads * Width + lane * Vector, key_values);
        load_decode_vector<Element, Vector>(value + position * key_heads * Width + lane * Vector, value_values);
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

template <typename Element, int Width>
__global__ void decode_merge(Launch plan, const float* partials, std::int64_t partitions) {
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
    for (int column = 0; column < Width / 256; ++column) {
        const float value = result[column] / denominator;
        auto* out = static_cast<Element*>(plan.output.data) + static_cast<std::int64_t>(blockIdx.x) * Width +
                    threadIdx.x + column * 256;
        if constexpr (std::is_same_v<Element, __nv_bfloat16>)
            *out = __float2bfloat16(value);
        else
            *out = value;
    }
}

template <typename Element, int Width>
__global__ void parallel_attention(Launch plan, std::int64_t heads, std::int64_t key_heads, std::int64_t keys) {
    extern __shared__ float probabilities[];
    __shared__ float partial[8];
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const auto queries = plan.inputs[0].shape[1];
    const auto head = blockIdx.x % heads, row = blockIdx.x / heads % queries, batch = blockIdx.x / heads / queries;
    const auto key_head = head / (heads / key_heads);
    const auto offset = plan.attribute_count >= 3 ? plan.attributes[2] : 0;
    const auto key_batch = plan.inputs[1].shape[0] == 1 ? 0 : batch;
    const auto value_batch = plan.inputs[2].shape[0] == 1 ? 0 : batch;
    const auto* query =
        static_cast<const Element*>(plan.inputs[0].data) + static_cast<std::int64_t>(blockIdx.x) * Width;
    const auto* key = static_cast<const Element*>(address(plan.inputs[1])) +
                      (key_batch * plan.inputs[1].shape[1] + offset) * key_heads * Width + key_head * Width;
    const auto* value = static_cast<const Element*>(address(plan.inputs[2])) +
                        (value_batch * plan.inputs[2].shape[1] + offset) * key_heads * Width + key_head * Width;
    float activation[Width / 32];
#pragma unroll
    for (int chunk = 0; chunk < Width / 32; ++chunk) activation[chunk] = static_cast<float>(query[lane + chunk * 32]);
    const float scale = plan.epsilon > 0 ? plan.epsilon : rsqrtf(static_cast<float>(Width));
    View score_view{};
    score_view.rank = 4;
    score_view.shape[0] = plan.inputs[0].shape[0];
    score_view.shape[1] = heads;
    score_view.shape[2] = queries;
    score_view.shape[3] = keys;
    for (std::int64_t position = warp; position < keys; position += 8) {
        float score = 0;
#pragma unroll
        for (int chunk = 0; chunk < Width / 32; ++chunk)
            score = fmaf(activation[chunk], static_cast<float>(key[position * key_heads * Width + lane + chunk * 32]),
                         score);
        for (int delta = 16; delta > 0; delta /= 2) score += __shfl_down_sync(0xffffffff, score, delta);
        if (lane == 0) {
            score *= scale;
            if (plan.count == 4)
                score +=
                    read(plan.inputs[3], broadcast_index(((batch * heads + head) * queries + row) * keys + position,
                                                         score_view, plan.inputs[3]));
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
                fmaf(probabilities[position], static_cast<float>(value[position * key_heads * Width + column]), result);
        static_cast<Element*>(plan.output.data)[static_cast<std::int64_t>(blockIdx.x) * Width + column] =
            static_cast<Element>(result / denominator);
    }
}

template <typename Element, int Width>
void run_parallel_attention(Launch plan, std::int64_t heads, std::int64_t key_heads, std::int64_t keys,
                            cudaStream_t stream) {
    const auto rows = static_cast<unsigned int>(plan.elements / Width);
    parallel_attention<Element, Width><<<rows, 256, keys * sizeof(float), stream>>>(plan, heads, key_heads, keys);
    check(cudaGetLastError());
}

template <typename Element>
auto parallel_attention_launcher(std::int64_t width) -> AttentionLauncher {
    switch (width) {
        case 32:
            return run_parallel_attention<Element, 32>;
        case 64:
            return run_parallel_attention<Element, 64>;
        case 128:
            return run_parallel_attention<Element, 128>;
        case 256:
            return run_parallel_attention<Element, 256>;
        case 512:
            return run_parallel_attention<Element, 512>;
        default:
            return nullptr;
    }
}

#if defined(KIDI_CUDA_USE_CUTLASS_INT8)
auto to_flash_view(const View& source) -> FlashView {
    FlashView result{};
    result.data = source.data;
    result.binding = source.binding;
    result.binding_offset = source.binding_offset;
    result.type = source.type;
    result.rank = source.rank;
    std::copy(source.shape, source.shape + 4, result.shape);
    return result;
}

template <int Width>
void dispatch_flash_attention(Launch plan, std::int64_t heads, std::int64_t key_heads, std::int64_t keys,
                              cudaStream_t stream) {
    const bool aligned = reinterpret_cast<std::uintptr_t>(plan.inputs[0].data) % 16 == 0 &&
                         reinterpret_cast<std::uintptr_t>(plan.inputs[1].data) % 16 == 0 &&
                         reinterpret_cast<std::uintptr_t>(plan.inputs[2].data) % 16 == 0 &&
                         reinterpret_cast<std::uintptr_t>(plan.output.data) % 16 == 0;
    const bool bfloat = plan.inputs[0].type == static_cast<int>(Type::BF16);
    if (!aligned) {
        if (keys <= 8192) {
            if (bfloat)
                run_parallel_attention<__nv_bfloat16, Width>(plan, heads, key_heads, keys, stream);
            else
                run_parallel_attention<float, Width>(plan, heads, key_heads, keys, stream);
            return;
        }
        throw std::runtime_error("CUDA: unaligned attention buffers with keys > 8192");
    }
    FlashAttentionArgs args{};
    args.query = to_flash_view(plan.inputs[0]);
    args.key = to_flash_view(plan.inputs[1]);
    args.value = to_flash_view(plan.inputs[2]);
    args.has_mask = plan.count == 4;
    if (args.has_mask) {
        args.mask = to_flash_view(plan.inputs[3]);
        const auto mask_type = static_cast<Type>(plan.inputs[3].type);
        if (mask_type == Type::F32)
            args.mask.type = 0;
        else if (mask_type == Type::BF16)
            args.mask.type = 1;
        else if (mask_type == Type::F16)
            args.mask.type = 2;
        else
            throw std::runtime_error("CUDA: unsupported FlashAttention mask dtype");
    }
    args.output = to_flash_view(plan.output);
    args.batches = plan.inputs[0].shape[0];
    args.queries = plan.inputs[0].shape[1];
    args.heads = heads;
    args.key_heads = key_heads;
    args.key_offset = plan.attribute_count >= 3 ? plan.attributes[2] : 0;
    args.keys = keys;
    args.head_dim = Width;
    args.scale = plan.epsilon > 0.F ? plan.epsilon : 1.F / std::sqrt(static_cast<float>(Width));
    const bool launched = bfloat ? flash_attention_bf16(args, stream) : flash_attention_f32(args, stream);
    if (!launched) throw std::runtime_error("CUDA: unsupported FlashAttention launch configuration");
    check(cudaGetLastError());
}

auto flash_attention_launcher(std::int64_t width) -> AttentionLauncher {
    switch (width) {
        case 64:
            return dispatch_flash_attention<64>;
        case 128:
            return dispatch_flash_attention<128>;
        case 256:
            return dispatch_flash_attention<256>;
        case 512:
            return dispatch_flash_attention<512>;
        default:
            return nullptr;
    }
}
#endif

} // namespace

auto select_parallel_attention(const Descriptor& descriptor, std::int64_t keys, std::int64_t width, int device)
    -> AttentionLauncher {
    const auto type = descriptor.inputs[0].type;
    if (descriptor.reference_attention || keys <= 0 || type != descriptor.inputs[1].type ||
        type != descriptor.output.type)
        return nullptr;
#if defined(KIDI_CUDA_USE_CUTLASS_INT8)
    if (type == Type::BF16 && descriptor.inputs[0].shape[1] > 1 && dense_device(device).major >= 8) {
        if (auto launcher = flash_attention_launcher(width)) return launcher;
    }
    if (type == Type::F32 && (descriptor.inputs[0].shape[1] >= 64 || keys > 512) && dense_device(device).major >= 8) {
        if (auto launcher = flash_attention_launcher(width)) return launcher;
    }
#else
    (void)device;
#endif
    if (keys > 8192) return nullptr;
    if (type == Type::BF16) return parallel_attention_launcher<__nv_bfloat16>(width);
#if defined(KIDI_CUDA_USE_CUTLASS_INT8)
    if (type == Type::F32 && descriptor.inputs[0].shape[1] > 1) return parallel_attention_launcher<float>(width);
#endif
    return nullptr;
}

auto decode_attention(const Launch& plan, float* partials, std::int64_t batches, std::int64_t heads,
                      std::int64_t key_heads, std::int64_t keys, std::int64_t width, int input_type,
                      cudaStream_t stream) -> void {
    const auto partitions = (keys + 63) / 64;
    const int group = (heads / key_heads) % 2 == 0 ? 2 : 1;
    const dim3 grid(static_cast<unsigned int>(batches), static_cast<unsigned int>(heads / group),
                    static_cast<unsigned int>(partitions));
    const auto dispatch_decode = [&](auto tag) {
        using Element = decltype(tag);
        if (width == 256) {
            if (group == 2)
                decode_partitions<Element, 256, 2>
                    <<<grid, 128, 0, stream>>>(plan, partials, heads, key_heads, keys, partitions);
            else
                decode_partitions<Element, 256, 1>
                    <<<grid, 128, 0, stream>>>(plan, partials, heads, key_heads, keys, partitions);
            decode_merge<Element, 256>
                <<<static_cast<unsigned int>(batches * heads), 256, 0, stream>>>(plan, partials, partitions);
        } else {
            if (group == 2)
                decode_partitions<Element, 512, 2>
                    <<<grid, 128, 0, stream>>>(plan, partials, heads, key_heads, keys, partitions);
            else
                decode_partitions<Element, 512, 1>
                    <<<grid, 128, 0, stream>>>(plan, partials, heads, key_heads, keys, partitions);
            decode_merge<Element, 512>
                <<<static_cast<unsigned int>(batches * heads), 256, 0, stream>>>(plan, partials, partitions);
        }
    };
    if (input_type == static_cast<int>(Type::BF16))
        dispatch_decode(__nv_bfloat16{});
    else
        dispatch_decode(float{});
}

auto fallback_attention(const Launch& plan, float* scratch, std::int64_t batches, std::int64_t heads,
                        std::int64_t key_heads, std::int64_t queries, std::int64_t keys, cudaStream_t stream) -> void {
    const auto blocks = static_cast<unsigned int>((plan.elements + 255) / 256);
    const auto score_count = batches * heads * queries * keys;
    attention_scores<<<static_cast<unsigned int>((score_count + 255) / 256), 256, 0, stream>>>(plan, scratch, heads,
                                                                                               key_heads, keys);
    Launch softmax{};
    softmax.operation = static_cast<int>(Op::SOFTMAX);
    softmax.inputs[0] = softmax.output = {scratch, static_cast<int>(Type::F32), 4, {batches, heads, queries, keys}};
    softmax.elements = static_cast<std::size_t>(score_count);
    normalize(softmax, Op::SOFTMAX, stream);
    attention_values<<<blocks, 256, 0, stream>>>(plan, scratch, heads, key_heads, keys);
}

} // namespace kidi::cuda
