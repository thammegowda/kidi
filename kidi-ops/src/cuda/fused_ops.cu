#include "cuda/fused_ops.h"
#include "cuda/flash_attention.h"
#include "cuda/ptx_mma.cuh"

#include <cmath>
#include <cstring>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <limits>
#include <mma.h>

namespace kidi::cuda {
namespace {
namespace kernel {

constexpr int MAX_SOURCES = 7, MAX_QUERIES = 4, MAX_LANE_VALUES = 8;
// Decode attention has one block per head, so more warps keep more keys in flight.
constexpr int ATTENTION_WARPS = 32;
// Decode mixes a single token in one block, so a full block spreads the hidden columns as widely as possible.
constexpr int MIX_THREADS = 1024;

__device__ float e4m3(unsigned char bits) {
    const int exponent = (bits >> 3) & 15, mantissa = bits & 7;
    const float magnitude = exponent ? __uint_as_float(static_cast<unsigned>(exponent + 120) << 23) *
                                           (1.F + static_cast<float>(mantissa) / 8.F)
                                     : static_cast<float>(mantissa) / 512.F;
    if (exponent == 15 && mantissa == 7) return nanf("");
    return bits & 0x80 ? -magnitude : magnitude;
}

__device__ float expert_value(const Launch& plan, std::size_t index) {
    const bool nvfp4 = plan.inputs[0].type == static_cast<int>(Type::U8);
    const auto rows = plan.inputs[0].shape[1], columns = plan.inputs[0].shape[2] * (nvfp4 ? 2 : 1);
    const auto expert = index / (rows * columns), row = index / columns % rows, column = index % columns;
    const auto source_row = plan.attributes[0] ? (row % (rows / 2)) * 2 + row / (rows / 2) : row;
    const auto source = (expert * rows + source_row) * columns + column;
    const auto* values = static_cast<const unsigned char*>(address(plan.inputs[0]));
    const auto* scales = static_cast<const unsigned char*>(address(plan.inputs[1]));
    if (nvfp4) {
        constexpr float E2M1[]{0.F,  0.5F,  1.F,  1.5F,  2.F,  3.F,  4.F,  6.F,
                               -0.F, -0.5F, -1.F, -1.5F, -2.F, -3.F, -4.F, -6.F};
        const auto byte = values[source / 2];
        return E2M1[column % 2 ? byte >> 4 : byte & 15] * e4m3(scales[source / 16]) * read(plan.inputs[2], expert);
    }
    return e4m3(values[source]) * ldexpf(1.F, static_cast<int>(scales[source / 32]) - 127);
}

__global__ void expert_row_scales(Launch plan) {
    __shared__ float maxima[256];
    const auto columns = plan.inputs[0].shape[2] * (plan.inputs[0].type == static_cast<int>(Type::U8) ? 2 : 1);
    const auto row = blockIdx.x;
    float maximum = 0.F;
    for (auto column = threadIdx.x; column < columns; column += blockDim.x) {
        const auto value = expert_value(plan, row * columns + column);
        maximum = isfinite(value) ? fmaxf(maximum, fabsf(value)) : INFINITY;
    }
    maxima[threadIdx.x] = maximum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride; stride /= 2) {
        if (threadIdx.x < stride) maxima[threadIdx.x] = fmaxf(maxima[threadIdx.x], maxima[threadIdx.x + stride]);
        __syncthreads();
    }
    if (threadIdx.x == 0) write(plan.output, row, maxima[0] == 0.F ? 1.F : maxima[0] / 127.F);
}

__global__ void dequantize_experts(Launch plan) {
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= plan.elements) return;
    const auto value = expert_value(plan, index);
    if (plan.output.type == static_cast<int>(Type::U8)) {
        const auto scale = read(plan.inputs[plan.count - 1], index / plan.output.shape[2]);
        const auto code = static_cast<int>(fminf(127.F, fmaxf(-127.F, roundf(value / scale))));
        static_cast<unsigned char*>(address(plan.output))[index] = static_cast<unsigned char>(code);
        return;
    }
    auto bits = __float_as_uint(value);
    bits += 0x7FFF + ((bits >> 16) & 1);
    static_cast<unsigned short*>(address(plan.output))[index] = static_cast<unsigned short>(bits >> 16);
}

__device__ float warp_sum(float value) {
    for (int offset = 16; offset; offset >>= 1) value += __shfl_xor_sync(0xffffffffu, value, offset);
    return value;
}

// Formats: 0 = E2M1 (NVFP4), 1 = E4M3 values with UE8M0 or E4M3 scales (MXFP8), 2 = signed INT4.
__device__ float block_scale(const View& scales, int format, std::int64_t index) {
    const auto type = static_cast<Type>(scales.type);
    if (type != Type::E4M3 && type != Type::U8) return read(scales, index);
    const auto byte = static_cast<const unsigned char*>(address(scales))[index];
    if (format == 1 && type == Type::U8)
        return byte == 255 ? nanf("") : __uint_as_float(byte ? static_cast<unsigned>(byte) << 23 : 0x00400000);
    return e4m3(byte);
}

__device__ float block_code(int format, unsigned bits) {
    if (format == 2) return bits < 8 ? static_cast<float>(bits) : static_cast<float>(static_cast<int>(bits) - 16);
    const auto magnitude = bits & 7;
    const auto encoded =
        magnitude < 2 ? magnitude * 0x3F000000U : ((magnitude / 2 + 126) << 23) | ((magnitude & 1) << 22);
    return __uint_as_float(encoded | ((bits & 8) << 28));
}

__device__ float block_scaled_weight(const Launch& plan, std::int64_t row, std::int64_t column) {
    const auto width = plan.inputs[0].shape[plan.inputs[0].rank - 1];
    const auto index = row * width + column;
    const auto* values = static_cast<const unsigned char*>(address(plan.inputs[1]));
    const auto group = plan.attributes[4], shift = plan.attributes[6];
    const auto format = static_cast<int>(plan.attributes[3]);
    const auto scale = block_scale(plan.inputs[2], format, shift >= 0 ? index >> shift : index / group);
    const auto byte = values[format == 1 ? index : index / 2];
    const float value = format == 1 ? e4m3(byte) : block_code(format, column % 2 ? byte >> 4 : byte & 15);
    return value * scale * (plan.attributes[5] ? read(plan.inputs[3], 0) : 1.F);
}

__device__ float block_scaled_gate(const Launch& plan, float gate, float up, float routing) {
    if (plan.inputs[0].type == static_cast<int>(Type::BF16)) {
        gate = __bfloat162float(__float2bfloat16(gate));
        up = __bfloat162float(__float2bfloat16(up));
    }
    const auto gate_cap = __int_as_float(static_cast<int>(plan.attributes[1]));
    const auto up_cap = __int_as_float(static_cast<int>(plan.attributes[2]));
    return tanhf(gate / gate_cap) * gate_cap / (1 + expf(-gate)) * tanhf(up / up_cap) * up_cap * routing;
}

// Decodes nibble i of both 16-bit halves of a 4-bit chunk word into an FP16 pair: FP4 (E2M1) magnitude and sign bits
// are moved into FP16 positions (giving the value times 2^-14, rescaled by the caller); INT4 nibbles use the
// 1024-offset FP16 trick (offset removed by the caller).
template <int FORMAT>
__device__ __half2 nibble_pair(unsigned word, int i) {
    const auto shifted = word >> (4 * i);
    const unsigned bits = FORMAT == 0 ? ((shifted << 9) & 0x0E000E00U) | ((shifted << 12) & 0x80008000U)
                                      : ((shifted & 0x000F000FU) ^ 0x00080008U) | 0x64006400U;
    return *reinterpret_cast<const __half2*>(&bits);
}

// Dot product of one 16-byte weight chunk with its activations in FP32, before block scaling; `partial` holds one
// sum per 16-value scale group. `x` pairs value k of each 8-value word with value k + 4 for 4-bit formats, and holds
// consecutive pairs for FP8. The 2^14 FP4 factor and the 1032 INT4 offset are applied here.
template <int FORMAT>
__device__ void chunk_dot(const uint4& chunk, const float2* x, float* partial) {
    const unsigned words[4] = {chunk.x, chunk.y, chunk.z, chunk.w};
#pragma unroll
    for (int word = 0; word < 4; ++word) {
#pragma unroll
        for (int i = 0; i < (FORMAT == 1 ? 2 : 4); ++i) {
            float2 value;
            if constexpr (FORMAT == 1) {
                const auto raw = __nv_cvt_fp8x2_to_halfraw2(
                    static_cast<__nv_fp8x2_storage_t>(words[word] >> (16 * i)), __NV_E4M3);
                value = __half22float2(*reinterpret_cast<const __half2*>(&raw));
            } else {
                value = __half22float2(nibble_pair<FORMAT>(words[word], i));
                if (FORMAT == 2) value.x -= 1032.F, value.y -= 1032.F;
            }
            const int group = FORMAT == 1 ? 0 : word / 2, pair = FORMAT == 1 ? word * 2 + i : word * 4 + i;
            partial[group] = fmaf(x[pair].x, value.x, partial[group]);
            partial[group] = fmaf(x[pair].y, value.y, partial[group]);
        }
    }
    if (FORMAT == 0) partial[0] *= 16384.F, partial[1] *= 16384.F;
}

// Decode GEMV over block-scaled weights. Each warp computes ROWS consecutive outputs of one token (for gated
// weights, ROWS interleaved gate/up row pairs read in place), so every activation chunk is loaded and converted once
// and reused for all rows. Each lane loads 16 weight bytes per row at a time (32 FP4/INT4 or 16 FP8 values) and
// issues every row's chunk before decoding any, keeping enough requests in flight to approach memory bandwidth.
// Scales multiply each 16-value group sum rather than every weight. SCALE selects how scales are decoded: 0 = E4M3
// pairs for 16-value groups, 1 = UE8M0 bytes, 2 = any other type or group. Weights and activations must be 16-byte
// aligned and each token's outputs a multiple of ROWS (checked on launch). Dense: slots = 1. Routed over stacked
// experts [experts, rows, packed] with routes [tokens, slots, (expert, weight)]: gated outputs are
// [tokens, slots, rows / 2]; down outputs [tokens, rows] sum every slot.
template <bool BF16, bool ROUTED, bool GATED, int FORMAT, int SCALE, int ROWS>
__device__ void block_scaled_warp_gemv(const Launch& plan, std::int64_t first_output, int lane) {
    constexpr int PAIRS = GATED ? 2 : 1, LINES = ROWS * PAIRS, PER_CHUNK = FORMAT == 1 ? 16 : 32;
    constexpr int BYTES_PER_VALUE_SHIFT = FORMAT == 1 ? 0 : 1;
    const auto group = plan.attributes[4];
    const int shift = static_cast<int>(plan.attributes[6]);
    const bool has_global = plan.attributes[5];
    const int width = static_cast<int>(plan.inputs[0].shape[plan.inputs[0].rank - 1]);
    const auto columns = plan.output.shape[plan.output.rank - 1];
    const auto& routes = plan.inputs[plan.count - 1];
    const auto rows = plan.inputs[1].shape[ROUTED ? 1 : 0];
    int slots = 1, row = 0;
    std::int64_t route_base = 0, activation_base = 0;
    float routing = 1.F;
    if (ROUTED) {
        if (GATED) {
            route_base = first_output / columns;
            row = static_cast<int>(first_output % columns) * 2;
            activation_base = route_base / routes.shape[1] * width;
            routing = read(routes, route_base * 2 + 1);
        } else {
            slots = static_cast<int>(routes.shape[1]);
            route_base = first_output / columns * slots;
            row = static_cast<int>(first_output % columns);
            activation_base = route_base * width;
        }
    } else {
        const auto token = first_output / columns;
        row = static_cast<int>(first_output % columns) * PAIRS;
        activation_base = token * width;
        if (GATED) routing = read(plan.inputs[plan.count - 1], token);
    }
    const auto* values = static_cast<const unsigned char*>(address(plan.inputs[1]));
    const auto* scale_bytes = static_cast<const unsigned char*>(address(plan.inputs[2]));
    const auto* routes_data = ROUTED ? static_cast<const float*>(address(routes)) : nullptr;
    const auto* globals = has_global ? static_cast<const float*>(address(plan.inputs[3])) : nullptr;
    const auto* activations = address(plan.inputs[0]);
    const int chunks = width / PER_CHUNK, items = slots * chunks;
    // Element offsets within one expert fit 32 bits; the expert base is applied to pointers.
    const auto expert_elements = static_cast<std::int64_t>(rows) * width;
    const auto generic_scale = [&](std::int64_t index) {
        return block_scale(plan.inputs[2], FORMAT, shift >= 0 ? index >> shift : index / group);
    };
    float sums[LINES]{};
    for (int item = lane; item < items; item += 32) {
        const int slot = ROUTED && !GATED ? item / chunks : 0;
        const int column = (item - slot * chunks) * PER_CHUNK;
        const unsigned char* expert_values = values;
        const unsigned char* expert_scales = scale_bytes;
        std::int64_t expert_base = 0;
        float global = has_global ? globals[0] : 1.F;
        if (ROUTED) {
            const int expert = static_cast<int>(routes_data[(route_base + slot) * 2]);
            expert_base = expert * expert_elements;
            expert_values += expert_base >> BYTES_PER_VALUE_SHIFT;
            if (SCALE != 2) expert_scales += expert_base >> shift;
            if (has_global) global = globals[expert];
        }
        const int offset = row * width + column;
        uint4 weights[LINES];
#pragma unroll
        for (int line = 0; line < LINES; ++line)
            weights[line] = __ldg(
                reinterpret_cast<const uint4*>(expert_values + ((offset + line * width) >> BYTES_PER_VALUE_SHIFT)));
        float scales[LINES][2];
#pragma unroll
        for (int line = 0; line < LINES; ++line) {
            const int index = offset + line * width;
            if constexpr (SCALE == 0) {
                const auto pair = __ldg(reinterpret_cast<const unsigned short*>(expert_scales + (index >> 4)));
                const auto raw = __nv_cvt_fp8x2_to_halfraw2(pair, __NV_E4M3);
                const auto decoded = __half22float2(*reinterpret_cast<const __half2*>(&raw));
                scales[line][0] = decoded.x * global, scales[line][1] = decoded.y * global;
            } else if constexpr (SCALE == 1) {
                const unsigned byte = __ldg(expert_scales + (index >> shift));
                scales[line][0] = byte == 255 ? nanf("") : __uint_as_float(byte ? byte << 23 : 0x00400000) * global;
                scales[line][1] = 0.F;
            } else {
                scales[line][0] = generic_scale(expert_base + index) * global;
                scales[line][1] = FORMAT != 1 ? generic_scale(expert_base + index + 16) * global : 0.F;
            }
        }
        // Load activation pairs directly in the order chunk_dot reads them (without an intermediate x[] array).
        float2 pairs[PER_CHUNK / 2];
        const auto activation_at = activation_base + static_cast<std::int64_t>(slot) * width + column;
        if constexpr (BF16) {
            const auto* source =
                reinterpret_cast<const uint4*>(static_cast<const __nv_bfloat16*>(activations) + activation_at);
            if constexpr (FORMAT == 1) {
#pragma unroll
                for (int quad = 0; quad < 2; ++quad) {
                    const auto packed = __ldg(source + quad);
                    const auto* halves = reinterpret_cast<const __nv_bfloat162*>(&packed);
#pragma unroll
                    for (int i = 0; i < 4; ++i) pairs[quad * 4 + i] = __bfloat1622float2(halves[i]);
                }
            } else {
#pragma unroll
                for (int quad = 0; quad < 4; ++quad) {
                    const auto packed = __ldg(source + quad);
                    const auto* halves = reinterpret_cast<const __nv_bfloat162*>(&packed);
                    const float2 f0 = __bfloat1622float2(halves[0]);
                    const float2 f1 = __bfloat1622float2(halves[1]);
                    const float2 f2 = __bfloat1622float2(halves[2]);
                    const float2 f3 = __bfloat1622float2(halves[3]);
                    pairs[quad * 4 + 0] = make_float2(f0.x, f2.x);
                    pairs[quad * 4 + 1] = make_float2(f0.y, f2.y);
                    pairs[quad * 4 + 2] = make_float2(f1.x, f3.x);
                    pairs[quad * 4 + 3] = make_float2(f1.y, f3.y);
                }
            }
        } else {
            const auto* source = reinterpret_cast<const float4*>(static_cast<const float*>(activations) + activation_at);
            if constexpr (FORMAT == 1) {
#pragma unroll
                for (int quad = 0; quad < 4; ++quad) {
                    const auto packed = __ldg(source + quad);
                    pairs[quad * 2 + 0] = make_float2(packed.x, packed.y);
                    pairs[quad * 2 + 1] = make_float2(packed.z, packed.w);
                }
            } else {
#pragma unroll
                for (int quad = 0; quad < 4; ++quad) {
                    const auto p0 = __ldg(source + quad * 2);
                    const auto p1 = __ldg(source + quad * 2 + 1);
                    pairs[quad * 4 + 0] = make_float2(p0.x, p1.x);
                    pairs[quad * 4 + 1] = make_float2(p0.y, p1.y);
                    pairs[quad * 4 + 2] = make_float2(p0.z, p1.z);
                    pairs[quad * 4 + 3] = make_float2(p0.w, p1.w);
                }
            }
        }
#pragma unroll
        for (int line = 0; line < LINES; ++line) {
            float partial[2]{};
            chunk_dot<FORMAT>(weights[line], pairs, partial);
            sums[line] = fmaf(partial[0], scales[line][0], sums[line]);
            if (FORMAT != 1) sums[line] = fmaf(partial[1], scales[line][1], sums[line]);
        }
    }
#pragma unroll
    for (int output = 0; output < ROWS; ++output) {
        const auto gate = warp_sum(sums[output * PAIRS]);
        if constexpr (GATED) {
            const auto up = warp_sum(sums[output * PAIRS + 1]);
            if (lane == 0) write(plan.output, first_output + output, block_scaled_gate(plan, gate, up, routing));
        } else if (lane == 0)
            write(plan.output, first_output + output, gate);
    }
}

constexpr int GEMV_WARPS = 4;

template <bool BF16, bool ROUTED, bool GATED, int FORMAT, int SCALE, int ROWS>
__global__ void __launch_bounds__(GEMV_WARPS * 32, 6) block_scaled_block_gemv(const __grid_constant__ Launch plan) {
    const int lane = threadIdx.x & 31;
    const auto first_output = (static_cast<std::int64_t>(blockIdx.x) * GEMV_WARPS + (threadIdx.x >> 5)) * ROWS;
    if (first_output >= static_cast<std::int64_t>(plan.elements)) return;
    block_scaled_warp_gemv<BF16, ROUTED, GATED, FORMAT, SCALE, ROWS>(plan, first_output, lane);
}

// One warp per token: sigmoid scores, biased top-k selection, then weights normalized over the selected scores.
__global__ void route_experts(const __grid_constant__ Launch plan) {
    __shared__ float scores[1024], biased[1024];
    const auto token = static_cast<std::int64_t>(blockIdx.x);
    const int lane = threadIdx.x;
    const auto experts = plan.inputs[0].shape[1], top = plan.attributes[0];
    for (auto expert = lane; expert < experts; expert += 32) {
        scores[expert] = 1.F / (1.F + expf(-read(plan.inputs[0], token * experts + expert)));
        biased[expert] = scores[expert] + read(plan.inputs[1], expert);
    }
    __syncwarp();
    float total = 0;
    for (std::int64_t slot = 0; slot < top; ++slot) {
        float best = -INFINITY;
        int index = static_cast<int>(experts);
        for (auto expert = lane; expert < experts; expert += 32)
            if (biased[expert] > best) best = biased[expert], index = expert;
        for (int offset = 16; offset; offset >>= 1) {
            const auto other = __shfl_xor_sync(0xffffffffu, best, offset);
            const auto other_index = __shfl_xor_sync(0xffffffffu, index, offset);
            if (other > best || (other == best && other_index < index)) best = other, index = other_index;
        }
        if (lane == 0) {
            write(plan.output, (token * top + slot) * 2, static_cast<float>(index));
            write(plan.output, (token * top + slot) * 2 + 1, scores[index]);
            total += scores[index];
            biased[index] = -INFINITY;
        }
        __syncwarp();
    }
    if (lane == 0) {
        const auto scale = __int_as_float(static_cast<int>(plan.attributes[1])) / (total + 1e-20F);
        auto* output = static_cast<float*>(address(plan.output));
        for (std::int64_t slot = 0; slot < top; ++slot) output[(token * top + slot) * 2 + 1] *= scale;
    }
}

// Decode fallback for widths or groups that are not multiples of eight.
// grid_constant prevents references to the descriptor from copying it into every thread's local memory.
__global__ void block_scaled_gemv(const __grid_constant__ Launch plan) {
    __shared__ float gates[8], ups[8];
    const auto index = blockIdx.x;
    if (index >= plan.elements) return;
    const auto width = plan.inputs[0].shape[plan.inputs[0].rank - 1];
    const auto columns = plan.output.shape[plan.output.rank - 1];
    const auto token = index / columns, row = index % columns * (plan.attributes[0] ? 2 : 1);
    float gate = 0, up = 0;
    for (auto column = static_cast<std::int64_t>(threadIdx.x); column < width; column += blockDim.x) {
        const float activation = read(plan.inputs[0], token * width + column);
        float weight = block_scaled_weight(plan, row, column);
        if (plan.inputs[0].type == static_cast<int>(Type::BF16)) weight = __bfloat162float(__float2bfloat16(weight));
        gate += activation * weight;
        if (plan.attributes[0]) {
            weight = block_scaled_weight(plan, row + 1, column);
            if (plan.inputs[0].type == static_cast<int>(Type::BF16))
                weight = __bfloat162float(__float2bfloat16(weight));
            up += activation * weight;
        }
    }
    gate = warp_sum(gate);
    up = warp_sum(up);
    if (threadIdx.x % 32 == 0) {
        gates[threadIdx.x / 32] = gate;
        ups[threadIdx.x / 32] = up;
    }
    __syncthreads();
    if (threadIdx.x < 32) {
        gate = warp_sum(threadIdx.x < blockDim.x / 32 ? gates[threadIdx.x] : 0.F);
        up = warp_sum(threadIdx.x < blockDim.x / 32 ? ups[threadIdx.x] : 0.F);
        if (threadIdx.x == 0)
            write(plan.output, index, plan.attributes[0] ? block_scaled_gate(plan, gate, up, read(plan.inputs[plan.count - 1], token)) : gate);
    }
}

// Prefill: decode only a 16-output, 32-column shard into shared BF16, then use tensor cores.
__global__ void block_scaled_gemm(const __grid_constant__ Launch plan) {
    using namespace nvcuda;
    __shared__ __align__(32) __nv_bfloat16 a[16 * 32], b[16 * 32];
    __shared__ __align__(32) float result[16 * 16];
    const auto width = plan.inputs[0].shape[plan.inputs[0].rank - 1];
    const auto rows = plan.inputs[1].shape[0];
    const auto tokens = plan.elements / plan.output.shape[plan.output.rank - 1];
    const auto token_start = static_cast<std::int64_t>(blockIdx.y) * 16;
    const auto row_start = static_cast<std::int64_t>(blockIdx.x) * 16;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> af;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> bf;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> accumulator;
    wmma::fill_fragment(accumulator, 0.F);
    for (std::int64_t start = 0; start < width; start += 32) {
        for (int index = threadIdx.x; index < 16 * 32; index += 32) {
            const auto outer = index / 32;
            const auto column = start + index % 32;
            a[index] = __float2bfloat16(token_start + outer < tokens && column < width
                                            ? read(plan.inputs[0], (token_start + outer) * width + column)
                                            : 0.F);
            b[index] = __float2bfloat16(row_start + outer < rows && column < width
                                            ? block_scaled_weight(plan, row_start + outer, column)
                                            : 0.F);
        }
        __syncwarp();
        for (int offset = 0; offset < 32; offset += 16) {
            wmma::load_matrix_sync(af, a + offset, 32);
            wmma::load_matrix_sync(bf, b + offset, 32);
            wmma::mma_sync(accumulator, af, bf, accumulator);
        }
        __syncwarp();
    }
    wmma::store_matrix_sync(result, accumulator, 16, wmma::mem_row_major);
    __syncwarp();
    const auto output_width = plan.output.shape[plan.output.rank - 1];
    const int tile_width = plan.attributes[0] ? 8 : 16;
    for (int index = threadIdx.x; index < 16 * tile_width; index += 32) {
        const auto token = token_start + index / tile_width;
        const auto column = row_start / (plan.attributes[0] ? 2 : 1) + index % tile_width;
        if (token >= tokens || column >= output_width) continue;
        const auto position = (index / tile_width) * 16 + index % tile_width * (plan.attributes[0] ? 2 : 1);
        const auto value = plan.attributes[0] ? block_scaled_gate(plan, result[position], result[position + 1],
                                                                  read(plan.inputs[plan.count - 1], token))
                                              : result[position];
        write(plan.output, token * output_width + column, value);
    }
}

// Tiled prefill GEMM over block-scaled weights: each 128-thread block computes up to 64 activation rows against a
// 64-row weight tile, decoding 16-byte weight chunks (NVFP4 with E4M3 scales per 16 values, or MXFP8 with UE8M0
// scales) in place into shared BF16 for tensor cores. Gated weights keep their interleaved gate/up rows, so each
// tile yields 32 activated outputs. The next K step's weights and activations are loaded into registers while the
// current one multiplies.
constexpr int TILE = 64, TILE_STRIDE = TILE + 8, TILE_THREADS = 128;

struct TileShared {
    __align__(32) __nv_bfloat16 a[TILE][TILE_STRIDE];
    __align__(32) __nv_bfloat16 b[TILE][TILE_STRIDE];
    __align__(32) float c[TILE][TILE + 4];
    // Activation row and output row of each tile row (-1 when unused), and its routing weight.
    int source[TILE], target[TILE];
    float routing[TILE];
};

template <int FORMAT>
struct TileLoads {
    static constexpr int CHUNKS = FORMAT == 1 ? 2 : 1;
    uint4 weights[CHUNKS];
    unsigned scales[CHUNKS];
};

template <int FORMAT>
__device__ void load_weight_tile(TileLoads<FORMAT>& loads, const unsigned char* values, const unsigned char* scales,
                                 int shift, std::int64_t row0, int width, int k0) {
    const int t = threadIdx.x;
    if constexpr (FORMAT == 0) {
        const auto index = (row0 + (t >> 1)) * width + k0 + (t & 1) * 32;
        loads.weights[0] = __ldg(reinterpret_cast<const uint4*>(values + index / 2));
        loads.scales[0] = __ldg(reinterpret_cast<const unsigned short*>(scales + (index >> 4)));
    } else {
#pragma unroll
        for (int chunk = 0; chunk < 2; ++chunk) {
            const int item = t + chunk * TILE_THREADS;
            const auto index = (row0 + (item >> 2)) * width + k0 + (item & 3) * 16;
            loads.weights[chunk] = __ldg(reinterpret_cast<const uint4*>(values + index));
            loads.scales[chunk] = __ldg(scales + (index >> shift));
        }
    }
}

template <int FORMAT>
__device__ void store_weight_tile(TileShared& shared, const TileLoads<FORMAT>& loads, float global) {
    const int t = threadIdx.x;
    if constexpr (FORMAT == 0) {
        const auto raw = __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(loads.scales[0]), __NV_E4M3);
        const auto pair = __half22float2(*reinterpret_cast<const __half2*>(&raw));
        // nibble_pair yields FP4 values times 2^-14.
        const float scale[2] = {pair.x * global * 16384.F, pair.y * global * 16384.F};
        const unsigned words[4] = {loads.weights[0].x, loads.weights[0].y, loads.weights[0].z, loads.weights[0].w};
        __align__(16) __nv_bfloat16 decoded[32];
#pragma unroll
        for (int word = 0; word < 4; ++word)
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const auto value = __half22float2(nibble_pair<0>(words[word], i));
                decoded[word * 8 + i] = __float2bfloat16(value.x * scale[word / 2]);
                decoded[word * 8 + 4 + i] = __float2bfloat16(value.y * scale[word / 2]);
            }
        auto* target = reinterpret_cast<uint4*>(&shared.b[t >> 1][(t & 1) * 32]);
#pragma unroll
        for (int i = 0; i < 4; ++i) target[i] = reinterpret_cast<const uint4*>(decoded)[i];
    } else {
#pragma unroll
        for (int chunk = 0; chunk < 2; ++chunk) {
            const int item = t + chunk * TILE_THREADS;
            const unsigned byte = loads.scales[chunk];
            const float scale = (byte == 255 ? nanf("") : __uint_as_float(byte ? byte << 23 : 0x00400000)) * global;
            const unsigned words[4] = {loads.weights[chunk].x, loads.weights[chunk].y, loads.weights[chunk].z,
                                       loads.weights[chunk].w};
            __align__(16) __nv_bfloat162 decoded[8];
#pragma unroll
            for (int word = 0; word < 4; ++word)
#pragma unroll
                for (int i = 0; i < 2; ++i) {
                    const auto raw = __nv_cvt_fp8x2_to_halfraw2(
                        static_cast<__nv_fp8x2_storage_t>(words[word] >> (16 * i)), __NV_E4M3);
                    const auto value = __half22float2(*reinterpret_cast<const __half2*>(&raw));
                    decoded[word * 2 + i] = __floats2bfloat162_rn(value.x * scale, value.y * scale);
                }
            auto* target = reinterpret_cast<uint4*>(&shared.b[item >> 2][(item & 3) * 16]);
            target[0] = reinterpret_cast<const uint4*>(decoded)[0];
            target[1] = reinterpret_cast<const uint4*>(decoded)[1];
        }
    }
}

// BF16 activation rows load as 8-value chunks; unused tile rows read as zeros.
__device__ void load_activation_tile(uint4* loads, const TileShared& shared, const __nv_bfloat16* input, int width,
                                     int k0) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int item = threadIdx.x + i * TILE_THREADS;
        const int source = shared.source[item >> 3];
        loads[i] = source < 0 ? make_uint4(0, 0, 0, 0)
                              : __ldg(reinterpret_cast<const uint4*>(
                                    input + static_cast<std::int64_t>(source) * width + k0 + (item & 7) * 8));
    }
}

__device__ void store_activation_tile(TileShared& shared, const uint4* loads) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int item = threadIdx.x + i * TILE_THREADS;
        *reinterpret_cast<uint4*>(&shared.a[item >> 3][(item & 7) * 8]) = loads[i];
    }
}

// Multiplies the rows listed in `shared.source` by weight rows [row0, row0 + 64) and writes `count` result rows:
// gated tiles apply the softcapped SiLU gate and routing weight; `partial` (routed down projections) receives
// unsummed FP32 rows, otherwise results go to the output tensor.
template <bool GATED, int FORMAT>
__device__ void block_scaled_tile(const Launch& plan, TileShared& shared, const unsigned char* values,
                                  const unsigned char* scales, float global, std::int64_t row0, int count,
                                  float* partial) {
    using namespace nvcuda;
    const int width = static_cast<int>(plan.inputs[0].shape[plan.inputs[0].rank - 1]);
    const int shift = static_cast<int>(plan.attributes[6]);
    const auto* input = static_cast<const __nv_bfloat16*>(address(plan.inputs[0]));
    const int warp = threadIdx.x / 32;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> accumulators[TILE / 16];
#pragma unroll
    for (auto& accumulator : accumulators) wmma::fill_fragment(accumulator, 0.F);
    TileLoads<FORMAT> weights;
    uint4 activations[4];
    load_weight_tile(weights, values, scales, shift, row0, width, 0);
    load_activation_tile(activations, shared, input, width, 0);
    for (int k0 = 0; k0 < width; k0 += TILE) {
        __syncthreads();
        store_weight_tile(shared, weights, global);
        store_activation_tile(shared, activations);
        __syncthreads();
        if (k0 + TILE < width) {
            load_weight_tile(weights, values, scales, shift, row0, width, k0 + TILE);
            load_activation_tile(activations, shared, input, width, k0 + TILE);
        }
#pragma unroll
        for (int kk = 0; kk < TILE; kk += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a;
            wmma::load_matrix_sync(a, &shared.a[warp * 16][kk], TILE_STRIDE);
#pragma unroll
            for (int n = 0; n < TILE / 16; ++n) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b;
                wmma::load_matrix_sync(b, &shared.b[n * 16][kk], TILE_STRIDE);
                wmma::mma_sync(accumulators[n], a, b, accumulators[n]);
            }
        }
    }
#pragma unroll
    for (int n = 0; n < TILE / 16; ++n)
        wmma::store_matrix_sync(&shared.c[warp * 16][n * 16], accumulators[n], TILE + 4, wmma::mem_row_major);
    __syncthreads();
    const auto rows = plan.inputs[1].shape[plan.inputs[1].rank - 2];
    if constexpr (GATED) {
        for (int index = threadIdx.x; index < count * TILE / 2; index += TILE_THREADS) {
            const int m = index / (TILE / 2), j = index % (TILE / 2);
            write(plan.output, static_cast<std::int64_t>(shared.target[m]) * (rows / 2) + row0 / 2 + j,
                  block_scaled_gate(plan, shared.c[m][2 * j], shared.c[m][2 * j + 1], shared.routing[m]));
        }
    } else {
        for (int index = threadIdx.x; index < count * TILE; index += TILE_THREADS) {
            const int m = index / TILE, j = index % TILE;
            const auto at = static_cast<std::int64_t>(shared.target[m]) * rows + row0 + j;
            if (partial)
                partial[at] = shared.c[m][j];
            else
                write(plan.output, at, shared.c[m][j]);
        }
    }
    __syncthreads();
}

template <bool GATED, int FORMAT>
__global__ void __launch_bounds__(TILE_THREADS) block_scaled_tile_gemm(const __grid_constant__ Launch plan) {
    __shared__ TileShared shared;
    const auto tokens = static_cast<std::int64_t>(plan.elements / plan.output.shape[plan.output.rank - 1]);
    const auto token0 = static_cast<std::int64_t>(blockIdx.y) * TILE;
    const int count = static_cast<int>(min(static_cast<std::int64_t>(TILE), tokens - token0));
    if (threadIdx.x < TILE) {
        const bool used = static_cast<int>(threadIdx.x) < count;
        shared.source[threadIdx.x] = shared.target[threadIdx.x] = used ? static_cast<int>(token0 + threadIdx.x) : -1;
        shared.routing[threadIdx.x] = GATED && used ? read(plan.inputs[plan.count - 1], token0 + threadIdx.x) : 1.F;
    }
    __syncthreads();
    block_scaled_tile<GATED, FORMAT>(
        plan, shared, static_cast<const unsigned char*>(address(plan.inputs[1])),
        static_cast<const unsigned char*>(address(plan.inputs[2])), plan.attributes[5] ? read(plan.inputs[3], 0) : 1.F,
        static_cast<std::int64_t>(blockIdx.x) * TILE, count, nullptr);
}

struct MoeBlock {
    int start;
    int expert_and_count;
};

constexpr int MARLIN_M_SMALL = 16;
constexpr int MARLIN_M_LARGE = 32;
constexpr int MARLIN_N_SMALL = 64;
constexpr int MARLIN_N_LARGE = 128;
constexpr int MARLIN_K = 64;
constexpr int MARLIN_STAGES = 3;
constexpr int MARLIN_THREADS = 128;

template <int FORMAT, int M_TILE, int N_TILE>
struct MarlinMoeShared {
    static constexpr int B_STRIDE = FORMAT == 0 ? 48 : 80;
    __align__(16) __nv_bfloat16 a[MARLIN_STAGES][M_TILE][MARLIN_K];
    __align__(16) unsigned char b[MARLIN_STAGES][N_TILE][B_STRIDE];
    __align__(16) unsigned s[MARLIN_STAGES][N_TILE];
    int source[M_TILE];
    int target[M_TILE];
    float routing[M_TILE];
};

// Buckets routed (token, slot) pairs by expert and emits dense active M_TILE blocks (moe_align_block_size style),
// eliminating empty/underfilled expert CTAs and matching 16/32-row Tensor Core tiles.
template <int M_TILE>
__global__ void sort_routes_blocks(const __grid_constant__ Launch plan, int* active_blocks, MoeBlock* blocks_out,
                                   int* routes_by_expert) {
    __shared__ int counts[1024], block_offsets[1024];
    const auto& routes = plan.inputs[plan.count - 1];
    const auto* data = static_cast<const float*>(address(routes));
    const int experts = static_cast<int>(plan.inputs[1].shape[0]);
    const int total = static_cast<int>(routes.shape[0] * routes.shape[1]);
    for (int expert = threadIdx.x; expert < experts; expert += blockDim.x) counts[expert] = 0;
    __syncthreads();
    for (int route = threadIdx.x; route < total; route += blockDim.x) {
        const int expert = static_cast<int>(data[route * 2]);
        if (expert >= 0 && expert < experts) atomicAdd(&counts[expert], 1);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        int offset = 0, block_offset = 0;
        for (int expert = 0; expert < experts; ++expert) {
            const int count = counts[expert];
            counts[expert] = offset;
            block_offsets[expert] = block_offset;
            offset += count;
            block_offset += (count + M_TILE - 1) / M_TILE;
        }
        counts[experts] = offset;
        block_offsets[experts] = block_offset;
        *active_blocks = block_offset;
    }
    __syncthreads();
    if (threadIdx.x < experts) {
        const int expert = static_cast<int>(threadIdx.x);
        const int begin = counts[expert];
        const int b0 = block_offsets[expert], b1 = block_offsets[expert + 1];
        const int n_blocks = b1 - b0;
        if (n_blocks > 0) {
            const int total_expert_routes = counts[expert + 1] - begin;
            for (int b = 0; b < n_blocks; ++b) {
                const int start = begin + b * M_TILE;
                const int cnt = min(M_TILE, total_expert_routes - b * M_TILE);
                blocks_out[b0 + b] = MoeBlock{start, (expert << 8) | cnt};
            }
        }
    }
    __syncthreads();
    for (int route = threadIdx.x; route < total; route += blockDim.x) {
        const int expert = static_cast<int>(data[route * 2]);
        if (expert >= 0 && expert < experts) routes_by_expert[atomicAdd(&counts[expert], 1)] = route;
    }
}

template <int FORMAT, int M_TILE, int N_TILE>
__device__ __forceinline__ void marlin_issue_stage(MarlinMoeShared<FORMAT, M_TILE, N_TILE>& shared, int stage,
                                                   const __nv_bfloat16* input, const unsigned char* values,
                                                   const unsigned char* scales, std::int64_t row0, int width,
                                                   int shift, int k0, bool scales_aligned4) {
    const int t = static_cast<int>(threadIdx.x);
#pragma unroll
    for (int item = t; item < M_TILE * 8; item += MARLIN_THREADS) {
        const int r = item >> 3;
        const int c = item & 7;
        const int src_row = shared.source[r];
        const void* src_ptr =
            src_row >= 0 ? input + static_cast<std::int64_t>(src_row) * width + k0 + c * 8 : nullptr;
        ptx::cp_async_zfill_16<false>(&shared.a[stage][r][(c ^ (r & 7)) * 8], src_ptr, src_row >= 0);
    }
    if constexpr (FORMAT == 0) {
#pragma unroll
        for (int item = t; item < N_TILE * 2; item += MARLIN_THREADS) {
            const int r = item >> 1;
            const int h = item & 1;
            const unsigned char* src = values + (row0 + r) * (width >> 1) + (k0 >> 1) + h * 16;
            ptx::cp_async_zfill_16<true>(&shared.b[stage][r][h * 16], src, true);
        }
    } else {
#pragma unroll
        for (int item = t; item < N_TILE * 4; item += MARLIN_THREADS) {
            const int r = item >> 2;
            const int c = item & 3;
            const unsigned char* src = values + (row0 + r) * width + k0 + c * 16;
            ptx::cp_async_zfill_16<true>(&shared.b[stage][r][c * 16], src, true);
        }
    }
    if (t < N_TILE) {
        const auto* src_s = scales + (row0 + t) * (width >> shift) + (k0 >> shift);
        if constexpr (FORMAT == 0) {
            if (scales_aligned4) {
                ptx::cp_async_4(&shared.s[stage][t], src_s);
            } else {
                shared.s[stage][t] = static_cast<unsigned>(src_s[0]) | (static_cast<unsigned>(src_s[1]) << 8) |
                                     (static_cast<unsigned>(src_s[2]) << 16) |
                                     (static_cast<unsigned>(src_s[3]) << 24);
            }
        } else {
            unsigned packed = static_cast<unsigned>(src_s[0]);
            if (shift == 5) {
                packed |= static_cast<unsigned>(src_s[1]) << 8;
            } else if (shift == 4) {
                packed |= (static_cast<unsigned>(src_s[1]) << 8) | (static_cast<unsigned>(src_s[2]) << 16) |
                          (static_cast<unsigned>(src_s[3]) << 24);
            }
            shared.s[stage][t] = packed;
        }
    }
}

template <bool GATED, int FORMAT, int M_TILE, int N_TILE>
__device__ __forceinline__ void marlin_block_scaled_tile(const Launch& plan,
                                                         MarlinMoeShared<FORMAT, M_TILE, N_TILE>& shared,
                                                         const unsigned char* values, const unsigned char* scales,
                                                         float global, std::int64_t row0, int count, float* partial) {
    constexpr int M_TILES = M_TILE / 16;
    constexpr int WARPS = MARLIN_THREADS / 32;
    constexpr int N_WARP_TILES = (N_TILE / WARPS) / 8;
    const int width = static_cast<int>(plan.inputs[0].shape[plan.inputs[0].rank - 1]);
    const int shift = static_cast<int>(plan.attributes[6]);
    const auto* input = static_cast<const __nv_bfloat16*>(address(plan.inputs[0]));
    const bool scales_aligned4 = FORMAT == 0 && (reinterpret_cast<std::uintptr_t>(scales) & 3) == 0 &&
                                 ((width >> shift) & 3) == 0;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int group = lane >> 2;
    const int tid4 = lane & 3;
    const int n_base = warp * (N_TILE / WARPS);

    ptx::Fragment<float, 4> o_acc[M_TILES][N_WARP_TILES];
#pragma unroll
    for (int m = 0; m < M_TILES; ++m)
#pragma unroll
        for (int n = 0; n < N_WARP_TILES; ++n) o_acc[m][n].clear();

    const int num_k_tiles = width / MARLIN_K;
#pragma unroll
    for (int s = 0; s < MARLIN_STAGES - 1; ++s) {
        if (s < num_k_tiles) {
            marlin_issue_stage<FORMAT, M_TILE, N_TILE>(shared, s, input, values, scales, row0, width, shift,
                                                       s * MARLIN_K, scales_aligned4);
        }
        ptx::cp_async_fence();
    }

    for (int k_iter = 0; k_iter < num_k_tiles; ++k_iter) {
        const int stage = k_iter % MARLIN_STAGES;
        const int prefetch_iter = k_iter + (MARLIN_STAGES - 1);
        if (prefetch_iter < num_k_tiles) {
            marlin_issue_stage<FORMAT, M_TILE, N_TILE>(shared, prefetch_iter % MARLIN_STAGES, input, values, scales,
                                                       row0, width, shift, prefetch_iter * MARLIN_K, scales_aligned4);
        }
        ptx::cp_async_fence();
        ptx::cp_async_wait<MARLIN_STAGES - 1>();
        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
            ptx::Fragment<unsigned, 4> a_frag[M_TILES];
#pragma unroll
            for (int m = 0; m < M_TILES; ++m) {
                const int a_row = m * 16 + (lane & 15);
                const int a_chunk = (kk * 2 + (lane >> 4)) ^ (a_row & 7);
                ptx::ldmatrix_x4<false>(a_frag[m], &shared.a[stage][a_row][a_chunk * 8]);
            }

#pragma unroll
            for (int n_tile = 0; n_tile < N_WARP_TILES; ++n_tile) {
                const int b_row = n_base + n_tile * 8 + group;
                ptx::Fragment<unsigned, 2> b_frag;
                if constexpr (FORMAT == 0) {
                    const uint2 raw = *reinterpret_cast<const uint2*>(&shared.b[stage][b_row][kk * 8]);
                    const unsigned byte_lo = (raw.x >> (8 * tid4)) & 0xFFU;
                    const unsigned byte_hi = (raw.y >> (8 * tid4)) & 0xFFU;
                    const unsigned s_byte = (shared.s[stage][b_row] >> (8 * kk)) & 0xFFU;
                    const auto s_half = __nv_cvt_fp8_to_halfraw(static_cast<__nv_fp8_storage_t>(s_byte), __NV_E4M3);
                    const float s_scaled = __half2float(*reinterpret_cast<const __half*>(&s_half)) * global;

                    const unsigned p0 = (byte_lo & 0x0FU) | ((byte_lo & 0xF0U) << 12);
                    const unsigned p1 = (byte_hi & 0x0FU) | ((byte_hi & 0xF0U) << 12);
                    const unsigned h0 = ((p0 & 0x00080008U) << 12) | ((p0 & 0x00070007U) << 6);
                    const unsigned h1 = ((p1 & 0x00080008U) << 12) | ((p1 & 0x00070007U) << 6);
                    constexpr unsigned bf16_bias_126 = 0x7E807E80U;
                    const __nv_bfloat162 bias_reg = *reinterpret_cast<const __nv_bfloat162*>(&bf16_bias_126);
                    const __nv_bfloat162 s_bf162 = __float2bfloat162_rn(s_scaled);
                    const __nv_bfloat162 bf0 =
                        __hmul2(__hmul2(*reinterpret_cast<const __nv_bfloat162*>(&h0), bias_reg), s_bf162);
                    const __nv_bfloat162 bf1 =
                        __hmul2(__hmul2(*reinterpret_cast<const __nv_bfloat162*>(&h1), bias_reg), s_bf162);
                    b_frag[0] = *reinterpret_cast<const unsigned*>(&bf0);
                    b_frag[1] = *reinterpret_cast<const unsigned*>(&bf1);
                } else {
                    const uint4 raw = *reinterpret_cast<const uint4*>(&shared.b[stage][b_row][kk * 16]);
                    const unsigned w01 = ((tid4 < 2 ? raw.x : raw.y) >> (16 * (tid4 & 1))) & 0xFFFFU;
                    const unsigned w23 = ((tid4 < 2 ? raw.z : raw.w) >> (16 * (tid4 & 1))) & 0xFFFFU;
                    const int s_idx = (kk * 16) >> shift;
                    const unsigned s_byte = (shared.s[stage][b_row] >> (8 * s_idx)) & 0xFFU;
                    const float s_scaled =
                        (s_byte == 255 ? nanf("") : __uint_as_float(s_byte ? s_byte << 23 : 0x00400000)) * global;
                    const auto h0 = __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(w01), __NV_E4M3);
                    const auto h1 = __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(w23), __NV_E4M3);
                    const float2 f0 = __half22float2(*reinterpret_cast<const __half2*>(&h0));
                    const float2 f1 = __half22float2(*reinterpret_cast<const __half2*>(&h1));
                    const __nv_bfloat162 bf0 = __floats2bfloat162_rn(f0.x * s_scaled, f0.y * s_scaled);
                    const __nv_bfloat162 bf1 = __floats2bfloat162_rn(f1.x * s_scaled, f1.y * s_scaled);
                    b_frag[0] = *reinterpret_cast<const unsigned*>(&bf0);
                    b_frag[1] = *reinterpret_cast<const unsigned*>(&bf1);
                }
#pragma unroll
                for (int m = 0; m < M_TILES; ++m) {
                    ptx::mma_bf16_m16n8k16(o_acc[m][n_tile], a_frag[m], b_frag, o_acc[m][n_tile]);
                }
            }
        }
        __syncthreads();
    }

    const auto rows = plan.inputs[1].shape[plan.inputs[1].rank - 2];
#pragma unroll
    for (int m = 0; m < M_TILES; ++m) {
        const int m0 = m * 16 + group;
        const int m1 = m0 + 8;
#pragma unroll
        for (int n_tile = 0; n_tile < N_WARP_TILES; ++n_tile) {
            if constexpr (GATED) {
                const auto out_col = row0 / 2 + n_base / 2 + n_tile * 4 + tid4;
                if (m0 < count) {
                    write(plan.output, static_cast<std::int64_t>(shared.target[m0]) * (rows / 2) + out_col,
                          block_scaled_gate(plan, o_acc[m][n_tile][0], o_acc[m][n_tile][1], shared.routing[m0]));
                }
                if (m1 < count) {
                    write(plan.output, static_cast<std::int64_t>(shared.target[m1]) * (rows / 2) + out_col,
                          block_scaled_gate(plan, o_acc[m][n_tile][2], o_acc[m][n_tile][3], shared.routing[m1]));
                }
            } else {
                const auto out_col = row0 + n_base + n_tile * 8 + tid4 * 2;
                if (partial) {
                    if (m0 < count) {
                        *reinterpret_cast<float2*>(&partial[static_cast<std::int64_t>(shared.target[m0]) * rows +
                                                            out_col]) =
                            make_float2(o_acc[m][n_tile][0], o_acc[m][n_tile][1]);
                    }
                    if (m1 < count) {
                        *reinterpret_cast<float2*>(&partial[static_cast<std::int64_t>(shared.target[m1]) * rows +
                                                            out_col]) =
                            make_float2(o_acc[m][n_tile][2], o_acc[m][n_tile][3]);
                    }
                } else {
                    if (m0 < count) {
                        const auto at = static_cast<std::int64_t>(shared.target[m0]) * rows + out_col;
                        write(plan.output, at, o_acc[m][n_tile][0]);
                        write(plan.output, at + 1, o_acc[m][n_tile][1]);
                    }
                    if (m1 < count) {
                        const auto at = static_cast<std::int64_t>(shared.target[m1]) * rows + out_col;
                        write(plan.output, at, o_acc[m][n_tile][2]);
                        write(plan.output, at + 1, o_acc[m][n_tile][3]);
                    }
                }
            }
        }
    }
}

// Grid (weight-row tiles, active M_TILE blocks): each block processes one active expert tile directly from
// compact shared memory via cp.async + register dequantization + mma.sync.m16n8k16.
template <bool GATED, int FORMAT, int M_TILE, int N_TILE>
__global__ void __launch_bounds__(MARLIN_THREADS)
    marlin_expert_tile_gemm(const __grid_constant__ Launch plan, const int* active_blocks, const MoeBlock* blocks_in,
                            const int* routes_by_expert, float* partial) {
    const int tile_idx = static_cast<int>(blockIdx.y);
    if (tile_idx >= *active_blocks) return;
    __shared__ MarlinMoeShared<FORMAT, M_TILE, N_TILE> shared;
    const MoeBlock blk = blocks_in[tile_idx];
    const int expert = blk.expert_and_count >> 8;
    const int count = blk.expert_and_count & 0xFF;
    const int start = blk.start;

    const auto& routes = plan.inputs[plan.count - 1];
    const auto* route_data = static_cast<const float*>(address(routes));
    const int slots = static_cast<int>(routes.shape[1]);
    const auto rows = plan.inputs[1].shape[1];
    const auto width = plan.inputs[0].shape[plan.inputs[0].rank - 1];
    const auto expert_elements = static_cast<std::int64_t>(expert) * rows * width;
    const auto* values =
        static_cast<const unsigned char*>(address(plan.inputs[1])) + (expert_elements >> (FORMAT == 1 ? 0 : 1));
    const auto* scales =
        static_cast<const unsigned char*>(address(plan.inputs[2])) + (expert_elements >> plan.attributes[6]);
    const float global = plan.attributes[5] ? read(plan.inputs[3], expert) : 1.F;

    if (threadIdx.x < M_TILE) {
        const int route = static_cast<int>(threadIdx.x) < count ? routes_by_expert[start + threadIdx.x] : -1;
        shared.source[threadIdx.x] = route < 0 ? -1 : GATED ? route / slots : route;
        shared.target[threadIdx.x] = route;
        shared.routing[threadIdx.x] = GATED && route >= 0 ? route_data[route * 2 + 1] : 1.F;
    }
    __syncthreads();
    marlin_block_scaled_tile<GATED, FORMAT, M_TILE, N_TILE>(
        plan, shared, values, scales, global, static_cast<std::int64_t>(blockIdx.x) * N_TILE, count,
        GATED ? nullptr : partial);
}

// Sums [tokens, slots, rows] partial projections over slots, in slot order.
__global__ void sum_slots(const __grid_constant__ Launch plan, const float* partial) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= plan.elements) return;
    const auto rows = plan.output.shape[plan.output.rank - 1];
    const auto slots = plan.inputs[plan.count - 1].shape[1];
    const auto token = static_cast<std::int64_t>(index / rows), row = static_cast<std::int64_t>(index % rows);
    float sum = 0;
    for (std::int64_t slot = 0; slot < slots; ++slot) sum += partial[(token * slots + slot) * rows + row];
    write(plan.output, index, sum);
}

__device__ float attribute_float(const Launch& plan, int index) {
    return __int_as_float(static_cast<int>(plan.attributes[index]));
}

// 8-byte BF16 vector helpers for register-cached kernels.
__device__ __forceinline__ void load_bf16x4(const __nv_bfloat16* p, float out[4]) {
    const uint2 raw = __ldg(reinterpret_cast<const uint2*>(p));
    const auto* halves = reinterpret_cast<const __nv_bfloat162*>(&raw);
    const float2 p0 = __bfloat1622float2(halves[0]);
    const float2 p1 = __bfloat1622float2(halves[1]);
    out[0] = p0.x;
    out[1] = p0.y;
    out[2] = p1.x;
    out[3] = p1.y;
}

__device__ __forceinline__ void store_bf16x4(__nv_bfloat16* p, const float in[4]) {
    const __nv_bfloat162 pairs[2] = {__floats2bfloat162_rn(in[0], in[1]), __floats2bfloat162_rn(in[2], in[3])};
    *reinterpret_cast<uint2*>(p) = *reinterpret_cast<const uint2*>(pairs);
}

constexpr int MIX_FAST_THREADS = 576;
constexpr int MIX_FAST_WIDTH = 2304;

/// One block per token: RMS-normalizes each source, softmax pseudo-query logits over sources, mix.
__global__ void __launch_bounds__(MIX_THREADS) residual_mix(Launch plan) {
    // Every norm and query dot product accumulates in one pass and reduces once, instead of one block-wide
    // reduction per (query, source); decode runs a single block, so that dependency chain was the cost.
    constexpr int SUMS = MAX_SOURCES * (MAX_QUERIES + 1);
    __shared__ float reduction[MIX_THREADS / 32][SUMS];
    __shared__ float inverse[MAX_SOURCES], weights[MAX_QUERIES][MAX_SOURCES];
    const int sources = plan.count - 1;
    const auto& queries = plan.inputs[sources];
    const auto width = queries.shape[1], queries_count = queries.shape[0], tokens = plan.inputs[0].shape[0];
    const auto token = blockIdx.x;
    float sums[SUMS]{};
    for (auto column = threadIdx.x; column < width; column += blockDim.x) {
        float query_values[MAX_QUERIES];
#pragma unroll
        for (int query = 0; query < MAX_QUERIES; ++query)
            query_values[query] = query < queries_count ? read(queries, query * width + column) : 0.F;
#pragma unroll
        for (int source = 0; source < MAX_SOURCES; ++source)
            if (source < sources) {
                const float value = read(plan.inputs[source], token * width + column);
                sums[source] += value * value;
#pragma unroll
                for (int query = 0; query < MAX_QUERIES; ++query)
                    sums[MAX_SOURCES * (query + 1) + source] += query_values[query] * value;
            }
    }
#pragma unroll
    for (int index = 0; index < SUMS; ++index) {
        const float value = warp_sum(sums[index]);
        if (threadIdx.x % 32 == 0) reduction[threadIdx.x / 32][index] = value;
    }
    __syncthreads();
    if (threadIdx.x < SUMS) {
        float total = 0;
        for (int warp = 0; warp < static_cast<int>(blockDim.x / 32); ++warp) total += reduction[warp][threadIdx.x];
        reduction[0][threadIdx.x] = total;
    }
    __syncthreads();
    if (threadIdx.x < sources) inverse[threadIdx.x] = rsqrtf(reduction[0][threadIdx.x] / width + plan.epsilon);
    __syncthreads();
    if (threadIdx.x < queries_count)
        for (int source = 0; source < sources; ++source)
            weights[threadIdx.x][source] = reduction[0][MAX_SOURCES * (threadIdx.x + 1) + source] * inverse[source] *
                                           rsqrtf(static_cast<float>(width));
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

/// Single-pass shared-memory-cached residual_mix for width <= 2304, F32 queries/output, BF16/F32 sources.
template <int MAX_S, int MAX_Q>
__global__ void __launch_bounds__(MIX_FAST_THREADS) residual_mix_fast(const __grid_constant__ Launch plan) {
    constexpr int SUMS = MAX_S * (MAX_Q + 1);
    __shared__ float reduction[MIX_FAST_THREADS / 32][SUMS];
    __shared__ float inverse[MAX_S], weights[MAX_Q][MAX_S];
    __shared__ __align__(16) __nv_bfloat16 cache_bf16[MAX_S > 1 ? MAX_S - 1 : 1][MIX_FAST_WIDTH];
    __shared__ __align__(16) float cache_last[MIX_FAST_WIDTH];
    const int sources = plan.count - 1;
    const auto& queries = plan.inputs[sources];
    const auto width = queries.shape[1];
    const int queries_count = static_cast<int>(queries.shape[0]);
    const auto tokens = plan.inputs[0].shape[0];
    const auto token = static_cast<std::int64_t>(blockIdx.x);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    float sums[SUMS];
#pragma unroll
    for (int i = 0; i < SUMS; ++i) sums[i] = 0.F;
    const auto* q_ptr = static_cast<const float*>(address(queries));
    auto* out = static_cast<float*>(address(plan.output));
    bool vec4 = (width & 3) == 0 && width <= MIX_FAST_WIDTH &&
                (reinterpret_cast<std::uintptr_t>(q_ptr) & 15) == 0 &&
                (reinterpret_cast<std::uintptr_t>(out) & 15) == 0;
#pragma unroll
    for (int s = 0; s < MAX_S; ++s) {
        if (s < sources && (reinterpret_cast<std::uintptr_t>(address(plan.inputs[s])) & 15) != 0) vec4 = false;
    }
    if (vec4) {
        const auto c4 = static_cast<std::int64_t>(threadIdx.x) * 4;
        if (c4 < width) {
            float4 q_vals[MAX_Q];
#pragma unroll
            for (int q = 0; q < MAX_Q; ++q)
                q_vals[q] = q < queries_count ? __ldg(reinterpret_cast<const float4*>(q_ptr + q * width + c4))
                                              : make_float4(0.F, 0.F, 0.F, 0.F);
#pragma unroll
            for (int s = 0; s < MAX_S; ++s) {
                if (s < sources) {
                    const void* ptr = address(plan.inputs[s]);
                    const bool is_bf16 = plan.inputs[s].type == static_cast<int>(Type::BF16);
                    float4 v4;
                    if (is_bf16) {
                        const uint2 raw =
                            __ldg(reinterpret_cast<const uint2*>(static_cast<const __nv_bfloat16*>(ptr) + token * width + c4));
                        const auto* h = reinterpret_cast<const __nv_bfloat162*>(&raw);
                        const float2 p0 = __bfloat1622float2(h[0]), p1 = __bfloat1622float2(h[1]);
                        v4 = make_float4(p0.x, p0.y, p1.x, p1.y);
                        if (s + 1 < sources)
                            *reinterpret_cast<uint2*>(&cache_bf16[s][c4]) = raw;
                        else
                            *reinterpret_cast<float4*>(&cache_last[c4]) = v4;
                    } else {
                        v4 = __ldg(reinterpret_cast<const float4*>(static_cast<const float*>(ptr) + token * width + c4));
                        if (s + 1 < sources) {
                            const __nv_bfloat162 h[2] = {__floats2bfloat162_rn(v4.x, v4.y),
                                                         __floats2bfloat162_rn(v4.z, v4.w)};
                            *reinterpret_cast<uint2*>(&cache_bf16[s][c4]) = *reinterpret_cast<const uint2*>(h);
                        } else {
                            *reinterpret_cast<float4*>(&cache_last[c4]) = v4;
                        }
                    }
                    sums[s] = fmaf(v4.x, v4.x, fmaf(v4.y, v4.y, fmaf(v4.z, v4.z, fmaf(v4.w, v4.w, sums[s]))));
#pragma unroll
                    for (int q = 0; q < MAX_Q; ++q) {
                        if (q < queries_count) {
                            const int idx = MAX_S * (q + 1) + s;
                            sums[idx] = fmaf(q_vals[q].x, v4.x,
                                             fmaf(q_vals[q].y, v4.y,
                                                  fmaf(q_vals[q].z, v4.z, fmaf(q_vals[q].w, v4.w, sums[idx]))));
                        }
                    }
                }
            }
        }
    } else {
        for (auto col = static_cast<std::int64_t>(threadIdx.x); col < width; col += MIX_FAST_THREADS) {
            float q_vals[MAX_Q];
#pragma unroll
            for (int q = 0; q < MAX_Q; ++q) q_vals[q] = q < queries_count ? __ldg(q_ptr + q * width + col) : 0.F;
#pragma unroll
            for (int s = 0; s < MAX_S; ++s) {
                if (s < sources) {
                    const void* ptr = address(plan.inputs[s]);
                    const bool is_bf16 = plan.inputs[s].type == static_cast<int>(Type::BF16);
                    const float v =
                        is_bf16 ? __bfloat162float(__ldg(static_cast<const __nv_bfloat16*>(ptr) + token * width + col))
                                : __ldg(static_cast<const float*>(ptr) + token * width + col);
                    if (s + 1 < sources)
                        cache_bf16[s][col] = __float2bfloat16(v);
                    else
                        cache_last[col] = v;
                    sums[s] = fmaf(v, v, sums[s]);
#pragma unroll
                    for (int q = 0; q < MAX_Q; ++q)
                        if (q < queries_count) sums[MAX_S * (q + 1) + s] = fmaf(q_vals[q], v, sums[MAX_S * (q + 1) + s]);
                }
            }
        }
    }
#pragma unroll
    for (int s = 0; s < MAX_S; ++s) {
        if (s < sources) {
            const float v = warp_sum(sums[s]);
            if (lane == 0) reduction[warp][s] = v;
#pragma unroll
            for (int q = 0; q < MAX_Q; ++q) {
                if (q < queries_count) {
                    const int idx = MAX_S * (q + 1) + s;
                    const float dq = warp_sum(sums[idx]);
                    if (lane == 0) reduction[warp][idx] = dq;
                }
            }
        }
    }
    __syncthreads();
    if (warp == 0) {
        for (int idx = lane; idx < (queries_count + 1) * MAX_S; idx += 32) {
            const int group = idx / MAX_S, s = idx % MAX_S;
            if (s < sources) {
                float total = 0;
#pragma unroll
                for (int w = 0; w < MIX_FAST_THREADS / 32; ++w) total += reduction[w][idx];
                if (group == 0)
                    inverse[s] = rsqrtf(total / static_cast<float>(width) + plan.epsilon);
                else
                    reduction[0][idx] = total;
            }
        }
    }
    __syncthreads();
    if (threadIdx.x < queries_count) {
        auto* row = weights[threadIdx.x];
        const float inv_sqrt_d = rsqrtf(static_cast<float>(width));
        float peak = -INFINITY;
        for (int s = 0; s < sources; ++s) {
            const float logit = reduction[0][MAX_S * (threadIdx.x + 1) + s] * inverse[s] * inv_sqrt_d;
            row[s] = logit;
            peak = fmaxf(peak, logit);
        }
        float total = 0;
        for (int s = 0; s < sources; ++s) total += row[s] = expf(row[s] - peak);
        const float inv_total = 1.F / total;
        for (int s = 0; s < sources; ++s) row[s] *= inv_total;
    }
    __syncthreads();
    if (vec4) {
        const auto c4 = static_cast<std::int64_t>(threadIdx.x) * 4;
        if (c4 < width) {
            const float4 last = *reinterpret_cast<const float4*>(&cache_last[c4]);
#pragma unroll
            for (int q = 0; q < MAX_Q; ++q) {
                if (q < queries_count) {
                    const float w_last = weights[q][sources - 1] * inverse[sources - 1];
                    float4 acc = make_float4(w_last * last.x, w_last * last.y, w_last * last.z, w_last * last.w);
#pragma unroll
                    for (int s = 0; s + 1 < MAX_S; ++s) {
                        if (s + 1 < sources) {
                            const float ws = weights[q][s] * inverse[s];
                            const uint2 raw = *reinterpret_cast<const uint2*>(&cache_bf16[s][c4]);
                            const auto* h = reinterpret_cast<const __nv_bfloat162*>(&raw);
                            const float2 p0 = __bfloat1622float2(h[0]), p1 = __bfloat1622float2(h[1]);
                            acc.x = fmaf(ws, p0.x, acc.x);
                            acc.y = fmaf(ws, p0.y, acc.y);
                            acc.z = fmaf(ws, p1.x, acc.z);
                            acc.w = fmaf(ws, p1.y, acc.w);
                        }
                    }
                    *reinterpret_cast<float4*>(out + (q * tokens + token) * width + c4) = acc;
                }
            }
        }
    } else {
        for (auto col = static_cast<std::int64_t>(threadIdx.x); col < width; col += MIX_FAST_THREADS) {
#pragma unroll
            for (int q = 0; q < MAX_Q; ++q) {
                if (q < queries_count) {
                    float acc = weights[q][sources - 1] * inverse[sources - 1] * cache_last[col];
#pragma unroll
                    for (int s = 0; s + 1 < MAX_S; ++s)
                        if (s + 1 < sources)
                            acc = fmaf(weights[q][s] * inverse[s], __bfloat162float(cache_bf16[s][col]), acc);
                    out[(q * tokens + token) * width + col] = acc;
                }
            }
        }
    }
}

/// Causal depthwise convolution over `padded` rows (the previous W - 1 inputs first), lag-0-first taps, or
/// ring-buffered over `input` and `state [ring, channels]` when `attribute_count >= 2` (with optional residual when `attribute_count == 3`).
__global__ void short_convolution(const __grid_constant__ Launch plan) {
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= plan.elements) return;
    const auto width = plan.attributes[0], channels = plan.output.shape[1];
    const auto token = static_cast<std::int64_t>(index / channels), channel = static_cast<std::int64_t>(index % channels);
    if (plan.attribute_count >= 2) {
        const auto ring = plan.attributes[1];
        const auto slot = static_cast<std::int64_t>(static_cast<const int*>(address(plan.inputs[3]))[0]);
        const auto flags = plan.attribute_count == 3 ? plan.attributes[2] : 0;
        const bool dynamic = plan.attribute_count == 3 ? ((flags & 1) != 0) : (plan.count == 5);
        const bool has_residual = plan.attribute_count == 3 && ((flags & 2) != 0);
        const bool update_state = plan.attribute_count == 3 && ((flags & 4) != 0);
        const auto groups = dynamic ? plan.inputs[4].shape[1] / width : 1;
        const auto group = channel / (channels / groups);
        const float cur_val = read(plan.inputs[0], token * channels + channel);
        float sum = cur_val;
        for (std::int64_t lag = 0; lag < width; ++lag) {
            float tap = read(plan.inputs[2], lag * channels + channel);
            if (dynamic) tap += read(plan.inputs[4], (token * groups + group) * width + lag);
            const auto prior = token - lag;
            float value;
            if (prior >= 0) {
                value = read(plan.inputs[0], prior * channels + channel);
            } else {
                const auto state_row = ((slot + prior) % ring + ring) % ring;
                value = read(plan.inputs[1], state_row * channels + channel);
            }
            sum += tap * value;
        }
        auto out = sum * plan.epsilon;
        if (has_residual) out += read(plan.inputs[plan.count - 1], token * channels + channel);
        if (update_state) {
            const auto state_index = ((slot % ring + ring) % ring) * channels + channel;
            if (plan.inputs[1].type == static_cast<int>(Type::BF16))
                static_cast<__nv_bfloat16*>(address(plan.inputs[1]))[state_index] = __float2bfloat16(cur_val);
            else if (plan.inputs[1].type == static_cast<int>(Type::F16))
                static_cast<__half*>(address(plan.inputs[1]))[state_index] = __float2half(cur_val);
            else
                static_cast<float*>(address(plan.inputs[1]))[state_index] = cur_val;
        }
        write(plan.output, index, out);
        return;
    }
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

/// Vectorized BF16 ring-buffered short_convolution for width == 4, ring == 4, channels % 8 == 0, with optional
/// residual addition and BF16 or F32 output.
__global__ void short_convolution_bf16_w4(const __grid_constant__ Launch plan) {
    const std::size_t vec_index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const auto channels = plan.output.shape[1];
    const auto vec_cols = channels / 8;
    const std::size_t total_vecs = plan.elements / 8;
    if (vec_index >= total_vecs) return;
    const auto token = static_cast<std::int64_t>(vec_index / vec_cols);
    const auto c0 = static_cast<std::int64_t>(vec_index % vec_cols) * 8;
    const int slot = static_cast<const int*>(address(plan.inputs[3]))[0] & 3;
    const auto* input = static_cast<const __nv_bfloat16*>(address(plan.inputs[0]));
    auto* state = static_cast<__nv_bfloat16*>(address(plan.inputs[1]));
    const auto* weight = static_cast<const __nv_bfloat16*>(address(plan.inputs[2]));
    const auto flags = plan.attribute_count == 3 ? plan.attributes[2] : 0;
    const bool dynamic = plan.attribute_count == 3 ? ((flags & 1) != 0) : (plan.count == 5);
    const bool has_residual = plan.attribute_count == 3 && ((flags & 2) != 0);
    const bool update_state = plan.attribute_count == 3 && ((flags & 4) != 0);
    const auto groups = dynamic ? plan.inputs[4].shape[1] / 4 : 1;
    const auto group = c0 / (channels / groups);

    float sum[8]{};
    const auto decode8 = [](const uint4& raw, float out[8]) {
        const auto* halves = reinterpret_cast<const __nv_bfloat162*>(&raw);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float2 pair = __bfloat1622float2(halves[i]);
            out[2 * i] = pair.x;
            out[2 * i + 1] = pair.y;
        }
    };
    const auto load8 = [&](const __nv_bfloat16* p, float out[8]) {
        decode8(__ldg(reinterpret_cast<const uint4*>(p)), out);
    };
    const uint4 raw_cur = __ldg(reinterpret_cast<const uint4*>(input + token * channels + c0));
    float cur[8];
    decode8(raw_cur, cur);
#pragma unroll
    for (int i = 0; i < 8; ++i) sum[i] = cur[i];
#pragma unroll
    for (int lag = 0; lag < 4; ++lag) {
        float w[8], x[8];
        load8(weight + lag * channels + c0, w);
        const float dyn = dynamic ? read(plan.inputs[4], (token * groups + group) * 4 + lag) : 0.F;
        if (lag == 0) {
#pragma unroll
            for (int i = 0; i < 8; ++i) x[i] = cur[i];
        } else if (token >= lag) {
            load8(input + (token - lag) * channels + c0, x);
        } else {
            const int row = (slot + static_cast<int>(token) - lag + 4) & 3;
            load8(state + row * channels + c0, x);
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) sum[i] = fmaf(w[i] + dyn, x[i], sum[i]);
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) sum[i] *= plan.epsilon;
    if (has_residual) {
        const auto& res_view = plan.inputs[plan.count - 1];
        const auto offset = token * channels + c0;
        if (res_view.type == static_cast<int>(Type::BF16)) {
            float res[8];
            load8(static_cast<const __nv_bfloat16*>(address(res_view)) + offset, res);
#pragma unroll
            for (int i = 0; i < 8; ++i) sum[i] += res[i];
        } else {
            const auto* rptr = static_cast<const float*>(address(res_view)) + offset;
            const float4 r0 = __ldg(reinterpret_cast<const float4*>(rptr));
            const float4 r1 = __ldg(reinterpret_cast<const float4*>(rptr + 4));
            sum[0] += r0.x;
            sum[1] += r0.y;
            sum[2] += r0.z;
            sum[3] += r0.w;
            sum[4] += r1.x;
            sum[5] += r1.y;
            sum[6] += r1.z;
            sum[7] += r1.w;
        }
    }
    const auto out_offset = token * channels + c0;
    if (plan.output.type == static_cast<int>(Type::BF16)) {
        __nv_bfloat162 out_pairs[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) out_pairs[i] = __floats2bfloat162_rn(sum[2 * i], sum[2 * i + 1]);
        *reinterpret_cast<uint4*>(static_cast<__nv_bfloat16*>(address(plan.output)) + out_offset) =
            *reinterpret_cast<const uint4*>(out_pairs);
    } else {
        auto* optr = static_cast<float*>(address(plan.output)) + out_offset;
        *reinterpret_cast<float4*>(optr) = make_float4(sum[0], sum[1], sum[2], sum[3]);
        *reinterpret_cast<float4*>(optr + 4) = make_float4(sum[4], sum[5], sum[6], sum[7]);
    }
    if (update_state) *reinterpret_cast<uint4*>(state + slot * channels + c0) = raw_cur;
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

/// Fast single-pass register-cached BF16 l2_clip for width == 128 or width == 16.
__global__ void l2_clip_bf16_fast(Launch plan) {
    const auto width = plan.inputs[0].shape[plan.inputs[0].rank - 1];
    const std::size_t row = (static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (row * width >= plan.elements) return;
    const bool has_scale = plan.count == 2;
    const auto scale_rows = has_scale ? plan.inputs[1].shape[0] : 0;
    const auto scale_row = has_scale ? static_cast<std::int64_t>(row % scale_rows) : 0;
    const auto* src = static_cast<const __nv_bfloat16*>(address(plan.inputs[0])) + row * width;
    const auto* scale_ptr = has_scale ? static_cast<const __nv_bfloat16*>(address(plan.inputs[1])) + scale_row * width : nullptr;
    auto* dst = static_cast<__nv_bfloat16*>(address(plan.output)) + row * width;
    if (width == 128) {
        float v[4];
        load_bf16x4(src + lane * 4, v);
        if (has_scale) {
            float s[4];
            load_bf16x4(scale_ptr + lane * 4, s);
#pragma unroll
            for (int i = 0; i < 4; ++i) v[i] *= s[i];
        }
        const float sq = v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3];
        const float factor = fminf(1.F, plan.epsilon / (sqrtf(warp_sum(sq)) + 1e-6F));
#pragma unroll
        for (int i = 0; i < 4; ++i) v[i] *= factor;
        store_bf16x4(dst + lane * 4, v);
        return;
    }
    // width == 16: lanes 0..15 hold 1 value each.
    float v = 0.F;
    if (lane < 16) {
        v = __bfloat162float(__ldg(src + lane));
        if (has_scale) v *= __bfloat162float(__ldg(scale_ptr + lane));
    }
    const float factor = fminf(1.F, plan.epsilon / (sqrtf(warp_sum(v * v)) + 1e-6F));
    if (lane < 16) dst[lane] = __float2bfloat16(v * factor);
}

/// Quantizes rows [tokens, width] to INT8 with one F32 scale per 32 columns (max |x| / 127, rounded to nearest even)
/// and writes them to cache row `positions[token]`. One warp per 32-column group.
__global__ void quantize_cache(Launch plan) {
    const auto& cache = plan.inputs[2];
    const auto width = cache.shape[cache.rank - 1];
    const auto group = static_cast<std::int64_t>(blockIdx.x) * (blockDim.x / 32) + threadIdx.x / 32;
    if (group * 32 >= width) return;
    const int lane = threadIdx.x & 31;
    const auto token = static_cast<std::int64_t>(blockIdx.y);
    const auto row = static_cast<std::int64_t>(static_cast<const int*>(address(plan.inputs[1]))[token]);
    const float value = read(plan.inputs[0], token * width + group * 32 + lane);
    float maximum = fabsf(value);
    for (int offset = 16; offset > 0; offset >>= 1)
        maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, offset));
    // IEEE division and rounding match the CPU cache quantizer.
    const float scale = maximum > 0 ? __fdiv_rn(maximum, 127.F) : 1.F;
    const float code = fminf(fmaxf(rintf(__fmul_rn(value, __fdiv_rn(1.F, scale))), -127.F), 127.F);
    static_cast<signed char*>(address(cache))[row * width + group * 32 + lane] = static_cast<signed char>(code);
    if (lane == 0) static_cast<float*>(address(plan.inputs[3]))[row * (width / 32) + group] = scale;
}

/// Position of query `token`: `start + token`, or the I32 positions operand. Linear caches clamp it to their last row;
/// ring caches (eighth attribute) keep absolute positions and store key `k` in row `k % rows`.
__device__ std::int64_t relative_query_position(const Launch& plan, std::int64_t token, bool ring) {
    if (plan.count < 7) return plan.attributes[2] + token;
    const auto position = static_cast<std::int64_t>(static_cast<const int*>(address(plan.inputs[6]))[token]);
    return ring ? position : min(position, plan.inputs[1].shape[0] - 1);
}

/// One block per (query, head), one warp per key stride. The sink is a zero-valued key with logit `sink`;
/// keys closer than `extent` add the per-query relative logit, and a sigmoid gate scales the output.
/// A positive log floor scales the query's key logits (not the sink) by
/// 1 + alpha * log(max(1, (position + 1) / floor)). INT8 caches (nine operands) carry F32 scales [rows, width / 32].
__global__ void relative_attention(Launch plan) {
    __shared__ float peaks[ATTENTION_WARPS], totals[ATTENTION_WARPS], partial[ATTENTION_WARPS][MAX_LANE_VALUES * 32];
    const auto heads = plan.attributes[0], key_heads = plan.attributes[1], window = plan.attributes[3],
               extent = plan.attributes[4];
    const float log_alpha = attribute_float(plan, 5), log_floor = attribute_float(plan, 6);
    const auto token = static_cast<std::int64_t>(blockIdx.x), head = static_cast<std::int64_t>(blockIdx.y);
    const auto width = plan.inputs[0].shape[1] / heads, key_head = head / (heads / key_heads);
    const auto key_stride = plan.inputs[1].shape[1], rows = plan.inputs[1].shape[0];
    const bool ring = plan.attribute_count > 7 && plan.attributes[7];
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32, warps = blockDim.x / 32;
    const int values = static_cast<int>(width / 32);
    const auto position = relative_query_position(plan, token, ring);
    const auto first = window ? max(static_cast<std::int64_t>(0), position - window + 1) : 0;
    const float tau =
        log_floor > 0 ? 1.F + log_alpha * logf(fmaxf(1.F, static_cast<float>(position + 1) / log_floor)) : 1.F;
    const float scale = rsqrtf(static_cast<float>(width));
    // Each 32-lane value chunk of a head is one INT8 scale group.
    const bool scaled = plan.count == 9;
    const auto groups = key_stride / 32, head_group = key_head * width / 32;
    const auto* key_scales = scaled ? static_cast<const float*>(address(plan.inputs[7])) : nullptr;
    const auto* value_scales = scaled ? static_cast<const float*>(address(plan.inputs[8])) : nullptr;
    float query[MAX_LANE_VALUES], accumulator[MAX_LANE_VALUES];
    for (int value = 0; value < values; ++value) {
        query[value] = read(plan.inputs[0], (token * heads + head) * width + value * 32 + lane) * scale;
        accumulator[value] = 0;
    }
    float peak = -1e30F, total = 0;
    for (auto key = first + warp; key <= position; key += warps) {
        const auto row = ring ? key % rows : key;
        const auto base = row * key_stride + key_head * width;
        float dot = 0;
        for (int value = 0; value < values; ++value)
            dot += query[value] * read(plan.inputs[1], base + value * 32 + lane) *
                   (scaled ? key_scales[row * groups + head_group + value] : 1.F);
        float logit = warp_sum(dot);
        const auto distance = position - key;
        if (distance < extent) logit += read(plan.inputs[3], (token * heads + head) * extent + distance);
        logit *= tau;
        const float next = fmaxf(peak, logit), decay = expf(peak - next), weight = expf(logit - next);
        total = total * decay + weight;
        for (int value = 0; value < values; ++value) {
            const float stored = read(plan.inputs[2], base + value * 32 + lane) *
                                 (scaled ? value_scales[row * groups + head_group + value] : 1.F);
            accumulator[value] = accumulator[value] * decay + weight * stored;
        }
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

/// Vectorized decode attention for width == 128 BF16 queries with configurable warps per head. Caches are BF16, or
/// INT8 with one F32 scale per 32 columns (nine operands).
template <int WARPS, bool INT8>
__global__ void __launch_bounds__(WARPS * 32) relative_attention_d128(Launch plan) {
    __shared__ float peaks[WARPS], totals[WARPS], warp_scales[WARPS], partial[WARPS][128];
    const auto heads = plan.attributes[0], key_heads = plan.attributes[1], window = plan.attributes[3],
               extent = plan.attributes[4];
    const float log_alpha = attribute_float(plan, 5), log_floor = attribute_float(plan, 6);
    const auto token = static_cast<std::int64_t>(blockIdx.x), head = static_cast<std::int64_t>(blockIdx.y);
    constexpr int WIDTH = 128;
    const auto key_head = head / (heads / key_heads);
    const auto key_stride = plan.inputs[1].shape[1], rows = plan.inputs[1].shape[0];
    const bool ring = plan.attribute_count > 7 && plan.attributes[7];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const auto position = relative_query_position(plan, token, ring);
    const auto first = window ? max(static_cast<std::int64_t>(0), position - window + 1) : 0;
    const float tau =
        log_floor > 0 ? 1.F + log_alpha * logf(fmaxf(1.F, static_cast<float>(position + 1) / log_floor)) : 1.F;
    const float scale = rsqrtf(static_cast<float>(WIDTH));
    const auto* q_ptr =
        static_cast<const __nv_bfloat16*>(address(plan.inputs[0])) + (token * heads + head) * WIDTH + lane * 4;
    const auto* k_cache = address(plan.inputs[1]);
    const auto* v_cache = address(plan.inputs[2]);
    const auto* key_scales = INT8 ? static_cast<const float*>(address(plan.inputs[7])) : nullptr;
    const auto* value_scales = INT8 ? static_cast<const float*>(address(plan.inputs[8])) : nullptr;
    // Each lane loads four columns of one 32-column scale group.
    const auto column = key_head * WIDTH + lane * 4, groups = key_stride / 32;
    const auto load = [&](const void* cache, const float* scales, std::int64_t row, float (&values)[4]) {
        if constexpr (INT8) {
            const auto codes = *reinterpret_cast<const char4*>(static_cast<const signed char*>(cache) +
                                                               row * key_stride + column);
            const float factor = scales[row * groups + column / 32];
            values[0] = codes.x * factor;
            values[1] = codes.y * factor;
            values[2] = codes.z * factor;
            values[3] = codes.w * factor;
        } else {
            load_bf16x4(static_cast<const __nv_bfloat16*>(cache) + row * key_stride + column, values);
        }
    };
    float q[4], acc[4]{};
    load_bf16x4(q_ptr, q);
#pragma unroll
    for (int i = 0; i < 4; ++i) q[i] *= scale;
    const auto* rel_ptr = static_cast<const __nv_bfloat16*>(address(plan.inputs[3])) + (token * heads + head) * extent;
    const bool rel_bf16 = plan.inputs[3].type == static_cast<int>(Type::BF16);
    float peak = -1e30F, total = 0;
    for (auto key = first + warp; key <= position; key += WARPS) {
        const auto row = ring ? key % rows : key;
        float kv[4];
        load(k_cache, key_scales, row, kv);
        const float dot = q[0] * kv[0] + q[1] * kv[1] + q[2] * kv[2] + q[3] * kv[3];
        float logit = warp_sum(dot);
        const auto distance = position - key;
        if (distance < extent)
            logit += rel_bf16 ? __bfloat162float(__ldg(rel_ptr + distance)) : read(plan.inputs[3], (token * heads + head) * extent + distance);
        logit *= tau;
        const float next = fmaxf(peak, logit), decay = expf(peak - next), weight = expf(logit - next);
        total = total * decay + weight;
        load(v_cache, value_scales, row, kv);
#pragma unroll
        for (int i = 0; i < 4; ++i) acc[i] = acc[i] * decay + weight * kv[i];
        peak = next;
    }
    if (lane == 0) {
        peaks[warp] = peak;
        totals[warp] = total;
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) partial[warp][lane * 4 + i] = acc[i];
    __syncthreads();
    if (warp == 0 && lane == 0) {
        const float sink = read(plan.inputs[4], token * heads + head);
        float combined = sink;
#pragma unroll
        for (int w = 0; w < WARPS; ++w) combined = fmaxf(combined, peaks[w]);
        float denominator = expf(sink - combined);
#pragma unroll
        for (int w = 0; w < WARPS; ++w) {
            const float s = expf(peaks[w] - combined);
            warp_scales[w] = s;
            denominator += totals[w] * s;
        }
        const float inv_denom = 1.F / denominator;
#pragma unroll
        for (int w = 0; w < WARPS; ++w) warp_scales[w] *= inv_denom;
    }
    __syncthreads();
    const auto* gate_ptr = static_cast<const __nv_bfloat16*>(address(plan.inputs[5])) + (token * heads + head) * WIDTH;
    auto* out_ptr = static_cast<__nv_bfloat16*>(address(plan.output)) + (token * heads + head) * WIDTH;
    if (threadIdx.x < WIDTH) {
        const int col = threadIdx.x;
        float sum = 0;
#pragma unroll
        for (int w = 0; w < WARPS; ++w) sum = fmaf(partial[w][col], warp_scales[w], sum);
        const float gate = __bfloat162float(__ldg(gate_ptr + col));
        out_ptr[col] = __float2bfloat16(sum / (1.F + expf(-gate)));
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

constexpr int BLEND_THREADS = 576;

/// Unit-RMS-normalizes `destination` and `message` [tokens, width] in one pass and writes concatenated BF16 [tokens, 2 * width].
__global__ void __launch_bounds__(BLEND_THREADS) blend_prologue(const __grid_constant__ Launch plan) {
    __shared__ float shared[2][BLEND_THREADS / 32];
    const auto token = static_cast<std::int64_t>(blockIdx.x);
    const auto width = plan.inputs[0].shape[1];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const auto* d_base = static_cast<const float*>(address(plan.inputs[0])) + token * width;
    const auto* m_base = static_cast<const float*>(address(plan.inputs[1])) + token * width;
    auto* out = static_cast<__nv_bfloat16*>(address(plan.output)) + token * 2 * width;
    const bool vec4 = plan.inputs[0].type == static_cast<int>(Type::F32) &&
                      plan.inputs[1].type == static_cast<int>(Type::F32) && (width & 3) == 0 &&
                      width <= BLEND_THREADS * 4 && (reinterpret_cast<std::uintptr_t>(d_base) & 15) == 0 &&
                      (reinterpret_cast<std::uintptr_t>(m_base) & 15) == 0 &&
                      (reinterpret_cast<std::uintptr_t>(out) & 7) == 0;
    float4 dv4{}, mv4{};
    float sum_d = 0, sum_m = 0;
    if (vec4) {
        const auto c4 = static_cast<std::int64_t>(threadIdx.x) * 4;
        if (c4 < width) {
            dv4 = __ldg(reinterpret_cast<const float4*>(d_base + c4));
            mv4 = __ldg(reinterpret_cast<const float4*>(m_base + c4));
            sum_d = fmaf(dv4.x, dv4.x, fmaf(dv4.y, dv4.y, fmaf(dv4.z, dv4.z, dv4.w * dv4.w)));
            sum_m = fmaf(mv4.x, mv4.x, fmaf(mv4.y, mv4.y, fmaf(mv4.z, mv4.z, mv4.w * mv4.w)));
        }
    } else {
        for (auto c = static_cast<std::int64_t>(threadIdx.x); c < width; c += BLEND_THREADS) {
            const float dv = read(plan.inputs[0], token * width + c);
            const float mv = read(plan.inputs[1], token * width + c);
            sum_d = fmaf(dv, dv, sum_d);
            sum_m = fmaf(mv, mv, sum_m);
        }
    }
    sum_d = warp_sum(sum_d);
    sum_m = warp_sum(sum_m);
    if (lane == 0) {
        shared[0][warp] = sum_d;
        shared[1][warp] = sum_m;
    }
    __syncthreads();
    if (warp == 0) {
        float sd = lane < BLEND_THREADS / 32 ? shared[0][lane] : 0.F;
        float sm = lane < BLEND_THREADS / 32 ? shared[1][lane] : 0.F;
        sd = warp_sum(sd);
        sm = warp_sum(sm);
        if (lane == 0) {
            shared[0][0] = rsqrtf(sd / static_cast<float>(width) + plan.epsilon);
            shared[1][0] = rsqrtf(sm / static_cast<float>(width) + plan.epsilon);
        }
    }
    __syncthreads();
    const float inv_d = shared[0][0], inv_m = shared[1][0];
    if (vec4) {
        const auto c4 = static_cast<std::int64_t>(threadIdx.x) * 4;
        if (c4 < width) {
            const __nv_bfloat162 d_out[2] = {__floats2bfloat162_rn(dv4.x * inv_d, dv4.y * inv_d),
                                             __floats2bfloat162_rn(dv4.z * inv_d, dv4.w * inv_d)};
            const __nv_bfloat162 m_out[2] = {__floats2bfloat162_rn(mv4.x * inv_m, mv4.y * inv_m),
                                             __floats2bfloat162_rn(mv4.z * inv_m, mv4.w * inv_m)};
            *reinterpret_cast<uint2*>(out + c4) = *reinterpret_cast<const uint2*>(d_out);
            *reinterpret_cast<uint2*>(out + width + c4) = *reinterpret_cast<const uint2*>(m_out);
        }
    } else {
        for (auto c = static_cast<std::int64_t>(threadIdx.x); c < width; c += BLEND_THREADS) {
            out[c] = __float2bfloat16(read(plan.inputs[0], token * width + c) * inv_d);
            out[width + c] = __float2bfloat16(read(plan.inputs[1], token * width + c) * inv_m);
        }
    }
}

/// mixed = dest * (1 + tanh(forget)) + msg * tanh(gain); output BF16(rms_norm(mixed, norm)).
__global__ void __launch_bounds__(BLEND_THREADS) blend_epilogue(const __grid_constant__ Launch plan) {
    __shared__ float shared[BLEND_THREADS / 32];
    const auto token = static_cast<std::int64_t>(blockIdx.x);
    const auto width = plan.inputs[0].shape[1];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const auto* d_base = static_cast<const float*>(address(plan.inputs[0])) + token * width;
    const auto* m_base = static_cast<const float*>(address(plan.inputs[1])) + token * width;
    const auto* gf_base = static_cast<const __nv_bfloat16*>(address(plan.inputs[2])) + token * 2 * width;
    const auto* norm_ptr = static_cast<const float*>(address(plan.inputs[3]));
    auto* out = static_cast<__nv_bfloat16*>(address(plan.output)) + token * width;
    const bool vec4 = plan.inputs[0].type == static_cast<int>(Type::F32) &&
                      plan.inputs[1].type == static_cast<int>(Type::F32) &&
                      plan.inputs[2].type == static_cast<int>(Type::BF16) && (width & 3) == 0 &&
                      width <= BLEND_THREADS * 4 && (reinterpret_cast<std::uintptr_t>(d_base) & 15) == 0 &&
                      (reinterpret_cast<std::uintptr_t>(m_base) & 15) == 0 &&
                      (reinterpret_cast<std::uintptr_t>(gf_base) & 7) == 0 &&
                      (reinterpret_cast<std::uintptr_t>(norm_ptr) & 15) == 0 &&
                      (reinterpret_cast<std::uintptr_t>(out) & 7) == 0;
    float4 mixed4{};
    float sum = 0;
    if (vec4) {
        const auto c4 = static_cast<std::int64_t>(threadIdx.x) * 4;
        if (c4 < width) {
            const float4 dv4 = __ldg(reinterpret_cast<const float4*>(d_base + c4));
            const float4 mv4 = __ldg(reinterpret_cast<const float4*>(m_base + c4));
            const uint2 g_raw = __ldg(reinterpret_cast<const uint2*>(gf_base + c4));
            const uint2 f_raw = __ldg(reinterpret_cast<const uint2*>(gf_base + width + c4));
            const auto* gh = reinterpret_cast<const __nv_bfloat162*>(&g_raw);
            const auto* fh = reinterpret_cast<const __nv_bfloat162*>(&f_raw);
            const float2 g0 = __bfloat1622float2(gh[0]), g1 = __bfloat1622float2(gh[1]);
            const float2 f0 = __bfloat1622float2(fh[0]), f1 = __bfloat1622float2(fh[1]);
            mixed4.x = fmaf(dv4.x, 1.F + tanhf(f0.x), mv4.x * tanhf(g0.x));
            mixed4.y = fmaf(dv4.y, 1.F + tanhf(f0.y), mv4.y * tanhf(g0.y));
            mixed4.z = fmaf(dv4.z, 1.F + tanhf(f1.x), mv4.z * tanhf(g1.x));
            mixed4.w = fmaf(dv4.w, 1.F + tanhf(f1.y), mv4.w * tanhf(g1.y));
            sum = fmaf(mixed4.x, mixed4.x,
                       fmaf(mixed4.y, mixed4.y, fmaf(mixed4.z, mixed4.z, mixed4.w * mixed4.w)));
        }
    } else {
        for (auto c = static_cast<std::int64_t>(threadIdx.x); c < width; c += BLEND_THREADS) {
            const float g = tanhf(read(plan.inputs[2], token * 2 * width + c));
            const float f = tanhf(read(plan.inputs[2], token * 2 * width + width + c));
            const float v = fmaf(read(plan.inputs[0], token * width + c), 1.F + f,
                                 read(plan.inputs[1], token * width + c) * g);
            sum = fmaf(v, v, sum);
        }
    }
    sum = warp_sum(sum);
    if (lane == 0) shared[warp] = sum;
    __syncthreads();
    if (warp == 0) {
        float total = lane < BLEND_THREADS / 32 ? shared[lane] : 0.F;
        total = warp_sum(total);
        if (lane == 0) shared[0] = rsqrtf(total / static_cast<float>(width) + plan.epsilon);
    }
    __syncthreads();
    const float inv = shared[0];
    if (vec4) {
        const auto c4 = static_cast<std::int64_t>(threadIdx.x) * 4;
        if (c4 < width) {
            const float4 n4 = __ldg(reinterpret_cast<const float4*>(norm_ptr + c4));
            const __nv_bfloat162 out2[2] = {
                __floats2bfloat162_rn(mixed4.x * inv * n4.x, mixed4.y * inv * n4.y),
                __floats2bfloat162_rn(mixed4.z * inv * n4.z, mixed4.w * inv * n4.w)};
            *reinterpret_cast<uint2*>(out + c4) = *reinterpret_cast<const uint2*>(out2);
        }
    } else {
        for (auto c = static_cast<std::int64_t>(threadIdx.x); c < width; c += BLEND_THREADS) {
            const float g = tanhf(read(plan.inputs[2], token * 2 * width + c));
            const float f = tanhf(read(plan.inputs[2], token * 2 * width + width + c));
            const float v = fmaf(read(plan.inputs[0], token * width + c), 1.F + f,
                                 read(plan.inputs[1], token * width + c) * g);
            out[c] = __float2bfloat16(v * inv * norm_ptr[c]);
        }
    }
}

} // namespace kernel

auto blocks(std::size_t elements, int threads) -> unsigned { return static_cast<unsigned>((elements + threads - 1) / threads); }

auto aligned(const View& view) -> bool {
    return !((view.binding ? view.binding_offset : reinterpret_cast<std::uintptr_t>(view.data)) & 15);
}

// Outputs per warp: enough to share activation chunks, few enough to keep every SM busy for small decode grids.
constexpr int GEMV_ROWS = 2;

// The chunked GEMV reads 16-byte weight and activation chunks, so rows must hold whole chunks, scale groups must
// span them, both tensors must start on a 16-byte boundary, and each warp's rows must belong to one token.
auto chunked_gemv(const Launch& plan) -> bool {
    const auto width = plan.inputs[0].shape[plan.inputs[0].rank - 1];
    const auto columns = plan.output.shape[plan.output.rank - 1];
    return width % (plan.attributes[3] == 1 ? 16 : 32) == 0 && plan.attributes[4] % 16 == 0 &&
           columns % GEMV_ROWS == 0 && width <= INT32_MAX / plan.inputs[1].shape[plan.inputs[1].rank - 2] &&
           aligned(plan.inputs[0]) && aligned(plan.inputs[1]);
}

template <bool BF16, bool ROUTED, bool GATED, int FORMAT, int SCALE>
auto dispatch_block_gemv(const Launch& plan, cudaStream_t stream) -> void {
    constexpr int ROWS = (ROUTED && GATED) ? 1 : GEMV_ROWS;
    kernel::block_scaled_block_gemv<BF16, ROUTED, GATED, FORMAT, SCALE, ROWS>
        <<<blocks(plan.elements / ROWS, kernel::GEMV_WARPS), kernel::GEMV_WARPS * 32, 0, stream>>>(plan);
}

template <bool ROUTED, bool BF16, bool GATED>
auto dispatch_block_gemv_format(const Launch& plan, cudaStream_t stream) -> void {
    switch (plan.attributes[3] * 3 + plan.attributes[7]) {
        case 0: dispatch_block_gemv<BF16, ROUTED, GATED, 0, 0>(plan, stream); break;
        case 2: dispatch_block_gemv<BF16, ROUTED, GATED, 0, 2>(plan, stream); break;
        case 4: dispatch_block_gemv<BF16, ROUTED, GATED, 1, 1>(plan, stream); break;
        case 5: dispatch_block_gemv<BF16, ROUTED, GATED, 1, 2>(plan, stream); break;
        default: dispatch_block_gemv<BF16, ROUTED, GATED, 2, 2>(plan, stream); break;
    }
}

template <bool ROUTED>
auto launch_block_gemv(const Launch& source, cudaStream_t stream) -> void {
    auto plan = source;
    const auto format = plan.attributes[3], shift = plan.attributes[6];
    const auto scale_type = static_cast<Type>(plan.inputs[2].type);
    if (format == 0 && (scale_type == Type::E4M3 || scale_type == Type::U8) && shift == 4)
        plan.attributes[7] = 0;
    else if (format == 1 && scale_type == Type::U8 && shift >= 0)
        plan.attributes[7] = 1;
    else
        plan.attributes[7] = 2;
    const bool bfloat = plan.inputs[0].type == static_cast<int>(Type::BF16), gated = plan.attributes[0];
    if (bfloat && gated)
        dispatch_block_gemv_format<ROUTED, true, true>(plan, stream);
    else if (bfloat)
        dispatch_block_gemv_format<ROUTED, true, false>(plan, stream);
    else if (gated)
        dispatch_block_gemv_format<ROUTED, false, true>(plan, stream);
    else
        dispatch_block_gemv_format<ROUTED, false, false>(plan, stream);
}

// The tiled GEMM multiplies BF16 activations by NVFP4 with E4M3 scales per 16 values or MXFP8 with UE8M0 scales,
// from 16-byte-aligned rows that hold whole 64-column tiles. Returns the tile format, or -1 when another kernel must run.
auto tile_format(const Launch& plan) -> int {
    const auto& values = plan.inputs[1];
    const auto width = plan.inputs[0].shape[plan.inputs[0].rank - 1], rows = values.shape[values.rank - 2];
    const auto scale_type = static_cast<Type>(plan.inputs[2].type);
    const auto format = plan.attributes[3], shift = plan.attributes[6];
    if (width % kernel::TILE || rows % kernel::TILE || !aligned(plan.inputs[0]) || !aligned(values) ||
        plan.inputs[0].type != static_cast<int>(Type::BF16))
        return -1;
    if (format == 0 && (scale_type == Type::E4M3 || scale_type == Type::U8) && shift == 4) return 0;
    if (format == 1 && scale_type == Type::U8 && shift >= 4) return 1;
    return -1;
}

// Instantiates a tiled kernel for the runtime gated/format combination.
template <template <bool, int> class Kernel, typename... Arguments>
auto launch_tiled(const Launch& plan, int format, dim3 grid, cudaStream_t stream, Arguments... arguments) -> void {
    const auto launch = [&]<bool GATED>() {
        if (format == 0)
            Kernel<GATED, 0>::launch(grid, stream, plan, arguments...);
        else
            Kernel<GATED, 1>::launch(grid, stream, plan, arguments...);
    };
    if (plan.attributes[0])
        launch.template operator()<true>();
    else
        launch.template operator()<false>();
}

template <bool GATED, int FORMAT>
struct DenseTiles {
    static auto launch(dim3 grid, cudaStream_t stream, const Launch& plan) -> void {
        kernel::block_scaled_tile_gemm<GATED, FORMAT><<<grid, kernel::TILE_THREADS, 0, stream>>>(plan);
    }
};

template <bool GATED, int FORMAT>
struct MarlinExpertTiles {
    static auto launch(dim3 grid, cudaStream_t stream, const Launch& plan, int m_tile, int n_tile,
                       const int* active_blocks, const kernel::MoeBlock* blocks_in, const int* routes,
                       float* partial) -> void {
        if (n_tile == kernel::MARLIN_N_LARGE) {
            if (m_tile == kernel::MARLIN_M_LARGE)
                kernel::marlin_expert_tile_gemm<GATED, FORMAT, kernel::MARLIN_M_LARGE, kernel::MARLIN_N_LARGE>
                    <<<grid, kernel::MARLIN_THREADS, 0, stream>>>(plan, active_blocks, blocks_in, routes, partial);
            else
                kernel::marlin_expert_tile_gemm<GATED, FORMAT, kernel::MARLIN_M_SMALL, kernel::MARLIN_N_LARGE>
                    <<<grid, kernel::MARLIN_THREADS, 0, stream>>>(plan, active_blocks, blocks_in, routes, partial);
        } else {
            if (m_tile == kernel::MARLIN_M_LARGE)
                kernel::marlin_expert_tile_gemm<GATED, FORMAT, kernel::MARLIN_M_LARGE, kernel::MARLIN_N_SMALL>
                    <<<grid, kernel::MARLIN_THREADS, 0, stream>>>(plan, active_blocks, blocks_in, routes, partial);
            else
                kernel::marlin_expert_tile_gemm<GATED, FORMAT, kernel::MARLIN_M_SMALL, kernel::MARLIN_N_SMALL>
                    <<<grid, kernel::MARLIN_THREADS, 0, stream>>>(plan, active_blocks, blocks_in, routes, partial);
        }
    }
};

// Batches from this many tokens group routed rows by expert, so each expert's weights are decoded once per tile of
// rows instead of once per (token, slot) pair.
constexpr std::int64_t GROUPED_EXPERT_TOKENS = 8;
constexpr int MAX_GROUPED_EXPERTS = 1024;

auto round_up(std::size_t bytes) -> std::size_t { return (bytes + 255) / 256 * 256; }

auto marlin_m_tile(std::int64_t tokens) -> int {
    return tokens >= 256 ? kernel::MARLIN_M_LARGE : kernel::MARLIN_M_SMALL;
}

auto marlin_max_blocks(std::int64_t tokens, std::int64_t slots, std::int64_t experts) -> std::size_t {
    const auto routes = static_cast<std::size_t>(tokens * slots);
    const auto m_tile = static_cast<std::size_t>(marlin_m_tile(tokens));
    return std::min<std::size_t>(static_cast<std::size_t>(experts) + routes / m_tile, routes);
}

template <int MAX_S>
auto launch_residual_mix_fast(const Launch& plan, unsigned grid, std::int64_t queries_count, cudaStream_t stream)
    -> void {
    if (queries_count <= 1)
        kernel::residual_mix_fast<MAX_S, 1><<<grid, kernel::MIX_FAST_THREADS, 0, stream>>>(plan);
    else if (queries_count == 2)
        kernel::residual_mix_fast<MAX_S, 2><<<grid, kernel::MIX_FAST_THREADS, 0, stream>>>(plan);
    else
        kernel::residual_mix_fast<MAX_S, 4><<<grid, kernel::MIX_FAST_THREADS, 0, stream>>>(plan);
}
} // namespace

auto expert_workspace_bytes(std::int64_t tokens, std::int64_t slots, std::int64_t experts, std::int64_t rows,
                            bool gated) -> std::size_t {
    if (tokens < GROUPED_EXPERT_TOKENS || experts > MAX_GROUPED_EXPERTS) return 0;
    const auto routes = static_cast<std::size_t>(tokens * slots);
    const auto max_blocks = marlin_max_blocks(tokens, slots, experts);
    return round_up(16 + max_blocks * sizeof(kernel::MoeBlock) + routes * sizeof(int)) +
           (gated ? 0 : routes * static_cast<std::size_t>(rows) * sizeof(float));
}

auto expert_linear(const Launch& plan, void* workspace, cudaStream_t stream) -> void {
    const auto format = workspace ? tile_format(plan) : -1;
    if (format < 0) {
        if (!chunked_gemv(plan))
            throw std::runtime_error("CUDA: routed expert projection needs aligned, chunk-sized rows");
        launch_block_gemv<true>(plan, stream);
        return;
    }
    const auto& routes = plan.inputs[plan.count - 1];
    const auto tokens = routes.shape[0], slots = routes.shape[1];
    const auto experts = plan.inputs[1].shape[0], rows = plan.inputs[1].shape[1];
    const auto route_count = static_cast<std::size_t>(tokens * slots);
    const auto max_blocks = marlin_max_blocks(tokens, slots, experts);
    const int m_tile = marlin_m_tile(tokens);
    const int n_tile = rows % kernel::MARLIN_N_LARGE == 0 ? kernel::MARLIN_N_LARGE : kernel::MARLIN_N_SMALL;
    auto* active_blocks = static_cast<int*>(workspace);
    auto* moe_blocks = reinterpret_cast<kernel::MoeBlock*>(active_blocks + 4);
    auto* routes_by_expert = reinterpret_cast<int*>(moe_blocks + max_blocks);
    auto* partial = reinterpret_cast<float*>(
        static_cast<unsigned char*>(workspace) +
        round_up(16 + max_blocks * sizeof(kernel::MoeBlock) + route_count * sizeof(int)));
    if (m_tile == kernel::MARLIN_M_LARGE)
        kernel::sort_routes_blocks<kernel::MARLIN_M_LARGE>
            <<<1, 1024, 0, stream>>>(plan, active_blocks, moe_blocks, routes_by_expert);
    else
        kernel::sort_routes_blocks<kernel::MARLIN_M_SMALL>
            <<<1, 1024, 0, stream>>>(plan, active_blocks, moe_blocks, routes_by_expert);
    launch_tiled<MarlinExpertTiles>(plan, format,
                                    dim3(static_cast<unsigned>(rows / n_tile), static_cast<unsigned>(max_blocks)),
                                    stream, m_tile, n_tile, static_cast<const int*>(active_blocks),
                                    static_cast<const kernel::MoeBlock*>(moe_blocks),
                                    static_cast<const int*>(routes_by_expert), partial);
    if (!plan.attributes[0]) kernel::sum_slots<<<blocks(plan.elements, 256), 256, 0, stream>>>(plan, partial);
}

auto fused_operation(const Launch& plan, Op operation, cudaStream_t stream) -> void {
    switch (operation) {
        case Op::BLOCK_SCALED_LINEAR: {
            const auto tokens = plan.elements / plan.output.shape[plan.output.rank - 1];
            if (const auto format = tokens >= GROUPED_EXPERT_TOKENS ? tile_format(plan) : -1; format >= 0) {
                launch_tiled<DenseTiles>(plan, format,
                                         dim3(static_cast<unsigned>(plan.inputs[1].shape[0] / kernel::TILE),
                                              blocks(tokens, kernel::TILE)),
                                         stream);
            } else if (plan.inputs[0].type == static_cast<int>(Type::BF16) && tokens >= 32) {
                const dim3 grid(blocks(plan.inputs[1].shape[0], 16), blocks(tokens, 16));
                kernel::block_scaled_gemm<<<grid, 32, 0, stream>>>(plan);
            } else if (chunked_gemv(plan))
                launch_block_gemv<false>(plan, stream);
            else
                kernel::block_scaled_gemv<<<blocks(plan.elements, 1), 256, 0, stream>>>(plan);
            break;
        }
        case Op::EXPERT_LINEAR:
            expert_linear(plan, nullptr, stream);
            break;
        case Op::ROUTE_EXPERTS:
            kernel::route_experts<<<static_cast<unsigned>(plan.inputs[0].shape[0]), 32, 0, stream>>>(plan);
            break;
        case Op::DEQUANTIZE_EXPERTS:
            if (plan.output.type == static_cast<int>(Type::F32))
                kernel::expert_row_scales<<<static_cast<unsigned>(plan.elements), 256, 0, stream>>>(plan);
            else
                kernel::dequantize_experts<<<blocks(plan.elements, 256), 256, 0, stream>>>(plan);
            break;
        case Op::RESIDUAL_MIX: {
            const int sources = plan.count - 1;
            const auto& queries = plan.inputs[sources];
            const auto width = queries.shape[1], queries_count = queries.shape[0];
            bool all_ok = width <= kernel::MIX_FAST_WIDTH && queries.type == static_cast<int>(Type::F32) &&
                          plan.output.type == static_cast<int>(Type::F32) && sources >= 1 &&
                          sources <= kernel::MAX_SOURCES && queries_count <= kernel::MAX_QUERIES;
            for (int s = 0; s < sources; ++s) {
                const auto t = plan.inputs[s].type;
                if (t != static_cast<int>(Type::BF16) && t != static_cast<int>(Type::F32)) all_ok = false;
                if (s + 1 < sources && t == static_cast<int>(Type::F32)) all_ok = false;
            }
            if (all_ok) {
                const auto grid = static_cast<unsigned>(plan.inputs[0].shape[0]);
                if (sources <= 2)
                    launch_residual_mix_fast<2>(plan, grid, queries_count, stream);
                else if (sources <= 4)
                    launch_residual_mix_fast<4>(plan, grid, queries_count, stream);
                else
                    launch_residual_mix_fast<7>(plan, grid, queries_count, stream);
            } else
                kernel::residual_mix<<<static_cast<unsigned>(plan.inputs[0].shape[0]), kernel::MIX_THREADS, 0,
                                       stream>>>(plan);
            break;
        }
        case Op::SHORT_CONVOLUTION: {
            const auto channels = plan.output.shape[1];
            const auto flags = plan.attribute_count == 3 ? plan.attributes[2] : 0;
            const bool dynamic = plan.attribute_count == 3 ? ((flags & 1) != 0) : (plan.count == 5);
            const bool has_residual = plan.attribute_count == 3 && ((flags & 2) != 0);
            const bool out_ok =
                plan.output.type == static_cast<int>(Type::BF16) || plan.output.type == static_cast<int>(Type::F32);
            const bool res_ok = !has_residual || plan.inputs[plan.count - 1].type == static_cast<int>(Type::BF16) ||
                                plan.inputs[plan.count - 1].type == static_cast<int>(Type::F32);
            const bool bf16_w4 = plan.attribute_count >= 2 && plan.attributes[0] == 4 && plan.attributes[1] == 4 &&
                                 plan.inputs[0].type == static_cast<int>(Type::BF16) &&
                                 plan.inputs[1].type == static_cast<int>(Type::BF16) &&
                                 plan.inputs[2].type == static_cast<int>(Type::BF16) && out_ok && res_ok &&
                                 channels % 8 == 0 && (!dynamic || channels % ((plan.inputs[4].shape[1] / 4) * 8) == 0);
            if (bf16_w4)
                kernel::short_convolution_bf16_w4<<<blocks(plan.elements / 8, 256), 256, 0, stream>>>(plan);
            else
                kernel::short_convolution<<<blocks(plan.elements, 256), 256, 0, stream>>>(plan);
            break;
        }
        case Op::L2_CLIP: {
            const auto width = plan.inputs[0].shape[plan.inputs[0].rank - 1];
            const bool fast = (width == 128 || width == 16) && plan.inputs[0].type == static_cast<int>(Type::BF16) &&
                              plan.output.type == static_cast<int>(Type::BF16) &&
                              (plan.count == 1 || plan.inputs[1].type == static_cast<int>(Type::BF16));
            if (fast)
                kernel::l2_clip_bf16_fast<<<blocks(plan.elements / width * 32, 128), 128, 0, stream>>>(plan);
            else
                kernel::l2_clip<<<blocks(plan.elements / width * 32, 256), 256, 0, stream>>>(plan);
            break;
        }
        case Op::RELATIVE_ATTENTION: {
            const auto tokens = plan.inputs[0].shape[0], heads = plan.attributes[0], key_heads = plan.attributes[1],
                       start = plan.attributes[2], window = plan.attributes[3], extent = plan.attributes[4];
            const auto width = plan.inputs[0].shape[1] / heads;
            const bool bf16_kv = plan.inputs[1].type == static_cast<int>(Type::BF16) &&
                                 plan.inputs[2].type == static_cast<int>(Type::BF16);
            const bool int8_kv = plan.count == 9;
            const bool q_bf16_or_f32 =
                plan.inputs[0].type == static_cast<int>(Type::BF16) || plan.inputs[0].type == static_cast<int>(Type::F32);
            const bool flash_aligned = aligned(plan.inputs[0]) && aligned(plan.inputs[1]) && aligned(plan.inputs[2]) &&
                                       aligned(plan.inputs[5]) && aligned(plan.output);
            const bool flash_ok = tokens > 1 && (bf16_kv || int8_kv) && q_bf16_or_f32 &&
                                  plan.inputs[0].type == plan.inputs[5].type &&
                                  plan.inputs[0].type == plan.output.type && flash_aligned &&
                                  (width == 64 || width == 128 || width == 256 || width == 512);
            if (flash_ok) {
                const auto to_flash = [](const View& view) {
                    FlashView fv{};
                    fv.data = view.data;
                    fv.binding = view.binding;
                    fv.binding_offset = view.binding_offset;
                    fv.type = view.type;
                    fv.rank = view.rank;
                    for (int i = 0; i < std::min(view.rank, 4); ++i) fv.shape[i] = view.shape[i];
                    return fv;
                };
                FlashAttentionArgs args{};
                args.query = to_flash(plan.inputs[0]);
                args.key = to_flash(plan.inputs[1]);
                args.value = to_flash(plan.inputs[2]);
                args.output = to_flash(plan.output);
                args.relative = to_flash(plan.inputs[3]);
                args.sink = to_flash(plan.inputs[4]);
                args.gate = to_flash(plan.inputs[5]);
                args.has_relative = true;
                args.batches = 1;
                args.queries = tokens;
                args.heads = heads;
                args.key_heads = key_heads;
                args.key_offset = start;
                args.mask_window = window;
                // Ring caches address keys by absolute position, so only the positions bound them.
                const bool ring = plan.attribute_count > 7 && plan.attributes[7];
                args.keys = ring ? std::numeric_limits<int>::max() : plan.inputs[1].shape[0];
                args.ring_rows = ring ? plan.inputs[1].shape[0] : 0;
                args.extent = extent;
                args.head_dim = static_cast<int>(width);
                args.scale = 1.F / std::sqrt(static_cast<float>(width));
                int raw_alpha = static_cast<int>(plan.attributes[5]), raw_floor = static_cast<int>(plan.attributes[6]);
                std::memcpy(&args.log_alpha, &raw_alpha, sizeof(float));
                std::memcpy(&args.log_floor, &raw_floor, sizeof(float));
                if (int8_kv) {
                    args.key_scales = to_flash(plan.inputs[7]);
                    args.value_scales = to_flash(plan.inputs[8]);
                }
                if (plan.count >= 7) {
                    args.has_mask = true;
                    args.mask = to_flash(plan.inputs[6]);
                    args.mask.type = 3;
                    args.mask.rank = 4;
                    args.mask.shape[0] = 1;
                    args.mask.shape[1] = 1;
                    args.mask.shape[2] = tokens;
                    args.mask.shape[3] = 1;
                }
                if (flash_attention_bf16(args, stream)) break;
            }
            const bool d128 = width == 128 && plan.inputs[0].type == static_cast<int>(Type::BF16) &&
                              (bf16_kv || int8_kv) && plan.inputs[5].type == static_cast<int>(Type::BF16) &&
                              plan.output.type == static_cast<int>(Type::BF16);
            if (d128) {
                const dim3 grid(static_cast<unsigned>(tokens), static_cast<unsigned>(heads));
                const bool few_keys = (window > 0 ? window : plan.inputs[1].shape[0]) <= 256;
                if (int8_kv && few_keys)
                    kernel::relative_attention_d128<8, true><<<grid, 8 * 32, 0, stream>>>(plan);
                else if (int8_kv)
                    kernel::relative_attention_d128<32, true><<<grid, 32 * 32, 0, stream>>>(plan);
                else if (few_keys)
                    kernel::relative_attention_d128<8, false><<<grid, 8 * 32, 0, stream>>>(plan);
                else
                    kernel::relative_attention_d128<32, false><<<grid, 32 * 32, 0, stream>>>(plan);
            } else {
                const dim3 grid(static_cast<unsigned>(tokens), static_cast<unsigned>(heads));
                kernel::relative_attention<<<grid, kernel::ATTENTION_WARPS * 32, 0, stream>>>(plan);
            }
            break;
        }
        case Op::QUANTIZE_CACHE: {
            const auto& cache = plan.inputs[2];
            const auto groups = cache.shape[cache.rank - 1] / 32;
            const dim3 grid(static_cast<unsigned>((groups + 7) / 8), static_cast<unsigned>(plan.inputs[0].shape[0]));
            kernel::quantize_cache<<<grid, 256, 0, stream>>>(plan);
            break;
        }
        case Op::SILU_GATE:
            kernel::softcapped_silu_gate<<<blocks(plan.elements, 256), 256, 0, stream>>>(plan);
            break;
        case Op::BLEND_PROLOGUE:
            kernel::blend_prologue<<<static_cast<unsigned>(plan.inputs[0].shape[0]), kernel::BLEND_THREADS, 0,
                                     stream>>>(plan);
            break;
        case Op::BLEND_EPILOGUE:
            kernel::blend_epilogue<<<static_cast<unsigned>(plan.inputs[0].shape[0]), kernel::BLEND_THREADS, 0,
                                     stream>>>(plan);
            break;
        default:
            throw std::runtime_error("CUDA: unsupported fused operation");
    }
    check(cudaGetLastError());
}

} // namespace kidi::cuda
