#include "cuda/projection.h"
#include "cuda/elementwise.h"
#if defined(KIDI_CUDA_USE_CUTLASS_INT8)
#include "cuda/dense_gemm.h"
#include "cuda/int8_gemm.h"
#endif

#include <climits>
#include <cstring>

namespace kidi::cuda {
namespace {
namespace kernel {

__global__ void quantize_rows(Launch plan, signed char* quantized, float* scales, int* zeros) {
    __shared__ float s_min[8], s_max[8];
    const auto width = plan.inputs[0].shape[plan.inputs[0].rank - 1];
    const auto row = blockIdx.x;
    const bool calibrated = plan.operation == static_cast<int>(Op::PACKED_LINEAR) && plan.epsilon > 0;
    const auto base = static_cast<std::size_t>(row) * width;
    const bool aligned_f32 = plan.inputs[0].type == static_cast<int>(Type::F32) && width % 4 == 0 &&
                             reinterpret_cast<std::uintptr_t>(address(plan.inputs[0])) % 16 == 0;
    if (calibrated) {
        const float scale = plan.epsilon;
        if (threadIdx.x == 0) {
            scales[row] = scale;
            zeros[row] = 0;
        }
        if (aligned_f32) {
            const auto* src = static_cast<const float4*>(address(plan.inputs[0])) + base / 4;
            auto* dst = reinterpret_cast<char4*>(quantized + base);
            for (std::int64_t vec = threadIdx.x; vec < width / 4; vec += blockDim.x) {
                const float4 v = src[vec];
                char4 q;
                q.x = static_cast<signed char>(fminf(127.F, fmaxf(-128.F, nearbyintf(v.x / scale))));
                q.y = static_cast<signed char>(fminf(127.F, fmaxf(-128.F, nearbyintf(v.y / scale))));
                q.z = static_cast<signed char>(fminf(127.F, fmaxf(-128.F, nearbyintf(v.z / scale))));
                q.w = static_cast<signed char>(fminf(127.F, fmaxf(-128.F, nearbyintf(v.w / scale))));
                dst[vec] = q;
            }
        } else {
            for (std::int64_t column = threadIdx.x; column < width; column += blockDim.x)
                quantized[base + column] = static_cast<signed char>(
                    fminf(127.F, fmaxf(-128.F, nearbyintf(read(plan.inputs[0], base + column) / scale))));
        }
        return;
    }

    float minimum = 0.F, maximum = 0.F;
    if (aligned_f32) {
        const auto* src = static_cast<const float4*>(address(plan.inputs[0])) + base / 4;
        for (std::int64_t vec = threadIdx.x; vec < width / 4; vec += blockDim.x) {
            const float4 v = src[vec];
            minimum = fminf(minimum, fminf(fminf(v.x, v.y), fminf(v.z, v.w)));
            maximum = fmaxf(maximum, fmaxf(fmaxf(v.x, v.y), fmaxf(v.z, v.w)));
        }
    } else {
        for (std::int64_t column = threadIdx.x; column < width; column += blockDim.x) {
            const float value = read(plan.inputs[0], base + column);
            minimum = fminf(minimum, value);
            maximum = fmaxf(maximum, value);
        }
    }
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int delta = 16; delta > 0; delta >>= 1) {
        minimum = fminf(minimum, __shfl_down_sync(0xffffffffu, minimum, delta));
        maximum = fmaxf(maximum, __shfl_down_sync(0xffffffffu, maximum, delta));
    }
    if (lane == 0) {
        s_min[warp] = minimum;
        s_max[warp] = maximum;
    }
    __syncthreads();
    if (warp == 0) {
        minimum = lane < 8 ? s_min[lane] : 0.F;
        maximum = lane < 8 ? s_max[lane] : 0.F;
        for (int delta = 16; delta > 0; delta >>= 1) {
            minimum = fminf(minimum, __shfl_down_sync(0xffffffffu, minimum, delta));
            maximum = fmaxf(maximum, __shfl_down_sync(0xffffffffu, maximum, delta));
        }
        if (lane == 0) {
            s_min[0] = minimum;
            s_max[0] = maximum;
        }
    }
    __syncthreads();
    minimum = s_min[0];
    maximum = s_max[0];
    const float difference = maximum - minimum;
    // Match the CPU's dynamic quantization, including half-away-from-zero input rounding.
    const float scale = difference == 0.F ? 1.F : difference / 255.F;
    const float inverse_scale = difference == 0.F ? 0.F : 255.F / difference;
    int zero = -128;
    if (difference > 0.F) {
        // Separate rounding prevents fused multiply-add from changing a zero-point tie.
        const float scaled_minimum = __fmul_rn(minimum, inverse_scale);
        const float scaled_maximum = __fmul_rn(maximum, inverse_scale);
        const float minimum_error = __fadd_rn(-128.F, scaled_minimum);
        const float maximum_error = __fadd_rn(127.F, scaled_maximum);
        const float candidate = minimum_error + maximum_error > 0.F ? __fadd_rn(-128.F, -scaled_minimum)
                                                                    : __fadd_rn(127.F, -scaled_maximum);
        zero = static_cast<int>(nearbyintf(fminf(127.F, fmaxf(-128.F, candidate))));
    }
    if (threadIdx.x == 0) {
        scales[row] = scale;
        zeros[row] = zero;
    }
    if (aligned_f32) {
        const auto* src = static_cast<const float4*>(address(plan.inputs[0])) + base / 4;
        auto* dst = reinterpret_cast<char4*>(quantized + base);
        for (std::int64_t vec = threadIdx.x; vec < width / 4; vec += blockDim.x) {
            const float4 v = src[vec];
            char4 q;
            q.x = static_cast<signed char>(fminf(127.F, fmaxf(-128.F, roundf(v.x * inverse_scale) + zero)));
            q.y = static_cast<signed char>(fminf(127.F, fmaxf(-128.F, roundf(v.y * inverse_scale) + zero)));
            q.z = static_cast<signed char>(fminf(127.F, fmaxf(-128.F, roundf(v.z * inverse_scale) + zero)));
            q.w = static_cast<signed char>(fminf(127.F, fmaxf(-128.F, roundf(v.w * inverse_scale) + zero)));
            dst[vec] = q;
        }
    } else {
        for (std::int64_t column = threadIdx.x; column < width; column += blockDim.x)
            quantized[base + column] = static_cast<signed char>(
                fminf(127.F, fmaxf(-128.F, roundf(read(plan.inputs[0], base + column) * inverse_scale) + zero)));
    }
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

#if defined(KIDI_CUDA_USE_CUTLASS_INT8)
__global__ void projection_epilogue(Launch plan, const int* accumulators, const int* sums, const float* scales,
                                    const int* zeros) {
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= plan.elements) return;
    const auto columns = plan.output.shape[plan.output.rank - 1];
    const auto row = index / columns, column = index % columns;
    write_projection(plan, row, column, accumulators[index] - zeros[row] * sums[column], scales);
}
#endif

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

template <int Rows, int ColsPerWarp>
__global__ void __launch_bounds__(128)
    quantized_vector_fast(Launch plan, const signed char* quantized, const float* scales, const int* zeros) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const auto columns = plan.output.shape[plan.output.rank - 1];
    const auto width = plan.inputs[0].shape[plan.inputs[0].rank - 1];
    const auto col_base = (static_cast<std::int64_t>(blockIdx.x) * 4 + warp) * ColsPerWarp;
    if (col_base >= columns) return;
    const auto* weights = static_cast<const signed char*>(plan.inputs[1].data);
    int sum[Rows][ColsPerWarp]{};
    int wsum[ColsPerWarp]{};
    int z[Rows];
    bool need_wsum = false;
#pragma unroll
    for (int m = 0; m < Rows; ++m) {
        z[m] = zeros[m];
        need_wsum |= (z[m] != 0);
    }

    for (std::int64_t k = lane * 16; k < width; k += 32 * 16) {
        if constexpr (Rows == 1) {
            const uint4 x = *reinterpret_cast<const uint4*>(quantized + k);
#pragma unroll
            for (int j = 0; j < ColsPerWarp; ++j) {
                if (col_base + j >= columns) break;
                const uint4 w = *reinterpret_cast<const uint4*>(weights + (col_base + j) * width + k);
                sum[0][j] = __dp4a(static_cast<int>(x.x), static_cast<int>(w.x), sum[0][j]);
                sum[0][j] = __dp4a(static_cast<int>(x.y), static_cast<int>(w.y), sum[0][j]);
                sum[0][j] = __dp4a(static_cast<int>(x.z), static_cast<int>(w.z), sum[0][j]);
                sum[0][j] = __dp4a(static_cast<int>(x.w), static_cast<int>(w.w), sum[0][j]);
                if (need_wsum) {
                    wsum[j] = __dp4a(0x01010101, static_cast<int>(w.x), wsum[j]);
                    wsum[j] = __dp4a(0x01010101, static_cast<int>(w.y), wsum[j]);
                    wsum[j] = __dp4a(0x01010101, static_cast<int>(w.z), wsum[j]);
                    wsum[j] = __dp4a(0x01010101, static_cast<int>(w.w), wsum[j]);
                }
            }
        } else {
#pragma unroll
            for (int j = 0; j < ColsPerWarp; ++j) {
                if (col_base + j >= columns) break;
                const uint4 w = *reinterpret_cast<const uint4*>(weights + (col_base + j) * width + k);
                if (need_wsum) {
                    wsum[j] = __dp4a(0x01010101, static_cast<int>(w.x), wsum[j]);
                    wsum[j] = __dp4a(0x01010101, static_cast<int>(w.y), wsum[j]);
                    wsum[j] = __dp4a(0x01010101, static_cast<int>(w.z), wsum[j]);
                    wsum[j] = __dp4a(0x01010101, static_cast<int>(w.w), wsum[j]);
                }
#pragma unroll
                for (int m = 0; m < Rows; ++m) {
                    const uint4 x = *reinterpret_cast<const uint4*>(quantized + m * width + k);
                    sum[m][j] = __dp4a(static_cast<int>(x.x), static_cast<int>(w.x), sum[m][j]);
                    sum[m][j] = __dp4a(static_cast<int>(x.y), static_cast<int>(w.y), sum[m][j]);
                    sum[m][j] = __dp4a(static_cast<int>(x.z), static_cast<int>(w.z), sum[m][j]);
                    sum[m][j] = __dp4a(static_cast<int>(x.w), static_cast<int>(w.w), sum[m][j]);
                }
            }
        }
    }

#pragma unroll
    for (int j = 0; j < ColsPerWarp; ++j) {
#pragma unroll
        for (int m = 0; m < Rows; ++m)
            for (int delta = 16; delta > 0; delta >>= 1) sum[m][j] += __shfl_down_sync(0xffffffffu, sum[m][j], delta);
        if (need_wsum)
            for (int delta = 16; delta > 0; delta >>= 1) wsum[j] += __shfl_down_sync(0xffffffffu, wsum[j], delta);
    }
    if (lane == 0) {
#pragma unroll
        for (int j = 0; j < ColsPerWarp; ++j) {
            const auto col = col_base + j;
            if (col >= columns) break;
#pragma unroll
            for (int m = 0; m < Rows; ++m) write_projection(plan, m, col, sum[m][j] - z[m] * wsum[j], scales);
        }
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

__global__ void bfloat_vector_wide(Launch plan, std::int64_t width, std::int64_t columns) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const auto col_base = (static_cast<std::int64_t>(blockIdx.x) * 4 + warp) * 4;
    if (col_base >= columns) return;
    const auto* input = static_cast<const __nv_bfloat16*>(plan.inputs[0].data);
    const auto* weights = static_cast<const __nv_bfloat16*>(plan.inputs[1].data);
    float acc[4]{};
    for (std::int64_t contraction = lane * 8; contraction < width; contraction += 32 * 8) {
        union Vector {
            uint4 packed;
            __nv_bfloat162 pairs[4];
        };
        Vector left;
        left.packed = *reinterpret_cast<const uint4*>(input + contraction);
        float2 acts[4];
#pragma unroll
        for (int pair = 0; pair < 4; ++pair) acts[pair] = __bfloat1622float2(left.pairs[pair]);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            if (col_base + j >= columns) break;
            Vector right;
            right.packed = *reinterpret_cast<const uint4*>(weights + (col_base + j) * width + contraction);
#pragma unroll
            for (int pair = 0; pair < 4; ++pair) {
                const float2 param = __bfloat1622float2(right.pairs[pair]);
                acc[j] = fmaf(acts[pair].x, param.x, acc[j]);
                acc[j] = fmaf(acts[pair].y, param.y, acc[j]);
            }
        }
    }
#pragma unroll
    for (int j = 0; j < 4; ++j)
        for (int delta = 16; delta > 0; delta >>= 1) acc[j] += __shfl_down_sync(0xffffffff, acc[j], delta);
    if (lane == 0) {
        auto* out = static_cast<__nv_bfloat16*>(plan.output.data);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            if (col_base + j < columns) out[col_base + j] = __float2bfloat16(acc[j]);
        }
    }
}

__global__ void w8a16_matrix(Launch plan) {
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= plan.elements) return;
    const auto columns = plan.output.shape[plan.output.rank - 1];
    const auto width = plan.inputs[0].shape[plan.inputs[0].rank - 1];
    const auto row = index / columns, column = index % columns;
    const auto group = plan.attributes[1];
    const auto groups = width / group;
    const auto* input = static_cast<const __nv_bfloat16*>(plan.inputs[0].data) + row * width;
    const auto* weights = static_cast<const std::int8_t*>(plan.inputs[1].data) + column * width;
    const auto* scales = static_cast<const float*>(plan.inputs[2].data) + column * groups;
    float result = 0;
    for (std::int64_t block = 0; block < groups; ++block) {
        float sum = 0;
        for (std::int64_t slot = 0; slot < group; ++slot) {
            const auto k = block * group + slot;
            sum = fmaf(__bfloat162float(input[k]), static_cast<float>(weights[k]), sum);
        }
        result = fmaf(sum, scales[block], result);
    }
    static_cast<__nv_bfloat16*>(plan.output.data)[index] = __float2bfloat16(result);
}

template <int Rows, int ColsPerWarp, bool Grouped>
__global__ void __launch_bounds__(128)
    w8a16_vector(Launch plan, std::int64_t width, std::int64_t columns, std::int64_t group_size) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const auto col_base = (static_cast<std::int64_t>(blockIdx.x) * 4 + warp) * ColsPerWarp;
    if (col_base >= columns) return;
    const auto groups = width / group_size;
    const auto* input = static_cast<const __nv_bfloat16*>(plan.inputs[0].data);
    const auto* weights = static_cast<const std::int8_t*>(plan.inputs[1].data);
    const auto* scales = static_cast<const float*>(plan.inputs[2].data);
    float acc[Rows][ColsPerWarp]{};

    for (std::int64_t k = lane * 16; k < width; k += 32 * 16) {
        float scale_vals[ColsPerWarp];
        if constexpr (Grouped) {
            const auto g = k / group_size;
#pragma unroll
            for (int j = 0; j < ColsPerWarp; ++j)
                scale_vals[j] = col_base + j < columns ? scales[(col_base + j) * groups + g] : 0.F;
        }
        if constexpr (Rows == 1) {
            const uint4 x_lo = *reinterpret_cast<const uint4*>(input + k);
            const uint4 x_hi = *reinterpret_cast<const uint4*>(input + k + 8);
            const auto* x2_lo = reinterpret_cast<const __nv_bfloat162*>(&x_lo);
            const auto* x2_hi = reinterpret_cast<const __nv_bfloat162*>(&x_hi);
            float2 fx[8];
#pragma unroll
            for (int p = 0; p < 4; ++p) {
                fx[p] = __bfloat1622float2(x2_lo[p]);
                fx[4 + p] = __bfloat1622float2(x2_hi[p]);
            }
#pragma unroll
            for (int j = 0; j < ColsPerWarp; ++j) {
                if (col_base + j >= columns) break;
                const uint4 packed_w = *reinterpret_cast<const uint4*>(weights + (col_base + j) * width + k);
                const auto* w_pairs = reinterpret_cast<const std::uint16_t*>(&packed_w);
                float partial = 0.F;
#pragma unroll
                for (int p = 0; p < 8; ++p) {
                    const std::uint32_t u = w_pairs[p];
                    std::uint32_t lo_hi = 0, pos = 0, neg = 0;
                    asm("prmt.b32 %0, %1, 0, 0x4140;" : "=r"(lo_hi) : "r"(u));
                    constexpr std::uint32_t imm = (0xf0U & 0xccU) ^ 0xaaU;
                    asm("lop3.b32 %0, %1, 0x007f007f, 0x43004300, %2;" : "=r"(pos) : "r"(lo_hi), "n"(imm));
                    asm("lop3.b32 %0, %1, 0x00800080, 0x43004300, %2;" : "=r"(neg) : "r"(lo_hi), "n"(imm));
                    const __nv_bfloat162 fw = __hsub2(reinterpret_cast<const __nv_bfloat162&>(pos),
                                                      reinterpret_cast<const __nv_bfloat162&>(neg));
                    const float2 wf = __bfloat1622float2(fw);
                    partial = fmaf(fx[p].x, wf.x, partial);
                    partial = fmaf(fx[p].y, wf.y, partial);
                }
                if constexpr (Grouped)
                    acc[0][j] = fmaf(partial, scale_vals[j], acc[0][j]);
                else
                    acc[0][j] += partial;
            }
        } else {
#pragma unroll
            for (int j = 0; j < ColsPerWarp; ++j) {
                if (col_base + j >= columns) break;
                const uint4 packed_w = *reinterpret_cast<const uint4*>(weights + (col_base + j) * width + k);
                const auto* w_pairs = reinterpret_cast<const std::uint16_t*>(&packed_w);
#pragma unroll
                for (int m = 0; m < Rows; ++m) {
                    const uint4 x_lo = *reinterpret_cast<const uint4*>(input + m * width + k);
                    const uint4 x_hi = *reinterpret_cast<const uint4*>(input + m * width + k + 8);
                    const auto* x2_lo = reinterpret_cast<const __nv_bfloat162*>(&x_lo);
                    const auto* x2_hi = reinterpret_cast<const __nv_bfloat162*>(&x_hi);
                    float partial = 0.F;
#pragma unroll
                    for (int p = 0; p < 8; ++p) {
                        const std::uint32_t u = w_pairs[p];
                        std::uint32_t lo_hi = 0, pos = 0, neg = 0;
                        asm("prmt.b32 %0, %1, 0, 0x4140;" : "=r"(lo_hi) : "r"(u));
                        constexpr std::uint32_t imm = (0xf0U & 0xccU) ^ 0xaaU;
                        asm("lop3.b32 %0, %1, 0x007f007f, 0x43004300, %2;" : "=r"(pos) : "r"(lo_hi), "n"(imm));
                        asm("lop3.b32 %0, %1, 0x00800080, 0x43004300, %2;" : "=r"(neg) : "r"(lo_hi), "n"(imm));
                        const __nv_bfloat162 fw = __hsub2(reinterpret_cast<const __nv_bfloat162&>(pos),
                                                          reinterpret_cast<const __nv_bfloat162&>(neg));
                        const float2 wf = __bfloat1622float2(fw);
                        const float2 fx = __bfloat1622float2(p < 4 ? x2_lo[p] : x2_hi[p - 4]);
                        partial = fmaf(fx.x, wf.x, partial);
                        partial = fmaf(fx.y, wf.y, partial);
                    }
                    if constexpr (Grouped)
                        acc[m][j] = fmaf(partial, scale_vals[j], acc[m][j]);
                    else
                        acc[m][j] += partial;
                }
            }
        }
    }

#pragma unroll
    for (int m = 0; m < Rows; ++m)
#pragma unroll
        for (int j = 0; j < ColsPerWarp; ++j)
            for (int delta = 16; delta > 0; delta >>= 1) acc[m][j] += __shfl_down_sync(0xffffffff, acc[m][j], delta);

    if (lane == 0) {
        auto* output = static_cast<__nv_bfloat16*>(plan.output.data);
#pragma unroll
        for (int j = 0; j < ColsPerWarp; ++j) {
            const auto col = col_base + j;
            if (col >= columns) break;
            const float scale = Grouped ? 1.F : scales[col];
#pragma unroll
            for (int m = 0; m < Rows; ++m) output[m * columns + col] = __float2bfloat16(acc[m][j] * scale);
        }
    }
}

template <int ColsPerWarp, bool Grouped>
void run_w8a16_vector(const Launch& launch, unsigned int grid, std::int64_t rows, std::int64_t width,
                      std::int64_t columns, std::int64_t group_size, cudaStream_t stream) {
    if (rows == 1)
        w8a16_vector<1, ColsPerWarp, Grouped><<<grid, 128, 0, stream>>>(launch, width, columns, group_size);
    else if (rows == 2)
        w8a16_vector<2, ColsPerWarp, Grouped><<<grid, 128, 0, stream>>>(launch, width, columns, group_size);
    else if (rows == 3)
        w8a16_vector<3, ColsPerWarp, Grouped><<<grid, 128, 0, stream>>>(launch, width, columns, group_size);
    else
        w8a16_vector<4, ColsPerWarp, Grouped><<<grid, 128, 0, stream>>>(launch, width, columns, group_size);
}

#if defined(KIDI_CUDA_USE_CUTLASS_INT8)
__global__ void float_vector(Launch plan, std::int64_t width, std::int64_t columns) {
    const int lane = threadIdx.x % 32;
    const auto column = static_cast<std::int64_t>(blockIdx.x) * 4 + threadIdx.x / 32;
    if (column >= columns) return;
    const auto* input = static_cast<const float*>(plan.inputs[0].data);
    const auto* weight = static_cast<const float*>(plan.inputs[1].data) + column * width;
    float4 sums{};
    for (std::int64_t contraction = lane * 4; contraction < width; contraction += 32 * 4) {
        const auto left = *reinterpret_cast<const float4*>(input + contraction);
        const auto right = *reinterpret_cast<const float4*>(weight + contraction);
        sums.x = fmaf(left.x, right.x, sums.x);
        sums.y = fmaf(left.y, right.y, sums.y);
        sums.z = fmaf(left.z, right.z, sums.z);
        sums.w = fmaf(left.w, right.w, sums.w);
    }
    float sum = sums.x + sums.y + sums.z + sums.w;
    for (int delta = 16; delta > 0; delta /= 2) sum += __shfl_down_sync(0xffffffff, sum, delta);
    if (lane == 0) {
        if (plan.count == 3) sum += read(plan.inputs[2], column);
        static_cast<float*>(plan.output.data)[column] = sum;
    }
}
#endif

} // namespace kernel
} // namespace

auto prepare_projection_weights(const Launch& plan, int bits, void* expanded, void* sums, cudaStream_t stream) -> void {
    const auto columns = static_cast<unsigned int>(plan.output.shape[plan.output.rank - 1]);
    if (bits == 2)
        kernel::prepare_projection_weights<2>
            <<<columns, 256, 0, stream>>>(plan, static_cast<signed char*>(expanded), static_cast<int*>(sums));
    else if (bits == 4)
        kernel::prepare_projection_weights<4>
            <<<columns, 256, 0, stream>>>(plan, static_cast<signed char*>(expanded), static_cast<int*>(sums));
    else
        kernel::prepare_projection_weights<8><<<columns, 256, 0, stream>>>(plan, nullptr, static_cast<int*>(sums));
}

auto quantize_rows(const Launch& plan, std::int64_t rows, void* scratch, void* scales, void* zeros, cudaStream_t stream)
    -> void {
    kernel::quantize_rows<<<static_cast<unsigned int>(rows), 256, 0, stream>>>(
        plan, static_cast<signed char*>(scratch), static_cast<float*>(scales), static_cast<int*>(zeros));
}

auto execute_matrix(const Launch& plan, int device, bool optimized, void* workspace, int slices, cudaStream_t stream)
    -> void {
    const auto blocks = static_cast<unsigned int>((plan.elements + 255) / 256);
    const auto& left = plan.inputs[0];
    const auto& right = plan.inputs[1];
    const auto width = left.shape[left.rank - 1];
    const auto columns = plan.output.shape[plan.output.rank - 1];
    const bool bfloat = left.type == static_cast<int>(Type::BF16) && right.type == static_cast<int>(Type::BF16) &&
                        plan.output.type == static_cast<int>(Type::BF16);
    if (optimized && bfloat && plan.count == 2 && right.rank == 2 && plan.attribute_count && plan.attributes[0] &&
        plan.elements == static_cast<std::size_t>(columns) && width > 0 && width <= 65536 && columns > 0 &&
        reinterpret_cast<std::uintptr_t>(left.data) % 16 == 0 &&
        reinterpret_cast<std::uintptr_t>(right.data) % 16 == 0) {
        if (columns <= 4096 && width % 256 == 0) {
            if (width >= 1024)
                kernel::bfloat_vector_block<<<static_cast<unsigned int>(columns), 128, 0, stream>>>(plan, width);
            else
                kernel::bfloat_vector<<<static_cast<unsigned int>((columns + 3) / 4), 128, 0, stream>>>(plan, width,
                                                                                                        columns);
            return;
        }
        if (width % 8 == 0) {
            kernel::bfloat_vector_wide<<<static_cast<unsigned int>((columns + 15) / 16), 128, 0, stream>>>(plan, width,
                                                                                                           columns);
            return;
        }
    }
    (void)device;
    (void)workspace;
    (void)slices;
#if defined(KIDI_CUDA_USE_CUTLASS_INT8)
    const bool fp32 = left.type == static_cast<int>(Type::F32) && right.type == static_cast<int>(Type::F32) &&
                      plan.output.type == static_cast<int>(Type::F32);
    if (optimized && (fp32 || (bfloat && plan.count == 2 && width % 8 == 0 && columns % 8 == 0)) && right.rank == 2 &&
        plan.attribute_count && plan.attributes[0] && width > 0 && width <= INT32_MAX && columns > 0 &&
        columns <= INT32_MAX && plan.elements / columns <= INT32_MAX &&
        reinterpret_cast<std::uintptr_t>(left.data) % 16 == 0 &&
        reinterpret_cast<std::uintptr_t>(right.data) % 16 == 0 &&
        reinterpret_cast<std::uintptr_t>(plan.output.data) % 16 == 0) {
        const auto rows = static_cast<int>(plan.elements / columns);
        if (fp32 && rows == 1 && width % 4 == 0) {
            kernel::float_vector<<<static_cast<unsigned int>((columns + 3) / 4), 128, 0, stream>>>(plan, width,
                                                                                                   columns);
            return;
        }
        const auto major = dense_device(device).major;
        if (fp32) {
            float_gemm(rows, static_cast<int>(columns), static_cast<int>(width), left.data, right.data,
                       plan.output.data, stream, major >= 8, workspace, slices);
            if (plan.count == 3) {
                Launch bias = plan;
                bias.operation = static_cast<int>(Op::ADD);
                bias.inputs[0] = plan.output;
                bias.inputs[1] = plan.inputs[2];
                pointwise(bias, stream);
            }
            return;
        }
        if (major >= 8) {
            bfloat_gemm(rows, static_cast<int>(columns), static_cast<int>(width), left.data, right.data,
                        plan.output.data, stream, workspace, slices);
            return;
        }
    }
#endif
    kernel::matrix<<<blocks, 256, 0, stream>>>(plan);
}

auto floating_projection(const Launch& plan, int device, bool reference_projection, void* shared_workspace,
                         void* scratch, int dense_slices, cudaStream_t stream) -> void {
    if (plan.output.type == static_cast<int>(Type::BF16)) {
        if (reference_projection || plan.count == 3) {
            const auto blocks = static_cast<unsigned int>((plan.elements + 255) / 256);
            kernel::matrix<<<blocks, 256, 0, stream>>>(plan);
        } else {
            execute_matrix(plan, device, true, shared_workspace ? shared_workspace : scratch, dense_slices, stream);
        }
        return;
    }
    execute_matrix(plan, device, !reference_projection, scratch, dense_slices, stream);
}

auto quantized_projection(const Launch& plan, int device, bool reference_projection, bool vector_projection,
                          bool tensor_projection, bool fused_w8a8, void* scratch, void* scales, void* zeros,
                          const void* prepared_expanded, const void* prepared_sums, void* accumulators,
                          void* shared_workspace, int dense_slices, cudaStream_t stream) -> void {
    const auto blocks = static_cast<unsigned int>((plan.elements + 255) / 256);
    const auto output_columns = plan.output.shape[plan.output.rank - 1];
    const auto rows = plan.elements / output_columns;
    const auto width = plan.inputs[0].shape[plan.inputs[0].rank - 1];
    if (vector_projection && reinterpret_cast<std::uintptr_t>(plan.inputs[1].data) % 4 == 0) {
        const auto bits = plan.operation == static_cast<int>(Op::PACKED_LINEAR) ? plan.attributes[0] : 8;
        if (bits == 8 && width % 16 == 0 && reinterpret_cast<std::uintptr_t>(plan.inputs[1].data) % 16 == 0) {
            const bool wide = rows == 1 && output_columns >= 32768;
            const auto grid = static_cast<unsigned int>((output_columns + (wide ? 15 : 3)) / (wide ? 16 : 4));
            if (wide)
                kernel::quantized_vector_fast<1, 4>
                    <<<grid, 128, 0, stream>>>(plan, static_cast<const signed char*>(scratch),
                                               static_cast<const float*>(scales), static_cast<const int*>(zeros));
            else if (rows == 1)
                kernel::quantized_vector_fast<1, 1>
                    <<<grid, 128, 0, stream>>>(plan, static_cast<const signed char*>(scratch),
                                               static_cast<const float*>(scales), static_cast<const int*>(zeros));
            else if (rows == 2)
                kernel::quantized_vector_fast<2, 1>
                    <<<grid, 128, 0, stream>>>(plan, static_cast<const signed char*>(scratch),
                                               static_cast<const float*>(scales), static_cast<const int*>(zeros));
            else if (rows == 3)
                kernel::quantized_vector_fast<3, 1>
                    <<<grid, 128, 0, stream>>>(plan, static_cast<const signed char*>(scratch),
                                               static_cast<const float*>(scales), static_cast<const int*>(zeros));
            else
                kernel::quantized_vector_fast<4, 1>
                    <<<grid, 128, 0, stream>>>(plan, static_cast<const signed char*>(scratch),
                                               static_cast<const float*>(scales), static_cast<const int*>(zeros));
            return;
        }
        const dim3 grid(static_cast<unsigned int>((output_columns + 3) / 4), static_cast<unsigned int>(rows));
        if (bits == 2)
            kernel::quantized_vector<2><<<grid, 128, 0, stream>>>(plan, static_cast<const signed char*>(scratch),
                                                                  static_cast<const float*>(scales),
                                                                  static_cast<const int*>(zeros));
        else if (bits == 4)
            kernel::quantized_vector<4><<<grid, 128, 0, stream>>>(plan, static_cast<const signed char*>(scratch),
                                                                  static_cast<const float*>(scales),
                                                                  static_cast<const int*>(zeros));
        else
            kernel::quantized_vector<8><<<grid, 128, 0, stream>>>(plan, static_cast<const signed char*>(scratch),
                                                                  static_cast<const float*>(scales),
                                                                  static_cast<const int*>(zeros));
        return;
    }
    (void)device;
    (void)reference_projection;
    (void)tensor_projection;
    (void)fused_w8a8;
    (void)prepared_expanded;
    (void)prepared_sums;
    (void)accumulators;
    (void)shared_workspace;
    (void)dense_slices;
#if defined(KIDI_CUDA_USE_CUTLASS_INT8)
    constexpr std::size_t WEIGHT_ALIGNMENT = 16;
    if (!tensor_projection || vector_projection ||
        reinterpret_cast<std::uintptr_t>(plan.inputs[1].data) % WEIGHT_ALIGNMENT)
#endif
    {
        kernel::quantized_matrix<<<blocks, 256, 0, stream>>>(plan, static_cast<const signed char*>(scratch),
                                                             static_cast<const float*>(scales),
                                                             static_cast<const int*>(zeros));
        return;
    }
#if defined(KIDI_CUDA_USE_CUTLASS_INT8)
    if (fused_w8a8) {
        const float* bias_ptr =
            plan.operation == static_cast<int>(Op::QUANTIZED_LINEAR) && plan.count >= 4 && plan.inputs[3].data
                ? static_cast<const float*>(plan.inputs[3].data)
                : nullptr;
        float output_scale = 0.F;
        if (plan.operation == static_cast<int>(Op::PACKED_LINEAR) && plan.attribute_count > 2) {
            const auto raw_scale = static_cast<std::int32_t>(plan.attributes[2]);
            std::memcpy(&output_scale, &raw_scale, sizeof(output_scale));
        }
        w8a8_gemm(static_cast<int>(rows), static_cast<int>(output_columns), static_cast<int>(width),
                  static_cast<const std::int8_t*>(scratch),
                  static_cast<const std::int8_t*>(prepared_expanded ? prepared_expanded : plan.inputs[1].data),
                  static_cast<const float*>(scales), static_cast<const int*>(zeros),
                  static_cast<const int*>(prepared_sums), static_cast<const float*>(plan.inputs[2].data), bias_ptr,
                  output_scale, plan.output.data, plan.output.type == static_cast<int>(Type::BF16), stream,
                  shared_workspace ? shared_workspace : accumulators, dense_slices);
        return;
    }
    int8_gemm(static_cast<int>(rows), static_cast<int>(output_columns), static_cast<int>(width),
              static_cast<const std::int8_t*>(scratch),
              static_cast<const std::int8_t*>(prepared_expanded ? prepared_expanded : plan.inputs[1].data),
              static_cast<std::int32_t*>(accumulators), stream);
    kernel::projection_epilogue<<<blocks, 256, 0, stream>>>(
        plan, static_cast<const int*>(accumulators), static_cast<const int*>(prepared_sums),
        static_cast<const float*>(scales), static_cast<const int*>(zeros));
#endif
}

auto w8a16_projection(const Launch& plan, int device, bool reference_projection, void* shared_workspace, void* scratch,
                      int dense_slices, cudaStream_t stream) -> void {
    const auto blocks = static_cast<unsigned int>((plan.elements + 255) / 256);
    const auto output_columns = plan.output.shape[plan.output.rank - 1];
    const auto width = plan.inputs[0].shape[plan.inputs[0].rank - 1];
    const auto rows = static_cast<std::int64_t>(plan.elements / output_columns);
    const auto group_size = plan.attributes[1];
    const bool aligned = reinterpret_cast<std::uintptr_t>(plan.inputs[0].data) % 16 == 0 &&
                         reinterpret_cast<std::uintptr_t>(plan.inputs[1].data) % 16 == 0 &&
                         reinterpret_cast<std::uintptr_t>(plan.output.data) % 4 == 0;
    if (!reference_projection && aligned && rows >= 1 && rows <= 4 && width > 0 && width <= 65536 && width % 16 == 0 &&
        group_size > 0 && group_size % 16 == 0 && width % group_size == 0) {
        const bool grouped = group_size < width;
        const bool wide = output_columns >= 4096;
        const auto grid = static_cast<unsigned int>((output_columns + (wide ? 15 : 3)) / (wide ? 16 : 4));
        if (grouped) {
            if (wide)
                kernel::run_w8a16_vector<4, true>(plan, grid, rows, width, output_columns, group_size, stream);
            else
                kernel::run_w8a16_vector<1, true>(plan, grid, rows, width, output_columns, group_size, stream);
        } else {
            if (wide)
                kernel::run_w8a16_vector<4, false>(plan, grid, rows, width, output_columns, group_size, stream);
            else
                kernel::run_w8a16_vector<1, false>(plan, grid, rows, width, output_columns, group_size, stream);
        }
        return;
    }
    (void)device;
    (void)shared_workspace;
    (void)scratch;
    (void)dense_slices;
#if defined(KIDI_CUDA_USE_CUTLASS_INT8)
    if (!reference_projection && aligned && rows > 4 && rows <= INT32_MAX && width > 0 && width <= INT32_MAX &&
        width % 64 == 0 && group_size > 0 && group_size % 64 == 0 && width % group_size == 0 && output_columns > 0 &&
        output_columns <= INT32_MAX && output_columns % 8 == 0 && dense_device(device).major >= 8) {
        w8a16_gemm(static_cast<int>(rows), static_cast<int>(output_columns), static_cast<int>(width),
                   static_cast<int>(group_size), plan.inputs[0].data, plan.inputs[1].data,
                   static_cast<const float*>(plan.inputs[2].data), plan.output.data, stream,
                   shared_workspace ? shared_workspace : scratch, dense_slices);
        return;
    }
#endif
    kernel::w8a16_matrix<<<blocks, 256, 0, stream>>>(plan);
}

} // namespace kidi::cuda
