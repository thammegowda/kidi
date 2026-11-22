#include "cuda/dense_gemm.h"
#include "cuda/ptx_mma.cuh"

#include <cuda_bf16.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace kidi::cuda {
namespace {

constexpr int FloatRows = 32, FloatColumns = 64, FloatDepth = 32, FloatStages = 3, FloatThreads = 128;
using Bfloat = __nv_bfloat16;
using BfloatA = ptx::Fragment<unsigned, 4>;
using BfloatB = ptx::Fragment<unsigned, 2>;
using BfloatC = ptx::Fragment<float, 4>;

constexpr int W8Depth = 64, W8WeightStride = 80, W8Stages = 3;

__device__ __forceinline__ std::uint32_t convert_int8_pair_to_bfloat2(std::uint16_t pair) {
    std::uint32_t unpacked = 0, positive = 0, negative = 0;
    asm("prmt.b32 %0, %1, 0, 0x4140;" : "=r"(unpacked) : "r"(static_cast<std::uint32_t>(pair)));
    constexpr std::uint32_t imm = (0xf0U & 0xccU) ^ 0xaaU;
    asm("lop3.b32 %0, %1, 0x007f007f, 0x43004300, %2;" : "=r"(positive) : "r"(unpacked), "n"(imm));
    asm("lop3.b32 %0, %1, 0x00800080, 0x43004300, %2;" : "=r"(negative) : "r"(unpacked), "n"(imm));
    const __nv_bfloat162 result =
        __hsub2(reinterpret_cast<const __nv_bfloat162&>(positive), reinterpret_cast<const __nv_bfloat162&>(negative));
    return reinterpret_cast<const std::uint32_t&>(result);
}

template <int TileRows, int TileCols>
struct alignas(16) W8Shared {
    __nv_bfloat16 activations[W8Stages][TileRows][W8Depth];
    std::int8_t parameters[W8Stages][TileCols][W8WeightStride];
};

template <int TileRows, int Threads>
__device__ void stage_w8a16_activations(__nv_bfloat16 (*shared)[W8Depth], const __nv_bfloat16* input, int rows,
                                        int width, int origin, int depth) {
#pragma unroll
    for (int vector = threadIdx.x; vector < TileRows * (W8Depth / 8); vector += Threads) {
        const int row = vector >> 3, chunk = vector & 7;
        const bool valid = origin + row < rows && depth + chunk * 8 < width;
        const auto* source =
            valid ? input + static_cast<std::int64_t>(origin + row) * width + depth + chunk * 8 : input;
        ptx::cp_async_zfill_16<false>(&shared[row][(chunk ^ (row & 3)) * 8], source, valid);
    }
}

template <int TileCols, int Threads>
__device__ void stage_w8a16_weights(std::int8_t (*shared)[W8WeightStride], const std::int8_t* weights, int columns,
                                    int width, int origin, int depth) {
#pragma unroll
    for (int vector = threadIdx.x; vector < TileCols * (W8Depth / 16); vector += Threads) {
        const int column = vector >> 2, chunk = vector & 3;
        const bool valid = origin + column < columns && depth + chunk * 16 < width;
        const auto* source =
            valid ? weights + static_cast<std::int64_t>(origin + column) * width + depth + chunk * 16 : weights;
        ptx::cp_async_zfill_16<true>(&shared[column][chunk * 16], source, valid);
    }
}

template <int TileRows, int TileCols, int MWarps, int NWarps, bool Grouped>
__global__ void __launch_bounds__(MWarps * NWarps * 32)
    pipelined_w8a16_gemm(int rows, int columns, int width, int group_size, const __nv_bfloat16* __restrict__ input,
                         const std::int8_t* __restrict__ weights, const float* __restrict__ scales, void* output) {
    extern __shared__ __align__(16) unsigned char dynamic_shared[];
    auto& shared = *reinterpret_cast<W8Shared<TileRows, TileCols>*>(dynamic_shared);
    constexpr int Threads = MWarps * NWarps * 32;
    constexpr int WarpRows = TileRows / MWarps, RowMmas = WarpRows / 16, ColMmas = (TileCols / NWarps) / 8;
    const int row_start = blockIdx.y * TileRows, col_start = blockIdx.x * TileCols;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int warp_m = warp / NWarps, warp_n = warp % NWarps;
    const int group_lane = lane >> 2, tid = lane & 3;
    const int groups = width / group_size, group_tiles = group_size / W8Depth;

    const int total_tiles = width / W8Depth;
    int start = 0, tiles = total_tiles, start_group = 0;
    if constexpr (Grouped) {
        if (gridDim.z > 1) {
            start_group = groups * blockIdx.z / gridDim.z;
            const int end_group = groups * (blockIdx.z + 1) / gridDim.z;
            start = start_group * group_tiles;
            tiles = (end_group - start_group) * group_tiles;
        }
    } else if (gridDim.z > 1) {
        start = total_tiles * blockIdx.z / gridDim.z;
        tiles = total_tiles * (blockIdx.z + 1) / gridDim.z - start;
    }
    if (tiles <= 0) return;

    BfloatC accumulators[RowMmas][ColMmas];
    BfloatC totals[Grouped ? RowMmas : 1][Grouped ? ColMmas : 1];
#pragma unroll
    for (int mi = 0; mi < RowMmas; ++mi)
#pragma unroll
        for (int ni = 0; ni < ColMmas; ++ni) {
            accumulators[mi][ni].clear();
            if constexpr (Grouped) totals[mi][ni].clear();
        }

    for (int tile = 0; tile < W8Stages - 1; ++tile) {
        if (tile < tiles) {
            stage_w8a16_activations<TileRows, Threads>(shared.activations[tile], input, rows, width, row_start,
                                                       (start + tile) * W8Depth);
            stage_w8a16_weights<TileCols, Threads>(shared.parameters[tile], weights, columns, width, col_start,
                                                   (start + tile) * W8Depth);
        }
        ptx::cp_async_fence();
    }

    const int sub = lane >> 3, sub_row = (sub & 1) * 8 + (lane & 7), sub_col = sub >> 1;
    for (int tile = 0; tile < tiles; ++tile) {
        const int next = tile + W8Stages - 1, next_slot = next % W8Stages;
        if (next < tiles) {
            stage_w8a16_activations<TileRows, Threads>(shared.activations[next_slot], input, rows, width, row_start,
                                                       (start + next) * W8Depth);
            stage_w8a16_weights<TileCols, Threads>(shared.parameters[next_slot], weights, columns, width, col_start,
                                                   (start + next) * W8Depth);
        }
        ptx::cp_async_fence();
        ptx::cp_async_wait<W8Stages - 1>();
        __syncthreads();
        const int slot = tile % W8Stages;

#pragma unroll
        for (int step = 0; step < 4; ++step) {
            BfloatA frag_a[RowMmas];
#pragma unroll
            for (int mi = 0; mi < RowMmas; ++mi) {
                const int r = warp_m * WarpRows + mi * 16 + sub_row;
                const int c8 = (step * 2 + sub_col) ^ (r & 3);
                ptx::ldmatrix_x4<false>(frag_a[mi], &shared.activations[slot][r][c8 * 8]);
            }
            BfloatB frag_b[ColMmas];
#pragma unroll
            for (int ni = 0; ni < ColMmas; ++ni) {
                const int n = warp_n * (TileCols / NWarps) + ni * 8 + group_lane;
                const auto* row_b = &shared.parameters[slot][n][step * 16];
                frag_b[ni][0] =
                    convert_int8_pair_to_bfloat2(*reinterpret_cast<const std::uint16_t*>(row_b + tid * 2));
                frag_b[ni][1] =
                    convert_int8_pair_to_bfloat2(*reinterpret_cast<const std::uint16_t*>(row_b + 8 + tid * 2));
            }
#pragma unroll
            for (int mi = 0; mi < RowMmas; ++mi)
#pragma unroll
                for (int ni = 0; ni < ColMmas; ++ni)
                    ptx::mma_bf16_m16n8k16(accumulators[mi][ni], frag_a[mi], frag_b[ni], accumulators[mi][ni]);
        }
        __syncthreads();

        if constexpr (Grouped) {
            if ((tile + 1) % group_tiles == 0) {
                const int group = start_group + tile / group_tiles;
#pragma unroll
                for (int ni = 0; ni < ColMmas; ++ni) {
                    const int col0 = col_start + warp_n * (TileCols / NWarps) + ni * 8 + tid * 2;
                    const float s0 = col0 < columns ? scales[static_cast<std::int64_t>(col0) * groups + group] : 0.F;
                    const float s1 =
                        col0 + 1 < columns ? scales[static_cast<std::int64_t>(col0 + 1) * groups + group] : 0.F;
#pragma unroll
                    for (int mi = 0; mi < RowMmas; ++mi) {
                        totals[mi][ni][0] = fmaf(accumulators[mi][ni][0], s0, totals[mi][ni][0]);
                        totals[mi][ni][1] = fmaf(accumulators[mi][ni][1], s1, totals[mi][ni][1]);
                        totals[mi][ni][2] = fmaf(accumulators[mi][ni][2], s0, totals[mi][ni][2]);
                        totals[mi][ni][3] = fmaf(accumulators[mi][ni][3], s1, totals[mi][ni][3]);
                        accumulators[mi][ni].clear();
                    }
                }
            }
        }
    }

    if (gridDim.z > 1) {
        auto* partial = static_cast<float*>(output) + static_cast<std::int64_t>(blockIdx.z) * rows * columns;
#pragma unroll
        for (int mi = 0; mi < RowMmas; ++mi) {
            const int r0 = row_start + warp_m * WarpRows + mi * 16 + group_lane;
            const int r1 = r0 + 8;
#pragma unroll
            for (int ni = 0; ni < ColMmas; ++ni) {
                const auto& src = Grouped ? totals[mi][ni] : accumulators[mi][ni];
                const int c0 = col_start + warp_n * (TileCols / NWarps) + ni * 8 + tid * 2;
                if (c0 + 1 < columns) {
                    if (r0 < rows) {
                        partial[static_cast<std::int64_t>(r0) * columns + c0] = src[0];
                        partial[static_cast<std::int64_t>(r0) * columns + c0 + 1] = src[1];
                    }
                    if (r1 < rows) {
                        partial[static_cast<std::int64_t>(r1) * columns + c0] = src[2];
                        partial[static_cast<std::int64_t>(r1) * columns + c0 + 1] = src[3];
                    }
                }
            }
        }
        return;
    }

    auto* destination = static_cast<__nv_bfloat16*>(output);
#pragma unroll
    for (int ni = 0; ni < ColMmas; ++ni) {
        const int c0 = col_start + warp_n * (TileCols / NWarps) + ni * 8 + tid * 2;
        if (c0 + 1 >= columns) continue;
        const float s0 = Grouped ? 1.F : scales[c0];
        const float s1 = Grouped ? 1.F : scales[c0 + 1];
#pragma unroll
        for (int mi = 0; mi < RowMmas; ++mi) {
            const auto& src = Grouped ? totals[mi][ni] : accumulators[mi][ni];
            const int r0 = row_start + warp_m * WarpRows + mi * 16 + group_lane;
            const int r1 = r0 + 8;
            if (r0 < rows) {
                *reinterpret_cast<__nv_bfloat162*>(destination + static_cast<std::int64_t>(r0) * columns + c0) =
                    __halves2bfloat162(__float2bfloat16(src[0] * s0), __float2bfloat16(src[1] * s1));
            }
            if (r1 < rows) {
                *reinterpret_cast<__nv_bfloat162*>(destination + static_cast<std::int64_t>(r1) * columns + c0) =
                    __halves2bfloat162(__float2bfloat16(src[2] * s0), __float2bfloat16(src[3] * s1));
            }
        }
    }
}

__global__ void reduce_w8a16_partials(const float* __restrict__ workspace, const float* __restrict__ scales,
                                      __nv_bfloat16* __restrict__ output, int rows, int columns, int slices) {
    const auto pair_index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const auto total_pairs = static_cast<std::size_t>(rows) * (columns / 2);
    if (pair_index >= total_pairs) return;
    const auto element = pair_index * 2;
    const int col = static_cast<int>(element % columns);
    const std::size_t stride = static_cast<std::size_t>(rows) * columns;
    float sum0 = 0.F, sum1 = 0.F;
    for (int slice = 0; slice < slices; ++slice) {
        const float2 values = *reinterpret_cast<const float2*>(workspace + slice * stride + element);
        sum0 += values.x;
        sum1 += values.y;
    }
    const float s0 = scales ? scales[col] : 1.F;
    const float s1 = scales ? scales[col + 1] : 1.F;
    *reinterpret_cast<__nv_bfloat162*>(output + element) =
        __halves2bfloat162(__float2bfloat16(sum0 * s0), __float2bfloat16(sum1 * s1));
}

template <int TileRows>
__device__ void stage_float_tile(float (*shared)[FloatDepth], const float* operand, int rows, int width, int origin,
                                 int depth) {
    for (int vector = threadIdx.x; vector < TileRows * (FloatDepth / 4); vector += FloatThreads) {
        const int row = vector / (FloatDepth / 4), column = (vector % (FloatDepth / 4)) * 4;
        const bool valid = origin + row < rows && depth + column < width;
        const auto* source =
            valid ? operand + static_cast<std::int64_t>(origin + row) * width + depth + column : operand;
        ptx::cp_async_zfill_16<true>(&shared[row][column ^ ((row & 7) * 4)], source, valid);
    }
}

__global__ void pipelined_float_gemm(int rows, int columns, int width, const float* input, const float* weights,
                                     float* output) {
    __shared__ __align__(16) float activations[FloatStages][FloatRows][FloatDepth];
    __shared__ __align__(16) float parameters[FloatStages][FloatColumns][FloatDepth];
    const int row_start = blockIdx.y * FloatRows, column_start = blockIdx.x * FloatColumns;
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const int row_lane = warp / 2 * 16 + lane / 8, column_lane = warp % 2 * 32 + lane % 8;
    float sums[4][4]{};
    const int total_tiles = (width - 1) / FloatDepth + 1;
    int start = 0, tiles = total_tiles;
    if (gridDim.z > 1) {
        start = total_tiles * blockIdx.z / gridDim.z;
        tiles = total_tiles * (blockIdx.z + 1) / gridDim.z - start;
        output += static_cast<std::int64_t>(blockIdx.z) * rows * columns;
    }

    for (int tile = 0; tile < FloatStages - 1; ++tile) {
        if (tile < tiles) {
            stage_float_tile<FloatRows>(activations[tile], input, rows, width, row_start, (start + tile) * FloatDepth);
            stage_float_tile<FloatColumns>(parameters[tile], weights, columns, width, column_start,
                                           (start + tile) * FloatDepth);
        }
        ptx::cp_async_fence();
    }
    for (int tile = 0; tile < tiles; ++tile) {
        const int next = tile + FloatStages - 1, next_slot = next % FloatStages;
        if (next < tiles) {
            stage_float_tile<FloatRows>(activations[next_slot], input, rows, width, row_start,
                                        (start + next) * FloatDepth);
            stage_float_tile<FloatColumns>(parameters[next_slot], weights, columns, width, column_start,
                                           (start + next) * FloatDepth);
        }
        ptx::cp_async_fence();
        ptx::cp_async_wait<FloatStages - 1>();
        __syncthreads();
        const int slot = tile % FloatStages;
#pragma unroll
        for (int depth = 0; depth < FloatDepth; depth += 4) {
            float4 left[4], right[4];
#pragma unroll
            for (int value = 0; value < 4; ++value) {
                const int row = row_lane + value * 4, column = column_lane + value * 8;
                left[value] = *reinterpret_cast<const float4*>(&activations[slot][row][depth ^ ((row & 7) * 4)]);
                right[value] = *reinterpret_cast<const float4*>(&parameters[slot][column][depth ^ ((column & 7) * 4)]);
            }
#pragma unroll
            for (int element = 0; element < 4; ++element)
#pragma unroll
                for (int row = 0; row < 4; ++row)
#pragma unroll
                    for (int column = 0; column < 4; ++column)
                        sums[row][column] =
                            fmaf(reinterpret_cast<const float*>(&left[row])[element],
                                 reinterpret_cast<const float*>(&right[column])[element], sums[row][column]);
        }
        __syncthreads();
    }
#pragma unroll
    for (int row = 0; row < 4; ++row) {
        const int target_row = row_start + row_lane + row * 4;
#pragma unroll
        for (int column = 0; column < 4; ++column) {
            const int target_column = column_start + column_lane + column * 8;
            if (target_row < rows && target_column < columns)
                output[static_cast<std::int64_t>(target_row) * columns + target_column] = sums[row][column];
        }
    }
}

__global__ void bfloat_mma_gemm(int rows, int columns, int width, const Bfloat* input, const Bfloat* weights,
                                Bfloat* output) {
    __shared__ __align__(16) Bfloat activations[16][16];
    __shared__ __align__(16) Bfloat parameters[16][16];
    const int row_start = static_cast<int>(blockIdx.y) * 16;
    const int column_start = static_cast<int>(blockIdx.x) * 16;
    const int lane = threadIdx.x;
    BfloatC accumulators[2];
    accumulators[0].clear();
    accumulators[1].clear();

    for (int depth = 0; depth < width; depth += 16) {
        for (int index = lane; index < 16 * 16; index += 32) {
            const int row = index / 16;
            const int column = index % 16;
            const int input_row = row_start + row;
            const int input_column = depth + column;
            const int weight_row = column_start + row;
            activations[row][column] =
                input_row < rows && input_column < width ? input[input_row * width + input_column] : __float2bfloat16(0.F);
            parameters[row][column] = weight_row < columns && input_column < width
                                          ? weights[weight_row * width + input_column]
                                          : __float2bfloat16(0.F);
        }
        __syncthreads();

        BfloatA a;
        BfloatA packed_b;
        const int row = lane & 15;
        const int depth_chunk = (lane >> 4) * 8;
        ptx::ldmatrix_x4<false>(a, &activations[row][depth_chunk]);
        const int weight_row = (lane >> 4) * 8 + (lane & 7);
        const int weight_depth_chunk = ((lane >> 3) & 1) * 8;
        ptx::ldmatrix_x4<false>(packed_b, &parameters[weight_row][weight_depth_chunk]);
        const BfloatB b0{{packed_b[0], packed_b[1]}};
        const BfloatB b1{{packed_b[2], packed_b[3]}};
        ptx::mma_bf16_m16n8k16(accumulators[0], a, b0, accumulators[0]);
        ptx::mma_bf16_m16n8k16(accumulators[1], a, b1, accumulators[1]);
        __syncthreads();
    }

    const int lane_row = lane >> 2;
    const int lane_column = (lane & 3) * 2;
    const int row0 = row_start + lane_row;
    const int row1 = row0 + 8;
    const int column0 = column_start + lane_column;
    const int column1 = column0 + 8;
    if (row0 < rows) {
        if (column0 < columns) output[row0 * columns + column0] = __float2bfloat16(accumulators[0][0]);
        if (column0 + 1 < columns) output[row0 * columns + column0 + 1] = __float2bfloat16(accumulators[0][1]);
        if (column1 < columns) output[row0 * columns + column1] = __float2bfloat16(accumulators[1][0]);
        if (column1 + 1 < columns) output[row0 * columns + column1 + 1] = __float2bfloat16(accumulators[1][1]);
    }
    if (row1 < rows) {
        if (column0 < columns) output[row1 * columns + column0] = __float2bfloat16(accumulators[0][2]);
        if (column0 + 1 < columns) output[row1 * columns + column0 + 1] = __float2bfloat16(accumulators[0][3]);
        if (column1 < columns) output[row1 * columns + column1] = __float2bfloat16(accumulators[1][2]);
        if (column1 + 1 < columns) output[row1 * columns + column1 + 1] = __float2bfloat16(accumulators[1][3]);
    }
}

__global__ void scalar_float_gemm(int rows, int columns, int width, const float* input, const float* weights,
                                  float* output) {
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const auto elements = static_cast<std::size_t>(rows) * columns;
    if (index >= elements) return;
    const int row = static_cast<int>(index / columns);
    const int column = static_cast<int>(index % columns);
    float sum = 0.F;
    for (int depth = 0; depth < width; ++depth)
        sum = fmaf(input[static_cast<std::size_t>(row) * width + depth],
                   weights[static_cast<std::size_t>(column) * width + depth], sum);
    output[index] = sum;
}

__global__ void sum_float_partials(const float* workspace, float* output, std::size_t elements, int slices) {
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    float sum = workspace[index];
    for (int slice = 1; slice < slices; ++slice) sum += workspace[slice * elements + index];
    output[index] = sum;
}

} // namespace

template <int TileRows, int TileCols, int MWarps, int NWarps, bool Grouped>
void run_w8a16(dim3 grid, cudaStream_t stream, int rows, int columns, int width, int group_size,
               const __nv_bfloat16* activations, const std::int8_t* parameters, const float* scales, void* target) {
    constexpr int smem = static_cast<int>(sizeof(W8Shared<TileRows, TileCols>));
    auto* kernel = pipelined_w8a16_gemm<TileRows, TileCols, MWarps, NWarps, Grouped>;
    if constexpr (smem >= (48 << 10)) {
        static thread_local std::uint64_t configured = 0;
        int device = 0;
        cudaGetDevice(&device);
        const std::uint64_t bit = device >= 0 && device < 64 ? (std::uint64_t{1} << device) : 0;
        if (!bit || !(configured & bit)) {
            cudaStreamCaptureStatus capturing = cudaStreamCaptureStatusNone;
            if (cudaStreamIsCapturing(stream, &capturing) == cudaSuccess && capturing == cudaStreamCaptureStatusNone) {
                const auto attr = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
                if (attr != cudaSuccess) throw std::runtime_error(cudaGetErrorString(attr));
                configured |= bit;
            }
        }
    }
    kernel<<<grid, static_cast<unsigned int>(MWarps * NWarps * 32), smem, stream>>>(
        rows, columns, width, group_size, activations, parameters, scales, target);
}

auto dense_gemm_slices(int rows, int columns, int width, int multiprocessors, bool bfloat) -> int {
    if (rows <= 1 || rows > 256 || columns <= 0 || columns > 4096 || width < 256) return 1;
    const auto tile_rows = !bfloat ? FloatRows : (rows <= 16 ? 16 : (columns <= 512 ? 32 : 64));
    const auto tile_columns = bfloat ? 64 : FloatColumns;
    const auto tiles = ((rows - 1) / tile_rows + 1) * ((columns - 1) / tile_columns + 1);
    if (tiles >= multiprocessors) return 1;
    const std::size_t workspace_budget = (bfloat && rows > 32 ? 8 : 2) * 1024 * 1024;
    const auto workspace_slices = static_cast<int>(workspace_budget / (rows * columns * sizeof(float)));
    const int max_slices = (bfloat && rows > 16 && columns <= 512) ? 8 : 4;
    return std::max(1,
                    std::min({max_slices, width / 256, (2 * multiprocessors + tiles - 1) / tiles, workspace_slices}));
}

auto w8a16_gemm_slices(int rows, int columns, int width, int group_size, int multiprocessors) -> int {
    if (rows <= 4 || rows > 256 || columns <= 0 || width < 1024 || group_size <= 0 || width % group_size != 0) return 1;
    const int tile_cols = columns >= 4096 ? 128 : 64;
    const int tiles = ((rows - 1) / 64 + 1) * ((columns - 1) / tile_cols + 1);
    if (tiles >= multiprocessors) return 1;
    const std::size_t workspace_budget = (rows > 32 ? 8 : 2) * 1024 * 1024;
    const auto workspace_slices = static_cast<int>(workspace_budget / (rows * columns * sizeof(float)));
    const int max_by_depth = group_size < width ? std::min(8, width / group_size) : std::min(8, width / 256);
    const int target_waves = (columns < 4096 && width >= 4096) ? 4 : 3;
    return std::max(1,
                    std::min({max_by_depth, (target_waves * multiprocessors + tiles - 1) / tiles, workspace_slices}));
}

auto bfloat_gemm(int rows, int columns, int width, const void* input, const void* weights, void* output,
                 cudaStream_t stream, void* workspace, int slices) -> void {
    (void)workspace;
    (void)slices;
    const dim3 grid(static_cast<unsigned int>((columns + 15) / 16),
                    static_cast<unsigned int>((rows + 15) / 16));
    bfloat_mma_gemm<<<grid, 32, 0, stream>>>(rows, columns, width, static_cast<const Bfloat*>(input),
                                             static_cast<const Bfloat*>(weights), static_cast<Bfloat*>(output));
    const auto status = cudaGetLastError();
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("CUDA: BF16 GEMM failed: ") + cudaGetErrorString(status));
}

auto w8a16_gemm(int rows, int columns, int width, int group_size, const void* input, const void* weights,
                const float* scales, void* output, cudaStream_t stream, void* workspace, int slices) -> void {
    const auto* activations = static_cast<const __nv_bfloat16*>(input);
    const auto* parameters = static_cast<const std::int8_t*>(weights);
    const bool grouped = group_size < width;
    const int active_slices = workspace ? std::max(1, slices) : 1;
    const bool wide = active_slices == 1 && columns >= 4096;
    const int tile_cols = wide ? 128 : 64;
    const dim3 grid((columns + tile_cols - 1) / tile_cols, (rows + 63) / 64, active_slices);
    void* target = active_slices > 1 ? workspace : output;
    if (grouped) {
        if (wide)
            run_w8a16<64, 128, 2, 4, true>(grid, stream, rows, columns, width, group_size, activations, parameters,
                                           scales, target);
        else
            run_w8a16<64, 64, 2, 2, true>(grid, stream, rows, columns, width, group_size, activations, parameters,
                                          scales, target);
    } else {
        if (wide)
            run_w8a16<64, 128, 1, 4, false>(grid, stream, rows, columns, width, group_size, activations, parameters,
                                            scales, target);
        else
            run_w8a16<64, 64, 1, 2, false>(grid, stream, rows, columns, width, group_size, activations, parameters,
                                           scales, target);
    }
    if (active_slices > 1) {
        const auto pairs = static_cast<std::size_t>(rows) * (columns / 2);
        reduce_w8a16_partials<<<static_cast<unsigned int>((pairs + 255) / 256), 256, 0, stream>>>(
            static_cast<const float*>(workspace), grouped ? nullptr : scales, static_cast<__nv_bfloat16*>(output), rows,
            columns, active_slices);
    }
    const auto status = cudaGetLastError();
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

auto float_gemm(int rows, int columns, int width, const void* input, const void* weights, void* output,
                cudaStream_t stream, bool pipelined, void* workspace, int slices) -> void {
    if (pipelined && width % 4 == 0 && rows <= 65535 * FloatRows) {
        const dim3 grid((columns - 1) / FloatColumns + 1, (rows - 1) / FloatRows + 1, slices);
        pipelined_float_gemm<<<grid, FloatThreads, 0, stream>>>(rows, columns, width, static_cast<const float*>(input),
                                                                static_cast<const float*>(weights),
                                                                static_cast<float*>(slices > 1 ? workspace : output));
        if (slices > 1) {
            const auto elements = static_cast<std::size_t>(rows) * columns;
            sum_float_partials<<<(elements + 255) / 256, 256, 0, stream>>>(
                static_cast<const float*>(workspace), static_cast<float*>(output), elements, slices);
        }
        const auto status = cudaGetLastError();
        if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
        return;
    }
    (void)workspace;
    (void)slices;
    const auto elements = static_cast<std::size_t>(rows) * columns;
    scalar_float_gemm<<<static_cast<unsigned int>((elements + 255) / 256), 256, 0, stream>>>(
        rows, columns, width, static_cast<const float*>(input), static_cast<const float*>(weights),
        static_cast<float*>(output));
    const auto status = cudaGetLastError();
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("CUDA: FP32 GEMM failed: ") + cudaGetErrorString(status));
}

} // namespace kidi::cuda