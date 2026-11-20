#include "cuda/dense_gemm.h"

#include <cuda_bf16.h>
#include <cutlass/arch/memory_sm75.h>
#include <cutlass/arch/memory_sm80.h>
#include <cutlass/arch/mma_sm80.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/device/gemm_splitk_parallel.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace kidi::cuda {
namespace {

template <typename Gemm, typename Element>
void run_gemm(int rows, int columns, int width, const void* input, const void* weights, void* output,
              cudaStream_t stream) {
    auto* result = static_cast<Element*>(output);
    const typename Gemm::Arguments arguments{{rows, columns, width},
                                             {static_cast<const Element*>(input), width},
                                             {static_cast<const Element*>(weights), width},
                                             {result, columns},
                                             {result, columns},
                                             {1, 0}};
    const auto status = Gemm{}(arguments, nullptr, stream);
    if (status != cutlass::Status::kSuccess)
        throw std::runtime_error(std::string("CUDA: CUTLASS dense GEMM failed: ") + cutlassGetStatusString(status));
}

template <typename Split>
void split_gemm(int rows, int columns, int width, const void* input, const void* weights, void* output,
                cudaStream_t stream, void* workspace, int slices) {
    using Bfloat = cutlass::bfloat16_t;
    auto* result = static_cast<Bfloat*>(output);
    const typename Split::Arguments arguments{{rows, columns, width},
                                              {static_cast<const Bfloat*>(input), width},
                                              {static_cast<const Bfloat*>(weights), width},
                                              {result, columns},
                                              {result, columns},
                                              {1, 0},
                                              slices};
    const auto status = Split{}(arguments, workspace, stream);
    if (status != cutlass::Status::kSuccess)
        throw std::runtime_error(std::string("CUDA: CUTLASS BF16 split-K failed: ") + cutlassGetStatusString(status));
}

constexpr int FloatRows = 32, FloatColumns = 64, FloatDepth = 32, FloatStages = 3, FloatThreads = 128;
using Bfloat16x64 = cutlass::gemm::GemmShape<16, 64, 32>;
using BfloatWarp16x64 = cutlass::gemm::GemmShape<16, 32, 32>;
using Bfloat32x64 = cutlass::gemm::GemmShape<32, 64, 64>;
using BfloatWarp32x64 = cutlass::gemm::GemmShape<16, 32, 64>;
using Bfloat64x64 = cutlass::gemm::GemmShape<64, 64, 64>;
using BfloatWarp64x64 = cutlass::gemm::GemmShape<32, 32, 64>;
using Bfloat64x128 = cutlass::gemm::GemmShape<64, 128, 64>;
using BfloatWarp64x128 = cutlass::gemm::GemmShape<32, 64, 64>;
using BfloatInstruction = cutlass::gemm::GemmShape<16, 8, 16>;
using BfloatEpilogue = cutlass::epilogue::thread::LinearCombination<cutlass::bfloat16_t, 8, float, float>;
using Bfloat = cutlass::bfloat16_t;

template <typename Tile, typename Warp, int Stages = 3>
using BfloatGemm =
    cutlass::gemm::device::Gemm<Bfloat, cutlass::layout::RowMajor, Bfloat, cutlass::layout::ColumnMajor, Bfloat,
                                cutlass::layout::RowMajor, float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
                                Tile, Warp, BfloatInstruction, BfloatEpilogue,
                                cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>, Stages, 8, 8>;

template <typename Tile, typename Warp, int Stages = 3>
using BfloatSplitGemm = cutlass::gemm::device::GemmSplitKParallel<
    Bfloat, cutlass::layout::RowMajor, Bfloat, cutlass::layout::ColumnMajor, Bfloat, cutlass::layout::RowMajor, float,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, Tile, Warp, BfloatInstruction, BfloatEpilogue,
    cutlass::epilogue::thread::Convert<float, 8, float>, cutlass::reduction::thread::ReduceAdd<float, float, 8>,
    cutlass::gemm::threadblock::GemmSplitKHorizontalThreadblockSwizzle, Stages, 8, 8>;

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
        cutlass::arch::cp_async_zfill<16, cutlass::arch::CacheOperation::Always>(&shared[row][(chunk ^ (row & 3)) * 8],
                                                                                 source, valid);
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
        cutlass::arch::cp_async_zfill<16, cutlass::arch::CacheOperation::Global>(&shared[column][chunk * 16], source,
                                                                                 valid);
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

    using Mma = cutlass::arch::Mma<BfloatInstruction, 32, Bfloat, cutlass::layout::RowMajor, Bfloat,
                                   cutlass::layout::ColumnMajor, float, cutlass::layout::RowMajor,
                                   cutlass::arch::OpMultiplyAdd>;
    typename Mma::FragmentC accumulators[RowMmas][ColMmas];
    typename Mma::FragmentC totals[Grouped ? RowMmas : 1][Grouped ? ColMmas : 1];
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
        cutlass::arch::cp_async_fence();
    }

    Mma mma;
    const int sub = lane >> 3, sub_row = (sub & 1) * 8 + (lane & 7), sub_col = sub >> 1;
    for (int tile = 0; tile < tiles; ++tile) {
        const int next = tile + W8Stages - 1, next_slot = next % W8Stages;
        if (next < tiles) {
            stage_w8a16_activations<TileRows, Threads>(shared.activations[next_slot], input, rows, width, row_start,
                                                       (start + next) * W8Depth);
            stage_w8a16_weights<TileCols, Threads>(shared.parameters[next_slot], weights, columns, width, col_start,
                                                   (start + next) * W8Depth);
        }
        cutlass::arch::cp_async_fence();
        cutlass::arch::cp_async_wait<W8Stages - 1>();
        __syncthreads();
        const int slot = tile % W8Stages;

#pragma unroll
        for (int step = 0; step < 4; ++step) {
            typename Mma::FragmentA frag_a[RowMmas];
#pragma unroll
            for (int mi = 0; mi < RowMmas; ++mi) {
                const int r = warp_m * WarpRows + mi * 16 + sub_row;
                const int c8 = (step * 2 + sub_col) ^ (r & 3);
                cutlass::arch::ldsm<cutlass::layout::RowMajor, 4>(
                    reinterpret_cast<cutlass::Array<unsigned, 4>&>(frag_a[mi]), &shared.activations[slot][r][c8 * 8]);
            }
            typename Mma::FragmentB frag_b[ColMmas];
#pragma unroll
            for (int ni = 0; ni < ColMmas; ++ni) {
                const int n = warp_n * (TileCols / NWarps) + ni * 8 + group_lane;
                const auto* row_b = &shared.parameters[slot][n][step * 16];
                auto* packed = reinterpret_cast<std::uint32_t*>(&frag_b[ni]);
                packed[0] = convert_int8_pair_to_bfloat2(*reinterpret_cast<const std::uint16_t*>(row_b + tid * 2));
                packed[1] = convert_int8_pair_to_bfloat2(*reinterpret_cast<const std::uint16_t*>(row_b + 8 + tid * 2));
            }
#pragma unroll
            for (int mi = 0; mi < RowMmas; ++mi)
#pragma unroll
                for (int ni = 0; ni < ColMmas; ++ni)
                    mma(accumulators[mi][ni], frag_a[mi], frag_b[ni], accumulators[mi][ni]);
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
        cutlass::arch::cp_async_zfill<16, cutlass::arch::CacheOperation::Global>(&shared[row][column ^ ((row & 7) * 4)],
                                                                                 source, valid);
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
        cutlass::arch::cp_async_fence();
    }
    for (int tile = 0; tile < tiles; ++tile) {
        const int next = tile + FloatStages - 1, next_slot = next % FloatStages;
        if (next < tiles) {
            stage_float_tile<FloatRows>(activations[next_slot], input, rows, width, row_start,
                                        (start + next) * FloatDepth);
            stage_float_tile<FloatColumns>(parameters[next_slot], weights, columns, width, column_start,
                                           (start + next) * FloatDepth);
        }
        cutlass::arch::cp_async_fence();
        cutlass::arch::cp_async_wait<FloatStages - 1>();
        __syncthreads();
        const int slot = tile % FloatStages;
#pragma unroll
        for (int depth = 0; depth < FloatDepth; depth += 4) {
            using Pack = cutlass::AlignedArray<float, 4>;
            Pack left[4], right[4];
#pragma unroll
            for (int value = 0; value < 4; ++value) {
                const int row = row_lane + value * 4, column = column_lane + value * 8;
                left[value] = *reinterpret_cast<const Pack*>(&activations[slot][row][depth ^ ((row & 7) * 4)]);
                right[value] = *reinterpret_cast<const Pack*>(&parameters[slot][column][depth ^ ((column & 7) * 4)]);
            }
#pragma unroll
            for (int element = 0; element < 4; ++element)
#pragma unroll
                for (int row = 0; row < 4; ++row)
#pragma unroll
                    for (int column = 0; column < 4; ++column)
                        sums[row][column] = fmaf(left[row][element], right[column][element], sums[row][column]);
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
    if (rows > 32 && columns >= 4096) {
        run_gemm<BfloatGemm<Bfloat64x128, BfloatWarp64x128, 3>, Bfloat>(rows, columns, width, input, weights, output,
                                                                        stream);
        return;
    }
    if (rows > 16 && columns <= 512) {
        if (slices > 1) {
            split_gemm<BfloatSplitGemm<Bfloat32x64, BfloatWarp32x64, 3>>(rows, columns, width, input, weights, output,
                                                                         stream, workspace, slices);
        } else {
            run_gemm<BfloatGemm<Bfloat32x64, BfloatWarp32x64, 3>, Bfloat>(rows, columns, width, input, weights, output,
                                                                          stream);
        }
        return;
    }
    if (rows > 16) {
        if (slices > 1) {
            split_gemm<BfloatSplitGemm<Bfloat64x64, BfloatWarp64x64, 3>>(rows, columns, width, input, weights, output,
                                                                         stream, workspace, slices);
        } else {
            run_gemm<BfloatGemm<Bfloat64x64, BfloatWarp64x64, 3>, Bfloat>(rows, columns, width, input, weights, output,
                                                                          stream);
        }
        return;
    }
    if (slices > 1) {
        split_gemm<BfloatSplitGemm<Bfloat16x64, BfloatWarp16x64, 4>>(rows, columns, width, input, weights, output,
                                                                     stream, workspace, slices);
        return;
    }
    run_gemm<BfloatGemm<Bfloat16x64, BfloatWarp16x64, 4>, Bfloat>(rows, columns, width, input, weights, output, stream);
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
    using Gemm = cutlass::gemm::device::Gemm<float, cutlass::layout::RowMajor, float, cutlass::layout::ColumnMajor,
                                             float, cutlass::layout::RowMajor, float, cutlass::arch::OpClassSimt,
                                             cutlass::arch::Sm50, cutlass::gemm::GemmShape<32, 32, 8>,
                                             cutlass::gemm::GemmShape<16, 16, 8>, cutlass::gemm::GemmShape<1, 1, 1>>;
    run_gemm<Gemm, float>(rows, columns, width, input, weights, output, stream);
}

} // namespace kidi::cuda