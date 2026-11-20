#include "cuda/dense_gemm.h"

#include <cutlass/arch/memory_sm80.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/device/gemm_splitk_parallel.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace kidi::cuda {
namespace {

template <typename Gemm, typename Element>
void launch(int rows, int columns, int width, const void* input, const void* weights, void* output,
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

constexpr int FloatRows = 32, FloatColumns = 64, FloatDepth = 32, FloatStages = 3, FloatThreads = 128;
using BfloatTile = cutlass::gemm::GemmShape<16, 64, 32>;
using BfloatWarp = cutlass::gemm::GemmShape<16, 32, 32>;
using BfloatInstruction = cutlass::gemm::GemmShape<16, 8, 16>;
using BfloatEpilogue = cutlass::epilogue::thread::LinearCombination<cutlass::bfloat16_t, 8, float, float>;
using Bfloat = cutlass::bfloat16_t;

template <typename Tile, typename Warp>
using BfloatGemm = cutlass::gemm::device::Gemm<Bfloat, cutlass::layout::RowMajor, Bfloat, cutlass::layout::ColumnMajor,
                                               Bfloat, cutlass::layout::RowMajor, float, cutlass::arch::OpClassTensorOp,
                                               cutlass::arch::Sm80, Tile, Warp, BfloatInstruction, BfloatEpilogue,
                                               cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 3, 8, 8>;

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

auto dense_gemm_slices(int rows, int columns, int width, int multiprocessors, bool bfloat) -> int {
    if (rows <= 1 || rows > 128 || columns <= 0 || columns > 4096 || width < 1536) return 1;
    const auto tile_rows = bfloat ? BfloatTile::kM : FloatRows;
    const auto tile_columns = bfloat ? BfloatTile::kN : FloatColumns;
    const auto tiles = ((rows - 1) / tile_rows + 1) * ((columns - 1) / tile_columns + 1);
    if (tiles >= multiprocessors) return 1;
    constexpr auto workspace_budget = 2 * 1024 * 1024;
    const auto workspace_slices = static_cast<int>(workspace_budget / (rows * columns * sizeof(float)));
    return std::max(1, std::min({8, width / 256, (2 * multiprocessors + tiles - 1) / tiles, workspace_slices}));
}

auto launch_bfloat_gemm(int rows, int columns, int width, const void* input, const void* weights, void* output,
                        cudaStream_t stream, void* workspace, int slices) -> void {
    if (slices > 1) {
        using Split =
            cutlass::gemm::device::GemmSplitKParallel<Bfloat, cutlass::layout::RowMajor, Bfloat,
                                                      cutlass::layout::ColumnMajor, Bfloat, cutlass::layout::RowMajor,
                                                      float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
                                                      BfloatTile, BfloatWarp, BfloatInstruction, BfloatEpilogue>;
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
            throw std::runtime_error(std::string("CUDA: CUTLASS BF16 split-K failed: ") +
                                     cutlassGetStatusString(status));
        return;
    }
    if (rows >= 64 && columns >= 4096) {
        using Tall = BfloatGemm<cutlass::gemm::GemmShape<64, 64, 32>, cutlass::gemm::GemmShape<32, 32, 32>>;
        launch<Tall, Bfloat>(rows, columns, width, input, weights, output, stream);
    } else {
        launch<BfloatGemm<BfloatTile, BfloatWarp>, Bfloat>(rows, columns, width, input, weights, output, stream);
    }
}

auto launch_float_gemm(int rows, int columns, int width, const void* input, const void* weights, void* output,
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
    launch<Gemm, float>(rows, columns, width, input, weights, output, stream);
}

} // namespace kidi::cuda