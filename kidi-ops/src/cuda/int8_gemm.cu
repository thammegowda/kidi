#include "cuda/int8_gemm.h"

#include <cuda_bf16.h>
#include <cutlass/arch/memory_sm75.h>
#include <cutlass/arch/memory_sm80.h>
#include <cutlass/arch/mma_sm80.h>
#include <cutlass/array.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/layout/matrix.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <stdexcept>
#include <string>

namespace kidi::cuda {
namespace {

constexpr int kW8A8Depth = 64;
constexpr int kW8A8Stages = 3;
constexpr int kW8A8Stride = 80; // 80 bytes = 20 banks -> bank-conflict-free for 16B cp.async & ldmatrix

template <int TileRows, int TileCols>
struct W8A8Shared {
    __align__(16) std::int8_t activations[kW8A8Stages][TileRows][kW8A8Stride];
    __align__(16) std::int8_t weights[kW8A8Stages][TileCols][kW8A8Stride];
};

template <int TileRows, int Threads>
__device__ __forceinline__ void stage_w8a8_tile(std::int8_t (*shared)[kW8A8Stride], const std::int8_t* src, int rows,
                                                int width, int origin, int depth) {
    constexpr int VecsPerRow = kW8A8Depth / 16;
    constexpr int TotalVecs = TileRows * VecsPerRow;
    for (int vec = threadIdx.x; vec < TotalVecs; vec += Threads) {
        const int row = vec / VecsPerRow;
        const int col = (vec % VecsPerRow) * 16;
        const bool valid = origin + row < rows && depth + col < width;
        const auto* ptr = valid ? src + static_cast<std::int64_t>(origin + row) * width + depth + col : src;
        cutlass::arch::cp_async_zfill<16, cutlass::arch::CacheOperation::Always>(&shared[row][col], ptr, valid);
    }
}

__device__ __forceinline__ float apply_w8a8_epilogue(int acc, int row_zero, int weight_sum, float row_scale,
                                                     float col_scale, const float* bias, int col, float output_scale) {
    float val = static_cast<float>(acc - row_zero * weight_sum) * row_scale * col_scale;
    if (bias) {
        val += bias[col];
    } else if (output_scale > 0.F) {
        val = fminf(127.F, fmaxf(-128.F, nearbyintf(val / output_scale))) * output_scale;
    }
    return val;
}

template <int TileRows, int TileCols, int MWarps, int NWarps>
__global__ void __launch_bounds__(MWarps * NWarps * 32)
    pipelined_w8a8_gemm(int rows, int columns, int width, const std::int8_t* input, const std::int8_t* weights,
                        const float* row_scales, const int* row_zeros, const int* weight_sums, const float* col_scales,
                        const float* bias, float output_scale, void* output, bool bfloat_out) {
    constexpr int Threads = MWarps * NWarps * 32;
    constexpr int WarpRows = TileRows / MWarps;
    constexpr int WarpCols = TileCols / NWarps;
    constexpr int RowMmas = WarpRows / 16;
    constexpr int ColMmas = WarpCols / 8;

    extern __shared__ unsigned char smem_raw[];
    auto& shared = *reinterpret_cast<W8A8Shared<TileRows, TileCols>*>(smem_raw);

    const int row_start = static_cast<int>(blockIdx.y) * TileRows;
    const int col_start = static_cast<int>(blockIdx.x) * TileCols;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int warp_m = warp / NWarps;
    const int warp_n = warp % NWarps;

    const int total_tiles = (width + kW8A8Depth - 1) / kW8A8Depth;
    const int slices = static_cast<int>(gridDim.z);
    const int slice_id = static_cast<int>(blockIdx.z);
    const int tile_begin = slices > 1 ? total_tiles * slice_id / slices : 0;
    const int tile_end = slices > 1 ? total_tiles * (slice_id + 1) / slices : total_tiles;
    const int tiles = tile_end - tile_begin;

    using MmaOp = cutlass::arch::Mma<cutlass::gemm::GemmShape<16, 8, 32>, 32, int8_t, cutlass::layout::RowMajor, int8_t,
                                     cutlass::layout::ColumnMajor, int, cutlass::layout::RowMajor,
                                     cutlass::arch::OpMultiplyAddSaturate>;
    MmaOp mma;
    using FragmentA = typename MmaOp::FragmentA;
    using FragmentB = typename MmaOp::FragmentB;
    using FragmentC = typename MmaOp::FragmentC;

    FragmentC acc[RowMmas][ColMmas];
#pragma unroll
    for (int m = 0; m < RowMmas; ++m)
#pragma unroll
        for (int n = 0; n < ColMmas; ++n) acc[m][n].clear();

    for (int stage = 0; stage < kW8A8Stages - 1; ++stage) {
        if (stage < tiles) {
            const int depth = (tile_begin + stage) * kW8A8Depth;
            stage_w8a8_tile<TileRows, Threads>(shared.activations[stage], input, rows, width, row_start, depth);
            stage_w8a8_tile<TileCols, Threads>(shared.weights[stage], weights, columns, width, col_start, depth);
        }
        cutlass::arch::cp_async_fence();
    }

    for (int tile = 0; tile < tiles; ++tile) {
        const int next = tile + kW8A8Stages - 1;
        const int next_slot = next % kW8A8Stages;
        if (next < tiles) {
            const int depth = (tile_begin + next) * kW8A8Depth;
            stage_w8a8_tile<TileRows, Threads>(shared.activations[next_slot], input, rows, width, row_start, depth);
            stage_w8a8_tile<TileCols, Threads>(shared.weights[next_slot], weights, columns, width, col_start, depth);
        }
        cutlass::arch::cp_async_fence();
        cutlass::arch::cp_async_wait<kW8A8Stages - 1>();
        __syncthreads();

        const int slot = tile % kW8A8Stages;
#pragma unroll
        for (int k_sub = 0; k_sub < kW8A8Depth / 32; ++k_sub) {
            FragmentA a_frags[RowMmas];
#pragma unroll
            for (int m = 0; m < RowMmas; ++m) {
                const int a_row = warp_m * WarpRows + m * 16 + (lane & 15);
                const int a_col = k_sub * 32 + ((lane >> 4) & 1) * 16;
                cutlass::Array<unsigned, 4> packed;
                cutlass::arch::ldsm<cutlass::layout::RowMajor, 4>(packed, &shared.activations[slot][a_row][a_col]);
                a_frags[m] = reinterpret_cast<const FragmentA&>(packed);
            }
#pragma unroll
            for (int np = 0; np < ColMmas / 2; ++np) {
                const int b_row = warp_n * WarpCols + np * 16 + (lane & 15);
                const int b_col = k_sub * 32 + ((lane >> 4) & 1) * 16;
                cutlass::Array<unsigned, 4> packed;
                cutlass::arch::ldsm<cutlass::layout::RowMajor, 4>(packed, &shared.weights[slot][b_row][b_col]);
                FragmentB b_even, b_odd;
                reinterpret_cast<cutlass::Array<unsigned, 2>&>(b_even)[0] = packed[0];
                reinterpret_cast<cutlass::Array<unsigned, 2>&>(b_even)[1] = packed[2];
                reinterpret_cast<cutlass::Array<unsigned, 2>&>(b_odd)[0] = packed[1];
                reinterpret_cast<cutlass::Array<unsigned, 2>&>(b_odd)[1] = packed[3];
#pragma unroll
                for (int m = 0; m < RowMmas; ++m) {
                    mma(acc[m][2 * np + 0], a_frags[m], b_even, acc[m][2 * np + 0]);
                    mma(acc[m][2 * np + 1], a_frags[m], b_odd, acc[m][2 * np + 1]);
                }
            }
        }
        __syncthreads();
    }

    const int lane_row = lane >> 2;
    const int lane_col = (lane & 3) * 2;
    if (slices > 1) {
        auto* partials = static_cast<std::int32_t*>(output) + static_cast<std::size_t>(slice_id) * rows * columns;
#pragma unroll
        for (int m = 0; m < RowMmas; ++m) {
            const int r0 = row_start + warp_m * WarpRows + m * 16 + lane_row;
            const int r1 = r0 + 8;
#pragma unroll
            for (int n = 0; n < ColMmas; ++n) {
                const int c = col_start + warp_n * WarpCols + n * 8 + lane_col;
                if (c + 1 < columns) {
                    if (r0 < rows)
                        *reinterpret_cast<int2*>(partials + static_cast<std::size_t>(r0) * columns + c) =
                            make_int2(acc[m][n][0], acc[m][n][1]);
                    if (r1 < rows)
                        *reinterpret_cast<int2*>(partials + static_cast<std::size_t>(r1) * columns + c) =
                            make_int2(acc[m][n][2], acc[m][n][3]);
                } else if (c < columns) {
                    if (r0 < rows) partials[static_cast<std::size_t>(r0) * columns + c] = acc[m][n][0];
                    if (r1 < rows) partials[static_cast<std::size_t>(r1) * columns + c] = acc[m][n][2];
                }
            }
        }
        return;
    }

#pragma unroll
    for (int m = 0; m < RowMmas; ++m) {
        const int r0 = row_start + warp_m * WarpRows + m * 16 + lane_row;
        const int r1 = r0 + 8;
        const float rs0 = r0 < rows ? row_scales[r0] : 0.F;
        const float rs1 = r1 < rows ? row_scales[r1] : 0.F;
        const int rz0 = r0 < rows ? row_zeros[r0] : 0;
        const int rz1 = r1 < rows ? row_zeros[r1] : 0;
#pragma unroll
        for (int n = 0; n < ColMmas; ++n) {
            const int c = col_start + warp_n * WarpCols + n * 8 + lane_col;
            if (c + 1 < columns) {
                const float cs0 = col_scales[c];
                const float cs1 = col_scales[c + 1];
                const int ws0 = weight_sums[c];
                const int ws1 = weight_sums[c + 1];
                if (r0 < rows) {
                    const float v0 = apply_w8a8_epilogue(acc[m][n][0], rz0, ws0, rs0, cs0, bias, c, output_scale);
                    const float v1 = apply_w8a8_epilogue(acc[m][n][1], rz0, ws1, rs0, cs1, bias, c + 1, output_scale);
                    if (bfloat_out)
                        *reinterpret_cast<__nv_bfloat162*>(static_cast<__nv_bfloat16*>(output) +
                                                           static_cast<std::size_t>(r0) * columns + c) =
                            __floats2bfloat162_rn(v0, v1);
                    else
                        *reinterpret_cast<float2*>(static_cast<float*>(output) +
                                                   static_cast<std::size_t>(r0) * columns + c) = make_float2(v0, v1);
                }
                if (r1 < rows) {
                    const float v2 = apply_w8a8_epilogue(acc[m][n][2], rz1, ws0, rs1, cs0, bias, c, output_scale);
                    const float v3 = apply_w8a8_epilogue(acc[m][n][3], rz1, ws1, rs1, cs1, bias, c + 1, output_scale);
                    if (bfloat_out)
                        *reinterpret_cast<__nv_bfloat162*>(static_cast<__nv_bfloat16*>(output) +
                                                           static_cast<std::size_t>(r1) * columns + c) =
                            __floats2bfloat162_rn(v2, v3);
                    else
                        *reinterpret_cast<float2*>(static_cast<float*>(output) +
                                                   static_cast<std::size_t>(r1) * columns + c) = make_float2(v2, v3);
                }
            } else if (c < columns) {
                const float cs0 = col_scales[c];
                const int ws0 = weight_sums[c];
                if (r0 < rows) {
                    const float v0 = apply_w8a8_epilogue(acc[m][n][0], rz0, ws0, rs0, cs0, bias, c, output_scale);
                    if (bfloat_out)
                        static_cast<__nv_bfloat16*>(output)[static_cast<std::size_t>(r0) * columns + c] =
                            __float2bfloat16(v0);
                    else
                        static_cast<float*>(output)[static_cast<std::size_t>(r0) * columns + c] = v0;
                }
                if (r1 < rows) {
                    const float v2 = apply_w8a8_epilogue(acc[m][n][2], rz1, ws0, rs1, cs0, bias, c, output_scale);
                    if (bfloat_out)
                        static_cast<__nv_bfloat16*>(output)[static_cast<std::size_t>(r1) * columns + c] =
                            __float2bfloat16(v2);
                    else
                        static_cast<float*>(output)[static_cast<std::size_t>(r1) * columns + c] = v2;
                }
            }
        }
    }
}

__global__ void reduce_w8a8_partials(const std::int32_t* workspace, int rows, int columns, int slices,
                                     const float* row_scales, const int* row_zeros, const int* weight_sums,
                                     const float* col_scales, const float* bias, float output_scale, void* output,
                                     bool bfloat_out) {
    const std::size_t elements = static_cast<std::size_t>(rows) * columns;
    const std::size_t idx = (static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x) * 2;
    if (idx >= elements) return;
    if (idx + 1 < elements && (idx % columns) + 1 < static_cast<std::size_t>(columns)) {
        int2 sum = *reinterpret_cast<const int2*>(workspace + idx);
        for (int s = 1; s < slices; ++s) {
            const int2 part = *reinterpret_cast<const int2*>(workspace + static_cast<std::size_t>(s) * elements + idx);
            sum.x += part.x;
            sum.y += part.y;
        }
        const int row = static_cast<int>(idx / columns);
        const int col = static_cast<int>(idx % columns);
        const float rs = row_scales[row];
        const int rz = row_zeros[row];
        const float v0 = apply_w8a8_epilogue(sum.x, rz, weight_sums[col], rs, col_scales[col], bias, col, output_scale);
        const float v1 =
            apply_w8a8_epilogue(sum.y, rz, weight_sums[col + 1], rs, col_scales[col + 1], bias, col + 1, output_scale);
        if (bfloat_out)
            *reinterpret_cast<__nv_bfloat162*>(static_cast<__nv_bfloat16*>(output) + idx) =
                __floats2bfloat162_rn(v0, v1);
        else
            *reinterpret_cast<float2*>(static_cast<float*>(output) + idx) = make_float2(v0, v1);
        return;
    }
    const int row = static_cast<int>(idx / columns);
    const int col = static_cast<int>(idx % columns);
    int sum = workspace[idx];
    for (int s = 1; s < slices; ++s) sum += workspace[static_cast<std::size_t>(s) * elements + idx];
    const float v = apply_w8a8_epilogue(sum, row_zeros[row], weight_sums[col], row_scales[row], col_scales[col], bias,
                                        col, output_scale);
    if (bfloat_out)
        static_cast<__nv_bfloat16*>(output)[idx] = __float2bfloat16(v);
    else
        static_cast<float*>(output)[idx] = v;
}

template <int TileRows, int TileCols, int MWarps, int NWarps>
void run_w8a8(int slices, cudaStream_t stream, int rows, int columns, int width, const std::int8_t* input,
              const std::int8_t* weights, const float* row_scales, const int* row_zeros, const int* weight_sums,
              const float* col_scales, const float* bias, float output_scale, void* target, bool bfloat_out) {
    constexpr int smem = static_cast<int>(sizeof(W8A8Shared<TileRows, TileCols>));
    const dim3 grid(static_cast<unsigned int>((columns - 1) / TileCols + 1),
                    static_cast<unsigned int>((rows - 1) / TileRows + 1), static_cast<unsigned int>(slices));
    if constexpr (smem > 48 * 1024) {
        static thread_local std::uint64_t configured = 0;
        int device = 0;
        cudaGetDevice(&device);
        const auto bit = device >= 0 && device < 64 ? std::uint64_t{1} << device : 0;
        if (!bit || !(configured & bit)) {
            cudaStreamCaptureStatus capturing = cudaStreamCaptureStatusNone;
            if (cudaStreamIsCapturing(stream, &capturing) == cudaSuccess &&
                capturing == cudaStreamCaptureStatusNone) {
                const auto status =
                    cudaFuncSetAttribute(pipelined_w8a8_gemm<TileRows, TileCols, MWarps, NWarps>,
                                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
                if (status != cudaSuccess)
                    throw std::runtime_error(std::string("CUDA: setting W8A8 shared-memory limit failed: ") +
                                             cudaGetErrorString(status));
                configured |= bit;
            }
        }
    }
    pipelined_w8a8_gemm<TileRows, TileCols, MWarps, NWarps>
        <<<grid, static_cast<unsigned int>(MWarps * NWarps * 32), smem, stream>>>(
            rows, columns, width, input, weights, row_scales, row_zeros, weight_sums, col_scales, bias, output_scale,
            target, bfloat_out);
}

#if defined(KIDI_CUDA_HAS_SM90)
constexpr int kSm90MinRows = 128;

auto aligned16(std::initializer_list<const void*> pointers) -> bool {
    for (const auto* pointer : pointers)
        if (reinterpret_cast<std::uintptr_t>(pointer) % 16) return false;
    return true;
}

auto hopper_device() -> bool {
    static int cached[64] = {};
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess || device < 0 || device >= 64) return false;
    if (cached[device] == 0) {
        int major = 0;
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
        cached[device] = major == 9 ? 1 : -1;
    }
    return cached[device] > 0;
}
#endif

} // namespace

auto w8a8_gemm_slices(int rows, int columns, int width, int multiprocessors) -> int {
    if (rows <= 4 || rows > 256 || columns <= 0 || width < 256 || multiprocessors <= 0) return 1;
    const int tile_rows = rows <= 16 ? 16 : 64;
    const int tile_cols = 64;
    const int tiles = ((rows - 1) / tile_rows + 1) * ((columns - 1) / tile_cols + 1);
    if (tiles >= multiprocessors) return 1;
    const int k_tiles = (width + kW8A8Depth - 1) / kW8A8Depth;
    const std::size_t workspace_budget = (rows > 32 ? 8 : 2) * 1024 * 1024;
    const auto element_bytes = static_cast<std::size_t>(rows) * columns * sizeof(std::int32_t);
    const int max_by_workspace = element_bytes ? static_cast<int>(workspace_budget / element_bytes) : 1;
    int target = (multiprocessors - 1) / tiles + 1;
    return std::max(1, std::min({target, k_tiles, max_by_workspace, 16}));
}

auto w8a8_gemm(int rows, int columns, int width, const std::int8_t* input, const std::int8_t* weights,
               const float* row_scales, const int* row_zeros, const int* weight_sums, const float* col_scales,
               const float* bias, float output_scale, void* output, bool bfloat_out, cudaStream_t stream,
               void* workspace, int slices) -> void {
    const int active_slices = slices > 1 && workspace ? slices : 1;
#if defined(KIDI_CUDA_HAS_SM90)
    // Split-K still wins for skinny outputs below kSm90MinRows; above it the Hopper kernel is always faster.
    if (output_scale <= 0.F && (rows >= kSm90MinRows || (rows > 16 && active_slices == 1)) && width % 16 == 0 &&
        columns % 8 == 0 && aligned16({input, weights, output, col_scales, weight_sums, bias}) && hopper_device()) {
        w8a8_gemm_sm90(rows, columns, width, input, weights, row_scales, row_zeros, weight_sums, col_scales, bias,
                       output, bfloat_out, stream);
        return;
    }
#endif
    void* target = active_slices > 1 ? workspace : output;
    if (rows <= 16)
        run_w8a8<16, 64, 1, 4>(active_slices, stream, rows, columns, width, input, weights, row_scales, row_zeros, weight_sums,
                               col_scales, bias, output_scale, target, bfloat_out);
    else if (rows >= 1024)
        run_w8a8<128, 128, 4, 2>(active_slices, stream, rows, columns, width, input, weights, row_scales, row_zeros, weight_sums,
                               col_scales, bias, output_scale, target, bfloat_out);
    else
        run_w8a8<64, 64, 2, 2>(active_slices, stream, rows, columns, width, input, weights, row_scales, row_zeros, weight_sums,
                               col_scales, bias, output_scale, target, bfloat_out);
    if (active_slices > 1) {
        const std::size_t elements = static_cast<std::size_t>(rows) * columns;
        const auto blocks = static_cast<unsigned int>(((elements + 1) / 2 + 255) / 256);
        reduce_w8a8_partials<<<blocks, 256, 0, stream>>>(static_cast<const std::int32_t*>(workspace), rows, columns,
                                                         active_slices, row_scales, row_zeros, weight_sums, col_scales,
                                                         bias, output_scale, output, bfloat_out);
    }
    const auto err = cudaGetLastError();
    if (err != cudaSuccess) throw std::runtime_error(std::string("CUDA: W8A8 GEMM failed: ") + cudaGetErrorString(err));
}

auto int8_gemm(int rows, int columns, int width, const std::int8_t* input, const std::int8_t* weights,
               std::int32_t* output, cudaStream_t stream) -> void {
    using Gemm = cutlass::gemm::device::Gemm<
        std::int8_t, cutlass::layout::RowMajor, std::int8_t, cutlass::layout::ColumnMajor, std::int32_t,
        cutlass::layout::RowMajor, std::int32_t, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<32, 64, 64>, cutlass::gemm::GemmShape<16, 32, 64>, cutlass::gemm::GemmShape<16, 8, 32>,
        cutlass::epilogue::thread::LinearCombination<int32_t, 4, int32_t, int32_t>,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 3, 16, 16>;
    Gemm gemm;
    const typename Gemm::Arguments arguments{{rows, columns, width}, {input, width},    {weights, width},
                                             {output, columns},      {output, columns}, {1, 0}};
    const auto status = gemm(arguments, nullptr, stream);
    if (status != cutlass::Status::kSuccess)
        throw std::runtime_error(std::string("CUDA: CUTLASS INT8 GEMM failed: ") + cutlassGetStatusString(status));
}

} // namespace kidi::cuda