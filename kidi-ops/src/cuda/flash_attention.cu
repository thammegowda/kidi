#include "cuda/flash_attention.h"
#include "cuda/flash_attention_device.cuh"
#include "cuda/runtime.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cutlass/arch/memory_sm75.h>
#include <cutlass/arch/memory_sm80.h>
#include <cutlass/arch/mma_sm80.h>
#include <cutlass/array.h>
#include <cutlass/bfloat16.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/layout/matrix.h>

#include <cmath>
#include <cstdint>
#include <type_traits>

namespace kidi::cuda {
namespace {
using namespace flash_detail;

constexpr int kBr = 16;
constexpr int kBc = 32;
constexpr int kWarps = 4;
constexpr int kThreads = kWarps * 32;
constexpr int kSStride = kBc + 4;

__device__ __forceinline__ float read_mask_elem(const void* data, int type, std::size_t idx) {
    switch (type) {
        case 0: // F32
            return static_cast<const float*>(data)[idx];
        case 1: // BF16
            return __bfloat162float(static_cast<const __nv_bfloat16*>(data)[idx]);
        case 2: // F16
            return __half2float(static_cast<const __half*>(data)[idx]);
        default:
            return 0.F;
    }
}

__device__ __forceinline__ int kv_swizzle_chunk(int row, int chunk) { return chunk ^ (row & 7); }

__device__ __forceinline__ int p_swizzle_chunk(int row, int chunk) { return chunk ^ ((row >> 1) & 3); }

template <int HeadDim, bool FloatIO>
__global__ void __launch_bounds__(kThreads)
    flash_attention_bf16_kernel(FlashAttentionArgs args, int g_pack, int q_per_cta, int head_groups, int q_tiles,
                                std::int64_t m_s0, std::int64_t m_s1, std::int64_t m_s2, std::int64_t m_s3) {
    constexpr int DSub = HeadDim / kWarps;
    constexpr int KChunks = HeadDim / 8;
    constexpr int QMmas = DSub / 16;
    constexpr int DNtiles = DSub / 8;
    constexpr int SNtiles = kBc / 8;

    extern __shared__ unsigned char smem_raw[];
    auto* kv_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    auto* s_smem = reinterpret_cast<float*>(kv_smem + kBc * HeadDim);
    auto* p_smem = reinterpret_cast<__nv_bfloat16*>(s_smem + kWarps * kBr * kSStride);
    __shared__ float row_m[kBr];
    __shared__ float row_l[kBr];
    __shared__ float row_alpha[kBr];
    __shared__ int structured_key_begin;
    __shared__ int structured_key_end;
    __shared__ int structured_full_key_begin;
    __shared__ int structured_full_key_end;

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int group = lane >> 2;
    const int tid4 = lane & 3;

    if (tid < kBr) {
        row_m[tid] = -INFINITY;
        row_l[tid] = 0.F;
        row_alpha[tid] = 0.F;
    }

    int idx = blockIdx.x;
    const int q_tile = idx % q_tiles;
    idx /= q_tiles;
    const int head_group = idx % head_groups;
    idx /= head_groups;
    const int key_head = idx % static_cast<int>(args.key_heads);
    const int batch = idx / static_cast<int>(args.key_heads);

    const int g_total = static_cast<int>(args.heads / args.key_heads);
    const int base_h = key_head * g_total + head_group * g_pack;
    const int base_q = q_tile * q_per_cta;

    using IO = std::conditional_t<FloatIO, float, __nv_bfloat16>;
    const auto* q_ptr = static_cast<const IO*>(resolve_view(args.query));
    const auto* k_ptr = static_cast<const __nv_bfloat16*>(resolve_view(args.key));
    const auto* v_ptr = static_cast<const __nv_bfloat16*>(resolve_view(args.value));
    const void* mask_ptr = args.has_mask ? resolve_view(args.mask) : nullptr;
    auto* o_ptr = static_cast<IO*>(resolve_mut(args.output));

    const std::int64_t q_row_stride = args.heads * HeadDim;
    const std::int64_t k_batch = args.key.shape[0] == 1 ? 0 : batch;
    const std::int64_t v_batch = args.value.shape[0] == 1 ? 0 : batch;
    const std::int64_t kv_stride = args.key_heads * HeadDim;
    const auto* k_base = k_ptr + (k_batch * args.key.shape[1] + args.key_offset) * kv_stride +
                         static_cast<std::int64_t>(key_head) * HeadDim;
    const auto* v_base = v_ptr + (v_batch * args.value.shape[1] + args.key_offset) * kv_stride +
                         static_cast<std::int64_t>(key_head) * HeadDim;

    int key_begin = 0;
    int key_end = static_cast<int>(args.keys);
    int full_key_begin = 0;
    int full_key_end = 0;
    const bool structured_bounds =
        args.has_mask && args.mask.type == 3 && m_s1 == 0 && m_s3 == 0 && args.queries > 1 && args.keys > 512;
    if (structured_bounds) {
        if (tid == 0) {
            structured_key_bounds(args, mask_ptr, batch, base_q, q_per_cta, m_s0, m_s2, structured_key_begin,
                                  structured_key_end, structured_full_key_begin, structured_full_key_end);
        }
        __syncthreads();
        key_begin = structured_key_begin;
        key_end = structured_key_end;
        full_key_begin = structured_full_key_begin;
        full_key_end = structured_full_key_end;
    }

    // 1. Async load Q [16 x HeadDim] into kv_smem (first 16 rows), then load q_frag via ldsm.
    constexpr int QVecs = kBr * KChunks;
    for (int v = tid; v < QVecs; v += kThreads) {
        const int r = v / KChunks;
        const int chunk = v % KChunks;
        const int q_idx = base_q + r / g_pack;
        const int h_idx = base_h + r % g_pack;
        const bool valid = q_idx < args.queries;
        const std::int64_t g_off =
            (static_cast<std::int64_t>(batch) * args.queries + (valid ? q_idx : 0)) * q_row_stride +
            static_cast<std::int64_t>(h_idx) * HeadDim + chunk * 8;
        auto* destination = kv_smem + r * HeadDim + kv_swizzle_chunk(r, chunk) * 8;
        if constexpr (FloatIO) {
#pragma unroll
            for (int column = 0; column < 8; ++column)
                destination[column] = __float2bfloat16(valid ? q_ptr[g_off + column] : 0.F);
        } else {
            cutlass::arch::cp_async_zfill<16, cutlass::arch::CacheOperation::Always>(destination, q_ptr + g_off, valid);
        }
    }
    cutlass::arch::cp_async_fence();
    cutlass::arch::cp_async_wait<0>();
    __syncthreads();

    using MmaOp = cutlass::arch::Mma<cutlass::gemm::GemmShape<16, 8, 16>, 32, cutlass::bfloat16_t,
                                     cutlass::layout::RowMajor, cutlass::bfloat16_t, cutlass::layout::ColumnMajor,
                                     float, cutlass::layout::RowMajor, cutlass::arch::OpMultiplyAdd>;
    MmaOp mma;
    using FragmentA = typename MmaOp::FragmentA;
    using FragmentB = typename MmaOp::FragmentB;
    using FragmentC = typename MmaOp::FragmentC;

    FragmentA q_frag[QMmas];
    const int d_start = warp * DSub;
#pragma unroll
    for (int m = 0; m < QMmas; ++m) {
        const int q_row = lane & 15;
        const int q_chunk = (d_start + m * 16) / 8 + ((lane >> 4) & 1);
        cutlass::Array<unsigned, 4> packed;
        cutlass::arch::ldsm<cutlass::layout::RowMajor, 4>(
            packed, kv_smem + q_row * HeadDim + kv_swizzle_chunk(q_row, q_chunk) * 8);
        q_frag[m] = reinterpret_cast<const FragmentA&>(packed);
    }
    __syncthreads();

    FragmentC o_acc[DNtiles];
#pragma unroll
    for (int d = 0; d < DNtiles; ++d) o_acc[d].clear();

    const int keys = static_cast<int>(args.keys);
    constexpr int KVVecs = kBc * KChunks;

    for (int t0 = key_begin; t0 < key_end; t0 += kBc) {
        float mask_vals[4]{};
        const bool fully_valid_tile =
            structured_bounds && t0 >= full_key_begin && t0 + kBc <= full_key_end;
        if (args.has_mask && !fully_valid_tile) {
            bool all_neg_inf = true;
            bool all_very_neg = true;
            const int k_idx = t0 + lane;
#pragma unroll
            for (int rl = 0; rl < 4; ++rl) {
                const int r = warp * 4 + rl;
                const int q_idx = base_q + r / g_pack;
                const int h_idx = base_h + r % g_pack;
                if (q_idx < args.queries && k_idx < keys) {
                    const std::size_t m_off =
                        static_cast<std::size_t>(batch) * m_s0 + static_cast<std::size_t>(h_idx) * m_s1 +
                        static_cast<std::size_t>(q_idx) * m_s2 + static_cast<std::size_t>(k_idx) * m_s3;
                    float mv = read_mask_elem(mask_ptr, args.mask.type, m_off);
                    if (args.mask.type == 3) {
                        const auto query_position =
                            static_cast<const std::int32_t*>(mask_ptr)[m_off];
                        const auto key_position = args.key_offset + k_idx;
                        mv = key_position <= query_position &&
                                     (!args.mask_window || query_position - key_position < args.mask_window)
                                 ? 0.F
                                 : -INFINITY;
                    }
                    mask_vals[rl] = mv;
                    all_neg_inf &= (mv == -INFINITY);
                    all_very_neg &= (mv <= -1e8F);
                }
            }
            const bool warm = (tid >= kBr) || (base_q + tid / g_pack >= args.queries) || (row_m[tid] > -1e4F);
            if (__syncthreads_and(all_neg_inf)) continue;
            if (__syncthreads_and(all_very_neg) && __syncthreads_and(warm)) continue;
        }

        // Load K [kBc x HeadDim] into kv_smem.
        for (int v = tid; v < KVVecs; v += kThreads) {
            const int row = v / KChunks;
            const int chunk = v % KChunks;
            const int k_pos = t0 + row;
            const bool valid = k_pos < keys;
            cutlass::arch::cp_async_zfill<16, cutlass::arch::CacheOperation::Always>(
                kv_smem + row * HeadDim + kv_swizzle_chunk(row, chunk) * 8,
                k_base + static_cast<std::int64_t>(valid ? k_pos : 0) * kv_stride + chunk * 8, valid);
        }
        cutlass::arch::cp_async_fence();
        cutlass::arch::cp_async_wait<0>();
        __syncthreads();

        // Phase 1: S_partial = Q_warp @ K_warp^T.
        FragmentC s_acc[SNtiles];
#pragma unroll
        for (int n = 0; n < SNtiles; ++n) s_acc[n].clear();

#pragma unroll
        for (int m = 0; m < QMmas; ++m) {
#pragma unroll
            for (int np = 0; np < SNtiles / 2; ++np) {
                const int k_row = np * 16 + ((lane >> 4) * 8) + (lane & 7);
                const int k_chunk = (d_start + m * 16) / 8 + ((lane >> 3) & 1);
                cutlass::Array<unsigned, 4> k_packed;
                cutlass::arch::ldsm<cutlass::layout::RowMajor, 4>(
                    k_packed, kv_smem + k_row * HeadDim + kv_swizzle_chunk(k_row, k_chunk) * 8);
                FragmentB b_even, b_odd;
                reinterpret_cast<cutlass::Array<unsigned, 2>&>(b_even)[0] = k_packed[0];
                reinterpret_cast<cutlass::Array<unsigned, 2>&>(b_even)[1] = k_packed[1];
                reinterpret_cast<cutlass::Array<unsigned, 2>&>(b_odd)[0] = k_packed[2];
                reinterpret_cast<cutlass::Array<unsigned, 2>&>(b_odd)[1] = k_packed[3];
                mma(s_acc[2 * np + 0], q_frag[m], b_even, s_acc[2 * np + 0]);
                mma(s_acc[2 * np + 1], q_frag[m], b_odd, s_acc[2 * np + 1]);
            }
        }

        // Write partial S [16 x 32] per warp to s_smem.
        auto* s_warp = s_smem + warp * kBr * kSStride;
#pragma unroll
        for (int n = 0; n < SNtiles; ++n) {
            const int col = n * 8 + tid4 * 2;
            *reinterpret_cast<float2*>(s_warp + group * kSStride + col) = make_float2(s_acc[n][0], s_acc[n][1]);
            *reinterpret_cast<float2*>(s_warp + (group + 8) * kSStride + col) = make_float2(s_acc[n][2], s_acc[n][3]);
        }
        __syncthreads();

        // Start async load of V [kBc x HeadDim] into kv_smem while computing softmax.
        for (int v = tid; v < KVVecs; v += kThreads) {
            const int row = v / KChunks;
            const int chunk = v % KChunks;
            const int k_pos = t0 + row;
            const bool valid = k_pos < keys;
            cutlass::arch::cp_async_zfill<16, cutlass::arch::CacheOperation::Always>(
                kv_smem + row * HeadDim + kv_swizzle_chunk(row, chunk) * 8,
                v_base + static_cast<std::int64_t>(valid ? k_pos : 0) * kv_stride + chunk * 8, valid);
        }
        cutlass::arch::cp_async_fence();

        // Phase 2: Reduce S across 4 warps, apply mask, online softmax, write P_bf16 to p_smem.
        const int k_pos = t0 + lane;
        const bool k_valid = k_pos < keys;
        const int p_chunk = lane >> 3;
        const int p_in = lane & 7;
#pragma unroll
        for (int rl = 0; rl < 4; ++rl) {
            const int r = warp * 4 + rl;
            float score = -INFINITY;
            if (k_valid) {
                score = (s_smem[r * kSStride + lane] + s_smem[(kBr + r) * kSStride + lane] +
                         s_smem[(2 * kBr + r) * kSStride + lane] + s_smem[(3 * kBr + r) * kSStride + lane]) *
                        args.scale;
                if (args.has_mask) score += mask_vals[rl];
            }
            float rmax = score;
#pragma unroll
            for (int delta = 16; delta > 0; delta >>= 1) rmax = fmaxf(rmax, __shfl_xor_sync(0xffffffff, rmax, delta));

            const float m_old = row_m[r];
            const float m_new = fmaxf(m_old, rmax);
            const float alpha = m_old == -INFINITY ? 0.F : __expf(m_old - m_new);
            const float p_val = (m_new == -INFINITY || !k_valid) ? 0.F : __expf(score - m_new);
            float p_sum = p_val;
#pragma unroll
            for (int delta = 16; delta > 0; delta >>= 1) p_sum += __shfl_xor_sync(0xffffffff, p_sum, delta);

            if (lane == 0) {
                row_m[r] = m_new;
                row_l[r] = fmaf(row_l[r], alpha, p_sum);
                row_alpha[r] = alpha;
            }
            p_smem[r * kBc + p_swizzle_chunk(r, p_chunk) * 8 + p_in] = __float2bfloat16(p_val);
        }

        cutlass::arch::cp_async_wait<0>();
        __syncthreads();

        // Phase 3: Rescale O and accumulate O += P @ V.
        const float alpha0 = row_alpha[group];
        const float alpha1 = row_alpha[group + 8];
#pragma unroll
        for (int d = 0; d < DNtiles; ++d) {
            if (alpha0 != 1.F) {
                o_acc[d][0] *= alpha0;
                o_acc[d][1] *= alpha0;
            }
            if (alpha1 != 1.F) {
                o_acc[d][2] *= alpha1;
                o_acc[d][3] *= alpha1;
            }
        }

#pragma unroll
        for (int kp = 0; kp < kBc / 16; ++kp) {
            const int p_row = lane & 15;
            const int p_c = kp * 2 + ((lane >> 4) & 1);
            cutlass::Array<unsigned, 4> p_packed;
            cutlass::arch::ldsm<cutlass::layout::RowMajor, 4>(p_packed,
                                                              p_smem + p_row * kBc + p_swizzle_chunk(p_row, p_c) * 8);
            const auto& p_frag = reinterpret_cast<const FragmentA&>(p_packed);

#pragma unroll
            for (int dp = 0; dp < DNtiles / 2; ++dp) {
                const int v_row = kp * 16 + (lane & 15);
                const int v_chunk = (d_start + dp * 16) / 8 + ((lane >> 4) & 1);
                cutlass::Array<unsigned, 4> v_packed;
                cutlass::arch::ldsm<cutlass::layout::ColumnMajor, 4>(
                    v_packed, kv_smem + v_row * HeadDim + kv_swizzle_chunk(v_row, v_chunk) * 8);
                FragmentB v_even, v_odd;
                reinterpret_cast<cutlass::Array<unsigned, 2>&>(v_even)[0] = v_packed[0];
                reinterpret_cast<cutlass::Array<unsigned, 2>&>(v_even)[1] = v_packed[1];
                reinterpret_cast<cutlass::Array<unsigned, 2>&>(v_odd)[0] = v_packed[2];
                reinterpret_cast<cutlass::Array<unsigned, 2>&>(v_odd)[1] = v_packed[3];
                mma(o_acc[2 * dp + 0], p_frag, v_even, o_acc[2 * dp + 0]);
                mma(o_acc[2 * dp + 1], p_frag, v_odd, o_acc[2 * dp + 1]);
            }
        }
        __syncthreads();
    }

    // Epilogue: normalize by row_l.
    const float l0 = row_l[group];
    const float l1 = row_l[group + 8];
    const float inv_l0 = l0 > 0.F ? 1.F / l0 : 0.F;
    const float inv_l1 = l1 > 0.F ? 1.F / l1 : 0.F;
    const int q0 = base_q + group / g_pack;
    const int h0 = base_h + group % g_pack;
    const int q1 = base_q + (group + 8) / g_pack;
    const int h1 = base_h + (group + 8) % g_pack;
    const bool valid0 = q0 < args.queries;
    const bool valid1 = q1 < args.queries;
    auto* out0 = valid0 ? o_ptr + (static_cast<std::int64_t>(batch) * args.queries + q0) * q_row_stride +
                              static_cast<std::int64_t>(h0) * HeadDim
                        : nullptr;
    auto* out1 = valid1 ? o_ptr + (static_cast<std::int64_t>(batch) * args.queries + q1) * q_row_stride +
                              static_cast<std::int64_t>(h1) * HeadDim
                        : nullptr;
#pragma unroll
    for (int d = 0; d < DNtiles; ++d) {
        const int col = d_start + d * 8 + tid4 * 2;
        if constexpr (FloatIO) {
            if (valid0)
                *reinterpret_cast<float2*>(out0 + col) = make_float2(o_acc[d][0] * inv_l0, o_acc[d][1] * inv_l0);
            if (valid1)
                *reinterpret_cast<float2*>(out1 + col) = make_float2(o_acc[d][2] * inv_l1, o_acc[d][3] * inv_l1);
        } else {
            if (valid0)
                *reinterpret_cast<__nv_bfloat162*>(out0 + col) =
                    __floats2bfloat162_rn(o_acc[d][0] * inv_l0, o_acc[d][1] * inv_l0);
            if (valid1)
                *reinterpret_cast<__nv_bfloat162*>(out1 + col) =
                    __floats2bfloat162_rn(o_acc[d][2] * inv_l1, o_acc[d][3] * inv_l1);
        }
    }
}

// FlashAttention-2 style prefill kernel (Dao, 2023): every warp owns 16 query rows over the full head dimension, so
// Q·Kᵀ, the online softmax and P stay in registers and P feeds the P·V MMA straight from the accumulator layout.
// K/V tiles are double-buffered with cp.async. Wide heads split the output columns over two warp sets, which
// recompute the (cheaper) score tile instead of exchanging it through shared memory.
constexpr int kFaMinRows = 64;

template <int HeadDim>
struct FaConfig {
    static constexpr int DSplit = HeadDim > 256 ? 2 : 1;
    static constexpr int RowWarps = HeadDim == 256 ? 8 : 4;
    static constexpr int Rows = 16 * RowWarps;
    static constexpr int OutCols = HeadDim / DSplit;
    static constexpr int Bc = HeadDim > 128 ? 32 : 64;
    static constexpr int Threads = RowWarps * DSplit * 32;
    static constexpr std::size_t Smem = static_cast<std::size_t>(Rows + 4 * Bc) * HeadDim * sizeof(__nv_bfloat16);
};

__device__ __forceinline__ unsigned pack_bf16(float low, float high) {
    const __nv_bfloat162 packed = __floats2bfloat162_rn(low, high);
    return reinterpret_cast<const unsigned&>(packed);
}

template <int HeadDim, bool FloatIO>
__global__ void __launch_bounds__(FaConfig<HeadDim>::Threads)
    flash_attention_fa2(FlashAttentionArgs args, int g_pack, int q_per_cta, int head_groups, int q_tiles,
                        std::int64_t m_s0, std::int64_t m_s2) {
    using Config = FaConfig<HeadDim>;
    constexpr int Bc = Config::Bc;
    constexpr int KChunks = HeadDim / 8;
    constexpr int STiles = Bc / 8;
    constexpr int OTiles = Config::OutCols / 8;
    constexpr int Threads = Config::Threads;

    extern __shared__ __align__(16) unsigned char smem_raw[];
    auto* q_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    auto* k_smem = q_smem + Config::Rows * HeadDim;
    auto* v_smem = k_smem + 2 * Bc * HeadDim;
    __shared__ int bounds[4];

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int group = lane >> 2;
    const int tid4 = lane & 3;
    const int row_warp = warp % Config::RowWarps;
    const int d_start = (warp / Config::RowWarps) * Config::OutCols;

    int idx = blockIdx.x;
    const int q_tile = q_tiles - 1 - idx % q_tiles; // latest (most expensive causal) tiles first
    idx /= q_tiles;
    const int head_group = idx % head_groups;
    idx /= head_groups;
    const int key_head = idx % static_cast<int>(args.key_heads);
    const int batch = idx / static_cast<int>(args.key_heads);
    const int base_h = key_head * static_cast<int>(args.heads / args.key_heads) + head_group * g_pack;
    const int base_q = q_tile * q_per_cta;

    using IO = std::conditional_t<FloatIO, float, __nv_bfloat16>;
    const auto* q_ptr = static_cast<const IO*>(resolve_view(args.query));
    const auto* positions = args.has_mask ? static_cast<const std::int32_t*>(resolve_view(args.mask)) : nullptr;
    auto* o_ptr = static_cast<IO*>(resolve_mut(args.output));
    const std::int64_t q_row_stride = args.heads * HeadDim;
    const std::int64_t kv_stride = args.key_heads * HeadDim;
    const auto* k_base = static_cast<const __nv_bfloat16*>(resolve_view(args.key)) +
                         ((args.key.shape[0] == 1 ? 0 : batch) * args.key.shape[1] + args.key_offset) * kv_stride +
                         static_cast<std::int64_t>(key_head) * HeadDim;
    const auto* v_base = static_cast<const __nv_bfloat16*>(resolve_view(args.value)) +
                         ((args.value.shape[0] == 1 ? 0 : batch) * args.value.shape[1] + args.key_offset) * kv_stride +
                         static_cast<std::int64_t>(key_head) * HeadDim;
    const int keys = static_cast<int>(args.keys);

    int key_begin = 0, key_end = keys, full_begin = 0, full_end = positions ? 0 : keys;
    if (positions) {
        if (tid == 0)
            structured_key_bounds(args, positions, batch, base_q, q_per_cta, m_s0, m_s2, bounds[0], bounds[1],
                                  bounds[2], bounds[3]);
        __syncthreads();
        key_begin = bounds[0];
        key_end = bounds[1];
        full_begin = bounds[2];
        full_end = bounds[3];
    }

    for (int v = tid; v < Config::Rows * KChunks; v += Threads) {
        const int r = v / KChunks;
        const int chunk = v % KChunks;
        const int q_idx = base_q + r / g_pack;
        const bool valid = q_idx < args.queries && r / g_pack < q_per_cta;
        const std::int64_t offset =
            (static_cast<std::int64_t>(batch) * args.queries + (valid ? q_idx : 0)) * q_row_stride +
            static_cast<std::int64_t>(base_h + r % g_pack) * HeadDim + chunk * 8;
        auto* destination = q_smem + r * HeadDim + kv_swizzle_chunk(r, chunk) * 8;
        if constexpr (FloatIO) {
            const float4 low = valid ? *reinterpret_cast<const float4*>(q_ptr + offset) : make_float4(0, 0, 0, 0);
            const float4 high = valid ? *reinterpret_cast<const float4*>(q_ptr + offset + 4) : make_float4(0, 0, 0, 0);
            *reinterpret_cast<uint4*>(destination) = make_uint4(pack_bf16(low.x, low.y), pack_bf16(low.z, low.w),
                                                                pack_bf16(high.x, high.y), pack_bf16(high.z, high.w));
        } else {
            cutlass::arch::cp_async_zfill<16, cutlass::arch::CacheOperation::Always>(destination, q_ptr + offset,
                                                                                     valid);
        }
    }

    auto load_tile = [&](int buffer, int first_key) {
        for (int v = tid; v < Bc * KChunks; v += Threads) {
            const int row = v / KChunks;
            const int chunk = v % KChunks;
            const int key = first_key + row;
            const bool valid = key < keys;
            const std::int64_t offset = static_cast<std::int64_t>(valid ? key : 0) * kv_stride + chunk * 8;
            const int slot = (buffer * Bc + row) * HeadDim + kv_swizzle_chunk(row, chunk) * 8;
            cutlass::arch::cp_async_zfill<16, cutlass::arch::CacheOperation::Global>(k_smem + slot, k_base + offset,
                                                                                     valid);
            cutlass::arch::cp_async_zfill<16, cutlass::arch::CacheOperation::Global>(v_smem + slot, v_base + offset,
                                                                                     valid);
        }
    };

    // Query positions of the two rows this thread accumulates (row and row + 8 of its warp's 16-row slice).
    int query_position[2]{};
    bool row_valid[2]{};
#pragma unroll
    for (int half = 0; half < 2; ++half) {
        const int r = row_warp * 16 + group + half * 8;
        const int q_idx = base_q + r / g_pack;
        row_valid[half] = q_idx < args.queries && r / g_pack < q_per_cta;
        if (positions && row_valid[half])
            query_position[half] = positions[static_cast<std::size_t>(batch) * m_s0 + static_cast<std::size_t>(q_idx) * m_s2];
    }

    using MmaOp = cutlass::arch::Mma<cutlass::gemm::GemmShape<16, 8, 16>, 32, cutlass::bfloat16_t,
                                     cutlass::layout::RowMajor, cutlass::bfloat16_t, cutlass::layout::ColumnMajor,
                                     float, cutlass::layout::RowMajor, cutlass::arch::OpMultiplyAdd>;
    MmaOp mma;
    using FragmentA = typename MmaOp::FragmentA;
    using FragmentB = typename MmaOp::FragmentB;
    using FragmentC = typename MmaOp::FragmentC;

    FragmentC o_acc[OTiles];
#pragma unroll
    for (int d = 0; d < OTiles; ++d) o_acc[d].clear();
    float row_max[2] = {-INFINITY, -INFINITY};
    float row_sum[2] = {0.F, 0.F};
    const float scale = args.scale * 1.4426950408889634F;

    const int tiles = key_end > key_begin ? (key_end - key_begin + Bc - 1) / Bc : 0;
    if (tiles > 0) load_tile(0, key_begin);
    cutlass::arch::cp_async_fence();
    for (int tile = 0; tile < tiles; ++tile) {
        const int t0 = key_begin + tile * Bc;
        if (tile + 1 < tiles) load_tile((tile + 1) & 1, t0 + Bc);
        cutlass::arch::cp_async_fence();
        cutlass::arch::cp_async_wait<1>();
        __syncthreads();
        const auto* k_tile = k_smem + (tile & 1) * Bc * HeadDim;
        const auto* v_tile = v_smem + (tile & 1) * Bc * HeadDim;

        FragmentC s_acc[STiles];
#pragma unroll
        for (int n = 0; n < STiles; ++n) s_acc[n].clear();
#pragma unroll
        for (int kk = 0; kk < HeadDim / 16; ++kk) {
            const int q_row = row_warp * 16 + (lane & 15);
            cutlass::Array<unsigned, 4> q_packed;
            cutlass::arch::ldsm<cutlass::layout::RowMajor, 4>(
                q_packed, q_smem + q_row * HeadDim + kv_swizzle_chunk(q_row, kk * 2 + (lane >> 4)) * 8);
            const auto& q_frag = reinterpret_cast<const FragmentA&>(q_packed);
#pragma unroll
            for (int np = 0; np < STiles / 2; ++np) {
                const int k_row = np * 16 + ((lane >> 4) * 8) + (lane & 7);
                cutlass::Array<unsigned, 4> k_packed;
                cutlass::arch::ldsm<cutlass::layout::RowMajor, 4>(
                    k_packed, k_tile + k_row * HeadDim + kv_swizzle_chunk(k_row, kk * 2 + ((lane >> 3) & 1)) * 8);
                FragmentB b_even, b_odd;
                reinterpret_cast<cutlass::Array<unsigned, 2>&>(b_even)[0] = k_packed[0];
                reinterpret_cast<cutlass::Array<unsigned, 2>&>(b_even)[1] = k_packed[1];
                reinterpret_cast<cutlass::Array<unsigned, 2>&>(b_odd)[0] = k_packed[2];
                reinterpret_cast<cutlass::Array<unsigned, 2>&>(b_odd)[1] = k_packed[3];
                mma(s_acc[2 * np], q_frag, b_even, s_acc[2 * np]);
                mma(s_acc[2 * np + 1], q_frag, b_odd, s_acc[2 * np + 1]);
            }
        }

        const bool unmasked = t0 >= full_begin && t0 + Bc <= full_end;
        float tile_max[2] = {-INFINITY, -INFINITY};
#pragma unroll
        for (int n = 0; n < STiles; ++n) {
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int half = e >> 1;
                const int key = t0 + n * 8 + tid4 * 2 + (e & 1);
                bool valid = key < keys;
                if (!unmasked && positions) {
                    const std::int64_t key_position = args.key_offset + key;
                    valid = valid && key_position <= query_position[half] &&
                            (!args.mask_window || query_position[half] - key_position < args.mask_window);
                }
                const float score = valid ? s_acc[n][e] * scale : -INFINITY;
                s_acc[n][e] = score;
                tile_max[half] = fmaxf(tile_max[half], score);
            }
        }
        float alpha[2];
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            tile_max[half] = fmaxf(tile_max[half], __shfl_xor_sync(0xffffffff, tile_max[half], 1));
            tile_max[half] = fmaxf(tile_max[half], __shfl_xor_sync(0xffffffff, tile_max[half], 2));
            const float next = fmaxf(row_max[half], tile_max[half]);
            alpha[half] = row_max[half] == -INFINITY ? 0.F : exp2f(row_max[half] - next);
            row_max[half] = next;
            row_sum[half] *= alpha[half];
        }
#pragma unroll
        for (int n = 0; n < STiles; ++n) {
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const float maximum = row_max[e >> 1];
                const float p = maximum == -INFINITY ? 0.F : exp2f(s_acc[n][e] - maximum);
                s_acc[n][e] = p;
                row_sum[e >> 1] += p;
            }
        }
#pragma unroll
        for (int d = 0; d < OTiles; ++d) {
            o_acc[d][0] *= alpha[0];
            o_acc[d][1] *= alpha[0];
            o_acc[d][2] *= alpha[1];
            o_acc[d][3] *= alpha[1];
        }

#pragma unroll
        for (int kp = 0; kp < Bc / 16; ++kp) {
            cutlass::Array<unsigned, 4> p_packed;
            p_packed[0] = pack_bf16(s_acc[2 * kp][0], s_acc[2 * kp][1]);
            p_packed[1] = pack_bf16(s_acc[2 * kp][2], s_acc[2 * kp][3]);
            p_packed[2] = pack_bf16(s_acc[2 * kp + 1][0], s_acc[2 * kp + 1][1]);
            p_packed[3] = pack_bf16(s_acc[2 * kp + 1][2], s_acc[2 * kp + 1][3]);
            const auto& p_frag = reinterpret_cast<const FragmentA&>(p_packed);
#pragma unroll
            for (int dp = 0; dp < OTiles / 2; ++dp) {
                const int v_row = kp * 16 + (lane & 15);
                const int v_chunk = (d_start + dp * 16) / 8 + (lane >> 4);
                cutlass::Array<unsigned, 4> v_packed;
                cutlass::arch::ldsm<cutlass::layout::ColumnMajor, 4>(
                    v_packed, v_tile + v_row * HeadDim + kv_swizzle_chunk(v_row, v_chunk) * 8);
                FragmentB v_even, v_odd;
                reinterpret_cast<cutlass::Array<unsigned, 2>&>(v_even)[0] = v_packed[0];
                reinterpret_cast<cutlass::Array<unsigned, 2>&>(v_even)[1] = v_packed[1];
                reinterpret_cast<cutlass::Array<unsigned, 2>&>(v_odd)[0] = v_packed[2];
                reinterpret_cast<cutlass::Array<unsigned, 2>&>(v_odd)[1] = v_packed[3];
                mma(o_acc[2 * dp], p_frag, v_even, o_acc[2 * dp]);
                mma(o_acc[2 * dp + 1], p_frag, v_odd, o_acc[2 * dp + 1]);
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int half = 0; half < 2; ++half) {
        row_sum[half] += __shfl_xor_sync(0xffffffff, row_sum[half], 1);
        row_sum[half] += __shfl_xor_sync(0xffffffff, row_sum[half], 2);
        if (!row_valid[half]) continue;
        const int r = row_warp * 16 + group + half * 8;
        const float inverse = row_sum[half] > 0.F ? 1.F / row_sum[half] : 0.F;
        auto* out = o_ptr + (static_cast<std::int64_t>(batch) * args.queries + base_q + r / g_pack) * q_row_stride +
                    static_cast<std::int64_t>(base_h + r % g_pack) * HeadDim + d_start;
#pragma unroll
        for (int d = 0; d < OTiles; ++d) {
            const float x = o_acc[d][half * 2] * inverse;
            const float y = o_acc[d][half * 2 + 1] * inverse;
            if constexpr (FloatIO)
                *reinterpret_cast<float2*>(out + d * 8 + tid4 * 2) = make_float2(x, y);
            else
                *reinterpret_cast<__nv_bfloat162*>(out + d * 8 + tid4 * 2) = __floats2bfloat162_rn(x, y);
        }
    }
}

template <int HeadDim>
auto run_fa2(const FlashAttentionArgs& args, std::int64_t m_s0, std::int64_t m_s2, cudaStream_t stream) -> bool {
    using Config = FaConfig<HeadDim>;
    const int g_total = static_cast<int>(args.heads / args.key_heads);
    const int g_pack = g_total <= Config::Rows && Config::Rows % g_total == 0 ? g_total : 1;
    const int q_per_cta = Config::Rows / g_pack;
    const int head_groups = g_total / g_pack;
    const auto q_tiles = (args.queries + q_per_cta - 1) / q_per_cta;
    const auto blocks = static_cast<unsigned int>(args.batches * args.key_heads * head_groups * q_tiles);
    const bool float_io = args.query.type == static_cast<int>(Type::F32);
    auto* kernel = float_io ? flash_attention_fa2<HeadDim, true> : flash_attention_fa2<HeadDim, false>;
    static bool configured[2] = {};
    if (!configured[float_io]) {
        if (cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 static_cast<int>(Config::Smem)) != cudaSuccess)
            return false;
        configured[float_io] = true;
    }
    kernel<<<blocks, Config::Threads, Config::Smem, stream>>>(args, g_pack, q_per_cta, head_groups,
                                                              static_cast<int>(q_tiles), m_s0, m_s2);
    return true;
}

template <int HeadDim>
void run_bf16(const FlashAttentionArgs& args, int g_pack, int q_per_cta, int head_groups, int q_tiles,
              std::int64_t m_s0, std::int64_t m_s1, std::int64_t m_s2, std::int64_t m_s3, cudaStream_t stream) {
    const std::size_t smem =
        (kBc * HeadDim + kBr * kBc) * sizeof(__nv_bfloat16) + kWarps * kBr * kSStride * sizeof(float);
    const unsigned int blocks = static_cast<unsigned int>(args.batches * args.key_heads * head_groups * q_tiles);
    if (args.query.type == static_cast<int>(Type::F32))
        flash_attention_bf16_kernel<HeadDim, true>
            <<<blocks, kThreads, smem, stream>>>(args, g_pack, q_per_cta, head_groups, q_tiles, m_s0, m_s1, m_s2, m_s3);
    else
        flash_attention_bf16_kernel<HeadDim, false>
            <<<blocks, kThreads, smem, stream>>>(args, g_pack, q_per_cta, head_groups, q_tiles, m_s0, m_s1, m_s2, m_s3);
}

} // namespace

auto flash_attention_bf16(const FlashAttentionArgs& args, cudaStream_t stream) -> bool {
    if (args.head_dim != 64 && args.head_dim != 128 && args.head_dim != 256 && args.head_dim != 512) return false;
    if (args.batches <= 0 || args.queries <= 0 || args.heads <= 0 || args.key_heads <= 0 || args.keys <= 0 ||
        args.heads % args.key_heads != 0)
        return false;

    const int g_total = static_cast<int>(args.heads / args.key_heads);
    const int g_pack = (g_total <= kBr && (kBr % g_total) == 0) ? g_total : 1;
    const int q_per_cta = kBr / g_pack;
    const int head_groups = g_total / g_pack;
    const int q_tiles = static_cast<int>((args.queries + q_per_cta - 1) / q_per_cta);

    std::int64_t m_strides[4]{};
    if (args.has_mask) {
        std::int64_t stride = 1;
        for (int axis = 3; axis >= 0; --axis) {
            const int input_axis = axis - 4 + args.mask.rank;
            if (input_axis >= 0) {
                if (args.mask.shape[input_axis] != 1) m_strides[axis] = stride;
                stride *= args.mask.shape[input_axis];
            }
        }
    }

    const int fa_pack = g_total <= kFaMinRows && kFaMinRows % g_total == 0 ? g_total : 1;
    const bool structured = args.has_mask && args.mask.type == 3 && m_strides[1] == 0 && m_strides[3] == 0;
    if ((structured || !args.has_mask) && args.queries * fa_pack >= kFaMinRows) {
        switch (args.head_dim) {
            case 64:
                if (run_fa2<64>(args, m_strides[0], m_strides[2], stream)) return true;
                break;
            case 128:
                if (run_fa2<128>(args, m_strides[0], m_strides[2], stream)) return true;
                break;
            case 256:
                if (run_fa2<256>(args, m_strides[0], m_strides[2], stream)) return true;
                break;
            case 512:
                if (run_fa2<512>(args, m_strides[0], m_strides[2], stream)) return true;
                break;
            default:
                break;
        }
    }

    switch (args.head_dim) {
        case 64:
            run_bf16<64>(args, g_pack, q_per_cta, head_groups, q_tiles, m_strides[0], m_strides[1], m_strides[2],
                         m_strides[3], stream);
            return true;
        case 128:
            run_bf16<128>(args, g_pack, q_per_cta, head_groups, q_tiles, m_strides[0], m_strides[1], m_strides[2],
                          m_strides[3], stream);
            return true;
        case 256:
            run_bf16<256>(args, g_pack, q_per_cta, head_groups, q_tiles, m_strides[0], m_strides[1], m_strides[2],
                          m_strides[3], stream);
            return true;
        case 512:
            run_bf16<512>(args, g_pack, q_per_cta, head_groups, q_tiles, m_strides[0], m_strides[1], m_strides[2],
                          m_strides[3], stream);
            return true;
        default:
            return false;
    }
}

constexpr int kBrF32 = 8;
constexpr int kBcF32 = 16;

template <int Vec>
__device__ __forceinline__ void load_f32_vec(const float* ptr, float (&out)[Vec]) {
    if constexpr (Vec == 2) {
        const float2 v = *reinterpret_cast<const float2*>(ptr);
        out[0] = v.x;
        out[1] = v.y;
    } else {
#pragma unroll
        for (int i = 0; i < Vec / 4; ++i) {
            const float4 v = *reinterpret_cast<const float4*>(ptr + i * 4);
            out[i * 4 + 0] = v.x;
            out[i * 4 + 1] = v.y;
            out[i * 4 + 2] = v.z;
            out[i * 4 + 3] = v.w;
        }
    }
}

template <int Vec>
__device__ __forceinline__ void store_f32_vec(float* ptr, const float (&in)[Vec], float inv_l) {
    if constexpr (Vec == 2) {
        *reinterpret_cast<float2*>(ptr) = make_float2(in[0] * inv_l, in[1] * inv_l);
    } else {
#pragma unroll
        for (int i = 0; i < Vec / 4; ++i) {
            *reinterpret_cast<float4*>(ptr + i * 4) =
                make_float4(in[i * 4 + 0] * inv_l, in[i * 4 + 1] * inv_l, in[i * 4 + 2] * inv_l, in[i * 4 + 3] * inv_l);
        }
    }
}

template <int HeadDim>
__global__ void __launch_bounds__(kThreads, 4)
    flash_attention_f32_kernel(FlashAttentionArgs args, int g_pack, int q_per_cta, int head_groups, int q_tiles,
                               std::int64_t m_s0, std::int64_t m_s1, std::int64_t m_s2, std::int64_t m_s3) {
    constexpr int Vec = HeadDim / 32;
    constexpr int KChunks4 = HeadDim / 4;
    constexpr int KVVecs = kBcF32 * KChunks4;
    constexpr bool DoubleBuffer = HeadDim <= 256;

    extern __shared__ float f32_smem[];
    float* k_smem = f32_smem;
    float* v_smem = DoubleBuffer ? (f32_smem + kBcF32 * HeadDim) : f32_smem;
    __shared__ int structured_key_begin;
    __shared__ int structured_key_end;
    __shared__ int structured_full_key_begin;
    __shared__ int structured_full_key_end;

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    int idx = blockIdx.x;
    const int q_tile = idx % q_tiles;
    idx /= q_tiles;
    const int head_group = idx % head_groups;
    idx /= head_groups;
    const int key_head = idx % static_cast<int>(args.key_heads);
    const int batch = idx / static_cast<int>(args.key_heads);

    const int g_total = static_cast<int>(args.heads / args.key_heads);
    const int base_h = key_head * g_total + head_group * g_pack;
    const int base_q = q_tile * q_per_cta;

    const auto* q_ptr = static_cast<const float*>(resolve_view(args.query));
    const auto* k_ptr = static_cast<const float*>(resolve_view(args.key));
    const auto* v_ptr = static_cast<const float*>(resolve_view(args.value));
    const void* mask_ptr = args.has_mask ? resolve_view(args.mask) : nullptr;
    auto* o_ptr = static_cast<float*>(resolve_mut(args.output));

    const std::int64_t q_row_stride = args.heads * HeadDim;
    const std::int64_t k_batch = args.key.shape[0] == 1 ? 0 : batch;
    const std::int64_t v_batch = args.value.shape[0] == 1 ? 0 : batch;
    const std::int64_t kv_stride = args.key_heads * HeadDim;
    const auto* k_base = k_ptr + (k_batch * args.key.shape[1] + args.key_offset) * kv_stride +
                         static_cast<std::int64_t>(key_head) * HeadDim;
    const auto* v_base = v_ptr + (v_batch * args.value.shape[1] + args.key_offset) * kv_stride +
                         static_cast<std::int64_t>(key_head) * HeadDim;

    int key_begin = 0;
    int key_end = static_cast<int>(args.keys);
    int full_key_begin = 0;
    int full_key_end = 0;
    const bool structured_bounds =
        args.has_mask && args.mask.type == 3 && m_s1 == 0 && m_s3 == 0 && args.queries > 1 && args.keys > 512;
    if (structured_bounds) {
        if (tid == 0) {
            structured_key_bounds(args, mask_ptr, batch, base_q, q_per_cta, m_s0, m_s2, structured_key_begin,
                                  structured_key_end, structured_full_key_begin, structured_full_key_end);
        }
        __syncthreads();
        key_begin = structured_key_begin;
        key_end = structured_key_end;
        full_key_begin = structured_full_key_begin;
        full_key_end = structured_full_key_end;
    }

    float q_reg[2][Vec]{};
    float o_acc[2][Vec]{};
    float m_row[2]{-INFINITY, -INFINITY};
    float l_row[2]{};
    int row_mask = 0;

#pragma unroll
    for (int rl = 0; rl < 2; ++rl) {
        const int r = warp * 2 + rl;
        const int q_idx = base_q + r / g_pack;
        const int h_idx = base_h + r % g_pack;
        if (q_idx < args.queries) {
            row_mask |= (1 << rl);
            const std::int64_t g_off = (static_cast<std::int64_t>(batch) * args.queries + q_idx) * q_row_stride +
                                       static_cast<std::int64_t>(h_idx) * HeadDim + lane * Vec;
            load_f32_vec<Vec>(q_ptr + g_off, q_reg[rl]);
        }
    }

    const int keys = static_cast<int>(args.keys);
    for (int t0 = key_begin; t0 < key_end; t0 += kBcF32) {
        float mask_vals[2]{};
        const bool fully_valid_tile =
            structured_bounds && t0 >= full_key_begin && t0 + kBcF32 <= full_key_end;
        if (args.has_mask && !fully_valid_tile) {
            bool all_neg_inf = true;
            bool all_very_neg = true;
            bool warm = true;
            const int k_idx = t0 + lane;
#pragma unroll
            for (int rl = 0; rl < 2; ++rl) {
                const int r = warp * 2 + rl;
                const int q_idx = base_q + r / g_pack;
                const int h_idx = base_h + r % g_pack;
                if (row_mask & (1 << rl)) {
                    warm &= (m_row[rl] > -1e4F);
                    if (lane < kBcF32 && k_idx < keys) {
                        const std::size_t m_off =
                            static_cast<std::size_t>(batch) * m_s0 + static_cast<std::size_t>(h_idx) * m_s1 +
                            static_cast<std::size_t>(q_idx) * m_s2 + static_cast<std::size_t>(k_idx) * m_s3;
                        float mv = read_mask_elem(mask_ptr, args.mask.type, m_off);
                        if (args.mask.type == 3) {
                            const auto query_position =
                                static_cast<const std::int32_t*>(mask_ptr)[m_off];
                            const auto key_position = args.key_offset + k_idx;
                            mv = key_position <= query_position &&
                                         (!args.mask_window || query_position - key_position < args.mask_window)
                                     ? 0.F
                                     : -INFINITY;
                        }
                        mask_vals[rl] = mv;
                        all_neg_inf &= (mv == -INFINITY);
                        all_very_neg &= (mv <= -1e8F);
                    }
                }
            }
            if (__syncthreads_and(all_neg_inf)) continue;
            if (__syncthreads_and(all_very_neg) && __syncthreads_and(warm)) continue;
        }

        for (int v = tid; v < KVVecs; v += kThreads) {
            const int row = v / KChunks4;
            const int col4 = (v % KChunks4) * 4;
            const int k_pos = t0 + row;
            const bool valid = k_pos < keys;
            cutlass::arch::cp_async_zfill<16, cutlass::arch::CacheOperation::Always>(
                k_smem + row * HeadDim + col4, k_base + static_cast<std::int64_t>(valid ? k_pos : 0) * kv_stride + col4,
                valid);
        }
        cutlass::arch::cp_async_fence();
        if constexpr (DoubleBuffer) {
            for (int v = tid; v < KVVecs; v += kThreads) {
                const int row = v / KChunks4;
                const int col4 = (v % KChunks4) * 4;
                const int k_pos = t0 + row;
                const bool valid = k_pos < keys;
                cutlass::arch::cp_async_zfill<16, cutlass::arch::CacheOperation::Always>(
                    v_smem + row * HeadDim + col4,
                    v_base + static_cast<std::int64_t>(valid ? k_pos : 0) * kv_stride + col4, valid);
            }
            cutlass::arch::cp_async_fence();
            cutlass::arch::cp_async_wait<1>();
        } else {
            cutlass::arch::cp_async_wait<0>();
        }
        __syncthreads();

        float s_val[2]{-INFINITY, -INFINITY};
        for (int k = 0; k < kBcF32; ++k) {
            if (t0 + k >= keys) break;
            float kv[Vec];
            load_f32_vec<Vec>(k_smem + k * HeadDim + lane * Vec, kv);
            float d0 = 0.F, d1 = 0.F;
#pragma unroll
            for (int c = 0; c < Vec; ++c) {
                d0 = fmaf(q_reg[0][c], kv[c], d0);
                d1 = fmaf(q_reg[1][c], kv[c], d1);
            }
#pragma unroll
            for (int delta = 16; delta > 0; delta >>= 1) {
                d0 += __shfl_xor_sync(0xffffffffu, d0, delta);
                d1 += __shfl_xor_sync(0xffffffffu, d1, delta);
            }
            if (lane == k) {
                s_val[0] = d0 * args.scale + (args.has_mask ? mask_vals[0] : 0.F);
                s_val[1] = d1 * args.scale + (args.has_mask ? mask_vals[1] : 0.F);
            }
        }

        if constexpr (!DoubleBuffer) {
            __syncthreads();
            for (int v = tid; v < KVVecs; v += kThreads) {
                const int row = v / KChunks4;
                const int col4 = (v % KChunks4) * 4;
                const int k_pos = t0 + row;
                const bool valid = k_pos < keys;
                cutlass::arch::cp_async_zfill<16, cutlass::arch::CacheOperation::Always>(
                    v_smem + row * HeadDim + col4,
                    v_base + static_cast<std::int64_t>(valid ? k_pos : 0) * kv_stride + col4, valid);
            }
            cutlass::arch::cp_async_fence();
        }

        float p_val[2]{};
#pragma unroll
        for (int rl = 0; rl < 2; ++rl) {
            float rmax = s_val[rl];
#pragma unroll
            for (int delta = 8; delta > 0; delta >>= 1) rmax = fmaxf(rmax, __shfl_xor_sync(0xffffffffu, rmax, delta));
            rmax = __shfl_sync(0xffffffffu, rmax, lane & 15);
            const float m_old = m_row[rl];
            const float m_new = fmaxf(m_old, rmax);
            const float alpha = m_old == -INFINITY ? 0.F : expf(m_old - m_new);
            const float pv =
                (m_new == -INFINITY || lane >= kBcF32 || t0 + lane >= keys) ? 0.F : expf(s_val[rl] - m_new);
            p_val[rl] = pv;
            float p_sum = lane < kBcF32 ? pv : 0.F;
#pragma unroll
            for (int delta = 8; delta > 0; delta >>= 1) p_sum += __shfl_xor_sync(0xffffffffu, p_sum, delta);
            p_sum = __shfl_sync(0xffffffffu, p_sum, lane & 15);
            m_row[rl] = m_new;
            l_row[rl] = fmaf(l_row[rl], alpha, p_sum);
            if (alpha != 1.F) {
#pragma unroll
                for (int c = 0; c < Vec; ++c) o_acc[rl][c] *= alpha;
            }
        }

        cutlass::arch::cp_async_wait<0>();
        __syncthreads();

        for (int k = 0; k < kBcF32; ++k) {
            if (t0 + k >= keys) break;
            const float p0 = __shfl_sync(0xffffffffu, p_val[0], k);
            const float p1 = __shfl_sync(0xffffffffu, p_val[1], k);
            float vv[Vec];
            load_f32_vec<Vec>(v_smem + k * HeadDim + lane * Vec, vv);
#pragma unroll
            for (int c = 0; c < Vec; ++c) {
                o_acc[0][c] = fmaf(p0, vv[c], o_acc[0][c]);
                o_acc[1][c] = fmaf(p1, vv[c], o_acc[1][c]);
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int rl = 0; rl < 2; ++rl) {
        if (!(row_mask & (1 << rl))) continue;
        const int r = warp * 2 + rl;
        const int q_idx = base_q + r / g_pack;
        const int h_idx = base_h + r % g_pack;
        const float inv_l = l_row[rl] > 0.F ? 1.F / l_row[rl] : 0.F;
        const std::int64_t g_off = (static_cast<std::int64_t>(batch) * args.queries + q_idx) * q_row_stride +
                                   static_cast<std::int64_t>(h_idx) * HeadDim + lane * Vec;
        store_f32_vec<Vec>(o_ptr + g_off, o_acc[rl], inv_l);
    }
}

template <int HeadDim>
void run_f32(const FlashAttentionArgs& args, int g_pack, int q_per_cta, int head_groups, int q_tiles, std::int64_t m_s0,
             std::int64_t m_s1, std::int64_t m_s2, std::int64_t m_s3, cudaStream_t stream) {
    constexpr int Buffers = HeadDim <= 256 ? 2 : 1;
    const std::size_t smem = static_cast<std::size_t>(Buffers) * kBcF32 * HeadDim * sizeof(float);
    const unsigned int blocks = static_cast<unsigned int>(args.batches * args.key_heads * head_groups * q_tiles);
    flash_attention_f32_kernel<HeadDim>
        <<<blocks, kThreads, smem, stream>>>(args, g_pack, q_per_cta, head_groups, q_tiles, m_s0, m_s1, m_s2, m_s3);
}

auto flash_attention_f32(const FlashAttentionArgs& args, cudaStream_t stream) -> bool {
    if (args.head_dim != 64 && args.head_dim != 128 && args.head_dim != 256 && args.head_dim != 512) return false;
    if (args.batches <= 0 || args.queries <= 0 || args.heads <= 0 || args.key_heads <= 0 || args.keys <= 0 ||
        args.heads % args.key_heads != 0)
        return false;

    const int g_total = static_cast<int>(args.heads / args.key_heads);
    const int g_pack = (g_total <= kBrF32 && (kBrF32 % g_total) == 0) ? g_total : 1;
    const int q_per_cta = kBrF32 / g_pack;
    const int head_groups = g_total / g_pack;
    const int q_tiles = static_cast<int>((args.queries + q_per_cta - 1) / q_per_cta);

    std::int64_t m_strides[4]{};
    if (args.has_mask) {
        std::int64_t stride = 1;
        for (int axis = 3; axis >= 0; --axis) {
            const int input_axis = axis - 4 + args.mask.rank;
            if (input_axis >= 0) {
                if (args.mask.shape[input_axis] != 1) m_strides[axis] = stride;
                stride *= args.mask.shape[input_axis];
            }
        }
    }

    switch (args.head_dim) {
        case 64:
            run_f32<64>(args, g_pack, q_per_cta, head_groups, q_tiles, m_strides[0], m_strides[1], m_strides[2],
                        m_strides[3], stream);
            return true;
        case 128:
            run_f32<128>(args, g_pack, q_per_cta, head_groups, q_tiles, m_strides[0], m_strides[1], m_strides[2],
                         m_strides[3], stream);
            return true;
        case 256:
            run_f32<256>(args, g_pack, q_per_cta, head_groups, q_tiles, m_strides[0], m_strides[1], m_strides[2],
                         m_strides[3], stream);
            return true;
        case 512:
            run_f32<512>(args, g_pack, q_per_cta, head_groups, q_tiles, m_strides[0], m_strides[1], m_strides[2],
                         m_strides[3], stream);
            return true;
        default:
            return false;
    }
}

} // namespace kidi::cuda
