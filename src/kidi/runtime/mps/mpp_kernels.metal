#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

inline void store_h4(threadgroup half* dst, float4 v) {
    dst[0] = half(v.x);
    dst[1] = half(v.y);
    dst[2] = half(v.z);
    dst[3] = half(v.w);
}

struct RelativeAttentionGeometry {
    uint tokens, heads, key_heads, start, window, extent, width, key_stride, capacity, has_positions;
    float log_alpha, log_floor;
    uint kv_dtype, scale_stride, scale_group, ring_mode;
};

inline void load_kv_32_half(device const uchar* buf, device const float* scales, uint slot, uint col_base, uint stride,
                            uint kv_dtype, uint scale_stride, uint scale_group, threadgroup half* dst) {
    if (kv_dtype == 0u) {
        const device packed_float4* ptr = (const device packed_float4*)(((device const float*)buf) + ulong(slot) * stride + col_base);
#pragma unroll
        for (uint v = 0; v < 8; ++v) store_h4(dst + 4 * v, float4(ptr[v]));
    } else if (kv_dtype == 1u) {
        const device ushort* ptr = ((device const ushort*)buf) + ulong(slot) * stride + col_base;
#pragma unroll
        for (uint v = 0; v < 32; ++v) dst[v] = half(as_type<float>(uint(ptr[v]) << 16));
    } else {
        const device char* ptr = ((device const char*)buf) + ulong(slot) * stride + col_base;
        const float s = scales[ulong(slot) * scale_stride + col_base / scale_group];
#pragma unroll
        for (uint v = 0; v < 32; ++v) dst[v] = half(float(ptr[v]) * s);
    }
}

// Flash-style tiled relative attention (BQ=32, BK=32, D=128) using Metal 4 Neural Accelerators (FP16 tiles,
// FP32 online softmax & accumulation). Supports F32, BF16, and group-32 INT8 KV caches + SWA ring modulo.
kernel void relative_attention_mpp(device const float* query [[buffer(0)]], device const uchar* key [[buffer(1)]],
                                   device const uchar* value [[buffer(2)]], device const float* relative [[buffer(3)]],
                                   device const float* sink [[buffer(4)]], device const float* gate [[buffer(5)]],
                                   device const int* positions [[buffer(6)]], device float* output [[buffer(7)]],
                                   constant RelativeAttentionGeometry& geometry [[buffer(8)]],
                                   device const float* key_scales [[buffer(9)]],
                                   device const float* value_scales [[buffer(10)]],
                                   uint2 group [[threadgroup_position_in_grid]],
                                   uint tid [[thread_index_in_threadgroup]],
                                   uint simd [[simdgroup_index_in_threadgroup]],
                                   uint lane [[thread_index_in_simdgroup]]) {
    constexpr int BQ = 32, BK = 32, D = 128;
    const uint q0 = group.x * BQ, head = group.y;
    const uint m = min(uint(BQ), geometry.tokens - q0);
    const uint key_head = head / (geometry.heads / geometry.key_heads);
    const float q_scale = precise::rsqrt(float(D));

    threadgroup half Qs[BQ * D];       // 8 KB
    threadgroup half KVs[BK * D];      // 8 KB (shared for K then V)
    threadgroup float Scores[BQ * BK]; // 4 KB
    threadgroup half Ps[BQ * BK];      // 2 KB
    threadgroup int q_pos[BQ], q_first[BQ], k_range[2];
    threadgroup float q_tau[BQ], q_peak[BQ], q_total[BQ], q_decay[BQ];

    if (tid < m) {
        const int raw_pos = geometry.has_positions ? positions[q0 + tid] : int(geometry.start + q0 + tid);
        const int pos = geometry.ring_mode ? raw_pos : min(raw_pos, int(geometry.capacity) - 1);
        q_pos[tid] = pos;
        q_first[tid] = geometry.window ? max(0, pos - int(geometry.window) + 1) : 0;
        q_tau[tid] = geometry.log_floor > 0
                         ? 1.0f + geometry.log_alpha * precise::log(max(1.0f, float(pos + 1) / geometry.log_floor))
                         : 1.0f;
        q_peak[tid] = -1e30f;
        q_total[tid] = 0.0f;
        q_decay[tid] = 1.0f;
    } else if (tid < BQ) {
        q_pos[tid] = -1;
        q_first[tid] = 0x7fffffff;
        q_tau[tid] = 1.0f;
        q_peak[tid] = -1e30f;
        q_total[tid] = 0.0f;
        q_decay[tid] = 1.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd == 0) {
        const int mn = simd_min(q_first[lane]);
        const int mx = simd_max(q_pos[lane]);
        if (lane == 0) {
            k_range[0] = mn;
            k_range[1] = mx;
        }
    }

    // Load Q[q0..q0+m-1, head, 0..127] scaled by rsqrt(128): 128 threads x 32 floats = 4096 floats.
    {
        const uint qr = tid >> 2, qc = (tid & 3) * 32;
        if (qr < m) {
            const device packed_float4* qptr =
                (const device packed_float4*)(query + ulong((q0 + qr) * geometry.heads + head) * D + qc);
#pragma unroll
            for (uint v = 0; v < 8; ++v) store_h4(Qs + qr * D + qc + 4 * v, float4(qptr[v]) * q_scale);
        } else {
            threadgroup half* dst = Qs + qr * D + qc;
#pragma unroll
            for (uint v = 0; v < 32; ++v) dst[v] = half(0.0f);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    auto tQ = tensor<threadgroup half, extents<int32_t, D, BQ>, tensor_inline>(Qs, extents<int32_t, D, BQ>());
    auto tKV_k = tensor<threadgroup half, extents<int32_t, D, BK>, tensor_inline>(KVs, extents<int32_t, D, BK>());
    auto tKV_v = tensor<threadgroup half, extents<int32_t, D, BK>, tensor_inline>(KVs, extents<int32_t, D, BK>());
    auto tS = tensor<threadgroup float, extents<int32_t, BK, BQ>, tensor_inline>(Scores, extents<int32_t, BK, BQ>());
    auto tP = tensor<threadgroup half, extents<int32_t, BK, BQ>, tensor_inline>(Ps, extents<int32_t, BK, BQ>());

    constexpr auto desc_qk =
        matmul2d_descriptor(BQ, BK, D, false, true, false, matmul2d_descriptor::mode::multiply);
    constexpr auto desc_pv =
        matmul2d_descriptor(BQ, D, BK, false, false, false, matmul2d_descriptor::mode::multiply);
    matmul2d<desc_qk, execution_simdgroups<4>> op_qk;
    matmul2d<desc_pv, execution_simdgroups<4>> op_pv;

    auto cO = op_pv.template get_destination_cooperative_tensor<decltype(tP), decltype(tKV_v), float>();
    auto cTmp = op_pv.template get_destination_cooperative_tensor<decltype(tP), decltype(tKV_v), float>();
#pragma unroll
    for (uint16_t i = 0; i < cO.get_capacity(); ++i)
        if (cO.is_valid_element(i)) cO[i] = 0.0f;

    const int min_first = k_range[0], max_pos = k_range[1];
    const uint kr = tid >> 2, kc = (tid & 3) * 32;
    const uint kv_col_base = key_head * D + kc;

    for (int k0 = min_first; k0 <= max_pos; k0 += BK) {
        const int k_idx = k0 + int(kr);
        const bool k_valid = k_idx >= 0 && k_idx <= max_pos && (geometry.ring_mode || uint(k_idx) < geometry.capacity);
        if (k_valid) {
            const uint slot = geometry.ring_mode ? (uint(k_idx) % geometry.capacity) : uint(k_idx);
            load_kv_32_half(key, key_scales, slot, kv_col_base, geometry.key_stride, geometry.kv_dtype,
                            geometry.scale_stride, geometry.scale_group, KVs + kr * D + kc);
        } else {
            threadgroup half* dst = KVs + kr * D + kc;
#pragma unroll
            for (uint v = 0; v < 32; ++v) dst[v] = half(0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        op_qk.run(tQ, tKV_k, tS);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (k_valid) {
            const uint slot = geometry.ring_mode ? (uint(k_idx) % geometry.capacity) : uint(k_idx);
            load_kv_32_half(value, value_scales, slot, kv_col_base, geometry.key_stride, geometry.kv_dtype,
                            geometry.scale_stride, geometry.scale_group, KVs + kr * D + kc);
        } else {
            threadgroup half* dst = KVs + kr * D + kc;
#pragma unroll
            for (uint v = 0; v < 32; ++v) dst[v] = half(0.0f);
        }

        const int kl = k0 + int(lane);
        for (uint q = simd * 8; q < simd * 8 + 8; ++q) {
            if (q < m) {
                const int pos = q_pos[q], first = q_first[q];
                const bool valid = kl >= first && kl <= pos;
                float logit = -1e30f;
                if (valid) {
                    logit = Scores[q * BK + lane];
                    const uint dist = uint(pos - kl);
                    if (dist < geometry.extent)
                        logit += relative[ulong((q0 + q) * geometry.heads + head) * geometry.extent + dist];
                    logit *= q_tau[q];
                }
                const float bmax = simd_max(logit);
                const float old_peak = q_peak[q];
                const float next_peak = max(old_peak, bmax);
                const float dec = precise::exp(old_peak - next_peak);
                const float w = valid ? precise::exp(logit - next_peak) : 0.0f;
                const float wsum = simd_sum(w);
                if (lane == 0) {
                    q_peak[q] = next_peak;
                    q_total[q] = q_total[q] * dec + wsum;
                    q_decay[q] = dec;
                }
                Ps[q * BK + lane] = half(w);
            } else {
                Ps[q * BK + lane] = half(0.0f);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        op_pv.run(tP, tKV_v, cTmp);
#pragma unroll
        for (uint16_t i = 0; i < cO.get_capacity(); ++i) {
            if (cO.is_valid_element(i)) {
                const uint q = cO.get_multidimensional_index(i)[1];
                cO[i] = cO[i] * q_decay[q] + cTmp[i];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid < m) {
        const float s_logit = sink[ulong(q0 + tid) * geometry.heads + head];
        const float combined = max(q_peak[tid], s_logit);
        const float scale_o = precise::exp(q_peak[tid] - combined);
        const float denom = q_total[tid] * scale_o + precise::exp(s_logit - combined);
        q_decay[tid] = scale_o / denom;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

#pragma unroll
    for (uint16_t i = 0; i < cO.get_capacity(); ++i) {
        if (cO.is_valid_element(i)) {
            const auto idx = cO.get_multidimensional_index(i);
            const uint col = idx[0], q = idx[1];
            if (q < m) {
                const ulong out_idx = ulong((q0 + q) * geometry.heads + head) * D + col;
                output[out_idx] = cO[i] * q_decay[q] / (1.0f + precise::exp(-gate[out_idx]));
            }
        }
    }
}
