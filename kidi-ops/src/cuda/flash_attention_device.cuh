#pragma once

#include "cuda/flash_attention.h"

#include <cuda_bf16.h>

#include <cstddef>
#include <cstdint>

namespace kidi::cuda::flash_detail {

__device__ __forceinline__ const void* resolve_view(const FlashView& v) {
    return v.binding ? static_cast<const unsigned char*>(*v.binding) + v.binding_offset : v.data;
}

__device__ __forceinline__ void* resolve_mut(const FlashView& v) {
    return v.binding ? static_cast<unsigned char*>(*v.binding) + v.binding_offset : const_cast<void*>(v.data);
}

/// Cache row holding key `key`: the key itself for linear caches, or its ring slot.
__device__ __forceinline__ std::int64_t key_row(const FlashAttentionArgs& args, int key) {
    return args.ring_rows ? key % args.ring_rows : key;
}

/// Writes eight columns of an INT8 cache row to shared memory as BF16. Rows hold `width` INT8 values, and `scales`
/// holds one F32 scale per 32 columns of each row. Invalid rows are written as zeros.
__device__ __forceinline__ void load_int8_chunk(__nv_bfloat16* destination, const signed char* cache,
                                                const float* scales, std::int64_t row, std::int64_t column,
                                                std::int64_t width, bool valid) {
    uint4 packed{};
    if (valid) {
        const uint2 bytes = *reinterpret_cast<const uint2*>(cache + row * width + column);
        const float scale = scales[row * (width / 32) + column / 32];
        const auto* values = reinterpret_cast<const signed char*>(&bytes);
        const auto pack = [&](int index) {
            const __nv_bfloat162 pair = __floats2bfloat162_rn(values[index] * scale, values[index + 1] * scale);
            return reinterpret_cast<const unsigned&>(pair);
        };
        packed = make_uint4(pack(0), pack(2), pack(4), pack(6));
    }
    *reinterpret_cast<uint4*>(destination) = packed;
}

__device__ __forceinline__ void structured_key_bounds(const FlashAttentionArgs& args, const void* positions,
                                                      int batch, int base_q, int q_per_cta, std::int64_t m_s0,
                                                      std::int64_t m_s2, int& key_begin, int& key_end,
                                                      int& full_key_begin, int& full_key_end) {
    bool has_query = false;
    std::int64_t min_query = 0;
    std::int64_t max_query = 0;
    const auto* query_positions = static_cast<const std::int32_t*>(positions);
    int query_end = base_q + q_per_cta;
    if (query_end > args.queries) query_end = static_cast<int>(args.queries);
    for (int q = base_q; q < query_end; ++q) {
        const auto offset = static_cast<std::size_t>(batch) * m_s0 + static_cast<std::size_t>(q) * m_s2;
        const std::int64_t position = query_positions[offset];
        if (!has_query) {
            min_query = position;
            max_query = position;
            has_query = true;
        } else {
            if (position < min_query) min_query = position;
            if (position > max_query) max_query = position;
        }
    }

    if (!has_query) {
        key_begin = 0;
        key_end = 0;
        full_key_begin = 0;
        full_key_end = 0;
        return;
    }

    const std::int64_t lower =
        args.mask_window > 0 ? min_query - args.mask_window + 1 - args.key_offset : 0;
    const std::int64_t full_lower =
        args.mask_window > 0 ? max_query - args.mask_window + 1 - args.key_offset : 0;
    const std::int64_t upper = max_query - args.key_offset + 1;
    const std::int64_t full_upper = min_query - args.key_offset + 1;
    key_begin = lower <= 0 ? 0 : (lower < args.keys ? static_cast<int>(lower) : static_cast<int>(args.keys));
    key_end = upper <= 0 ? 0 : (upper < args.keys ? static_cast<int>(upper) : static_cast<int>(args.keys));
    full_key_begin =
        full_lower <= 0 ? 0 : (full_lower < args.keys ? static_cast<int>(full_lower) : static_cast<int>(args.keys));
    full_key_end =
        full_upper <= 0 ? 0 : (full_upper < args.keys ? static_cast<int>(full_upper) : static_cast<int>(args.keys));
    if (key_end < key_begin) key_end = key_begin;
}

} // namespace kidi::cuda::flash_detail
