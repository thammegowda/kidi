#pragma once

#include "cuda/flash_attention.h"

#include <cstddef>
#include <cstdint>

namespace kidi::cuda::flash_detail {

__device__ __forceinline__ const void* resolve_view(const FlashView& v) {
    return v.binding ? static_cast<const unsigned char*>(*v.binding) + v.binding_offset : v.data;
}

__device__ __forceinline__ void* resolve_mut(const FlashView& v) {
    return v.binding ? static_cast<unsigned char*>(*v.binding) + v.binding_offset : const_cast<void*>(v.data);
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
