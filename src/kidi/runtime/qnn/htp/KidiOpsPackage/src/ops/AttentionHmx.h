#pragma once

#ifdef __hexagon__

#include "AttentionHvx.h"

#include <hmx_hexagon_protos.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace kidi::qnn::htp {

inline constexpr std::size_t HMX_TILE = 32;
inline constexpr std::size_t HMX_TILE_ELEMENTS = HMX_TILE * HMX_TILE;
inline constexpr std::size_t HMX_SCALE_ELEMENTS = 128;

inline constexpr auto hmx_tile_index(std::size_t row, std::size_t column) -> std::size_t {
    return (row / 2) * 64 + column * 2 + row % 2;
}

inline constexpr auto hmx_attention_scratch_elements(std::size_t query_length, std::size_t head_dim) -> std::size_t {
    const auto padded_query = (query_length + HMX_TILE - 1) / HMX_TILE * HMX_TILE;
    return 4 * padded_query * head_dim + padded_query * padded_query + HMX_SCALE_ELEMENTS;
}

inline void hmx_initialize_scales(__fp16* scales, float scale) {
    std::fill_n(scales, HMX_SCALE_ELEMENTS, static_cast<__fp16>(0.0f));
    for (std::size_t column = 0; column < HMX_TILE; ++column) scales[column * 2] = static_cast<__fp16>(scale);
}

__attribute__((noinline)) inline void hmx_accumulate_tile(const __fp16* left, const __fp16* right) {
    Q6_activation_hf_mxmem_RR(reinterpret_cast<std::uintptr_t>(left), 2047);
    Q6_weight_hf_mxmem_RR(reinterpret_cast<std::uintptr_t>(right), 2047);
}

inline void hmx_pack_query(__fp16* packed, const Float16* query, std::size_t query_length, std::size_t query_heads,
                           std::size_t head, std::size_t head_dim) {
    const std::size_t query_tiles = (query_length + HMX_TILE - 1) / HMX_TILE;
    const std::size_t depth_tiles = head_dim / HMX_TILE;
    std::memset(packed, 0, query_tiles * depth_tiles * HMX_TILE_ELEMENTS * sizeof(__fp16));
    for (std::size_t row = 0; row < query_length; ++row)
        for (std::size_t depth = 0; depth < head_dim; ++depth) {
            const std::size_t query_tile = row / HMX_TILE;
            const std::size_t depth_tile = depth / HMX_TILE;
            auto* tile = packed + (query_tile * depth_tiles + depth_tile) * HMX_TILE_ELEMENTS;
            tile[hmx_tile_index(row % HMX_TILE, depth % HMX_TILE)] =
                static_cast<__fp16>(static_cast<float>(query[(row * query_heads + head) * head_dim + depth]));
        }
}

inline void hmx_pack_key(__fp16* packed, const Float16* key, std::size_t query_length, std::size_t head_dim) {
    const std::size_t query_tiles = (query_length + HMX_TILE - 1) / HMX_TILE;
    const std::size_t depth_tiles = head_dim / HMX_TILE;
    std::memset(packed, 0, query_tiles * depth_tiles * HMX_TILE_ELEMENTS * sizeof(__fp16));
    for (std::size_t token = 0; token < query_length; ++token)
        for (std::size_t depth = 0; depth < head_dim; ++depth) {
            const std::size_t token_tile = token / HMX_TILE;
            const std::size_t depth_tile = depth / HMX_TILE;
            auto* tile = packed + (token_tile * depth_tiles + depth_tile) * HMX_TILE_ELEMENTS;
            tile[hmx_tile_index(depth % HMX_TILE, token % HMX_TILE)] =
                static_cast<__fp16>(static_cast<float>(key[token * head_dim + depth]));
        }
}

inline void hmx_pack_value(__fp16* packed, const Float16* value, std::size_t query_length, std::size_t head_dim) {
    const std::size_t query_tiles = (query_length + HMX_TILE - 1) / HMX_TILE;
    const std::size_t depth_tiles = head_dim / HMX_TILE;
    std::memset(packed, 0, query_tiles * depth_tiles * HMX_TILE_ELEMENTS * sizeof(__fp16));
    for (std::size_t token = 0; token < query_length; ++token)
        for (std::size_t depth = 0; depth < head_dim; ++depth) {
            const std::size_t token_tile = token / HMX_TILE;
            const std::size_t depth_tile = depth / HMX_TILE;
            auto* tile = packed + (depth_tile * query_tiles + token_tile) * HMX_TILE_ELEMENTS;
            tile[hmx_tile_index(token % HMX_TILE, depth % HMX_TILE)] =
                static_cast<__fp16>(static_cast<float>(value[token * head_dim + depth]));
        }
}

inline void hmx_unpack_output(Float16* output, const __fp16* packed, std::size_t query_length, std::size_t query_heads,
                              std::size_t head, std::size_t head_dim) {
    const std::size_t query_tiles = (query_length + HMX_TILE - 1) / HMX_TILE;
    for (std::size_t row = 0; row < query_length; ++row) {
        auto* output_row = output + (row * query_heads + head) * head_dim;
        const std::size_t query_tile = row / HMX_TILE;
        const std::size_t tile_vector = (row % HMX_TILE) / 2;
        for (std::size_t depth = 0; depth < head_dim; depth += 64) {
            const auto* tile0 = packed + ((depth / HMX_TILE) * query_tiles + query_tile) * HMX_TILE_ELEMENTS;
            const auto* tile1 = tile0 + query_tiles * HMX_TILE_ELEMENTS;
            const auto unpacked = Q6_W_vdeal_VVR(reinterpret_cast<const HVX_Vector*>(tile1)[tile_vector],
                                                 reinterpret_cast<const HVX_Vector*>(tile0)[tile_vector], -2);
            q6op_vstu_AV(output_row + depth, row % 2 == 0 ? Q6_V_lo_W(unpacked) : Q6_V_hi_W(unpacked));
        }
    }
}

inline void hmx_softmax(__fp16* scores, const std::int32_t* positions, std::size_t query_length, std::int32_t window) {
    const std::size_t query_tiles = (query_length + HMX_TILE - 1) / HMX_TILE;
    for (std::size_t row = 0; row < query_length; ++row) {
        const std::int32_t end = std::min<std::int32_t>(positions[row] + 1, static_cast<std::int32_t>(query_length));
        const std::int32_t start = window > 0 ? std::max<std::int32_t>(0, end - window) : 0;
        float maximum = -1.0e30f;
        for (std::int32_t token = start; token < end; ++token) {
            const auto* tile = scores + ((row / HMX_TILE) * query_tiles + token / HMX_TILE) * HMX_TILE_ELEMENTS;
            maximum = std::max(maximum, static_cast<float>(tile[hmx_tile_index(row % HMX_TILE, token % HMX_TILE)]));
        }
        float denominator = 0.0f;
        for (std::int32_t token = start; token < end; ++token) {
            auto* tile = scores + ((row / HMX_TILE) * query_tiles + token / HMX_TILE) * HMX_TILE_ELEMENTS;
            auto& element = tile[hmx_tile_index(row % HMX_TILE, token % HMX_TILE)];
            const float probability = fast_exp_negative(static_cast<float>(element) - maximum);
            element = static_cast<__fp16>(probability);
            denominator += probability;
        }
        const float inverse = denominator > 0.0f ? 1.0f / denominator : 0.0f;
        for (std::size_t token = 0; token < query_tiles * HMX_TILE; ++token) {
            auto* tile = scores + ((row / HMX_TILE) * query_tiles + token / HMX_TILE) * HMX_TILE_ELEMENTS;
            auto& element = tile[hmx_tile_index(row % HMX_TILE, token % HMX_TILE)];
            element = token >= static_cast<std::size_t>(start) && token < static_cast<std::size_t>(end)
                          ? static_cast<__fp16>(static_cast<float>(element) * inverse)
                          : static_cast<__fp16>(0.0f);
        }
    }
}

inline void structured_attention_hmx(Float16* output, Float16* scratch, const Float16* query,
                                     const Float16* current_key, const Float16* current_value,
                                     const std::int32_t* positions, float attention_scale, std::int32_t window,
                                     std::size_t query_length, std::size_t query_heads, std::size_t head_dim,
                                     std::size_t first_head, std::size_t head_count) {
    const std::size_t padded_query = (query_length + HMX_TILE - 1) / HMX_TILE * HMX_TILE;
    const std::size_t query_tiles = padded_query / HMX_TILE;
    const std::size_t depth_tiles = head_dim / HMX_TILE;
    const std::size_t matrix_elements = padded_query * head_dim;
    auto* query_tiles_buffer = reinterpret_cast<__fp16*>(scratch);
    auto* key_tiles_buffer = query_tiles_buffer + matrix_elements;
    auto* value_tiles_buffer = key_tiles_buffer + matrix_elements;
    auto* score_tiles_buffer = value_tiles_buffer + matrix_elements;
    auto* output_tiles_buffer = score_tiles_buffer + padded_query * padded_query;
    auto* scales = output_tiles_buffer + matrix_elements;

    hmx_pack_key(key_tiles_buffer, current_key, query_length, head_dim);
    hmx_pack_value(value_tiles_buffer, current_value, query_length, head_dim);
    for (std::size_t head = first_head; head < first_head + head_count; ++head) {
        hmx_pack_query(query_tiles_buffer, query, query_length, query_heads, head, head_dim);
        hmx_initialize_scales(scales, attention_scale);
        Q6_bias_mxmem2_A(scales);
        for (std::size_t query_tile = 0; query_tile < query_tiles; ++query_tile)
            for (std::size_t key_tile = 0; key_tile < query_tiles; ++key_tile) {
                Q6_mxclracc_hf();
                for (std::size_t depth_tile = 0; depth_tile < depth_tiles; ++depth_tile)
                    hmx_accumulate_tile(
                        query_tiles_buffer + (query_tile * depth_tiles + depth_tile) * HMX_TILE_ELEMENTS,
                        key_tiles_buffer + (key_tile * depth_tiles + depth_tile) * HMX_TILE_ELEMENTS);
                Q6_mxmem_AR_after_hf(score_tiles_buffer + (query_tile * query_tiles + key_tile) * HMX_TILE_ELEMENTS, 0);
            }

        hmx_softmax(score_tiles_buffer, positions, query_length, window);
        hmx_initialize_scales(scales, 1.0f);
        Q6_bias_mxmem2_A(scales);
        for (std::size_t query_tile = 0; query_tile < query_tiles; ++query_tile)
            for (std::size_t depth_tile = 0; depth_tile < depth_tiles; ++depth_tile) {
                Q6_mxclracc_hf();
                for (std::size_t key_tile = 0; key_tile < query_tiles; ++key_tile)
                    hmx_accumulate_tile(score_tiles_buffer + (query_tile * query_tiles + key_tile) * HMX_TILE_ELEMENTS,
                                        value_tiles_buffer + (depth_tile * query_tiles + key_tile) * HMX_TILE_ELEMENTS);
                Q6_mxmem_AR_after_hf(output_tiles_buffer + (depth_tile * query_tiles + query_tile) * HMX_TILE_ELEMENTS,
                                     0);
            }

        for (std::size_t row = 0; row < query_length; ++row)
            for (std::size_t depth = 0; depth < head_dim; ++depth) {
                const auto* tile =
                    output_tiles_buffer + ((depth / HMX_TILE) * query_tiles + row / HMX_TILE) * HMX_TILE_ELEMENTS;
                output[(row * query_heads + head) * head_dim + depth] =
                    Float16(static_cast<float>(tile[hmx_tile_index(row % HMX_TILE, depth % HMX_TILE)]));
            }
    }
}

} // namespace kidi::qnn::htp

#endif
