#pragma once

#ifdef __hexagon__

#include "HTP/core/intrinsics.h"
#include "HTP/core/memory_layout.h"
#include "HTP/core/qhpi.h"

#include "AttentionHvx.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>

namespace kidi::qnn::htp {

class Crouton16Reader {
public:
    explicit Crouton16Reader(const QHPI_Tensor* tensor) : blocks_(qhpi_tensor_block_table(tensor)) {
        const auto padded = qhpi_tensor_padded_shape(tensor);
        const auto padding = qhpi_tensor_padding(tensor);
        for (std::size_t dimension = 0; dimension < 4; ++dimension) {
            dims_[dimension] = padded.dims[dimension];
            padding_[dimension] = padding.dims[dimension];
        }
    }

    Crouton16Reader(void** blocks, std::array<std::size_t, 4> dims, std::array<std::size_t, 4> padding = {})
        : blocks_(blocks), dims_(dims), padding_(padding) {}

    auto load64(std::size_t sequence, std::size_t head, std::size_t depth) const -> HVX_Vector {
        const auto first = block_vector(sequence, head, depth);
        const auto second = block_vector(sequence, head, depth + 32);
        const auto unpacked = Q6_W_vdeal_VVR(second, first, -2);
        return (head + padding_[2]) % 2 == 0 ? Q6_V_lo_W(unpacked) : Q6_V_hi_W(unpacked);
    }

private:
    auto block_vector(std::size_t sequence, std::size_t head, std::size_t depth) const -> HVX_Vector {
        const auto padded_head = head + padding_[2];
        const std::array<std::size_t, 4> coordinates{padding_[0], sequence + padding_[1], padded_head & ~std::size_t{1},
                                                     depth + padding_[3]};
        const auto block = R4Crouton2Layout::chunk_index(coordinates, dims_);
        const auto offset = R4Crouton2Layout::chunk_offset(coordinates, dims_);
        return q6op_V_vldu_A(static_cast<const Float16*>(blocks_[block]) + offset);
    }

    void** blocks_;
    std::array<std::size_t, 4> dims_;
    std::array<std::size_t, 4> padding_;
};

template <std::size_t VECTORS>
inline void load_crouton_vectors(std::array<HVX_Vector, VECTORS>& vectors, const Crouton16Reader& reader,
                                 std::size_t sequence, std::size_t head) {
    for (std::size_t index = 0; index < VECTORS; ++index) vectors[index] = reader.load64(sequence, head, index * 64);
}

template <std::size_t VECTORS>
inline void structured_attention_crouton(Float16* output, const Crouton16Reader& query,
                                         const Crouton16Reader& current_key, const Crouton16Reader& current_value,
                                         const std::uint8_t* key_cache, const std::uint8_t* value_cache,
                                         const std::int32_t* positions, float attention_scale, std::int32_t window,
                                         std::uint32_t query_length, std::uint32_t query_heads, std::uint32_t capacity,
                                         float key_scale, std::int32_t key_offset, float value_scale,
                                         std::int32_t value_offset, std::uint32_t first_task,
                                         std::uint32_t task_count) {
    constexpr std::size_t HEAD_DIM = VECTORS * 64;
    const std::int32_t first_position = positions[0];
    for (std::uint32_t task = first_task; task < first_task + task_count; ++task) {
        const std::uint32_t query_index = task / query_heads;
        const std::uint32_t head = task % query_heads;
        const std::int32_t end = std::min<std::int32_t>(positions[query_index] + 1, capacity);
        const std::int32_t start = window > 0 ? std::max<std::int32_t>(0, end - window) : 0;
        std::array<HVX_Vector, VECTORS> query_vectors;
        std::array<HVX_Vector, VECTORS> accumulator;
        load_crouton_vectors(query_vectors, query, query_index, head);
        std::fill(accumulator.begin(), accumulator.end(), Q6_V_vzero());

        float maximum = 0.0f;
        float denominator = 0.0f;
        for (std::int32_t token = start; token < end; ++token) {
            float score;
            if (token < first_position) {
                score = dot_quant_uint8(query_vectors, key_cache + static_cast<std::size_t>(token) * HEAD_DIM,
                                        key_offset, key_scale);
            } else {
                std::array<HVX_Vector, VECTORS> key_vectors;
                load_crouton_vectors(key_vectors, current_key, token - first_position, 0);
                score = dot_float16_vectors(query_vectors, key_vectors);
            }
            score *= attention_scale;
            const auto weights = online_softmax_weights(maximum, score, denominator == 0.0f);
            denominator = denominator * weights.previous + weights.current;
            if (token < first_position) {
                accumulate_quant_uint8(accumulator, value_cache + static_cast<std::size_t>(token) * HEAD_DIM,
                                       value_offset, value_scale, weights.previous, weights.current);
            } else {
                std::array<HVX_Vector, VECTORS> value_vectors;
                load_crouton_vectors(value_vectors, current_value, token - first_position, 0);
                accumulate_float16_vectors(accumulator, value_vectors, weights.previous, weights.current);
            }
            maximum = weights.maximum;
        }
        store_normalized(accumulator, output + (query_index * query_heads + head) * HEAD_DIM, denominator);
    }
}

} // namespace kidi::qnn::htp

#endif
