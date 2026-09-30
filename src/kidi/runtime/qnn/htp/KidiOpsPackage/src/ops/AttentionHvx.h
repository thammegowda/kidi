#pragma once

#ifdef __hexagon__

#include "AttentionMath.h"

#include "HTP/core/intrinsics.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace kidi::qnn::htp {

inline auto float_from_bits(std::int32_t bits) -> float {
    union {
        std::int32_t bits;
        float value;
    } cast{bits};
    return cast.value;
}

inline auto horizontal_sum_qf32(HVX_Vector value) -> float {
    const auto zero = Q6_V_vzero();
    for (int shift = 64; shift >= 4; shift >>= 1)
        value = Q6_Vqf32_vadd_Vqf32Vqf32(value, Q6_V_vlalign_VVR(value, zero, shift));
    return float_from_bits(Q6_R_vextract_VR(Q6_Vsf_equals_Vqf32(value), 124));
}

template <std::size_t VECTORS>
inline void load_float16_vectors(std::array<HVX_Vector, VECTORS>& vectors, const Float16* source) {
    for (std::size_t index = 0; index < VECTORS; ++index) vectors[index] = q6op_V_vldu_A(source + index * 64);
}

template <std::size_t VECTORS>
inline auto dot_float16_vectors(const std::array<HVX_Vector, VECTORS>& query,
                                const std::array<HVX_Vector, VECTORS>& key) -> float {
    auto sum_low = Q6_V_vzero();
    auto sum_high = Q6_V_vzero();
    for (std::size_t index = 0; index < VECTORS; ++index) {
        const auto product = Q6_Wqf32_vmpy_VhfVhf(query[index], key[index]);
        sum_low = Q6_Vqf32_vadd_Vqf32Vqf32(sum_low, Q6_V_lo_W(product));
        sum_high = Q6_Vqf32_vadd_Vqf32Vqf32(sum_high, Q6_V_hi_W(product));
    }
    return horizontal_sum_qf32(Q6_Vqf32_vadd_Vqf32Vqf32(sum_low, sum_high));
}

template <std::size_t VECTORS>
inline auto dot_float16(const std::array<HVX_Vector, VECTORS>& query, const Float16* key) -> float {
    std::array<HVX_Vector, VECTORS> key_vectors;
    load_float16_vectors(key_vectors, key);
    return dot_float16_vectors(query, key_vectors);
}

template <std::size_t VECTORS>
inline auto dot_quant_uint8(const std::array<HVX_Vector, VECTORS>& query, const std::uint8_t* key, std::int32_t offset,
                            float scale) -> float {
    static_assert(VECTORS % 2 == 0);
    auto sum_low = Q6_V_vzero();
    auto sum_high = Q6_V_vzero();
    const auto offset_vector = Q6_Vh_vsplat_R(offset);
    for (std::size_t index = 0; index < VECTORS / 2; ++index) {
        const auto unpacked = Q6_Wuh_vunpack_Vub(q6op_V_vldu_A(key + index * 128));
        const auto key_low = Q6_Vhf_equals_Vh(Q6_Vh_vadd_VhVh(Q6_V_lo_W(unpacked), offset_vector));
        const auto key_high = Q6_Vhf_equals_Vh(Q6_Vh_vadd_VhVh(Q6_V_hi_W(unpacked), offset_vector));
        const auto product_low = Q6_Wqf32_vmpy_VhfVhf(query[index * 2], key_low);
        const auto product_high = Q6_Wqf32_vmpy_VhfVhf(query[index * 2 + 1], key_high);
        sum_low = Q6_Vqf32_vadd_Vqf32Vqf32(sum_low,
                                           Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(product_low), Q6_V_lo_W(product_high)));
        sum_high = Q6_Vqf32_vadd_Vqf32Vqf32(sum_high,
                                            Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_hi_W(product_low), Q6_V_hi_W(product_high)));
    }
    return horizontal_sum_qf32(Q6_Vqf32_vadd_Vqf32Vqf32(sum_low, sum_high)) * scale;
}

template <std::size_t VECTORS>
inline void accumulate_float16_vectors(std::array<HVX_Vector, VECTORS>& accumulator,
                                       const std::array<HVX_Vector, VECTORS>& value, float previous_weight,
                                       float current_weight) {
    const auto previous = q6op_V_vsplat_float16(Float16(previous_weight));
    const auto current = q6op_V_vsplat_float16(Float16(current_weight));
    for (std::size_t index = 0; index < VECTORS; ++index) {
        const auto retained = Q6_Vqf16_vmpy_Vqf16Vhf(accumulator[index], previous);
        const auto added = Q6_Vqf16_vmpy_VhfVhf(value[index], current);
        accumulator[index] = Q6_Vqf16_vadd_Vqf16Vqf16(retained, added);
    }
}

template <std::size_t VECTORS>
inline void accumulate_float16(std::array<HVX_Vector, VECTORS>& accumulator, const Float16* value,
                               float previous_weight, float current_weight) {
    std::array<HVX_Vector, VECTORS> value_vectors;
    load_float16_vectors(value_vectors, value);
    accumulate_float16_vectors(accumulator, value_vectors, previous_weight, current_weight);
}

template <std::size_t VECTORS>
inline void accumulate_quant_uint8(std::array<HVX_Vector, VECTORS>& accumulator, const std::uint8_t* value,
                                   std::int32_t offset, float scale, float previous_weight, float current_weight) {
    static_assert(VECTORS % 2 == 0);
    const auto previous = q6op_V_vsplat_float16(Float16(previous_weight));
    const auto current = q6op_V_vsplat_float16(Float16(current_weight * scale));
    const auto offset_vector = Q6_Vh_vsplat_R(offset);
    for (std::size_t index = 0; index < VECTORS / 2; ++index) {
        const auto unpacked = Q6_Wuh_vunpack_Vub(q6op_V_vldu_A(value + index * 128));
        const auto value_low = Q6_Vhf_equals_Vh(Q6_Vh_vadd_VhVh(Q6_V_lo_W(unpacked), offset_vector));
        const auto value_high = Q6_Vhf_equals_Vh(Q6_Vh_vadd_VhVh(Q6_V_hi_W(unpacked), offset_vector));
        const auto retained_low = Q6_Vqf16_vmpy_Vqf16Vhf(accumulator[index * 2], previous);
        const auto retained_high = Q6_Vqf16_vmpy_Vqf16Vhf(accumulator[index * 2 + 1], previous);
        accumulator[index * 2] = Q6_Vqf16_vadd_Vqf16Vqf16(retained_low, Q6_Vqf16_vmpy_VhfVhf(value_low, current));
        accumulator[index * 2 + 1] = Q6_Vqf16_vadd_Vqf16Vqf16(retained_high, Q6_Vqf16_vmpy_VhfVhf(value_high, current));
    }
}

template <std::size_t VECTORS>
inline void store_normalized(const std::array<HVX_Vector, VECTORS>& accumulator, Float16* output, float denominator) {
    const auto inverse = q6op_V_vsplat_float16(Float16(denominator > 0.0f ? 1.0f / denominator : 0.0f));
    for (std::size_t index = 0; index < VECTORS; ++index) {
        const auto normalized = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_Vqf16Vhf(accumulator[index], inverse));
        q6op_vstu_AV(output + index * 64, normalized);
    }
}

template <std::size_t VECTORS, typename Position>
inline void structured_attention_hvx(Float16* output, const Float16* query, const Float16* current_key,
                                     const Float16* current_value, const std::uint8_t* key_cache,
                                     const std::uint8_t* value_cache, const Position* positions, float attention_scale,
                                     std::int32_t window, std::uint32_t query_length, std::uint32_t query_heads,
                                     std::uint32_t capacity, float key_scale, std::int32_t key_offset,
                                     float value_scale, std::int32_t value_offset, std::uint32_t first_task = 0,
                                     std::uint32_t task_count = 0) {
    static_assert(std::is_integral_v<Position> && sizeof(Position) == 4);
    constexpr std::size_t HEAD_DIM = VECTORS * 64;
    const std::int32_t first_position = positions[0];
    const std::uint32_t total_tasks = query_length * query_heads;
    if (task_count == 0 || first_task + task_count > total_tasks) task_count = total_tasks - first_task;
    for (std::uint32_t task = first_task; task < first_task + task_count; ++task) {
        const std::uint32_t query_index = task / query_heads;
        const std::uint32_t head = task % query_heads;
        const std::int32_t position = static_cast<std::int32_t>(positions[query_index]);
        const std::int32_t end = std::min<std::int32_t>(position + 1, static_cast<std::int32_t>(capacity));
        const std::int32_t start = window > 0 ? std::max<std::int32_t>(0, end - window) : 0;
        const auto* query_row = query + (query_index * query_heads + head) * HEAD_DIM;
        std::array<HVX_Vector, VECTORS> query_vectors;
        std::array<HVX_Vector, VECTORS> accumulator;
        load_float16_vectors(query_vectors, query_row);
        std::fill(accumulator.begin(), accumulator.end(), Q6_V_vzero());

        float maximum = 0.0f;
        float denominator = 0.0f;
        for (std::int32_t token = start; token < end; ++token) {
            float score;
            if (token < first_position) {
                score = dot_quant_uint8(query_vectors, key_cache + static_cast<std::size_t>(token) * HEAD_DIM,
                                        key_offset, key_scale);
            } else {
                score = dot_float16(query_vectors,
                                    current_key + static_cast<std::size_t>(token - first_position) * HEAD_DIM);
            }
            score *= attention_scale;
            const auto weights = online_softmax_weights(maximum, score, denominator == 0.0f);
            denominator = denominator * weights.previous + weights.current;
            if (token < first_position) {
                accumulate_quant_uint8(accumulator, value_cache + static_cast<std::size_t>(token) * HEAD_DIM,
                                       value_offset, value_scale, weights.previous, weights.current);
            } else {
                accumulate_float16(accumulator,
                                   current_value + static_cast<std::size_t>(token - first_position) * HEAD_DIM,
                                   weights.previous, weights.current);
            }
            maximum = weights.maximum;
        }
        store_normalized(accumulator, output + (query_index * query_heads + head) * HEAD_DIM, denominator);
    }
}

} // namespace kidi::qnn::htp

#endif
