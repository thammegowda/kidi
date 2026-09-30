#include "AttentionHvx.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>

namespace {

template <std::size_t VECTORS>
auto run_primitive_test() -> bool {
    constexpr std::size_t SIZE = VECTORS * 64;
    constexpr std::int32_t OFFSET = -128;
    constexpr float SCALE = 1.0f / 128.0f;

    alignas(128) std::array<Float16, SIZE> query;
    alignas(128) std::array<Float16, SIZE> key;
    alignas(128) std::array<Float16, SIZE> value;
    alignas(128) std::array<std::uint8_t, SIZE> quantized;
    alignas(128) std::array<Float16, SIZE> output;
    std::array<float, SIZE> reference_accumulator{};

    for (std::size_t index = 0; index < SIZE; ++index) {
        query[index] = static_cast<float>(static_cast<int>(index % 29) - 14) / 32.0f;
        key[index] = static_cast<float>(static_cast<int>((index * 7) % 31) - 15) / 64.0f;
        value[index] = static_cast<float>(static_cast<int>((index * 11) % 37) - 18) / 48.0f;
        quantized[index] = static_cast<std::uint8_t>((index * 19 + 23) & 255);
    }

    std::array<HVX_Vector, VECTORS> query_vectors;
    kidi::qnn::htp::load_float16_vectors(query_vectors, query.data());

    float reference_dot = 0.0f;
    float reference_quantized_dot = 0.0f;
    for (std::size_t index = 0; index < SIZE; ++index) {
        const float query_value = static_cast<float>(static_cast<int>(index % 29) - 14) / 32.0f;
        const float key_value = static_cast<float>(static_cast<int>((index * 7) % 31) - 15) / 64.0f;
        reference_dot += query_value * key_value;
        reference_quantized_dot +=
            query_value * static_cast<float>(static_cast<int>(quantized[index]) + OFFSET) * SCALE;
    }

    const float actual_dot = kidi::qnn::htp::dot_float16(query_vectors, key.data());
    const float actual_quantized_dot = kidi::qnn::htp::dot_quant_uint8(query_vectors, quantized.data(), OFFSET, SCALE);
    if (std::abs(actual_dot - reference_dot) > 2.0e-3f ||
        std::abs(actual_quantized_dot - reference_quantized_dot) > 2.0e-3f) {
        std::printf("dot mismatch size=%zu fp16=%g/%g quant=%g/%g\n", SIZE, actual_dot, reference_dot,
                    actual_quantized_dot, reference_quantized_dot);
        return false;
    }

    std::array<HVX_Vector, VECTORS> accumulator;
    std::fill(accumulator.begin(), accumulator.end(), Q6_V_vzero());
    constexpr std::array<float, 4> PREVIOUS{{0.0f, 0.75f, 1.0f, 0.625f}};
    constexpr std::array<float, 4> CURRENT{{1.0f, 0.5f, 0.125f, 1.0f}};
    for (std::size_t step = 0; step < PREVIOUS.size(); ++step) {
        if ((step & 1) == 0) {
            kidi::qnn::htp::accumulate_float16(accumulator, value.data(), PREVIOUS[step], CURRENT[step]);
            for (std::size_t index = 0; index < SIZE; ++index)
                reference_accumulator[index] =
                    reference_accumulator[index] * PREVIOUS[step] + static_cast<float>(value[index]) * CURRENT[step];
        } else {
            kidi::qnn::htp::accumulate_quant_uint8(accumulator, quantized.data(), OFFSET, SCALE, PREVIOUS[step],
                                                   CURRENT[step]);
            for (std::size_t index = 0; index < SIZE; ++index)
                reference_accumulator[index] =
                    reference_accumulator[index] * PREVIOUS[step] +
                    static_cast<float>(static_cast<int>(quantized[index]) + OFFSET) * SCALE * CURRENT[step];
        }
    }

    constexpr float DENOMINATOR = 2.375f;
    kidi::qnn::htp::store_normalized(accumulator, output.data(), DENOMINATOR);
    float maximum_error = 0.0f;
    float squared_error = 0.0f;
    for (std::size_t index = 0; index < SIZE; ++index) {
        const float error = static_cast<float>(output[index]) - reference_accumulator[index] / DENOMINATOR;
        maximum_error = std::max(maximum_error, std::abs(error));
        squared_error += error * error;
    }
    const float rms_error = std::sqrt(squared_error / static_cast<float>(SIZE));
    std::printf("size=%zu dot=%g quant_dot=%g value_max_error=%g value_rms_error=%g\n", SIZE, actual_dot,
                actual_quantized_dot, maximum_error, rms_error);
    return maximum_error < 3.0e-3f && rms_error < 1.0e-3f;
}

template <std::size_t VECTORS>
auto run_attention_test(std::int32_t window) -> bool {
    constexpr std::size_t HEAD_DIM = VECTORS * 64;
    constexpr std::size_t QUERY_LENGTH = 3;
    constexpr std::size_t QUERY_HEADS = 2;
    constexpr std::size_t CAPACITY = 12;
    constexpr std::int32_t FIRST_POSITION = 5;
    constexpr std::int32_t KEY_OFFSET = -127;
    constexpr std::int32_t VALUE_OFFSET = -129;
    constexpr float KEY_SCALE = 1.0f / 96.0f;
    constexpr float VALUE_SCALE = 1.0f / 112.0f;

    alignas(128) static std::array<Float16, QUERY_LENGTH * QUERY_HEADS * HEAD_DIM> query;
    alignas(128) static std::array<Float16, QUERY_LENGTH * HEAD_DIM> current_key;
    alignas(128) static std::array<Float16, QUERY_LENGTH * HEAD_DIM> current_value;
    alignas(128) static std::array<std::uint8_t, CAPACITY * HEAD_DIM> key_cache;
    alignas(128) static std::array<std::uint8_t, CAPACITY * HEAD_DIM> value_cache;
    alignas(128) static std::array<Float16, QUERY_LENGTH * QUERY_HEADS * HEAD_DIM> output;
    static std::array<float, QUERY_LENGTH * QUERY_HEADS * HEAD_DIM> reference;
    constexpr std::array<std::int32_t, QUERY_LENGTH> POSITIONS{
        {FIRST_POSITION, FIRST_POSITION + 1, FIRST_POSITION + 2}};

    for (std::size_t index = 0; index < query.size(); ++index)
        query[index] = static_cast<float>(static_cast<int>((index * 13) % 43) - 21) / 48.0f;
    for (std::size_t index = 0; index < current_key.size(); ++index) {
        current_key[index] = static_cast<float>(static_cast<int>((index * 7) % 31) - 15) / 56.0f;
        current_value[index] = static_cast<float>(static_cast<int>((index * 17) % 47) - 23) / 52.0f;
    }
    for (std::size_t index = 0; index < key_cache.size(); ++index) {
        key_cache[index] = static_cast<std::uint8_t>((index * 29 + 5) & 255);
        value_cache[index] = static_cast<std::uint8_t>((index * 23 + 17) & 255);
    }

    const float attention_scale = 1.0f / std::sqrt(static_cast<float>(HEAD_DIM));
    constexpr std::uint32_t SLICES = 4;
    constexpr std::uint32_t TASKS = QUERY_LENGTH * QUERY_HEADS;
    for (std::uint32_t slice = 0; slice < SLICES; ++slice) {
        const auto first = TASKS * slice / SLICES;
        const auto end = TASKS * (slice + 1) / SLICES;
        kidi::qnn::htp::structured_attention_hvx<VECTORS>(
            output.data(), query.data(), current_key.data(), current_value.data(), key_cache.data(), value_cache.data(),
            POSITIONS.data(), attention_scale, window, QUERY_LENGTH, QUERY_HEADS, CAPACITY, KEY_SCALE, KEY_OFFSET,
            VALUE_SCALE, VALUE_OFFSET, first, end - first);
    }

    for (std::size_t query_index = 0; query_index < QUERY_LENGTH; ++query_index) {
        const std::int32_t end = POSITIONS[query_index] + 1;
        const std::int32_t start = window > 0 ? std::max<std::int32_t>(0, end - window) : 0;
        for (std::size_t head = 0; head < QUERY_HEADS; ++head) {
            std::array<float, CAPACITY> scores{};
            float maximum = -1.0e30f;
            for (std::int32_t token = start; token < end; ++token) {
                float score = 0.0f;
                for (std::size_t channel = 0; channel < HEAD_DIM; ++channel) {
                    const float query_element =
                        static_cast<float>(query[(query_index * QUERY_HEADS + head) * HEAD_DIM + channel]);
                    const float key_element =
                        token < FIRST_POSITION
                            ? static_cast<float>(static_cast<int>(key_cache[token * HEAD_DIM + channel]) + KEY_OFFSET) *
                                  KEY_SCALE
                            : static_cast<float>(current_key[(token - FIRST_POSITION) * HEAD_DIM + channel]);
                    score += query_element * key_element;
                }
                scores[token] = score * attention_scale;
                maximum = std::max(maximum, scores[token]);
            }
            float denominator = 0.0f;
            for (std::int32_t token = start; token < end; ++token) denominator += std::exp(scores[token] - maximum);
            for (std::size_t channel = 0; channel < HEAD_DIM; ++channel) {
                float accumulator = 0.0f;
                for (std::int32_t token = start; token < end; ++token) {
                    const float value_element =
                        token < FIRST_POSITION
                            ? static_cast<float>(static_cast<int>(value_cache[token * HEAD_DIM + channel]) +
                                                 VALUE_OFFSET) *
                                  VALUE_SCALE
                            : static_cast<float>(current_value[(token - FIRST_POSITION) * HEAD_DIM + channel]);
                    accumulator += std::exp(scores[token] - maximum) * value_element;
                }
                reference[(query_index * QUERY_HEADS + head) * HEAD_DIM + channel] = accumulator / denominator;
            }
        }
    }

    float maximum_error = 0.0f;
    float squared_error = 0.0f;
    float reference_squared = 0.0f;
    for (std::size_t index = 0; index < output.size(); ++index) {
        const float error = static_cast<float>(output[index]) - reference[index];
        maximum_error = std::max(maximum_error, std::abs(error));
        squared_error += error * error;
        reference_squared += reference[index] * reference[index];
    }
    const float rms_error = std::sqrt(squared_error / static_cast<float>(output.size()));
    const float relative_rms = std::sqrt(squared_error / reference_squared);
    std::printf("attention size=%zu window=%ld max_error=%g rms_error=%g relative_rms=%g\n", HEAD_DIM,
                static_cast<long>(window), maximum_error, rms_error, relative_rms);
    return maximum_error < 5.0e-3f && relative_rms < 5.0e-3f;
}

} // namespace

int main() {
    return run_primitive_test<4>() && run_primitive_test<8>() && run_attention_test<4>(0) && run_attention_test<4>(4) &&
                   run_attention_test<8>(0) && run_attention_test<8>(4)
               ? 0
               : 1;
}
