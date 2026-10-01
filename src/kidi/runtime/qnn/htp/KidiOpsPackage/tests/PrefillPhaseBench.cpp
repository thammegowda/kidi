// Breaks the HVX prefill attention inner loop into phases, in cycles per (query, key) pair.
#include "AttentionHvx.h"

#include <array>
#include <cstdint>
#include <cstdio>

#include <hexagon_sim_timer.h>

namespace {

constexpr std::size_t VECTORS = 4;
constexpr std::size_t HEAD_DIM = VECTORS * 64;
constexpr std::size_t QUERY = 32;
constexpr std::size_t PAIRS = QUERY * (QUERY + 1) / 2;

alignas(128) std::array<Float16, QUERY * HEAD_DIM> query;
alignas(128) std::array<Float16, QUERY * HEAD_DIM> key;
alignas(128) std::array<Float16, QUERY * HEAD_DIM> value;
alignas(128) std::array<std::uint8_t, QUERY * HEAD_DIM> cache;
alignas(128) std::array<Float16, QUERY * HEAD_DIM> output;
std::array<int, QUERY> positions;
volatile float sink;

template <typename Body>
auto measure(const char* label, Body body) -> void {
    const auto start = hexagon_sim_read_pcycles();
    body();
    const auto cycles = hexagon_sim_read_pcycles() - start;
    std::printf("%-34s %8llu cycles  %6.1f cycles/pair\n", label, cycles, static_cast<double>(cycles) / PAIRS);
}

} // namespace

int main() {
    for (std::size_t index = 0; index < query.size(); ++index) {
        query[index] = static_cast<float>(static_cast<int>(index % 37) - 18) / 64.0f;
        key[index] = static_cast<float>(static_cast<int>(index % 41) - 20) / 64.0f;
        value[index] = static_cast<float>(static_cast<int>(index % 43) - 21) / 64.0f;
        cache[index] = static_cast<std::uint8_t>(index * 29 + 5);
    }
    for (std::size_t index = 0; index < QUERY; ++index) positions[index] = static_cast<int>(index);

    using namespace kidi::qnn::htp;
    measure("dot FP16 + horizontal reduce", [] {
        float total = 0;
        for (std::size_t row = 0; row < QUERY; ++row) {
            std::array<HVX_Vector, VECTORS> q;
            load_float16_vectors(q, query.data() + row * HEAD_DIM);
            for (std::size_t token = 0; token <= row; ++token) total += dot_float16(q, key.data() + token * HEAD_DIM);
        }
        sink = total;
    });
    measure("dot INT8 cache + reduce", [] {
        float total = 0;
        for (std::size_t row = 0; row < QUERY; ++row) {
            std::array<HVX_Vector, VECTORS> q;
            load_float16_vectors(q, query.data() + row * HEAD_DIM);
            for (std::size_t token = 0; token <= row; ++token)
                total += dot_quant_uint8(q, cache.data() + token * HEAD_DIM, -128, 0.01f);
        }
        sink = total;
    });
    measure("scalar online-softmax update", [] {
        float maximum = 0, denominator = 0;
        for (std::size_t row = 0; row < QUERY; ++row)
            for (std::size_t token = 0; token <= row; ++token) {
                const float score = static_cast<float>((row * 7 + token * 3) % 17) * 0.1f;
                const auto weights = online_softmax_weights(maximum, score, denominator == 0.0f);
                denominator = denominator * weights.previous + weights.current;
                maximum = weights.maximum;
            }
        sink = denominator;
    });
    measure("PV accumulate (weights->fp16 splat)", [] {
        for (std::size_t row = 0; row < QUERY; ++row) {
            std::array<HVX_Vector, VECTORS> accumulator{};
            for (std::size_t token = 0; token <= row; ++token)
                accumulate_float16(accumulator, value.data() + token * HEAD_DIM, 0.75f, 0.25f);
            store_normalized(accumulator, output.data() + row * HEAD_DIM, 1.0f);
        }
    });
    measure("full kernel (1 head)", [] {
        structured_attention_hvx<VECTORS>(output.data(), query.data(), key.data(), value.data(), cache.data(),
                                          cache.data(), positions.data(), 0.0625f, 0, QUERY, 1, QUERY, 0.01f, -128,
                                          0.01f, -128);
    });
}
