#include "AttentionHvx.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>

#include <hexagon_sim_timer.h>

namespace {

template <std::size_t VECTORS>
struct Buffers {
    static constexpr std::size_t HEAD_DIM = VECTORS * 64;
    static constexpr std::size_t MAX_QUERY = 64;
    static constexpr std::size_t CAPACITY = 1024;

    alignas(128) std::array<Float16, MAX_QUERY * HEAD_DIM> query;
    alignas(128) std::array<Float16, MAX_QUERY * HEAD_DIM> current_key;
    alignas(128) std::array<Float16, MAX_QUERY * HEAD_DIM> current_value;
    alignas(128) std::array<std::uint8_t, CAPACITY * HEAD_DIM> key_cache;
    alignas(128) std::array<std::uint8_t, CAPACITY * HEAD_DIM> value_cache;
    alignas(128) std::array<Float16, MAX_QUERY * HEAD_DIM> output;
    std::array<int, MAX_QUERY> positions;

    Buffers() {
        for (std::size_t index = 0; index < query.size(); ++index) {
            query[index] = static_cast<float>(static_cast<int>((index * 13) % 43) - 21) / 48.0f;
            current_key[index] = static_cast<float>(static_cast<int>((index * 7) % 31) - 15) / 56.0f;
            current_value[index] = static_cast<float>(static_cast<int>((index * 17) % 47) - 23) / 52.0f;
        }
        for (std::size_t index = 0; index < key_cache.size(); ++index) {
            key_cache[index] = static_cast<std::uint8_t>((index * 29 + 5) & 255);
            value_cache[index] = static_cast<std::uint8_t>((index * 23 + 17) & 255);
        }
    }
};

template <std::size_t VECTORS>
auto run_kernel(Buffers<VECTORS>& buffers, std::uint32_t query_length, std::int32_t window) -> unsigned long long {
    constexpr float KEY_SCALE = 1.0f / 96.0f;
    constexpr float VALUE_SCALE = 1.0f / 112.0f;
    constexpr std::int32_t KEY_OFFSET = -127;
    constexpr std::int32_t VALUE_OFFSET = -129;
    const float attention_scale = 1.0f / std::sqrt(static_cast<float>(Buffers<VECTORS>::HEAD_DIM));

    const auto start = hexagon_sim_read_pcycles();
    kidi::qnn::htp::structured_attention_hvx<VECTORS>(
        buffers.output.data(), buffers.query.data(), buffers.current_key.data(), buffers.current_value.data(),
        buffers.key_cache.data(), buffers.value_cache.data(), buffers.positions.data(), attention_scale, window,
        query_length, 1, Buffers<VECTORS>::CAPACITY, KEY_SCALE, KEY_OFFSET, VALUE_SCALE, VALUE_OFFSET);
    return hexagon_sim_read_pcycles() - start;
}

template <std::size_t VECTORS>
void benchmark(Buffers<VECTORS>& buffers) {
    constexpr std::array<int, 4> DECODE_LENGTHS{{64, 256, 512, 1024}};

    for (int length : DECODE_LENGTHS) {
        buffers.positions[0] = length - 1;
        const auto global_cycles = run_kernel(buffers, 1, 0);
        const auto local_cycles = run_kernel(buffers, 1, 512);
        std::printf("decode head_dim=%zu length=%d global_cycles=%llu window512_cycles=%llu\n",
                    Buffers<VECTORS>::HEAD_DIM, length, global_cycles, local_cycles);
    }

    for (std::uint32_t query_length : {16U, 64U}) {
        for (std::uint32_t index = 0; index < query_length; ++index) buffers.positions[index] = static_cast<int>(index);
        const auto cycles = run_kernel(buffers, query_length, 0);
        std::printf("prefill head_dim=%zu query=%lu causal_cycles=%llu cycles_per_query=%llu\n",
                    Buffers<VECTORS>::HEAD_DIM, static_cast<unsigned long>(query_length), cycles,
                    cycles / query_length);
    }

    std::printf("checksum head_dim=%zu value=%g\n", Buffers<VECTORS>::HEAD_DIM, static_cast<float>(buffers.output[0]));
}

Buffers<4> buffers_256;
Buffers<8> buffers_512;

} // namespace

int main() {
    benchmark(buffers_256);
    benchmark(buffers_512);
}
