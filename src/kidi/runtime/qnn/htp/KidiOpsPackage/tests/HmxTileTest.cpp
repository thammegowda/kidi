#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>

#include <hmx_hexagon_protos.h>

namespace {

constexpr std::size_t TILE = 32;
constexpr std::size_t TILE_ELEMENTS = TILE * TILE;
constexpr std::size_t TILE_BYTES = TILE_ELEMENTS * sizeof(__fp16);

constexpr auto tile_index(std::size_t row, std::size_t column) -> std::size_t {
    return (row / 2) * 64 + column * 2 + row % 2;
}

void pack_tile(__fp16* output, const __fp16* input, std::size_t row_stride) {
    for (std::size_t row = 0; row < TILE; ++row)
        for (std::size_t column = 0; column < TILE; ++column)
            output[tile_index(row, column)] = input[row * row_stride + column];
}

void unpack_tile(__fp16* output, const __fp16* input, std::size_t row_stride) {
    for (std::size_t row = 0; row < TILE; ++row)
        for (std::size_t column = 0; column < TILE; ++column)
            output[row * row_stride + column] = input[tile_index(row, column)];
}

template <std::size_t DEPTH>
auto test_matmul() -> bool {
    static_assert(DEPTH % TILE == 0);
    constexpr std::size_t DEPTH_TILES = DEPTH / TILE;
    alignas(128) std::array<__fp16, TILE * DEPTH> left{};
    alignas(128) std::array<__fp16, DEPTH * TILE> right{};
    alignas(128) std::array<__fp16, DEPTH_TILES * TILE_ELEMENTS> left_tiles{};
    alignas(128) std::array<__fp16, DEPTH_TILES * TILE_ELEMENTS> right_tiles{};
    alignas(128) std::array<__fp16, TILE_ELEMENTS> packed_output{};
    alignas(128) std::array<__fp16, TILE_ELEMENTS> output{};
    alignas(128) std::array<std::uint16_t, 128> scales{};

    for (std::size_t index = 0; index < left.size(); ++index)
        left[index] = static_cast<float>(static_cast<int>((index * 13) % 37) - 18) / 256.0f;
    for (std::size_t index = 0; index < right.size(); ++index)
        right[index] = static_cast<float>(static_cast<int>((index * 17) % 41) - 20) / 256.0f;
    for (std::size_t tile = 0; tile < DEPTH_TILES; ++tile) {
        std::array<__fp16, TILE_ELEMENTS> left_block{};
        std::array<__fp16, TILE_ELEMENTS> right_block{};
        for (std::size_t row = 0; row < TILE; ++row)
            for (std::size_t column = 0; column < TILE; ++column) {
                left_block[row * TILE + column] = left[row * DEPTH + tile * TILE + column];
                right_block[row * TILE + column] = right[(tile * TILE + row) * TILE + column];
            }
        pack_tile(left_tiles.data() + tile * TILE_ELEMENTS, left_block.data(), TILE);
        pack_tile(right_tiles.data() + tile * TILE_ELEMENTS, right_block.data(), TILE);
    }
    for (std::size_t column = 0; column < TILE; ++column) scales[column * 2] = 0x3c00;

    Q6_bias_mxmem2_A(scales.data());
    Q6_mxclracc_hf();
    for (std::size_t tile = 0; tile < DEPTH_TILES; ++tile) {
        Q6_activation_hf_mxmem_RR(reinterpret_cast<std::uintptr_t>(left_tiles.data() + tile * TILE_ELEMENTS),
                                  static_cast<std::int32_t>(TILE_BYTES - 1));
        Q6_weight_hf_mxmem_RR(reinterpret_cast<std::uintptr_t>(right_tiles.data() + tile * TILE_ELEMENTS),
                              static_cast<std::int32_t>(TILE_BYTES - 1));
    }
    Q6_mxmem_AR_after_hf(packed_output.data(), 0);
    unpack_tile(output.data(), packed_output.data(), TILE);

    float maximum_error = 0.0f;
    float squared_error = 0.0f;
    float reference_squared = 0.0f;
    for (std::size_t row = 0; row < TILE; ++row)
        for (std::size_t column = 0; column < TILE; ++column) {
            float expected = 0.0f;
            for (std::size_t depth = 0; depth < DEPTH; ++depth)
                expected +=
                    static_cast<float>(left[row * DEPTH + depth]) * static_cast<float>(right[depth * TILE + column]);
            const float error = static_cast<float>(output[row * TILE + column]) - expected;
            maximum_error = std::max(maximum_error, std::abs(error));
            squared_error += error * error;
            reference_squared += expected * expected;
        }
    const float relative_rms = std::sqrt(squared_error / reference_squared);
    std::printf("HMX depth=%zu max_error=%g relative_rms=%g first=%g\n", DEPTH, maximum_error, relative_rms,
                static_cast<float>(output[0]));
    return maximum_error < 2.0e-3f && relative_rms < 2.0e-3f;
}

auto test_direct32() -> bool {
    alignas(128) std::array<__fp16, TILE_ELEMENTS> left{};
    alignas(128) std::array<__fp16, TILE_ELEMENTS> right{};
    alignas(128) std::array<__fp16, TILE_ELEMENTS> output{};
    alignas(128) std::array<std::uint16_t, 128> scales{};
    for (std::size_t index = 0; index < TILE_ELEMENTS; ++index) {
        left[index] = 1.0f;
        right[index] = 1.0f;
    }
    for (std::size_t column = 0; column < TILE; ++column) scales[column * 2] = 0x3c00;
    Q6_bias_mxmem2_A(scales.data());
    Q6_mxclracc_hf();
    Q6_activation_hf_mxmem_RR(reinterpret_cast<std::uintptr_t>(left.data()), 32767);
    Q6_weight_hf_mxmem_RR(reinterpret_cast<std::uintptr_t>(right.data()), 1920);
    Q6_mxmem_AR_after_hf(output.data(), 0);

    float maximum_error = 0.0f;
    float squared_error = 0.0f;
    float reference_squared = 0.0f;
    for (std::size_t row = 0; row < TILE; ++row)
        for (std::size_t column = 0; column < TILE; ++column) {
            float expected = 0.0f;
            for (std::size_t depth = 0; depth < TILE; ++depth)
                expected +=
                    static_cast<float>(left[row * TILE + depth]) * static_cast<float>(right[depth * TILE + column]);
            const float error = static_cast<float>(output[row * TILE + column]) - expected;
            maximum_error = std::max(maximum_error, std::abs(error));
            squared_error += error * error;
            reference_squared += expected * expected;
        }
    const float relative_rms = std::sqrt(squared_error / reference_squared);
    std::printf("HMX direct32 max_error=%g relative_rms=%g first=%g\n", maximum_error, relative_rms,
                static_cast<float>(output[0]));
    return maximum_error < 2.0e-3f && relative_rms < 2.0e-3f;
}

} // namespace

int main() {
    const bool direct32 = test_direct32();
    const bool depth32 = test_matmul<32>();
    const bool depth256 = test_matmul<256>();
    const bool depth512 = test_matmul<512>();
    return direct32 && depth32 && depth256 && depth512 ? 0 : 1;
}
