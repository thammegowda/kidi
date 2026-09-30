#include "AttentionHmx.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>

namespace {

constexpr std::size_t QUERY = 128;
constexpr std::size_t HEADS = 4;
constexpr std::size_t DEPTH = 256;
constexpr std::size_t MATRIX_ELEMENTS = QUERY * DEPTH;

alignas(128) std::array<Float16, QUERY * HEADS * DEPTH> query{};
alignas(128) std::array<Float16, QUERY * DEPTH> value{};
alignas(128) std::array<__fp16, MATRIX_ELEMENTS> expected_tiles{};
alignas(128) std::array<__fp16, MATRIX_ELEMENTS> actual_tiles{};
alignas(128) std::array<Float16, QUERY * HEADS * DEPTH> expected_output{};
alignas(128) std::array<Float16, QUERY * HEADS * DEPTH> actual_output{};

void pack_query_hvx(__fp16* packed, const Float16* source, std::size_t head) {
    constexpr std::size_t query_tiles = QUERY / kidi::qnn::htp::HMX_TILE;
    constexpr std::size_t depth_tiles = DEPTH / kidi::qnn::htp::HMX_TILE;
    std::memset(packed, 0, MATRIX_ELEMENTS * sizeof(__fp16));
    for (std::size_t row = 0; row < QUERY; row += 2) {
        const auto* row0 = source + (row * HEADS + head) * DEPTH;
        const auto* row1 = source + ((row + 1) * HEADS + head) * DEPTH;
        auto* tile_base = packed + (row / kidi::qnn::htp::HMX_TILE) * depth_tiles * kidi::qnn::htp::HMX_TILE_ELEMENTS;
        const std::size_t tile_vector = (row % kidi::qnn::htp::HMX_TILE) / 2;
        for (std::size_t depth = 0; depth < DEPTH; depth += 64) {
            const auto packed_rows = Q6_W_vshuff_VVR(q6op_V_vldu_A(row1 + depth), q6op_V_vldu_A(row0 + depth), -2);
            auto* dual_tile = tile_base + (depth / kidi::qnn::htp::HMX_TILE) * kidi::qnn::htp::HMX_TILE_ELEMENTS;
            reinterpret_cast<HVX_Vector*>(dual_tile)[tile_vector] = Q6_V_lo_W(packed_rows);
            reinterpret_cast<HVX_Vector*>(dual_tile)[16 + tile_vector] = Q6_V_hi_W(packed_rows);
        }
    }
    (void)query_tiles;
}

void pack_value_hvx(__fp16* packed, const Float16* source) {
    constexpr std::size_t query_tiles = QUERY / kidi::qnn::htp::HMX_TILE;
    std::memset(packed, 0, MATRIX_ELEMENTS * sizeof(__fp16));
    for (std::size_t token = 0; token < QUERY; token += 2) {
        const auto* row0 = source + token * DEPTH;
        const auto* row1 = source + (token + 1) * DEPTH;
        const std::size_t token_tile = token / kidi::qnn::htp::HMX_TILE;
        const std::size_t tile_vector = (token % kidi::qnn::htp::HMX_TILE) / 2;
        auto* first_tile = packed + token_tile * kidi::qnn::htp::HMX_TILE_ELEMENTS;
        for (std::size_t depth = 0; depth < DEPTH; depth += 64) {
            const auto packed_rows = Q6_W_vshuff_VVR(q6op_V_vldu_A(row1 + depth), q6op_V_vldu_A(row0 + depth), -2);
            auto* tile0 =
                first_tile + (depth / kidi::qnn::htp::HMX_TILE) * query_tiles * kidi::qnn::htp::HMX_TILE_ELEMENTS;
            auto* tile1 = tile0 + query_tiles * kidi::qnn::htp::HMX_TILE_ELEMENTS;
            reinterpret_cast<HVX_Vector*>(tile0)[tile_vector] = Q6_V_lo_W(packed_rows);
            reinterpret_cast<HVX_Vector*>(tile1)[tile_vector] = Q6_V_hi_W(packed_rows);
        }
    }
}

void unpack_output_hvx(Float16* output, const __fp16* packed, std::size_t head) {
    constexpr std::size_t query_tiles = QUERY / kidi::qnn::htp::HMX_TILE;
    for (std::size_t row = 0; row < QUERY; ++row) {
        auto* output_row = output + (row * HEADS + head) * DEPTH;
        const std::size_t query_tile = row / kidi::qnn::htp::HMX_TILE;
        const std::size_t tile_vector = (row % kidi::qnn::htp::HMX_TILE) / 2;
        for (std::size_t depth = 0; depth < DEPTH; depth += 64) {
            const auto* tile0 = packed + ((depth / kidi::qnn::htp::HMX_TILE) * query_tiles + query_tile) *
                                             kidi::qnn::htp::HMX_TILE_ELEMENTS;
            const auto* tile1 = tile0 + query_tiles * kidi::qnn::htp::HMX_TILE_ELEMENTS;
            const auto unpacked = Q6_W_vdeal_VVR(reinterpret_cast<const HVX_Vector*>(tile1)[tile_vector],
                                                 reinterpret_cast<const HVX_Vector*>(tile0)[tile_vector], -2);
            q6op_vstu_AV(output_row + depth, row % 2 == 0 ? Q6_V_lo_W(unpacked) : Q6_V_hi_W(unpacked));
        }
    }
}

auto compare_bits(const __fp16* expected, const __fp16* actual, std::size_t count, const char* label) -> bool {
    const auto bits = [](const __fp16* values, std::size_t index) {
        std::uint16_t result;
        std::memcpy(&result, values + index, sizeof(result));
        return result;
    };
    for (std::size_t index = 0; index < count; ++index) {
        const auto expected_bits = bits(expected, index);
        const auto actual_bits = bits(actual, index);
        if (expected_bits != actual_bits) {
            std::printf("%s mismatch index=%zu expected=%04x actual=%04x\n", label, index, expected_bits, actual_bits);
            std::size_t expected_location = count;
            std::size_t actual_location = count;
            for (std::size_t probe = 0; probe < count; ++probe) {
                if (expected_location == count && bits(expected, probe) == actual_bits) expected_location = probe;
                if (actual_location == count && bits(actual, probe) == expected_bits) actual_location = probe;
            }
            std::printf("%s actual value occurs in expected at %zu; expected value occurs in actual at %zu\n", label,
                        expected_location, actual_location);
            return false;
        }
    }
    return true;
}

} // namespace

int main() {
    for (std::size_t index = 0; index < query.size(); ++index)
        query[index] = Float16(static_cast<float>(static_cast<int>(index % 251) - 125) / 128.0f);
    for (std::size_t index = 0; index < value.size(); ++index)
        value[index] = Float16(static_cast<float>(static_cast<int>(index % 239) - 119) / 128.0f);

    kidi::qnn::htp::hmx_pack_query(expected_tiles.data(), query.data(), QUERY, HEADS, 2, DEPTH);
    pack_query_hvx(actual_tiles.data(), query.data(), 2);
    if (!compare_bits(expected_tiles.data(), actual_tiles.data(), MATRIX_ELEMENTS, "query")) return 1;

    kidi::qnn::htp::hmx_pack_value(expected_tiles.data(), value.data(), QUERY, DEPTH);
    pack_value_hvx(actual_tiles.data(), value.data());
    if (!compare_bits(expected_tiles.data(), actual_tiles.data(), MATRIX_ELEMENTS, "value")) return 1;

    std::fill(expected_output.begin(), expected_output.end(), Float16(0.0f));
    std::fill(actual_output.begin(), actual_output.end(), Float16(0.0f));
    for (std::size_t row = 0; row < QUERY; ++row)
        for (std::size_t depth = 0; depth < DEPTH; ++depth) {
            const auto* tile =
                expected_tiles.data() + ((depth / kidi::qnn::htp::HMX_TILE) * (QUERY / kidi::qnn::htp::HMX_TILE) +
                                         row / kidi::qnn::htp::HMX_TILE) *
                                            kidi::qnn::htp::HMX_TILE_ELEMENTS;
            expected_output[(row * HEADS + 1) * DEPTH + depth] =
                Float16(static_cast<float>(tile[kidi::qnn::htp::hmx_tile_index(row % kidi::qnn::htp::HMX_TILE,
                                                                               depth % kidi::qnn::htp::HMX_TILE)]));
        }
    unpack_output_hvx(actual_output.data(), expected_tiles.data(), 1);
    if (!compare_bits(reinterpret_cast<const __fp16*>(expected_output.data()),
                      reinterpret_cast<const __fp16*>(actual_output.data()), expected_output.size(), "output"))
        return 1;
    std::printf("HMX packing transforms match scalar layout\n");
}
