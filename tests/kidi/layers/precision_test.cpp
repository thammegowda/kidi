#include "kidi/layers/transformer.h"
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>

#include <gtest/gtest.h>

#include "kidi/checkpoint/weights.h"
#include "kidi/core/precision.h"

namespace {

auto write_bf16_weights(const std::filesystem::path& path) -> void {
    std::string header =
        R"({"linear.weight":{"dtype":"BF16","shape":[4,3],"data_offsets":[0,24]},"linear.bias":{"dtype":"F32","shape":[3],"data_offsets":[24,36]},"tgt_embed.0.lut.weight":{"dtype":"BF16","shape":[3,4],"data_offsets":[36,60]}})";
    while (header.size() % 8 != 0) header.push_back(' ');

    std::ofstream output(path, std::ios::binary);
    std::uint64_t header_size = header.size();
    if constexpr (std::endian::native == std::endian::big) header_size = std::byteswap(header_size);
    output.write(reinterpret_cast<const char*>(&header_size), sizeof(header_size));
    output.write(header.data(), static_cast<std::streamsize>(header.size()));

    constexpr std::array<std::uint16_t, 12> WEIGHT = {
        0x3f00, 0xbf80, 0x3e00, 0xbe80, 0x3f00, 0x3e80, 0x3f80, 0x3e80, 0xbf00, 0x0000, 0x3f40, 0x3f80,
    };
    constexpr std::array BIAS = {0.1F, -0.2F, 0.3F};
    output.write(reinterpret_cast<const char*>(WEIGHT.data()), sizeof(WEIGHT));
    output.write(reinterpret_cast<const char*>(BIAS.data()), sizeof(BIAS));
    constexpr std::array<std::uint16_t, 12> EMBEDDING = {
        0x3f00, 0xbe80, 0x3f80, 0x0000, 0xbf80, 0x3f00, 0x3e80, 0x3f40, 0x3e00, 0x3e80, 0xbf00, 0x3f80,
    };
    output.write(reinterpret_cast<const char*>(EMBEDDING.data()), sizeof(EMBEDDING));
}

constexpr float TOLERANCE = 1.0e-5F;

TEST(PrecisionTest, ParsesInferencePrecisions) {
    using namespace kidi;
    using core::InferencePrecision;
    constexpr auto MESSAGE = "inference precision parsing failed";
    EXPECT_EQ(core::parse_precision("q8"), InferencePrecision::Q8) << MESSAGE;
    EXPECT_EQ(core::parse_precision("q4"), InferencePrecision::Q4) << MESSAGE;
    EXPECT_EQ(core::parse_precision("qat-fp32"), InferencePrecision::QAT_FP32) << MESSAGE;
    EXPECT_FALSE(core::parse_precision("i8a8")) << MESSAGE;
    EXPECT_FALSE(core::parse_precision("unknown")) << MESSAGE;
}

TEST(PrecisionTest, ParsesKVCachePrecisions) {
    using namespace kidi;
    constexpr auto MESSAGE = "KV-cache precision parsing failed";
    EXPECT_EQ(core::parse_kv_cache_precision("int8"), core::KVCachePrecision::INT8) << MESSAGE;
    EXPECT_FALSE(core::parse_kv_cache_precision("e4m3")) << MESSAGE;
    EXPECT_EQ(core::to_string(core::KVCachePrecision::FP32), "fp32") << MESSAGE;
    EXPECT_FALSE(core::parse_kv_cache_precision("unknown")) << MESSAGE;
}

class PackedPrecisionTest : public ::testing::Test {
protected:
    static constexpr std::array INPUT_VALUES{0.15F, -0.2F};
    static constexpr std::array<std::uint8_t, 4> WEIGHT_VALUES{2, 254, 1, 1};
    static constexpr std::array SCALE_VALUES{0.5F, 0.25F};

    kidi::tensor::Tensor input =
        kidi::ops::require(kidi::tensor::Tensor::from_host({1, 2}, std::span<const float>(INPUT_VALUES)));
    kidi::tensor::Tensor weight =
        kidi::ops::require(kidi::tensor::Tensor::from_host({2, 2}, std::span<const std::uint8_t>(WEIGHT_VALUES)));
    kidi::tensor::Tensor scales =
        kidi::ops::require(kidi::tensor::Tensor::from_host({2, 1}, std::span<const float>(SCALE_VALUES)));
};

TEST_F(PackedPrecisionTest, Bf16ReferenceIgnoresActivationRounding) {
    using namespace kidi;
    using ops::require;
    ops::Context context;
    context.set_precision(core::InferencePrecision::BF16);
    const auto output = context.packed_linear(input, weight, scales, 8, 2, 0.1F, 0.1F);
    const auto values = require(output.data<float>());
    EXPECT_NEAR(values[0], 0.35F, 2.0e-3F) << "BF16 reference changed packed-linear result";
    EXPECT_NEAR(values[1], -0.0125F, 2.0e-3F) << "BF16 reference changed packed-linear result";
}

TEST_F(PackedPrecisionTest, DequantCacheScales) {
    using namespace kidi;
    using core::InferencePrecision;
    using ops::require;
    const std::array alternate_scale_values{1.F, 0.5F};
    const auto alternate_scales =
        require(tensor::Tensor::from_host({2, 1}, std::span<const float>(alternate_scale_values)));
    // Zero activation scales disable QAT rounding, leaving the FP32 dequantized product.
    ops::Context cache_context;
    cache_context.set_precision(InferencePrecision::QAT_FP32);
    static_cast<void>(cache_context.packed_linear(input, weight, scales, 8, 2, 0.F, 0.F));
    const auto alternate = cache_context.packed_linear(input, weight, alternate_scales, 8, 2, 0.F, 0.F);
    const auto alternate_values = require(alternate.data<float>());
    EXPECT_NEAR(alternate_values[0], 0.7F, TOLERANCE) << "dequantized weight cache ignored changed scales";
    EXPECT_NEAR(alternate_values[1], -0.025F, TOLERANCE) << "dequantized weight cache ignored changed scales";
}

TEST_F(PackedPrecisionTest, QatFp32KeepsActivationRounding) {
    using namespace kidi;
    using ops::require;
    ops::Context context;
    context.set_precision(core::InferencePrecision::QAT_FP32);
    const auto output = context.packed_linear(input, weight, scales, 8, 2, 0.1F, 0.1F);
    const auto values = require(output.data<float>());
    EXPECT_NEAR(values[0], 0.4F, TOLERANCE) << "QAT-FP32 projection produced unexpected SRQ output";
    EXPECT_NEAR(values[1], 0.F, TOLERANCE) << "QAT-FP32 projection produced unexpected SRQ output";
}

TEST(PrecisionTest, UnpacksLowBitWeights) {
    using namespace kidi;
    using core::InferencePrecision;
    using ops::require;
    const std::array low_bit_input_values{1.F, 2.F, 3.F, 4.F};
    const std::array<std::uint8_t, 2> low_bit_weight_values{0xe1, 0xc3};
    const std::array low_bit_scale_values{0.5F};
    const auto low_bit_input =
        require(tensor::Tensor::from_host({1, 4}, std::span<const float>(low_bit_input_values)));
    const auto low_bit_weight =
        require(tensor::Tensor::from_host({1, 2}, std::span<const std::uint8_t>(low_bit_weight_values)));
    const auto low_bit_scales =
        require(tensor::Tensor::from_host({1}, std::span<const float>(low_bit_scale_values)));
    ops::Context low_bit_context;
    low_bit_context.set_precision(InferencePrecision::QAT_FP32);
    const auto low_bit_output =
        low_bit_context.packed_linear(low_bit_input, low_bit_weight, low_bit_scales, 4, 4, 0.F, 0.F);
    EXPECT_NEAR(require(low_bit_output.data<float>())[0], -5.F, TOLERANCE)
        << "QAT-FP32 precision did not unpack signed 4-bit weights";
}

TEST(PrecisionTest, Bf16LinearAndTiedLinear) {
    using namespace kidi;
    const auto path = std::filesystem::temp_directory_path() / "kidi-bf16-precision-test.safetensors";
    write_bf16_weights(path);
    auto weights = ops::require(checkpoint::Weights::load(path));
    const ModuleScope construction(tensor::DType::BF16, false);
    layers::Linear linear(4, 3);
    ops::require(linear->set_state(StateDict{{"weight", ops::require(weights.tensor("linear.weight"))},
                                             {"bias", ops::require(weights.tensor("linear.bias"))}}));
    layers::Linear tied(4, 3, true);
    ops::require(tied->set_state(StateDict{{"weight", ops::require(weights.tensor("tgt_embed.0.lut.weight"))},
                                           {"bias", ops::require(weights.tensor("linear.bias"))}}));
    const std::array inputs{1.F, 2.F, 3.F, 4.F, -1.F, 0.5F, 2.F, -0.5F};
    const std::array expected{3.1F, 3.55F, 3.425F, 1.475F, 1.175F, -1.2F};
    std::vector devices{tensor::Device::cpu()};
#if defined(__APPLE__)
    devices.push_back(tensor::Device::apple_gpu());
#endif
    for (auto device : devices) {
        ops::Context context(device);
        auto input = ops::require(tensor::Tensor::from_host({1, 2, 4}, std::span<const float>(inputs), device));
        auto output = linear->forward(context, input);
        auto tied_output = tied->forward(context, input);
        context.synchronize();
        auto values = ops::require(output.data<float>());
        auto tied_values = ops::require(tied_output.data<float>());
        for (std::size_t index = 0; index < expected.size(); ++index) {
            EXPECT_NEAR(values[index], expected[index], TOLERANCE) << "linear index " << index;
            EXPECT_NEAR(tied_values[index], expected[index], TOLERANCE) << "tied index " << index;
        }
    }
    std::filesystem::remove(path);
}

} // namespace
