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
    EXPECT_EQ(core::parse_precision("fp32"), InferencePrecision::FP32) << MESSAGE;
    EXPECT_EQ(core::parse_precision("lowbit-parity"), InferencePrecision::LOWBIT_PARITY) << MESSAGE;
    EXPECT_EQ(core::parse_precision("i2a16"), InferencePrecision::Q2A16) << MESSAGE;
    EXPECT_EQ(core::parse_precision("i4a16"), InferencePrecision::Q4A16) << MESSAGE;
    EXPECT_EQ(core::parse_precision("i8a16"), InferencePrecision::Q8A16) << MESSAGE;
    EXPECT_EQ(core::parse_precision("i8a8"), InferencePrecision::Q8AE4M3) << MESSAGE;
    EXPECT_EQ(core::parse_precision("w8afp8"), InferencePrecision::Q8AE4M3) << MESSAGE;
    EXPECT_EQ(core::to_string(InferencePrecision::Q4AE5M2), "q4ae5m2") << MESSAGE;
    EXPECT_FALSE(core::parse_precision("unknown")) << MESSAGE;
}

TEST(PrecisionTest, ParsesKVCachePrecisions) {
    using namespace kidi;
    constexpr auto MESSAGE = "KV-cache precision parsing failed";
    EXPECT_EQ(core::parse_kv_cache_precision("i8"), core::KVCachePrecision::INT8) << MESSAGE;
    EXPECT_EQ(core::parse_kv_cache_precision("fp8-e4m3"), core::KVCachePrecision::E4M3) << MESSAGE;
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

TEST_F(PackedPrecisionTest, PrecisionIndependent) {
    using namespace kidi;
    using core::InferencePrecision;
    using ops::require;
    for (const auto precision : {InferencePrecision::FP32, InferencePrecision::BF16,
                                 InferencePrecision::Q8A16}) {
        SCOPED_TRACE(core::to_string(precision));
        ops::Context context;
        context.set_precision(precision);
        const auto output = context.packed_linear(input, weight, scales, 8, 2, 0.1F, 0.1F);
        const auto values = require(output.data<float>());
        const auto tolerance = precision == InferencePrecision::FP32 ? 1.0e-5F : 2.0e-3F;
        EXPECT_NEAR(values[0], 0.35F, tolerance) << "inference precision changed packed-linear result";
        EXPECT_NEAR(values[1], -0.0125F, tolerance) << "inference precision changed packed-linear result";
    }
}

TEST_F(PackedPrecisionTest, DequantCacheScales) {
    using namespace kidi;
    using core::InferencePrecision;
    using ops::require;
    const std::array alternate_scale_values{1.F, 0.5F};
    const auto alternate_scales =
        require(tensor::Tensor::from_host({2, 1}, std::span<const float>(alternate_scale_values)));
    ops::Context cache_context;
    cache_context.set_precision(InferencePrecision::FP32);
    static_cast<void>(cache_context.packed_linear(input, weight, scales, 8, 2, 0.1F, 0.1F));
    const auto alternate =
        cache_context.packed_linear(input, weight, alternate_scales, 8, 2, 0.1F, 0.1F);
    const auto alternate_values = require(alternate.data<float>());
    EXPECT_NEAR(alternate_values[0], 0.7F, TOLERANCE) << "dequantized weight cache ignored changed scales";
    EXPECT_NEAR(alternate_values[1], -0.025F, TOLERANCE) << "dequantized weight cache ignored changed scales";
}

#if defined(__APPLE__)
TEST_F(PackedPrecisionTest, LowBitParity) {
    using namespace kidi;
    using core::InferencePrecision;
    using ops::require;
    ops::Context lowbit_context;
    lowbit_context.set_precision(InferencePrecision::LOWBIT_PARITY);
    const auto lowbit = lowbit_context.packed_linear(input, weight, scales, 8, 2, 0.1F, 0.1F);
    const auto lowbit_values = require(lowbit.data<float>());
    EXPECT_NEAR(lowbit_values[0], 0.4F, TOLERANCE) << "low-bit parity projection produced unexpected SRQ output";
    EXPECT_NEAR(lowbit_values[1], 0.F, TOLERANCE) << "low-bit parity projection produced unexpected SRQ output";
}
#endif

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
    low_bit_context.set_precision(InferencePrecision::FP32);
    const auto low_bit_output = low_bit_context.packed_linear(
        low_bit_input, low_bit_weight, low_bit_scales, 4, 4, 0.1F, 0.1F);
    EXPECT_NEAR(require(low_bit_output.data<float>())[0], -5.F, TOLERANCE)
        << "FP32 precision did not unpack signed 4-bit weights";
    for (const auto [precision, expected] :
         {std::pair{InferencePrecision::Q2A16, -6.F},
          std::pair{InferencePrecision::Q4A16, -4.857143F}}) {
        ops::Context context;
        context.set_precision(precision);
        const auto output =
            context.packed_linear(low_bit_input, low_bit_weight, low_bit_scales, 4, 4, 0.1F, 0.1F);
        const auto actual = require(output.data<float>())[0];
        EXPECT_NEAR(actual, expected, 0.03F) << "explicit weight precision result differs";
    }
}

TEST_F(PackedPrecisionTest, Fp8Activations) {
    using namespace kidi;
    using core::InferencePrecision;
    using ops::require;
    for (const auto [precision, dtype] :
         {std::pair{InferencePrecision::Q8AE4M3, tensor::DType::E4M3},
          std::pair{InferencePrecision::Q8AE5M2, tensor::DType::E5M2}}) {
        SCOPED_TRACE(core::to_string(precision));
        ops::Context context;
        context.set_precision(precision);
        const auto rounded = context.cast(context.cast(input, dtype), tensor::DType::F32);
        const auto rounded_values = require(rounded.data<float>());
        const auto expected_second = dtype == tensor::DType::E4M3 ? -0.203125F : -0.1875F;
        EXPECT_NEAR(rounded_values[0], 0.15625F, TOLERANCE) << "CPU FP8 conversion produced unexpected values";
        EXPECT_NEAR(rounded_values[1], expected_second, TOLERANCE) << "CPU FP8 conversion produced unexpected values";
        const std::array expected{rounded_values[0] - rounded_values[1],
                                  0.25F * rounded_values[0] + 0.25F * rounded_values[1]};
        const auto output = context.packed_linear(input, weight, scales, 8, 2, 0.1F, 0.1F);
        const auto values = require(output.data<float>());
        EXPECT_NEAR(values[0], expected[0], TOLERANCE) << "FP8 activation precision changed packed-linear result";
        EXPECT_NEAR(values[1], expected[1], TOLERANCE) << "FP8 activation precision changed packed-linear result";
    }
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
