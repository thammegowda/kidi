#include "kidi/layers/audio.h"
#include "kidi/layers/higgs_audio.h"

#include <array>
#include <cmath>

#include <gtest/gtest.h>

namespace {

constexpr float TOLERANCE = 1e-2F;

auto set_quantized(kidi::Module& module, std::int8_t weight, float scale, std::int8_t bias = 0, float bias_scale = 1.F)
    -> void {
    for (auto& [name, tensor] : module.state_dict()) {
        if (tensor.dtype() == kidi::tensor::DType::I8) {
            auto values = tensor.data<std::int8_t>();
            ASSERT_TRUE(values) << name;
            std::ranges::fill(*values, name.ends_with("bias") ? bias : weight);
        } else {
            auto values = tensor.data<float>();
            ASSERT_TRUE(values) << name;
            std::ranges::fill(*values, name.ends_with("bias_scale") ? bias_scale : scale);
        }
    }
}

auto higgs_tokens(kidi::ops::Context& context, std::vector<float>& expected) -> void {
    using namespace kidi;
    layers::HiggsDecoder decoder(layers::HiggsDecoderShape{2, 4, 2, 2, 2, 4, {2}});
    ASSERT_NO_FATAL_FAILURE(set_quantized(*decoder, 1, 0.5F));
    for (auto& [name, tensor] : decoder->state_dict())
        if (name.ends_with("codebook.weight")) {
            auto values = ops::require(tensor.data<std::int8_t>());
            for (std::size_t index = 0; index < values.size(); ++index)
                values[index] = static_cast<std::int8_t>(index + 1);
        }
    constexpr std::array<std::int32_t, 4> CODES{0, 1, 2, 3};
    const auto tokens = ops::require(
        ops::require(tensor::Tensor::from_host({2, 2}, std::span<const std::int32_t>(CODES))).to(context.device()));
    const auto output = decoder->forward(context, tokens);
    context.synchronize();
    const auto values = ops::require(output.data<float>());
    expected.assign(values.begin(), values.end());
    ASSERT_EQ(expected.size(), 4U);
    EXPECT_TRUE(std::ranges::all_of(expected, [](float value) { return std::isfinite(value); }));

    constexpr std::array<std::int32_t, 6> PADDED{9, 9, 0, 1, 2, 3};
    const auto padded = ops::require(
        ops::require(tensor::Tensor::from_host({3, 2}, std::span<const std::int32_t>(PADDED))).to(context.device()));
    const auto view = ops::require(padded.narrow(0, 1, 2));
    const auto replayed = decoder->forward(context, view);
    context.synchronize();
    const auto actual = ops::require(replayed.data<float>());
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t index = 0; index < actual.size(); ++index)
        EXPECT_LE(std::abs(actual[index] - expected[index]), 1e-5F * std::max(1.F, std::abs(expected[index])))
            << "index " << index;

    const auto rejects = [&](const tensor::Tensor& input) {
        try {
            (void)decoder->forward(context, input);
            return false;
        } catch (const ops::Failure& error) {
            return error.error().code == ErrorCode::INVALID_ARGUMENT;
        }
    };
    EXPECT_TRUE(rejects(ops::require(tokens.reshape({4}))));
    constexpr std::array<std::int32_t, 4> INVALID{0, 4, 1, 2};
    const auto invalid = ops::require(
        ops::require(tensor::Tensor::from_host({2, 2}, std::span<const std::int32_t>(INVALID))).to(context.device()));
    EXPECT_TRUE(rejects(invalid));
}

class AudioTest : public ::testing::Test {
protected:
    const kidi::ModuleScope scope{kidi::tensor::DType::I8, true, kidi::tensor::Device::cpu()};
    kidi::ops::Context context{kidi::tensor::Device::cpu()};
    const std::array<float, 3> input_values{1.F, 2.F, 3.F};
};

TEST_F(AudioTest, Conv1d) {
    using namespace kidi;
    layers::Conv1d convolution(1, 1, 3, 1, 1);
    ASSERT_NO_FATAL_FAILURE(set_quantized(*convolution, 1, 1.F, 1, 1.F));
    auto input = tensor::Tensor::from_host({1, 3, 1}, std::span<const float>(input_values));
    ASSERT_TRUE(input);
    const auto convolved = convolution->forward(context, *input);
    context.synchronize();
    const auto convolved_values = convolved.data<float>();
    ASSERT_TRUE(convolved_values);
    ASSERT_EQ(convolved_values->size(), 3U);
    EXPECT_NEAR((*convolved_values)[0], 4.F, TOLERANCE);
    EXPECT_NEAR((*convolved_values)[1], 7.F, TOLERANCE);
    EXPECT_NEAR((*convolved_values)[2], 6.F, TOLERANCE);
}

TEST_F(AudioTest, ConvTranspose1d) {
    using namespace kidi;
    layers::ConvTranspose1d transposed(1, 1, 4, 2, 1);
    ASSERT_NO_FATAL_FAILURE(set_quantized(*transposed, 1, 1.F));
    const std::array transposed_input_values{1.F, 2.F};
    auto transposed_input = tensor::Tensor::from_host({1, 2, 1}, std::span<const float>(transposed_input_values));
    ASSERT_TRUE(transposed_input);
    const auto expanded = transposed->forward(context, *transposed_input);
    const auto expanded_values = expanded.data<float>();
    const std::array expected{1.F, 3.F, 3.F, 2.F};
    ASSERT_TRUE(expanded_values);
    ASSERT_EQ(expanded_values->size(), expected.size());
    for (std::size_t index = 0; index < expected.size(); ++index)
        EXPECT_NEAR((*expanded_values)[index], expected[index], TOLERANCE) << "index " << index;
}

TEST_F(AudioTest, Snake1d) {
    using namespace kidi;
    auto input = tensor::Tensor::from_host({1, 3, 1}, std::span<const float>(input_values));
    ASSERT_TRUE(input);
    layers::Snake1d snake(1);
    ASSERT_NO_FATAL_FAILURE(set_quantized(*snake, 1, 1.F));
    const auto activated = snake->forward(context, *input);
    const auto activated_values = activated.data<float>();
    ASSERT_TRUE(activated_values);
    ASSERT_EQ(activated_values->size(), input_values.size());
    for (std::size_t index = 0; index < input_values.size(); ++index)
        EXPECT_NEAR((*activated_values)[index],
                    input_values[index] + std::sin(input_values[index]) * std::sin(input_values[index]), TOLERANCE)
            << "index " << index;
}

TEST_F(AudioTest, HiggsDecoder) {
    std::vector<float> cpu_higgs;
    ASSERT_NO_FATAL_FAILURE(higgs_tokens(context, cpu_higgs));
#if defined(__APPLE__)
    {
        using namespace kidi;
        const ModuleScope metal_scope(tensor::DType::I8, true, tensor::Device::apple_gpu());
        ops::Context metal(tensor::Device::apple_gpu());
        std::vector<float> metal_higgs;
        ASSERT_NO_FATAL_FAILURE(higgs_tokens(metal, metal_higgs));
        ASSERT_EQ(metal_higgs.size(), cpu_higgs.size());
        for (std::size_t index = 0; index < cpu_higgs.size(); ++index)
            EXPECT_LE(std::abs(metal_higgs[index] - cpu_higgs[index]), 1e-2F * std::max(1.F, std::abs(cpu_higgs[index])))
                << "index " << index;
    }
#endif
}

} // namespace
