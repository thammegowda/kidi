#include "kidi/ops/context.h"

#include <array>
#include <cmath>

#include <gtest/gtest.h>

namespace {
using namespace kidi;

constexpr float TOLERANCE = 1e-5F;

auto make_tensor(std::vector<std::int64_t> shape, std::span<const float> values) -> tensor::Tensor {
    return ops::require(tensor::Tensor::from_host(std::move(shape), values));
}

TEST(FusedCpuTest, UnweightedRmsNorm) {
    ops::Context context;
    const std::array<float, 2> norm_values{3.F, 4.F};
    const auto normalized_tensor = context.rms_norm(make_tensor({1, 2}, norm_values), {}, 0.01F);
    const auto normalized = ops::require(normalized_tensor.data<float>());
    const auto inverse = 1.F / std::sqrt(12.51F);
    EXPECT_NEAR(normalized[0], 3.F * inverse, TOLERANCE) << "unweighted RMS norm mismatch";
    EXPECT_NEAR(normalized[1], 4.F * inverse, TOLERANCE) << "unweighted RMS norm mismatch";
}

TEST(FusedCpuTest, ResidualMix) {
    ops::Context context;
    const std::array<float, 4> first_values{1.F, 0.F, 0.F, 1.F};
    const std::array<float, 4> second_values{0.F, 1.F, 1.F, 0.F};
    const std::array<float, 2> query_values{0.F, 0.F};
    const std::array sources{make_tensor({2, 2}, first_values), make_tensor({2, 2}, second_values)};
    const auto mixed = context.residual_mix(sources, make_tensor({1, 2}, query_values), 1e-6F);
    ASSERT_EQ(mixed.shape()[0], 1) << "residual mix shape mismatch";
    ASSERT_EQ(mixed.shape()[1], 2) << "residual mix shape mismatch";
    ASSERT_EQ(mixed.shape()[2], 2) << "residual mix shape mismatch";
    const auto mixed_values = ops::require(mixed.data<float>());
    constexpr float normalized_half = 0.70710678F;
    for (std::size_t index = 0; index < 4; ++index)
        EXPECT_NEAR(mixed_values[index], normalized_half, TOLERANCE) << "residual mix values mismatch at " << index;
}

TEST(FusedCpuTest, ShortConvolution) {
    ops::Context context;
    const std::array<float, 3> padded_values{1.F, 2.F, 3.F};
    const std::array<float, 2> filter_values{0.1F, 0.2F};
    const auto convolution =
        context.short_convolution(make_tensor({3, 1}, padded_values), make_tensor({2, 1}, filter_values), {}, 1.F);
    const auto convolution_values = ops::require(convolution.data<float>());
    EXPECT_NEAR(convolution_values[0], 2.4F, TOLERANCE) << "short convolution mismatch";
    EXPECT_NEAR(convolution_values[1], 3.7F, TOLERANCE) << "short convolution mismatch";
}

TEST(FusedCpuTest, L2Clip) {
    ops::Context context;
    const std::array<float, 2> clip_values{3.F, 4.F};
    const auto clipped = context.l2_clip(make_tensor({1, 2}, clip_values), {}, 2.F);
    const auto clipped_values = ops::require(clipped.data<float>());
    EXPECT_NEAR(clipped_values[0], 1.2F, TOLERANCE) << "L2 clip mismatch";
    EXPECT_NEAR(clipped_values[1], 1.6F, TOLERANCE) << "L2 clip mismatch";
}

TEST(FusedCpuTest, RelativeAttention) {
    ops::Context context;
    const std::array<float, 1> zero{0.F};
    const std::array<float, 2> key_values{0.F, 0.F}, value_values{10.F, 20.F};
    const auto attention = context.relative_attention(
        make_tensor({1, 1}, zero), make_tensor({2, 1}, key_values), make_tensor({2, 1}, value_values),
        make_tensor({1, 1, 1}, zero), make_tensor({1, 1}, zero), make_tensor({1, 1}, zero),
        {.heads = 1, .key_heads = 1, .start = 1, .window = 0, .extent = 1});
    EXPECT_NEAR(ops::require(attention.data<float>())[0], 5.F, TOLERANCE) << "relative attention mismatch";
}

TEST(FusedCpuTest, SoftcapSiluGate) {
    ops::Context context;
    const std::array<float, 2> gate_up_values{1.F, 2.F};
    const std::array<float, 1> row_weight{0.5F};
    const auto gated =
        context.silu_gate(make_tensor({1, 2}, gate_up_values), 10.F, 10.F, make_tensor({1}, row_weight));
    const auto gate_value = 10.F * std::tanh(0.1F);
    const auto up_value = 10.F * std::tanh(0.2F);
    const auto expected_gate = gate_value / (1.F + std::exp(-gate_up_values[0])) * up_value * 0.5F;
    EXPECT_NEAR(ops::require(gated.data<float>())[0], expected_gate, TOLERANCE) << "softcap SiLU gate mismatch";
}
} // namespace
