#include "kidi/graph/compiler.h"
#include "kidi/ops/context.h"
#include "kidi/runtime/operator.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <span>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>

namespace {
using kidi::ops::require;
using kidi::tensor::DType;
using kidi::tensor::Tensor;

auto tensor(std::span<const float> values, std::initializer_list<std::int64_t> shape) -> Tensor {
    return require(Tensor::from_host({shape.begin(), shape.end()}, values));
}

auto packed(std::span<const std::int8_t> values, std::int64_t rows, std::int64_t columns, int bits) -> Tensor {
    const auto per_byte = 8 / bits;
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(rows * columns / per_byte), 0);
    const auto mask = (1 << bits) - 1;
    for (std::size_t index = 0; index < values.size(); ++index)
        bytes[index / per_byte] |= static_cast<std::uint8_t>((values[index] & mask) << ((index % per_byte) * bits));
    return require(Tensor::from_host({rows, columns / per_byte}, std::span<const std::uint8_t>(bytes)));
}

TEST(QnnCompilerTest, ReplayMatchesEager) {
    setenv("KIDI_BACKGROUND_COMPILE", "0", 1);
    auto compiler = kidi::runtime::npu_step_compiler();
    if (!compiler) GTEST_SKIP() << "skipping QNN step compiler test: " << compiler.error().message;
    kidi::ops::StepCompilerScope scope(*compiler);
    kidi::ops::Context context(kidi::tensor::Device::cpu());
    auto input = tensor(std::array{0.25F, -0.5F, 0.75F, 1.0F}, {1, 1, 4});
    const std::array<std::int8_t, 16> gate_values{1, -2, 3, 1, -1, 2, -3, 1, 2, 1, -1, 3, -2, 1, 2, -1};
    const std::array<float, 4> gate_scales{0.05F, 0.04F, 0.03F, 0.02F};
    const std::array<std::int8_t, 8> down_values{2, -1, 1, 2, -2, 1, 3, -1};
    const std::array<float, 4> down_scales{0.06F, 0.05F, 0.04F, 0.03F};
    auto gate_weight = packed(gate_values, 4, 4, 4);
    auto gate_scale = tensor(gate_scales, {4, 1});
    auto down_weight = packed(down_values, 4, 2, 4);
    auto down_scale = tensor(down_scales, {4, 1});
    const auto step = [&](std::span<const Tensor> operands) {
        auto projected = context.packed_linear(operands[0], gate_weight, gate_scale, 4, 4, 0.1F, 0.05F);
        auto gate = context.slice(projected, -1, 0, 2);
        auto up = context.slice(projected, -1, 2, 2);
        auto hidden = context.gelu_multiply(gate, up);
        return std::vector{context.packed_linear(hidden, down_weight, down_scale, 4, 2, 0.02F, 0.05F)};
    };
    const std::array inputs{input};
    auto eager = step(inputs);
    constexpr std::string_view key = "gemma4_decode:1:1:512:0:logits";
    (void)context.replay(key, inputs, step);
    (void)context.replay(key, inputs, step);
    auto accelerated = context.replay(key, inputs, step);
    const auto expected = require(eager[0].data<float>());
    const auto actual = require(accelerated[0].data<float>());
    ASSERT_EQ(expected.size(), actual.size());
    for (std::size_t index = 0; index < actual.size(); ++index) {
        EXPECT_TRUE(std::isfinite(actual[index])) << "QNN output not finite at " << index;
        EXPECT_LE(std::abs(actual[index] - expected[index]), 0.12F)
            << "QNN output mismatch at " << index << ": " << actual[index] << " != " << expected[index];
    }
}
} // namespace
