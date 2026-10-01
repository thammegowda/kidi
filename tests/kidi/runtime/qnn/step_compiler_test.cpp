#include "kidi/ops/context.h"
#include "kidi/runtime/operator.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <span>
#include <string_view>
#include <vector>

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
} // namespace

auto main() -> int {
    try {
        setenv("KIDI_BACKGROUND_COMPILE", "0", 1);
        auto compiler = kidi::runtime::npu_step_compiler();
        if (!compiler) {
            std::cout << "skipping QNN step compiler test: " << compiler.error().message << '\n';
            return 0;
        }
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
        if (expected.size() != actual.size()) return 1;
        for (std::size_t index = 0; index < actual.size(); ++index)
            if (!std::isfinite(actual[index]) || std::abs(actual[index] - expected[index]) > 0.12F) {
                std::cerr << "QNN output mismatch at " << index << ": " << actual[index] << " != " << expected[index]
                          << '\n';
                return 1;
            }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
