#include "kidi/ops/context.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>

#if defined(KIDI_HAS_CUDA)
#include "cuda/runtime.h"
#endif
#include <gtest/gtest.h>

namespace {
using namespace kidi;

constexpr float TOLERANCE = 1e-5F;

auto make_tensor(std::vector<std::int64_t> shape, std::span<const float> values) -> tensor::Tensor {
    return ops::require(tensor::Tensor::from_host(std::move(shape), values));
}

TEST(FusedCpuTest, NativeBlockScaledProjectionsAndInterleavedGate) {
    using ops::require;
    using tensor::DType;
    using tensor::Tensor;
    std::vector devices{tensor::Device::cpu()};
#if defined(KIDI_HAS_CUDA)
    if (cuda::available()) devices.push_back(tensor::Device::cuda());
#endif
    const auto e4m3 = [](std::uint8_t byte) {
        const int exponent = (byte >> 3) & 15, mantissa = byte & 7;
        const float value = exponent ? std::ldexp(1.F + mantissa / 8.F, exponent - 7) : mantissa / 512.F;
        return byte & 128 ? -value : value;
    };
    const auto bfloat = [](float value) {
        const auto bits = std::bit_cast<std::uint32_t>(value);
        return std::bit_cast<float>((bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000);
    };
    constexpr std::array<float, 16> e2m1{0, .5F, 1, 1.5F, 2, 3, 4, 6, -0.F, -.5F, -1, -1.5F, -2, -3, -4, -6};
    constexpr std::int64_t ROWS = 18;
    for (const auto device : devices)
        for (const auto format :
             {ops::BlockScaledFormat::NVFP4, ops::BlockScaledFormat::MXFP8, ops::BlockScaledFormat::INT4})
            for (const std::int32_t group : {0, 32, 64, 128, 48})
                for (const bool has_global : {false, true})
                    for (const auto dtype : {DType::F32, DType::BF16})
                        for (const std::int64_t tokens : {1, 3, 17, 33}) {
                            const bool nvfp4 = format == ops::BlockScaledFormat::NVFP4;
                            const bool int4 = format == ops::BlockScaledFormat::INT4;
                            const bool packed4 = nvfp4 || int4;
                            SCOPED_TRACE(tensor::to_string(device));
                            SCOPED_TRACE(static_cast<int>(format));
                            SCOPED_TRACE(group);
                            SCOPED_TRACE(has_global);
                            SCOPED_TRACE(tokens);
                            const std::int64_t width = group ? 384 : (int4 ? 256 : (nvfp4 ? 48 : 96));
                            const std::int64_t block = group ? group : (int4 ? 128 : (nvfp4 ? 16 : 32));
                            std::vector<std::uint8_t> values(ROWS * width / (packed4 ? 2 : 1)),
                                scales(ROWS * width / block);
                            std::vector<float> float_scales(scales.size());
                            for (std::size_t index = 0; index < values.size(); ++index)
                                values[index] = packed4 ? static_cast<std::uint8_t>(index * 37)
                                                        : static_cast<std::uint8_t>(0x28 + index % 32);
                            for (std::size_t index = 0; index < scales.size(); ++index) {
                                scales[index] = nvfp4 ? static_cast<std::uint8_t>(0x30 + index % 8)
                                                      : static_cast<std::uint8_t>(125 + index % 4);
                                float_scales[index] = .125F + (index % 4) / 16.F;
                            }
                            std::vector<float> activations(tokens * width), routing(tokens);
                            for (std::size_t index = 0; index < activations.size(); ++index)
                                activations[index] = (static_cast<int>(index % 17) - 8) / 32.F;
                            for (std::size_t index = 0; index < routing.size(); ++index)
                                routing[index] = 1.F + (index % 3) / 4.F;
                            ops::Context context(device, false, false, dtype);
                            ops::Context host;
                            const auto input =
                                require(host.cast(make_tensor({1, tokens, width}, activations), dtype).to(device));
                            const ops::BlockScaledWeight weight{
                                format,
                                require(Tensor::from_bytes({ROWS, width / (packed4 ? 2 : 1)},
                                                           packed4 ? DType::U8 : DType::E4M3,
                                                           std::as_bytes(std::span(values)), device)),
                                int4 ? require(host.cast(make_tensor({ROWS, width / block}, float_scales),
                                                         tokens == 1 ? DType::F32
                                                                     : (tokens == 3 ? DType::F16 : DType::BF16))
                                                   .to(device))
                                     : require(Tensor::from_bytes({ROWS, width / block},
                                                                  nvfp4 && tokens != 3 ? DType::E4M3 : DType::U8,
                                                                  std::as_bytes(std::span(scales)), device)),
                                has_global ? require(make_tensor({1}, std::array{.75F}).to(device)) : Tensor{}, group};
                            const auto linear =
                                host.cast(require(context.block_scaled_linear(input, weight).to(tensor::Device::cpu())),
                                          DType::F32);
                            const auto gated =
                                host.cast(require(context
                                                      .block_scaled_silu(
                                                          input, weight,
                                                          require(make_tensor({tokens}, routing).to(device)), 3.F, 5.F)
                                                      .to(tensor::Device::cpu())),
                                          DType::F32);
                            ASSERT_EQ(gated.size(-1), ROWS / 2);
                            const auto actual = require(linear.data<float>()), activated = require(gated.data<float>());
                            std::vector<float> expected(tokens * ROWS);
                            for (std::int64_t token = 0; token < tokens; ++token)
                                for (std::int64_t row = 0; row < ROWS; ++row) {
                                    float sum = 0;
                                    for (std::int64_t column = 0; column < width; ++column) {
                                        const auto index = row * width + column;
                                        const auto scale = scales[index / block];
                                        const auto byte = values[packed4 ? index / 2 : index];
                                        const int code = column % 2 ? byte >> 4 : byte & 15;
                                        float value = int4 ? (code < 8 ? code : code - 16) * float_scales[index / block]
                                                      : nvfp4 ? e2m1[code] * e4m3(scale)
                                                              : e4m3(byte) * std::ldexp(1.F, scale - 127);
                                        if (has_global) value *= .75F;
                                        if (dtype == DType::BF16) value = bfloat(value);
                                        sum += value * activations[token * width + column];
                                    }
                                    if (dtype == DType::BF16) sum = bfloat(sum);
                                    expected[token * ROWS + row] = sum;
                                    EXPECT_NEAR(actual[token * ROWS + row], sum, dtype == DType::BF16 ? .02F : 1e-4F);
                                }
                            for (std::int64_t token = 0; token < tokens; ++token)
                                for (std::int64_t pair = 0; pair < ROWS / 2; ++pair) {
                                    const auto gate = expected[token * ROWS + pair * 2],
                                               up = expected[token * ROWS + pair * 2 + 1];
                                    float value = std::tanh(gate / 3.F) * 3.F / (1 + std::exp(-gate)) *
                                                  std::tanh(up / 5.F) * 5.F * routing[token];
                                    if (dtype == DType::BF16) value = bfloat(value);
                                    EXPECT_NEAR(activated[token * ROWS / 2 + pair], value,
                                                dtype == DType::BF16 ? .02F : 1e-4F);
                                }
                            EXPECT_EQ(require(weight.values.copy_to_host()),
                                      std::vector<std::byte>(std::as_bytes(std::span(values)).begin(),
                                                             std::as_bytes(std::span(values)).end()));
                            if (int4 && !has_global && dtype == DType::BF16 && tokens <= 3 &&
                                device.kind == tensor::DeviceKind::CUDA) {
                                const auto legacy = host.cast(
                                    require(context.packed_linear(input, weight.values, weight.scales, 4, block)
                                                .to(tensor::Device::cpu())),
                                    DType::F32);
                                const auto legacy_values = require(legacy.data<float>());
                                for (std::size_t index = 0; index < actual.size(); ++index)
                                    EXPECT_EQ(legacy_values[index], actual[index]);
                            }
                        }
}

TEST(FusedCpuTest, NativeBlockScaledRejectsInvalidMetadata) {
    using ops::require;
    using tensor::DType;
    using tensor::Tensor;
    ops::Context context;
    const auto input = require(Tensor::zeros({1, 32}, DType::F32));
    ops::BlockScaledWeight weight{ops::BlockScaledFormat::NVFP4,
                                  require(Tensor::zeros({3, 16}, DType::U8)),
                                  require(Tensor::zeros({3, 1}, DType::E4M3)),
                                  {}};
    EXPECT_THROW(context.block_scaled_linear(input, weight), ops::Failure);
    weight.global = make_tensor({1}, std::array{1.F});
    weight.scales = require(Tensor::zeros({3, 2}, DType::E4M3));
    for (const auto group : {-16, 18, 64}) {
        weight.group_size = group;
        EXPECT_THROW(context.block_scaled_linear(input, weight), ops::Failure);
    }
    weight.group_size = 0;
    EXPECT_THROW(context.block_scaled_silu(input, weight, make_tensor({1}, std::array{1.F}), 3.F, 5.F), ops::Failure);
    weight.format = ops::BlockScaledFormat::MXFP8;
    EXPECT_THROW(context.block_scaled_linear(input, weight), ops::Failure);
}

#if defined(KIDI_HAS_CUDA)
TEST(FusedCpuTest, RoutedExpertProjectionsMatchReference) {
    using ops::require;
    using tensor::DType;
    using tensor::Tensor;
    if (!cuda::available()) GTEST_SKIP() << "CUDA unavailable";
    constexpr std::int64_t EXPERTS = 6, TOKENS = 2, TOP = 3, WIDTH = 64, ROWS = 64, HIDDEN = ROWS / 2, OUTPUTS = 16,
                           GROUP = 16;
    constexpr float SCALE = 2.5F, GATE_CAP = 3.F, UP_CAP = 5.F;
    constexpr std::array<float, 16> e2m1{0, .5F, 1, 1.5F, 2, 3, 4, 6, -0.F, -.5F, -1, -1.5F, -2, -3, -4, -6};
    const auto e4m3 = [](std::uint8_t byte) {
        const int exponent = (byte >> 3) & 15, mantissa = byte & 7;
        return exponent ? std::ldexp(1.F + mantissa / 8.F, exponent - 7) : mantissa / 512.F;
    };
    const auto device = tensor::Device::cuda();
    ops::Context host;
    std::vector<float> logits(TOKENS * EXPERTS), bias(EXPERTS), globals(EXPERTS);
    for (std::size_t index = 0; index < logits.size(); ++index)
        logits[index] = static_cast<float>(static_cast<int>(index * 7 % 11) - 5) / 4.F;
    for (std::int64_t expert = 0; expert < EXPERTS; ++expert) {
        bias[expert] = static_cast<float>(expert % 3) / 8.F;
        globals[expert] = .5F + static_cast<float>(expert) / 8.F;
    }
    // Stacked [experts, rows, columns] NVFP4 weights with E4M3 group scales and one global per expert.
    struct Stack {
        std::vector<std::uint8_t> values, scales;
        std::int64_t rows, columns;
        auto weight(std::span<const float> globals, tensor::Device device) const -> ops::BlockScaledWeight {
            return {ops::BlockScaledFormat::NVFP4,
                    require(Tensor::from_bytes({EXPERTS, rows, columns / 2}, DType::U8, std::as_bytes(std::span(values)),
                                               device)),
                    require(Tensor::from_bytes({EXPERTS, rows, columns / GROUP}, DType::E4M3,
                                               std::as_bytes(std::span(scales)), device)),
                    require(make_tensor({EXPERTS}, globals).to(device)), GROUP};
        }
    };
    const auto stack = [](std::int64_t rows, std::int64_t columns, int seed) {
        Stack result{std::vector<std::uint8_t>(EXPERTS * rows * columns / 2),
                     std::vector<std::uint8_t>(EXPERTS * rows * columns / GROUP), rows, columns};
        for (std::size_t index = 0; index < result.values.size(); ++index)
            result.values[index] = static_cast<std::uint8_t>(index * 37 + seed);
        for (std::size_t index = 0; index < result.scales.size(); ++index)
            result.scales[index] = static_cast<std::uint8_t>(0x30 + (index + seed) % 8);
        return result;
    };
    const auto gate_up = stack(ROWS, WIDTH, 3), down = stack(OUTPUTS, HIDDEN, 5);
    const auto weight_at = [&](const Stack& weights, std::int64_t expert, std::int64_t row, std::int64_t column) {
        const auto index = (expert * weights.rows + row) * weights.columns + column;
        const auto byte = weights.values[index / 2];
        return e2m1[column % 2 ? byte >> 4 : byte & 15] * e4m3(weights.scales[index / GROUP]) * globals[expert];
    };
    std::vector<float> activations(TOKENS * WIDTH);
    for (std::size_t index = 0; index < activations.size(); ++index)
        activations[index] = static_cast<float>(static_cast<int>(index % 13) - 6) / 16.F;

    // Reference routing: sigmoid scores, top-k on biased scores, weights renormalized over the selected scores.
    std::vector<std::int64_t> experts(TOKENS * TOP);
    std::vector<float> routing(TOKENS * TOP);
    for (std::int64_t token = 0; token < TOKENS; ++token) {
        std::vector<float> scores(EXPERTS), biased(EXPERTS);
        for (std::int64_t expert = 0; expert < EXPERTS; ++expert) {
            scores[expert] = 1.F / (1.F + std::exp(-logits[token * EXPERTS + expert]));
            biased[expert] = scores[expert] + bias[expert];
        }
        float total = 0;
        for (std::int64_t slot = 0; slot < TOP; ++slot) {
            const auto best = std::max_element(biased.begin(), biased.end()) - biased.begin();
            experts[token * TOP + slot] = best;
            routing[token * TOP + slot] = scores[best];
            total += scores[best];
            biased[best] = -INFINITY;
        }
        for (std::int64_t slot = 0; slot < TOP; ++slot) routing[token * TOP + slot] *= SCALE / total;
    }

    for (const auto dtype : {DType::F32, DType::BF16}) {
        SCOPED_TRACE(tensor::to_string(dtype));
        ops::Context context(device, false, false, dtype);
        const auto routes = context.route_experts(require(make_tensor({TOKENS, EXPERTS}, logits).to(device)),
                                                  require(make_tensor({EXPERTS}, bias).to(device)), TOP, SCALE);
        const auto host_routes = require(routes.to(tensor::Device::cpu()));
        const auto route_values = require(host_routes.data<float>());
        for (std::int64_t index = 0; index < TOKENS * TOP; ++index) {
            EXPECT_EQ(route_values[index * 2], static_cast<float>(experts[index]));
            EXPECT_NEAR(route_values[index * 2 + 1], routing[index], 1e-5F);
        }
        const auto input = require(host.cast(make_tensor({TOKENS, WIDTH}, activations), dtype).to(device));
        const auto hidden = context.expert_silu(input, gate_up.weight(globals, device), routes, GATE_CAP, UP_CAP);
        ASSERT_TRUE(std::ranges::equal(hidden.shape(), std::array{TOKENS, TOP, HIDDEN}));
        const auto output = context.expert_linear(hidden, down.weight(globals, device), routes);
        ASSERT_TRUE(std::ranges::equal(output.shape(), std::array{TOKENS, OUTPUTS}));
        const auto hidden_values = host.cast(require(hidden.to(tensor::Device::cpu())), DType::F32);
        const auto actual_hidden = require(hidden_values.data<float>());
        const auto host_output = require(output.to(tensor::Device::cpu()));
        const auto actual = require(host_output.data<float>());
        const float tolerance = dtype == DType::BF16 ? .03F : 1e-4F;
        for (std::int64_t token = 0; token < TOKENS; ++token) {
            std::vector<float> expected(OUTPUTS);
            for (std::int64_t slot = 0; slot < TOP; ++slot) {
                const auto route = token * TOP + slot;
                const auto expert = experts[route];
                for (std::int64_t pair = 0; pair < HIDDEN; ++pair) {
                    float gate = 0, up = 0;
                    for (std::int64_t column = 0; column < WIDTH; ++column) {
                        const auto x = activations[token * WIDTH + column];
                        gate += weight_at(gate_up, expert, 2 * pair, column) * x;
                        up += weight_at(gate_up, expert, 2 * pair + 1, column) * x;
                    }
                    const float value = std::tanh(gate / GATE_CAP) * GATE_CAP / (1 + std::exp(-gate)) *
                                        std::tanh(up / UP_CAP) * UP_CAP * routing[route];
                    EXPECT_NEAR(actual_hidden[route * HIDDEN + pair], value,
                                tolerance * std::max(1.F, std::abs(value)));
                    // Down projection consumes the kernel's own activations so only its error is measured here.
                    for (std::int64_t row = 0; row < OUTPUTS; ++row)
                        expected[row] += weight_at(down, expert, row, pair) * actual_hidden[route * HIDDEN + pair];
                }
            }
            for (std::int64_t row = 0; row < OUTPUTS; ++row)
                EXPECT_NEAR(actual[token * OUTPUTS + row], expected[row], 1e-3F * std::max(1.F, std::abs(expected[row])));
        }
    }
}
#endif

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
