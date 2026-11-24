#include "kidi/ops/context.h"
#include "kidi/ops/cpu_kernels.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <numeric>
#include <random>

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
    const auto bf16 = context.cast(make_tensor({1, 2}, norm_values), tensor::DType::BF16);
    const auto bf16_normalized = context.rms_norm(bf16, {}, 0.01F);
    EXPECT_EQ(bf16_normalized.dtype(), tensor::DType::BF16);
    const auto actual = context.cast(bf16_normalized, tensor::DType::F32);
    const auto expected = context.cast(context.cast(normalized_tensor, tensor::DType::BF16), tensor::DType::F32);
    EXPECT_TRUE(std::ranges::equal(ops::require(actual.data<float>()), ops::require(expected.data<float>())));
}

TEST(FusedCpuTest, FastInt8PackingMatchesGeneric) {
    using ops::require;
    using tensor::DType;
    constexpr std::int64_t ROWS = 65, WIDTH = 67;
    std::vector<float> data(ROWS * WIDTH);
    for (std::size_t i = 0; i < data.size(); ++i)
        data[i] = static_cast<float>(static_cast<int>(i * 37 % 509) - 254) / 8.F;
    std::fill_n(data.begin(), WIDTH, 0.F);
    data[WIDTH] = 127.F;
    data[WIDTH + 1] = .5F;
    data[WIDTH + 2] = -.5F;
    ops::Context context;
    for (const auto dtype : {DType::F32, DType::BF16}) {
        const auto input = context.cast(make_tensor({ROWS, WIDTH}, data), dtype);
        const auto expected = require(ops::pack_weight(input, 8, WIDTH));
        const auto actual = require(ops::pack_rows_int8(input));
        EXPECT_EQ(require(actual.values.copy_to_host()), require(expected.values.copy_to_host()));
        EXPECT_EQ(require(actual.scales.copy_to_host()), require(expected.scales.copy_to_host()));
    }
    data[0] = std::numeric_limits<float>::infinity();
    EXPECT_FALSE(ops::pack_rows_int8(make_tensor({ROWS, WIDTH}, data)));
    data[0] = std::numeric_limits<float>::quiet_NaN();
    EXPECT_FALSE(ops::pack_rows_int8(make_tensor({ROWS, WIDTH}, data)));
    data.assign(ROWS * WIDTH, 0.F);
    data[0] = std::numeric_limits<float>::denorm_min();
    EXPECT_FALSE(ops::pack_rows_int8(make_tensor({ROWS, WIDTH}, data)));
}

TEST(FusedCpuTest, PackCheckpointExpertsToInt8) {
    using ops::require;
    using tensor::DType;
    using tensor::Tensor;
    constexpr std::int64_t EXPERTS = 2, ROWS = 6, WIDTH = 64;
    ops::Context context;
    for (const bool nvfp4 : {false, true}) {
        SCOPED_TRACE(nvfp4);
        const auto block = nvfp4 ? 16 : 32;
        auto values = require(Tensor::empty({EXPERTS, ROWS, WIDTH / (nvfp4 ? 2 : 1)},
                                           nvfp4 ? DType::U8 : DType::E4M3));
        auto scales = require(Tensor::empty({EXPERTS, ROWS, WIDTH / block}, DType::U8));
        auto bytes = require(values.host_bytes());
        const auto scale_bytes = require(scales.data<std::uint8_t>());
        for (std::size_t index = 0; index < bytes.size(); ++index) {
            const auto row = index / values.size(2);
            bytes[index] = row == EXPERTS * ROWS - 1
                               ? std::byte{0}
                               : std::byte(nvfp4 ? (row % 2 ? 0xCA : 0x42) : (row % 2 ? 0xC0 : 0x38));
        }
        std::ranges::fill(scale_bytes, nvfp4 ? 0x38 : 127);
        const std::array<float, 2> factors{0.5F, 2.F};
        const auto global = nvfp4 ? make_tensor({EXPERTS}, factors) : Tensor{};
        for (const bool interleaved : {false, true}) {
            const auto packed = context.pack_experts(values, scales, global, interleaved);
            ASSERT_EQ(packed.bits, 8);
            ASSERT_EQ(packed.group_size, WIDTH);
            const auto codes = require(packed.values.data<std::uint8_t>());
            const auto steps = require(packed.scales.data<float>());
            for (std::int64_t expert = 0; expert < EXPERTS; ++expert)
                for (std::int64_t row = 0; row < ROWS; ++row)
                    for (std::int64_t column = 0; column < WIDTH; ++column) {
                        const auto source_row = interleaved ? (row % 3) * 2 + row / 3 : row;
                        const auto sign = source_row % 2 ? -1.F : 1.F;
                        const auto expected =
                            expert == EXPERTS - 1 && source_row == ROWS - 1
                                ? 0.F
                                : nvfp4 ? sign * (column % 2 ? 2.F : 1.F) * factors[expert]
                                         : (source_row % 2 ? -2.F : 1.F);
                        const auto step = steps[expert * ROWS + row];
                        const auto code = std::bit_cast<std::int8_t>(codes[(expert * ROWS + row) * WIDTH + column]);
                        EXPECT_NEAR(code * step, expected, step * 0.501F);
                    }
        }
        if (nvfp4) EXPECT_THROW(context.pack_experts(values, scales, {}, true), ops::Failure);
        bytes[0] = std::byte{0x7F};
        if (nvfp4)
            scale_bytes[0] = 0x7F;
        EXPECT_THROW(context.pack_experts(values, scales, global, true), ops::Failure);
    }
}

TEST(FusedCpuTest, RingConvolutionUpdatesAfterReadingHistory) {
    using ops::require;
    using tensor::Tensor;
    constexpr std::int64_t WIDTH = 4, TOKENS = 7, CHANNELS = 2;
    ops::Context context;
    const std::array<float, 8> filters{0.1F, 0.2F, -0.3F, 0.4F, 0.5F, -0.6F, 0.7F, 0.8F};
    const auto weight = make_tensor({WIDTH, CHANNELS}, filters);
    std::vector<float> input(TOKENS * CHANNELS), padded((TOKENS + WIDTH - 1) * CHANNELS);
    std::iota(input.begin(), input.end(), 1.F);
    std::ranges::copy(input, padded.begin() + (WIDTH - 1) * CHANNELS);
    const auto reference = context.short_convolution(
        make_tensor({TOKENS + WIDTH - 1, CHANNELS}, padded), weight, {}, 0.5F);
    const auto expected = require(reference.data<float>());
    auto state = require(Tensor::zeros({1, WIDTH, CHANNELS}, tensor::DType::F32));
    const auto ring = require(state.reshape({WIDTH, CHANNELS}));
    const auto values = make_tensor({TOKENS, CHANNELS}, input);
    const auto initial_slots = require(Tensor::from_host({3}, std::span<const std::int32_t>(
                                                                std::array<std::int32_t, 3>{0, 1, 2})));
    context.scatter_unique_(state, require(require(values.narrow(0, 0, 3)).reshape({1, 3, CHANNELS})), initial_slots);
    for (std::int64_t token = 3; token < TOKENS; ++token) {
        const std::array<std::int32_t, 1> slot{static_cast<std::int32_t>(token % WIDTH)};
        const auto position = require(Tensor::from_host({1}, std::span<const std::int32_t>(slot)));
        const auto row = require(values.narrow(0, token, 1));
        const auto output = context.short_convolution(row, ring, weight, {}, position, 0.5F, row, {}, true);
        const auto actual = require(output.data<float>());
        for (std::int64_t channel = 0; channel < CHANNELS; ++channel)
            EXPECT_NEAR(actual[channel], expected[token * CHANNELS + channel] + input[token * CHANNELS + channel],
                        TOLERANCE);
    }
    const auto history = require(state.data<float>());
    for (std::int64_t token = 3; token < TOKENS; ++token)
        for (std::int64_t channel = 0; channel < CHANNELS; ++channel)
            EXPECT_EQ(history[(token % WIDTH) * CHANNELS + channel], input[token * CHANNELS + channel]);
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

TEST(FusedCpuTest, GroupedRelativeAttention) {
    ops::Context context;
    const std::array<float, 4> zero{};
    const std::array<float, 4> value{10.F, 20.F, 30.F, 40.F};
    const auto output = context.relative_attention(
        make_tensor({1, 4}, zero), make_tensor({2, 2}, zero), make_tensor({2, 2}, value),
        make_tensor({1, 4, 1}, zero), make_tensor({1, 4}, zero), make_tensor({1, 4}, zero),
        {.heads = 4, .key_heads = 2, .start = 1, .window = 0, .extent = 1});
    const auto actual = ops::require(output.data<float>());
    for (std::size_t head = 0; head < 4; ++head)
        EXPECT_NEAR(actual[head], head < 2 ? 20.F / 3.F : 10.F, TOLERANCE);
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
    const auto bf16_input = context.cast(make_tensor({1, 2}, gate_up_values), tensor::DType::BF16);
    const auto bf16_gated = context.silu_gate(bf16_input, 10.F, 10.F, make_tensor({1}, row_weight));
    EXPECT_EQ(bf16_gated.dtype(), tensor::DType::BF16);
    const auto actual = context.cast(bf16_gated, tensor::DType::F32);
    const auto expected = context.cast(context.cast(gated, tensor::DType::BF16), tensor::DType::F32);
    EXPECT_EQ(ops::require(actual.data<float>())[0], ops::require(expected.data<float>())[0]);
}
TEST(FusedCpuTest, ExpertDecodeMatchesFloatReference) {
    // 128 and 64 are multiples of 64 (VNNI path); 96 exercises the portable path and row tails.
    for (const auto [hidden, intermediate] : {std::pair<std::int64_t, std::int64_t>{128, 64}, {96, 40}}) {
        std::mt19937 random(7);
        std::uniform_int_distribution<int> code(-127, 127);
        std::uniform_real_distribution<float> uniform(-1.F, 1.F);
        constexpr int EXPERTS = 3;
        constexpr float GATE_CAP = 4.F, UP_CAP = 6.F;
        std::vector<float> input(static_cast<std::size_t>(hidden));
        for (auto& value : input) value = uniform(random);
        std::vector<std::vector<std::int8_t>> gate_up(EXPERTS), down(EXPERTS);
        std::vector<std::vector<float>> gate_up_scales(EXPERTS), down_scales(EXPERTS);
        std::vector<ops::cpu::ExpertRows> rows;
        for (int e = 0; e < EXPERTS; ++e) {
            gate_up[e].resize(static_cast<std::size_t>(2 * intermediate * hidden));
            down[e].resize(static_cast<std::size_t>(hidden * intermediate));
            for (auto& value : gate_up[e]) value = static_cast<std::int8_t>(code(random));
            for (auto& value : down[e]) value = static_cast<std::int8_t>(code(random));
            gate_up_scales[e].assign(static_cast<std::size_t>(2 * intermediate), 0.002F);
            down_scales[e].assign(static_cast<std::size_t>(hidden), 0.003F);
            rows.push_back({gate_up[e].data(), gate_up_scales[e].data(), down[e].data(), down_scales[e].data(),
                            0.25F * static_cast<float>(e + 1)});
        }
        std::vector<float> expected(static_cast<std::size_t>(hidden), 0.F);
        for (int e = 0; e < EXPERTS; ++e) {
            std::vector<float> activated(static_cast<std::size_t>(intermediate));
            for (std::int64_t i = 0; i < intermediate; ++i) {
                float g = 0, u = 0;
                for (std::int64_t k = 0; k < hidden; ++k) {
                    g += input[k] * gate_up[e][i * hidden + k] * gate_up_scales[e][i];
                    u += input[k] * gate_up[e][(intermediate + i) * hidden + k] * gate_up_scales[e][intermediate + i];
                }
                activated[i] = GATE_CAP * std::tanh(g / GATE_CAP) / (1.F + std::exp(-g)) * UP_CAP *
                               std::tanh(u / UP_CAP) * rows[e].route;
            }
            for (std::int64_t r = 0; r < hidden; ++r)
                for (std::int64_t i = 0; i < intermediate; ++i)
                    expected[r] += activated[i] * down[e][r * intermediate + i] * down_scales[e][r];
        }
        std::vector<float> actual(static_cast<std::size_t>(hidden));
        ASSERT_TRUE(
            ops::cpu::expert_decode(input.data(), rows, hidden, intermediate, GATE_CAP, UP_CAP, actual.data()));
        float error = 0, norm = 0;
        for (std::int64_t r = 0; r < hidden; ++r) {
            error += (actual[r] - expected[r]) * (actual[r] - expected[r]);
            norm += expected[r] * expected[r];
        }
        // INT8 activations add ~1% quantization noise relative to the float reference.
        EXPECT_LT(std::sqrt(error / norm), 0.03F) << "hidden " << hidden;
    }
}
TEST(FusedCpuTest, Int8LinearMatchesFloatReference) {
    // K=16 with many tokens mirrors the tiny blend projections; K=4608 crosses several work items.
    for (const auto [tokens, k, n] : {std::array<std::int64_t, 3>{24, 16, 40}, {3, 4608, 72}, {3072, 16, 40}}) {
        std::mt19937 random(11);
        std::uniform_int_distribution<int> code(-127, 127);
        std::uniform_real_distribution<float> uniform(-1.F, 1.F);
        std::vector<float> input(static_cast<std::size_t>(tokens * k)), scales(static_cast<std::size_t>(n));
        std::vector<std::int8_t> weight(static_cast<std::size_t>(n * k));
        for (auto& value : input) value = uniform(random);
        for (auto& value : weight) value = static_cast<std::int8_t>(code(random));
        for (auto& value : scales) value = 0.001F + 0.01F * std::abs(uniform(random));
        std::vector<std::uint16_t> output(static_cast<std::size_t>(tokens * n));
        ASSERT_TRUE(ops::cpu::int8_linear(input.data(), tokens, k, weight.data(), scales.data(), n, output.data()));
        float error = 0, norm = 0;
        for (std::int64_t t = 0; t < tokens; ++t)
            for (std::int64_t j = 0; j < n; ++j) {
                float expected = 0;
                for (std::int64_t i = 0; i < k; ++i) expected += input[t * k + i] * weight[j * k + i];
                expected *= scales[j];
                const auto actual = std::bit_cast<float>(std::uint32_t{output[t * n + j]} << 16);
                error += (actual - expected) * (actual - expected);
                norm += expected * expected;
            }
        EXPECT_LT(std::sqrt(error / norm), 0.02F) << "k " << k;
    }
}

TEST(FusedCpuTest, BatchedExpertsMatchDecode) {
    // Expert 0 sees more tokens than one 16-token AMX tile so tiled prefill covers partial and full groups.
    constexpr std::int64_t TOKENS = 21, HIDDEN = 128, INTERMEDIATE = 64;
    std::mt19937 random(13);
    std::uniform_int_distribution<int> code(-127, 127);
    std::uniform_real_distribution<float> uniform(-1.F, 1.F);
    std::vector<float> input(TOKENS * HIDDEN);
    for (auto& x : input) x = uniform(random);
    std::array<std::vector<std::int8_t>, 3> gate_up, down;
    std::vector<float> up_scales(2 * INTERMEDIATE, 0.002F), down_scales(HIDDEN, 0.003F);
    std::array<std::vector<std::int32_t>, 3> tokens{{{}, {1, 2, 3}, {0, 2, 20}}};
    for (std::int32_t t = 0; t < 19; ++t) tokens[0].push_back(t == 4 ? 20 : t);
    std::array<std::vector<float>, 3> routes;
    for (std::size_t e = 0; e < 3; ++e)
        for (std::size_t r = 0; r < tokens[e].size(); ++r)
            routes[e].push_back(0.1F + 0.05F * static_cast<float>((r + e) % 7));
    std::vector<ops::cpu::RoutedExpert> experts;
    for (std::size_t e = 0; e < 3; ++e) {
        gate_up[e].resize(2 * INTERMEDIATE * HIDDEN);
        down[e].resize(HIDDEN * INTERMEDIATE);
        for (auto& x : gate_up[e]) x = static_cast<std::int8_t>(code(random));
        for (auto& x : down[e]) x = static_cast<std::int8_t>(code(random));
        experts.push_back({{gate_up[e].data(), up_scales.data(), down[e].data(), down_scales.data(), 0.F},
                           tokens[e], routes[e]});
    }
    std::vector<float> actual(TOKENS * HIDDEN), expected(HIDDEN);
    ASSERT_TRUE(ops::cpu::expert_prefill(input.data(), TOKENS, experts, HIDDEN, INTERMEDIATE, 4.F, 6.F, actual.data()));
    for (std::int32_t t = 0; t < TOKENS; ++t) {
        std::vector<ops::cpu::ExpertRows> chosen;
        for (const auto& e : experts)
            for (std::size_t r = 0; r < e.tokens.size(); ++r)
                if (e.tokens[r] == t) {
                    auto weight = e.weight;
                    weight.route = e.routes[r];
                    chosen.push_back(weight);
                }
        ASSERT_TRUE(ops::cpu::expert_decode(input.data() + t * HIDDEN, chosen, HIDDEN, INTERMEDIATE,
                                           4.F, 6.F, expected.data()));
        for (std::int64_t c = 0; c < HIDDEN; ++c) EXPECT_NEAR(actual[t * HIDDEN + c], expected[c], 1e-5);
    }
}

TEST(FusedCpuTest, Int8CacheUpdatesAndAttention) {
    using ops::require;
    using tensor::Tensor;
    using tensor::DType;
    ops::Context context;
    constexpr std::int64_t CAPACITY = 2000, HEADS = 2, WIDTH = 64, QUERIES = 6, GROUP = 32;
    constexpr auto GROUPS = HEADS * WIDTH / GROUP;
    auto cache = require(Tensor::zeros({1, CAPACITY, HEADS * WIDTH}, DType::I8));
    auto scales = require(Tensor::zeros({CAPACITY, GROUPS}, DType::F32));
    std::vector<float> input(3 * HEADS * WIDTH);
    for (std::size_t i = 0; i < input.size(); ++i) input[i] = std::sin(static_cast<float>(i)) * (1 + i / WIDTH);
    std::fill_n(input.data(), WIDTH, 0.F);
    const std::array<std::int32_t, 3> indices{7, 1998, 1999};
    const auto positions = require(Tensor::from_host({3}, std::span<const std::int32_t>(indices)));
    const auto source = context.cast(make_tensor({3, HEADS, WIDTH}, input), DType::BF16);
    context.quantize_cache_(cache, scales, source, positions);
    const auto rounded = context.cast(source, DType::F32);
    const auto original = require(rounded.data<float>());
    const auto codes = require(cache.data<std::int8_t>());
    const auto factors = require(scales.data<float>());
    std::vector<float> restored(CAPACITY * HEADS * WIDTH);
    for (std::int64_t row = 0; row < CAPACITY; ++row)
        for (std::int64_t h = 0; h < HEADS; ++h)
            for (std::int64_t c = 0; c < WIDTH; ++c) {
                const auto offset = (row * HEADS + h) * WIDTH + c;
                restored[offset] = codes[offset] * factors[offset / GROUP];
            }
    for (std::size_t t = 0; t < indices.size(); ++t)
        for (std::int64_t h = 0; h < HEADS; ++h)
            for (std::int64_t c = 0; c < WIDTH; ++c)
                EXPECT_NEAR(restored[(indices[t] * HEADS + h) * WIDTH + c],
                            original[(t * HEADS + h) * WIDTH + c],
                            factors[(indices[t] * HEADS * WIDTH + h * WIDTH + c) / GROUP] * 0.51F);
    EXPECT_EQ(factors[7 * GROUPS], 1.F);
    EXPECT_EQ(factors[8 * GROUPS], 0.F);
    std::vector<float> query(QUERIES * WIDTH, 0.5F), zero(query.size()), relative(QUERIES * 4, 0.1F);
    std::vector<float> sink(QUERIES, 0.3F);
    const auto q = make_tensor({1, QUERIES * WIDTH}, query);
    const auto r = make_tensor({1, QUERIES, 4}, relative);
    const auto s = make_tensor({1, QUERIES}, sink);
    const auto g = make_tensor({1, QUERIES * WIDTH}, zero);
    const auto floats = make_tensor({CAPACITY, HEADS * WIDTH}, restored);
    const auto bytes = require(cache.reshape({CAPACITY, HEADS * WIDTH}));
    for (const auto window : {0, 2}) {
        const ops::Context::RelativeAttention shape{.heads = QUERIES, .key_heads = HEADS, .start = CAPACITY - 1,
                                           .window = window, .extent = 4, .log_alpha = 0.2F, .log_floor = 16.F};
        const auto actual = context.relative_attention(q, bytes, bytes, r, s, g, shape, {}, scales, scales);
        const auto expected = context.relative_attention(q, floats, floats, r, s, g, shape);
        const auto a = require(actual.data<float>()), e = require(expected.data<float>());
        for (std::size_t i = 0; i < a.size(); ++i) EXPECT_NEAR(a[i], e[i], 1e-5F);
    }
    const std::array<std::int32_t, 3> bad_indices{0, CAPACITY, 1};
    const auto bad = require(Tensor::from_host({3}, std::span<const std::int32_t>(bad_indices)));
    EXPECT_THROW(context.quantize_cache_(cache, scales, source, bad), ops::Failure);
    EXPECT_EQ(factors[0], 0.F);
    EXPECT_THROW(context.relative_attention(q, bytes, bytes, r, s, g,
                                            {.heads = QUERIES, .key_heads = HEADS, .start = CAPACITY - 1, .extent = 4}),
                 ops::Failure);
}

TEST(FusedCpuTest, Int8PrefillAttentionMatchesDecode) {
    // Batched prefill over an INT8 cache takes the INT8 flash path; one-token calls take the decode path.
    constexpr std::int64_t HEADS = 6, KEY_HEADS = 2, WIDTH = 128, GROUP = 32, EXTENT = 8, ROWS = 1600, TOKENS = 37;
    constexpr auto KV = KEY_HEADS * WIDTH;
    std::mt19937 random(11);
    std::uniform_real_distribution<float> uniform(-1.F, 1.F);
    std::vector<float> floats(ROWS * KV);
    std::vector<std::int8_t> key(floats.size()), value(floats.size());
    std::vector<float> key_scales(ROWS * KV / GROUP), value_scales(key_scales.size());
    for (auto& x : floats) x = uniform(random);
    ops::cpu::quantize_rows(floats.data(), ROWS, KV, GROUP, key.data(), key_scales.data());
    for (auto& x : floats) x = uniform(random);
    ops::cpu::quantize_rows(floats.data(), ROWS, KV, GROUP, value.data(), value_scales.data());
    const ops::cpu::Rows key_rows{key.data(), ops::cpu::Rows::Storage::I8, KV, key_scales.data(), GROUP};
    const ops::cpu::Rows value_rows{value.data(), ops::cpu::Rows::Storage::I8, KV, value_scales.data(), GROUP};
    std::vector<float> query(TOKENS * HEADS * WIDTH), gate(query.size()), relative(TOKENS * HEADS * EXTENT),
        sink(TOKENS * HEADS), output(query.size()), single(HEADS * WIDTH);
    for (auto* values : {&query, &gate, &relative, &sink})
        for (auto& x : *values) x = 2.F * uniform(random);
    std::vector<std::int32_t> positions(TOKENS);
    std::iota(positions.begin(), positions.end(), static_cast<std::int32_t>(ROWS - TOKENS));
    for (const auto window : {std::int64_t{0}, std::int64_t{100}}) {
        const ops::cpu::RelativeAttentionShape shape{TOKENS, HEADS, KEY_HEADS, WIDTH, EXTENT, window, 0.1F, 64.F};
        ASSERT_TRUE(ops::cpu::relative_attention(query.data(), key_rows, value_rows, relative.data(), sink.data(),
                                                 gate.data(), positions, shape, output.data()));
        auto one = shape;
        one.tokens = 1;
        for (std::int64_t t = 0; t < TOKENS; ++t) {
            const auto offset = t * HEADS * WIDTH;
            ASSERT_TRUE(ops::cpu::relative_attention(
                query.data() + offset, key_rows, value_rows, relative.data() + t * HEADS * EXTENT,
                sink.data() + t * HEADS, gate.data() + offset, std::span(positions).subspan(t, 1), one, single.data()));
            // Two INT8 codes per query group (value and residual) keep float-query accuracy.
            for (std::int64_t i = 0; i < HEADS * WIDTH; ++i)
                ASSERT_NEAR(output[offset + i], single[i], 5e-5) << "window " << window << " token " << t;
        }
    }
}

TEST(FusedCpuTest, LongContextGroupedAttentionMatchesReference) {
    // Decode forces key-range splitting; unordered positions and a tile tail exercise prefill causal masking.
    constexpr std::int64_t HEADS = 6, KEY_HEADS = 2, WIDTH = 32, EXTENT = 4, ROWS = 2000;
    std::mt19937 random(5);
    std::uniform_real_distribution<float> uniform(-1.F, 1.F);
    std::vector<float> key(ROWS * KEY_HEADS * WIDTH), value(key.size());
    for (auto& x : key) x = uniform(random);
    for (auto& x : value) x = uniform(random);
    for (const auto& positions : {std::vector<std::int32_t>{ROWS - 1}, std::vector<std::int32_t>{9, 4, 12, 6, 7}}) {
        const auto tokens = static_cast<std::int64_t>(positions.size());
        std::vector<float> query(tokens * HEADS * WIDTH), gate(query.size()), relative(tokens * HEADS * EXTENT),
            sink(tokens * HEADS), output(query.size());
        for (auto* values : {&query, &gate, &relative, &sink})
            for (auto& x : *values) x = 2.F * uniform(random);
        const ops::cpu::RelativeAttentionShape shape{tokens, HEADS, KEY_HEADS, WIDTH, EXTENT, 0, 0.F, 0.F};
        const ops::cpu::Rows key_rows{key.data(), ops::cpu::Rows::Storage::F32, KEY_HEADS * WIDTH};
        const ops::cpu::Rows value_rows{value.data(), ops::cpu::Rows::Storage::F32, KEY_HEADS * WIDTH};
        ASSERT_TRUE(ops::cpu::relative_attention(query.data(), key_rows, value_rows, relative.data(), sink.data(),
                                                 gate.data(), positions, shape, output.data()));
        for (std::int64_t t = 0; t < tokens; ++t)
            for (std::int64_t h = 0; h < HEADS; ++h) {
                const auto row = t * HEADS + h;
                const auto column = h / (HEADS / KEY_HEADS) * WIDTH;
                const auto position = positions[static_cast<std::size_t>(t)];
                std::vector<double> logits;
                double maximum = sink[row];
                for (std::int64_t i = 0; i <= position; ++i) {
                    double logit = 0;
                    for (std::int64_t c = 0; c < WIDTH; ++c)
                        logit += query[row * WIDTH + c] * key[i * KEY_HEADS * WIDTH + column + c];
                    logit /= std::sqrt(static_cast<double>(WIDTH));
                    if (position - i < EXTENT) logit += relative[row * EXTENT + position - i];
                    logits.push_back(logit);
                    maximum = std::max(maximum, logit);
                }
                double denominator = std::exp(sink[row] - maximum);
                for (auto& logit : logits) denominator += logit = std::exp(logit - maximum);
                for (std::int64_t c = 0; c < WIDTH; ++c) {
                    double expected = 0;
                    for (std::int64_t i = 0; i <= position; ++i)
                        expected += logits[i] * value[i * KEY_HEADS * WIDTH + column + c];
                    expected /= denominator * (1. + std::exp(-gate[row * WIDTH + c]));
                    EXPECT_NEAR(output[row * WIDTH + c], expected, 1e-4) << "token " << t << " head " << h;
                }
            }
    }
}
} // namespace
