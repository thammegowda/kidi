#include "kidi/ops/context.h"
#include "kidi/ops/quantization.h"
#include "kidi/tensor/arena.h"
#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

namespace {

using namespace kidi;

auto devices() -> std::vector<tensor::Device> {
    std::vector devices{tensor::Device::cpu()};
#if defined(__APPLE__)
    devices.push_back(tensor::Device::apple_gpu());
#endif
    return devices;
}

auto device_name(const ::testing::TestParamInfo<tensor::Device>& info) -> std::string {
    std::string name;
    for (const char character : tensor::to_string(info.param))
        if (std::isalnum(static_cast<unsigned char>(character))) name.push_back(character);
    return name;
}

auto fails_with_code(const std::function<void()>& call, ErrorCode code) -> ::testing::AssertionResult {
    try {
        call();
    } catch (const ops::Failure& error) {
        if (error.error().code == code) return ::testing::AssertionSuccess();
        return ::testing::AssertionFailure() << "failed with a different error: " << error.error().message;
    }
    return ::testing::AssertionFailure() << "call did not fail";
}

class EagerTest : public ::testing::TestWithParam<tensor::Device> {
protected:
    auto make_input() -> tensor::Tensor {
        return ops::require(
            tensor::Tensor::from_host({1, 4}, std::span<const float>(std::array{1.F, 2.F, 3.F, 4.F}), device));
    }

    tensor::Device device = GetParam();
    ops::Context context{device};
};

INSTANTIATE_TEST_SUITE_P(, EagerTest, ::testing::ValuesIn(devices()), device_name);

TEST_P(EagerTest, ArenaTemporaries) {
    tensor::Tensor escaped;
    if (device == tensor::Device::cpu()) {
        tensor::Arena arena(1024);
        ops::require(arena.reserve(1024));
        auto first = ops::require(arena.allocate({4}, tensor::DType::F32));
        escaped = ops::require(arena.allocate({4}, tensor::DType::F32));
        EXPECT_TRUE(first.owns_unique_storage());
        EXPECT_TRUE(escaped.owns_unique_storage());
        EXPECT_NE(escaped.storage_offset(), 0);
        auto first_values = ops::require(first.data<float>());
        std::fill(first_values.begin(), first_values.end(), 2.F);
        auto values = ops::require(escaped.data<float>());
        std::fill(values.begin(), values.end(), 3.F);
        ops::Context arena_context(device);
        auto sum = arena_context.add(first, escaped);
        arena_context.add_(escaped, first);
        arena_context.synchronize();
        EXPECT_EQ(ops::require(sum.data<float>())[0], 5.F);
        EXPECT_EQ(values[0], 5.F);
        EXPECT_EQ(first_values[0], 2.F);
        auto overflow = arena.allocate({-1}, tensor::DType::F32);
        EXPECT_FALSE(overflow);
        auto larger = ops::require(arena.allocate({1024}, tensor::DType::F32));
        EXPECT_EQ(values[0], 5.F);
        EXPECT_GT(arena.reserved_bytes(), 1024U);
    }
    if (escaped.defined()) EXPECT_EQ(ops::require(escaped.data<float>())[0], 5.F);
    {
        ops::Context temporary(device);
        auto source = ops::require(tensor::Tensor::from_host({2}, std::span<const float>(std::array{2.F, 3.F}), device));
        escaped = ops::require(temporary.add(source, source).narrow(0, 1, 1));
    }
    EXPECT_EQ(ops::require(escaped.data<float>())[0], 6.F);
}

TEST_P(EagerTest, CastToInt8) {
    const std::array values{1.F, 2.F, -1.F, -2.F};
    const auto input = ops::require(tensor::Tensor::from_host({4}, std::span<const float>(values), device));
    const auto bytes = context.cast(input, tensor::DType::I8);
    const auto unchanged = context.cast(bytes, tensor::DType::I8);
    context.synchronize();
    EXPECT_TRUE(std::ranges::equal(ops::require(bytes.data<std::int8_t>()), std::array{1, 2, -1, -2}));
    EXPECT_EQ(unchanged.storage_identity(), bytes.storage_identity());
    const auto calibrated = context.cast(input, tensor::DType::I8, 0.25F);
    context.synchronize();
    EXPECT_TRUE(std::ranges::equal(ops::require(calibrated.data<std::int8_t>()), std::array{4, 8, -4, -8}));
}

TEST(EagerCpuTest, CalibratedProjection) {
    ops::Context context(tensor::Device::cpu());
    for (const int bits : {2, 4, 8}) {
        for (const std::int64_t columns : {3, 35, 67, 257, 513}) {
            constexpr std::int64_t WIDTH = 256;
            const auto values_per_byte = 8 / bits;
            auto bytes = std::make_shared<std::vector<std::uint8_t>>(columns * WIDTH / values_per_byte);
            auto factors = std::make_shared<std::vector<float>>(columns, 0.001953125F);
            std::vector<std::int32_t> integers(columns * WIDTH);
            for (std::size_t index = 0; index < integers.size(); ++index) {
                integers[index] =
                    static_cast<int>((index * 13 + index / WIDTH * 7 + index / (WIDTH * 3) + 3) % (1 << bits)) -
                    (1 << (bits - 1));
                (*bytes)[index / values_per_byte] |= (integers[index] & ((1 << bits) - 1))
                                                     << (index % values_per_byte * bits);
            }
            const auto weight = ops::require(
                tensor::Tensor::from_blob({columns, WIDTH / values_per_byte}, tensor::DType::U8,
                                          std::as_bytes(std::span<const std::uint8_t>(*bytes)), bytes));
            const auto scales = ops::require(tensor::Tensor::from_blob(
                {columns, 1}, tensor::DType::F32, std::as_bytes(std::span<const float>(*factors)), factors));
            std::array<float, WIDTH> values{};
            for (std::size_t index = 0; index < values.size(); ++index)
                values[index] = (static_cast<int>(index % 17) - 8) * 0.125F;
            values[0] = -1000.F;
            values[1] = 1000.F;
            const auto input = ops::require(tensor::Tensor::from_host({1, WIDTH}, std::span<const float>(values)));
            const auto output = context.packed_linear(input, weight, scales, bits, WIDTH, 0.25F, 0.125F);
            const auto actual = ops::require(output.data<float>());
            for (std::size_t column = 0; column < actual.size(); ++column) {
                std::int32_t sum = 0;
                for (std::size_t channel = 0; channel < values.size(); ++channel)
                    sum += static_cast<std::int32_t>(
                               std::clamp(std::nearbyint(values[channel] / 0.25F), -128.F, 127.F)) *
                           integers[column * WIDTH + channel];
                const auto expected =
                    std::clamp(std::nearbyint(sum * (*factors)[column] * 0.25F / 0.125F), -128.F, 127.F) * 0.125F;
                ASSERT_EQ(actual[column], expected) << "native calibrated projection mismatch bits=" << bits
                                                    << " columns=" << columns << " column=" << column;
            }
        }
    }
}

TEST_P(EagerTest, PackedLinearUpdates) {
    for (const std::int64_t rows : {1, 8}) {
        SCOPED_TRACE(rows);
        auto matrix = ops::require(tensor::Tensor::zeros({8, 8}, tensor::DType::F32));
        auto entries = ops::require(matrix.data<float>());
        for (std::size_t diagonal = 0; diagonal < 8; ++diagonal) entries[diagonal * 8 + diagonal] = 1.F;
        auto weight = ops::require(ops::pack_weight(matrix, 8, 8));
        auto operand = ops::require(tensor::Tensor::empty({rows, 8}, tensor::DType::F32, device));
        std::ranges::fill(ops::require(operand.data<float>()), 0.5F);
        auto increment = ops::require(tensor::Tensor::empty({rows, 8}, tensor::DType::F32, device));
        std::ranges::fill(ops::require(increment.data<float>()), 0.25F);
        const auto project = [&] {
            return context.packed_linear(operand, weight.values, weight.scales, 8, 8, 0.25F, 0.125F);
        };
        const auto first = project();
        const auto repeated = project();
        context.add_(operand, increment);
        const auto updated = project();
        context.copy_slice_(operand, increment, 0, 0);
        const auto copied = project();
        context.synchronize();
        for (const auto& [output, expected] : std::array{std::pair{first, 0.5F}, std::pair{repeated, 0.5F},
                                                         std::pair{updated, 0.75F}, std::pair{copied, 0.25F}})
            for (auto value : ops::require(output.data<float>())) EXPECT_NEAR(value, expected, 1e-6F);
        std::ranges::fill(ops::require(operand.data<float>()), 1.F);
        const auto host_changed = project();
        context.synchronize();
        for (auto value : ops::require(host_changed.data<float>())) EXPECT_NEAR(value, 1.F, 1e-6F);
    }
}

TEST_P(EagerTest, GatedActivations) {
    const std::array values{-100.F, -10.F, -3.F, -0.F, 0.25F, 3.F, 10.F, 100.F};
    const std::array factors{-1.F, 0.F, 0.5F, 2.F, -0.75F, 1.25F, -2.F, 0.125F};
    auto gate = ops::require(tensor::Tensor::from_host({2, 4}, std::span<const float>(values), device));
    auto factor = ops::require(tensor::Tensor::from_host({2, 4}, std::span<const float>(factors), device));
    auto separate = context.multiply(context.gelu(gate, true), factor);
    auto fused = context.gelu_multiply(gate, factor);
    context.synchronize();
    EXPECT_TRUE(std::ranges::equal(ops::require(separate.data<float>()), ops::require(fused.data<float>())));
    if (device == tensor::Device::cpu()) {
        const std::array parts{gate, factor};
        const auto combined = context.concat(parts, 1);
        const auto packed = context.gelu_gate(combined);
        context.synchronize();
        EXPECT_TRUE(std::ranges::equal(ops::require(fused.data<float>()), ops::require(packed.data<float>())));
    }
    EXPECT_THROW(context.gelu_multiply(gate, ops::require(factor.reshape({8}))), ops::Failure);
    const auto silu = context.silu_multiply(gate, factor);
    tensor::Tensor split_silu;
    if (device == tensor::Device::cpu() || device == tensor::Device::apple_gpu()) {
        const std::array parts{gate, factor};
        const auto combined = context.concat(parts, 1);
        split_silu = context.silu_gate(combined);
    }
    context.synchronize();
    const auto actual = ops::require(silu.data<float>());
    const auto split_values = split_silu.defined() ? ops::require(split_silu.data<float>()) : std::span<const float>{};
    for (std::size_t index = 0; index < actual.size(); ++index) {
        const auto expected = values[index] / (1.F + std::exp(-values[index])) * factors[index];
        EXPECT_NEAR(actual[index], expected, 1e-5F) << "SiLU mismatch at " << index;
        if (!split_values.empty()) EXPECT_NEAR(split_values[index], expected, 1e-5F) << "split SiLU mismatch at " << index;
    }
}

TEST_P(EagerTest, RmsRotary) {
    for (const std::int64_t width : {40, 256, 512}) {
        SCOPED_TRACE(width);
        auto input = ops::require(tensor::Tensor::empty({2, 3, 4, width}, tensor::DType::F32, device));
        auto scale = ops::require(tensor::Tensor::empty({width}, tensor::DType::F32, device));
        auto cosine = ops::require(tensor::Tensor::empty({1, 3, 1, width / 2}, tensor::DType::F32, device));
        auto sine = ops::require(tensor::Tensor::empty({1, 3, 1, width / 2}, tensor::DType::F32, device));
        auto values = ops::require(input.data<float>());
        for (std::size_t index = 0; index < values.size(); ++index)
            values[index] = (static_cast<int>(index % 29) - 14) * 0.137F;
        auto scales = ops::require(scale.data<float>());
        for (std::size_t index = 0; index < scales.size(); ++index)
            scales[index] = (static_cast<int>(index % 5) - 2) * 0.71F;
        auto cosines = ops::require(cosine.data<float>()), sines = ops::require(sine.data<float>());
        for (std::size_t index = 0; index < cosines.size(); ++index) {
            cosines[index] = std::cos(index * 0.071F);
            sines[index] = std::sin(index * 0.071F);
        }
        auto reference = context.rotary(context.rms_norm(input, scale, 1e-6F), cosine, sine);
        auto fused = context.rms_rotary(input, scale, cosine, sine, 1e-6F);
        context.synchronize();
        const auto expected = ops::require(reference.data<float>()), actual = ops::require(fused.data<float>());
        for (std::size_t index = 0; index < actual.size(); ++index) {
            ASSERT_TRUE(std::isfinite(actual[index])) << "index " << index;
            ASSERT_NEAR(actual[index], expected[index], 1e-6F) << "index " << index;
        }
    }
}

#if defined(__APPLE__)
TEST(EagerMetalTest, CalibratedEpilogue) {
    const auto device = tensor::Device::apple_gpu();
    ops::Context context(device);
    auto identity = ops::require(tensor::Tensor::zeros({8, 8}, tensor::DType::F32));
    for (std::size_t index = 0; index < 8; ++index) ops::require(identity.data<float>())[index * 8 + index] = 1.F;
    auto packed = ops::require(ops::pack_weight(identity, 2, 8));
    std::vector<float> values;
    const std::array row{-100.F, -1.25F, -0.75F, -0.25F, 0.25F, 0.75F, 1.25F, 100.F};
    for (int index = 0; index < 4; ++index) values.insert(values.end(), row.begin(), row.end());
    auto input = ops::require(tensor::Tensor::from_host({4, 8}, std::span<const float>(values), device));
    auto output = context.packed_linear(input, packed.values, packed.scales, 2, 8, 0.F, 0.5F);
    context.synchronize();
    const std::array expected{-64.F, -1.F, -1.F, 0.F, 0.F, 1.F, 1.F, 63.5F};
    const auto actual = ops::require(output.data<float>());
    for (std::size_t index = 0; index < actual.size(); ++index)
        EXPECT_EQ(actual[index], expected[index % 8])
            << "calibrated epilogue mismatch " << tensor::to_string(device) << " at " << index;
}
#endif

TEST_P(EagerTest, GreedyToken) {
    for (const std::int64_t width : {7, 1027, 262144}) {
        SCOPED_TRACE(width);
        const auto infinity = std::numeric_limits<float>::infinity();
        std::vector<float> scores(6 * width, -2.F);
        scores[1] = scores[width - 1] = 4.F;
        std::fill(scores.begin() + width, scores.begin() + 2 * width, -infinity);
        scores[2 * width + width - 1] = std::numeric_limits<float>::quiet_NaN();
        scores[3 * width + 1] = infinity;
        scores[6 * width - 1] = 3.F;
        auto logits = ops::require(tensor::Tensor::from_host({6, width}, std::span<const float>(scores), device));
        auto selected = context.greedy_token(logits);
        context.synchronize();
        EXPECT_TRUE(std::ranges::equal(
            ops::require(selected.data<std::int32_t>()),
            std::array<std::int32_t, 6>{1, -1, -1, -1, 0, static_cast<std::int32_t>(width - 1)}));
        auto finite = ops::require(
            tensor::Tensor::from_host({1, 3}, std::span<const float>(std::array{-3.F, 0.F, -0.F}), device));
        auto queued = context.greedy_token(context.add(finite, finite));
        context.synchronize();
        EXPECT_EQ(ops::require(queued.data<std::int32_t>())[0], 1);
    }
    EXPECT_THROW(context.greedy_token(ops::require(tensor::Tensor::zeros({}, tensor::DType::F32, device))),
                 ops::Failure);
}

TEST_P(EagerTest, LongOperationChains) {
    auto increment = ops::require(
        tensor::Tensor::from_host({1, 4}, std::span<const float>(std::array{1.F, 2.F, 3.F, 4.F}), device));
    auto identity = ops::require(tensor::Tensor::from_host(
        {4, 4},
        std::span<const float>(std::array{1.F, 0.F, 0.F, 0.F, 0.F, 1.F, 0.F, 0.F, 0.F, 0.F, 1.F, 0.F, 0.F, 0.F, 0.F, 1.F}),
        device));
    auto current = context.add(increment, increment);
    auto retained = ops::require(ops::require(current.reshape({4})).narrow(0, 0, 2));
    for (int iteration = 0; iteration < 64; ++iteration) {
        current = context.add(current, increment);
        current = context.matmul(current, identity);
    }
    context.synchronize();
    EXPECT_TRUE(std::ranges::equal(ops::require(current.data<float>()), std::array{66.F, 132.F, 198.F, 264.F}));
    EXPECT_TRUE(std::ranges::equal(ops::require(retained.data<float>()), std::array{2.F, 4.F}));
    if (device == tensor::Device::apple_gpu()) {
        ops::Context eviction(device);
        auto scale = ops::require(
            tensor::Tensor::from_host({4}, std::span<const float>(std::array{1.F, 1.F, 1.F, 1.F}), device));
        auto preserved = eviction.add(increment, increment);
        constexpr int ITERATIONS = 4102;
        for (int iteration = 0; iteration < ITERATIONS; ++iteration) {
            current = eviction.rms_norm(increment, scale, (iteration + 1) * 0.001F);
            const auto preparation = eviction.preparation_ns();
            eviction.add(increment, increment);
            ASSERT_EQ(eviction.preparation_ns(), preparation) << "iteration " << iteration;
        }
        eviction.synchronize();
        EXPECT_TRUE(std::ranges::equal(ops::require(preserved.data<float>()), std::array{2.F, 4.F, 6.F, 8.F}));
        const auto actual = ops::require(current.data<float>());
        for (std::size_t channel = 0; channel < actual.size(); ++channel)
            EXPECT_NEAR(actual[channel], (channel + 1) / std::sqrt(7.5F + ITERATIONS * 0.001F), 1e-5F)
                << "channel " << channel;
    }
}

TEST_P(EagerTest, PackedWeightBits) {
    for (int bits : {2, 4, 8}) {
        SCOPED_TRACE(bits);
        const std::array weights{0.F, -7.F, 7.F, 1.F, 0.F, 14.F, -14.F, 2.F, 0.F, 0.F, 0.F, 0.F, 7.F, -7.F, 1.F, -1.F};
        auto original = ops::require(tensor::Tensor::from_host({2, 8}, std::span<const float>(weights)));
        auto packed = ops::require(ops::pack_weight(original, bits, 4));
        const std::array activation{0.F, 1.F, 2.F, 3.F, 4.F, 5.F, 6.F, 255.F};
        auto operand = ops::require(tensor::Tensor::from_host({1, 8}, std::span<const float>(activation), device));
        auto output = context.packed_linear(operand, packed.values, packed.scales, bits, 4);
        context.synchronize();
        const auto bytes = ops::require(packed.values.data<std::uint8_t>());
        const auto scales = ops::require(packed.scales.data<float>());
        const auto values = ops::require(output.data<float>());
        for (std::size_t row = 0; row < 2; ++row) {
            float expected = 0;
            for (std::size_t channel = 0; channel < 8; ++channel) {
                const auto offset = row * 8 + channel;
                const auto raw = (bytes[offset / (8 / bits)] >> ((offset % (8 / bits)) * bits)) & ((1 << bits) - 1);
                const auto integer = (raw ^ (1 << (bits - 1))) - (1 << (bits - 1));
                expected += activation[channel] * integer * scales[row * 2 + channel / 4];
            }
            ASSERT_TRUE(std::isfinite(values[row])) << "row " << row;
            ASSERT_NEAR(values[row], expected, 1e-3F) << "row " << row;
        }
        EXPECT_FALSE(ops::pack_weight(original, bits, 3));
        context.prepare_linear_weights(original, bits, 4);
        auto rebound = context.linear(operand, original, {}, true);
        context.synchronize();
        const auto rebound_values = ops::require(rebound.data<float>());
        for (std::size_t row = 0; row < 2; ++row) ASSERT_NEAR(rebound_values[row], values[row], 1e-3F) << "row " << row;
        std::vector<float> many_values(33 * 8);
        for (std::size_t row = 0; row < 33; ++row)
            std::copy(activation.begin(), activation.end(), many_values.begin() + row * 8);
        auto many = ops::require(tensor::Tensor::from_host({33, 8}, std::span<const float>(many_values), device));
        auto tiled = context.linear(many, original, {}, true);
        context.synchronize();
        const auto tiled_values = ops::require(tiled.data<float>());
        for (std::size_t row = 0; row < 33; ++row)
            for (std::size_t column = 0; column < 2; ++column) {
                ASSERT_TRUE(std::isfinite(tiled_values[row * 2 + column])) << "row " << row << " column " << column;
                ASSERT_NEAR(tiled_values[row * 2 + column], values[column], 0.1F)
                    << "row " << row << " column " << column;
            }
    }
}

TEST_P(EagerTest, PackedLinear4BitWide) {
    constexpr std::size_t WIDTH = 256, COLUMNS = 35;
    std::vector<float> weights(COLUMNS * WIDTH);
    for (std::size_t column = 0; column < COLUMNS; ++column)
        for (std::size_t channel = 0; channel < WIDTH; ++channel)
            weights[column * WIDTH + channel] =
                (static_cast<int>((column * 7 + channel) % 15) - 7) * (channel < 128 ? 0.125F : 0.25F);
    auto original = ops::require(tensor::Tensor::from_host({COLUMNS, WIDTH}, std::span<const float>(weights)));
    auto packed = ops::require(ops::pack_weight(original));
    for (std::size_t rows : {1, 3, 33}) {
        SCOPED_TRACE(rows);
        std::vector<float> activations(rows * WIDTH);
        for (std::size_t index = 0; index < activations.size(); ++index) activations[index] = float(index % 256);
        auto operand = ops::require(tensor::Tensor::from_host({static_cast<std::int64_t>(rows), WIDTH},
                                                              std::span<const float>(activations), device));
        const auto output = context.packed_linear(operand, packed.values, packed.scales, 4, 128);
        context.synchronize();
        const auto values = ops::require(output.data<float>());
        for (std::size_t row = 0; row < rows; ++row)
            for (std::size_t column = 0; column < COLUMNS; ++column) {
                float expected = 0;
                for (std::size_t channel = 0; channel < WIDTH; ++channel)
                    expected += activations[row * WIDTH + channel] * weights[column * WIDTH + channel];
                ASSERT_TRUE(std::isfinite(values[row * COLUMNS + column])) << "row " << row << " column " << column;
                ASSERT_NEAR(values[row * COLUMNS + column], expected, 0.05F) << "row " << row << " column " << column;
            }
    }
}

TEST_P(EagerTest, AddAndSoftmax) {
    auto input = make_input();
    auto result = context.add(input, input);
    auto normalized = context.softmax(result);
    context.synchronize();
    auto data = ops::require(result.data<float>());
    EXPECT_EQ(data[0], 2);
    EXPECT_EQ(data[3], 8);
    auto probabilities = ops::require(normalized.data<float>());
    float total = 0;
    for (auto value : probabilities) total += value;
    EXPECT_NEAR(total, 1, 1e-5F);
}

TEST_P(EagerTest, RmsNorm) {
    const std::array rms_input_values{1.F, 2.F, 3.F, 4.F, 0.F, 0.F, 0.F, 0.F};
    const std::array rms_scale_values{1.F, -2.F, 0.5F, 3.F};
    auto rms_input = ops::require(tensor::Tensor::from_host({2, 4}, std::span<const float>(rms_input_values), device));
    auto rms_scale = ops::require(tensor::Tensor::from_host({4}, std::span<const float>(rms_scale_values), device));
    auto rms_output = context.rms_norm(rms_input, rms_scale, 1e-6F);
    auto other_scale = ops::require(
        tensor::Tensor::from_host({4}, std::span<const float>(std::array{2.F, -4.F, 1.F, 6.F}), device));
    auto other_rms = context.rms_norm(rms_input, other_scale, 1e-6F);
    auto rms_residual = context.rms_norm_residual(rms_input, rms_scale, rms_input, 1e-6F);
    auto output_scale = ops::require(tensor::Tensor::from_host({1}, std::span<const float>(std::array{0.5F}), device));
    auto rms_scaled = context.rms_norm_residual(rms_input, rms_scale, rms_input, 1e-6F, output_scale);
    context.synchronize();
    const auto rms_values = ops::require(rms_output.data<float>());
    for (std::size_t channel = 0; channel < 4; ++channel) {
        SCOPED_TRACE(channel);
        const auto expected = rms_input_values[channel] * rms_scale_values[channel] / std::sqrt(7.5F + 1e-6F);
        EXPECT_NEAR(rms_values[channel], expected, 1e-5F);
        EXPECT_EQ(rms_values[4 + channel], 0.F);
        EXPECT_NEAR(ops::require(other_rms.data<float>())[channel], 2.F * expected, 1e-5F);
        EXPECT_NEAR(ops::require(rms_residual.data<float>())[channel], rms_input_values[channel] + expected, 1e-5F);
        EXPECT_NEAR(ops::require(rms_scaled.data<float>())[channel], 0.5F * (rms_input_values[channel] + expected),
                    1e-5F);
    }
    EXPECT_TRUE(fails_with_code([&] { context.rms_norm(rms_input, rms_scale, 0.F); }, ErrorCode::INVALID_ARGUMENT));
}

TEST_P(EagerTest, ActivationsRounding) {
    auto activation_input = ops::require(
        tensor::Tensor::from_host({2, 2}, std::span<const float>(std::array{-4.F, -1.F, 0.F, 3.F}), device));
    auto activated = context.gelu(activation_input, true);
    const auto rounding_input = ops::require(tensor::Tensor::from_host(
        {8}, std::span<const float>(std::array{-100.F, -1.25F, -0.75F, -0.25F, 0.25F, 0.75F, 1.25F, 100.F}), device));
    const auto rounded = context.static_round(rounding_input, 0.5F);
    context.synchronize();
    const std::array rounding_expected{-64.F, -1.F, -1.F, 0.F, 0.F, 1.F, 1.F, 63.5F};
    EXPECT_TRUE(std::ranges::equal(ops::require(rounded.data<float>()), rounding_expected));
    auto large = ops::require(
        tensor::Tensor::from_host({4}, std::span<const float>(std::array{-1000.F, -20.F, 20.F, 1000.F}), device));
    auto large_gelu = context.gelu(large, true);
    auto hyperbolic = context.tanh(activation_input);
    auto projection = context.linear(activation_input, activation_input, {}, true);
    context.synchronize();
    const auto activation_values = ops::require(activation_input.data<float>());
    const auto activated_values = ops::require(activated.data<float>());
    const auto large_values = ops::require(large_gelu.data<float>());
    EXPECT_EQ(large_values[0], 0.F);
    EXPECT_EQ(large_values[1], 0.F);
    EXPECT_EQ(large_values[2], 20.F);
    EXPECT_EQ(large_values[3], 1000.F);
    const auto hyperbolic_values = ops::require(hyperbolic.data<float>());
    for (std::size_t index = 0; index < activation_values.size(); ++index) {
        const auto value = activation_values[index];
        const auto expected =
            0.5F * value *
            (1.F + std::tanh(std::sqrt(2.F / 3.14159265358979323846F) * (value + 0.044715F * value * value * value)));
        EXPECT_NEAR(activated_values[index], expected, 1e-5F) << "index " << index;
        EXPECT_NEAR(hyperbolic_values[index], std::tanh(value), 1e-5F) << "index " << index;
    }
    const auto projection_values = ops::require(projection.data<float>());
    EXPECT_EQ(projection_values[0], 17.F);
    EXPECT_EQ(projection_values[1], -3.F);
    EXPECT_EQ(projection_values[3], 9.F);
}

TEST_P(EagerTest, RotaryGqa) {
    auto input = make_input();
    auto rope_input = context.reshape(input, {1, 1, 1, 4});
    auto cosine =
        ops::require(tensor::Tensor::from_host({1, 1, 1, 2}, std::span<const float>(std::array{0.F, 1.F}), device));
    auto sine =
        ops::require(tensor::Tensor::from_host({1, 1, 1, 2}, std::span<const float>(std::array{1.F, 0.F}), device));
    auto rotated = context.rotary(rope_input, cosine, sine);
    auto query = context.reshape(input, {1, 1, 4});
    auto memory = ops::require(
        tensor::Tensor::from_host({1, 2, 2}, std::span<const float>(std::array{1.F, 0.F, 0.F, 1.F}), device));
    auto mask = ops::require(tensor::Tensor::zeros({1, 1, 1, 2}, tensor::DType::F32, device));
    auto attended = context.grouped_query_attention(query, memory, memory, 2, 1, mask);
    auto broadcast_mask = ops::require(tensor::Tensor::zeros({1}, tensor::DType::F32, device));
    auto broadcast_attended = context.grouped_query_attention(query, memory, memory, 2, 1, broadcast_mask);
    context.synchronize();
    const auto rotated_values = ops::require(rotated.data<float>());
    EXPECT_EQ(rotated_values[0], -3.F);
    EXPECT_EQ(rotated_values[1], 2.F);
    EXPECT_EQ(rotated_values[2], 1.F);
    EXPECT_EQ(rotated_values[3], 4.F);
    const auto attention_values = ops::require(attended.data<float>());
    const auto broadcast_values = ops::require(broadcast_attended.data<float>());
    for (std::size_t index = 0; index < attention_values.size(); ++index)
        EXPECT_NEAR(attention_values[index], broadcast_values[index], 1e-5F) << "index " << index;
    const auto probability = 1.F / (1.F + std::exp(1.F));
    for (std::size_t head = 0; head < 2; ++head) {
        EXPECT_NEAR(attention_values[head * 2], probability, 1e-5F) << "head " << head;
        EXPECT_NEAR(attention_values[head * 2 + 1], 1.F - probability, 1e-5F) << "head " << head;
    }
    for (const std::int64_t heads : {2, 8}) {
        SCOPED_TRACE(heads);
        const ops::DecodeScope decoding(true);
        constexpr std::int64_t ROWS = 3;
        const auto width = heads * 2;
        std::vector<float> queries(ROWS * width);
        for (std::size_t index = 0; index < queries.size(); ++index)
            queries[index] = (static_cast<int>(index % 11) - 5) * 0.25F;
        const std::array<float, 6> masks{0.F, -1e9F, 0.F, 0.F, -1e9F, 0.F};
        const auto batch_query =
            ops::require(tensor::Tensor::from_host({1, ROWS, width}, std::span<const float>(queries), device));
        const auto batch_mask =
            ops::require(tensor::Tensor::from_host({1, 1, ROWS, 2}, std::span<const float>(masks), device));
        const auto batch_result = context.grouped_query_attention(batch_query, memory, memory, heads, 1, batch_mask);
        context.synchronize();
        const auto batch_values = ops::require(batch_result.data<float>());
        const std::vector<float> expected(batch_values.begin(), batch_values.end());
        for (std::int64_t row = 0; row < ROWS; ++row) {
            const auto row_query = ops::require(tensor::Tensor::from_host(
                {1, 1, width}, std::span<const float>(queries).subspan(row * width, width), device));
            const auto row_mask = ops::require(
                tensor::Tensor::from_host({1, 1, 1, 2}, std::span<const float>(masks).subspan(row * 2, 2), device));
            const auto actual = context.grouped_query_attention(row_query, memory, memory, heads, 1, row_mask);
            context.synchronize();
            const auto actual_values = ops::require(actual.data<float>());
            for (std::int64_t column = 0; column < width; ++column)
                EXPECT_NEAR(actual_values[column], expected[row * width + column], 1e-5F)
                    << "row " << row << " column " << column;
        }
    }
}

TEST_P(EagerTest, Int8CacheAttention) {
    if (!device.capabilities().attention.int8_cache_tokens)
        GTEST_SKIP() << "device does not support int8 cache attention";
    constexpr std::int64_t HEADS = 4, KEY_HEADS = 2, WIDTH = 40, QUERIES = 3, KEYS = 259, START = 5;
    constexpr float KEY_SCALE = 0.05F, VALUE_SCALE = 0.02F;
    std::vector<float> query_values(QUERIES * HEADS * WIDTH);
    std::vector<float> key_values(KEYS * KEY_HEADS * WIDTH);
    std::vector<float> value_values(key_values.size());
    std::vector<float> mask_values(QUERIES * (KEYS - START));
    for (std::size_t index = 0; index < query_values.size(); ++index) query_values[index] = std::sin(float(index) * 0.37F);
    for (std::size_t index = 0; index < key_values.size(); ++index) {
        key_values[index] = std::sin(float(index) * 0.011F) * 0.8F;
        value_values[index] = std::cos(float(index) * 0.017F);
    }
    auto test_query = ops::require(
        tensor::Tensor::from_host({1, QUERIES, HEADS * WIDTH}, std::span<const float>(query_values), device));
    auto test_key = ops::require(
        tensor::Tensor::from_host({1, KEYS, KEY_HEADS * WIDTH}, std::span<const float>(key_values), device));
    auto test_value = ops::require(
        tensor::Tensor::from_host({1, KEYS, KEY_HEADS * WIDTH}, std::span<const float>(value_values), device));
    auto test_mask = ops::require(
        tensor::Tensor::from_host({1, 1, QUERIES, KEYS - START}, std::span<const float>(mask_values), device));
    auto rounded_key = context.static_round(test_key, KEY_SCALE);
    auto rounded_value = context.static_round(test_value, VALUE_SCALE);
    auto byte_key = context.cast(test_key, tensor::DType::I8, KEY_SCALE);
    auto byte_value = context.cast(test_value, tensor::DType::I8, VALUE_SCALE);
    const ops::BlockwiseQuantization key_quantization{{KEY_SCALE}, {0}, WIDTH};
    const ops::BlockwiseQuantization value_quantization{{VALUE_SCALE}, {0}, WIDTH};
    if (device == tensor::Device::cpu()) context.set_precision(core::InferencePrecision::QAT_FP32);
    auto expected = context.grouped_query_attention(test_query, rounded_key, rounded_value, HEADS, KEY_HEADS, test_mask,
                                                    1.F, START);
    auto actual = context.grouped_query_attention(test_query, byte_key, byte_value, HEADS, KEY_HEADS, test_mask, 1.F,
                                                  START, key_quantization, value_quantization);
    context.synchronize();
    const auto expected_values = ops::require(expected.data<float>());
    const auto actual_values = ops::require(actual.data<float>());
    for (std::size_t index = 0; index < expected_values.size(); ++index)
        ASSERT_NEAR(actual_values[index], expected_values[index], 1e-4F) << "index " << index;
    if (device == tensor::Device::cpu()) {
        const ops::DecodeScope decoding(true);
        const auto batched = context.grouped_query_attention(test_query, byte_key, byte_value, HEADS, KEY_HEADS,
                                                             test_mask, 1.F, START, key_quantization, value_quantization);
        context.synchronize();
        const auto batched_values = ops::require(batched.data<float>());
        for (std::int64_t row = 0; row < QUERIES; ++row) {
            const auto query_row = ops::require(tensor::Tensor::from_host(
                {1, 1, HEADS * WIDTH}, std::span<const float>(query_values).subspan(row * HEADS * WIDTH, HEADS * WIDTH),
                device));
            const auto mask_row = ops::require(tensor::Tensor::from_host(
                {1, 1, 1, KEYS - START},
                std::span<const float>(mask_values).subspan(row * (KEYS - START), KEYS - START), device));
            const auto serial = context.grouped_query_attention(query_row, byte_key, byte_value, HEADS, KEY_HEADS,
                                                                mask_row, 1.F, START, key_quantization,
                                                                value_quantization);
            context.synchronize();
            const auto serial_values = ops::require(serial.data<float>());
            EXPECT_TRUE(std::ranges::equal(serial_values, batched_values.subspan(row * HEADS * WIDTH, HEADS * WIDTH)))
                << "row " << row;
        }
    }
    if (device == tensor::Device::cpu()) context.set_precision(core::InferencePrecision::CHECKPOINT);
}

TEST_P(EagerTest, LongGqa) {
    constexpr std::int64_t HEADS = 4, KEY_HEADS = 2, WIDTH = 40, LENGTH = 259;
    std::vector<float> queries(HEADS * WIDTH), keys(LENGTH * KEY_HEADS * WIDTH), values(keys.size());
    std::vector<float> mask_values(LENGTH, -1e9F);
    for (std::size_t index = 0; index < queries.size(); ++index) queries[index] = std::sin(float(index)) * 0.2F;
    for (std::size_t index = 0; index < keys.size(); ++index) {
        keys[index] = std::cos(float(index) * 0.13F);
        values[index] = std::sin(float(index) * 0.17F);
    }
    for (std::size_t position = 131; position < LENGTH; ++position) mask_values[position] = 0.F;
    const auto tensor_query =
        ops::require(tensor::Tensor::from_host({1, 1, HEADS * WIDTH}, std::span<const float>(queries), device));
    const auto tensor_key = ops::require(
        tensor::Tensor::from_host({1, LENGTH, KEY_HEADS * WIDTH}, std::span<const float>(keys), device));
    const auto tensor_value = ops::require(
        tensor::Tensor::from_host({1, LENGTH, KEY_HEADS * WIDTH}, std::span<const float>(values), device));
    auto tensor_mask =
        ops::require(tensor::Tensor::from_host({1, 1, 1, LENGTH}, std::span<const float>(mask_values), device));
    const auto first =
        context.grouped_query_attention(tensor_query, tensor_key, tensor_value, HEADS, KEY_HEADS, tensor_mask, 0.5F);
    const auto second =
        context.grouped_query_attention(tensor_query, tensor_key, tensor_value, HEADS, KEY_HEADS, tensor_mask, 0.5F);
    context.synchronize();
    const auto actual = ops::require(first.data<float>()), repeated = ops::require(second.data<float>());
    for (std::size_t head = 0; head < HEADS; ++head) {
        std::vector<float> scores(LENGTH);
        for (std::size_t position = 0; position < LENGTH; ++position) {
            float dot = 0;
            for (std::size_t channel = 0; channel < WIDTH; ++channel)
                dot += queries[head * WIDTH + channel] * keys[(position * KEY_HEADS + head / 2) * WIDTH + channel];
            scores[position] = dot * 0.5F + mask_values[position];
        }
        const auto maximum = *std::ranges::max_element(scores);
        float denominator = 0;
        for (auto& score : scores) {
            score = std::exp(score - maximum);
            denominator += score;
        }
        for (std::size_t channel = 0; channel < WIDTH; ++channel) {
            float expected_value = 0;
            for (std::size_t position = 0; position < LENGTH; ++position)
                expected_value +=
                    scores[position] * values[(position * KEY_HEADS + head / 2) * WIDTH + channel] / denominator;
            const auto offset = head * WIDTH + channel;
            ASSERT_TRUE(std::isfinite(actual[offset])) << "offset " << offset;
            ASSERT_NEAR(actual[offset], expected_value, 2e-5F) << "offset " << offset;
            ASSERT_NEAR(repeated[offset], expected_value, 2e-5F) << "repeated offset " << offset;
        }
    }
}

TEST_P(EagerTest, RetainedTensors) {
    auto input = make_input();
    auto result = context.add(input, input);
    context.synchronize();
    auto saved = result;
    for (int iteration = 0; iteration < 12; ++iteration) {
        auto temporary = context.multiply(result, result);
        context.synchronize();
        ASSERT_EQ(ops::require(temporary.data<float>())[3], 64.F) << "iteration " << iteration;
    }
    EXPECT_EQ(ops::require(saved.data<float>())[3], 8.F);
    auto slice = context.slice(result, 1, 0, 2);
    context.synchronize();
    EXPECT_EQ(ops::require(slice.data<float>())[1], 4.F);
    auto retained_slice = slice;
    for (int iteration = 0; iteration < 10; ++iteration) {
        auto temporary = context.add(result, result);
        context.synchronize();
        ASSERT_EQ(ops::require(temporary.data<float>())[3], 16.F) << "iteration " << iteration;
    }
    EXPECT_EQ(ops::require(retained_slice.data<float>())[1], 4.F);
}

TEST_P(EagerTest, ScatterGather) {
    // Restores the thread-local in-place flag even if an assertion aborts the test early.
    struct InPlaceReset {
        ~InPlaceReset() { ops::is_inplace = false; }
    } reset;
    auto input = make_input();
    auto result = context.add(input, input);
    auto index = ops::require(
        tensor::Tensor::from_host({1}, std::span<const std::int32_t>(std::array<std::int32_t, 1>{1}), device));
    auto cache = ops::require(tensor::Tensor::zeros({1, 3, 4}, tensor::DType::F32, device));
    auto updated = context.scatter(cache, context.reshape(result, {1, 1, 4}), index);
    context.synchronize();
    EXPECT_EQ(ops::require(updated.data<float>())[7], 8.F);
    EXPECT_EQ(ops::require(cache.data<float>())[7], 0.F);
    auto rows = ops::require(
        tensor::Tensor::from_host({2, 2}, std::span<const float>(std::array{1.F, 2.F, 3.F, 4.F}), device));
    auto selected = context.gather(rows, index);
    context.synchronize();
    EXPECT_EQ(ops::require(selected.data<float>())[0], 3.F);
    EXPECT_TRUE(fails_with_code([&] { context.add(input, rows); }, ErrorCode::INVALID_ARGUMENT));
    auto mutable_input = ops::require(
        tensor::Tensor::from_host({1, 4}, std::span<const float>(std::array{1.F, 2.F, 3.F, 4.F}), device));
    auto alias = mutable_input;
    const auto address = ops::require(mutable_input.data<float>()).data();
    const auto immutable = context.add(mutable_input, mutable_input);
    EXPECT_EQ(&context.add_(mutable_input, mutable_input), &mutable_input);
    EXPECT_FALSE(ops::is_inplace);
    context.multiply_(mutable_input, mutable_input);
    context.synchronize();
    EXPECT_EQ(ops::require(alias.data<float>())[3], 64.F);
    EXPECT_EQ(ops::require(mutable_input.data<float>()).data(), address);
    EXPECT_EQ(ops::require(immutable.data<float>())[3], 8.F);
    auto cache_alias = cache;
    {
        auto destination = ops::require(tensor::Tensor::zeros({2, 4, 2}, tensor::DType::F32, device));
        auto source = ops::require(tensor::Tensor::from_host(
            {2, 2, 2}, std::span<const float>(std::array{1.F, 2.F, 3.F, 4.F, 5.F, 6.F, 7.F, 8.F}), device));
        const auto alias = destination;
        const auto address = ops::require(destination.host_bytes()).data();
        EXPECT_EQ(&context.copy_slice_(destination, source, 1, 2), &destination);
        EXPECT_FALSE(ops::is_inplace);
        context.synchronize();
        const std::array expected{0.F, 0.F, 0.F, 0.F, 1.F, 2.F, 3.F, 4.F, 0.F, 0.F, 0.F, 0.F, 5.F, 6.F, 7.F, 8.F};
        EXPECT_TRUE(std::ranges::equal(ops::require(alias.data<float>()), expected));
        EXPECT_EQ(ops::require(destination.host_bytes()).data(), address);
        for (auto start : {std::int64_t{-1}, std::int64_t{3}, INT64_MAX}) {
            SCOPED_TRACE(start);
            EXPECT_THROW(context.copy_slice_(destination, source, 1, start), ops::Failure);
            EXPECT_FALSE(ops::is_inplace);
            EXPECT_TRUE(std::ranges::equal(ops::require(alias.data<float>()), expected));
        }
        auto storage = ops::require(tensor::Tensor::from_host(
            {8}, std::span<const float>(std::array{0.F, 1.F, 2.F, 3.F, 4.F, 5.F, 6.F, 7.F}), device));
        auto view = ops::require(storage.narrow(0, 1, 6));
        auto overlapping = ops::require(storage.narrow(0, 0, 4));
        context.copy_slice_(view, overlapping, 0, 1);
        context.synchronize();
        EXPECT_TRUE(std::ranges::equal(ops::require(storage.data<float>()),
                                       std::array{0.F, 1.F, 0.F, 1.F, 2.F, 3.F, 6.F, 7.F}));
        {
            auto queued = context.add(source, source);
            context.copy_slice_(destination, queued, -2, 0);
        }
        auto consumed = context.add(destination, destination);
        context.synchronize();
        EXPECT_EQ(ops::require(consumed.data<float>())[0], 4.F);
        EXPECT_EQ(ops::require(consumed.data<float>())[8], 20.F);
    }
    context.scatter_(cache, context.reshape(result, {1, 1, 4}), index);
    context.synchronize();
    EXPECT_EQ(ops::require(cache_alias.data<float>())[7], 8.F);
    EXPECT_FALSE(ops::is_inplace);
    ops::is_inplace = true;
    auto protected_input = context.add(mutable_input, mutable_input);
    EXPECT_TRUE(ops::is_inplace);
    context.synchronize();
    EXPECT_EQ(ops::require(alias.data<float>())[3], 64.F);
    EXPECT_THROW(context.add_(input, rows), ops::Failure);
    EXPECT_TRUE(ops::is_inplace);
    bool worker_default = false;
    std::thread worker([&] { worker_default = !ops::is_inplace; });
    worker.join();
    EXPECT_TRUE(worker_default);
    EXPECT_TRUE(ops::is_inplace);
    ops::is_inplace = false;
}

TEST_P(EagerTest, InPlaceUnaryNorm) {
    auto input = make_input();
    auto result = context.add(input, input);
    auto mutable_input = ops::require(
        tensor::Tensor::from_host({1, 4}, std::span<const float>(std::array{4.F, 16.F, 36.F, 64.F}), device));
    const auto expected_softmax = context.softmax(mutable_input);
    context.softmax_(mutable_input);
    context.synchronize();
    const auto expected_values = ops::require(expected_softmax.data<float>());
    const auto actual_values = ops::require(mutable_input.data<float>());
    for (std::size_t index = 0; index < expected_values.size(); ++index)
        EXPECT_NEAR(expected_values[index], actual_values[index], 1e-5F) << "softmax index " << index;
    const auto expected_gelu = context.gelu(mutable_input);
    context.gelu_(mutable_input);
    context.synchronize();
    auto expected_gelu_values = ops::require(expected_gelu.data<float>());
    auto actual_gelu_values = ops::require(mutable_input.data<float>());
    for (std::size_t index = 0; index < expected_gelu_values.size(); ++index)
        EXPECT_NEAR(expected_gelu_values[index], actual_gelu_values[index], 1e-5F) << "gelu index " << index;
    auto scale = ops::require(tensor::Tensor::from_host({4}, std::span<const float>(std::array{1.F, 1.F, 1.F, 1.F})));
    auto bias = ops::require(tensor::Tensor::zeros({4}, tensor::DType::F32));
    auto paired = context.residual_layer_norm(input, input, scale, bias, 1e-5F);
    auto paired_expected = context.layer_norm(paired[0], scale, bias, 1e-5F);
    auto paired_again = context.residual_layer_norm(result, result, scale, bias, 1e-5F);
    context.add_(paired_again[0], result);
    context.synchronize();
    EXPECT_EQ(ops::require(paired[0].data<float>())[3], 8.F);
    EXPECT_EQ(ops::require(paired_again[0].data<float>())[3], 24.F);
    const auto paired_values = ops::require(paired[1].data<float>());
    const auto paired_expected_values = ops::require(paired_expected.data<float>());
    for (std::size_t index = 0; index < paired_values.size(); ++index)
        EXPECT_NEAR(paired_values[index], paired_expected_values[index], 1e-5F) << "paired index " << index;
    const auto expected_norm = context.layer_norm(mutable_input, scale, bias, 1e-5F);
    context.layer_norm_(mutable_input, scale, bias, 1e-5F);
    context.synchronize();
    auto expected_norm_values = ops::require(expected_norm.data<float>());
    auto actual_norm_values = ops::require(mutable_input.data<float>());
    for (std::size_t index = 0; index < expected_norm_values.size(); ++index)
        EXPECT_NEAR(expected_norm_values[index], actual_norm_values[index], 1e-5F) << "layer norm index " << index;
    auto scalar = ops::require(tensor::Tensor::from_host({}, std::span<const float>(std::array{1.F}), device));
    EXPECT_THROW(context.add_(scalar, input), ops::Failure);
    context.synchronize();
    EXPECT_FALSE(ops::is_inplace);
    EXPECT_EQ(ops::require(scalar.data<float>())[0], 1.F);
}

TEST_P(EagerTest, ScatterOverlapping) {
    auto index = ops::require(
        tensor::Tensor::from_host({1}, std::span<const std::int32_t>(std::array<std::int32_t, 1>{1}), device));
    auto shifted =
        ops::require(tensor::Tensor::from_host({1, 3, 1}, std::span<const float>(std::array{1.F, 2.F, 3.F}), device));
    auto flat = context.reshape(shifted, {3, 1});
    auto overlapping = context.reshape(ops::require(flat.narrow(0, 0, 2)), {1, 2, 1});
    auto shift_indices = ops::require(
        tensor::Tensor::from_host({2}, std::span<const std::int32_t>(std::array<std::int32_t, 2>{1, 2}), device));
    context.scatter_(shifted, overlapping, shift_indices);
    context.synchronize();
    auto shifted_values = ops::require(shifted.data<float>());
    EXPECT_EQ(shifted_values[0], 1.F);
    EXPECT_EQ(shifted_values[1], 1.F);
    EXPECT_EQ(shifted_values[2], 2.F);
    auto flat_view = context.reshape(shifted, {3});
    context.multiply_(flat_view, flat_view);
    context.synchronize();
    EXPECT_EQ(ops::require(shifted.data<float>())[2], 4.F);
    auto repeated_indices = ops::require(
        tensor::Tensor::from_host({2}, std::span<const std::int32_t>(std::array<std::int32_t, 2>{1, 1}), device));
    auto replacements =
        ops::require(tensor::Tensor::from_host({1, 2, 1}, std::span<const float>(std::array{5.F, 6.F}), device));
    context.scatter_(shifted, replacements, repeated_indices);
    context.synchronize();
    EXPECT_EQ(ops::require(shifted.data<float>())[1], 6.F);
    auto invalid_indices = ops::require(
        tensor::Tensor::from_host({2}, std::span<const std::int32_t>(std::array<std::int32_t, 2>{0, 3}), device));
    {
        ops::Context isolated(device);
        EXPECT_THROW(
            {
                isolated.scatter_(shifted, overlapping, invalid_indices);
                isolated.synchronize();
            },
            ops::Failure);
    }
    EXPECT_FALSE(ops::is_inplace);
    EXPECT_EQ(ops::require(shifted.data<float>())[0], 1.F);
    EXPECT_EQ(ops::require(shifted.data<float>())[1], 6.F);
    EXPECT_EQ(ops::require(shifted.data<float>())[2], 4.F);
    auto backing = ops::require(tensor::Tensor::zeros({4, 3, 1}, tensor::DType::F32, device));
    auto cache_view = ops::require(backing.narrow(0, 1, 2));
    auto batch_updates =
        ops::require(tensor::Tensor::from_host({2, 1, 1}, std::span<const float>(std::array{7.F, 9.F}), device));
    context.scatter_(cache_view, batch_updates, index);
    context.synchronize();
    auto batch_values = ops::require(backing.data<float>());
    for (std::size_t offset = 0; offset < batch_values.size(); ++offset)
        EXPECT_EQ(batch_values[offset], (offset == 4 ? 7.F : offset == 7 ? 9.F : 0.F)) << "offset " << offset;
}

TEST_P(EagerTest, ScatterQueuedIndex) {
    auto batch_updates =
        ops::require(tensor::Tensor::from_host({2, 1, 1}, std::span<const float>(std::array{7.F, 9.F}), device));
    ops::Context fresh(device);
    auto target = ops::require(tensor::Tensor::zeros({2, 3, 1}, tensor::DType::F32, device));
    auto first_index = ops::require(
        tensor::Tensor::from_host({1}, std::span<const std::int32_t>(std::array<std::int32_t, 1>{0}), device));
    auto delta = ops::require(
        tensor::Tensor::from_host({1}, std::span<const std::int32_t>(std::array<std::int32_t, 1>{1}), device));
    fresh.scatter_(target, batch_updates, first_index);
    auto next_index = fresh.add(first_index, delta);
    auto separate = fresh.scatter(target, batch_updates, next_index);
    fresh.synchronize();
    EXPECT_EQ(ops::require(target.data<float>())[1], 0.F);
    EXPECT_EQ(ops::require(separate.data<float>())[1], 7.F);
    EXPECT_EQ(ops::require(separate.data<float>())[4], 9.F);
}

TEST(EagerCpuTest, ReadonlyRejectsWrites) {
    const auto device = tensor::Device::cpu();
    ops::Context context(device);
    auto input = ops::require(
        tensor::Tensor::from_host({1, 4}, std::span<const float>(std::array{1.F, 2.F, 3.F, 4.F}), device));
    auto owner = std::make_shared<std::array<float, 4>>(std::array{1.F, 2.F, 3.F, 4.F});
    auto readonly = ops::require(
        tensor::Tensor::from_blob({1, 4}, tensor::DType::F32, std::as_bytes(std::span<const float>(*owner)), owner));
    EXPECT_THROW(context.add_(readonly, input), ops::Failure);
    EXPECT_FALSE(ops::is_inplace);
    EXPECT_EQ((*owner)[0], 1.F);
    EXPECT_THROW(context.copy_slice_(readonly, input, 0, 0), ops::Failure);
    EXPECT_FALSE(ops::is_inplace);
    EXPECT_EQ((*owner)[0], 1.F);
}

TEST(EagerCpuTest, RejectsUnsupportedDevice) {
    EXPECT_TRUE(fails_with_code([] { ops::Context unsupported(tensor::Device{tensor::DeviceKind::CPU, 1}); },
                                ErrorCode::UNSUPPORTED));
}

} // namespace
