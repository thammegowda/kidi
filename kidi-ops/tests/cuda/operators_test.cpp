#include "kidi/ops/context.h"
#include "kidi/layers/gemma4.h"
#include "kidi/layers/audio.h"
#include "cuda/runtime.h"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>

#include <gtest/gtest.h>

namespace {
static_assert(
    kidi::tensor::Device::cuda().capabilities().sampling.supports(kidi::tensor::TokenSelection::CHAINED_DEVICE));
static_assert(
    !kidi::tensor::Device::cuda().capabilities().sampling.supports(kidi::tensor::TokenSelection::BATCHED_DEVICE));
static_assert(
    !kidi::tensor::Device::apple_gpu().capabilities().sampling.supports(kidi::tensor::TokenSelection::CHAINED_DEVICE));

class CudaOpsTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!kidi::cuda::available()) GTEST_SKIP() << "CUDA device unavailable";
    }
};

TEST_F(CudaOpsTest, DequantizeExpertsMatchesCheckpoint) {
    using namespace kidi;
    using ops::require;
    using tensor::DType;
    using tensor::Tensor;
    const auto device = tensor::Device::cuda();
    ops::Context context(device, false, false, DType::BF16);
    const auto e4m3 = [](std::uint8_t bits) {
        const int exponent = (bits >> 3) & 15, mantissa = bits & 7;
        const float magnitude = exponent ? std::ldexp(1.F + static_cast<float>(mantissa) / 8.F, exponent - 7)
                                         : std::ldexp(static_cast<float>(mantissa) / 8.F, -6);
        return bits & 0x80 ? -magnitude : magnitude;
    };
    constexpr std::int64_t EXPERTS = 2, ROWS = 4, COLUMNS = 256;
    constexpr std::array<float, 16> E2M1{0.F,  0.5F,  1.F,  1.5F,  2.F,  3.F,  4.F,  6.F,
                                         -0.F, -0.5F, -1.F, -1.5F, -2.F, -3.F, -4.F, -6.F};
    const std::array<float, 2> factors{0.37F, 1.25F};
    for (const bool nvfp4 : {false, true})
        for (const bool interleaved : {false, true}) {
            SCOPED_TRACE(nvfp4 ? "NVFP4" : "MXFP8");
            SCOPED_TRACE(interleaved ? "interleaved" : "plain");
            const auto packed_columns = COLUMNS / (nvfp4 ? 2 : 1);
            const auto block = nvfp4 ? 16 : 32;
            std::vector<std::uint8_t> values(EXPERTS * ROWS * packed_columns), scales(EXPERTS * ROWS * COLUMNS / block);
            for (std::size_t index = 0; index < values.size(); ++index) {
                values[index] = static_cast<std::uint8_t>(index);
                if (!nvfp4 && (values[index] & 0x7F) == 0x7F) values[index] = 0x7E;
            }
            for (std::size_t index = 0; index < scales.size(); ++index)
                scales[index] =
                    nvfp4 ? static_cast<std::uint8_t>(index % 0x7F) : static_cast<std::uint8_t>(120 + index % 16);
            const auto packed =
                require(Tensor::from_bytes({EXPERTS, ROWS, packed_columns}, nvfp4 ? DType::U8 : DType::E4M3,
                                           std::as_bytes(std::span(values)), device));
            const auto scale =
                require(Tensor::from_bytes({EXPERTS, ROWS, COLUMNS / block}, nvfp4 ? DType::E4M3 : DType::U8,
                                           std::as_bytes(std::span(scales)), device));
            const auto global =
                nvfp4 ? require(Tensor::from_host({EXPERTS}, std::span<const float>(factors), device)) : Tensor{};
            const auto expanded = context.dequantize_experts(packed, scale, global, interleaved);
            ASSERT_EQ(expanded.dtype(), DType::BF16);
            ASSERT_TRUE(std::ranges::equal(expanded.shape(), std::array{EXPERTS, ROWS, COLUMNS}));
            const auto bytes = require(expanded.copy_to_host());
            std::vector<float> reference(EXPERTS * ROWS * COLUMNS);
            for (std::int64_t expert = 0; expert < EXPERTS; ++expert)
                for (std::int64_t row = 0; row < ROWS; ++row)
                    for (std::int64_t column = 0; column < COLUMNS; ++column) {
                        const auto source = (expert * ROWS + row) * COLUMNS + column;
                        float value;
                        if (nvfp4) {
                            const auto byte = values[source / 2];
                            value = E2M1[column % 2 ? byte >> 4 : byte & 15] * e4m3(scales[source / block]) *
                                    factors[expert];
                        } else {
                            value =
                                e4m3(values[source]) * std::ldexp(1.F, static_cast<int>(scales[source / block]) - 127);
                        }
                        auto bits = std::bit_cast<std::uint32_t>(value);
                        bits += 0x7FFF + ((bits >> 16) & 1);
                        const auto target_row = interleaved ? (row % 2 ? ROWS / 2 + row / 2 : row / 2) : row;
                        const auto target = (expert * ROWS + target_row) * COLUMNS + column;
                        reference[target] = value;
                        std::uint16_t actual;
                        std::memcpy(&actual, bytes.data() + target * sizeof(actual), sizeof(actual));
                        ASSERT_EQ(actual, bits >> 16) << "expert=" << expert << " row=" << row << " column=" << column;
                    }
            const auto packed_experts = context.pack_experts(packed, scale, global, interleaved);
            const auto expected = require(ops::pack_weight(
                require(Tensor::from_host({EXPERTS * ROWS, COLUMNS}, std::span<const float>(reference))), 8, COLUMNS));
            EXPECT_EQ(require(packed_experts.values.copy_to_host()), require(expected.values.copy_to_host()));
            EXPECT_EQ(require(packed_experts.scales.copy_to_host()), require(expected.scales.copy_to_host()));
        }
}

TEST_F(CudaOpsTest, DequantizeExpertsRejectsInvalidScales) {
    using namespace kidi;
    using ops::require;
    using tensor::DType;
    using tensor::Tensor;
    const auto device = tensor::Device::cuda();
    ops::Context context(device, false, false, DType::BF16);
    const auto values = require(Tensor::zeros({2, 4, 16}, DType::U8, device));
    const auto scales = require(Tensor::zeros({2, 4, 2}, DType::E4M3, device));
    const auto global = require(Tensor::zeros({2}, DType::F32, device));
    EXPECT_THROW(context.dequantize_experts(values, scales), ops::Failure);
    EXPECT_THROW(context.dequantize_experts(values, require(Tensor::zeros({2, 4, 1}, DType::E4M3, device)), global),
                 ops::Failure);
    EXPECT_THROW(context.dequantize_experts(values, scales, require(global.narrow(0, 0, 1))), ops::Failure);
    EXPECT_THROW(context.dequantize_experts(require(Tensor::zeros({2, 3, 16}, DType::U8, device)),
                                            require(Tensor::zeros({2, 3, 2}, DType::E4M3, device)), global, true),
                 ops::Failure);
    const std::vector<std::uint8_t> invalid(32, 0x7F);
    EXPECT_THROW(context.pack_experts(
                     require(Tensor::from_bytes({1, 1, 32}, DType::E4M3, std::as_bytes(std::span(invalid)), device)),
                     require(Tensor::zeros({1, 1, 1}, DType::U8, device))),
                 ops::Failure);
}

TEST_F(CudaOpsTest, AttentionBindings) {
    SCOPED_TRACE("CUDA attention retained a stale KV pointer");
    using namespace kidi;
    using ops::require;
    using tensor::DType;
    using tensor::Tensor;
    const auto device = tensor::Device::cuda();
    for (const std::int64_t width : {8, 256, 512}) {
        ops::Context context(device, false, true, DType::BF16);
        const auto query = context.cast(
            require(Tensor::from_host({1, 1, width}, std::span<const float>(std::vector<float>(width, 1.F)), device)),
            DType::BF16);
        const auto key_update = require(Tensor::zeros({1, 1, width}, DType::BF16, device));
        const auto index =
            require(Tensor::from_host({1}, std::span<const std::int32_t>(std::array<std::int32_t, 1>{128}), device));
        std::array<float, 128> masks{};
        std::ranges::fill(masks, -INFINITY);
        masks[0] = 0.F;
        const auto mask = require(Tensor::from_host({1, 1, 1, 128}, std::span<const float>(masks), device));
        Tensor original;
        for (int iteration = 0; iteration < 5; ++iteration) {
            const auto key = require(Tensor::zeros({1, 256, width}, DType::BF16, device));
            const auto value = require(Tensor::zeros({1, 256, width}, DType::BF16, device));
            if (!iteration) original = value;
            const auto update = context.cast(
                require(Tensor::from_host(
                    {1, 1, width}, std::span<const float>(std::vector<float>(width, static_cast<float>(iteration + 1))),
                    device)),
                DType::BF16);
            const std::array arguments{query, key, value, key_update, update, index, mask};
            const auto outputs =
                context.replay("cuda-attention-bindings", arguments, [&](std::span<const Tensor> inputs) {
                    auto keys = inputs[1], values = inputs[2];
                    context.scatter_(keys, inputs[3], inputs[5]);
                    context.scatter_(values, inputs[4], inputs[5]);
                    return std::vector{
                        context.grouped_query_attention(inputs[0], keys, values, 1, 1, inputs[6], 1.F, 128)};
                });
            const auto copied = require(context.cast(outputs[0], DType::F32).to(tensor::Device::cpu()));
            for (const auto actual : require(copied.data<float>()))
                ASSERT_EQ(actual, static_cast<float>(iteration + 1)) << "width=" << width << " iteration=" << iteration;
        }
        const auto copied = require(context.cast(original, DType::F32).to(tensor::Device::cpu()));
        const auto values = require(copied.data<float>());
        for (std::size_t column = 0; column < static_cast<std::size_t>(width); ++column)
            ASSERT_EQ(values[128 * width + column], 1.F) << "width=" << width << " column=" << column;
    }
}

TEST_F(CudaOpsTest, StructuredCausalMasks) {
    SCOPED_TRACE("CUDA structured causal masks changed attention output");
    using namespace kidi;
    using ops::require;
    using tensor::DType;
    using tensor::Tensor;
    const auto device = tensor::Device::cuda();
    ops::Context context(device, false, true, DType::BF16);
    const std::vector<float> query_values(128, 0.F), key_values(192, 0.F);
    std::vector<float> value_values(192, 1.F);
    std::ranges::fill(value_values.begin() + 64, value_values.begin() + 128, 3.F);
    std::ranges::fill(value_values.begin() + 128, value_values.end(), 9.F);
    const auto query = context.cast(require(Tensor::from_host({1, 2, 64}, std::span<const float>(query_values), device)),
                                    DType::BF16);
    const auto key = context.cast(require(Tensor::from_host({1, 3, 64}, std::span<const float>(key_values), device)),
                                  DType::BF16);
    const auto value =
        context.cast(require(Tensor::from_host({1, 3, 64}, std::span<const float>(value_values), device)), DType::BF16);
    const std::array<float, 6> full_precision_values{0.F, -1e9F, -1e9F, 0.F, 0.F, 0.F};
    const auto full_precision_mask =
        require(Tensor::from_host({1, 1, 2, 3}, std::span<const float>(full_precision_values), device));
    const std::array<std::int32_t, 2> query_position{0, 2};
    const auto positions = require(Tensor::from_host({1, 1, 2, 1}, std::span<const std::int32_t>(query_position),
                                                     device));
    const auto expected = context.grouped_query_attention(query, key, value, 1, 1, full_precision_mask);
    const auto causal = context.grouped_query_attention_causal(query, key, value, 1, 1, positions, 0);
    const auto local = context.grouped_query_attention_causal(query, key, value, 1, 1, positions, 2);
    const auto expected_host = require(context.cast(expected, DType::F32).to(tensor::Device::cpu()));
    const auto causal_host = require(context.cast(causal, DType::F32).to(tensor::Device::cpu()));
    const auto local_host = require(context.cast(local, DType::F32).to(tensor::Device::cpu()));
    const auto expected_values = require(expected_host.data<float>());
    const auto causal_values = require(causal_host.data<float>());
    const auto local_values = require(local_host.data<float>());
    for (std::size_t index = 0; index < causal_values.size(); ++index)
        ASSERT_NEAR(causal_values[index], expected_values[index], 1e-5F)
            << "causal mask parity failed index=" << index;
    EXPECT_NEAR(expected_values[0], 1.F, 1e-5F) << "attention expected row outputs 1, 4.333, 1, 6";
    EXPECT_NEAR(expected_values[64], 13.F / 3.F, 2e-2F) << "attention expected row outputs 1, 4.333, 1, 6";
    EXPECT_NEAR(local_values[0], 1.F, 1e-5F) << "attention expected row outputs 1, 4.333, 1, 6";
    EXPECT_NEAR(local_values[64], 6.F, 1e-5F) << "attention expected row outputs 1, 4.333, 1, 6";

    constexpr std::size_t long_keys = 1024;
    std::vector<float> long_query_values(2 * 64, 0.F), long_key_values(long_keys * 64, 0.F);
    std::vector<float> long_value_values(long_keys * 64);
    std::array<float, 2 * long_keys> long_causal_mask{}, long_local_mask{}, long_single_mask{};
    for (std::size_t key_index = 0; key_index < long_keys; ++key_index) {
        std::ranges::fill(long_value_values.begin() + key_index * 64, long_value_values.begin() + (key_index + 1) * 64,
                          static_cast<float>(key_index / 128));
        long_causal_mask[key_index] = key_index <= 511 ? 0.F : -1e9F;
        long_causal_mask[long_keys + key_index] = key_index <= 510 ? 0.F : -1e9F;
        long_local_mask[key_index] = key_index >= 256 && key_index <= 511 ? 0.F : -1e9F;
        long_local_mask[long_keys + key_index] = key_index >= 255 && key_index <= 510 ? 0.F : -1e9F;
        long_single_mask[key_index] = key_index == 511 ? 0.F : -1e9F;
        long_single_mask[long_keys + key_index] = key_index == 510 ? 0.F : -1e9F;
    }
    const auto long_query = context.cast(
        require(Tensor::from_host({1, 2, 64}, std::span<const float>(long_query_values), device)), DType::BF16);
    const auto long_key = context.cast(
        require(Tensor::from_host({1, static_cast<std::int64_t>(long_keys), 64},
                                  std::span<const float>(long_key_values), device)),
        DType::BF16);
    const auto long_value = context.cast(
        require(Tensor::from_host({1, static_cast<std::int64_t>(long_keys), 64},
                                  std::span<const float>(long_value_values), device)),
        DType::BF16);
    const std::array<std::int32_t, 2> long_query_positions{511, 510};
    const auto long_positions = require(
        Tensor::from_host({1, 1, 2, 1}, std::span<const std::int32_t>(long_query_positions), device));
    const auto long_causal_reference =
        context.grouped_query_attention(long_query, long_key, long_value, 1, 1,
                                        require(Tensor::from_host({1, 1, 2, static_cast<std::int64_t>(long_keys)},
                                                                  std::span<const float>(long_causal_mask), device)));
    const auto long_local_reference =
        context.grouped_query_attention(long_query, long_key, long_value, 1, 1,
                                        require(Tensor::from_host({1, 1, 2, static_cast<std::int64_t>(long_keys)},
                                                                  std::span<const float>(long_local_mask), device)));
    const auto long_single_reference =
        context.grouped_query_attention(long_query, long_key, long_value, 1, 1,
                                        require(Tensor::from_host({1, 1, 2, static_cast<std::int64_t>(long_keys)},
                                                                  std::span<const float>(long_single_mask), device)));
    const auto long_causal =
        context.grouped_query_attention_causal(long_query, long_key, long_value, 1, 1, long_positions, 0);
    const auto long_local =
        context.grouped_query_attention_causal(long_query, long_key, long_value, 1, 1, long_positions, 256);
    const auto long_single =
        context.grouped_query_attention_causal(long_query, long_key, long_value, 1, 1, long_positions, 1);
    const auto long_causal_reference_host =
        require(context.cast(long_causal_reference, DType::F32).to(tensor::Device::cpu()));
    const auto long_local_reference_host =
        require(context.cast(long_local_reference, DType::F32).to(tensor::Device::cpu()));
    const auto long_single_reference_host =
        require(context.cast(long_single_reference, DType::F32).to(tensor::Device::cpu()));
    const auto long_causal_host = require(context.cast(long_causal, DType::F32).to(tensor::Device::cpu()));
    const auto long_local_host = require(context.cast(long_local, DType::F32).to(tensor::Device::cpu()));
    const auto long_single_host = require(context.cast(long_single, DType::F32).to(tensor::Device::cpu()));
    const auto long_causal_reference_values = require(long_causal_reference_host.data<float>());
    const auto long_local_reference_values = require(long_local_reference_host.data<float>());
    const auto long_single_reference_values = require(long_single_reference_host.data<float>());
    const auto long_causal_values = require(long_causal_host.data<float>());
    const auto long_local_values = require(long_local_host.data<float>());
    const auto long_single_values = require(long_single_host.data<float>());
    for (std::size_t index = 0; index < long_causal_values.size(); ++index) {
        ASSERT_NEAR(long_causal_values[index], long_causal_reference_values[index], 1e-2F)
            << "long attention key-range parity failed index=" << index;
        ASSERT_NEAR(long_local_values[index], long_local_reference_values[index], 1e-2F)
            << "long attention key-range parity failed index=" << index;
        ASSERT_NEAR(long_single_values[index], long_single_reference_values[index], 1e-2F)
            << "long attention key-range parity failed index=" << index;
    }

    ops::Context f32_context(device, false, true, DType::F32);
    const auto long_query_f32 =
        require(Tensor::from_host({1, 2, 64}, std::span<const float>(long_query_values), device));
    const auto long_key_f32 = require(
        Tensor::from_host({1, static_cast<std::int64_t>(long_keys), 64},
                          std::span<const float>(long_key_values), device));
    const auto long_value_f32 = require(
        Tensor::from_host({1, static_cast<std::int64_t>(long_keys), 64},
                          std::span<const float>(long_value_values), device));
    const auto long_causal_mask_f32 =
        require(Tensor::from_host({1, 1, 2, static_cast<std::int64_t>(long_keys)},
                                  std::span<const float>(long_causal_mask), device));
    const auto long_causal_reference_f32 =
        f32_context.grouped_query_attention(long_query_f32, long_key_f32, long_value_f32, 1, 1,
                                            long_causal_mask_f32);
    const auto long_causal_f32 =
        f32_context.grouped_query_attention_causal(long_query_f32, long_key_f32, long_value_f32, 1, 1,
                                                   long_positions, 0);
    const auto long_causal_reference_f32_host =
        require(long_causal_reference_f32.to(tensor::Device::cpu()));
    const auto long_causal_f32_host = require(long_causal_f32.to(tensor::Device::cpu()));
    const auto long_causal_reference_f32_values = require(long_causal_reference_f32_host.data<float>());
    const auto long_causal_f32_values = require(long_causal_f32_host.data<float>());
    for (std::size_t index = 0; index < long_causal_f32_values.size(); ++index) {
        ASSERT_TRUE(std::isfinite(long_causal_f32_values[index]))
            << "FP32 long causal attention parity failed index=" << index;
        ASSERT_NEAR(long_causal_f32_values[index], long_causal_reference_f32_values[index], 1e-2F)
            << "FP32 long causal attention parity failed index=" << index;
    }

    constexpr std::size_t key_start = 37;
    constexpr std::size_t cropped_keys = long_keys - key_start;
    std::array<float, 2 * cropped_keys> cropped_local_mask{};
    for (std::size_t row = 0; row < long_query_positions.size(); ++row) {
        const auto position = long_query_positions[row];
        for (std::size_t index = 0; index < cropped_keys; ++index) {
            const auto key_position = static_cast<std::int64_t>(key_start + index);
            cropped_local_mask[row * cropped_keys + index] =
                key_position <= position && position - key_position < 256 ? 0.F : -1e9F;
        }
    }
    const auto cropped_mask = require(
        Tensor::from_host({1, 1, 2, static_cast<std::int64_t>(cropped_keys)},
                          std::span<const float>(cropped_local_mask), device));
    const auto cropped_reference = f32_context.grouped_query_attention(
        long_query_f32, long_key_f32, long_value_f32, 1, 1, cropped_mask, 1.F, key_start);
    const auto cropped_actual = f32_context.grouped_query_attention_causal(
        long_query_f32, long_key_f32, long_value_f32, 1, 1, long_positions, 256, 1.F, key_start);
    const auto cropped_reference_host = require(cropped_reference.to(tensor::Device::cpu()));
    const auto cropped_actual_host = require(cropped_actual.to(tensor::Device::cpu()));
    const auto cropped_reference_values = require(cropped_reference_host.data<float>());
    const auto cropped_actual_values = require(cropped_actual_host.data<float>());
    for (std::size_t index = 0; index < cropped_actual_values.size(); ++index) {
        ASSERT_TRUE(std::isfinite(cropped_actual_values[index]))
            << "FP32 cropped window attention parity failed index=" << index;
        ASSERT_NEAR(cropped_actual_values[index], cropped_reference_values[index], 1e-2F)
            << "FP32 cropped window attention parity failed index=" << index;
    }
}

TEST_F(CudaOpsTest, AsyncReadback) {
    SCOPED_TRACE("CUDA nonblocking readback snapshot failed");
    using namespace kidi;
    using ops::require;
    using tensor::Tensor;
    const std::array<std::int32_t, 4> values{1, 2, 3, 4};
    const auto host = require(Tensor::from_host({4}, std::span<const std::int32_t>(values)));
    const auto immediate = require(host.readback());
    ASSERT_TRUE(require(immediate->ready()));
    ASSERT_EQ(require(immediate->bytes()).size(), sizeof(values));
    ops::Context context(tensor::Device::cuda());
    auto input = require(host.to(context.device()));
    std::vector<std::shared_ptr<tensor::Readback>> pending;
    for (int iteration = 0; iteration < 8; ++iteration) {
        input = context.add(input, input);
        const auto middle = require(input.narrow(0, 1, 2));
        auto ticket = require(middle.readback());
        require(ticket->ready());
        pending.push_back(std::move(ticket));
    }
    input = {};
    context.synchronize();
    for (std::size_t index = 0; index < pending.size(); ++index) {
        ASSERT_TRUE(require(pending[index]->ready())) << "index=" << index;
        const auto bytes = require(pending[index]->bytes());
        std::array<std::int32_t, 2> actual{};
        std::memcpy(actual.data(), bytes.data(), bytes.size());
        const auto multiplier = std::int32_t{1} << (index + 1);
        ASSERT_EQ(actual, (std::array<std::int32_t, 2>{2 * multiplier, 3 * multiplier})) << "index=" << index;
    }
}

TEST_F(CudaOpsTest, ProjectionGroups) {
    SCOPED_TRACE("CUDA shared-input projection fusion failed");
    using namespace kidi;
    using ops::require;
    using tensor::DType;
    using tensor::Tensor;
    const auto device = tensor::Device::cuda();
    ops::Context context(device, false, true, DType::BF16), eager(device, false, false, DType::BF16);
    std::vector<Tensor> weights, scales;
    for (const int columns : {64, 128, 256}) {
        std::vector<float> values(columns * 256);
        for (std::size_t index = 0; index < values.size(); ++index)
            values[index] = (static_cast<int>(index * 11 % 17) - 8) * 0.03125F;
        weights.push_back(context.cast(
            require(Tensor::from_host({columns, 256}, std::span<const float>(values), device)), DType::BF16));
        scales.push_back(
            require(Tensor::from_host({columns}, std::span<const float>(std::vector<float>(columns, 1.F)), device)));
    }
    const auto scale = require(Tensor::from_host({256}, std::span<const float>(std::vector<float>(256, 1.F)), device));
    std::size_t captures = 0;
    for (int iteration = 0; iteration < 5; ++iteration) {
        std::array<float, 256> values{};
        for (std::size_t index = 0; index < values.size(); ++index)
            values[index] = (static_cast<int>((index + iteration) * 7 % 23) - 11) * 0.125F;
        const auto input =
            context.cast(require(Tensor::from_host({1, 1, 256}, std::span<const float>(values), device)), DType::BF16);
        const std::array arguments{input};
        const auto outputs = context.replay("cuda-projections", arguments, [&](std::span<const Tensor> operands) {
            ++captures;
            const auto normalized = context.rms_norm(operands[0], scale, 1e-6F);
            std::vector<Tensor> result;
            for (std::size_t index = 0; index < weights.size(); ++index)
                result.push_back(
                    context.rms_norm(context.linear(normalized, weights[index], {}, true), scales[index], 1e-6F));
            return result;
        });
        const auto normalized = eager.rms_norm(input, scale, 1e-6F);
        for (std::size_t index = 0; index < weights.size(); ++index) {
            const auto expected =
                eager.rms_norm(eager.linear(normalized, weights[index], {}, true), scales[index], 1e-6F);
            ASSERT_EQ(require(outputs[index].copy_to_host()), require(expected.copy_to_host()))
                << "iteration=" << iteration << " projection=" << index;
        }
    }
    EXPECT_EQ(captures, 2U);
}

TEST_F(CudaOpsTest, W8A16ProjectionGroups) {
    SCOPED_TRACE("CUDA shared-input W8A16 projection fusion failed");
    using namespace kidi;
    using ops::require;
    using tensor::DType;
    using tensor::Tensor;
    const auto device = tensor::Device::cuda();
    for (const std::int32_t group_size : {256, 128}) {
        ops::Context context(device, false, true, DType::BF16), eager(device, false, false, DType::BF16);
        context.set_precision(core::InferencePrecision::Q8);
        eager.set_precision(core::InferencePrecision::Q8);
        const std::int64_t groups = 256 / group_size;
        std::vector<Tensor> weights, proj_scales, norm_scales;
        for (const int columns : {64, 128, 256}) {
            std::vector<std::uint8_t> w(columns * 256);
            for (std::size_t index = 0; index < w.size(); ++index)
                w[index] = static_cast<std::uint8_t>(static_cast<int>(index * 13 % 251) - 125);
            weights.push_back(
                require(Tensor::from_bytes({columns, 256}, DType::U8, std::as_bytes(std::span(w)), device)));
            std::vector<float> pscales(columns * groups);
            for (std::size_t index = 0; index < pscales.size(); ++index) pscales[index] = ((index % 7) + 1) * 0.015625F;
            proj_scales.push_back(
                require(Tensor::from_host({columns, groups}, std::span<const float>(pscales), device)));
            norm_scales.push_back(require(
                Tensor::from_host({columns}, std::span<const float>(std::vector<float>(columns, 1.F)), device)));
        }
        const auto scale =
            require(Tensor::from_host({256}, std::span<const float>(std::vector<float>(256, 1.F)), device));
        std::size_t captures = 0;
        for (int iteration = 0; iteration < 5; ++iteration) {
            std::array<float, 256> values{};
            for (std::size_t index = 0; index < values.size(); ++index)
                values[index] = (static_cast<int>((index + iteration) * 7 % 23) - 11) * 0.125F;
            const auto input = context.cast(
                require(Tensor::from_host({1, 1, 256}, std::span<const float>(values), device)), DType::BF16);
            const std::array arguments{input};
            const auto outputs =
                context.replay("cuda-w8a16-projections", arguments, [&](std::span<const Tensor> operands) {
                    ++captures;
                    const auto normalized = context.rms_norm(operands[0], scale, 1e-6F);
                    std::vector<Tensor> result;
                    for (std::size_t index = 0; index < weights.size(); ++index)
                        result.push_back(
                            context.rms_norm(context.packed_linear(normalized, weights[index], proj_scales[index], 8,
                                                                   group_size, 0.F, 0.F),
                                             norm_scales[index], 1e-6F));
                    return result;
                });
            const auto normalized = eager.rms_norm(input, scale, 1e-6F);
            for (std::size_t index = 0; index < weights.size(); ++index) {
                const auto expected = eager.rms_norm(
                    eager.packed_linear(normalized, weights[index], proj_scales[index], 8, group_size, 0.F, 0.F),
                    norm_scales[index], 1e-6F);
                ASSERT_EQ(require(outputs[index].copy_to_host()), require(expected.copy_to_host()))
                    << "group_size=" << group_size << " iteration=" << iteration << " projection=" << index;
            }
        }
        ASSERT_EQ(captures, 2U) << "group_size=" << group_size;
    }
    // Verify F32 W8A8 sibling projection fusion.
    {
        ops::Context context(device, false, true), eager(device, false, false);
        std::vector<Tensor> weights, proj_scales, norm_scales;
        for (const int columns : {64, 128, 256}) {
            std::vector<std::uint8_t> w(columns * 256);
            for (std::size_t index = 0; index < w.size(); ++index)
                w[index] = static_cast<std::uint8_t>(static_cast<int>(index * 13 % 251) - 125);
            weights.push_back(
                require(Tensor::from_bytes({columns, 256}, DType::U8, std::as_bytes(std::span(w)), device)));
            std::vector<float> pscales(columns);
            for (std::size_t index = 0; index < pscales.size(); ++index) pscales[index] = ((index % 7) + 1) * 0.015625F;
            proj_scales.push_back(require(Tensor::from_host({columns, 1}, std::span<const float>(pscales), device)));
            norm_scales.push_back(require(
                Tensor::from_host({columns}, std::span<const float>(std::vector<float>(columns, 1.F)), device)));
        }
        const auto scale =
            require(Tensor::from_host({256}, std::span<const float>(std::vector<float>(256, 1.F)), device));
        std::size_t captures = 0;
        for (int iteration = 0; iteration < 5; ++iteration) {
            std::array<float, 256> values{};
            for (std::size_t index = 0; index < values.size(); ++index)
                values[index] = (static_cast<int>((index + iteration) * 7 % 23) - 11) * 0.125F;
            const auto input = require(Tensor::from_host({1, 1, 256}, std::span<const float>(values), device));
            const std::array arguments{input};
            const auto outputs =
                context.replay("cuda-w8a8-projections", arguments, [&](std::span<const Tensor> operands) {
                    ++captures;
                    const auto normalized = context.rms_norm(operands[0], scale, 1e-6F);
                    std::vector<Tensor> result;
                    for (std::size_t index = 0; index < weights.size(); ++index)
                        result.push_back(context.rms_norm(
                            context.packed_linear(normalized, weights[index], proj_scales[index], 8, 256, 0.F, 0.F),
                            norm_scales[index], 1e-6F));
                    return result;
                });
            const auto normalized = eager.rms_norm(input, scale, 1e-6F);
            for (std::size_t index = 0; index < weights.size(); ++index) {
                const auto expected = eager.rms_norm(
                    eager.packed_linear(normalized, weights[index], proj_scales[index], 8, 256, 0.F, 0.F),
                    norm_scales[index], 1e-6F);
                ASSERT_EQ(require(outputs[index].copy_to_host()), require(expected.copy_to_host()))
                    << "F32 W8A8 iteration=" << iteration << " projection=" << index;
            }
        }
        EXPECT_EQ(captures, 2U) << "F32 W8A8";
    }
}

// A fused RMS-norm or GELU-gate prologue must quantize exactly like the separate operation it replaces.
TEST_F(CudaOpsTest, ProjectionPrologues) {
    SCOPED_TRACE("CUDA fused projection prologue changed output");
    using namespace kidi;
    using ops::require;
    using tensor::DType;
    using tensor::Tensor;
    const auto device = tensor::Device::cuda();
    ops::Context context(device);
    constexpr std::int64_t rows = 64, columns = 128;
    const auto filled = [&](std::vector<std::int64_t> shape, std::size_t count, float step) {
        std::vector<float> values(count);
        for (std::size_t index = 0; index < count; ++index)
            values[index] = (static_cast<int>(index * 37 % 101) - 50) * step;
        return require(Tensor::from_host(shape, std::span<const float>(values), device));
    };
    using Kind = ops::ProjectionPrologue::Kind;
    for (const auto& [kind, width] : {std::pair{Kind::RMS_NORM, 512}, {Kind::RMS_NORM, 3072}, {Kind::GELU_GATE, 1024}}) {
        const bool gate = kind == Kind::GELU_GATE;
        const auto input_width = gate ? 2 * width : width;
        const auto input = filled({1, rows, input_width}, rows * input_width, 0.03125F);
        const auto norm = filled({width}, width, 0.015625F);
        std::vector<std::uint8_t> packed(columns * width);
        for (std::size_t index = 0; index < packed.size(); ++index)
            packed[index] = static_cast<std::uint8_t>(index * 29 % 251);
        const auto weight =
            require(Tensor::from_bytes({columns, width}, DType::U8, std::as_bytes(std::span(packed)), device));
        std::vector<float> scale_values(columns);
        for (std::size_t index = 0; index < scale_values.size(); ++index) scale_values[index] = (index % 5 + 1) * 0.01F;
        const auto scales = require(Tensor::from_host({columns, 1}, std::span<const float>(scale_values), device));
        const ops::ProjectionPrologue prologue{.kind = kind, .scale = gate ? nullptr : &norm, .epsilon = 1e-6F};
        ASSERT_TRUE(context.fuses(input, prologue)) << "width=" << width;
        for (const float input_scale : {0.F, 0.05F}) {
            const auto fused = context.packed_linear(input, weight, scales, 8, width, input_scale, 0.F, {}, prologue);
            const auto expected =
                context.packed_linear(context.apply(input, prologue), weight, scales, 8, width, input_scale, 0.F);
            ASSERT_EQ(require(fused.copy_to_host()), require(expected.copy_to_host()))
                << "width=" << width << " input_scale=" << input_scale;
        }
    }
}

TEST_F(CudaOpsTest, InputStaging) {
    SCOPED_TRACE("CUDA reusable input staging failed");
    using namespace kidi;
    using ops::require;
    using tensor::Tensor;
    const auto device = tensor::Device::cuda();
    ops::Context context(device);
    std::array<float, 8> values{1, 2, 3, 4, 5, 6, 7, 8};
    const std::array<std::int32_t, 3> ids{2, 7, 9};
    auto input = require(Tensor::from_host({2, 4}, std::span<const float>(values)));
    const auto indices = require(Tensor::from_host({3}, std::span<const std::int32_t>(ids)));
    std::size_t stable_bytes = 0;
    for (int iteration = 0; iteration < 5; ++iteration) {
        require(input.copy_from_host(std::as_bytes(std::span(values))));
        const std::array operands{input, indices};
        const auto staged = context.stage_inputs(operands);
        std::array<float, 8> overwritten{};
        require(input.copy_from_host(std::as_bytes(std::span(overwritten))));
        const auto copied = require(staged[0].to(tensor::Device::cpu()));
        const auto copied_ids = require(staged[1].to(tensor::Device::cpu()));
        ASSERT_TRUE(std::ranges::equal(require(copied.data<float>()), values)) << "iteration=" << iteration;
        ASSERT_TRUE(std::ranges::equal(require(copied_ids.data<std::int32_t>()), ids)) << "iteration=" << iteration;
        const auto already = context.stage_inputs(staged);
        ASSERT_EQ(already[0].storage_identity(), staged[0].storage_identity()) << "iteration=" << iteration;
        if (iteration == 1) stable_bytes = kidi::cuda::allocated_bytes();
        if (iteration > 1) ASSERT_EQ(kidi::cuda::allocated_bytes(), stable_bytes) << "iteration=" << iteration;
        for (auto& value : values) value += 1.F;
    }
}

TEST_F(CudaOpsTest, DynamicProjection) {
    SCOPED_TRACE("CUDA fusion froze dynamic step weights");
    using namespace kidi;
    using ops::require;
    using tensor::DType;
    using tensor::Tensor;
    const auto device = tensor::Device::cuda();
    ops::Context context(device, false, true, DType::BF16);
    const auto input = context.cast(
        require(Tensor::from_host({1, 1, 256}, std::span<const float>(std::vector<float>(256, 1.F)), device)),
        DType::BF16);
    const auto scale = require(Tensor::from_host({256}, std::span<const float>(std::vector<float>(256, 1.F)), device));
    auto first = require(Tensor::empty({64, 256}, DType::BF16, device));
    const auto second = context.cast(
        require(Tensor::from_host({64, 256}, std::span<const float>(std::vector<float>(64 * 256, 0.5F)), device)),
        DType::BF16);
    std::size_t captures = 0;
    for (int iteration = 0; iteration < 5; ++iteration) {
        const std::vector<std::uint16_t> weights(64 * 256, static_cast<std::uint16_t>(0x3f80 + iteration * 128));
        require(first.copy_from_host(std::as_bytes(std::span(weights))));
        const std::array arguments{input, first, second};
        const auto outputs =
            context.replay("cuda-dynamic-projections", arguments, [&](std::span<const Tensor> operands) {
                ++captures;
                const auto normalized = context.rms_norm(operands[0], scale, 1e-6F);
                return std::vector{context.add(context.linear(normalized, operands[1], {}, true),
                                               context.linear(normalized, operands[2], {}, true))};
            });
        const auto copied = require(context.cast(outputs[0], DType::F32).to(tensor::Device::cpu()));
        for (const auto value : require(copied.data<float>()))
            ASSERT_EQ(value, 256.F * (static_cast<float>(1 << iteration) + 0.5F)) << "iteration=" << iteration;
    }
    EXPECT_EQ(captures, 2U);
}

TEST_F(CudaOpsTest, ExportedCacheAlias) {
    SCOPED_TRACE("CUDA graph exported a stale cache alias");
    using namespace kidi;
    using ops::require;
    using tensor::DType;
    using tensor::Tensor;
    const auto device = tensor::Device::cuda();
    ops::Context context(device, false, true);
    const auto index =
        require(Tensor::from_host({1}, std::span<const std::int32_t>(std::array<std::int32_t, 1>{2}), device));
    std::size_t captures = 0;
    for (int iteration = 0; iteration < 5; ++iteration) {
        const auto cache = require(Tensor::zeros({1, 4, 1}, DType::F32, device));
        const auto update =
            require(Tensor::from_host({1, 1, 1}, std::span<const float>(std::array{iteration + 1.F}), device));
        const std::array arguments{cache, update, index};
        const auto outputs = context.replay("cuda-exported-cache", arguments, [&](std::span<const Tensor> operands) {
            ++captures;
            auto destination = operands[0];
            context.scatter_(destination, operands[1], operands[2]);
            return std::vector{destination};
        });
        const auto copied = require(outputs[0].to(tensor::Device::cpu()));
        const auto updated = require(cache.to(tensor::Device::cpu()));
        const auto actual = require(copied.data<float>());
        ASSERT_TRUE(std::ranges::equal(actual, require(updated.data<float>()))) << "iteration=" << iteration;
        ASSERT_EQ(actual[2], iteration + 1.F) << "iteration=" << iteration;
    }
    EXPECT_EQ(captures, 2U);
}

TEST_F(CudaOpsTest, ConstantCacheRelease) {
    SCOPED_TRACE("CUDA workspace release retained unused constants");
    using namespace kidi;
    using ops::require;
    using tensor::Tensor;
    const auto device = tensor::Device::cuda();
    ops::Context context(device);
    const auto input =
        require(Tensor::from_host({1, 4}, std::span<const float>(std::array{1.F, 2.F, 3.F, 4.F}), device));
    const auto weight =
        require(Tensor::from_host({2, 4}, std::span<const float>(std::array{1.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F, 1.F})));
    const auto bias = require(Tensor::from_host({2}, std::span<const float>(std::array{2.F, 3.F})));
    const auto before = kidi::cuda::allocated_bytes();
    {
        const auto output = context.linear(input, weight, bias, true);
        const auto copied = require(output.to(tensor::Device::cpu()));
        ASSERT_TRUE(std::ranges::equal(require(copied.data<float>()), std::array{3.F, 7.F}));
    }
    ASSERT_GT(kidi::cuda::allocated_bytes(), before);
    context.release_workspaces();
    EXPECT_EQ(kidi::cuda::allocated_bytes(), before);
}

TEST_F(CudaOpsTest, ReadbackSourceRelease) {
    SCOPED_TRACE("CUDA readback retained its completed source");
    using namespace kidi;
    using ops::require;
    using tensor::DType;
    using tensor::Tensor;
    const auto device = tensor::Device::cuda();
    ops::Context context(device);
    const auto before = kidi::cuda::allocated_bytes();
    std::shared_ptr<tensor::Readback> pending;
    {
        const auto source = require(Tensor::zeros({65536}, DType::I32, device));
        pending = require(require(source.narrow(0, 1, 1)).readback());
    }
    context.synchronize();
    ASSERT_TRUE(require(pending->ready()));
    ASSERT_EQ(require(pending->bytes()).size(), sizeof(std::int32_t));
    EXPECT_EQ(kidi::cuda::allocated_bytes(), before);
}

TEST_F(CudaOpsTest, GreedyPartitions) {
    SCOPED_TRACE("CUDA parallel token selection changed tie-breaking");
    using namespace kidi;
    using ops::require;
    using tensor::DType;
    using tensor::Tensor;
    const auto device = tensor::Device::cuda();
    ops::Context context(device, false, false, DType::BF16);
    for (const std::size_t width : {4096, 4097, 9000}) {
        std::vector<float> values(2 * width, -100.F);
        values[1999] = values[width - 1] = 42.F;
        values[2 * width - 1] = 1.F;
        const auto source =
            require(Tensor::from_host({2, static_cast<std::int64_t>(width)}, std::span<const float>(values), device));
        for (const auto dtype : {DType::F32, DType::BF16}) {
            const auto token = context.greedy_token(context.cast(source, dtype));
            const auto actual = require(token.data<std::int32_t>());
            ASSERT_EQ(actual[0], 1999) << "width=" << width;
            ASSERT_EQ(actual[1], static_cast<std::int32_t>(width - 1)) << "width=" << width;
        }
        std::fill(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(width), -INFINITY);
        std::fill(values.begin() + static_cast<std::ptrdiff_t>(width), values.end(), NAN);
        for (const bool positive_infinity : {false, true}) {
            if (positive_infinity) {
                std::fill(values.begin(), values.end(), -100.F);
                values[width - 1] = INFINITY;
                values[width + width / 2] = NAN;
            }
            const auto special = require(
                Tensor::from_host({2, static_cast<std::int64_t>(width)}, std::span<const float>(values), device));
            for (const auto dtype : {DType::F32, DType::BF16}) {
                const auto selected = context.greedy_token(context.cast(special, dtype));
                const auto actual = require(selected.data<std::int32_t>());
                ASSERT_EQ(actual[0], -1) << "width=" << width << " positive_infinity=" << positive_infinity;
                ASSERT_EQ(actual[1], -1) << "width=" << width << " positive_infinity=" << positive_infinity;
            }
        }
    }
}

TEST_F(CudaOpsTest, BfloatFusions) {
    SCOPED_TRACE("CUDA BF16 fusion changed intermediate rounding");
    using namespace kidi;
    using ops::require;
    using tensor::DType;
    using tensor::Tensor;
    const auto device = tensor::Device::cuda();
    ops::Context context(device, false, false, DType::BF16);
    std::array<float, 2 * 3 * 4 * 64> values{}, residuals{};
    std::array<float, 64> scales{};
    std::array<float, 3 * 32> cosines{}, sines{};
    for (std::size_t index = 0; index < values.size(); ++index) {
        values[index] = (static_cast<int>(index * 17 % 97) - 49) * 0.017F;
        residuals[index] = (static_cast<int>(index * 31 % 73) - 37) * 0.029F;
    }
    for (std::size_t index = 0; index < scales.size(); ++index) scales[index] = (index % 7 + 1) * 0.125F;
    for (std::size_t index = 0; index < cosines.size(); ++index) {
        cosines[index] = std::cos(static_cast<float>(index) * 0.013F);
        sines[index] = std::sin(static_cast<float>(index) * 0.013F);
    }
    const auto input =
        context.cast(require(Tensor::from_host({2, 3, 4, 64}, std::span<const float>(values), device)), DType::BF16);
    const auto residual =
        context.cast(require(Tensor::from_host({2, 3, 4, 64}, std::span<const float>(residuals), device)), DType::BF16);
    const auto scale = require(Tensor::from_host({64}, std::span<const float>(scales), device));
    const auto cosine = require(Tensor::from_host({1, 3, 1, 32}, std::span<const float>(cosines), device));
    const auto sine = require(Tensor::from_host({1, 3, 1, 32}, std::span<const float>(sines), device));
    const auto scalar = require(Tensor::from_host({}, std::span<const float>(std::array{0.78125F}), device));
    const auto normalized = context.rms_norm(input, scale, 1e-6F);
    const auto rotary = context.rms_rotary(input, scale, cosine, sine, 1e-6F);
    const auto rotary_reference = context.rotary(normalized, cosine, sine);
    EXPECT_EQ(require(rotary.copy_to_host()), require(rotary_reference.copy_to_host()));
    for (const bool scaled : {false, true}) {
        const auto fused = context.rms_norm_residual(input, scale, residual, 1e-6F, scaled ? scalar : Tensor{});
        const auto added = context.add(normalized, residual);
        const auto reference = scaled ? context.multiply(added, scalar) : added;
        EXPECT_EQ(require(fused.copy_to_host()), require(reference.copy_to_host())) << "scaled=" << scaled;
    }
    const auto gate = context.slice(input, -1, 0, 32), up = context.slice(input, -1, 32, 32);
    EXPECT_EQ(require(context.gelu_gate(input).copy_to_host()), require(context.gelu_multiply(gate, up).copy_to_host()));
    EXPECT_EQ(require(context.silu_gate(input).copy_to_host()), require(context.silu_multiply(gate, up).copy_to_host()));
}

TEST_F(CudaOpsTest, CudaGraph) {
    SCOPED_TRACE("CUDA graph input staging or cache writeback failed");
    using namespace kidi;
    using ops::require;
    using tensor::DType;
    using tensor::Tensor;
    const auto device = tensor::Device::cuda();
    ops::Context context(device, false, true, DType::BF16);
    std::vector<float> parameters(33 * 256, 0.F);
    for (std::size_t column = 0; column < 33; ++column) parameters[column * 256 + column] = 1.F;
    const auto weight =
        context.cast(require(Tensor::from_host({33, 256}, std::span<const float>(parameters), device)), DType::BF16);
    auto cache = require(Tensor::zeros({1, 4, 33}, DType::BF16, device));
    std::size_t eager_calls = 0;
    Tensor retained;
    for (std::int32_t iteration = 0; iteration < 6; ++iteration) {
        if (iteration == 3) cache = require(Tensor::zeros({1, 4, 33}, DType::BF16, device));
        std::array<float, 256> values{};
        values[iteration] = static_cast<float>(iteration + 1);
        const auto input =
            context.cast(require(Tensor::from_host({1, 1, 256}, std::span<const float>(values), device)), DType::BF16);
        const std::array<std::int32_t, 1> row{iteration % 4};
        const auto index = require(Tensor::from_host({1}, std::span<const std::int32_t>(row), device));
        const std::array operands{input, index, cache};
        const auto outputs = context.replay("cuda-test", operands, [&](std::span<const Tensor> arguments) {
            ++eager_calls;
            auto projected = context.linear(arguments[0], weight, {}, true);
            auto destination = arguments[2];
            context.scatter_(destination, projected, arguments[1]);
            return std::vector{context.greedy_token(projected), projected};
        });
        context.synchronize();
        ASSERT_EQ(outputs.size(), 2U) << "iteration=" << iteration;
        ASSERT_EQ(require(outputs[0].data<std::int32_t>())[0], iteration);
        const auto copied = require(context.cast(cache, DType::F32).to(tensor::Device::cpu()));
        const auto actual = require(copied.data<float>());
        for (std::size_t column = 0; column < 33; ++column)
            ASSERT_EQ(actual[row[0] * 33 + column],
                      (column == static_cast<std::size_t>(iteration) ? iteration + 1.F : 0.F))
                << "iteration=" << iteration << " column=" << column;
        retained = outputs[1];
    }
    ASSERT_EQ(eager_calls, 2U);
    context.clear_replays("cuda-test");
    const auto result = require(context.cast(retained, DType::F32).to(tensor::Device::cpu()));
    EXPECT_EQ(require(result.data<float>())[5], 6.F);
}

TEST_F(CudaOpsTest, BfloatContext) {
    SCOPED_TRACE("CUDA BF16 context pipeline failed");
    using namespace kidi;
    using ops::require;
    using tensor::DType;
    using tensor::Tensor;
    ops::Context context(tensor::Device::cuda(), false, std::nullopt, DType::BF16);
    ASSERT_EQ(context.activation_dtype(), DType::BF16);
    const auto source = require(Tensor::from_host(
        {1, 2, 4}, std::span<const float>(std::array{1.F, 2.F, 3.F, 4.F, -1.F, -2.F, -3.F, -4.F}), context.device()));
    const auto input = context.cast(source, DType::BF16);
    const auto scale =
        require(Tensor::from_host({4}, std::span<const float>(std::array{1.F, 1.F, 1.F, 1.F}), context.device()));
    const auto scalar = require(Tensor::from_host({}, std::span<const float>(std::array{0.5F}), context.device()));
    auto hidden = context.multiply(context.rms_norm(input, scale, 1e-6F), scalar);
    ASSERT_EQ(hidden.dtype(), DType::BF16);
    hidden = context.rms_norm_residual(hidden, scale, input, 1e-6F);
    const auto weight = context.cast(
        require(Tensor::from_host({4, 4},
                                  std::span<const float>(std::array{1.F, 0.F, 0.F, 0.F, 0.F, 1.F, 0.F, 0.F, 0.F, 0.F,
                                                                    1.F, 0.F, 0.F, 0.F, 0.F, 1.F}),
                                  context.device())),
        DType::BF16);
    const auto projected = context.linear(hidden, weight, {}, true);
    const auto mask = require(Tensor::zeros({1, 1, 2, 2}, DType::F32, context.device()));
    const auto attended = context.grouped_query_attention(projected, projected, projected, 1, 1, mask);
    const auto token = context.greedy_token(attended);
    const auto copied = require(context.cast(attended, DType::F32).to(tensor::Device::cpu()));
    for (const auto value : require(copied.data<float>())) ASSERT_TRUE(std::isfinite(value));
    EXPECT_EQ(projected.dtype(), DType::BF16);
    EXPECT_EQ(attended.dtype(), DType::BF16);
    EXPECT_EQ(token.dtype(), DType::I32);
}

TEST_F(CudaOpsTest, ScatterUpdates) {
    SCOPED_TRACE("CUDA sparse cache scatter failed");
    using namespace kidi;
    using ops::require;
    using tensor::DType;
    using tensor::Tensor;
    ops::Context context(tensor::Device::cuda());
    const std::array<float, 24> values{1,  2,  3,  4,  5,  6,  7,  8,  0.333333F, 10, 11, 12,
                                     13, 14, 15, 16, 17, 18, 19, 20, 21,        22, 23, 24};
    const std::array<std::int32_t, 3> rows{5, 9215, 5};
    const auto updates = require(Tensor::from_host({2, 3, 4}, std::span<const float>(values), context.device()));
    const auto indices = require(Tensor::from_host({3}, std::span<const std::int32_t>(rows), context.device()));
    for (const auto dtype : {DType::F32, DType::BF16}) {
        auto cache = require(Tensor::zeros({2, 9216, 4}, dtype, context.device()));
        const auto identity = cache.storage_identity();
        const auto before = kidi::cuda::allocated_bytes();
        context.scatter_(cache, updates, indices);
        context.synchronize();
        ASSERT_EQ(cache.storage_identity(), identity);
        ASSERT_EQ(kidi::cuda::allocated_bytes(), before);
        const auto rounded = require(context.cast(context.cast(updates, dtype), DType::F32).to(tensor::Device::cpu()));
        const auto expected_values = require(rounded.data<float>());
        for (const bool inplace : {true, false}) {
            const auto result = inplace ? cache : context.scatter(cache, updates, indices);
            const auto copied = require(context.cast(result, DType::F32).to(tensor::Device::cpu()));
            const auto actual = require(copied.data<float>());
            for (std::size_t batch = 0; batch < 2; ++batch)
                for (std::size_t row = 0; row < 9216; ++row)
                    for (std::size_t channel = 0; channel < 4; ++channel) {
                        const auto expected = row == 5      ? expected_values[batch * 12 + 8 + channel]
                                              : row == 9215 ? expected_values[batch * 12 + 4 + channel]
                                                            : 0.F;
                        ASSERT_EQ(actual[(batch * 9216 + row) * 4 + channel], expected)
                            << "inplace=" << inplace << " batch=" << batch << " row=" << row
                            << " channel=" << channel;
                    }
        }
    }
    const std::array<std::int32_t, 3> unique_rows{1, 3, 5};
    const auto unique_indices = require(Tensor::from_host({3}, std::span<const std::int32_t>(unique_rows),
                                                                  context.device()));
    auto unique_cache = require(Tensor::zeros({2, 6, 4}, DType::F32, context.device()));
    context.scatter_unique_(unique_cache, updates, unique_indices);
    context.synchronize();
    const auto unique_result = require(unique_cache.to(tensor::Device::cpu()));
    const auto unique_values = require(unique_result.data<float>());
    for (std::size_t batch = 0; batch < 2; ++batch)
        for (std::size_t row = 0; row < 6; ++row)
                    for (std::size_t channel = 0; channel < 4; ++channel) {
                        const auto expected = row == 1 ? values[batch * 12 + channel]
                                              : row == 3 ? values[batch * 12 + 4 + channel]
                                              : row == 5 ? values[batch * 12 + 8 + channel]
                                                         : 0.F;
                        const auto index = (batch * 6 + row) * 4 + channel;
                        ASSERT_EQ(unique_values[index], expected) << "unique-index scatter mismatch index=" << index;
                    }
    std::array<float, 16> initial{};
    for (std::size_t index = 0; index < initial.size(); ++index) initial[index] = static_cast<float>(index + 1);
    auto aliased = require(Tensor::from_host({1, 4, 4}, std::span<const float>(initial), context.device()));
    const auto matrix = require(aliased.reshape({4, 4}));
    const auto alias = require(require(matrix.narrow(0, 0, 2)).reshape({1, 2, 4}));
    const std::array<std::int32_t, 2> targets{2, 3};
    const auto target = require(Tensor::from_host({2}, std::span<const std::int32_t>(targets), context.device()));
    context.scatter_(aliased, alias, target);
    context.synchronize();
    const auto alias_copy = require(aliased.to(tensor::Device::cpu()));
    const auto alias_values = require(alias_copy.data<float>());
    for (std::size_t index = 0; index < initial.size(); ++index)
        ASSERT_EQ(alias_values[index], initial[index % 8]) << "aliased scatter index=" << index;
}

TEST_F(CudaOpsTest, KeyValueScatterFusion) {
    SCOPED_TRACE("CUDA fused key/value scatter failed");
    using namespace kidi;
    using ops::require;
    using tensor::DType;
    using tensor::Tensor;
    const auto device = tensor::Device::cuda();
    ops::Context context(device, false, true), eager(device);
    const std::array<float, 12> key_values{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    const std::array<float, 12> value_values{-1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12};
    const std::array<std::int32_t, 3> rows{1, 4, 6};
    const auto key_updates = require(Tensor::from_host({1, 3, 4}, std::span<const float>(key_values), device));
    const auto value_updates = require(Tensor::from_host({1, 3, 4}, std::span<const float>(value_values), device));
    const auto indices = require(Tensor::from_host({3}, std::span<const std::int32_t>(rows), device));
    auto key_cache = require(Tensor::zeros({1, 8, 4}, DType::BF16, device));
    auto value_cache = require(Tensor::zeros({1, 8, 4}, DType::BF16, device));
    const std::array arguments{key_cache, value_cache, key_updates, value_updates, indices};
    const auto step = [&](std::span<const Tensor> operands) {
        auto key = operands[0], value = operands[1];
        context.scatter_unique_(key, operands[2], operands[4]);
        context.scatter_unique_(value, operands[3], operands[4]);
        return std::vector<Tensor>{key, value};
    };
    auto outputs = context.replay("cuda-key-value-scatter", arguments, step);
    outputs = context.replay("cuda-key-value-scatter", arguments, step);

    auto expected_key = require(Tensor::zeros({1, 8, 4}, DType::BF16, device));
    auto expected_value = require(Tensor::zeros({1, 8, 4}, DType::BF16, device));
    eager.scatter_(expected_key, key_updates, indices);
    eager.scatter_(expected_value, value_updates, indices);
    eager.synchronize();
    EXPECT_EQ(require(outputs[0].copy_to_host()), require(expected_key.copy_to_host()));
    EXPECT_EQ(require(outputs[1].copy_to_host()), require(expected_value.copy_to_host()));
}

TEST_F(CudaOpsTest, ScopedLayers) {
    SCOPED_TRACE("CUDA scoped placement or layer parity failed");
    using namespace kidi;
    using ops::require;
    using tensor::DType;
    using tensor::Tensor;
    const auto previous_device = module_device;
    const auto previous_dtype = module_dtype;
    const auto previous_allocate = allocate_parameters;
    {
        const auto device = tensor::Device::cuda();
        const ModuleScope selected(DType::F32, true, device);
        constexpr float EPSILON = 1e-5F;
        layers::RmsNorm rms(4, EPSILON);
        layers::LayerNorm norm(4, EPSILON);
        layers::Linear linear(4, 4);
        const std::array<const Module*, 3> modules{rms.get(), norm.get(), linear.get()};
        for (const auto* module : modules) {
            ASSERT_EQ(module->device(), device);
            for (const auto& [name, value] : module->state_dict()) {
                ASSERT_TRUE(value.defined()) << name;
                ASSERT_EQ(value.device(), device) << name;
            }
        }
        constexpr std::array VALUES{1.F, 2.F, 3.F, 4.F, -2.F, 0.F, 2.F, 4.F};
        constexpr std::array ONES{1.F, 1.F, 1.F, 1.F};
        constexpr std::array ZEROS{0.F, 0.F, 0.F, 0.F};
        const auto host = require(Tensor::from_host({2, 4}, std::span<const float>(VALUES)));
        const auto scale = require(Tensor::from_host({4}, std::span<const float>(ONES)));
        const auto bias = require(Tensor::from_host({4}, std::span<const float>(ZEROS)));
        const auto input = require(host.to(rms->device()));
        ops::Context cpu(tensor::Device::cpu()), context(rms->device());
        const auto same_values = [&](const Tensor& output, const Tensor& reference) -> ::testing::AssertionResult {
            if (output.device() != device) return ::testing::AssertionFailure() << "output left the selected device";
            context.synchronize();
            const auto copied = require(output.to(tensor::Device::cpu()));
            const auto actual = require(copied.data<float>()), expected = require(reference.data<float>());
            if (actual.size() != expected.size())
                return ::testing::AssertionFailure() << "size " << actual.size() << " != " << expected.size();
            for (std::size_t index = 0; index < actual.size(); ++index)
                if (std::abs(actual[index] - expected[index]) > 2e-5F)
                    return ::testing::AssertionFailure() << "index=" << index << " actual=" << actual[index]
                                                         << " expected=" << expected[index];
            return ::testing::AssertionSuccess();
        };
        ASSERT_TRUE(same_values(rms->forward(context, input), cpu.rms_norm(host, scale, EPSILON))) << "RMS norm";
        ASSERT_TRUE(same_values(norm->forward(context, input), cpu.layer_norm(host, scale, bias, EPSILON)))
            << "layer norm";
        {
            const ModuleScope quantized(DType::I8);
            layers::Linear projection(4, 4, true, true, 0, true);
            layers::Embedding embedding(8, 4, 8);
            layers::Conv1d convolution(4, 4, 3, 1, 1);
            layers::ConvTranspose1d transpose(4, 4, 4, 2, 1);
            layers::Snake1d snake(4);
            const std::array<const Module*, 5> quantized_modules{projection.get(), embedding.get(), convolution.get(),
                                                                 transpose.get(), snake.get()};
            for (const auto* module : quantized_modules) {
                ASSERT_EQ(module->device(), device);
                for (const auto& [name, value] : module->state_dict())
                    if (name.ends_with("scale")) {
                        const auto copied = require(value.to(tensor::Device::cpu()));
                        for (const auto scale_value : require(copied.data<float>())) ASSERT_EQ(scale_value, 1.F) << name;
                    }
            }
            const auto zeros = require(Tensor::zeros({2, 4}, DType::F32));
            ASSERT_TRUE(same_values(projection->forward(context, input), zeros)) << "quantized projection";
        }
        {
            const ModuleScope nested(tensor::Device::cpu());
            layers::RmsNorm child(4, EPSILON);
            ASSERT_EQ(child->device(), tensor::Device::cpu());
        }
        ASSERT_EQ(module_device, rms->device());
        bool rejected = false;
        try {
            rms->forward(context, host);
        } catch (const ops::Failure& error) {
            rejected = error.error().code == ErrorCode::INVALID_ARGUMENT;
        }
        ASSERT_TRUE(rejected) << "host input to a CUDA layer was not rejected with INVALID_ARGUMENT";
        std::array<float, 32> weights{};
        for (std::size_t index = 0; index < weights.size(); ++index) weights[index] = static_cast<float>(index + 1);
        const auto table = require(Tensor::from_host({8, 4}, std::span<const float>(weights)));
        constexpr std::array<std::int32_t, 4> TOKENS{0, 1, 2, 3};
        const auto tokens = require(Tensor::from_host({2, 2}, std::span<const std::int32_t>(TOKENS), device));
        constexpr std::array EXPECTED{26.F, 28.F, 30.F, 32.F, 34.F, 36.F, 38.F, 40.F};
        const auto reference = require(Tensor::from_host({1, 2, 4}, std::span<const float>(EXPECTED)));
        ASSERT_TRUE(same_values(context.embedding_sum(tokens, table, {}, 4, 0, 1.F), reference)) << "embedding sum";
    }
    EXPECT_EQ(module_device, previous_device);
    EXPECT_EQ(module_dtype, previous_dtype);
    EXPECT_EQ(allocate_parameters, previous_allocate);
}

TEST_F(CudaOpsTest, DiffusionFeedback) {
    SCOPED_TRACE("CUDA diffusion embedding feedback failed");
    using namespace kidi;
    using ops::require;
    using tensor::Tensor;
    ops::Context context(tensor::Device::cuda());
    constexpr std::array<std::int32_t, 4> MASKED{3, 3, 3, 3};
    constexpr std::array LOGITS{0.F, 8.F, 0.F, -9.F, 0.F, 0.F, 6.F, -9.F, 5.F, 0.F, 0.F, -9.F, 0.F, 4.F, 0.F, -9.F};
    constexpr std::array NOISE{0.F};
    const auto tokens = require(Tensor::from_host({2, 2}, std::span<const std::int32_t>(MASKED), context.device()));
    const auto logits = require(Tensor::from_host({1, 2, 8}, std::span<const float>(LOGITS), context.device()));
    const auto noise = require(Tensor::from_host({1}, std::span<const float>(NOISE), context.device()));
    const auto candidates = context.diffusion_candidates(logits, 2, 2, 4, 3, 0.F);
    const auto partial = context.diffusion_update(tokens, candidates, noise, 2, 3, 0.F, 0.F);
    const auto copied_partial = require(partial.to(tensor::Device::cpu()));
    ASSERT_TRUE(std::ranges::equal(require(copied_partial.data<std::int32_t>()), std::array<std::int32_t, 4>{1, 3, 2, 3}));
    const auto completed = context.diffusion_update(partial, candidates, noise, 2, 3, 0.F, 0.F);
    const auto copied_tokens = require(completed.to(tensor::Device::cpu()));
    ASSERT_EQ(completed.device(), context.device());
    ASSERT_TRUE(std::ranges::equal(require(copied_tokens.data<std::int32_t>()), std::array<std::int32_t, 4>{1, 0, 2, 1}));
    constexpr std::array<std::int8_t, 16> WEIGHTS{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    constexpr std::array SCALES{0.5F, 0.5F, 0.5F, 0.5F, 0.5F, 0.5F, 0.5F, 0.5F};
    const auto weight = require(Tensor::from_host({8, 2}, std::span<const std::int8_t>(WEIGHTS)));
    const auto scales = require(Tensor::from_host({8, 1}, std::span<const float>(SCALES)));
    const auto embedding = context.embedding_sum(completed, weight, scales, 2, 8, 1.F);
    const auto copied_embedding = require(embedding.to(tensor::Device::cpu()));
    EXPECT_EQ(embedding.device(), context.device());
    EXPECT_TRUE(std::ranges::equal(embedding.shape(), std::array<std::int64_t, 3>{1, 2, 2}));
    EXPECT_TRUE(std::ranges::equal(require(copied_embedding.data<float>()), std::array{8.F, 9.F, 6.F, 7.F}));
}

TEST_F(CudaOpsTest, ProjectionPaths) {
    SCOPED_TRACE("accelerated projection CPU parity failed");
    using namespace kidi;
    using ops::require;
    using tensor::DType;
    using tensor::Tensor;
    ops::Context cpu;
    ops::Context cuda(tensor::Device::cuda());
    constexpr std::int64_t COLUMNS = 8;
    for (const std::int64_t width : {16, 17}) {
        for (const std::int64_t rows : {1, 8}) {
            std::vector<float> values(rows * width), scales(COLUMNS), biases(COLUMNS);
            for (std::size_t index = 0; index < values.size(); ++index)
                values[index] = index / width == 2   ? 0.F
                                : index / width == 3 ? (index % 2 ? -1000.F : 1000.F)
                                                     : (static_cast<int>(index * 13 % 41) - 17) * 0.125F;
            if (rows > 6) {
                constexpr std::array TIES{-126.5F, 128.5F, 0.5F, -0.5F, 1.5F, -1.5F, 2.5F, -2.5F};
                std::ranges::copy(TIES, values.begin() + 5 * width);
                // This zero point changes when the multiply and add are fused.
                values[6 * width] = -6901.607421875F;
                values[6 * width + 1] = 2637.199462890625F;
            }
            for (std::size_t index = 0; index < scales.size(); ++index) {
                scales[index] = static_cast<float>(index + 1) * 0.001F;
                biases[index] = static_cast<float>(index) * 0.01F;
            }
            const auto host = require(Tensor::from_host({rows, width}, std::span<const float>(values)));
            const auto input = require(host.to(tensor::Device::cuda()));
            const auto scale = require(Tensor::from_host({COLUMNS, 1}, std::span<const float>(scales)));
            const auto bias = require(Tensor::from_host({COLUMNS}, std::span<const float>(biases)));
            for (const int bits : {0, 2, 4, 8}) {
                if (bits && width % (8 / bits)) continue;
                std::vector<std::uint8_t> weights(COLUMNS * width / (bits ? 8 / bits : 1), 0);
                for (std::size_t index = 0; index < COLUMNS * width; ++index) {
                    if (!bits)
                        weights[index] = static_cast<std::uint8_t>(static_cast<int>(index * 11 % 255) - 127);
                    else {
                        const auto value =
                            static_cast<unsigned int>(static_cast<int>(index * 7 % (1 << bits)) - (1 << (bits - 1))) &
                            ((1 << bits) - 1);
                        weights[index / (8 / bits)] |= value << (index % (8 / bits) * bits);
                    }
                }
                const auto weight =
                    require(Tensor::from_bytes({COLUMNS, width / (bits ? 8 / bits : 1)}, bits ? DType::U8 : DType::I8,
                                               std::as_bytes(std::span(weights))));
                for (const float input_scale : {0.F, 0.125F}) {
                    if (!bits && input_scale) continue;
                    const float output_scale = input_scale ? 0.0625F : 0.F;
                    const auto expected =
                        bits ? cpu.packed_linear(host, weight, scale, bits, width, input_scale, output_scale)
                             : cpu.quantized_linear(host, weight, scale, bias, true);
                    for (int repetition = 0; repetition < 2; ++repetition) {
                        const auto actual = require(
                            (bits ? cuda.packed_linear(input, weight, scale, bits, width, input_scale, output_scale)
                                  : cuda.quantized_linear(input, weight, scale, bias, true))
                                .to(tensor::Device::cpu()));
                        const auto actual_values = require(actual.data<float>()),
                                   expected_values = require(expected.data<float>());
                        for (std::size_t index = 0; index < actual_values.size(); ++index) {
                            ASSERT_TRUE(std::isfinite(actual_values[index]))
                                << "projection mismatch width=" << width << " rows=" << rows << " bits=" << bits
                                << " input_scale=" << input_scale << " index=" << index;
                            ASSERT_NEAR(actual_values[index], expected_values[index],
                                        1e-4F + (bits ? 0.F : 1e-6F * std::abs(expected_values[index])))
                                << "projection mismatch width=" << width << " rows=" << rows << " bits=" << bits
                                << " input_scale=" << input_scale << " index=" << index;
                        }
                    }
                }
            }
        }
    }
}

TEST_F(CudaOpsTest, FusedOperations) {
    SCOPED_TRACE("CUDA fused operations failed");
    using namespace kidi;
    using ops::require;
    using tensor::DType;
    using tensor::Tensor;
    const auto device = tensor::Device::cuda();
    ops::Context context(device);
    std::uint32_t seed = 7;
    const auto random = [&](std::size_t count) {
        std::vector<float> values(count);
        for (auto& value : values) {
            seed = seed * 1664525U + 1013904223U;
            value = static_cast<float>(seed >> 8) / static_cast<float>(1U << 24) * 2.F - 1.F;
        }
        return values;
    };
    const auto upload = [&](std::vector<std::int64_t> shape, const std::vector<float>& values) {
        return require(Tensor::from_host(std::move(shape), std::span<const float>(values), device));
    };
    const auto close = [&](const Tensor& actual, const std::vector<float>& expected) -> ::testing::AssertionResult {
        const auto host = require(context.cast(actual, DType::F32).to(tensor::Device::cpu()));
        const auto values = require(host.data<float>());
        if (values.size() != expected.size())
            return ::testing::AssertionFailure() << "size " << values.size() << " != " << expected.size();
        for (std::size_t index = 0; index < values.size(); ++index)
            if (std::abs(values[index] - expected[index]) > 1e-4F * (1.F + std::abs(expected[index])))
                return ::testing::AssertionFailure() << "index=" << index << " actual=" << values[index]
                                                     << " expected=" << expected[index];
        return ::testing::AssertionSuccess();
    };

    {   // residual_mix: softmax over RMS-normalized sources
        constexpr std::size_t N = 3, Q = 2, T = 4, D = 64;
        std::vector<std::vector<float>> sources;
        std::vector<Tensor> inputs;
        for (std::size_t n = 0; n < N; ++n) {
            sources.push_back(random(T * D));
            for (auto& value : sources.back()) value *= static_cast<float>(n + 1);
            inputs.push_back(upload({T, D}, sources.back()));
        }
        const auto queries = random(Q * D);
        std::vector<float> expected(Q * T * D, 0.F);
        for (std::size_t t = 0; t < T; ++t) {
            std::array<float, N> inverse{};
            for (std::size_t n = 0; n < N; ++n) {
                float sum = 0;
                for (std::size_t d = 0; d < D; ++d) sum += sources[n][t * D + d] * sources[n][t * D + d];
                inverse[n] = 1.F / std::sqrt(sum / D + 1e-6F);
            }
            for (std::size_t q = 0; q < Q; ++q) {
                std::array<float, N> logits{};
                for (std::size_t n = 0; n < N; ++n) {
                    for (std::size_t d = 0; d < D; ++d) logits[n] += queries[q * D + d] * sources[n][t * D + d];
                    logits[n] *= inverse[n] / std::sqrt(static_cast<float>(D));
                }
                const auto peak = *std::ranges::max_element(logits);
                float total = 0;
                for (auto& logit : logits) total += logit = std::exp(logit - peak);
                for (std::size_t n = 0; n < N; ++n)
                    for (std::size_t d = 0; d < D; ++d)
                        expected[(q * T + t) * D + d] += logits[n] / total * sources[n][t * D + d] * inverse[n];
            }
        }
        EXPECT_TRUE(close(context.residual_mix(inputs, upload({Q, D}, queries), 1e-6F), expected)) << "residual_mix";
    }
    {   // short_convolution with grouped per-token taps
        constexpr std::size_t W = 4, T = 5, C = 8, G = 2;
        const auto padded = random((W - 1 + T) * C), weight = random(W * C), taps = random(T * G * W);
        const float scale = 1.F / 3.F;
        std::vector<float> fixed(T * C), dynamic(T * C);
        for (std::size_t t = 0; t < T; ++t)
            for (std::size_t c = 0; c < C; ++c) {
                const auto current = padded[(t + W - 1) * C + c];
                float plain = current, tapped = current;
                for (std::size_t lag = 0; lag < W; ++lag) {
                    const auto input = padded[(t + W - 1 - lag) * C + c];
                    plain += input * weight[lag * C + c];
                    tapped += input * (weight[lag * C + c] + taps[t * G * W + c / (C / G) * W + lag]);
                }
                fixed[t * C + c] = plain * scale;
                dynamic[t * C + c] = tapped * scale;
            }
        const auto input = upload({W - 1 + T, C}, padded), weights = upload({W, C}, weight);
        EXPECT_TRUE(close(context.short_convolution(input, weights, {}, scale), fixed)) << "short_convolution fixed";
        EXPECT_TRUE(close(context.short_convolution(input, weights, upload({T, G * W}, taps), scale), dynamic))
            << "short_convolution dynamic taps";
        // Ring-buffered state at slot 2 holds rows [?, 0, 1, 2] of padded -> slot 2 precedes row 0 of current input.
        std::vector<float> ring(W * C, 0.F);
        for (std::size_t lag = 1; lag < W; ++lag)
            for (std::size_t c = 0; c < C; ++c)
                ring[((2 - lag + W) % W) * C + c] = padded[(W - 1 - lag) * C + c];
        const auto current_input = upload({T, C}, std::vector<float>(padded.begin() + (W - 1) * C, padded.end()));
        const auto slot = require(Tensor::from_host({1}, std::span<const std::int32_t>(std::array<std::int32_t, 1>{2}),
                                                    device));
        EXPECT_TRUE(close(context.short_convolution(current_input, upload({W, C}, ring), weights, {}, slot, scale),
                          fixed))
            << "short_convolution ring fixed";
        EXPECT_TRUE(close(context.short_convolution(current_input, upload({W, C}, ring), weights,
                                                    upload({T, G * W}, taps), slot, scale),
                          dynamic))
            << "short_convolution ring dynamic taps";
    }
    {   // l2_clip: scaled rows capped at the threshold
        constexpr std::size_t T = 3, H = 2, D = 32;
        auto input = random(T * H * D);
        for (std::size_t d = 0; d < D; ++d) input[d] *= 0.01F; // keeps one row below the threshold
        const auto scale = random(H * D);
        const float threshold = 1.5F;
        std::vector<float> expected(input.size());
        for (std::size_t row = 0; row < T * H; ++row) {
            float norm = 0;
            for (std::size_t d = 0; d < D; ++d) {
                expected[row * D + d] = input[row * D + d] * scale[row % H * D + d];
                norm += expected[row * D + d] * expected[row * D + d];
            }
            const auto factor = std::min(1.F, threshold / (std::sqrt(norm) + 1e-6F));
            for (std::size_t d = 0; d < D; ++d) expected[row * D + d] *= factor;
        }
        EXPECT_TRUE(close(context.l2_clip(upload({T, H, D}, input), upload({H, D}, scale), threshold), expected))
            << "l2_clip";
    }
    {   // relative_attention: window, relative extent, sink, GQA and sigmoid gate
        constexpr std::int64_t H = 4, KVH = 2, D = 32, T = 3, START = 2, S = START + T, E = 2;
        for (const std::int64_t window : {std::int64_t{0}, std::int64_t{2}}) {
            const auto query = random(T * H * D), key = random(S * KVH * D), value = random(S * KVH * D),
                       relative = random(T * H * E), sink = random(T * H), gate = random(T * H * D);
            std::vector<float> expected(T * H * D);
            for (std::int64_t t = 0; t < T; ++t)
                for (std::int64_t h = 0; h < H; ++h) {
                    const auto position = START + t, kv = h / (H / KVH);
                    const auto first = window ? std::max<std::int64_t>(0, position - window + 1) : 0;
                    std::vector<float> logits;
                    for (auto s = first; s <= position; ++s) {
                        float logit = 0;
                        for (std::int64_t d = 0; d < D; ++d)
                            logit += query[(t * H + h) * D + d] * key[(s * KVH + kv) * D + d];
                        logit /= std::sqrt(static_cast<float>(D));
                        if (position - s < E) logit += relative[(t * H + h) * E + position - s];
                        logits.push_back(logit);
                    }
                    logits.push_back(sink[t * H + h]);
                    const auto peak = *std::ranges::max_element(logits);
                    float total = 0;
                    for (auto& logit : logits) total += logit = std::exp(logit - peak);
                    for (std::int64_t d = 0; d < D; ++d) {
                        float sum = 0;
                        for (auto s = first; s <= position; ++s)
                            sum += logits[s - first] / total * value[(s * KVH + kv) * D + d];
                        const auto g = gate[(t * H + h) * D + d];
                        expected[(t * H + h) * D + d] = sum / (1.F + std::exp(-g));
                    }
                }
            const auto actual = context.relative_attention(
                upload({T, H * D}, query), upload({S, KVH * D}, key), upload({S, KVH * D}, value),
                upload({T, H, E}, relative), upload({T, H}, sink), upload({T, H * D}, gate),
                {.heads = H, .key_heads = KVH, .start = START, .window = window, .extent = E});
            EXPECT_TRUE(close(actual, expected)) << "relative_attention window=" << window;
        }
        {   // Device positions + log scaling
            constexpr float ALPHA = 0.1F, FLOOR = 2.F;
            const auto query = random(T * H * D), key = random(S * KVH * D), value = random(S * KVH * D),
                       relative = random(T * H * E), sink = random(T * H), gate = random(T * H * D);
            std::vector<float> expected(T * H * D);
            for (std::int64_t t = 0; t < T; ++t)
                for (std::int64_t h = 0; h < H; ++h) {
                    const auto position = START + t, kv = h / (H / KVH);
                    const float tau = 1.F + ALPHA * std::log(std::max(1.F, static_cast<float>(position + 1) / FLOOR));
                    std::vector<float> logits;
                    for (std::int64_t s = 0; s <= position; ++s) {
                        float logit = 0;
                        for (std::int64_t d = 0; d < D; ++d)
                            logit += query[(t * H + h) * D + d] * key[(s * KVH + kv) * D + d];
                        logit /= std::sqrt(static_cast<float>(D));
                        if (position - s < E) logit += relative[(t * H + h) * E + position - s];
                        logits.push_back(logit * tau);
                    }
                    logits.push_back(sink[t * H + h]);
                    const auto peak = *std::ranges::max_element(logits);
                    float total = 0;
                    for (auto& logit : logits) total += logit = std::exp(logit - peak);
                    for (std::int64_t d = 0; d < D; ++d) {
                        float sum = 0;
                        for (std::int64_t s = 0; s <= position; ++s)
                            sum += logits[s] / total * value[(s * KVH + kv) * D + d];
                        const auto g = gate[(t * H + h) * D + d];
                        expected[(t * H + h) * D + d] = sum / (1.F + std::exp(-g));
                    }
                }
            std::vector<std::int32_t> pos_values(T);
            for (std::int64_t t = 0; t < T; ++t) pos_values[t] = static_cast<std::int32_t>(START + t);
            const auto positions = require(Tensor::from_host({T}, std::span<const std::int32_t>(pos_values), device));
            const auto actual = context.relative_attention(
                upload({T, H * D}, query), upload({S, KVH * D}, key), upload({S, KVH * D}, value),
                upload({T, H, E}, relative), upload({T, H}, sink), upload({T, H * D}, gate),
                {.heads = H, .key_heads = KVH, .start = 0, .window = 0, .extent = E, .log_alpha = ALPHA,
                 .log_floor = FLOOR},
                positions);
            EXPECT_TRUE(close(actual, expected)) << "relative_attention device positions + log scaling";
        }
    }
    {   // Softcapped SiLU gate with routing weights
        constexpr std::size_t R = 3, I = 5;
        const auto input = random(R * 2 * I);
        auto weights = random(R);
        std::vector<float> scaled(input.size());
        for (std::size_t index = 0; index < input.size(); ++index) scaled[index] = input[index] * 40.F;
        std::vector<float> expected(R * I);
        for (std::size_t r = 0; r < R; ++r)
            for (std::size_t i = 0; i < I; ++i) {
                const auto g = scaled[r * 2 * I + i], u = scaled[r * 2 * I + I + i];
                expected[r * I + i] =
                    std::tanh(g / 4.F) * 4.F / (1.F + std::exp(-g)) * std::tanh(u / 25.F) * 25.F * weights[r];
            }
        EXPECT_TRUE(close(context.silu_gate(upload({R, 2 * I}, scaled), 4.F, 25.F, upload({R}, weights)), expected))
            << "softcapped silu_gate";
    }
}

TEST_F(CudaOpsTest, ParsesCudaDevices) {
    using namespace kidi;
    EXPECT_EQ(tensor::parse_cuda_device("cuda"), tensor::Device::cuda(0));
    EXPECT_EQ(tensor::parse_cuda_device("cuda:0"), tensor::Device::cuda(0));
    EXPECT_EQ(tensor::parse_cuda_device("cuda:12"), tensor::Device::cuda(12));
    for (const auto invalid :
         {"cuda:", "cuda:-0", "cuda:-1", "cuda:+1", "cuda:1x", "cuda:999999999999999999", "cpu"})
        EXPECT_FALSE(tensor::parse_cuda_device(invalid)) << invalid;
}

// Numerical checks covering W8A8 projection, embedding, GEMM, softmax, RMS/layer norm, convolution and attention.
TEST_F(CudaOpsTest, StorageBroadcastParity) {
    using namespace kidi;
    const std::array<float, 6> values{1, 2, 3, 4, 5, 6};
    const std::array<float, 3> bias{10, 20, 30};
    auto input =
        ops::require(tensor::Tensor::from_host({2, 3}, std::span<const float>(values), tensor::Device::cuda()));
    auto other = ops::require(tensor::Tensor::from_host({3}, std::span<const float>(bias), tensor::Device::cuda()));
    ops::Context context(tensor::Device::cuda());
    const auto result = ops::require(context.add(input, other).to(tensor::Device::cpu()));
    const auto actual = ops::require(result.data<float>());
    ASSERT_TRUE(std::ranges::equal(actual, std::array<float, 6>{11, 22, 33, 14, 25, 36})) << "broadcast add";
    context.add_(input, other);
    const auto updated = ops::require(input.to(tensor::Device::cpu()));
    ASSERT_TRUE(std::ranges::equal(ops::require(updated.data<float>()), actual)) << "in-place add";
    ops::Context cpu;
    const auto host = ops::require(input.to(tensor::Device::cpu()));
    const auto gpu_softmax = ops::require(context.softmax(input).to(tensor::Device::cpu()));
    const auto cpu_softmax = cpu.softmax(host);
    const auto expected = ops::require(cpu_softmax.data<float>());
    const auto probabilities = ops::require(gpu_softmax.data<float>());
    for (std::size_t index = 0; index < expected.size(); ++index)
        EXPECT_NEAR(probabilities[index], expected[index], 1e-5F) << "softmax index=" << index;
    const std::array<std::int8_t, 6> weights{1, -2, 3, 4, 5, -6};
    const std::array<float, 2> scales{0.25F, 0.5F}, biases{0.1F, -0.2F};
    const auto weight = ops::require(tensor::Tensor::from_host({2, 3}, std::span<const std::int8_t>(weights)));
    const auto scale = ops::require(tensor::Tensor::from_host({2, 1}, std::span<const float>(scales)));
    const auto bias_tensor = ops::require(tensor::Tensor::from_host({2}, std::span<const float>(biases)));
    const auto gpu_projection =
        ops::require(context.quantized_linear(input, weight, scale, bias_tensor, true).to(tensor::Device::cpu()));
    const auto cpu_projection = cpu.quantized_linear(host, weight, scale, bias_tensor, true);
    const auto reference = ops::require(cpu_projection.data<float>());
    const auto projected = ops::require(gpu_projection.data<float>());
    for (std::size_t index = 0; index < reference.size(); ++index)
        EXPECT_NEAR(projected[index], reference[index], 1e-4F) << "quantized projection index=" << index;
    const std::array<std::int32_t, 2> ids{1, 0};
    const auto indices =
        ops::require(tensor::Tensor::from_host({2}, std::span<const std::int32_t>(ids), tensor::Device::cuda()));
    const auto embedded =
        ops::require(context.embedding(indices, weight, scale, 3, 8, 1.F).to(tensor::Device::cpu()));
    ASSERT_TRUE(std::ranges::equal(embedded.shape(), std::array<std::int64_t, 3>{1, 2, 3})) << "embedding shape";
    EXPECT_TRUE(std::ranges::equal(ops::require(embedded.data<float>()),
                                   std::array<float, 6>{2, 2.5F, -3, 0.25F, -0.5F, 0.75F}))
        << "embedding values";
    const auto dense_weight = ops::require(weight.to(tensor::Device::cpu()));
    const auto float_weight = cpu.cast(dense_weight, tensor::DType::F32);
    context.linear(input, float_weight, {}, true);
    const auto reshaped_weight = ops::require(float_weight.reshape({3, 2}));
    const auto narrow_input = context.slice(input, 1, 0, 2);
    const auto reshaped_projection =
        ops::require(context.linear(narrow_input, reshaped_weight, {}, true).to(tensor::Device::cpu()));
    const auto reshape_reference = cpu.linear(cpu.slice(host, 1, 0, 2), reshaped_weight, {}, true);
    EXPECT_TRUE(std::ranges::equal(ops::require(reshaped_projection.data<float>()),
                                   ops::require(reshape_reference.data<float>())))
        << "reshaped weight projection";
    bool rejected_dtype = false;
    try {
        context.cast(input, tensor::DType::U32);
    } catch (const ops::Failure& failure) {
        rejected_dtype = failure.error().code == ErrorCode::UNSUPPORTED;
    }
    EXPECT_TRUE(rejected_dtype) << "cast to U32 was not rejected as UNSUPPORTED";
    const auto empty = ops::require(tensor::Tensor::empty({0, 3}, tensor::DType::F32, tensor::Device::cuda()));
    const auto empty_output = context.add(empty, other);
    EXPECT_EQ(empty_output.numel(), 0) << "empty broadcast";
    EXPECT_TRUE(std::ranges::equal(empty_output.shape(), std::array<std::int64_t, 2>{0, 3})) << "empty broadcast";
    const auto gpu_matrix =
        ops::require(context.matmul(input, ops::require(float_weight.to(tensor::Device::cuda())), true)
                         .to(tensor::Device::cpu()));
    const auto cpu_matrix = cpu.matmul(host, float_weight, true);
    const auto dense_expected = ops::require(cpu_matrix.data<float>());
    const auto dense_actual = ops::require(gpu_matrix.data<float>());
    for (std::size_t index = 0; index < dense_expected.size(); ++index)
        EXPECT_NEAR(dense_actual[index], dense_expected[index], 1e-5F) << "matmul index=" << index;
    const std::array<float, 3> norm_scale_values{1, 0.5F, 2};
    const auto norm_scale = ops::require(tensor::Tensor::from_host({3}, std::span<const float>(norm_scale_values)));
    const auto normalized = ops::require(context.rms_norm(input, norm_scale, 1e-5F).to(tensor::Device::cpu()));
    const auto norm_reference = cpu.rms_norm(host, norm_scale, 1e-5F);
    const auto norm_expected = ops::require(norm_reference.data<float>());
    const auto norm_actual = ops::require(normalized.data<float>());
    for (std::size_t index = 0; index < norm_expected.size(); ++index)
        EXPECT_NEAR(norm_actual[index], norm_expected[index], 1e-5F) << "RMS norm index=" << index;
    const std::array<float, 4> signal_values{1, 2, 3, 4};
    const std::array<float, 3> kernel_values{1, 2, 1};
    const std::array<float, 1> zero{0};
    const auto signal = ops::require(
        tensor::Tensor::from_host({1, 4, 1}, std::span<const float>(signal_values), tensor::Device::cuda()));
    const auto kernel = ops::require(tensor::Tensor::from_host({1, 3}, std::span<const float>(kernel_values)));
    const auto offset = ops::require(tensor::Tensor::from_host({1}, std::span<const float>(zero)));
    const auto convolved =
        ops::require(context.conv1d(signal, kernel, {}, offset, 3, 1, 1, 1).to(tensor::Device::cpu()));
    EXPECT_TRUE(std::ranges::equal(ops::require(convolved.data<float>()), std::array<float, 4>{4, 8, 12, 11}))
        << "conv1d";
    const auto norm_bias = ops::require(tensor::Tensor::zeros({3}, tensor::DType::F32));
    const auto gpu_pair = context.residual_layer_norm(input, input, norm_scale, norm_bias, 1e-5F);
    const auto cpu_pair = cpu.residual_layer_norm(host, host, norm_scale, norm_bias, 1e-5F);
    for (std::size_t output_index = 0; output_index < 2; ++output_index) {
        const auto pair_host = ops::require(gpu_pair[output_index].to(tensor::Device::cpu()));
        const auto pair_actual = ops::require(pair_host.data<float>());
        const auto pair_expected = ops::require(cpu_pair[output_index].data<float>());
        for (std::size_t index = 0; index < pair_actual.size(); ++index)
            EXPECT_NEAR(pair_actual[index], pair_expected[index], 1e-5F)
                << "residual layer norm output=" << output_index << " index=" << index;
    }
    const auto sequence = context.reshape(input, {1, 2, 3});
    const auto attention = ops::require(
        context.scaled_dot_product_attention(sequence, sequence, sequence, 1).to(tensor::Device::cpu()));
    const auto host_sequence = cpu.reshape(host, {1, 2, 3});
    const auto expected_attention =
        cpu.scaled_dot_product_attention(host_sequence, host_sequence, host_sequence, 1);
    const auto attention_actual = ops::require(attention.data<float>());
    const auto attention_expected = ops::require(expected_attention.data<float>());
    for (std::size_t index = 0; index < attention_actual.size(); ++index)
        EXPECT_NEAR(attention_actual[index], attention_expected[index], 1e-4F) << "SDPA index=" << index;
    const std::array<float, 1> attention_scale_value{0.01F};
    const auto attention_scale = ops::require(
        tensor::Tensor::from_host({}, std::span<const float>(attention_scale_value), tensor::Device::cuda()));
    const auto small_query = context.multiply(sequence, attention_scale);
    const auto small_host_query = ops::require(small_query.to(tensor::Device::cpu()));
    const auto attention_mask =
        ops::require(tensor::Tensor::zeros({1, 1, 2, 2}, tensor::DType::F32, tensor::Device::cuda()));
    const auto explicit_attention = ops::require(
        context.grouped_query_attention(small_query, small_query, small_query, 1, 1, attention_mask, 1.F)
            .to(tensor::Device::cpu()));
    const auto explicit_reference =
        cpu.grouped_query_attention(small_host_query, small_host_query, small_host_query, 1, 1,
                                    ops::require(attention_mask.to(tensor::Device::cpu())), 1.F);
    const auto explicit_actual = ops::require(explicit_attention.data<float>());
    const auto explicit_expected = ops::require(explicit_reference.data<float>());
    for (std::size_t index = 0; index < explicit_actual.size(); ++index)
        EXPECT_NEAR(explicit_actual[index], explicit_expected[index], 1e-5F) << "GQA explicit mask index=" << index;
    const std::array<float, 8> packed_values{0.25F, -0.5F, 0.75F, -1, -0.25F, 0.5F, -0.75F, 1};
    const std::array<std::uint8_t, 4> packed_weights{0xE1, 0xC3, 0x54, 0x7A};
    const std::array<float, 4> packed_scales{0.25F, 0.5F, 0.125F, 0.25F};
    const auto packed_host = ops::require(tensor::Tensor::from_host({2, 4}, std::span<const float>(packed_values)));
    const auto packed_input = ops::require(packed_host.to(tensor::Device::cuda()));
    const auto packed_weight =
        ops::require(tensor::Tensor::from_host({2, 2}, std::span<const std::uint8_t>(packed_weights)));
    const auto packed_scale =
        ops::require(tensor::Tensor::from_host({2, 2}, std::span<const float>(packed_scales)));
    const auto packed_output =
        ops::require(context.packed_linear(packed_input, packed_weight, packed_scale, 4, 2, 0.25F, 0.125F)
                         .to(tensor::Device::cpu()));
    const auto packed_reference = cpu.packed_linear(packed_host, packed_weight, packed_scale, 4, 2, 0.25F, 0.125F);
    const auto packed_actual = ops::require(packed_output.data<float>());
    const auto packed_expected = ops::require(packed_reference.data<float>());
    for (std::size_t index = 0; index < packed_actual.size(); ++index)
        EXPECT_NEAR(packed_actual[index], packed_expected[index], 1e-5F) << "packed projection index=" << index;
}

TEST_F(CudaOpsTest, SelectedDevices) {
    using namespace kidi;
    const std::array<float, 6> values{1, 2, 3, 4, 5, 6};
    const std::array<float, 3> bias{10, 20, 30};
    auto input =
        ops::require(tensor::Tensor::from_host({2, 3}, std::span<const float>(values), tensor::Device::cuda()));
    auto other = ops::require(tensor::Tensor::from_host({3}, std::span<const float>(bias), tensor::Device::cuda()));
    ops::Context context(tensor::Device::cuda());
    context.add_(input, other);
    const auto host = ops::require(input.to(tensor::Device::cpu()));
    for (int index = 0; index < kidi::cuda::device_count(); ++index) {
        SCOPED_TRACE(::testing::Message() << "device " << index);
        const auto device = tensor::Device::cuda(index);
        const auto selected_input = ops::require(host.to(device));
        const auto selected_bias = ops::require(other.to(tensor::Device::cpu())).to(device);
        ops::Context selected_context(device);
        const auto selected_output = selected_context.add(selected_input, ops::require(selected_bias));
        ASSERT_EQ(selected_output.device(), device);
        const auto copied = ops::require(selected_output.to(tensor::Device::cpu()));
        EXPECT_TRUE(
            std::ranges::equal(ops::require(copied.data<float>()), std::array<float, 6>{21, 42, 63, 24, 45, 66}));
    }
    EXPECT_FALSE(tensor::Tensor::empty({1}, tensor::DType::F32, tensor::Device::cuda(kidi::cuda::device_count())))
        << "allocation on an out-of-range CUDA device succeeded";
}
} // namespace
