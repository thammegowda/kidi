#include "kidi/ops/context.h"
#include "kidi/ops/quantization.h"
#include "kidi/tensor/arena.h"
#include <array>
#include <cmath>
#include <iostream>
#include <thread>
#include <limits>

auto main() -> int {
    try {
        using namespace kidi;
        std::vector devices{tensor::Device::cpu()};
#if defined(__APPLE__)
        devices.push_back(tensor::Device::apple_gpu());
#endif
        for (auto device : devices) {
            tensor::Tensor escaped;
            if (device == tensor::Device::cpu()) {
                tensor::Arena arena(1024);
                ops::require(arena.reserve(1024));
                auto first = ops::require(arena.allocate({4}, tensor::DType::F32));
                escaped = ops::require(arena.allocate({4}, tensor::DType::F32));
                if (!first.owns_unique_storage() || !escaped.owns_unique_storage() || escaped.storage_offset() == 0)
                    return 1;
                auto first_values = ops::require(first.data<float>());
                std::fill(first_values.begin(), first_values.end(), 2.F);
                auto values = ops::require(escaped.data<float>());
                std::fill(values.begin(), values.end(), 3.F);
                ops::Context arena_context(device);
                auto sum = arena_context.add(first, escaped);
                arena_context.add_(escaped, first);
                arena_context.synchronize();
                if (ops::require(sum.data<float>())[0] != 5.F || values[0] != 5.F || first_values[0] != 2.F) return 1;
                auto overflow = arena.allocate({-1}, tensor::DType::F32);
                if (overflow) return 1;
                auto larger = ops::require(arena.allocate({1024}, tensor::DType::F32));
                if (values[0] != 5.F || arena.reserved_bytes() <= 1024) return 1;
            }
            if (escaped.defined() && ops::require(escaped.data<float>())[0] != 5.F) return 1;
            {
                ops::Context temporary(device);
                auto source =
                    ops::require(tensor::Tensor::from_host({2}, std::span<const float>(std::array{2.F, 3.F}), device));
                escaped = ops::require(temporary.add(source, source).narrow(0, 1, 1));
            }
            if (ops::require(escaped.data<float>())[0] != 6.F) return 1;
            ops::Context context(device);
            {
                const std::array values{1.F, 2.F, -1.F, -2.F};
                const auto input = ops::require(tensor::Tensor::from_host({4}, std::span<const float>(values), device));
                const auto bytes = context.cast(input, tensor::DType::I8);
                const auto unchanged = context.cast(bytes, tensor::DType::I8);
                context.synchronize();
                if (!std::ranges::equal(ops::require(bytes.data<std::int8_t>()), std::array{1, 2, -1, -2}) ||
                    unchanged.storage_identity() != bytes.storage_identity())
                    return 1;
                bool rejected = false;
                try {
                    context.cast(input, tensor::DType::I8, 0.25F);
                } catch (const ops::Failure& error) {
                    rejected = error.error().code == ErrorCode::UNSUPPORTED;
                }
                if (!rejected) return 1;
            }
            for (const std::int64_t rows : {1, 8}) {
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
                    for (auto value : ops::require(output.data<float>()))
                        if (std::abs(value - expected) > 1e-6F) return 1;
                std::ranges::fill(ops::require(operand.data<float>()), 1.F);
                const auto host_changed = project();
                context.synchronize();
                for (auto value : ops::require(host_changed.data<float>()))
                    if (std::abs(value - 1.F) > 1e-6F) return 1;
            }
            {
                const std::array values{-100.F, -10.F, -3.F, -0.F, 0.25F, 3.F, 10.F, 100.F};
                const std::array factors{-1.F, 0.F, 0.5F, 2.F, -0.75F, 1.25F, -2.F, 0.125F};
                auto gate = ops::require(tensor::Tensor::from_host({2, 4}, std::span<const float>(values), device));
                auto factor = ops::require(tensor::Tensor::from_host({2, 4}, std::span<const float>(factors), device));
                auto separate = context.multiply(context.gelu(gate, true), factor);
                auto fused = context.gelu_multiply(gate, factor);
                context.synchronize();
                if (!std::ranges::equal(ops::require(separate.data<float>()), ops::require(fused.data<float>())))
                    return 1;
                bool rejected = false;
                try {
                    context.gelu_multiply(gate, ops::require(factor.reshape({8})));
                } catch (const ops::Failure&) {
                    rejected = true;
                }
                if (!rejected) return 1;
            }
            for (const std::int64_t width : {40, 256, 512}) {
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
                for (std::size_t index = 0; index < actual.size(); ++index)
                    if (!std::isfinite(actual[index]) || std::abs(actual[index] - expected[index]) > 1e-6F) return 1;
            }
            if (device == tensor::Device::apple_gpu()) {
                auto identity = ops::require(tensor::Tensor::zeros({8, 8}, tensor::DType::F32));
                for (std::size_t index = 0; index < 8; ++index)
                    ops::require(identity.data<float>())[index * 8 + index] = 1.F;
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
                    if (actual[index] != expected[index % 8]) {
                        std::cerr << "calibrated epilogue mismatch " << tensor::to_string(device) << " at " << index
                                  << ": " << actual[index] << " vs " << expected[index % 8] << '\n';
                        return 1;
                    }
            }
            for (const std::int64_t width : {7, 1027, 262144}) {
                const auto infinity = std::numeric_limits<float>::infinity();
                std::vector<float> scores(6 * width, -2.F);
                scores[1] = scores[width - 1] = 4.F;
                std::fill(scores.begin() + width, scores.begin() + 2 * width, -infinity);
                scores[2 * width + width - 1] = std::numeric_limits<float>::quiet_NaN();
                scores[3 * width + 1] = infinity;
                scores[6 * width - 1] = 3.F;
                auto logits =
                    ops::require(tensor::Tensor::from_host({6, width}, std::span<const float>(scores), device));
                auto selected = context.greedy_token(logits);
                context.synchronize();
                if (!std::ranges::equal(
                        ops::require(selected.data<std::int32_t>()),
                        std::array<std::int32_t, 6>{1, -1, -1, -1, 0, static_cast<std::int32_t>(width - 1)}))
                    return 1;
                auto finite = ops::require(
                    tensor::Tensor::from_host({1, 3}, std::span<const float>(std::array{-3.F, 0.F, -0.F}), device));
                auto queued = context.greedy_token(context.add(finite, finite));
                context.synchronize();
                if (ops::require(queued.data<std::int32_t>())[0] != 1) return 1;
            }
            bool invalid_selection = false;
            try {
                context.greedy_token(ops::require(tensor::Tensor::zeros({}, tensor::DType::F32, device)));
            } catch (const ops::Failure&) {
                invalid_selection = true;
            }
            if (!invalid_selection) return 1;
            {
                auto increment = ops::require(
                    tensor::Tensor::from_host({1, 4}, std::span<const float>(std::array{1.F, 2.F, 3.F, 4.F}), device));
                auto identity = ops::require(tensor::Tensor::from_host(
                    {4, 4},
                    std::span<const float>(
                        std::array{1.F, 0.F, 0.F, 0.F, 0.F, 1.F, 0.F, 0.F, 0.F, 0.F, 1.F, 0.F, 0.F, 0.F, 0.F, 1.F}),
                    device));
                auto current = context.add(increment, increment);
                auto retained = ops::require(ops::require(current.reshape({4})).narrow(0, 0, 2));
                for (int iteration = 0; iteration < 64; ++iteration) {
                    current = context.add(current, increment);
                    current = context.matmul(current, identity);
                }
                context.synchronize();
                if (!std::ranges::equal(ops::require(current.data<float>()), std::array{66.F, 132.F, 198.F, 264.F}) ||
                    !std::ranges::equal(ops::require(retained.data<float>()), std::array{2.F, 4.F}))
                    return 1;
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
                        if (eviction.preparation_ns() != preparation) return 1;
                    }
                    eviction.synchronize();
                    if (!std::ranges::equal(ops::require(preserved.data<float>()), std::array{2.F, 4.F, 6.F, 8.F}))
                        return 1;
                    const auto actual = ops::require(current.data<float>());
                    for (std::size_t channel = 0; channel < actual.size(); ++channel)
                        if (std::abs(actual[channel] - (channel + 1) / std::sqrt(7.5F + ITERATIONS * 0.001F)) > 1e-5F)
                            return 1;
                }
            }
            {
                for (int bits : {2, 4, 8}) {
                    const std::array weights{0.F, -7.F, 7.F, 1.F, 0.F, 14.F, -14.F, 2.F,
                                             0.F, 0.F,  0.F, 0.F, 7.F, -7.F, 1.F,   -1.F};
                    auto original = ops::require(tensor::Tensor::from_host({2, 8}, std::span<const float>(weights)));
                    auto packed = ops::require(ops::pack_weight(original, bits, 4));
                    const std::array activation{0.F, 1.F, 2.F, 3.F, 4.F, 5.F, 6.F, 255.F};
                    auto operand =
                        ops::require(tensor::Tensor::from_host({1, 8}, std::span<const float>(activation), device));
                    auto output = context.packed_linear(operand, packed.values, packed.scales, bits, 4);
                    context.synchronize();
                    const auto bytes = ops::require(packed.values.data<std::uint8_t>());
                    const auto scales = ops::require(packed.scales.data<float>());
                    const auto values = ops::require(output.data<float>());
                    for (std::size_t row = 0; row < 2; ++row) {
                        float expected = 0;
                        for (std::size_t channel = 0; channel < 8; ++channel) {
                            const auto offset = row * 8 + channel;
                            const auto raw =
                                (bytes[offset / (8 / bits)] >> ((offset % (8 / bits)) * bits)) & ((1 << bits) - 1);
                            const auto integer = (raw ^ (1 << (bits - 1))) - (1 << (bits - 1));
                            expected += activation[channel] * integer * scales[row * 2 + channel / 4];
                        }
                        if (!std::isfinite(values[row]) || std::abs(values[row] - expected) > 1e-3F) return 1;
                    }
                    if (ops::pack_weight(original, bits, 3)) return 1;
                    context.prepare_linear_weights(original, bits, 4);
                    auto rebound = context.linear(operand, original, {}, true);
                    context.synchronize();
                    const auto rebound_values = ops::require(rebound.data<float>());
                    for (std::size_t row = 0; row < 2; ++row)
                        if (std::abs(rebound_values[row] - values[row]) > 1e-3F) return 1;
                    std::vector<float> many_values(33 * 8);
                    for (std::size_t row = 0; row < 33; ++row)
                        std::copy(activation.begin(), activation.end(), many_values.begin() + row * 8);
                    auto many =
                        ops::require(tensor::Tensor::from_host({33, 8}, std::span<const float>(many_values), device));
                    auto tiled = context.linear(many, original, {}, true);
                    context.synchronize();
                    const auto tiled_values = ops::require(tiled.data<float>());
                    for (std::size_t row = 0; row < 33; ++row)
                        for (std::size_t column = 0; column < 2; ++column)
                            if (!std::isfinite(tiled_values[row * 2 + column]) ||
                                std::abs(tiled_values[row * 2 + column] - values[column]) > 0.1F)
                                return 1;
                }
            }
            auto input = ops::require(
                tensor::Tensor::from_host({1, 4}, std::span<const float>(std::array{1.F, 2.F, 3.F, 4.F}), device));
            {
                constexpr std::size_t WIDTH = 256, COLUMNS = 35;
                std::vector<float> weights(COLUMNS * WIDTH);
                for (std::size_t column = 0; column < COLUMNS; ++column)
                    for (std::size_t channel = 0; channel < WIDTH; ++channel)
                        weights[column * WIDTH + channel] =
                            (static_cast<int>((column * 7 + channel) % 15) - 7) * (channel < 128 ? 0.125F : 0.25F);
                auto original =
                    ops::require(tensor::Tensor::from_host({COLUMNS, WIDTH}, std::span<const float>(weights)));
                auto packed = ops::require(ops::pack_weight(original));
                for (std::size_t rows : {1, 3, 33}) {
                    std::vector<float> activations(rows * WIDTH);
                    for (std::size_t index = 0; index < activations.size(); ++index)
                        activations[index] = float(index % 256);
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
                            if (!std::isfinite(values[row * COLUMNS + column]) ||
                                std::abs(values[row * COLUMNS + column] - expected) > 0.05F)
                                return 1;
                        }
                }
            }
            auto result = context.add(input, input);
            auto normalized = context.softmax(result);
            context.synchronize();
            auto data = ops::require(result.data<float>());
            if (data[0] != 2 || data[3] != 8) return 1;
            auto probabilities = ops::require(normalized.data<float>());
            float total = 0;
            for (auto value : probabilities) total += value;
            if (std::abs(total - 1) > 1e-5F) return 1;
            const std::array rms_input_values{1.F, 2.F, 3.F, 4.F, 0.F, 0.F, 0.F, 0.F};
            const std::array rms_scale_values{1.F, -2.F, 0.5F, 3.F};
            auto rms_input =
                ops::require(tensor::Tensor::from_host({2, 4}, std::span<const float>(rms_input_values), device));
            auto rms_scale =
                ops::require(tensor::Tensor::from_host({4}, std::span<const float>(rms_scale_values), device));
            auto rms_output = context.rms_norm(rms_input, rms_scale, 1e-6F);
            auto other_scale = ops::require(
                tensor::Tensor::from_host({4}, std::span<const float>(std::array{2.F, -4.F, 1.F, 6.F}), device));
            auto other_rms = context.rms_norm(rms_input, other_scale, 1e-6F);
            auto rms_residual = context.rms_norm_residual(rms_input, rms_scale, rms_input, 1e-6F);
            auto output_scale =
                ops::require(tensor::Tensor::from_host({1}, std::span<const float>(std::array{0.5F}), device));
            auto rms_scaled = context.rms_norm_residual(rms_input, rms_scale, rms_input, 1e-6F, output_scale);
            context.synchronize();
            const auto rms_values = ops::require(rms_output.data<float>());
            for (std::size_t channel = 0; channel < 4; ++channel) {
                const auto expected = rms_input_values[channel] * rms_scale_values[channel] / std::sqrt(7.5F + 1e-6F);
                if (std::abs(rms_values[channel] - expected) > 1e-5F || rms_values[4 + channel] != 0.F) return 1;
                if (std::abs(ops::require(other_rms.data<float>())[channel] - 2.F * expected) > 1e-5F) return 1;
                if (std::abs(ops::require(rms_residual.data<float>())[channel] -
                             (rms_input_values[channel] + expected)) > 1e-5F ||
                    std::abs(ops::require(rms_scaled.data<float>())[channel] -
                             0.5F * (rms_input_values[channel] + expected)) > 1e-5F)
                    return 1;
            }
            bool invalid_rms = false;
            try {
                context.rms_norm(rms_input, rms_scale, 0.F);
            } catch (const ops::Failure& error) {
                invalid_rms = error.error().code == ErrorCode::INVALID_ARGUMENT;
            }
            if (!invalid_rms) return 1;
            auto activation_input = ops::require(
                tensor::Tensor::from_host({2, 2}, std::span<const float>(std::array{-4.F, -1.F, 0.F, 3.F}), device));
            auto activated = context.gelu(activation_input, true);
            const auto rounding_input = ops::require(tensor::Tensor::from_host(
                {8}, std::span<const float>(std::array{-100.F, -1.25F, -0.75F, -0.25F, 0.25F, 0.75F, 1.25F, 100.F}),
                device));
            const auto rounded = context.static_round(rounding_input, 0.5F);
            context.synchronize();
            const std::array rounding_expected{-64.F, -1.F, -1.F, 0.F, 0.F, 1.F, 1.F, 63.5F};
            if (!std::ranges::equal(ops::require(rounded.data<float>()), rounding_expected)) return 1;
            auto large = ops::require(tensor::Tensor::from_host(
                {4}, std::span<const float>(std::array{-1000.F, -20.F, 20.F, 1000.F}), device));
            auto large_gelu = context.gelu(large, true);
            auto hyperbolic = context.tanh(activation_input);
            auto projection = context.linear(activation_input, activation_input, {}, true);
            context.synchronize();
            const auto activation_values = ops::require(activation_input.data<float>());
            const auto activated_values = ops::require(activated.data<float>());
            const auto large_values = ops::require(large_gelu.data<float>());
            if (large_values[0] != 0.F || large_values[1] != 0.F || large_values[2] != 20.F ||
                large_values[3] != 1000.F)
                return 1;
            const auto hyperbolic_values = ops::require(hyperbolic.data<float>());
            for (std::size_t index = 0; index < activation_values.size(); ++index) {
                const auto value = activation_values[index];
                const auto expected = 0.5F * value *
                                      (1.F + std::tanh(std::sqrt(2.F / 3.14159265358979323846F) *
                                                       (value + 0.044715F * value * value * value)));
                if (std::abs(activated_values[index] - expected) > 1e-5F ||
                    std::abs(hyperbolic_values[index] - std::tanh(value)) > 1e-5F)
                    return 1;
            }
            const auto projection_values = ops::require(projection.data<float>());
            if (projection_values[0] != 17.F || projection_values[1] != -3.F || projection_values[3] != 9.F) return 1;
            auto rope_input = context.reshape(input, {1, 1, 1, 4});
            auto cosine = ops::require(
                tensor::Tensor::from_host({1, 1, 1, 2}, std::span<const float>(std::array{0.F, 1.F}), device));
            auto sine = ops::require(
                tensor::Tensor::from_host({1, 1, 1, 2}, std::span<const float>(std::array{1.F, 0.F}), device));
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
            if (rotated_values[0] != -3.F || rotated_values[1] != 2.F || rotated_values[2] != 1.F ||
                rotated_values[3] != 4.F)
                return 1;
            const auto attention_values = ops::require(attended.data<float>());
            const auto broadcast_values = ops::require(broadcast_attended.data<float>());
            for (std::size_t index = 0; index < attention_values.size(); ++index)
                if (std::abs(attention_values[index] - broadcast_values[index]) > 1e-5F) return 1;
            const auto probability = 1.F / (1.F + std::exp(1.F));
            for (std::size_t head = 0; head < 2; ++head)
                if (std::abs(attention_values[head * 2] - probability) > 1e-5F ||
                    std::abs(attention_values[head * 2 + 1] - (1.F - probability)) > 1e-5F)
                    return 1;
            auto saved = result;
            {
                constexpr std::int64_t HEADS = 4, KEY_HEADS = 2, WIDTH = 40, LENGTH = 259;
                std::vector<float> queries(HEADS * WIDTH), keys(LENGTH * KEY_HEADS * WIDTH), values(keys.size());
                std::vector<float> mask_values(LENGTH, -1e9F);
                for (std::size_t index = 0; index < queries.size(); ++index)
                    queries[index] = std::sin(float(index)) * 0.2F;
                for (std::size_t index = 0; index < keys.size(); ++index) {
                    keys[index] = std::cos(float(index) * 0.13F);
                    values[index] = std::sin(float(index) * 0.17F);
                }
                for (std::size_t position = 131; position < LENGTH; ++position) mask_values[position] = 0.F;
                const auto tensor_query = ops::require(
                    tensor::Tensor::from_host({1, 1, HEADS * WIDTH}, std::span<const float>(queries), device));
                const auto tensor_key = ops::require(
                    tensor::Tensor::from_host({1, LENGTH, KEY_HEADS * WIDTH}, std::span<const float>(keys), device));
                const auto tensor_value = ops::require(
                    tensor::Tensor::from_host({1, LENGTH, KEY_HEADS * WIDTH}, std::span<const float>(values), device));
                auto tensor_mask = ops::require(
                    tensor::Tensor::from_host({1, 1, 1, LENGTH}, std::span<const float>(mask_values), device));
                const auto first = context.grouped_query_attention(tensor_query, tensor_key, tensor_value, HEADS,
                                                                   KEY_HEADS, tensor_mask, 0.5F);
                const auto second = context.grouped_query_attention(tensor_query, tensor_key, tensor_value, HEADS,
                                                                    KEY_HEADS, tensor_mask, 0.5F);
                context.synchronize();
                const auto actual = ops::require(first.data<float>()), repeated = ops::require(second.data<float>());
                for (std::size_t head = 0; head < HEADS; ++head) {
                    std::vector<float> scores(LENGTH);
                    for (std::size_t position = 0; position < LENGTH; ++position) {
                        float dot = 0;
                        for (std::size_t channel = 0; channel < WIDTH; ++channel)
                            dot += queries[head * WIDTH + channel] *
                                   keys[(position * KEY_HEADS + head / 2) * WIDTH + channel];
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
                            expected_value += scores[position] *
                                              values[(position * KEY_HEADS + head / 2) * WIDTH + channel] / denominator;
                        const auto offset = head * WIDTH + channel;
                        if (!std::isfinite(actual[offset]) || std::abs(actual[offset] - expected_value) > 2e-5F ||
                            std::abs(repeated[offset] - expected_value) > 2e-5F)
                            return 1;
                    }
                }
            }
            for (int iteration = 0; iteration < 12; ++iteration) {
                auto temporary = context.multiply(result, result);
                context.synchronize();
                if (ops::require(temporary.data<float>())[3] != 64.F) return 1;
            }
            if (ops::require(saved.data<float>())[3] != 8.F) return 1;
            auto slice = context.slice(result, 1, 0, 2);
            context.synchronize();
            if (ops::require(slice.data<float>())[1] != 4.F) return 1;
            auto retained_slice = slice;
            for (int iteration = 0; iteration < 10; ++iteration) {
                auto temporary = context.add(result, result);
                context.synchronize();
                if (ops::require(temporary.data<float>())[3] != 16.F) return 1;
            }
            if (ops::require(retained_slice.data<float>())[1] != 4.F) return 1;
            auto index = ops::require(
                tensor::Tensor::from_host({1}, std::span<const std::int32_t>(std::array<std::int32_t, 1>{1}), device));
            auto cache = ops::require(tensor::Tensor::zeros({1, 3, 4}, tensor::DType::F32, device));
            auto updated = context.scatter(cache, context.reshape(result, {1, 1, 4}), index);
            context.synchronize();
            if (ops::require(updated.data<float>())[7] != 8.F || ops::require(cache.data<float>())[7] != 0.F) return 1;
            auto rows = ops::require(
                tensor::Tensor::from_host({2, 2}, std::span<const float>(std::array{1.F, 2.F, 3.F, 4.F}), device));
            auto selected = context.gather(rows, index);
            context.synchronize();
            if (ops::require(selected.data<float>())[0] != 3.F) return 1;
            bool bad_shape = false;
            try {
                context.add(input, rows);
            } catch (const ops::Failure& error) {
                bad_shape = error.error().code == ErrorCode::INVALID_ARGUMENT;
            }
            if (!bad_shape) return 1;
            auto mutable_input = ops::require(
                tensor::Tensor::from_host({1, 4}, std::span<const float>(std::array{1.F, 2.F, 3.F, 4.F}), device));
            auto alias = mutable_input;
            const auto address = ops::require(mutable_input.data<float>()).data();
            const auto immutable = context.add(mutable_input, mutable_input);
            if (&context.add_(mutable_input, mutable_input) != &mutable_input || ops::is_inplace) return 1;
            context.multiply_(mutable_input, mutable_input);
            context.synchronize();
            if (ops::require(alias.data<float>())[3] != 64.F ||
                ops::require(mutable_input.data<float>()).data() != address ||
                ops::require(immutable.data<float>())[3] != 8.F)
                return 1;
            auto cache_alias = cache;
            {
                auto destination = ops::require(tensor::Tensor::zeros({2, 4, 2}, tensor::DType::F32, device));
                auto source = ops::require(tensor::Tensor::from_host(
                    {2, 2, 2}, std::span<const float>(std::array{1.F, 2.F, 3.F, 4.F, 5.F, 6.F, 7.F, 8.F}), device));
                const auto alias = destination;
                const auto address = ops::require(destination.host_bytes()).data();
                if (&context.copy_slice_(destination, source, 1, 2) != &destination || ops::is_inplace) return 1;
                context.synchronize();
                const std::array expected{0.F, 0.F, 0.F, 0.F, 1.F, 2.F, 3.F, 4.F,
                                          0.F, 0.F, 0.F, 0.F, 5.F, 6.F, 7.F, 8.F};
                if (!std::ranges::equal(ops::require(alias.data<float>()), expected) ||
                    ops::require(destination.host_bytes()).data() != address)
                    return 1;
                for (auto start : {std::int64_t{-1}, std::int64_t{3}, INT64_MAX}) {
                    bool rejected = false;
                    try {
                        context.copy_slice_(destination, source, 1, start);
                    } catch (const ops::Failure&) {
                        rejected = true;
                    }
                    if (!rejected || ops::is_inplace ||
                        !std::ranges::equal(ops::require(alias.data<float>()), expected))
                        return 1;
                }
                auto storage = ops::require(tensor::Tensor::from_host(
                    {8}, std::span<const float>(std::array{0.F, 1.F, 2.F, 3.F, 4.F, 5.F, 6.F, 7.F}), device));
                auto view = ops::require(storage.narrow(0, 1, 6));
                auto overlapping = ops::require(storage.narrow(0, 0, 4));
                context.copy_slice_(view, overlapping, 0, 1);
                context.synchronize();
                if (!std::ranges::equal(ops::require(storage.data<float>()),
                                        std::array{0.F, 1.F, 0.F, 1.F, 2.F, 3.F, 6.F, 7.F}))
                    return 1;
                {
                    auto queued = context.add(source, source);
                    context.copy_slice_(destination, queued, -2, 0);
                }
                auto consumed = context.add(destination, destination);
                context.synchronize();
                if (ops::require(consumed.data<float>())[0] != 4.F || ops::require(consumed.data<float>())[8] != 20.F)
                    return 1;
            }
            context.scatter_(cache, context.reshape(result, {1, 1, 4}), index);
            context.synchronize();
            if (ops::require(cache_alias.data<float>())[7] != 8.F || ops::is_inplace) return 1;
            ops::is_inplace = true;
            auto protected_input = context.add(mutable_input, mutable_input);
            if (!ops::is_inplace) return 1;
            context.synchronize();
            if (ops::require(alias.data<float>())[3] != 64.F) return 1;
            bool grew = false;
            try {
                context.add_(input, rows);
            } catch (const ops::Failure&) {
                grew = true;
            }
            if (!grew || !ops::is_inplace) return 1;
            bool worker_default = false;
            std::thread worker([&] { worker_default = !ops::is_inplace; });
            worker.join();
            if (!worker_default || !ops::is_inplace) return 1;
            ops::is_inplace = false;
            const auto expected_softmax = context.softmax(mutable_input);
            context.softmax_(mutable_input);
            context.synchronize();
            const auto expected_values = ops::require(expected_softmax.data<float>());
            const auto actual_values = ops::require(mutable_input.data<float>());
            for (std::size_t index = 0; index < expected_values.size(); ++index)
                if (std::abs(expected_values[index] - actual_values[index]) > 1e-5F) return 1;
            const auto expected_gelu = context.gelu(mutable_input);
            context.gelu_(mutable_input);
            context.synchronize();
            auto expected_gelu_values = ops::require(expected_gelu.data<float>());
            auto actual_gelu_values = ops::require(mutable_input.data<float>());
            for (std::size_t index = 0; index < expected_gelu_values.size(); ++index)
                if (std::abs(expected_gelu_values[index] - actual_gelu_values[index]) > 1e-5F) return 1;
            auto scale =
                ops::require(tensor::Tensor::from_host({4}, std::span<const float>(std::array{1.F, 1.F, 1.F, 1.F})));
            auto bias = ops::require(tensor::Tensor::zeros({4}, tensor::DType::F32));
            auto paired = context.residual_layer_norm(input, input, scale, bias, 1e-5F);
            auto paired_expected = context.layer_norm(paired[0], scale, bias, 1e-5F);
            auto paired_again = context.residual_layer_norm(result, result, scale, bias, 1e-5F);
            context.add_(paired_again[0], result);
            context.synchronize();
            if (ops::require(paired[0].data<float>())[3] != 8.F ||
                ops::require(paired_again[0].data<float>())[3] != 24.F)
                return 1;
            const auto paired_values = ops::require(paired[1].data<float>());
            const auto paired_expected_values = ops::require(paired_expected.data<float>());
            for (std::size_t index = 0; index < paired_values.size(); ++index)
                if (std::abs(paired_values[index] - paired_expected_values[index]) > 1e-5F) return 1;
            const auto expected_norm = context.layer_norm(mutable_input, scale, bias, 1e-5F);
            context.layer_norm_(mutable_input, scale, bias, 1e-5F);
            context.synchronize();
            auto expected_norm_values = ops::require(expected_norm.data<float>());
            auto actual_norm_values = ops::require(mutable_input.data<float>());
            for (std::size_t index = 0; index < expected_norm_values.size(); ++index)
                if (std::abs(expected_norm_values[index] - actual_norm_values[index]) > 1e-5F) return 1;
            auto scalar = ops::require(tensor::Tensor::from_host({}, std::span<const float>(std::array{1.F}), device));
            bool expansion_rejected = false;
            try {
                context.add_(scalar, input);
            } catch (const ops::Failure&) {
                expansion_rejected = true;
            }
            context.synchronize();
            if (!expansion_rejected || ops::is_inplace || ops::require(scalar.data<float>())[0] != 1.F) return 1;
            auto shifted = ops::require(
                tensor::Tensor::from_host({1, 3, 1}, std::span<const float>(std::array{1.F, 2.F, 3.F}), device));
            auto flat = context.reshape(shifted, {3, 1});
            auto overlapping = context.reshape(ops::require(flat.narrow(0, 0, 2)), {1, 2, 1});
            auto shift_indices = ops::require(tensor::Tensor::from_host(
                {2}, std::span<const std::int32_t>(std::array<std::int32_t, 2>{1, 2}), device));
            context.scatter_(shifted, overlapping, shift_indices);
            context.synchronize();
            auto shifted_values = ops::require(shifted.data<float>());
            if (shifted_values[0] != 1.F || shifted_values[1] != 1.F || shifted_values[2] != 2.F) return 1;
            auto flat_view = context.reshape(shifted, {3});
            context.multiply_(flat_view, flat_view);
            context.synchronize();
            if (ops::require(shifted.data<float>())[2] != 4.F) return 1;
            auto repeated_indices = ops::require(tensor::Tensor::from_host(
                {2}, std::span<const std::int32_t>(std::array<std::int32_t, 2>{1, 1}), device));
            auto replacements = ops::require(
                tensor::Tensor::from_host({1, 2, 1}, std::span<const float>(std::array{5.F, 6.F}), device));
            context.scatter_(shifted, replacements, repeated_indices);
            context.synchronize();
            if (ops::require(shifted.data<float>())[1] != 6.F) return 1;
            auto invalid_indices = ops::require(tensor::Tensor::from_host(
                {2}, std::span<const std::int32_t>(std::array<std::int32_t, 2>{0, 3}), device));
            bool rejected_indices = false;
            {
                ops::Context isolated(device);
                try {
                    isolated.scatter_(shifted, overlapping, invalid_indices);
                    isolated.synchronize();
                } catch (const ops::Failure&) {
                    rejected_indices = true;
                }
            }
            if (!rejected_indices || ops::is_inplace || ops::require(shifted.data<float>())[0] != 1.F ||
                ops::require(shifted.data<float>())[1] != 6.F || ops::require(shifted.data<float>())[2] != 4.F)
                return 1;
            auto backing = ops::require(tensor::Tensor::zeros({4, 3, 1}, tensor::DType::F32, device));
            auto cache_view = ops::require(backing.narrow(0, 1, 2));
            auto batch_updates = ops::require(
                tensor::Tensor::from_host({2, 1, 1}, std::span<const float>(std::array{7.F, 9.F}), device));
            context.scatter_(cache_view, batch_updates, index);
            context.synchronize();
            auto batch_values = ops::require(backing.data<float>());
            for (std::size_t offset = 0; offset < batch_values.size(); ++offset)
                if (batch_values[offset] != (offset == 4 ? 7.F : offset == 7 ? 9.F : 0.F)) return 1;
            {
                ops::Context fresh(device);
                auto target = ops::require(tensor::Tensor::zeros({2, 3, 1}, tensor::DType::F32, device));
                auto first_index = ops::require(tensor::Tensor::from_host(
                    {1}, std::span<const std::int32_t>(std::array<std::int32_t, 1>{0}), device));
                auto delta = ops::require(tensor::Tensor::from_host(
                    {1}, std::span<const std::int32_t>(std::array<std::int32_t, 1>{1}), device));
                fresh.scatter_(target, batch_updates, first_index);
                auto next_index = fresh.add(first_index, delta);
                auto separate = fresh.scatter(target, batch_updates, next_index);
                fresh.synchronize();
                if (ops::require(target.data<float>())[1] != 0.F || ops::require(separate.data<float>())[1] != 7.F ||
                    ops::require(separate.data<float>())[4] != 9.F)
                    return 1;
            }
            if (device == tensor::Device::cpu()) {
                auto owner = std::make_shared<std::array<float, 4>>(std::array{1.F, 2.F, 3.F, 4.F});
                auto readonly = ops::require(tensor::Tensor::from_blob(
                    {1, 4}, tensor::DType::F32, std::as_bytes(std::span<const float>(*owner)), owner));
                bool rejected_readonly = false;
                try {
                    context.add_(readonly, input);
                } catch (const ops::Failure&) {
                    rejected_readonly = true;
                }
                if (!rejected_readonly || ops::is_inplace || (*owner)[0] != 1.F) return 1;
                rejected_readonly = false;
                try {
                    context.copy_slice_(readonly, input, 0, 0);
                } catch (const ops::Failure&) {
                    rejected_readonly = true;
                }
                if (!rejected_readonly || ops::is_inplace || (*owner)[0] != 1.F) return 1;
            }
        }
        bool rejected = false;
        try {
            ops::Context unsupported(tensor::Device{tensor::DeviceKind::CPU, 1});
        } catch (const ops::Failure& error) {
            rejected = error.error().code == ErrorCode::UNSUPPORTED;
        }
        if (!rejected) return 1;
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}