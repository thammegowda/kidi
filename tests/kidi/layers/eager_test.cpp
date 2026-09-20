#include "kidi/ops/context.h"
#include "kidi/tensor/arena.h"
#include <array>
#include <cmath>
#include <iostream>
#include <thread>

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
            auto input = ops::require(
                tensor::Tensor::from_host({1, 4}, std::span<const float>(std::array{1.F, 2.F, 3.F, 4.F}), device));
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
            context.synchronize();
            const auto rms_values = ops::require(rms_output.data<float>());
            for (std::size_t channel = 0; channel < 4; ++channel) {
                const auto expected = rms_input_values[channel] * rms_scale_values[channel] / std::sqrt(7.5F + 1e-6F);
                if (std::abs(rms_values[channel] - expected) > 1e-5F || rms_values[4 + channel] != 0.F) return 1;
                if (std::abs(ops::require(other_rms.data<float>())[channel] - 2.F * expected) > 1e-5F) return 1;
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
            auto hyperbolic = context.tanh(activation_input);
            auto projection = context.linear(activation_input, activation_input, {}, true);
            context.synchronize();
            const auto activation_values = ops::require(activation_input.data<float>());
            const auto activated_values = ops::require(activated.data<float>());
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
            context.synchronize();
            const auto rotated_values = ops::require(rotated.data<float>());
            if (rotated_values[0] != -3.F || rotated_values[1] != 2.F || rotated_values[2] != 1.F ||
                rotated_values[3] != 4.F)
                return 1;
            const auto attention_values = ops::require(attended.data<float>());
            const auto probability = 1.F / (1.F + std::exp(1.F));
            for (std::size_t head = 0; head < 2; ++head)
                if (std::abs(attention_values[head * 2] - probability) > 1e-5F ||
                    std::abs(attention_values[head * 2 + 1] - (1.F - probability)) > 1e-5F)
                    return 1;
            auto saved = result;
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