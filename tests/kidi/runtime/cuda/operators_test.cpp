#include "kidi/ops/context.h"
#include "kidi/layers/gemma4.h"
#include "kidi/layers/audio.h"
#include "ynnpack/include/ynnpack_cuda.h"
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>

namespace {
static_assert(
    kidi::tensor::Device::cuda().capabilities().sampling.supports(kidi::tensor::TokenSelection::CHAINED_DEVICE));
static_assert(
    !kidi::tensor::Device::cuda().capabilities().sampling.supports(kidi::tensor::TokenSelection::BATCHED_DEVICE));
static_assert(
    !kidi::tensor::Device::apple_gpu().capabilities().sampling.supports(kidi::tensor::TokenSelection::CHAINED_DEVICE));

auto check_attention_bindings() -> bool {
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
                if (actual != static_cast<float>(iteration + 1)) return false;
        }
        const auto copied = require(context.cast(original, DType::F32).to(tensor::Device::cpu()));
        const auto values = require(copied.data<float>());
        for (std::size_t column = 0; column < static_cast<std::size_t>(width); ++column)
            if (values[128 * width + column] != 1.F) return false;
    }
    return true;
}

auto check_async_readback() -> bool {
    using namespace kidi;
    using ops::require;
    using tensor::Tensor;
    const std::array<std::int32_t, 4> values{1, 2, 3, 4};
    const auto host = require(Tensor::from_host({4}, std::span<const std::int32_t>(values)));
    const auto immediate = require(host.readback());
    if (!require(immediate->ready()) || require(immediate->bytes()).size() != sizeof(values)) return false;
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
        if (!require(pending[index]->ready())) return false;
        const auto bytes = require(pending[index]->bytes());
        std::array<std::int32_t, 2> actual{};
        std::memcpy(actual.data(), bytes.data(), bytes.size());
        const auto multiplier = std::int32_t{1} << (index + 1);
        if (actual != std::array<std::int32_t, 2>{2 * multiplier, 3 * multiplier}) return false;
    }
    return true;
}

auto check_projection_groups() -> bool {
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
            if (require(outputs[index].copy_to_host()) != require(expected.copy_to_host())) return false;
        }
    }
    return captures == 2;
}

auto check_input_staging() -> bool {
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
        if (!std::ranges::equal(require(copied.data<float>()), values) ||
            !std::ranges::equal(require(copied_ids.data<std::int32_t>()), ids))
            return false;
        const auto already = context.stage_inputs(staged);
        if (already[0].storage_identity() != staged[0].storage_identity()) return false;
        if (iteration == 1) stable_bytes = ynn::cuda::allocated_bytes();
        if (iteration > 1 && ynn::cuda::allocated_bytes() != stable_bytes) return false;
        for (auto& value : values) value += 1.F;
    }
    return true;
}

auto check_dynamic_projection_weights() -> bool {
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
            if (value != 256.F * (static_cast<float>(1 << iteration) + 0.5F)) return false;
    }
    return captures == 2;
}

auto check_exported_cache_alias() -> bool {
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
        if (!std::ranges::equal(actual, require(updated.data<float>())) || actual[2] != iteration + 1.F) return false;
    }
    return captures == 2;
}

auto check_constant_cache_release() -> bool {
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
    const auto before = ynn::cuda::allocated_bytes();
    {
        const auto output = context.linear(input, weight, bias, true);
        const auto copied = require(output.to(tensor::Device::cpu()));
        if (!std::ranges::equal(require(copied.data<float>()), std::array{3.F, 7.F})) return false;
    }
    if (ynn::cuda::allocated_bytes() <= before) return false;
    context.release_workspaces();
    return ynn::cuda::allocated_bytes() == before;
}

auto check_readback_source_release() -> bool {
    using namespace kidi;
    using ops::require;
    using tensor::DType;
    using tensor::Tensor;
    const auto device = tensor::Device::cuda();
    ops::Context context(device);
    const auto before = ynn::cuda::allocated_bytes();
    std::shared_ptr<tensor::Readback> pending;
    {
        const auto source = require(Tensor::zeros({65536}, DType::I32, device));
        pending = require(require(source.narrow(0, 1, 1)).readback());
    }
    context.synchronize();
    if (!require(pending->ready()) || require(pending->bytes()).size() != sizeof(std::int32_t)) return false;
    return ynn::cuda::allocated_bytes() == before;
}

auto check_greedy_partitions() -> bool {
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
            if (actual[0] != 1999 || actual[1] != static_cast<std::int32_t>(width - 1)) return false;
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
                if (actual[0] != -1 || actual[1] != -1) return false;
            }
        }
    }
    return true;
}

auto check_bfloat_fusions() -> bool {
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
    if (require(rotary.copy_to_host()) != require(rotary_reference.copy_to_host())) return false;
    for (const bool scaled : {false, true}) {
        const auto fused = context.rms_norm_residual(input, scale, residual, 1e-6F, scaled ? scalar : Tensor{});
        const auto added = context.add(normalized, residual);
        const auto reference = scaled ? context.multiply(added, scalar) : added;
        if (require(fused.copy_to_host()) != require(reference.copy_to_host())) return false;
    }
    const auto gate = context.slice(input, -1, 0, 32), up = context.slice(input, -1, 32, 32);
    if (require(context.gelu_gate(input).copy_to_host()) != require(context.gelu_multiply(gate, up).copy_to_host()) ||
        require(context.silu_gate(input).copy_to_host()) != require(context.silu_multiply(gate, up).copy_to_host()))
        return false;
    return true;
}

auto check_cuda_graph() -> bool {
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
        if (outputs.size() != 2 || require(outputs[0].data<std::int32_t>())[0] != iteration) return false;
        const auto copied = require(context.cast(cache, DType::F32).to(tensor::Device::cpu()));
        const auto actual = require(copied.data<float>());
        for (std::size_t column = 0; column < 33; ++column)
            if (actual[row[0] * 33 + column] != (column == static_cast<std::size_t>(iteration) ? iteration + 1.F : 0.F))
                return false;
        retained = outputs[1];
    }
    if (eager_calls != 2) return false;
    context.clear_replays("cuda-test");
    const auto result = require(context.cast(retained, DType::F32).to(tensor::Device::cpu()));
    return require(result.data<float>())[5] == 6.F;
}

auto check_bfloat_context() -> bool {
    using namespace kidi;
    using ops::require;
    using tensor::DType;
    using tensor::Tensor;
    ops::Context context(tensor::Device::cuda(), false, std::nullopt, DType::BF16);
    if (context.activation_dtype() != DType::BF16) return false;
    const auto source = require(Tensor::from_host(
        {1, 2, 4}, std::span<const float>(std::array{1.F, 2.F, 3.F, 4.F, -1.F, -2.F, -3.F, -4.F}), context.device()));
    const auto input = context.cast(source, DType::BF16);
    const auto scale =
        require(Tensor::from_host({4}, std::span<const float>(std::array{1.F, 1.F, 1.F, 1.F}), context.device()));
    const auto scalar = require(Tensor::from_host({}, std::span<const float>(std::array{0.5F}), context.device()));
    auto hidden = context.multiply(context.rms_norm(input, scale, 1e-6F), scalar);
    if (hidden.dtype() != DType::BF16) return false;
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
    for (const auto value : require(copied.data<float>()))
        if (!std::isfinite(value)) return false;
    return projected.dtype() == DType::BF16 && attended.dtype() == DType::BF16 && token.dtype() == DType::I32;
}

auto check_scatter_updates() -> bool {
    using namespace kidi;
    using ops::require;
    using tensor::DType;
    using tensor::Tensor;
    ops::Context context(tensor::Device::cuda());
    auto cache = require(Tensor::zeros({2, 9216, 4}, DType::F32, context.device()));
    const std::array<float, 24> values{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                                       13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
    const std::array<std::int32_t, 3> rows{5, 9215, 5};
    const auto updates = require(Tensor::from_host({2, 3, 4}, std::span<const float>(values), context.device()));
    const auto indices = require(Tensor::from_host({3}, std::span<const std::int32_t>(rows), context.device()));
    const auto identity = cache.storage_identity();
    const auto before = ynn::cuda::allocated_bytes();
    context.scatter_(cache, updates, indices);
    context.synchronize();
    if (cache.storage_identity() != identity || ynn::cuda::allocated_bytes() != before) return false;
    const auto copied = require(cache.to(tensor::Device::cpu()));
    const auto actual = require(copied.data<float>());
    for (std::size_t batch = 0; batch < 2; ++batch)
        for (std::size_t row = 0; row < 9216; ++row)
            for (std::size_t channel = 0; channel < 4; ++channel) {
                const auto expected = row == 5      ? values[batch * 12 + 8 + channel]
                                      : row == 9215 ? values[batch * 12 + 4 + channel]
                                                    : 0.F;
                if (actual[(batch * 9216 + row) * 4 + channel] != expected) return false;
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
        if (alias_values[index] != initial[index % 8]) return false;
    return true;
}

auto check_scoped_layers() -> bool {
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
            if (module->device() != device) return false;
            for (const auto& [name, value] : module->state_dict())
                if (!value.defined() || value.device() != device) return false;
        }
        constexpr std::array VALUES{1.F, 2.F, 3.F, 4.F, -2.F, 0.F, 2.F, 4.F};
        constexpr std::array ONES{1.F, 1.F, 1.F, 1.F};
        constexpr std::array ZEROS{0.F, 0.F, 0.F, 0.F};
        const auto host = require(Tensor::from_host({2, 4}, std::span<const float>(VALUES)));
        const auto scale = require(Tensor::from_host({4}, std::span<const float>(ONES)));
        const auto bias = require(Tensor::from_host({4}, std::span<const float>(ZEROS)));
        const auto input = require(host.to(rms->device()));
        ops::Context cpu(tensor::Device::cpu()), context(rms->device());
        const auto same_values = [&](const Tensor& output, const Tensor& reference) {
            if (output.device() != device) return false;
            context.synchronize();
            const auto copied = require(output.to(tensor::Device::cpu()));
            const auto actual = require(copied.data<float>()), expected = require(reference.data<float>());
            if (actual.size() != expected.size()) return false;
            for (std::size_t index = 0; index < actual.size(); ++index)
                if (std::abs(actual[index] - expected[index]) > 2e-5F) return false;
            return true;
        };
        if (!same_values(rms->forward(context, input), cpu.rms_norm(host, scale, EPSILON)) ||
            !same_values(norm->forward(context, input), cpu.layer_norm(host, scale, bias, EPSILON)))
            return false;
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
                if (module->device() != device) return false;
                for (const auto& [name, value] : module->state_dict())
                    if (name.ends_with("scale")) {
                        const auto copied = require(value.to(tensor::Device::cpu()));
                        for (const auto scale_value : require(copied.data<float>()))
                            if (scale_value != 1.F) return false;
                    }
            }
            const auto zeros = require(Tensor::zeros({2, 4}, DType::F32));
            if (!same_values(projection->forward(context, input), zeros)) return false;
        }
        {
            const ModuleScope nested(tensor::Device::cpu());
            layers::RmsNorm child(4, EPSILON);
            if (child->device() != tensor::Device::cpu()) return false;
        }
        if (module_device != rms->device()) return false;
        bool rejected = false;
        try {
            rms->forward(context, host);
        } catch (const ops::Failure& error) {
            rejected = error.error().code == ErrorCode::INVALID_ARGUMENT;
        }
        if (!rejected) return false;
        std::array<float, 32> weights{};
        for (std::size_t index = 0; index < weights.size(); ++index) weights[index] = static_cast<float>(index + 1);
        const auto table = require(Tensor::from_host({8, 4}, std::span<const float>(weights)));
        constexpr std::array<std::int32_t, 4> TOKENS{0, 1, 2, 3};
        const auto tokens = require(Tensor::from_host({2, 2}, std::span<const std::int32_t>(TOKENS), device));
        constexpr std::array EXPECTED{26.F, 28.F, 30.F, 32.F, 34.F, 36.F, 38.F, 40.F};
        const auto reference = require(Tensor::from_host({1, 2, 4}, std::span<const float>(EXPECTED)));
        if (!same_values(context.embedding_sum(tokens, table, {}, 4, 0, 1.F), reference)) return false;
    }
    return module_device == previous_device && module_dtype == previous_dtype &&
           allocate_parameters == previous_allocate;
}

auto check_projection_paths() -> bool {
    using namespace kidi;
    using ops::require;
    using tensor::DType;
    using tensor::Tensor;
    ops::Context cpu;
    ops::Context cuda(tensor::Device::cuda());
    constexpr std::int64_t WIDTH = 16, COLUMNS = 8;
    for (const std::int64_t rows : {1, 8}) {
        std::vector<float> values(rows * WIDTH), scales(COLUMNS), biases(COLUMNS);
        for (std::size_t index = 0; index < values.size(); ++index)
            values[index] = index / WIDTH == 2   ? 0.F
                            : index / WIDTH == 3 ? (index % 2 ? -1000.F : 1000.F)
                                                 : (static_cast<int>(index * 13 % 41) - 17) * 0.125F;
        for (std::size_t index = 0; index < scales.size(); ++index) {
            scales[index] = static_cast<float>(index + 1) * 0.001F;
            biases[index] = static_cast<float>(index) * 0.01F;
        }
        const auto host = require(Tensor::from_host({rows, WIDTH}, std::span<const float>(values)));
        const auto input = require(host.to(tensor::Device::cuda()));
        const auto scale = require(Tensor::from_host({COLUMNS, 1}, std::span<const float>(scales)));
        const auto bias = require(Tensor::from_host({COLUMNS}, std::span<const float>(biases)));
        for (const int bits : {0, 2, 4, 8}) {
            std::vector<std::uint8_t> weights(COLUMNS * WIDTH / (bits ? 8 / bits : 1), 0);
            for (std::size_t index = 0; index < COLUMNS * WIDTH; ++index) {
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
                require(Tensor::from_bytes({COLUMNS, WIDTH / (bits ? 8 / bits : 1)}, bits ? DType::U8 : DType::I8,
                                           std::as_bytes(std::span(weights))));
            const auto expected = bits ? cpu.packed_linear(host, weight, scale, bits, WIDTH, 0.125F, 0.0625F)
                                       : cpu.quantized_linear(host, weight, scale, bias, true);
            for (int repetition = 0; repetition < 2; ++repetition) {
                const auto actual =
                    require((bits ? cuda.packed_linear(input, weight, scale, bits, WIDTH, 0.125F, 0.0625F)
                                  : cuda.quantized_linear(input, weight, scale, bias, true))
                                .to(tensor::Device::cpu()));
                const auto actual_values = require(actual.data<float>()),
                           expected_values = require(expected.data<float>());
                for (std::size_t index = 0; index < actual_values.size(); ++index)
                    if (std::abs(actual_values[index] - expected_values[index]) >
                        1e-4F + (bits ? 0.F : 1e-6F * std::abs(expected_values[index]))) {
                        std::cerr << "projection mismatch rows=" << rows << " bits=" << bits << " index=" << index
                                  << " actual=" << actual_values[index] << " expected=" << expected_values[index]
                                  << '\n';
                        return false;
                    }
            }
        }
    }
    return true;
}
} // namespace

auto main() -> int {
    using namespace kidi;
    try {
        if (!ynn::cuda::available()) return 77;
        if (!check_constant_cache_release())
            throw std::runtime_error("CUDA workspace release retained unused constants");
        if (!check_dynamic_projection_weights()) throw std::runtime_error("CUDA fusion froze dynamic step weights");
        if (!check_exported_cache_alias()) throw std::runtime_error("CUDA graph exported a stale cache alias");
        if (!check_readback_source_release()) throw std::runtime_error("CUDA readback retained its completed source");
        if (!check_attention_bindings()) throw std::runtime_error("CUDA attention retained a stale KV pointer");
        if (!check_async_readback()) throw std::runtime_error("CUDA nonblocking readback snapshot failed");
        if (!check_projection_groups()) throw std::runtime_error("CUDA shared-input projection fusion failed");
        if (!check_input_staging()) throw std::runtime_error("CUDA reusable input staging failed");
        if (!check_greedy_partitions()) throw std::runtime_error("CUDA parallel token selection changed tie-breaking");
        if (!check_bfloat_fusions()) throw std::runtime_error("CUDA BF16 fusion changed intermediate rounding");
        if (!check_cuda_graph()) throw std::runtime_error("CUDA graph input staging or cache writeback failed");
        if (!check_bfloat_context()) throw std::runtime_error("CUDA BF16 context pipeline failed");
        if (!check_scatter_updates()) throw std::runtime_error("CUDA sparse cache scatter failed");
        if (!check_scoped_layers()) throw std::runtime_error("CUDA scoped placement or layer parity failed");
        if (!check_projection_paths()) throw std::runtime_error("accelerated projection CPU parity failed");
        if (tensor::parse_cuda_device("cuda") != tensor::Device::cuda(0) ||
            tensor::parse_cuda_device("cuda:0") != tensor::Device::cuda(0) ||
            tensor::parse_cuda_device("cuda:12") != tensor::Device::cuda(12))
            return 1;
        for (const auto invalid :
             {"cuda:", "cuda:-0", "cuda:-1", "cuda:+1", "cuda:1x", "cuda:999999999999999999", "cpu"})
            if (tensor::parse_cuda_device(invalid)) return 1;
        const std::array<float, 6> values{1, 2, 3, 4, 5, 6};
        const std::array<float, 3> bias{10, 20, 30};
        auto input =
            ops::require(tensor::Tensor::from_host({2, 3}, std::span<const float>(values), tensor::Device::cuda()));
        auto other = ops::require(tensor::Tensor::from_host({3}, std::span<const float>(bias), tensor::Device::cuda()));
        ops::Context context(tensor::Device::cuda());
        const auto result = ops::require(context.add(input, other).to(tensor::Device::cpu()));
        const auto actual = ops::require(result.data<float>());
        if (!std::ranges::equal(actual, std::array<float, 6>{11, 22, 33, 14, 25, 36})) return 1;
        context.add_(input, other);
        const auto updated = ops::require(input.to(tensor::Device::cpu()));
        if (!std::ranges::equal(ops::require(updated.data<float>()), actual)) return 1;
        ops::Context cpu;
        const auto host = ops::require(input.to(tensor::Device::cpu()));
        const auto gpu_softmax = ops::require(context.softmax(input).to(tensor::Device::cpu()));
        const auto cpu_softmax = cpu.softmax(host);
        const auto expected = ops::require(cpu_softmax.data<float>());
        const auto probabilities = ops::require(gpu_softmax.data<float>());
        for (std::size_t index = 0; index < expected.size(); ++index)
            if (std::abs(probabilities[index] - expected[index]) > 1e-5F) return 1;
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
            if (std::abs(projected[index] - reference[index]) > 1e-4F) return 1;
        const std::array<std::int32_t, 2> ids{1, 0};
        const auto indices =
            ops::require(tensor::Tensor::from_host({2}, std::span<const std::int32_t>(ids), tensor::Device::cuda()));
        const auto embedded =
            ops::require(context.embedding(indices, weight, scale, 3, 8, 1.F).to(tensor::Device::cpu()));
        if (!std::ranges::equal(embedded.shape(), std::array<std::int64_t, 3>{1, 2, 3})) return 1;
        if (!std::ranges::equal(ops::require(embedded.data<float>()),
                                std::array<float, 6>{2, 2.5F, -3, 0.25F, -0.5F, 0.75F}))
            return 1;
        const auto dense_weight = ops::require(weight.to(tensor::Device::cpu()));
        const auto float_weight = cpu.cast(dense_weight, tensor::DType::F32);
        context.linear(input, float_weight, {}, true);
        const auto reshaped_weight = ops::require(float_weight.reshape({3, 2}));
        const auto narrow_input = context.slice(input, 1, 0, 2);
        const auto reshaped_projection =
            ops::require(context.linear(narrow_input, reshaped_weight, {}, true).to(tensor::Device::cpu()));
        const auto reshape_reference = cpu.linear(cpu.slice(host, 1, 0, 2), reshaped_weight, {}, true);
        if (!std::ranges::equal(ops::require(reshaped_projection.data<float>()),
                                ops::require(reshape_reference.data<float>())))
            return 1;
        bool rejected_dtype = false;
        try {
            context.cast(input, tensor::DType::U32);
        } catch (const ops::Failure& failure) {
            rejected_dtype = failure.error().code == ErrorCode::UNSUPPORTED;
        }
        if (!rejected_dtype) return 1;
        const auto empty = ops::require(tensor::Tensor::empty({0, 3}, tensor::DType::F32, tensor::Device::cuda()));
        const auto empty_output = context.add(empty, other);
        if (empty_output.numel() != 0 || !std::ranges::equal(empty_output.shape(), std::array<std::int64_t, 2>{0, 3}))
            return 1;
        const auto gpu_matrix =
            ops::require(context.matmul(input, ops::require(float_weight.to(tensor::Device::cuda())), true)
                             .to(tensor::Device::cpu()));
        const auto cpu_matrix = cpu.matmul(host, float_weight, true);
        const auto dense_expected = ops::require(cpu_matrix.data<float>());
        const auto dense_actual = ops::require(gpu_matrix.data<float>());
        for (std::size_t index = 0; index < dense_expected.size(); ++index)
            if (std::abs(dense_actual[index] - dense_expected[index]) > 1e-5F) return 1;
        const std::array<float, 3> norm_scale_values{1, 0.5F, 2};
        const auto norm_scale = ops::require(tensor::Tensor::from_host({3}, std::span<const float>(norm_scale_values)));
        const auto normalized = ops::require(context.rms_norm(input, norm_scale, 1e-5F).to(tensor::Device::cpu()));
        const auto norm_reference = cpu.rms_norm(host, norm_scale, 1e-5F);
        const auto norm_expected = ops::require(norm_reference.data<float>());
        const auto norm_actual = ops::require(normalized.data<float>());
        for (std::size_t index = 0; index < norm_expected.size(); ++index)
            if (std::abs(norm_actual[index] - norm_expected[index]) > 1e-5F) return 1;
        const std::array<float, 4> signal_values{1, 2, 3, 4};
        const std::array<float, 3> kernel_values{1, 2, 1};
        const std::array<float, 1> zero{0};
        const auto signal = ops::require(
            tensor::Tensor::from_host({1, 4, 1}, std::span<const float>(signal_values), tensor::Device::cuda()));
        const auto kernel = ops::require(tensor::Tensor::from_host({1, 3}, std::span<const float>(kernel_values)));
        const auto offset = ops::require(tensor::Tensor::from_host({1}, std::span<const float>(zero)));
        const auto convolved =
            ops::require(context.conv1d(signal, kernel, {}, offset, 3, 1, 1, 1).to(tensor::Device::cpu()));
        if (!std::ranges::equal(ops::require(convolved.data<float>()), std::array<float, 4>{4, 8, 12, 11})) return 1;
        const auto norm_bias = ops::require(tensor::Tensor::zeros({3}, tensor::DType::F32));
        const auto gpu_pair = context.residual_layer_norm(input, input, norm_scale, norm_bias, 1e-5F);
        const auto cpu_pair = cpu.residual_layer_norm(host, host, norm_scale, norm_bias, 1e-5F);
        for (std::size_t output_index = 0; output_index < 2; ++output_index) {
            const auto pair_host = ops::require(gpu_pair[output_index].to(tensor::Device::cpu()));
            const auto pair_actual = ops::require(pair_host.data<float>());
            const auto pair_expected = ops::require(cpu_pair[output_index].data<float>());
            for (std::size_t index = 0; index < pair_actual.size(); ++index)
                if (std::abs(pair_actual[index] - pair_expected[index]) > 1e-5F) return 1;
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
            if (std::abs(attention_actual[index] - attention_expected[index]) > 1e-4F) return 1;
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
            if (std::abs(explicit_actual[index] - explicit_expected[index]) > 1e-5F) return 1;
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
            if (std::abs(packed_actual[index] - packed_expected[index]) > 1e-5F) return 1;
        std::cout << "CUDA numerical checks include W8A8 projection, embedding, cuBLAS GEMM, softmax, RMS/layer norm, "
                     "convolution, attention, and device routing\n";
        for (int index = 0; index < ynn::cuda::device_count(); ++index) {
            const auto device = tensor::Device::cuda(index);
            const auto selected_input = ops::require(host.to(device));
            const auto selected_bias = ops::require(other.to(tensor::Device::cpu())).to(device);
            ops::Context selected_context(device);
            const auto selected_output = selected_context.add(selected_input, ops::require(selected_bias));
            if (selected_output.device() != device) return 1;
            const auto copied = ops::require(selected_output.to(tensor::Device::cpu()));
            if (!std::ranges::equal(ops::require(copied.data<float>()), std::array<float, 6>{21, 42, 63, 24, 45, 66}))
                return 1;
        }
        if (tensor::Tensor::empty({1}, tensor::DType::F32, tensor::Device::cuda(ynn::cuda::device_count()))) return 1;
        std::cout << "Kidi CUDA storage, broadcast and in-place tests PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}