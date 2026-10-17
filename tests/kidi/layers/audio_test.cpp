#include "kidi/layers/audio.h"
#include "kidi/layers/higgs_audio.h"

#include <array>
#include <cassert>
#include <cmath>

namespace {

auto set_quantized(kidi::Module& module, std::int8_t weight, float scale, std::int8_t bias = 0, float bias_scale = 1.F)
    -> void {
    for (auto& [name, tensor] : module.state_dict()) {
        if (tensor.dtype() == kidi::tensor::DType::I8) {
            auto values = tensor.data<std::int8_t>();
            assert(values);
            std::ranges::fill(*values, name.ends_with("bias") ? bias : weight);
        } else {
            auto values = tensor.data<float>();
            assert(values);
            std::ranges::fill(*values, name.ends_with("bias_scale") ? bias_scale : scale);
        }
    }
}

auto close(float left, float right) -> bool { return std::abs(left - right) < 1e-2F; }

auto higgs_tokens(kidi::ops::Context& context) -> void {
    using namespace kidi;
    layers::HiggsDecoder decoder(layers::HiggsDecoderShape{2, 4, 2, 2, 2, 4, {2}});
    set_quantized(*decoder, 1, 0.5F);
    for (auto& [name, tensor] : decoder->state_dict())
        if (name.ends_with("codebook.weight")) {
            auto values = ops::require(tensor.data<std::int8_t>());
            for (std::size_t index = 0; index < values.size(); ++index)
                values[index] = static_cast<std::int8_t>(index + 1);
        }
    constexpr std::array<std::int32_t, 4> CODES{0, 1, 2, 3};
    const auto tokens = ops::require(tensor::Tensor::from_host({2, 2}, std::span<const std::int32_t>(CODES)));
    const auto output = decoder->forward(context, tokens);
    context.synchronize();
    const auto values = ops::require(output.data<float>());
    const std::vector expected(values.begin(), values.end());
    assert(expected.size() == 4);
    assert(std::ranges::all_of(expected, [](float value) { return std::isfinite(value); }));

    constexpr std::array<std::int32_t, 6> PADDED{9, 9, 0, 1, 2, 3};
    const auto padded = ops::require(tensor::Tensor::from_host({3, 2}, std::span<const std::int32_t>(PADDED)));
    const auto view = ops::require(padded.narrow(0, 1, 2));
    const auto replayed = decoder->forward(context, view);
    context.synchronize();
    const auto actual = ops::require(replayed.data<float>());
    assert(actual.size() == expected.size());
    for (std::size_t index = 0; index < actual.size(); ++index)
        assert(std::abs(actual[index] - expected[index]) <= 1e-5F * std::max(1.F, std::abs(expected[index])));

    const auto rejects = [&](const tensor::Tensor& input) {
        try {
            (void)decoder->forward(context, input);
            return false;
        } catch (const ops::Failure& error) {
            return error.error().code == ErrorCode::INVALID_ARGUMENT;
        }
    };
    assert(rejects(ops::require(tokens.reshape({4}))));
    constexpr std::array<std::int32_t, 4> INVALID{0, 4, 1, 2};
    assert(rejects(ops::require(tensor::Tensor::from_host({2, 2}, std::span<const std::int32_t>(INVALID)))));
}

} // namespace

int main() {
    using namespace kidi;
    const ModuleScope scope(tensor::DType::I8, true, tensor::Device::cpu());
    ops::Context context(tensor::Device::cpu());

    layers::Conv1d convolution(1, 1, 3, 1, 1);
    set_quantized(*convolution, 1, 1.F, 1, 1.F);
    const std::array input_values{1.F, 2.F, 3.F};
    auto input = tensor::Tensor::from_host({1, 3, 1}, std::span<const float>(input_values));
    assert(input);
    const auto convolved = convolution->forward(context, *input);
    context.synchronize();
    const auto convolved_values = convolved.data<float>();
    assert(convolved_values && convolved_values->size() == 3);
    assert(close((*convolved_values)[0], 4.F));
    assert(close((*convolved_values)[1], 7.F));
    assert(close((*convolved_values)[2], 6.F));

    layers::ConvTranspose1d transposed(1, 1, 4, 2, 1);
    set_quantized(*transposed, 1, 1.F);
    const std::array transposed_input_values{1.F, 2.F};
    auto transposed_input = tensor::Tensor::from_host({1, 2, 1}, std::span<const float>(transposed_input_values));
    assert(transposed_input);
    const auto expanded = transposed->forward(context, *transposed_input);
    const auto expanded_values = expanded.data<float>();
    const std::array expected{1.F, 3.F, 3.F, 2.F};
    assert(expanded_values && expanded_values->size() == expected.size());
    for (std::size_t index = 0; index < expected.size(); ++index)
        assert(close((*expanded_values)[index], expected[index]));

    layers::Snake1d snake(1);
    set_quantized(*snake, 1, 1.F);
    const auto activated = snake->forward(context, *input);
    const auto activated_values = activated.data<float>();
    assert(activated_values && activated_values->size() == input_values.size());
    for (std::size_t index = 0; index < input_values.size(); ++index)
        assert(close((*activated_values)[index],
                     input_values[index] + std::sin(input_values[index]) * std::sin(input_values[index])));
    higgs_tokens(context);
}
