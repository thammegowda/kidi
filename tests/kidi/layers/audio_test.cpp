#include "kidi/layers/audio.h"

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
}
