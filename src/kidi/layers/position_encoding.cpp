#include "kidi/layers/position_encoding.h"

#include <cmath>
#include <span>
#include <vector>

namespace kidi::layers {

auto sinusoidal_position_encoding(std::size_t length, std::int32_t hidden_size, tensor::Device device)
    -> Result<tensor::Tensor> {
    if (hidden_size <= 0 || hidden_size % 2 != 0) {
        return std::unexpected(
            Error{ErrorCode::INVALID_ARGUMENT, "sinusoidal position encoding requires a positive even hidden size"});
    }
    std::vector<float> values(length * static_cast<std::size_t>(hidden_size));
    for (std::size_t position = 0; position < length; ++position) {
        for (std::int32_t index = 0; index < hidden_size; index += 2) {
            const auto exponent = static_cast<float>(index) * -(std::log(10000.0F) / hidden_size);
            const auto angle = static_cast<float>(position) * std::exp(exponent);
            values[position * hidden_size + index] = std::sin(angle);
            values[position * hidden_size + index + 1] = std::cos(angle);
        }
    }
    return tensor::Tensor::from_host({static_cast<std::int64_t>(length), hidden_size}, std::span<const float>(values),
                                     device);
}

} // namespace kidi::layers