#pragma once

#include <cstddef>
#include <cstdint>

#include "kidi/core/error.h"
#include "kidi/tensor/device.h"
#include "kidi/tensor/tensor.h"

namespace kidi::layers {

/// Builds FP32 sinusoidal positions shaped [length, hidden_size] on the requested device.
/// hidden_size must be positive and even; even/odd channels contain sine/cosine values.
auto sinusoidal_position_encoding(std::size_t length, std::int32_t hidden_size,
                                  tensor::Device device = tensor::Device::cpu()) -> Result<tensor::Tensor>;

} // namespace kidi::layers