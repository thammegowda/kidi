#pragma once

#include "kidi/tensor/tensor.h"
#include <cstdint>
#include <span>

namespace kidi::runtime::parity {

/// SLEEF-compatible sine used by the pinned CPU reference.
auto sine(float value) -> float;
/// SLEEF-compatible cosine used by the pinned CPU reference.
auto cosine(float value) -> float;
/// In-place softmax with the pinned PyTorch CPU reduction order.
auto softmax(std::span<float> values, std::span<const float> mask) -> void;
/// Sum of squares with the pinned PyTorch CPU cascade reduction.
auto sum_squares(std::span<const float> values) -> float;
/// Packed-weight linear using bounded dequantization tiles and reference-compatible accumulation.
auto packed_linear(const tensor::Tensor& input, const tensor::Tensor& weight, const tensor::Tensor& scales,
                   std::int32_t bits, std::int32_t group_size, float input_scale, float output_scale)
    -> Result<tensor::Tensor>;

} // namespace kidi::runtime::parity
