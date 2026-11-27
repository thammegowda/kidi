#pragma once

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

} // namespace kidi::runtime::parity
