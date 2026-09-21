#pragma once

#include "kidi/tensor/tensor.h"

namespace kidi::ops {
struct PackedWeight {
    tensor::Tensor values, scales;
    std::int32_t bits, group_size;
};
auto pack_weight(const tensor::Tensor& weight, std::int32_t bits = 4, std::int32_t group_size = 128)
    -> Result<PackedWeight>;
} // namespace kidi::ops