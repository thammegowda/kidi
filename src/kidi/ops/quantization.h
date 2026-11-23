#pragma once

#include "kidi/tensor/tensor.h"

namespace kidi::ops {
enum class BlockScaledFormat { NVFP4, MXFP8, INT4 };
/// A composite weight format, not a scalar tensor dtype: packed values, block scales and optional global scale.
struct BlockScaledWeight {
    BlockScaledFormat format;
    tensor::Tensor values, scales, global;
    std::int32_t group_size = 0; // 0 selects NVFP4:16, MXFP8:32, symmetric INT4:128.
};
struct PackedWeight {
    tensor::Tensor values, scales, zeros;
    std::int32_t bits, group_size;
};
auto pack_weight(const tensor::Tensor& weight, std::int32_t bits = 4, std::int32_t group_size = 128,
                 bool asymmetric_clip = false) -> Result<PackedWeight>;
} // namespace kidi::ops