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
/// Same result as pack_weight(weight, 8, width) for a host matrix, packed in row chunks on the CPU worker pool.
auto pack_rows_int8(const tensor::Tensor& weight) -> Result<PackedWeight>;
/// Requantizes checkpoint experts [experts, rows, width] to row-scaled INT8, without expanding the whole bank.
auto pack_expert_weights(const tensor::Tensor& values, const tensor::Tensor& scales, const tensor::Tensor& global,
                         bool interleaved) -> Result<PackedWeight>;
} // namespace kidi::ops