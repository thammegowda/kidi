#pragma once

#include "kidi/tensor/tensor.h"

namespace kidi::ops {
enum class BlockScaledFormat { NVFP4, MXFP8, INT4, VQ3D, VQ4D };
/// A composite weight format, not a scalar tensor dtype: packed values, block scales and optional global scale.
/// VQ3D/VQ4D store 3 or 4 Hadamard-rotated weights per byte (2.84 or 2.14 bpw) with U8 group and F32 row scales.
struct BlockScaledWeight {
    BlockScaledFormat format;
    tensor::Tensor values, scales, global;
    std::int32_t group_size = 0; // 0 selects NVFP4:16, MXFP8:32, symmetric INT4:128, VQ3D:48, VQ4D:64.
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
/// Requantizes NVFP4/MXFP8 checkpoint experts directly to VQ3D or VQ4D, in row chunks on the CPU worker pool.
auto pack_vq_expert_weights(const tensor::Tensor& values, const tensor::Tensor& scales, const tensor::Tensor& global,
                            bool interleaved, BlockScaledFormat format) -> Result<BlockScaledWeight>;
/// Packs a floating host tensor [rows, cols] or [experts, rows, cols] into VQ3D or VQ4D.
auto pack_vq_weight(const tensor::Tensor& weight, BlockScaledFormat format) -> Result<BlockScaledWeight>;
} // namespace kidi::ops