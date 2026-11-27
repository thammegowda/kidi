#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include "kidi/tensor/tensor.h"
#include "kidi/runtime/operator.h"
#include "kidi/runtime/mps/command_batch.h"

namespace kidi::runtime::mps {
auto encode_expand_packed_weight(CommandBatch& batch, const tensor::Tensor& weight, const tensor::Tensor& scales,
                                 tensor::Tensor& output, std::int32_t bits, std::int32_t group_size) -> Result<void>;
auto encode_packed_linear(CommandBatch& batch, const tensor::Tensor& input, const tensor::Tensor& weight,
                          const tensor::Tensor& scales, tensor::Tensor& output, std::int32_t bits,
                          std::int32_t group_size, float input_scale = 0.F, float output_scale = 0.F,
                          bool vector_projection = false) -> Result<void>;
auto encode_packed_gate_up(CommandBatch& batch, const tensor::Tensor& input, const tensor::Tensor& weight,
                           const tensor::Tensor& scales, tensor::Tensor& output, float input_scale, float output_scale,
                           float hidden_scale) -> Result<void>;
/// Scratch words for interleaved VQ prefill, or zero for the GEMV path.
auto vq_expert_route_words(std::size_t tokens, std::size_t top, std::size_t experts) -> std::int64_t;
/// VQ3D/VQ4D routed MoE projection (`gated=true` computes softcapped SiLU(gate)*up*route_weight;
/// `gated=false` computes and sums across routed experts in rotated space before a single output Hadamard).
auto encode_vq_experts(CommandBatch& batch, const tensor::Tensor& input, const tensor::Tensor& values,
                       const tensor::Tensor& scales, const tensor::Tensor& global, const tensor::Tensor& routes,
                       tensor::Tensor& x_rot_scratch, tensor::Tensor& interleaved, tensor::Tensor& pair_scratch,
                       tensor::Tensor& route_scratch, tensor::Tensor& output, std::int32_t dim, bool gated,
                       float gate_cap, float up_cap) -> Result<void>;
/// Metal 4 Flash-style tiled relative attention (head width 128, tokens >= 16); returns false when Metal 4 is
/// unavailable so callers fall back to the Metal 3 kernel.
auto encode_relative_attention_mpp(CommandBatch& batch, std::span<const std::int64_t> attributes, TensorInputs inputs,
                                   tensor::Tensor& output) -> Result<bool>;

class QuantizedLinear {
public:
    QuantizedLinear(QuantizedLinear&&) noexcept;
    auto operator=(QuantizedLinear&&) noexcept -> QuantizedLinear&;
    ~QuantizedLinear();
    static auto create(const tensor::Tensor& weights, const tensor::Tensor& scales, const tensor::Tensor& bias)
        -> Result<QuantizedLinear>;
    auto prepare(const tensor::Tensor& input) -> Result<void>;
    auto encode(CommandBatch& batch, const tensor::Tensor& input, tensor::Tensor& output) -> Result<void>;
    auto run(const tensor::Tensor& input, tensor::Tensor& output) -> Result<void>;

private:
    struct Impl;
    explicit QuantizedLinear(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};
} // namespace kidi::runtime::mps