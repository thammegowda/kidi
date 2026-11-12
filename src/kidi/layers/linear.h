#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string_view>

#include "kidi/core/module.h"
#include "kidi/ops/context.h"

namespace kidi::layers {

using tensor::Tensor;
class GatedFeedForwardImpl;

struct QuantizedState {
    Tensor scale;
    mutable Tensor decoded;

    auto decode_vector(const Tensor& values, std::string_view name, tensor::Device target = tensor::Device::cpu()) const
        -> const Tensor&;
};

KIDI_MODULE(Linear);

/// Affine projection over the final dimension with bound weights and optional bias.
/// Supports FP32, BF16, per-channel INT8, and packed 2/4/8-bit weight encodings,
/// plus optional CPU HNSW sub-vocabulary selection for output heads.
class LinearImpl : public Module {
public:
    LinearImpl(std::int32_t input_size, std::int32_t output_size, bool transpose = false, bool bias = true,
               std::int32_t packed_bits = 0, bool quantized_bias = false, std::int32_t group_size = 0);
    ~LinearImpl();
    auto forward(ops::Context&, const Tensor&) const -> Tensor;
    /// Returns I32 [rows] token IDs; an indexed head is approximate and does not materialize vocabulary logits.
    auto select(ops::Context&, const Tensor&, ops::TokenOutput output = ops::TokenOutput::HOST) const -> Tensor;
    auto bind_hnsw(const std::filesystem::path& path, const Tensor& packed_zero = {}) -> Result<void>;
    auto has_hnsw() const noexcept -> bool;
    auto check_hnsw_binding() const -> void;
    auto release_workspaces() const -> void;
    auto weight() const -> const Tensor& { return weight_; }

private:
    friend class GatedFeedForwardImpl;
    struct HnswState;
    auto bias() const -> const Tensor&;

    Tensor weight_, bias_;
    std::optional<QuantizedState> qstate_, bias_qstate_;
    Tensor packed_scale_;
    Tensor input_scale_, output_scale_;
    std::int32_t packed_bits_ = 0, input_size_ = 0, group_size_ = 0;
    bool transpose_, has_bias_;
    mutable std::unique_ptr<HnswState> hnsw_;
};

} // namespace kidi::layers
