#pragma once

#include <map>

#include <yaml-cpp/yaml.h>

#include "kidi/layers/gemma4.h"
#include "kidi/layers/qwen3.h"

namespace kidi::model {

KIDI_MODULE(Qwen3);
class Qwen3Impl : public Module {
public:
    explicit Qwen3Impl(const YAML::Node& config);
    static auto validate_config(const YAML::Node& config) -> Result<void>;
    static auto create(const YAML::Node& config) -> Result<Qwen3>;

    auto embed(ops::Context& context, std::span<const std::int32_t> tokens) const -> tensor::Tensor;
    auto forward(ops::Context& context, const tensor::Tensor& embeddings) const -> tensor::Tensor;
    auto forward(ops::Context& context, const tensor::Tensor& embeddings, const tensor::Tensor& mask) const
        -> tensor::Tensor;
    auto forward_segmented(ops::Context& context, const tensor::Tensor& embeddings,
                           std::span<const std::size_t> segment_lengths) const -> tensor::Tensor;
    auto hidden_size() const noexcept -> std::int32_t { return hidden_; }
    auto vocabulary_size() const noexcept -> std::int32_t { return vocabulary_; }
    auto maximum_position() const noexcept -> std::int32_t { return maximum_position_; }
    auto clear_attention() -> void { attention_.clear(); }

private:
    struct AttentionState {
        tensor::Tensor cosine, sine, mask;
        std::uint64_t used = 0;
    };
    auto prepare_attention(std::size_t length) const -> const AttentionState&;

    std::int32_t vocabulary_, hidden_, layers_, heads_, key_value_heads_, head_width_, maximum_position_;
    float rope_theta_;
    layers::TokenEmbedding tokens_;
    ModuleList<layers::Qwen3BlockImpl> blocks_;
    layers::RmsNorm norm_;
    mutable std::map<std::size_t, AttentionState> attention_;
    mutable std::uint64_t attention_clock_ = 0;
};

} // namespace kidi::model
