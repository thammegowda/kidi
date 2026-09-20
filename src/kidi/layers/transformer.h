#pragma once

#include <array>
#include <string_view>

#include "kidi/core/module.h"
#include "kidi/model/weights.h"
#include "kidi/ops/context.h"

namespace kidi::layers {

using tensor::Tensor;

/// Transformer widths, attention head count, and layer-normalization epsilon.
struct Shape {
    std::int32_t hidden, feed_forward, heads;
    float epsilon;
};

/// Projected keys and values, each shaped [batch, sequence, hidden].
struct KeyValue {
    Tensor key, value;
};

KIDI_MODULE(Linear);

/// Affine projection over the final dimension with bound weights and bias.
/// Supports FP32, BF16, and per-channel INT8 weight encodings.
class LinearImpl : public Module {
public:
    LinearImpl(const model::Weights&, std::string_view prefix, model::WeightEncoding);
    LinearImpl(const model::Weights&, std::string_view weight, std::string_view bias, model::WeightEncoding,
               bool transpose = false);
    auto forward(ops::Context&, const Tensor&) const -> Tensor;

private:
    Tensor weight_, bias_, scale_;
    model::WeightEncoding encoding_;
    bool transpose_;
};

KIDI_MODULE(LayerNorm);

/// Normalizes the final dimension, then applies learned scale and bias.
class LayerNormImpl : public Module {
public:
    LayerNormImpl(const model::Weights&, std::string_view prefix, float epsilon);
    auto forward(ops::Context&, const Tensor&) const -> Tensor;

    /// Returns the residual sum and its normalized value without separate dispatches.
    auto forward_residual(ops::Context&, const Tensor& input, const Tensor& residual) const -> std::array<Tensor, 2>;

private:
    Tensor scale_, bias_;
    float epsilon_;
};

KIDI_MODULE(Embedding);

/// Token lookup scaled by sqrt(hidden), with sinusoidal positions added.
/// Maps row-major token IDs to FP32 tensors shaped [batch, sequence, hidden].
class EmbeddingImpl : public Module {
public:
    EmbeddingImpl(const model::Weights&, std::string_view weight, model::WeightEncoding, std::int32_t maximum_position);
    auto forward(ops::Context&, std::span<const std::int32_t> tokens, std::size_t batch, std::size_t position = 0) const
        -> Tensor;
    auto forward_(ops::Context&, Tensor& output, std::span<const std::int32_t> tokens, std::size_t batch,
                  std::size_t position = 0) const -> Tensor&;

private:
    Tensor weight_, scale_, positions_;
    model::WeightEncoding encoding_;
};

KIDI_MODULE(Attention);

/// Multi-head scaled dot-product attention over projected queries, keys, and values.
/// Applies an optional additive mask and a learned output projection.
class AttentionImpl : public Module {
public:
    AttentionImpl(const model::Weights&, std::string_view prefix, model::WeightEncoding, Shape);
    auto forward(ops::Context&, const Tensor& query, const KeyValue&, const Tensor& mask) const -> Tensor;

private:
    Linear output_;
    std::int32_t heads_;
};

KIDI_MODULE(FeedForward);

/// Positionwise hidden-to-intermediate-to-hidden projections with GELU between them.
/// Normalization and residual connections belong to the enclosing block.
class FeedForwardImpl : public Module {
public:
    FeedForwardImpl(const model::Weights&, std::string_view prefix, model::WeightEncoding);
    auto forward(ops::Context&, const Tensor&) const -> Tensor;

private:
    Linear first_, second_;
};

KIDI_MODULE(EncoderBlock);

/// Pre-normalized self-attention and feed-forward block with residual connections.
class EncoderBlockImpl : public Module {
public:
    EncoderBlockImpl(const model::Weights&, std::string_view prefix, model::WeightEncoding, Shape);
    auto forward(ops::Context&, const Tensor&, const Tensor& mask) const -> Tensor;

private:
    Linear qkv_;
    Attention attention_;
    LayerNorm attention_norm_, feed_forward_norm_;
    FeedForward feed_forward_;
};

KIDI_MODULE(DecoderBlock);

/// Pre-normalized self-attention, encoder cross-attention, and feed-forward block.
/// Supports full prefixes or incremental decoding with explicit key/value caches.
class DecoderBlockImpl : public Module {
public:
    DecoderBlockImpl(const model::Weights&, std::string_view prefix, model::WeightEncoding, Shape);

    /// Precomputes encoder keys and values for reuse across decoder steps.
    auto project_source(ops::Context&, const Tensor&) const -> KeyValue;

    /// When cache is supplied, updates its storage in place at cache_index; aliases observe the writes.
    auto forward(ops::Context&, const Tensor&, const KeyValue& source, const Tensor& source_mask,
                 const Tensor& self_mask, KeyValue* cache = nullptr, const Tensor& cache_index = {}) const -> Tensor;

private:
    Linear qkv_, query_, source_;
    Attention self_attention_, cross_attention_;
    LayerNorm self_norm_, cross_norm_, feed_forward_norm_;
    FeedForward feed_forward_;
};

} // namespace kidi::layers