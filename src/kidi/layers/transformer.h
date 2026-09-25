#pragma once

#include <array>

#include "kidi/core/module.h"
#include "kidi/ops/context.h"

namespace kidi::layers {

using tensor::Tensor;

/// Transformer widths, attention head count, and layer-normalization epsilon.
struct Shape {
    std::int32_t hidden, feed_forward, heads;
    float epsilon;
};

/// Projected keys and values, each shaped [batch, sequence, hidden].
/// An INT8 cache carries the scales its rows were rounded with, so layers sharing it need no scales of their own.
struct KeyValue {
    Tensor key, value;
    float key_scale = 0, value_scale = 0;
};

KIDI_MODULE(Linear);

/// Affine projection over the final dimension with bound weights and optional bias.
/// Supports FP32, BF16, and per-channel INT8 weight encodings.
class LinearImpl : public Module {
public:
    LinearImpl(std::int32_t input_size, std::int32_t output_size, bool transpose = false, bool bias = true,
               std::int32_t packed_bits = 0);
    auto forward(ops::Context&, const Tensor&) const -> Tensor;

private:
    Tensor weight_, bias_, scale_;
    Tensor input_scale_, output_scale_;
    std::int32_t packed_bits_ = 0, input_size_ = 0;
    bool transpose_, has_bias_;
};

KIDI_MODULE(LayerNorm);

/// Normalizes the final dimension, then applies learned scale and bias.
class LayerNormImpl : public Module {
public:
    LayerNormImpl(std::int32_t hidden_size, float epsilon);
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
    EmbeddingImpl(std::int32_t vocabulary_size, std::int32_t hidden_size, std::int32_t maximum_position);
    auto forward(ops::Context&, std::span<const std::int32_t> tokens, std::size_t batch, std::size_t position = 0) const
        -> Tensor;
    auto forward_(ops::Context&, Tensor& output, std::span<const std::int32_t> tokens, std::size_t batch,
                  std::size_t position = 0) const -> Tensor&;

private:
    Tensor weight_, scale_;
    mutable Tensor positions_;
    std::int32_t hidden_size_, maximum_position_;
};

KIDI_MODULE(Attention);

/// Multi-head scaled dot-product attention over projected queries, keys, and values.
/// Applies an optional additive mask and a learned output projection.
class AttentionImpl : public Module {
public:
    explicit AttentionImpl(Shape);
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
    FeedForwardImpl(std::int32_t hidden_size, std::int32_t feed_forward_size);
    auto forward(ops::Context&, const Tensor&) const -> Tensor;

private:
    Linear first_, second_;
};

KIDI_MODULE(EncoderBlock);

/// Pre-normalized self-attention and feed-forward block with residual connections.
class EncoderBlockImpl : public Module {
public:
    explicit EncoderBlockImpl(Shape);
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
    explicit DecoderBlockImpl(Shape);

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