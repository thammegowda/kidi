#pragma once

#include "kidi/layers/transformer.h"

namespace kidi::layers {
KIDI_MODULE(RmsNorm);
class RmsNormImpl : public Module {
public:
    RmsNormImpl(std::int32_t width, float epsilon, bool learned = true);
    auto forward(ops::Context& context, const Tensor& input) const -> Tensor;

private:
    Tensor weight_;
    float epsilon_;
};

KIDI_MODULE(TokenEmbedding);
class TokenEmbeddingImpl : public Module {
public:
    TokenEmbeddingImpl(std::int32_t vocabulary, std::int32_t width, float scale);
    auto forward(ops::Context& context, std::span<const std::int32_t> tokens) const -> Tensor;

private:
    Tensor weight_;
    float scale_;
};

KIDI_MODULE(GatedFeedForward);
class GatedFeedForwardImpl : public Module {
public:
    GatedFeedForwardImpl(std::int32_t hidden, std::int32_t intermediate);
    auto forward(ops::Context& context, const Tensor& input) const -> Tensor;

private:
    Linear gate_, up_, down_;
};

KIDI_MODULE(GemmaAttention);
class GemmaAttentionImpl : public Module {
public:
    GemmaAttentionImpl(std::int32_t hidden, std::int32_t heads, std::int32_t key_heads, std::int32_t head_width,
                       float epsilon, bool shared);
    auto forward(ops::Context& context, const Tensor& input, KeyValue& cache, const Tensor& index, const Tensor& mask,
                 const Tensor& cosine, const Tensor& sine) const -> Tensor;

private:
    Linear query_, key_{nullptr}, value_{nullptr}, output_;
    RmsNorm query_norm_, key_norm_{nullptr}, value_norm_{nullptr};
    std::int32_t heads_, key_heads_, head_width_;
};

KIDI_MODULE(GemmaBlock);
class GemmaBlockImpl : public Module {
public:
    GemmaBlockImpl(std::int32_t hidden, std::int32_t intermediate, std::int32_t heads, std::int32_t key_heads,
                   std::int32_t head_width, std::int32_t per_layer_width, float epsilon, bool shared);
    auto forward(ops::Context& context, const Tensor& input, const Tensor& per_layer_input, KeyValue& cache,
                 const Tensor& index, const Tensor& mask, const Tensor& cosine, const Tensor& sine) const -> Tensor;

private:
    GemmaAttention attention_;
    GatedFeedForward feed_forward_;
    RmsNorm input_norm_, attention_norm_, pre_feed_forward_norm_, post_feed_forward_norm_, per_layer_norm_;
    Linear per_layer_gate_, per_layer_projection_;
    Tensor scalar_;
};
} // namespace kidi::layers