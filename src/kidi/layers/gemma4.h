#pragma once

#include "kidi/layers/transformer.h"

namespace kidi::layers {
KIDI_MODULE(RmsNorm);
class RmsNormImpl : public Module {
public:
    RmsNormImpl(std::int32_t width, float epsilon, bool learned = true);
    auto forward(ops::Context& context, const Tensor& input) const -> Tensor;
    auto forward_rotary(ops::Context& context, const Tensor& input, const Tensor& cosine, const Tensor& sine) const
        -> Tensor;

    auto forward_residual(ops::Context& context, const Tensor& input, const Tensor& residual,
                          const Tensor& output_scale = {}) const -> Tensor;

private:
    Tensor weight_;
    float epsilon_;
};

KIDI_MODULE(TokenEmbedding);
class TokenEmbeddingImpl : public Module {
public:
    TokenEmbeddingImpl(std::int32_t vocabulary, std::int32_t width, float scale, std::int32_t packed_bits = 0,
                       std::int32_t scale_groups = 1);
    auto forward(ops::Context& context, std::span<const std::int32_t> tokens) const -> Tensor;

private:
    Tensor weight_;
    Tensor quantization_scale_;
    std::int32_t width_, packed_bits_;
    float scale_;
};

KIDI_MODULE(GatedFeedForward);
class GatedFeedForwardImpl : public Module {
public:
    GatedFeedForwardImpl(std::int32_t hidden, std::int32_t intermediate, std::int32_t packed_bits = 0);
    auto forward(ops::Context& context, const Tensor& input) const -> Tensor;

private:
    Linear gate_, up_, down_;
};

KIDI_MODULE(Gemma4Attention);
struct Gemma4AttentionSegment {
    KeyValue* cache;
    std::size_t position, length;
    const Tensor* mask;
    std::int64_t key_start = 0;
};
class Gemma4AttentionImpl : public Module {
public:
    Gemma4AttentionImpl(std::int32_t hidden, std::int32_t heads, std::int32_t key_heads, std::int32_t head_width,
                        float epsilon, bool shared, std::int32_t packed_bits = 0);
    auto forward(ops::Context& context, const Tensor& input, KeyValue& cache, std::size_t position, const Tensor& mask,
                 const Tensor& cosine, const Tensor& sine, std::int64_t key_start = 0) const -> Tensor;
    auto forward_segments(ops::Context& context, const Tensor& input, std::span<const Gemma4AttentionSegment> segments,
                          const Tensor& cosine, const Tensor& sine, bool cache_only = false) const -> Tensor;

private:
    Linear query_, key_{nullptr}, value_{nullptr}, output_;
    RmsNorm query_norm_, key_norm_{nullptr}, value_norm_{nullptr};
    Tensor key_scale_, value_scale_;
    std::int32_t heads_, key_heads_, head_width_;
};

KIDI_MODULE(Gemma4Block);
class Gemma4BlockImpl : public Module {
public:
    Gemma4BlockImpl(std::int32_t hidden, std::int32_t intermediate, std::int32_t heads, std::int32_t key_heads,
                    std::int32_t head_width, std::int32_t per_layer_width, float epsilon, bool shared,
                    std::int32_t mlp_bits = 0, std::int32_t attention_bits = 0, std::int32_t per_layer_bits = 0);
    auto forward(ops::Context& context, const Tensor& input, const Tensor& per_layer_input, KeyValue& cache,
                 std::size_t position, const Tensor& mask, const Tensor& cosine, const Tensor& sine,
                 std::int64_t key_start = 0) const -> Tensor;
    auto forward_segments(ops::Context& context, const Tensor& input, const Tensor& per_layer_input,
                          std::span<const Gemma4AttentionSegment> segments, const Tensor& cosine, const Tensor& sine,
                          bool cache_only = false) const -> Tensor;

private:
    Gemma4Attention attention_;
    GatedFeedForward feed_forward_;
    RmsNorm input_norm_, attention_norm_, pre_feed_forward_norm_, post_feed_forward_norm_, per_layer_norm_;
    Linear per_layer_gate_, per_layer_projection_;
    Tensor scalar_;
};
} // namespace kidi::layers