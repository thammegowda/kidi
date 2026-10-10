#pragma once

#include "kidi/layers/gemma4.h"

namespace kidi::layers {

struct Qwen3Shape {
    std::int32_t hidden;
    std::int32_t intermediate;
    std::int32_t heads;
    std::int32_t key_value_heads;
    std::int32_t head_width;
    float epsilon;
};

KIDI_MODULE(Qwen3Attention);
class Qwen3AttentionImpl : public Module {
public:
    explicit Qwen3AttentionImpl(Qwen3Shape shape);
    auto forward(ops::Context& context, const Tensor& input, const Tensor& mask, const Tensor& cosine,
                 const Tensor& sine) const -> Tensor;

private:
    Linear qkv_, output_;
    RmsNorm query_norm_, key_norm_;
    std::int32_t heads_, key_value_heads_, head_width_, query_width_, key_value_width_;
};

KIDI_MODULE(Qwen3Block);
class Qwen3BlockImpl : public Module {
public:
    explicit Qwen3BlockImpl(Qwen3Shape shape);
    auto forward(ops::Context& context, const Tensor& input, const Tensor& mask, const Tensor& cosine,
                 const Tensor& sine) const -> Tensor;

private:
    Qwen3Attention attention_;
    GatedFeedForward feed_forward_;
    RmsNorm input_norm_, post_attention_norm_;
};

} // namespace kidi::layers
