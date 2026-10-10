#include "kidi/layers/qwen3.h"

#include <cmath>

namespace kidi::layers {
using ops::require;
auto reshape_heads(ops::Context& context, const Tensor& input, std::int32_t heads, std::int32_t width) -> Tensor {
    return context.reshape(
        input, {static_cast<std::int64_t>(input.size(0)), static_cast<std::int64_t>(input.size(1)), heads, width});
}

Qwen3AttentionImpl::Qwen3AttentionImpl(Qwen3Shape shape)
    : qkv_(shape.hidden, (shape.heads + 2 * shape.key_value_heads) * shape.head_width, true, false),
      output_(shape.heads * shape.head_width, shape.hidden, true, false),
      query_norm_(shape.head_width, shape.epsilon, true, true),
      key_norm_(shape.head_width, shape.epsilon, true, true),
      heads_(shape.heads),
      key_value_heads_(shape.key_value_heads),
      head_width_(shape.head_width),
      query_width_(shape.heads * shape.head_width),
      key_value_width_(shape.key_value_heads * shape.head_width) {
    if (shape.hidden <= 0 || shape.heads <= 0 || shape.key_value_heads <= 0 || shape.heads % shape.key_value_heads ||
        shape.head_width <= 0)
        throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "invalid Qwen3 attention dimensions"});
    register_module("qkv_proj", qkv_);
    register_module("o_proj", output_);
    register_module("q_norm", query_norm_);
    register_module("k_norm", key_norm_);
}

auto Qwen3AttentionImpl::forward(ops::Context& context, const Tensor& input, const Tensor& mask, const Tensor& cosine,
                                 const Tensor& sine) const -> Tensor {
    const auto projected = qkv_->forward(context, input);
    auto query = context.slice(projected, -1, 0, query_width_);
    auto key = context.slice(projected, -1, query_width_, key_value_width_);
    auto value = context.slice(projected, -1, query_width_ + key_value_width_, key_value_width_);
    query = query_norm_->forward(context, reshape_heads(context, query, heads_, head_width_));
    key = key_norm_->forward(context, reshape_heads(context, key, key_value_heads_, head_width_));
    query = context.rotary(query, cosine, sine);
    key = context.rotary(key, cosine, sine);
    query = context.reshape(query, {static_cast<std::int64_t>(input.size(0)), static_cast<std::int64_t>(input.size(1)),
                                    heads_ * head_width_});
    key = context.reshape(key, {static_cast<std::int64_t>(input.size(0)), static_cast<std::int64_t>(input.size(1)),
                                key_value_heads_ * head_width_});
    auto attended = context.grouped_query_attention(query, key, value, heads_, key_value_heads_, mask,
                                                    1.F / std::sqrt(static_cast<float>(head_width_)));
    return output_->forward(context, attended);
}

Qwen3BlockImpl::Qwen3BlockImpl(Qwen3Shape shape)
    : attention_(shape),
      feed_forward_(shape.hidden, shape.intermediate, 0, GatedActivation::SILU),
      input_norm_(shape.hidden, shape.epsilon, true, true),
      post_attention_norm_(shape.hidden, shape.epsilon, true, true) {
    register_module("self_attn", attention_);
    register_module("mlp", feed_forward_);
    register_module("input_layernorm", input_norm_);
    register_module("post_attention_layernorm", post_attention_norm_);
}

auto Qwen3BlockImpl::forward(ops::Context& context, const Tensor& input, const Tensor& mask, const Tensor& cosine,
                             const Tensor& sine) const -> Tensor {
    auto hidden =
        context.add(input, attention_->forward(context, input_norm_->forward(context, input), mask, cosine, sine));
    return context.add(hidden, feed_forward_->forward(context, post_attention_norm_->forward(context, hidden)));
}

} // namespace kidi::layers
