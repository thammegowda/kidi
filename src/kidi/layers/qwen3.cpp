#include "kidi/layers/qwen3.h"

#include <cmath>
#include <vector>

namespace kidi::layers {
using ops::require;
auto reshape_heads(ops::Context& context, const Tensor& input, std::int32_t heads, std::int32_t width) -> Tensor {
    return context.reshape(
        input, {static_cast<std::int64_t>(input.size(0)), static_cast<std::int64_t>(input.size(1)), heads, width});
}

Qwen3AttentionImpl::Qwen3AttentionImpl(Qwen3Shape shape)
    : qkv_(shape.hidden, (shape.heads + 2 * shape.key_value_heads) * shape.head_width, true, false, shape.packed_bits),
      output_(shape.heads * shape.head_width, shape.hidden, true, false, shape.packed_bits),
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
    auto value = context.slice(projected, -1, query_width_ + key_value_width_, key_value_width_);
    Tensor query, key;
    if (context.device() == tensor::Device::cpu()) {
        query = query_norm_->forward_rotary_slice(context, projected, cosine, sine, heads_, head_width_, 0, 0,
                                                  input.size(1));
        key = key_norm_->forward_rotary_slice(context, projected, cosine, sine, key_value_heads_, head_width_,
                                              query_width_, 0, input.size(1));
    } else {
        query = context.slice(projected, -1, 0, query_width_);
        key = context.slice(projected, -1, query_width_, key_value_width_);
        query = query_norm_->forward_rotary(context, reshape_heads(context, query, heads_, head_width_), cosine, sine);
        key = key_norm_->forward_rotary(context, reshape_heads(context, key, key_value_heads_, head_width_), cosine,
                                        sine);
    }
    query = context.reshape(query, {static_cast<std::int64_t>(input.size(0)), static_cast<std::int64_t>(input.size(1)),
                                    heads_ * head_width_});
    key = context.reshape(key, {static_cast<std::int64_t>(input.size(0)), static_cast<std::int64_t>(input.size(1)),
                                key_value_heads_ * head_width_});
    auto attended = context.grouped_query_attention(query, key, value, heads_, key_value_heads_, mask,
                                                    1.F / std::sqrt(static_cast<float>(head_width_)));
    return output_->forward(context, attended);
}

auto Qwen3AttentionImpl::forward(ops::Context& context, const Tensor& input,
                                 std::span<const Qwen3AttentionSegment> segments) const -> Tensor {
    const auto projected = qkv_->forward(context, input);
    const auto value = context.slice(projected, -1, query_width_ + key_value_width_, key_value_width_);
    Tensor query, key;
    if (context.device() != tensor::Device::cpu()) {
        query = context.slice(projected, -1, 0, query_width_);
        key = context.slice(projected, -1, query_width_, key_value_width_);
    }
    std::vector<Tensor> attended;
    attended.reserve(segments.size());
    for (const auto& segment : segments) {
        const auto segment_value = context.slice(value, 1, segment.offset, segment.length);
        Tensor segment_query, segment_key;
        if (context.device() == tensor::Device::cpu()) {
            segment_query = query_norm_->forward_rotary_slice(context, projected, *segment.cosine, *segment.sine,
                                                              heads_, head_width_, 0, segment.offset, segment.length);
            segment_key =
                key_norm_->forward_rotary_slice(context, projected, *segment.cosine, *segment.sine, key_value_heads_,
                                                head_width_, query_width_, segment.offset, segment.length);
        } else {
            segment_query = context.slice(query, 1, segment.offset, segment.length);
            segment_key = context.slice(key, 1, segment.offset, segment.length);
            segment_query = query_norm_->forward_rotary(
                context, reshape_heads(context, segment_query, heads_, head_width_), *segment.cosine, *segment.sine);
            segment_key =
                key_norm_->forward_rotary(context, reshape_heads(context, segment_key, key_value_heads_, head_width_),
                                          *segment.cosine, *segment.sine);
        }
        segment_query =
            context.reshape(segment_query, {static_cast<std::int64_t>(input.size(0)),
                                            static_cast<std::int64_t>(segment.length), heads_ * head_width_});
        segment_key =
            context.reshape(segment_key, {static_cast<std::int64_t>(input.size(0)),
                                          static_cast<std::int64_t>(segment.length), key_value_heads_ * head_width_});
        attended.push_back(context.grouped_query_attention(segment_query, segment_key, segment_value, heads_,
                                                           key_value_heads_, *segment.mask,
                                                           1.F / std::sqrt(static_cast<float>(head_width_))));
    }
    return output_->forward(context, context.concat(attended, 1));
}

Qwen3BlockImpl::Qwen3BlockImpl(Qwen3Shape shape)
    : attention_(shape),
      feed_forward_(shape.hidden, shape.intermediate, shape.packed_bits, GatedActivation::SILU),
      input_norm_(shape.hidden, shape.epsilon, true, true),
      post_attention_norm_(shape.hidden, shape.epsilon, true, true) {
    register_module("self_attn", attention_);
    register_module("mlp", feed_forward_);
    register_module("input_layernorm", input_norm_);
    register_module("post_attention_layernorm", post_attention_norm_);
}

auto Qwen3BlockImpl::forward(ops::Context& context, Tensor input, const Tensor& mask, const Tensor& cosine,
                             const Tensor& sine, bool consume_input) const -> Tensor {
    const auto attended = attention_->forward(context, input_norm_->forward(context, input), mask, cosine, sine);
    Tensor hidden;
    if (consume_input) {
        context.add_(input, attended);
        hidden = std::move(input);
    } else {
        hidden = context.add(input, attended);
    }
    return context.add_(hidden, feed_forward_->forward(context, post_attention_norm_->forward(context, hidden)));
}

auto Qwen3BlockImpl::forward(ops::Context& context, Tensor input, std::span<const Qwen3AttentionSegment> segments,
                             bool consume_input) const -> Tensor {
    const auto attended = attention_->forward(context, input_norm_->forward(context, input), segments);
    Tensor hidden;
    if (consume_input) {
        context.add_(input, attended);
        hidden = std::move(input);
    } else {
        hidden = context.add(input, attended);
    }
    return context.add_(hidden, feed_forward_->forward(context, post_attention_norm_->forward(context, hidden)));
}

} // namespace kidi::layers
