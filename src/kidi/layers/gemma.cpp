#include "kidi/layers/gemma.h"

#include <algorithm>
#include <bit>

namespace kidi::layers {
using ops::require;

RmsNormImpl::RmsNormImpl(std::int32_t width, float epsilon, bool learned) : epsilon_(epsilon) {
    if (learned)
        register_parameter("weight", weight_, {width}, tensor::DType::F32);
    else
        weight_ = require(Tensor::empty({width}, tensor::DType::F32, device()));
    if (weight_.defined()) std::ranges::fill(require(weight_.data<float>()), 1.F);
}
auto RmsNormImpl::forward(ops::Context& context, const Tensor& input) const -> Tensor {
    return context.rms_norm(input, weight_, epsilon_);
}
TokenEmbeddingImpl::TokenEmbeddingImpl(std::int32_t vocabulary, std::int32_t width, float scale) : scale_(scale) {
    if (module_dtype != tensor::DType::F32 && module_dtype != tensor::DType::BF16)
        throw ops::Failure({ErrorCode::UNSUPPORTED, "token embeddings require FP32 or BF16 weights"});
    register_parameter("weight", weight_, {vocabulary, width});
}
auto TokenEmbeddingImpl::forward(ops::Context& context, std::span<const std::int32_t> tokens) const -> Tensor {
    if (!weight_.defined() || tokens.empty())
        throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "uninitialized embedding or empty tokens"});
    for (auto token : tokens)
        if (token < 0 || static_cast<std::size_t>(token) >= weight_.size(0))
            throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "embedding token outside vocabulary"});
    const auto width = weight_.size(1);
    auto output = require(Tensor::empty({1, static_cast<std::int64_t>(tokens.size()), static_cast<std::int64_t>(width)},
                                        tensor::DType::F32, context.device()));
    const auto bytes = require(weight_.host_bytes());
    auto values = require(output.data<float>());
    for (std::size_t index = 0; index < tokens.size(); ++index)
        for (std::size_t channel = 0; channel < width; ++channel) {
            const auto offset = static_cast<std::size_t>(tokens[index]) * width + channel;
            const auto value =
                weight_.dtype() == tensor::DType::BF16
                    ? std::bit_cast<float>(
                          static_cast<std::uint32_t>(reinterpret_cast<const std::uint16_t*>(bytes.data())[offset])
                          << 16)
                    : reinterpret_cast<const float*>(bytes.data())[offset];
            values[index * width + channel] = value * scale_;
        }
    return output;
}
GatedFeedForwardImpl::GatedFeedForwardImpl(std::int32_t hidden, std::int32_t intermediate)
    : gate_(hidden, intermediate, true, false),
      up_(hidden, intermediate, true, false),
      down_(intermediate, hidden, true, false) {
    register_module("gate_proj", gate_);
    register_module("up_proj", up_);
    register_module("down_proj", down_);
}
auto GatedFeedForwardImpl::forward(ops::Context& context, const Tensor& input) const -> Tensor {
    return down_->forward(
        context, context.multiply(context.gelu(gate_->forward(context, input), true), up_->forward(context, input)));
}
GemmaAttentionImpl::GemmaAttentionImpl(std::int32_t hidden, std::int32_t heads, std::int32_t key_heads,
                                       std::int32_t head_width, float epsilon, bool shared)
    : query_(hidden, heads * head_width, true, false),
      output_(heads * head_width, hidden, true, false),
      query_norm_(head_width, epsilon),
      heads_(heads),
      key_heads_(key_heads),
      head_width_(head_width) {
    register_module("q_proj", query_);
    register_module("o_proj", output_);
    register_module("q_norm", query_norm_);
    if (!shared) {
        key_ = Linear(hidden, key_heads * head_width, true, false);
        value_ = Linear(hidden, key_heads * head_width, true, false);
        key_norm_ = RmsNorm(head_width, epsilon);
        value_norm_ = RmsNorm(head_width, epsilon, false);
        register_module("k_proj", key_);
        register_module("v_proj", value_);
        register_module("k_norm", key_norm_);
    }
}
auto GemmaAttentionImpl::forward(ops::Context& context, const Tensor& input, KeyValue& cache, const Tensor& index,
                                 const Tensor& mask, const Tensor& cosine, const Tensor& sine) const -> Tensor {
    const auto length = static_cast<std::int64_t>(input.size(1));
    const auto reshape = [&](const Tensor& value, std::int32_t heads) {
        return context.reshape(value, {1, length, heads, head_width_});
    };
    auto query =
        context.rotary(query_norm_->forward(context, reshape(query_->forward(context, input), heads_)), cosine, sine);
    if (key_) {
        auto key = context.rotary(key_norm_->forward(context, reshape(key_->forward(context, input), key_heads_)),
                                  cosine, sine);
        auto value = value_norm_->forward(context, reshape(value_->forward(context, input), key_heads_));
        context.scatter_(cache.key, context.reshape(key, {1, length, key_heads_ * head_width_}), index);
        context.scatter_(cache.value, context.reshape(value, {1, length, key_heads_ * head_width_}), index);
    }
    auto hidden = context.grouped_query_attention(context.reshape(query, {1, length, heads_ * head_width_}), cache.key,
                                                  cache.value, heads_, key_heads_, mask);
    return output_->forward(context, hidden);
}
GemmaBlockImpl::GemmaBlockImpl(std::int32_t hidden, std::int32_t intermediate, std::int32_t heads,
                               std::int32_t key_heads, std::int32_t head_width, std::int32_t per_layer_width,
                               float epsilon, bool shared)
    : attention_(hidden, heads, key_heads, head_width, epsilon, shared),
      feed_forward_(hidden, intermediate),
      input_norm_(hidden, epsilon),
      attention_norm_(hidden, epsilon),
      pre_feed_forward_norm_(hidden, epsilon),
      post_feed_forward_norm_(hidden, epsilon),
      per_layer_norm_(hidden, epsilon),
      per_layer_gate_(hidden, per_layer_width, true, false),
      per_layer_projection_(per_layer_width, hidden, true, false) {
    register_module("self_attn", attention_);
    register_module("mlp", feed_forward_);
    register_module("input_layernorm", input_norm_);
    register_module("post_attention_layernorm", attention_norm_);
    register_module("pre_feedforward_layernorm", pre_feed_forward_norm_);
    register_module("post_feedforward_layernorm", post_feed_forward_norm_);
    register_module("post_per_layer_input_norm", per_layer_norm_);
    register_module("per_layer_input_gate", per_layer_gate_);
    register_module("per_layer_projection", per_layer_projection_);
    register_parameter("layer_scalar", scalar_, {1}, tensor::DType::F32);
    if (scalar_.defined()) require(scalar_.data<float>())[0] = 1.F;
}
auto GemmaBlockImpl::forward(ops::Context& context, const Tensor& input, const Tensor& per_layer_input, KeyValue& cache,
                             const Tensor& index, const Tensor& mask, const Tensor& cosine, const Tensor& sine) const
    -> Tensor {
    auto hidden = context.add(
        input, attention_norm_->forward(context, attention_->forward(context, input_norm_->forward(context, input),
                                                                     cache, index, mask, cosine, sine)));
    hidden = context.add(
        hidden, post_feed_forward_norm_->forward(
                    context, feed_forward_->forward(context, pre_feed_forward_norm_->forward(context, hidden))));
    auto gate = context.gelu(per_layer_gate_->forward(context, hidden), true);
    auto projected = per_layer_projection_->forward(context, context.multiply(gate, per_layer_input));
    hidden = context.add(hidden, per_layer_norm_->forward(context, projected));
    return context.multiply(hidden, scalar_);
}
} // namespace kidi::layers