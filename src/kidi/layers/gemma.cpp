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
auto RmsNormImpl::forward_residual(ops::Context& context, const Tensor& input, const Tensor& residual,
                                   const Tensor& output_scale) const -> Tensor {
    return context.rms_norm_residual(input, weight_, residual, epsilon_, output_scale);
}
TokenEmbeddingImpl::TokenEmbeddingImpl(std::int32_t vocabulary, std::int32_t width, float scale,
                                       std::int32_t packed_bits, std::int32_t scale_groups)
    : width_(width), packed_bits_(packed_bits), scale_(scale) {
    if (packed_bits) {
        if ((packed_bits != 2 && packed_bits != 4 && packed_bits != 8) || width % (8 / packed_bits) ||
            scale_groups <= 0 || width % scale_groups)
            throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "invalid packed embedding dimensions"});
        register_parameter("embedding_quantized", weight_, {vocabulary, width / (8 / packed_bits)}, tensor::DType::U8);
        register_parameter("embedding_scale", quantization_scale_, {vocabulary, scale_groups}, tensor::DType::F32);
        return;
    }
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
    const auto width = static_cast<std::size_t>(width_);
    auto output = require(Tensor::empty({1, static_cast<std::int64_t>(tokens.size()), static_cast<std::int64_t>(width)},
                                        tensor::DType::F32, context.device()));
    const auto bytes = require(weight_.host_bytes());
    auto values = require(output.data<float>());
    if (packed_bits_) {
        const auto scales = require(quantization_scale_.data<float>());
        const auto groups = quantization_scale_.size(1), group_width = width / groups;
        const auto data = reinterpret_cast<const std::uint8_t*>(bytes.data());
        for (std::size_t index = 0; index < tokens.size(); ++index)
            for (std::size_t channel = 0; channel < width; ++channel) {
                const auto offset = static_cast<std::size_t>(tokens[index]) * width + channel;
                const auto raw = (data[offset / (8 / packed_bits_)] >> ((offset % (8 / packed_bits_)) * packed_bits_)) &
                                 ((1 << packed_bits_) - 1);
                const auto integer = (raw ^ (1 << (packed_bits_ - 1))) - (1 << (packed_bits_ - 1));
                values[index * width + channel] =
                    integer * scales[tokens[index] * groups + channel / group_width] * scale_;
            }
        return output;
    }
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
GatedFeedForwardImpl::GatedFeedForwardImpl(std::int32_t hidden, std::int32_t intermediate, std::int32_t packed_bits)
    : gate_(hidden, intermediate, true, false, packed_bits),
      up_(hidden, intermediate, true, false, packed_bits),
      down_(intermediate, hidden, true, false, packed_bits) {
    register_module("gate_proj", gate_);
    register_module("up_proj", up_);
    register_module("down_proj", down_);
}
auto GatedFeedForwardImpl::forward(ops::Context& context, const Tensor& input) const -> Tensor {
    return down_->forward(
        context, context.multiply(context.gelu(gate_->forward(context, input), true), up_->forward(context, input)));
}
GemmaAttentionImpl::GemmaAttentionImpl(std::int32_t hidden, std::int32_t heads, std::int32_t key_heads,
                                       std::int32_t head_width, float epsilon, bool shared, std::int32_t packed_bits)
    : query_(hidden, heads * head_width, true, false, packed_bits),
      output_(heads * head_width, hidden, true, false, packed_bits),
      query_norm_(head_width, epsilon),
      heads_(heads),
      key_heads_(key_heads),
      head_width_(head_width) {
    register_module("q_proj", query_);
    register_module("o_proj", output_);
    register_module("q_norm", query_norm_);
    if (!shared) {
        key_ = Linear(hidden, key_heads * head_width, true, false, packed_bits);
        value_ = Linear(hidden, key_heads * head_width, true, false, packed_bits);
        key_norm_ = RmsNorm(head_width, epsilon);
        value_norm_ = RmsNorm(head_width, epsilon, false);
        register_module("k_proj", key_);
        register_module("v_proj", value_);
        register_module("k_norm", key_norm_);
        if (packed_bits) {
            register_parameter("k_cache_scale", key_scale_, {}, tensor::DType::F32);
            register_parameter("v_cache_scale", value_scale_, {}, tensor::DType::F32);
        }
    }
}
auto GemmaAttentionImpl::forward(ops::Context& context, const Tensor& input, KeyValue& cache, const Tensor& index,
                                 const Tensor& mask, const Tensor& cosine, const Tensor& sine,
                                 std::int64_t key_start) const -> Tensor {
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
        if (key_scale_.defined()) {
            key = context.static_round(key, require(key_scale_.data<float>())[0]);
            value = context.static_round(value, require(value_scale_.data<float>())[0]);
        }
        context.scatter_(cache.key, context.reshape(key, {1, length, key_heads_ * head_width_}), index);
        context.scatter_(cache.value, context.reshape(value, {1, length, key_heads_ * head_width_}), index);
    }
    const auto extent = static_cast<std::int64_t>(mask.size(-1));
    auto hidden = context.grouped_query_attention(
        context.reshape(query, {1, length, heads_ * head_width_}), context.slice(cache.key, 1, 0, extent + key_start),
        context.slice(cache.value, 1, 0, extent + key_start), heads_, key_heads_, mask, 1.F, key_start);
    return output_->forward(context, hidden);
}
GemmaBlockImpl::GemmaBlockImpl(std::int32_t hidden, std::int32_t intermediate, std::int32_t heads,
                               std::int32_t key_heads, std::int32_t head_width, std::int32_t per_layer_width,
                               float epsilon, bool shared, std::int32_t mlp_bits, std::int32_t attention_bits,
                               std::int32_t per_layer_bits)
    : attention_(hidden, heads, key_heads, head_width, epsilon, shared, attention_bits),
      feed_forward_(hidden, intermediate, mlp_bits),
      input_norm_(hidden, epsilon),
      attention_norm_(hidden, epsilon),
      pre_feed_forward_norm_(hidden, epsilon),
      post_feed_forward_norm_(hidden, epsilon),
      per_layer_norm_(hidden, epsilon),
      per_layer_gate_(hidden, per_layer_width, true, false, per_layer_bits),
      per_layer_projection_(per_layer_width, hidden, true, false, per_layer_bits) {
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
                             const Tensor& index, const Tensor& mask, const Tensor& cosine, const Tensor& sine,
                             std::int64_t key_start) const -> Tensor {
    auto hidden = attention_norm_->forward_residual(
        context,
        attention_->forward(context, input_norm_->forward(context, input), cache, index, mask, cosine, sine, key_start),
        input);
    hidden = post_feed_forward_norm_->forward_residual(
        context, feed_forward_->forward(context, pre_feed_forward_norm_->forward(context, hidden)), hidden);
    auto gate = context.gelu(per_layer_gate_->forward(context, hidden), true);
    auto projected = per_layer_projection_->forward(context, context.multiply(gate, per_layer_input));
    return per_layer_norm_->forward_residual(context, projected, hidden, scalar_);
}
} // namespace kidi::layers