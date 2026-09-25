#include "kidi/layers/gemma4.h"

#include <algorithm>
#include <bit>

namespace kidi::layers {
using ops::require;

RmsNormImpl::RmsNormImpl(std::int32_t width, float epsilon, bool learned) : epsilon_(epsilon) {
    if (learned)
        register_parameter("weight", weight_, {width}, tensor::DType::F32);
    else
        weight_ = require(Tensor::empty({width}, tensor::DType::F32, device()));
    if (weight_.defined()) {
        const std::vector<float> ones(width, 1.F);
        weight_ = require(Tensor::from_host({width}, std::span<const float>(ones), device()));
    }
}
auto RmsNormImpl::forward(ops::Context& context, const Tensor& input) const -> Tensor {
    return context.rms_norm(input, weight_, epsilon_);
}
auto RmsNormImpl::forward_rotary(ops::Context& context, const Tensor& input, const Tensor& cosine,
                                 const Tensor& sine) const -> Tensor {
    return context.rms_rotary(input, weight_, cosine, sine, epsilon_);
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
    if (context.device() == tensor::Device::web_gpu()) {
        const auto indices =
            require(Tensor::from_host({static_cast<std::int64_t>(tokens.size())}, tokens, context.device()));
        return context.embedding(indices, weight_, quantization_scale_, width_, packed_bits_, scale_);
    }
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
    return down_->forward(context, context.gelu_multiply(gate_->forward(context, input), up_->forward(context, input)));
}
Gemma4AttentionImpl::Gemma4AttentionImpl(std::int32_t hidden, std::int32_t heads, std::int32_t key_heads,
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
            register_parameter("k_cache_scale", key_scale_, {}, tensor::DType::F32, allocate_parameters,
                               tensor::Device::cpu());
            register_parameter("v_cache_scale", value_scale_, {}, tensor::DType::F32, allocate_parameters,
                               tensor::Device::cpu());
        }
    }
}
auto Gemma4AttentionImpl::forward(ops::Context& context, const Tensor& input, KeyValue& cache, std::size_t position,
                                  const Tensor& mask, const Tensor& cosine, const Tensor& sine,
                                  std::int64_t key_start) const -> Tensor {
    const std::array segments{Gemma4AttentionSegment{&cache, position, input.size(1), &mask, key_start}};
    return forward_segments(context, input, segments, cosine, sine);
}
auto Gemma4AttentionImpl::has_cache_scales() const -> bool {
    if (!key_ || !key_scale_.defined() || !value_scale_.defined()) return false;
    const auto key_scale = key_scale_.data<float>(), value_scale = value_scale_.data<float>();
    return key_scale && value_scale && (*key_scale)[0] > 0 && (*value_scale)[0] > 0;
}
auto Gemma4AttentionImpl::forward_segments(ops::Context& context, const Tensor& input,
                                           std::span<const Gemma4AttentionSegment> segments, const Tensor& cosine,
                                           const Tensor& sine, bool cache_only) const -> Tensor {
    const auto length = static_cast<std::int64_t>(input.size(1));
    std::size_t total = 0;
    for (const auto& segment : segments) {
        if (!segment.cache || !segment.mask || !segment.length || segment.length > input.size(1) - total ||
            segment.position > segment.cache->key.size(1) ||
            segment.length > segment.cache->key.size(1) - segment.position)
            throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "invalid packed attention segment"});
        total += segment.length;
    }
    if (segments.empty() || total != input.size(1))
        throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "attention segments do not cover input"});
    const auto reshape = [&](const Tensor& value, std::int32_t heads) {
        return context.reshape(value, {1, length, heads, head_width_});
    };
    Tensor query;
    if (!cache_only)
        query = query_norm_->forward_rotary(context, reshape(query_->forward(context, input), heads_), cosine, sine);
    Tensor key, value;
    float key_scale = 0, value_scale = 0;
    // A byte cache rounds and clamps inside its own cast, so the separate rounding pass would be redundant.
    const auto byte_cache = segments.front().cache->key.dtype() == tensor::DType::I8;
    if (key_) {
        key = key_norm_->forward_rotary(context, reshape(key_->forward(context, input), key_heads_), cosine, sine);
        value = value_norm_->forward(context, reshape(value_->forward(context, input), key_heads_));
        if (key_scale_.defined()) {
            key_scale = require(key_scale_.data<float>())[0];
            value_scale = require(value_scale_.data<float>())[0];
            if (!byte_cache) {
                key = context.static_round(key, key_scale);
                value = context.static_round(value, value_scale);
            }
        }
        key = context.reshape(key, {1, length, key_heads_ * head_width_});
        value = context.reshape(value, {1, length, key_heads_ * head_width_});
    }
    const auto prefix = [&](const Tensor& memory, std::int64_t extent) {
        if (memory.dimensions() != 3 || memory.size(0) != 1)
            throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "Gemma 4 cache must contain one sequence"});
        auto rows = require(memory.select(0, 0));
        auto view = require(rows.narrow(0, 0, extent));
        return context.reshape(view, {1, extent, static_cast<std::int64_t>(memory.size(2))});
    };
    if (!cache_only) query = context.reshape(query, {1, length, heads_ * head_width_});
    std::vector<Tensor> outputs;
    Tensor single_output;
    if (segments.size() > 1) outputs.reserve(segments.size());
    std::size_t offset = 0;
    for (const auto& segment : segments) {
        auto& cache = *segment.cache;
        if (key_) {
            auto keys = context.slice(key, 1, offset, segment.length);
            auto values = context.slice(value, 1, offset, segment.length);
            if (byte_cache) {
                keys = context.cast(keys, tensor::DType::I8, key_scale);
                values = context.cast(values, tensor::DType::I8, value_scale);
                cache.key_scale = key_scale;
                cache.value_scale = value_scale;
            }
            context.copy_slice_(cache.key, keys, 1, segment.position);
            context.copy_slice_(cache.value, values, 1, segment.position);
        }
        if (cache_only) {
            offset += segment.length;
            continue;
        }
        const auto extent = static_cast<std::int64_t>(segment.mask->size(-1)) + segment.key_start;
        auto attended = context.grouped_query_attention(
            context.slice(query, 1, offset, segment.length), prefix(cache.key, extent), prefix(cache.value, extent),
            heads_, key_heads_, *segment.mask, 1.F, segment.key_start, cache.key_scale, cache.value_scale);
        if (segments.size() == 1)
            single_output = std::move(attended);
        else
            outputs.push_back(std::move(attended));
        offset += segment.length;
    }
    if (cache_only) return {};
    auto hidden = segments.size() == 1 ? single_output : context.concat(outputs, 1);
    return output_->forward(context, hidden);
}
Gemma4BlockImpl::Gemma4BlockImpl(std::int32_t hidden, std::int32_t intermediate, std::int32_t heads,
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
    if (scalar_.defined()) scalar_ = require(Tensor::from_host({1}, std::span<const float>(std::array{1.F}), device()));
}
auto Gemma4BlockImpl::forward(ops::Context& context, const Tensor& input, const Tensor& per_layer_input,
                              KeyValue& cache, std::size_t position, const Tensor& mask, const Tensor& cosine,
                              const Tensor& sine, std::int64_t key_start) const -> Tensor {
    const std::array segments{Gemma4AttentionSegment{&cache, position, input.size(1), &mask, key_start}};
    return forward_segments(context, input, per_layer_input, segments, cosine, sine);
}
auto Gemma4BlockImpl::forward_segments(ops::Context& context, const Tensor& input, const Tensor& per_layer_input,
                                       std::span<const Gemma4AttentionSegment> segments, const Tensor& cosine,
                                       const Tensor& sine, bool cache_only) const -> Tensor {
    auto attended =
        attention_->forward_segments(context, input_norm_->forward(context, input), segments, cosine, sine, cache_only);
    if (cache_only) return {};
    auto hidden = attention_norm_->forward_residual(context, attended, input);
    hidden = post_feed_forward_norm_->forward_residual(
        context, feed_forward_->forward(context, pre_feed_forward_norm_->forward(context, hidden)), hidden);
    auto gate = per_layer_gate_->forward(context, hidden);
    auto projected = per_layer_projection_->forward(context, context.gelu_multiply(gate, per_layer_input));
    return per_layer_norm_->forward_residual(context, projected, hidden, scalar_);
}
} // namespace kidi::layers