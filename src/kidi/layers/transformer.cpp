#include "kidi/layers/transformer.h"
#include "kidi/layers/position_encoding.h"
#include <bit>
#include <cmath>
#include <string>

namespace kidi::layers {
using ops::require;
namespace {
auto name(std::string_view prefix, std::string_view suffix) -> std::string {
    return std::string(prefix) + std::string(suffix);
}
auto split_qkv(ops::Context& context, const Tensor& projected) -> std::array<Tensor, 3> {
    auto width = static_cast<std::int64_t>(projected.size(-1) / 3);
    return {context.slice(projected, 2, 0, width), context.slice(projected, 2, width, width),
            context.slice(projected, 2, 2 * width, width)};
}
} // namespace
LinearImpl::LinearImpl(const model::Weights& weights, std::string_view prefix, model::WeightEncoding encoding)
    : LinearImpl(weights, name(prefix, ".weight"), name(prefix, ".bias"), encoding) {}
LinearImpl::LinearImpl(const model::Weights& weights, std::string_view weight, std::string_view bias,
                       model::WeightEncoding encoding, bool transpose)
    : weight_(require(weights.tensor(weight))),
      bias_(require(weights.tensor(bias))),
      encoding_(encoding),
      transpose_(transpose) {
    if (encoding == model::WeightEncoding::INT8_PER_CHANNEL)
        scale_ = require(weights.tensor(model::quantization_scale_name(weight)));
    register_parameter("weight", weight_);
    register_parameter("bias", bias_);
    if (scale_.defined()) register_parameter("scale", scale_);
}
auto LinearImpl::forward(ops::Context& context, const Tensor& input) const -> Tensor {
    if (encoding_ == model::WeightEncoding::INT8_PER_CHANNEL)
        return context.quantized_linear(input, weight_, scale_, bias_);
    return context.linear(encoding_ == model::WeightEncoding::BF16 ? context.cast(input, tensor::DType::BF16) : input,
                          weight_, bias_, transpose_);
}
LayerNormImpl::LayerNormImpl(const model::Weights& weights, std::string_view prefix, float epsilon)
    : scale_(require(weights.tensor(name(prefix, ".weight")))),
      bias_(require(weights.tensor(name(prefix, ".bias")))),
      epsilon_(epsilon) {
    register_parameter("weight", scale_);
    register_parameter("bias", bias_);
}
auto LayerNormImpl::forward(ops::Context& context, const Tensor& input) const -> Tensor {
    return context.layer_norm(input, scale_, bias_, epsilon_);
}
auto LayerNormImpl::forward_residual(ops::Context& context, const Tensor& input, const Tensor& residual) const
    -> std::array<Tensor, 2> {
    return context.residual_layer_norm(input, residual, scale_, bias_, epsilon_);
}
EmbeddingImpl::EmbeddingImpl(const model::Weights& weights, std::string_view weight, model::WeightEncoding encoding,
                             std::int32_t maximum_position)
    : weight_(require(weights.tensor(weight))), encoding_(encoding) {
    if (weight_.dimensions() != 2 || weight_.dtype() != model::matrix_data_type(encoding) || maximum_position <= 0)
        throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "invalid embedding parameters"});
    if (encoding == model::WeightEncoding::INT8_PER_CHANNEL) {
        scale_ = require(weights.tensor(model::quantization_scale_name(weight)));
        if (scale_.dtype() != tensor::DType::F32 || scale_.dimensions() != 2 || scale_.size(0) != weight_.size(0) ||
            scale_.size(1) != 1)
            throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "invalid embedding scales"});
    }
    positions_ = require(sinusoidal_position_encoding(maximum_position, weight_.size(1)));
    register_parameter("weight", weight_);
    if (scale_.defined()) register_parameter("scale", scale_);
}
auto EmbeddingImpl::forward(ops::Context& context, std::span<const std::int32_t> tokens, std::size_t batch,
                            std::size_t position) const -> Tensor {
    if (!batch || tokens.empty() || tokens.size() % batch || position + tokens.size() / batch > positions_.size(0))
        throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "invalid embedding token/position dimensions"});
    const auto length = tokens.size() / batch, width = weight_.size(1);
    auto output = require(Tensor::empty(
        {static_cast<std::int64_t>(batch), static_cast<std::int64_t>(length), static_cast<std::int64_t>(width)},
        tensor::DType::F32, context.device()));
    forward_(context, output, tokens, batch, position);
    return output;
}
auto EmbeddingImpl::forward_(ops::Context& context, Tensor& output, std::span<const std::int32_t> tokens,
                             std::size_t batch, std::size_t position) const -> Tensor& {
    if (!batch || tokens.empty() || tokens.size() % batch || position > positions_.size(0) ||
        tokens.size() / batch > positions_.size(0) - position)
        throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "invalid embedding token/position dimensions"});
    const auto length = tokens.size() / batch, width = weight_.size(1);
    if (!output.defined() || output.dtype() != tensor::DType::F32 || output.device() != context.device() ||
        output.dimensions() != 3 || output.size(0) != batch || output.size(1) != length || output.size(2) != width)
        throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "embedding output shape or dtype mismatch"});
    auto values = require(output.data<float>());
    auto positions = require(positions_.data<float>());
    auto bytes = require(weight_.host_bytes());
    const auto scales = scale_.defined() ? require(scale_.data<float>()) : std::span<const float>{};
    const float scale = std::sqrt(static_cast<float>(width));
    for (std::size_t index = 0; index < tokens.size(); ++index) {
        if (tokens[index] < 0 || static_cast<std::size_t>(tokens[index]) >= weight_.size(0))
            throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "embedding token outside vocabulary"});
        const auto offset = tokens[index] * width;
        for (std::size_t channel = 0; channel < width; ++channel) {
            float value;
            if (encoding_ == model::WeightEncoding::INT8_PER_CHANNEL)
                value = reinterpret_cast<const std::int8_t*>(bytes.data())[offset + channel] * scales[tokens[index]];
            else if (encoding_ == model::WeightEncoding::BF16)
                value = std::bit_cast<float>(
                    static_cast<std::uint32_t>(reinterpret_cast<const std::uint16_t*>(bytes.data())[offset + channel])
                    << 16);
            else
                value = reinterpret_cast<const float*>(bytes.data())[offset + channel];
            values[index * width + channel] = value * scale + positions[(position + index % length) * width + channel];
        }
    }
    return output;
}
AttentionImpl::AttentionImpl(const model::Weights& weights, std::string_view prefix, model::WeightEncoding encoding,
                             Shape shape)
    : output_(std::make_shared<LinearImpl>(weights, name(prefix, ".out"), encoding)), heads_(shape.heads) {
    register_module("output", output_);
}
auto AttentionImpl::forward(ops::Context& context, const Tensor& query, const KeyValue& memory,
                            const Tensor& mask) const -> Tensor {
    return output_->forward(context,
                            context.scaled_dot_product_attention(query, memory.key, memory.value, heads_, mask));
}
FeedForwardImpl::FeedForwardImpl(const model::Weights& weights, std::string_view prefix, model::WeightEncoding encoding)
    : first_(std::make_shared<LinearImpl>(weights, name(prefix, ".w_1"), encoding)),
      second_(std::make_shared<LinearImpl>(weights, name(prefix, ".w_2"), encoding)) {
    register_module("first", first_);
    register_module("second", second_);
}
auto FeedForwardImpl::forward(ops::Context& context, const Tensor& input) const -> Tensor {
    return second_->forward(context, context.gelu(first_->forward(context, input)));
}
EncoderBlockImpl::EncoderBlockImpl(const model::Weights& weights, std::string_view prefix,
                                   model::WeightEncoding encoding, Shape shape)
    : qkv_(std::make_shared<LinearImpl>(weights, name(prefix, ".self_attn.qkv"), encoding)),
      attention_(std::make_shared<AttentionImpl>(weights, name(prefix, ".self_attn"), encoding, shape)),
      attention_norm_(std::make_shared<LayerNormImpl>(weights, name(prefix, ".sublayer.0.norm"), shape.epsilon)),
      feed_forward_norm_(std::make_shared<LayerNormImpl>(weights, name(prefix, ".sublayer.1.norm"), shape.epsilon)),
      feed_forward_(std::make_shared<FeedForwardImpl>(weights, name(prefix, ".feed_forward"), encoding)) {
    register_module("qkv", qkv_);
    register_module("attention", attention_);
    register_module("attention_norm", attention_norm_);
    register_module("feed_forward_norm", feed_forward_norm_);
    register_module("feed_forward", feed_forward_);
}
auto EncoderBlockImpl::forward(ops::Context& context, const Tensor& input, const Tensor& mask) const -> Tensor {
    const auto projected = split_qkv(context, qkv_->forward(context, attention_norm_->forward(context, input)));
    auto [hidden, normalized] = feed_forward_norm_->forward_residual(
        context, input, attention_->forward(context, projected[0], {projected[1], projected[2]}, mask));
    return context.add(hidden, feed_forward_->forward(context, normalized));
}
DecoderBlockImpl::DecoderBlockImpl(const model::Weights& weights, std::string_view prefix,
                                   model::WeightEncoding encoding, Shape shape)
    : qkv_(std::make_shared<LinearImpl>(weights, name(prefix, ".self_attn.qkv"), encoding)),
      query_(std::make_shared<LinearImpl>(weights, name(prefix, ".src_attn.q"), encoding)),
      source_(std::make_shared<LinearImpl>(weights, name(prefix, ".src_attn.kv"), encoding)),
      self_attention_(std::make_shared<AttentionImpl>(weights, name(prefix, ".self_attn"), encoding, shape)),
      cross_attention_(std::make_shared<AttentionImpl>(weights, name(prefix, ".src_attn"), encoding, shape)),
      self_norm_(std::make_shared<LayerNormImpl>(weights, name(prefix, ".sublayer.0.norm"), shape.epsilon)),
      cross_norm_(std::make_shared<LayerNormImpl>(weights, name(prefix, ".sublayer.1.norm"), shape.epsilon)),
      feed_forward_norm_(std::make_shared<LayerNormImpl>(weights, name(prefix, ".sublayer.2.norm"), shape.epsilon)),
      feed_forward_(std::make_shared<FeedForwardImpl>(weights, name(prefix, ".feed_forward"), encoding)) {
    register_module("qkv", qkv_);
    register_module("query", query_);
    register_module("source", source_);
    register_module("self_attention", self_attention_);
    register_module("cross_attention", cross_attention_);
    register_module("self_norm", self_norm_);
    register_module("cross_norm", cross_norm_);
    register_module("feed_forward_norm", feed_forward_norm_);
    register_module("feed_forward", feed_forward_);
}
auto DecoderBlockImpl::project_source(ops::Context& context, const Tensor& input) const -> KeyValue {
    auto projected = source_->forward(context, input);
    auto width = static_cast<std::int64_t>(input.size(2));
    return {context.slice(projected, 2, 0, width), context.slice(projected, 2, width, width)};
}
auto DecoderBlockImpl::forward(ops::Context& context, const Tensor& input, const KeyValue& source,
                               const Tensor& source_mask, const Tensor& self_mask, KeyValue* cache,
                               const Tensor& cache_index) const -> Tensor {
    auto projected = split_qkv(context, qkv_->forward(context, self_norm_->forward(context, input)));
    KeyValue memory{projected[1], projected[2]};
    if (cache) {
        context.scatter_(cache->key, memory.key, cache_index);
        context.scatter_(cache->value, memory.value, cache_index);
        memory = *cache;
    }
    auto [self_residual, normalized_self] = cross_norm_->forward_residual(
        context, input, self_attention_->forward(context, projected[0], memory, self_mask));
    auto [hidden, normalized] = feed_forward_norm_->forward_residual(
        context, self_residual,
        cross_attention_->forward(context, query_->forward(context, normalized_self), source, source_mask));
    return context.add(hidden, feed_forward_->forward(context, normalized));
}
} // namespace kidi::layers