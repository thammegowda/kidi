#include "kidi/layers/transformer.h"
#include "kidi/layers/position_encoding.h"
#include <bit>
#include <cmath>
#include <algorithm>

namespace kidi::layers {
using ops::require;
namespace {
auto check_precision(tensor::DType precision) -> void {
    if (precision != tensor::DType::F32 && precision != tensor::DType::BF16 && precision != tensor::DType::I8)
        throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "unsupported parameter precision"});
}
auto split_qkv(ops::Context& context, const Tensor& projected) -> std::array<Tensor, 3> {
    auto width = static_cast<std::int64_t>(projected.size(-1) / 3);
    return {context.slice(projected, 2, 0, width), context.slice(projected, 2, width, width),
            context.slice(projected, 2, 2 * width, width)};
}
} // namespace
LinearImpl::LinearImpl(std::int32_t input_size, std::int32_t output_size, bool transpose, bool bias,
                       std::int32_t packed_bits)
    : packed_bits_(packed_bits), input_size_(input_size), transpose_(transpose), has_bias_(bias) {
    if (packed_bits) {
        if ((packed_bits != 2 && packed_bits != 4 && packed_bits != 8) || !transpose || bias ||
            input_size % (8 / packed_bits))
            throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "invalid packed linear construction"});
        register_parameter("weight", weight_, {output_size, input_size / (8 / packed_bits)}, tensor::DType::U8);
        register_parameter("weight_scale", scale_, {output_size, 1}, tensor::DType::F32);
        register_parameter("input_activation_scale", input_scale_, {}, tensor::DType::F32);
        register_parameter("output_activation_scale", output_scale_, {}, tensor::DType::F32);
        return;
    }
    check_precision(module_dtype);
    if (module_dtype == tensor::DType::I8 && transpose)
        throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "INT8 linear requires input-output weight layout"});
    register_parameter("weight", weight_,
                       transpose ? std::vector<std::int64_t>{output_size, input_size}
                                 : std::vector<std::int64_t>{input_size, output_size});
    if (bias)
        register_parameter("bias", bias_, {output_size}, tensor::DType::F32);
    else if (module_dtype == tensor::DType::I8)
        bias_ = require(Tensor::zeros({output_size}, tensor::DType::F32, device()));
    if (module_dtype == tensor::DType::I8) {
        register_parameter("scale", scale_, {output_size, 1}, tensor::DType::F32);
        if (allocate_parameters) std::ranges::fill(require(scale_.data<float>()), 1.F);
    }
}
auto LinearImpl::forward(ops::Context& context, const Tensor& input) const -> Tensor {
    if (!weight_.defined() || (has_bias_ && !bias_.defined()))
        throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "linear state has not been initialized"});
    if (packed_bits_) {
        return context.packed_linear(input, weight_, scale_, packed_bits_, input_size_,
                                     require(input_scale_.data<float>())[0], require(output_scale_.data<float>())[0]);
    }
    if (weight_.dtype() == tensor::DType::I8) return context.quantized_linear(input, weight_, scale_, bias_);
    return context.linear(input, weight_, bias_, transpose_);
}
LayerNormImpl::LayerNormImpl(std::int32_t hidden_size, float epsilon) : epsilon_(epsilon) {
    if (!std::isfinite(epsilon) || epsilon <= 0)
        throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "invalid normalization epsilon"});
    register_parameter("weight", scale_, {hidden_size}, tensor::DType::F32);
    register_parameter("bias", bias_, {hidden_size}, tensor::DType::F32);
    if (allocate_parameters) std::ranges::fill(require(scale_.data<float>()), 1.F);
}
auto LayerNormImpl::forward(ops::Context& context, const Tensor& input) const -> Tensor {
    return context.layer_norm(input, scale_, bias_, epsilon_);
}
auto LayerNormImpl::forward_residual(ops::Context& context, const Tensor& input, const Tensor& residual) const
    -> std::array<Tensor, 2> {
    return context.residual_layer_norm(input, residual, scale_, bias_, epsilon_);
}
EmbeddingImpl::EmbeddingImpl(std::int32_t vocabulary_size, std::int32_t hidden_size, std::int32_t maximum_position)
    : hidden_size_(hidden_size), maximum_position_(maximum_position) {
    check_precision(module_dtype);
    if (hidden_size <= 0 || hidden_size % 2 || maximum_position <= 0)
        throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "invalid embedding parameters"});
    register_parameter("weight", weight_, {vocabulary_size, hidden_size});
    if (module_dtype == tensor::DType::I8) {
        register_parameter("scale", scale_, {vocabulary_size, 1}, tensor::DType::F32);
        if (allocate_parameters) std::ranges::fill(require(scale_.data<float>()), 1.F);
    }
    if (allocate_parameters)
        positions_ = require(sinusoidal_position_encoding(maximum_position_, hidden_size_, device()));
}
auto EmbeddingImpl::forward(ops::Context& context, std::span<const std::int32_t> tokens, std::size_t batch,
                            std::size_t position) const -> Tensor {
    if (!weight_.defined() || !batch || tokens.empty() || tokens.size() % batch ||
        position > static_cast<std::size_t>(maximum_position_) ||
        tokens.size() / batch > static_cast<std::size_t>(maximum_position_) - position)
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
    if (weight_.defined() && weight_.dtype() == tensor::DType::I8 && !scale_.defined())
        throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "embedding scales have not been initialized"});
    if (!weight_.defined() || !batch || tokens.empty() || tokens.size() % batch ||
        position > static_cast<std::size_t>(maximum_position_) ||
        tokens.size() / batch > static_cast<std::size_t>(maximum_position_) - position)
        throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "invalid embedding token/position dimensions"});
    const auto length = tokens.size() / batch, width = weight_.size(1);
    if (!output.defined() || output.dtype() != tensor::DType::F32 || output.device() != context.device() ||
        output.dimensions() != 3 || output.size(0) != batch || output.size(1) != length || output.size(2) != width)
        throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "embedding output shape or dtype mismatch"});
    auto values = require(output.data<float>());
    if (!positions_.defined())
        positions_ = require(sinusoidal_position_encoding(maximum_position_, hidden_size_, device()));
    auto positions = require(positions_.data<float>());
    auto bytes = require(weight_.host_bytes());
    const auto dtype = weight_.dtype();
    const auto scales = scale_.defined() ? require(scale_.data<float>()) : std::span<const float>{};
    const float scale = std::sqrt(static_cast<float>(width));
    for (std::size_t index = 0; index < tokens.size(); ++index) {
        if (tokens[index] < 0 || static_cast<std::size_t>(tokens[index]) >= weight_.size(0))
            throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "embedding token outside vocabulary"});
        const auto offset = tokens[index] * width;
        for (std::size_t channel = 0; channel < width; ++channel) {
            float value;
            if (dtype == tensor::DType::I8)
                value = reinterpret_cast<const std::int8_t*>(bytes.data())[offset + channel] * scales[tokens[index]];
            else if (dtype == tensor::DType::BF16)
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
AttentionImpl::AttentionImpl(Shape shape) : output_(shape.hidden, shape.hidden), heads_(shape.heads) {
    if (heads_ <= 0 || shape.hidden % heads_)
        throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "invalid attention dimensions"});
    register_module("output", output_);
}
auto AttentionImpl::forward(ops::Context& context, const Tensor& query, const KeyValue& memory,
                            const Tensor& mask) const -> Tensor {
    return output_->forward(context,
                            context.scaled_dot_product_attention(query, memory.key, memory.value, heads_, mask));
}
FeedForwardImpl::FeedForwardImpl(std::int32_t hidden_size, std::int32_t feed_forward_size)
    : first_(hidden_size, feed_forward_size), second_(feed_forward_size, hidden_size) {
    register_module("first", first_);
    register_module("second", second_);
}
auto FeedForwardImpl::forward(ops::Context& context, const Tensor& input) const -> Tensor {
    return second_->forward(context, context.gelu(first_->forward(context, input)));
}
EncoderBlockImpl::EncoderBlockImpl(Shape shape)
    : qkv_(shape.hidden, 3 * shape.hidden),
      attention_(shape),
      attention_norm_(shape.hidden, shape.epsilon),
      feed_forward_norm_(shape.hidden, shape.epsilon),
      feed_forward_(shape.hidden, shape.feed_forward) {
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
DecoderBlockImpl::DecoderBlockImpl(Shape shape)
    : qkv_(shape.hidden, 3 * shape.hidden),
      query_(shape.hidden, shape.hidden),
      source_(shape.hidden, 2 * shape.hidden),
      self_attention_(shape),
      cross_attention_(shape),
      self_norm_(shape.hidden, shape.epsilon),
      cross_norm_(shape.hidden, shape.epsilon),
      feed_forward_norm_(shape.hidden, shape.epsilon),
      feed_forward_(shape.hidden, shape.feed_forward) {
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