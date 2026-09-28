#include "kidi/layers/whisper.h"

#include <cmath>

namespace kidi::layers {
using ops::require;

WhisperConv1dImpl::WhisperConv1dImpl(std::int32_t input_channels, std::int32_t output_channels, std::int32_t stride)
    : input_channels_(input_channels), output_channels_(output_channels), stride_(stride) {
    if (input_channels <= 0 || output_channels <= 0 || (stride != 1 && stride != 2) ||
        (module_dtype != tensor::DType::F32 && module_dtype != tensor::DType::I8))
        throw ops::Failure({ErrorCode::UNSUPPORTED, "unsupported Whisper convolution"});
    register_parameter("weight", weight_, {output_channels, input_channels, 3});
    register_parameter("bias", bias_, {output_channels}, tensor::DType::F32);
    if (module_dtype == tensor::DType::I8)
        register_parameter("scale", scale_, {output_channels, 1}, tensor::DType::F32);
}

auto WhisperConv1dImpl::forward(ops::Context& context, const Tensor& input) const -> Tensor {
    if (context.device() != tensor::Device::cpu() || input.device() != context.device() ||
        input.dtype() != tensor::DType::F32 || input.dimensions() != 3 || input.size(2) != input_channels_)
        throw ops::Failure({ErrorCode::UNSUPPORTED, "Whisper convolution requires FP32 CPU input"});
    const auto batch = input.size(0), length = input.size(1);
    const auto output_length = (length - 1) / static_cast<std::size_t>(stride_) + 1;
    std::vector<float> columns(batch * output_length * input_channels_ * 3);
    const auto values = require(input.data<float>());
    for (std::size_t row = 0; row < batch; ++row)
        for (std::size_t output = 0; output < output_length; ++output)
            for (std::int32_t kernel = 0; kernel < 3; ++kernel) {
                const auto source = static_cast<std::int64_t>(output * stride_ + kernel) - 1;
                if (source < 0 || static_cast<std::size_t>(source) >= length) continue;
                const auto source_offset = (row * length + static_cast<std::size_t>(source)) * input_channels_;
                const auto destination = (row * output_length + output) * input_channels_ * 3;
                for (std::int32_t channel = 0; channel < input_channels_; ++channel)
                    columns[destination + channel * 3 + kernel] = values[source_offset + channel];
            }
    auto unfolded = require(Tensor::from_host(
        {static_cast<std::int64_t>(batch), static_cast<std::int64_t>(output_length), input_channels_ * 3},
        std::span<const float>(columns), context.device()));
    auto weight = context.reshape(weight_, {output_channels_, input_channels_ * 3});
    if (weight_.dtype() == tensor::DType::I8) return context.quantized_linear(unfolded, weight, scale_, bias_, true);
    return context.linear(unfolded, weight, bias_, true);
}

WhisperPositionEmbeddingImpl::WhisperPositionEmbeddingImpl(std::int32_t positions, std::int32_t width) {
    register_parameter("weight", weight_, {positions, width}, tensor::DType::F32);
}

auto WhisperPositionEmbeddingImpl::forward(ops::Context& context, std::size_t start, std::size_t length) const
    -> Tensor {
    if (!length || start > weight_.size(0) || length > weight_.size(0) - start)
        throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "Whisper position is outside the checkpoint table"});
    return context.reshape(context.slice(weight_, 0, start, length),
                           {1, static_cast<std::int64_t>(length), static_cast<std::int64_t>(weight_.size(1))});
}

WhisperAttentionImpl::WhisperAttentionImpl(std::int32_t hidden, std::int32_t heads)
    : key_(hidden, hidden, true, false),
      value_(hidden, hidden, true),
      query_(hidden, hidden, true),
      output_(hidden, hidden, true),
      heads_(heads) {
    if (heads <= 0 || hidden % heads)
        throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "invalid Whisper attention dimensions"});
    register_module("k_proj", key_);
    register_module("v_proj", value_);
    register_module("q_proj", query_);
    register_module("out_proj", output_);
}

auto WhisperAttentionImpl::project_memory(ops::Context& context, const Tensor& input) const -> KeyValue {
    return {key_->forward(context, input), value_->forward(context, input)};
}

auto WhisperAttentionImpl::forward(ops::Context& context, const Tensor& input, const KeyValue* memory,
                                   const Tensor& mask, KeyValue* cache, const Tensor& cache_index) const -> Tensor {
    auto projected = memory ? *memory : project_memory(context, input);
    if (cache) {
        context.scatter_(cache->key, projected.key, cache_index);
        context.scatter_(cache->value, projected.value, cache_index);
        projected = *cache;
    }
    return output_->forward(
        context, context.scaled_dot_product_attention(query_->forward(context, input), projected.key, projected.value,
                                                      heads_, mask));
}

WhisperEncoderBlockImpl::WhisperEncoderBlockImpl(std::int32_t hidden, std::int32_t intermediate, std::int32_t heads,
                                                 float epsilon)
    : self_attention_(hidden, heads),
      self_norm_(hidden, epsilon),
      final_norm_(hidden, epsilon),
      first_(hidden, intermediate, true),
      second_(intermediate, hidden, true) {
    register_module("self_attn", self_attention_);
    register_module("self_attn_layer_norm", self_norm_);
    register_module("fc1", first_);
    register_module("fc2", second_);
    register_module("final_layer_norm", final_norm_);
}

auto WhisperEncoderBlockImpl::forward(ops::Context& context, const Tensor& input) const -> Tensor {
    auto hidden = context.add(input, self_attention_->forward(context, self_norm_->forward(context, input)));
    return context.add(
        hidden,
        second_->forward(context, context.gelu(first_->forward(context, final_norm_->forward(context, hidden)))));
}

WhisperDecoderBlockImpl::WhisperDecoderBlockImpl(std::int32_t hidden, std::int32_t intermediate, std::int32_t heads,
                                                 float epsilon)
    : self_attention_(hidden, heads),
      cross_attention_(hidden, heads),
      self_norm_(hidden, epsilon),
      cross_norm_(hidden, epsilon),
      final_norm_(hidden, epsilon),
      first_(hidden, intermediate, true),
      second_(intermediate, hidden, true) {
    register_module("self_attn", self_attention_);
    register_module("self_attn_layer_norm", self_norm_);
    register_module("encoder_attn", cross_attention_);
    register_module("encoder_attn_layer_norm", cross_norm_);
    register_module("fc1", first_);
    register_module("fc2", second_);
    register_module("final_layer_norm", final_norm_);
}

auto WhisperDecoderBlockImpl::project_source(ops::Context& context, const Tensor& input) const -> KeyValue {
    return cross_attention_->project_memory(context, input);
}

auto WhisperDecoderBlockImpl::forward(ops::Context& context, const Tensor& input, const KeyValue& source,
                                      const Tensor& self_mask, KeyValue& cache, const Tensor& cache_index) const
    -> Tensor {
    auto hidden = context.add(input, self_attention_->forward(context, self_norm_->forward(context, input), nullptr,
                                                              self_mask, &cache, cache_index));
    hidden = context.add(hidden, cross_attention_->forward(context, cross_norm_->forward(context, hidden), &source));
    return context.add(
        hidden,
        second_->forward(context, context.gelu(first_->forward(context, final_norm_->forward(context, hidden)))));
}

WhisperEncoderImpl::WhisperEncoderImpl(std::int32_t mel_bins, std::int32_t hidden, std::int32_t intermediate,
                                       std::int32_t heads, std::int32_t layer_count, std::int32_t positions,
                                       float epsilon)
    : first_conv_([&] {
          const ModuleScope scope(tensor::DType::F32);
          return WhisperConv1d(mel_bins, hidden, 1);
      }()),
      second_conv_(hidden, hidden, 2),
      positions_(positions, hidden),
      norm_(hidden, epsilon) {
    register_module("conv1", first_conv_);
    register_module("conv2", second_conv_);
    register_module("embed_positions", positions_);
    for (std::int32_t layer = 0; layer < layer_count; ++layer)
        layers_->push_back(WhisperEncoderBlock(hidden, intermediate, heads, epsilon));
    register_module("layers", layers_);
    register_module("layer_norm", norm_);
}

auto WhisperEncoderImpl::convolve(ops::Context& context, const Tensor& input) const -> Tensor {
    auto hidden = context.gelu(first_conv_->forward(context, input));
    return context.gelu(second_conv_->forward(context, hidden));
}

auto WhisperEncoderImpl::encode(ops::Context& context, const Tensor& input) const -> Tensor {
    auto hidden = input;
    hidden = context.add(hidden, positions_->forward(context, 0, hidden.size(1)));
    for (const auto& layer : *layers_) hidden = layer->forward(context, hidden);
    return norm_->forward(context, hidden);
}

auto WhisperEncoderImpl::forward(ops::Context& context, const Tensor& input) const -> Tensor {
    return encode(context, convolve(context, input));
}

WhisperDecoderImpl::WhisperDecoderImpl(std::int32_t vocabulary, std::int32_t hidden, std::int32_t intermediate,
                                       std::int32_t heads, std::int32_t layer_count, std::int32_t positions,
                                       float epsilon)
    : tokens_(vocabulary, hidden, 1.F), positions_(positions, hidden), norm_(hidden, epsilon) {
    register_module("embed_tokens", tokens_);
    register_module("embed_positions", positions_);
    for (std::int32_t layer = 0; layer < layer_count; ++layer)
        layers_->push_back(WhisperDecoderBlock(hidden, intermediate, heads, epsilon));
    register_module("layers", layers_);
    register_module("layer_norm", norm_);
}

auto WhisperDecoderImpl::project_source(ops::Context& context, const Tensor& input) const -> std::vector<KeyValue> {
    std::vector<KeyValue> result;
    result.reserve(layers_->size());
    for (const auto& layer : *layers_) result.push_back(layer->project_source(context, input));
    return result;
}

auto WhisperDecoderImpl::forward(ops::Context& context, std::span<const std::int32_t> tokens, std::size_t position,
                                 std::span<const KeyValue> source, const Tensor& self_mask, std::span<KeyValue> cache,
                                 const Tensor& cache_index) const -> Tensor {
    if (tokens.empty() || source.size() != layers_->size() || cache.size() != layers_->size())
        throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "invalid Whisper decoder state"});
    auto hidden = context.add(tokens_->forward(context, tokens), positions_->forward(context, position, tokens.size()));
    for (std::size_t layer = 0; layer < layers_->size(); ++layer)
        hidden = layers_->at(layer)->forward(context, hidden, source[layer], self_mask, cache[layer], cache_index);
    return norm_->forward(context, hidden);
}

} // namespace kidi::layers