#include "kidi/layers/whisper.h"

#include <algorithm>
#include <cmath>

namespace kidi::layers {
using ops::require;

namespace {
// Encoder-length inputs run in fixed row slices: each weighted operator is prepared for one slice shape, so its retained
// buffers stay small and are shared by every audio length. Inputs of one slice (decoder steps) run whole.
constexpr std::int64_t ENCODER_ROWS = 128;
template <typename Function>
auto by_rows(ops::Context& context, const Tensor& input, Function&& function) -> Tensor {
    const auto length = static_cast<std::int64_t>(input.size(1));
    if (length <= ENCODER_ROWS) return function(input, 0);
    Tensor output;
    for (std::int64_t start = 0; start < length; start += ENCODER_ROWS) {
        const auto part = function(context.slice(input, 1, start, std::min(ENCODER_ROWS, length - start)), start);
        if (!output.defined())
            output = require(Tensor::empty({1, length, static_cast<std::int64_t>(part.size(2))}, part.dtype(),
                                            context.device()));
        context.copy_slice_(output, part, 1, start);
    }
    return output;
}
} // namespace

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
    if (input.size(0) != 1) throw ops::Failure({ErrorCode::UNSUPPORTED, "Whisper convolution takes one sequence"});
    const auto length = static_cast<std::int64_t>(input.size(1));
    const auto output_length = (length - 1) / stride_ + 1;
    const auto values = require(input.data<float>());
    const auto weight = context.reshape(weight_, {output_channels_, input_channels_ * 3});
    // Output rows are unfolded and projected one slice at a time, like the encoder blocks.
    const auto slice = std::min(ENCODER_ROWS, output_length);
    const auto required = static_cast<std::size_t>(slice * input_channels_ * 3);
    if (!columns_.defined() || columns_.numel() < required)
        columns_ = require(Tensor::empty({static_cast<std::int64_t>(required)}, tensor::DType::F32, context.device()));
    Tensor result;
    for (std::int64_t first = 0; first < output_length; first += slice) {
        const auto count = std::min(slice, output_length - first);
        // The previous slice's projection must finish reading the shared buffer before it is refilled.
        context.synchronize();
        auto columns = require(columns_.data<float>());
        for (std::int64_t output = 0; output < count; ++output)
            for (std::int32_t kernel = 0; kernel < 3; ++kernel) {
                const auto source = (first + output) * stride_ + kernel - 1;
                const auto destination = static_cast<std::size_t>(output * input_channels_ * 3);
                for (std::int32_t channel = 0; channel < input_channels_; ++channel)
                    columns[destination + channel * 3 + kernel] =
                        source < 0 || source >= length ? 0.F : values[source * input_channels_ + channel];
            }
        const auto unfolded = context.reshape(
            context.slice(columns_, 0, 0, count * input_channels_ * 3), {1, count, input_channels_ * 3});
        const auto part = weight_.dtype() == tensor::DType::I8
                              ? context.quantized_linear(unfolded, weight, scale_, bias_, true)
                              : context.linear(unfolded, weight, bias_, true);
        if (count == output_length) return part;
        if (!result.defined())
            result = require(Tensor::empty({1, output_length, output_channels_}, tensor::DType::F32, context.device()));
        context.copy_slice_(result, part, 1, first);
    }
    return result;
}

WhisperPositionEmbeddingImpl::WhisperPositionEmbeddingImpl(std::int32_t positions, std::int32_t width) {
    register_parameter("weight", weight_, {positions, width}, tensor::DType::F32);
}

auto WhisperPositionEmbeddingImpl::forward(ops::Context& context, std::size_t start,
                                           std::size_t length) const -> Tensor {
    if (!length || start > weight_.size(0) || length > weight_.size(0) - start)
        throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "Whisper position is outside the checkpoint table"});
    return context.reshape(context.slice(weight_, 0, start, length),
                           {1, static_cast<std::int64_t>(length), static_cast<std::int64_t>(weight_.size(1))});
}

auto WhisperPositionEmbeddingImpl::gather(ops::Context& context, const Tensor& positions) const -> Tensor {
    if (positions.dtype() != tensor::DType::I32 || positions.dimensions() != 1 || !positions.size(0))
        throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "Whisper positions must be a nonempty I32 vector"});
    return context.reshape(context.gather(weight_, positions), {1, static_cast<std::int64_t>(positions.size(0)),
                                                                static_cast<std::int64_t>(weight_.size(1))});
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
    return {by_rows(context, input, [&](const Tensor& rows, std::int64_t) { return key_->forward(context, rows); }),
            by_rows(context, input, [&](const Tensor& rows, std::int64_t) { return value_->forward(context, rows); })};
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
    const auto normalized =
        by_rows(context, input, [&](const Tensor& rows, std::int64_t) { return self_norm_->forward(context, rows); });
    const auto memory = self_attention_->project_memory(context, normalized);
    return by_rows(context, input, [&](const Tensor& rows, std::int64_t start) {
        const auto queries = context.slice(normalized, 1, start, static_cast<std::int64_t>(rows.size(1)));
        auto hidden = context.add(rows, self_attention_->forward(context, queries, &memory));
        return context.add(
            hidden,
            second_->forward(context, context.gelu(first_->forward(context, final_norm_->forward(context, hidden)))));
    });
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
                                      const Tensor& self_mask, KeyValue& cache,
                                      const Tensor& cache_index) const -> Tensor {
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
    return by_rows(context, hidden, [&](const Tensor& rows, std::int64_t) { return norm_->forward(context, rows); });
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

auto WhisperDecoderImpl::forward(ops::Context& context, const Tensor& tokens, std::span<const KeyValue> source,
                                 const Tensor& self_mask, std::span<KeyValue> cache,
                                 const Tensor& cache_index) const -> Tensor {
    if (source.size() != layers_->size() || cache.size() != layers_->size())
        throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "invalid Whisper decoder state"});
    auto hidden = context.add(tokens_->forward(context, tokens), positions_->gather(context, cache_index));
    for (std::size_t layer = 0; layer < layers_->size(); ++layer)
        hidden = layers_->at(layer)->forward(context, hidden, source[layer], self_mask, cache[layer], cache_index);
    return norm_->forward(context, hidden);
}

} // namespace kidi::layers