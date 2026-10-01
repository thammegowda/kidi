#pragma once

#include "kidi/layers/gemma4.h"
#include "kidi/layers/transformer.h"

namespace kidi::layers {

KIDI_MODULE(WhisperConv1d);
class WhisperConv1dImpl : public Module {
public:
    WhisperConv1dImpl(std::int32_t input_channels, std::int32_t output_channels, std::int32_t stride);
    auto forward(ops::Context& context, const Tensor& input) const -> Tensor;

private:
    Tensor weight_, bias_, scale_;
    mutable Tensor columns_;
    std::int32_t input_channels_, output_channels_, stride_;
};

KIDI_MODULE(WhisperPositionEmbedding);
class WhisperPositionEmbeddingImpl : public Module {
public:
    WhisperPositionEmbeddingImpl(std::int32_t positions, std::int32_t width);
    auto forward(ops::Context& context, std::size_t start, std::size_t length) const -> Tensor;
    /// Position rows selected by an I32 `[length]` index tensor, as `[1, length, width]`.
    auto gather(ops::Context& context, const Tensor& positions) const -> Tensor;

private:
    Tensor weight_;
};

KIDI_MODULE(WhisperAttention);
class WhisperAttentionImpl : public Module {
public:
    WhisperAttentionImpl(std::int32_t hidden, std::int32_t heads);
    auto project_memory(ops::Context& context, const Tensor& input) const -> KeyValue;
    auto forward(ops::Context& context, const Tensor& input, const KeyValue* memory = nullptr, const Tensor& mask = {},
                 KeyValue* cache = nullptr, const Tensor& cache_index = {}) const -> Tensor;

private:
    Linear key_, value_, query_, output_;
    std::int32_t heads_;
};

KIDI_MODULE(WhisperEncoderBlock);
class WhisperEncoderBlockImpl : public Module {
public:
    WhisperEncoderBlockImpl(std::int32_t hidden, std::int32_t intermediate, std::int32_t heads, float epsilon);
    auto forward(ops::Context& context, const Tensor& input) const -> Tensor;

private:
    WhisperAttention self_attention_;
    LayerNorm self_norm_, final_norm_;
    Linear first_, second_;
};

KIDI_MODULE(WhisperDecoderBlock);
class WhisperDecoderBlockImpl : public Module {
public:
    WhisperDecoderBlockImpl(std::int32_t hidden, std::int32_t intermediate, std::int32_t heads, float epsilon);
    auto project_source(ops::Context& context, const Tensor& input) const -> KeyValue;
    auto forward(ops::Context& context, const Tensor& input, const KeyValue& source, const Tensor& self_mask,
                 KeyValue& cache, const Tensor& cache_index) const -> Tensor;

private:
    WhisperAttention self_attention_, cross_attention_;
    LayerNorm self_norm_, cross_norm_, final_norm_;
    Linear first_, second_;
};

KIDI_MODULE(WhisperEncoder);
class WhisperEncoderImpl : public Module {
public:
    WhisperEncoderImpl(std::int32_t mel_bins, std::int32_t hidden, std::int32_t intermediate, std::int32_t heads,
                       std::int32_t layer_count, std::int32_t positions, float epsilon);
    auto convolve(ops::Context& context, const Tensor& input) const -> Tensor;
    auto encode(ops::Context& context, const Tensor& input) const -> Tensor;
    auto forward(ops::Context& context, const Tensor& input) const -> Tensor;

private:
    WhisperConv1d first_conv_, second_conv_;
    WhisperPositionEmbedding positions_;
    ModuleList<WhisperEncoderBlockImpl> layers_;
    LayerNorm norm_;
};

KIDI_MODULE(WhisperDecoder);
class WhisperDecoderImpl : public Module {
public:
    WhisperDecoderImpl(std::int32_t vocabulary, std::int32_t hidden, std::int32_t intermediate, std::int32_t heads,
                       std::int32_t layer_count, std::int32_t positions, float epsilon);
    auto project_source(ops::Context& context, const Tensor& input) const -> std::vector<KeyValue>;
    /// Decodes I32 `tokens` at the positions in `cache_index`, which also selects the self-attention cache rows.
    auto forward(ops::Context& context, const Tensor& tokens, std::span<const KeyValue> source, const Tensor& self_mask,
                 std::span<KeyValue> cache, const Tensor& cache_index) const -> Tensor;

private:
    TokenEmbedding tokens_;
    WhisperPositionEmbedding positions_;
    ModuleList<WhisperDecoderBlockImpl> layers_;
    LayerNorm norm_;
};

} // namespace kidi::layers