#pragma once

#include <span>
#include <vector>

#include "kidi/layers/audio.h"
#include "kidi/layers/gemma4.h"

namespace kidi::layers {

struct HiggsDecoderShape {
    std::int32_t codebooks;
    std::int32_t codebook_size;
    std::int32_t codebook_width;
    std::int32_t latent_width;
    std::int32_t acoustic_width;
    std::int32_t decoder_width;
    std::vector<std::int32_t> upsampling_ratios;
};

KIDI_MODULE(HiggsQuantizer);
class HiggsQuantizerImpl : public Module {
public:
    explicit HiggsQuantizerImpl(HiggsDecoderShape shape);
    auto forward(ops::Context& context, std::span<const std::int32_t> tokens) const -> Tensor;

private:
    TokenEmbedding codebook_;
    Linear project_out_;
};

KIDI_MODULE(DacResidualUnit);
class DacResidualUnitImpl : public Module {
public:
    DacResidualUnitImpl(std::int32_t channels, std::int32_t dilation);
    auto forward(ops::Context& context, const Tensor& input) const -> Tensor;

private:
    Snake1d first_snake_, second_snake_;
    Conv1d first_conv_, second_conv_;
};

KIDI_MODULE(DacDecoderBlock);
class DacDecoderBlockImpl : public Module {
public:
    DacDecoderBlockImpl(std::int32_t input_channels, std::int32_t output_channels, std::int32_t stride);
    auto forward(ops::Context& context, const Tensor& input) const -> Tensor;

private:
    Snake1d snake_;
    ConvTranspose1d upsample_;
    DacResidualUnit first_, second_, third_;
};

KIDI_MODULE(DacDecoder);
class DacDecoderImpl : public Module {
public:
    explicit DacDecoderImpl(HiggsDecoderShape shape);
    auto forward(ops::Context& context, const Tensor& input) const -> Tensor;

private:
    Conv1d input_, output_;
    ModuleList<DacDecoderBlockImpl> blocks_;
    Snake1d snake_;
};

KIDI_MODULE(HiggsDecoder);
class HiggsDecoderImpl : public Module {
public:
    explicit HiggsDecoderImpl(HiggsDecoderShape shape);
    auto forward(ops::Context& context, std::span<const std::int32_t> tokens, std::size_t length) const -> Tensor;

private:
    HiggsDecoderShape shape_;
    ModuleList<HiggsQuantizerImpl> quantizers_;
    Linear acoustic_projection_;
    DacDecoder decoder_;
};

} // namespace kidi::layers
