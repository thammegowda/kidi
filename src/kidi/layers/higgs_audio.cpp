#include "kidi/layers/higgs_audio.h"

#include <numeric>

namespace kidi::layers {

HiggsQuantizerImpl::HiggsQuantizerImpl(HiggsDecoderShape shape)
    : codebook_(shape.codebook_size, shape.codebook_width, 1.F),
      project_out_(shape.codebook_width, shape.latent_width, true, true, 0, true) {
    register_module("codebook", codebook_);
    register_module("project_out", project_out_);
}

auto HiggsQuantizerImpl::forward(ops::Context& context, const Tensor& tokens) const -> Tensor {
    return project_out_->forward(context, codebook_->forward(context, tokens));
}

DacResidualUnitImpl::DacResidualUnitImpl(std::int32_t channels, std::int32_t dilation)
    : first_snake_(channels),
      second_snake_(channels),
      first_conv_(channels, channels, 7, 1, 3 * dilation, dilation),
      second_conv_(channels, channels, 1, 1, 0) {
    register_module("snake1", first_snake_);
    register_module("conv1", first_conv_);
    register_module("snake2", second_snake_);
    register_module("conv2", second_conv_);
}

auto DacResidualUnitImpl::forward(ops::Context& context, Tensor input) const -> Tensor {
    auto output = first_snake_->forward(context, input, first_conv_);
    output = second_snake_->forward(context, output, second_conv_);
    if (!std::ranges::equal(input.shape(), output.shape()))
        throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "DAC residual unit changed the sequence shape"});
    return context.add_(input, output);
}

DacDecoderBlockImpl::DacDecoderBlockImpl(std::int32_t input_channels, std::int32_t output_channels, std::int32_t stride)
    : snake_(input_channels),
      upsample_(input_channels, output_channels, 2 * stride, stride, (stride + 1) / 2, stride % 2),
      first_(output_channels, 1),
      second_(output_channels, 3),
      third_(output_channels, 9) {
    register_module("snake1", snake_);
    register_module("conv_t1", upsample_);
    register_module("res_unit1", first_);
    register_module("res_unit2", second_);
    register_module("res_unit3", third_);
}

auto DacDecoderBlockImpl::forward(ops::Context& context, const Tensor& input) const -> Tensor {
    auto hidden = snake_->forward(context, input, upsample_);
    hidden = first_->forward(context, hidden);
    hidden = second_->forward(context, hidden);
    return third_->forward(context, hidden);
}

DacDecoderImpl::DacDecoderImpl(HiggsDecoderShape shape)
    : input_(shape.acoustic_width, shape.decoder_width, 7, 1, 3),
      output_(shape.decoder_width >> shape.upsampling_ratios.size(), 1, 7, 1, 3),
      snake_(shape.decoder_width >> shape.upsampling_ratios.size()) {
    if (shape.upsampling_ratios.empty() || shape.upsampling_ratios.size() >= 31 ||
        shape.decoder_width % (1 << shape.upsampling_ratios.size()))
        throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "invalid DAC decoder channel schedule"});
    register_module("conv1", input_);
    auto channels = shape.decoder_width;
    for (const auto stride : shape.upsampling_ratios) {
        const auto next = channels / 2;
        auto block = DacDecoderBlock(channels, next, stride);
        blocks_->push_back(block);
        channels = next;
    }
    register_module("block", blocks_);
    register_module("snake1", snake_);
    register_module("conv2", output_);
}

auto DacDecoderImpl::forward(ops::Context& context, const Tensor& input) const -> Tensor {
    if (context.device().policy().replay.region_layers) {
        auto hidden = input;
        std::size_t index = 0;
        for (const auto& block : *blocks_) {
            const std::array operands{hidden};
            hidden = context
                         .replay(
                             "dac:block:" + std::to_string(input.size(1)) + ':' + std::to_string(index), operands,
                             [&](std::span<const Tensor> values) {
                                 auto value = values[0];
                                 if (!index) value = input_->forward(context, value);
                                 return std::vector{block->forward(context, value)};
                             },
                             false)
                         .front();
            ++index;
        }
        const std::array operands{hidden};
        return context
            .replay(
                "dac:output:" + std::to_string(input.size(1)), operands,
                [&](std::span<const Tensor> values) {
                    return std::vector{snake_->forward(context, values[0], output_)};
                },
                false)
            .front();
    }
    auto hidden = input_->forward(context, input);
    for (const auto& block : *blocks_) hidden = block->forward(context, hidden);
    return snake_->forward(context, hidden, output_);
}

HiggsDecoderImpl::HiggsDecoderImpl(HiggsDecoderShape shape)
    : shape_(std::move(shape)),
      acoustic_projection_(shape_.latent_width, shape_.acoustic_width, true, true, 0, true),
      decoder_(shape_) {
    if (shape_.codebooks <= 0 || shape_.codebook_size <= 0 || shape_.codebook_width <= 0 || shape_.latent_width <= 0 ||
        shape_.acoustic_width <= 0 || shape_.decoder_width <= 0)
        throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "invalid Higgs decoder dimensions"});
    for (std::int32_t index = 0; index < shape_.codebooks; ++index) quantizers_->push_back(HiggsQuantizer(shape_));
    register_module("quantizers", quantizers_);
    register_module("fc2", acoustic_projection_);
    register_module("acoustic_decoder", decoder_);
}

auto HiggsDecoderImpl::forward(ops::Context& context, const Tensor& tokens) const -> Tensor {
    if (tokens.device() != context.device() || tokens.dtype() != tensor::DType::I32 || tokens.dimensions() != 2 ||
        tokens.size(0) != static_cast<std::size_t>(shape_.codebooks) || !tokens.size(1))
        throw ops::Failure({ErrorCode::INVALID_ARGUMENT,
                            "Higgs audio codes require an I32 [codebooks, length] tensor on the execution device"});
    Tensor codes = tokens;
    const auto& capabilities = context.device().capabilities();
    if (!capabilities.memory.offset_views && !capabilities.compute.embedding) {
        context.synchronize();
        const auto bytes = ops::require(tokens.copy_to_host());
        const auto values =
            std::span(reinterpret_cast<const std::int32_t*>(bytes.data()), bytes.size() / sizeof(std::int32_t));
        codes = ops::require(
            ops::require(Tensor::from_host({static_cast<std::int64_t>(tokens.size(0)),
                                            static_cast<std::int64_t>(tokens.size(1))},
                                           values))
                .to(context.device()));
    }
    const auto project = [&](std::span<const Tensor> inputs) {
        Tensor quantized;
        for (std::int32_t codebook = 0; codebook < shape_.codebooks; ++codebook) {
            const auto row = context.reshape(context.slice(inputs[0], 0, codebook, 1),
                                             {static_cast<std::int64_t>(inputs[0].size(1))});
            auto decoded = (*quantizers_)[codebook]->forward(context, row);
            if (quantized.defined())
                context.add_(quantized, decoded);
            else
                quantized = std::move(decoded);
        }
        return std::vector{acoustic_projection_->forward(context, quantized)};
    };
    Tensor acoustic;
    if (context.device().policy().replay.region_layers) {
        const std::array inputs{codes};
        acoustic = context.replay("higgs:codes:" + std::to_string(tokens.size(1)), inputs, project, false).front();
    } else {
        acoustic = project(std::array{codes}).front();
    }
    return decoder_->forward(context, acoustic);
}

} // namespace kidi::layers
