#pragma once

#include "kidi/layers/transformer.h"

namespace kidi::layers {

KIDI_MODULE(Conv1d);
class Conv1dImpl : public Module {
public:
    Conv1dImpl(std::int32_t input_channels, std::int32_t output_channels, std::int32_t kernel_size, std::int32_t stride,
               std::int32_t padding, std::int32_t dilation = 1);
    auto forward(ops::Context& context, const Tensor& input) const -> Tensor;

private:
    auto bias() const -> const Tensor&;

    Tensor weight_, bias_;
    std::optional<QuantizedState> qstate_, bias_qstate_;
    std::int32_t input_channels_, output_channels_, kernel_size_, stride_, padding_, dilation_;
};

KIDI_MODULE(ConvTranspose1d);
class ConvTranspose1dImpl : public Module {
public:
    ConvTranspose1dImpl(std::int32_t input_channels, std::int32_t output_channels, std::int32_t kernel_size,
                        std::int32_t stride, std::int32_t padding, std::int32_t output_padding = 0,
                        std::int32_t dilation = 1);
    auto forward(ops::Context& context, const Tensor& input) const -> Tensor;

private:
    auto bias() const -> const Tensor&;

    Tensor weight_, bias_, projection_bias_;
    std::optional<QuantizedState> qstate_, bias_qstate_;
    std::int32_t input_channels_, output_channels_, kernel_size_, stride_, padding_, output_padding_, dilation_;
};

KIDI_MODULE(Snake1d);
class Snake1dImpl : public Module {
public:
    explicit Snake1dImpl(std::int32_t channels);
    auto forward(ops::Context& context, const Tensor& input) const -> Tensor;

private:
    auto alpha() const -> const Tensor&;

    Tensor alpha_;
    std::optional<QuantizedState> qstate_;
    std::int32_t channels_;
};

} // namespace kidi::layers
