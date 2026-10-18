#include "kidi/layers/audio.h"

#include <algorithm>
#include <cmath>
#include <string_view>

#include "kidi/runtime/ynn/graph.h"

namespace kidi::layers {
using ops::require;
namespace {

auto audio_parameter_device() -> tensor::Device {
    return module_device == tensor::Device::apple_gpu() ? tensor::Device::cpu() : module_device;
}

auto require_cpu(const Tensor& input, std::int32_t channels, std::string_view operation) -> void {
    if (input.device() != tensor::Device::cpu() || input.dtype() != tensor::DType::F32 || input.dimensions() != 3 ||
        input.size(0) != 1 || input.size(2) != channels)
        throw ops::Failure(
            {ErrorCode::UNSUPPORTED, std::string(operation) + " requires FP32 [1, time, channels] CPU input"});
}

template <typename Function>
auto for_independent_ranges(std::size_t size, std::size_t work_per_item, Function&& function) -> void {
    const auto tasks = size * work_per_item < 32768 ? std::size_t{1} : std::min(runtime::ynn::thread_count(), size);
    if (tasks <= 1) return function(0, size);
    const auto block = (size + tasks - 1) / tasks;
    require(runtime::ynn::parallel_for(tasks, [&](std::size_t task) {
        const auto begin = task * block;
        if (begin < size) function(begin, std::min(begin + block, size));
    }));
}

} // namespace

Conv1dImpl::Conv1dImpl(std::int32_t input_channels, std::int32_t output_channels, std::int32_t kernel_size,
                       std::int32_t stride, std::int32_t padding, std::int32_t dilation)
    : input_channels_(input_channels),
      output_channels_(output_channels),
      kernel_size_(kernel_size),
      stride_(stride),
      padding_(padding),
      dilation_(dilation) {
    if (input_channels <= 0 || output_channels <= 0 || kernel_size <= 0 || stride <= 0 || padding < 0 ||
        dilation <= 0 || (module_dtype != tensor::DType::F32 && module_dtype != tensor::DType::I8))
        throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "invalid Conv1d construction"});
    register_parameter("weight", weight_, {output_channels, input_channels * kernel_size}, module_dtype,
                       allocate_parameters, audio_parameter_device());
    if (module_dtype == tensor::DType::I8) {
        qstate_.emplace();
        register_parameter("scale", qstate_->scale, {output_channels, 1}, tensor::DType::F32, allocate_parameters,
                           audio_parameter_device());
        bias_qstate_.emplace();
        const auto vectors = module_device == tensor::Device::web_gpu() ? tensor::Device::cpu()
                                                                        : audio_parameter_device();
        register_parameter("bias", bias_, {output_channels}, tensor::DType::I8, allocate_parameters, vectors);
        register_parameter("bias_scale", bias_qstate_->scale, {1}, tensor::DType::F32, allocate_parameters, vectors);
        if (allocate_parameters) {
            std::ranges::fill(require(qstate_->scale.data<float>()), 1.F);
            std::ranges::fill(require(bias_qstate_->scale.data<float>()), 1.F);
        }
    } else {
        register_parameter("bias", bias_, {output_channels}, tensor::DType::F32, allocate_parameters,
                           audio_parameter_device());
    }
}

auto Conv1dImpl::bias() const -> const Tensor& {
    return bias_qstate_ ? bias_qstate_->decode_vector(bias_, "Conv1d bias") : bias_;
}

auto Conv1dImpl::forward(ops::Context& context, const Tensor& input) const -> Tensor {
    if (context.device() == tensor::Device::web_gpu() || context.device() == tensor::Device::apple_gpu())
        return context.conv1d(input, weight_, qstate_ ? qstate_->scale : Tensor{}, bias(),
                               kernel_size_, stride_, padding_, dilation_);
    require_cpu(input, input_channels_, "Conv1d");
    const auto input_length = static_cast<std::int64_t>(input.size(1));
    const auto receptive = dilation_ * (kernel_size_ - 1) + 1;
    const auto numerator = input_length + 2 * padding_ - receptive;
    if (numerator < 0) throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "Conv1d kernel exceeds padded input"});
    const auto output_length = numerator / stride_ + 1;
    Tensor columns = input;
    if (kernel_size_ != 1 || stride_ != 1 || padding_ != 0 || dilation_ != 1) {
        columns = require(Tensor::zeros({1, output_length, input_channels_ * kernel_size_}, tensor::DType::F32));
        context.synchronize();
        const auto source = require(input.data<float>());
        auto unfolded = require(columns.data<float>());
        for_independent_ranges(output_length, input_channels_ * kernel_size_, [&](std::size_t begin, std::size_t end) {
            for (auto output = begin; output < end; ++output)
                for (std::int32_t channel = 0; channel < input_channels_; ++channel)
                    for (std::int32_t kernel = 0; kernel < kernel_size_; ++kernel) {
                        const auto position =
                            static_cast<std::int64_t>(output) * stride_ - padding_ + kernel * dilation_;
                        if (position >= 0 && position < input_length)
                            unfolded[(output * input_channels_ + channel) * kernel_size_ + kernel] =
                                source[position * input_channels_ + channel];
                    }
        });
    }
    if (qstate_) return context.quantized_linear(columns, weight_, qstate_->scale, bias(), true);
    return context.linear(columns, weight_, bias(), true);
}

auto Conv1dImpl::forward_snake(ops::Context& context, const Tensor& input, const Tensor& alpha) const -> Tensor {
    return context.snake_conv1d(input, alpha, weight_, qstate_ ? qstate_->scale : Tensor{}, bias(), kernel_size_,
                                stride_, padding_, dilation_);
}

ConvTranspose1dImpl::ConvTranspose1dImpl(std::int32_t input_channels, std::int32_t output_channels,
                                         std::int32_t kernel_size, std::int32_t stride, std::int32_t padding,
                                         std::int32_t output_padding, std::int32_t dilation)
    : input_channels_(input_channels),
      output_channels_(output_channels),
      kernel_size_(kernel_size),
      stride_(stride),
      padding_(padding),
      output_padding_(output_padding),
      dilation_(dilation) {
    if (input_channels <= 0 || output_channels <= 0 || kernel_size <= 0 || stride <= 0 || padding < 0 ||
        output_padding < 0 || output_padding >= stride || dilation <= 0 ||
        (module_dtype != tensor::DType::F32 && module_dtype != tensor::DType::I8))
        throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "invalid ConvTranspose1d construction"});
    register_parameter("weight", weight_, {output_channels * kernel_size, input_channels}, module_dtype,
                       allocate_parameters, audio_parameter_device());
    if (module_dtype == tensor::DType::I8) {
        qstate_.emplace();
        register_parameter("scale", qstate_->scale, {output_channels * kernel_size, 1}, tensor::DType::F32,
                           allocate_parameters, audio_parameter_device());
        bias_qstate_.emplace();
        const auto vectors = module_device == tensor::Device::web_gpu() ? tensor::Device::cpu()
                                                                        : audio_parameter_device();
        register_parameter("bias", bias_, {output_channels}, tensor::DType::I8, allocate_parameters, vectors);
        register_parameter("bias_scale", bias_qstate_->scale, {1}, tensor::DType::F32, allocate_parameters, vectors);
        if (allocate_parameters) {
            std::ranges::fill(require(qstate_->scale.data<float>()), 1.F);
            std::ranges::fill(require(bias_qstate_->scale.data<float>()), 1.F);
        }
    } else {
        register_parameter("bias", bias_, {output_channels}, tensor::DType::F32, allocate_parameters,
                           audio_parameter_device());
    }
    projection_bias_ =
        require(Tensor::zeros({output_channels * kernel_size}, tensor::DType::F32, tensor::Device::cpu()));
}

auto ConvTranspose1dImpl::bias() const -> const Tensor& {
    return bias_qstate_ ? bias_qstate_->decode_vector(bias_, "ConvTranspose1d bias") : bias_;
}

auto ConvTranspose1dImpl::forward(ops::Context& context, const Tensor& input) const -> Tensor {
    if (context.device() == tensor::Device::web_gpu() || context.device() == tensor::Device::apple_gpu()) {
        const auto projected = qstate_
            ? context.quantized_linear(input, weight_, qstate_->scale, projection_bias_, true)
            : context.linear(input, weight_, projection_bias_, true);
        return context.conv_transpose_fold(projected, bias(), kernel_size_, stride_, padding_, output_padding_, dilation_);
    }
    require_cpu(input, input_channels_, "ConvTranspose1d");
    const auto input_length = static_cast<std::int64_t>(input.size(1));
    const auto output_length =
        (input_length - 1) * stride_ - 2 * padding_ + dilation_ * (kernel_size_ - 1) + output_padding_ + 1;
    if (output_length <= 0)
        throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "ConvTranspose1d produced an empty sequence"});
    auto projected = qstate_ ? context.quantized_linear(input, weight_, qstate_->scale, projection_bias_, true)
                             : context.linear(input, weight_, projection_bias_, true);
    context.synchronize();
    const auto projected_values = require(projected.data<float>());
    const auto offsets = require(bias().data<float>());
    auto output = require(Tensor::empty({1, output_length, output_channels_}, tensor::DType::F32));
    auto result = require(output.data<float>());
    for_independent_ranges(
        output_channels_, output_length + input_length * kernel_size_, [&](std::size_t begin, std::size_t end) {
            for (auto channel = begin; channel < end; ++channel) {
                for (std::int64_t position = 0; position < output_length; ++position)
                    result[position * output_channels_ + channel] = offsets[channel];
                for (std::int64_t input_position = 0; input_position < input_length; ++input_position)
                    for (std::int32_t kernel = 0; kernel < kernel_size_; ++kernel) {
                        const auto output_position = input_position * stride_ - padding_ + kernel * dilation_;
                        if (output_position >= 0 && output_position < output_length)
                            result[output_position * output_channels_ + channel] +=
                                projected_values[(input_position * output_channels_ + channel) * kernel_size_ + kernel];
                    }
            }
        });
    return output;
}

auto ConvTranspose1dImpl::forward_snake(ops::Context& context, const Tensor& input, const Tensor& alpha) const
    -> Tensor {
    if (!qstate_) return forward(context, context.snake(input, alpha));
    const auto projected =
        context.snake_quantized_linear(input, alpha, weight_, qstate_->scale, projection_bias_, true);
    return context.conv_transpose_fold(projected, bias(), kernel_size_, stride_, padding_, output_padding_, dilation_);
}

Snake1dImpl::Snake1dImpl(std::int32_t channels) : channels_(channels) {
    if (channels <= 0 || (module_dtype != tensor::DType::F32 && module_dtype != tensor::DType::I8))
        throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "invalid Snake1d construction"});
    const auto vectors =
        module_device == tensor::Device::web_gpu() ? tensor::Device::cpu() : audio_parameter_device();
    register_parameter("alpha", alpha_, {channels}, module_dtype, allocate_parameters, vectors);
    if (module_dtype == tensor::DType::I8) {
        qstate_.emplace();
        register_parameter("alpha_scale", qstate_->scale, {1}, tensor::DType::F32, allocate_parameters, vectors);
        if (allocate_parameters) std::ranges::fill(require(qstate_->scale.data<float>()), 1.F);
    } else if (allocate_parameters) {
        std::ranges::fill(require(alpha_.data<float>()), 1.F);
    }
}

auto Snake1dImpl::alpha() const -> const Tensor& {
    return qstate_ ? qstate_->decode_vector(alpha_, "Snake1d alpha") : alpha_;
}

auto Snake1dImpl::forward(ops::Context& context, const Tensor& input) const -> Tensor {
    if (context.device() == tensor::Device::web_gpu() || context.device() == tensor::Device::apple_gpu())
        return context.snake(input, alpha());
    require_cpu(input, channels_, "Snake1d");
    context.synchronize();
    const auto source = require(input.data<float>());
    const auto parameters = require(alpha().data<float>());
    auto output = require(Tensor::empty({input.shape().begin(), input.shape().end()}, tensor::DType::F32));
    auto result = require(output.data<float>());
    for_independent_ranges(source.size(), 16, [&](std::size_t begin, std::size_t end) {
        for (auto index = begin; index < end; ++index) {
            const auto coefficient = parameters[index % channels_];
            if (std::abs(coefficient) < 1e-9F) {
                result[index] = source[index];
                continue;
            }
            const auto sine = std::sin(coefficient * source[index]);
            result[index] = source[index] + sine * sine / (coefficient + 1e-9F);
        }
    });
    return output;
}

auto Snake1dImpl::forward(ops::Context& context, const Tensor& input, const Conv1d& convolution) const -> Tensor {
    if (context.device() == tensor::Device::web_gpu() || context.device() == tensor::Device::apple_gpu())
        return convolution->forward_snake(context, input, alpha());
    return convolution->forward(context, forward(context, input));
}

auto Snake1dImpl::forward(ops::Context& context, const Tensor& input, const ConvTranspose1d& convolution) const
    -> Tensor {
    if (context.device() == tensor::Device::web_gpu() || context.device() == tensor::Device::apple_gpu())
        return convolution->forward_snake(context, input, alpha());
    return convolution->forward(context, forward(context, input));
}

} // namespace kidi::layers
