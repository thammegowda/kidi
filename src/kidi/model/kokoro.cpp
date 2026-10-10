#include "kidi/model/kokoro.h"

// Architecture equations follow hexgrad/kokoro (Apache-2.0). The native layer
// decomposition was cross-checked against wlejon/brosoundml (MIT).

#include <algorithm>
#include <bit>
#include <chrono>
#include <cctype>
#include <cmath>
#include <complex>
#include <map>
#include <numbers>
#include <numeric>
#include <optional>
#include <random>

#include "kidi/ops/context.h"
#include "kidi/text/phonemizer.h"

namespace kidi::model {
using ops::require;
using tensor::DType;
using tensor::Tensor;
namespace {

auto scale_name(std::string_view name) -> std::string {
    if (name.ends_with(".weight")) return std::string(name.substr(0, name.size() - 7)) + ".scale";
    return std::string(name) + "_scale";
}

class Parameters {
public:
    Parameters(checkpoint::Weights weights, std::int32_t group_size)
        : weights_(std::move(weights)), group_size_(group_size) {}

    auto tensor(std::string_view name) const -> Tensor { return require(weights_.tensor(name)); }

    auto vector(std::string_view name) const -> const Tensor& {
        const auto key = std::string(name);
        if (const auto found = decoded_.find(key); found != decoded_.end()) return found->second;
        const auto encoded = tensor(name);
        const auto scale = tensor(scale_name(name));
        if (encoded.dtype() != DType::I8 || scale.dtype() != DType::F32 ||
            (scale.numel() != 1 && scale.numel() != encoded.numel()))
            throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "invalid Kokoro vector: " + key});
        const auto source = require(encoded.data<std::int8_t>());
        const auto multipliers = require(scale.data<float>());
        std::vector<float> values(source.size());
        for (std::size_t index = 0; index < source.size(); ++index)
            values[index] = source[index] * multipliers[multipliers.size() == 1 ? 0 : index];
        return decoded_
            .emplace(key, require(Tensor::from_host({static_cast<std::int64_t>(values.size())},
                                                    std::span<const float>(values))))
            .first->second;
    }

    auto matrix(std::string_view name) const -> const Tensor& {
        const auto key = std::string(name);
        if (const auto found = decoded_.find(key); found != decoded_.end()) return found->second;
        const auto encoded = tensor(name);
        const auto scale = tensor(scale_name(name));
        if (encoded.dtype() != DType::I8 || encoded.dimensions() != 2 || scale.dtype() != DType::F32 ||
            scale.dimensions() != 2 || scale.size(0) != encoded.size(0) || group_size_ <= 0)
            throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "invalid Kokoro matrix: " + key});
        const auto source = require(encoded.data<std::int8_t>());
        const auto multipliers = require(scale.data<float>());
        const auto rows = static_cast<std::size_t>(encoded.size(0));
        const auto columns = static_cast<std::size_t>(encoded.size(1));
        const auto groups = static_cast<std::size_t>(scale.size(1));
        if (groups != (columns + group_size_ - 1) / group_size_)
            throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "invalid Kokoro matrix scales: " + key});
        std::vector<float> values(source.size());
        for (std::size_t row = 0; row < rows; ++row)
            for (std::size_t column = 0; column < columns; ++column)
                values[row * columns + column] =
                    source[row * columns + column] * multipliers[row * groups + column / group_size_];
        return decoded_
            .emplace(key,
                     require(Tensor::from_host({static_cast<std::int64_t>(rows), static_cast<std::int64_t>(columns)},
                                               std::span<const float>(values))))
            .first->second;
    }

    auto project(ops::Context& context, std::string_view weight_name, const Tensor& input,
                 std::string_view bias_name = {}) const -> Tensor {
        const auto& weight = matrix(weight_name);
        Tensor bias;
        if (!bias_name.empty()) {
            bias = vector(bias_name);
        } else {
            bias = require(Tensor::zeros({static_cast<std::int64_t>(weight.size(0))}, DType::F32));
        }
        return context.linear(input, weight, bias, true);
    }

    auto linear(ops::Context& context, std::string_view prefix, const Tensor& input) const -> Tensor {
        const auto weight = std::string(prefix) + ".weight";
        const auto bias = std::string(prefix) + ".bias";
        return project(context, weight, input, weights_.contains(bias) ? std::string_view(bias) : std::string_view{});
    }

    auto embedding(std::string_view name, std::span<const std::int32_t> ids) const -> Tensor {
        const auto weight = tensor(name);
        const auto scales = tensor(scale_name(name));
        if (weight.dtype() != DType::I8 || weight.dimensions() != 2 || scales.dtype() != DType::F32 ||
            scales.dimensions() != 2 || scales.size(0) != weight.size(0))
            throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "invalid Kokoro embedding: " + std::string(name)});
        const auto table = require(weight.data<std::int8_t>());
        const auto multipliers = require(scales.data<float>());
        const auto width = static_cast<std::size_t>(weight.size(1));
        const auto groups = static_cast<std::size_t>(scales.size(1));
        std::vector<float> values(ids.size() * width);
        for (std::size_t row = 0; row < ids.size(); ++row) {
            if (ids[row] < 0 || static_cast<std::size_t>(ids[row]) >= weight.size(0))
                throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "Kokoro embedding id is outside vocabulary"});
            const auto source = static_cast<std::size_t>(ids[row]) * width;
            for (std::size_t column = 0; column < width; ++column)
                values[row * width + column] =
                    table[source + column] * multipliers[ids[row] * groups + column / group_size_];
        }
        return require(Tensor::from_host({1, static_cast<std::int64_t>(ids.size()), static_cast<std::int64_t>(width)},
                                         std::span<const float>(values)));
    }

    auto voice(std::string_view name, std::size_t row) const -> std::vector<float> {
        const auto key = "voices." + std::string(name) + ".weight";
        const auto weight = tensor(key);
        const auto scales = tensor(scale_name(key));
        if (weight.dtype() != DType::I8 || weight.dimensions() != 2 || weight.size(1) != 256 || row >= weight.size(0) ||
            scales.dtype() != DType::F32 || scales.dimensions() != 2 || scales.size(0) != weight.size(0) ||
            scales.size(1) != (weight.size(1) + group_size_ - 1) / group_size_)
            throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "invalid Kokoro voice: " + std::string(name)});
        const auto values = require(weight.data<std::int8_t>());
        const auto multipliers = require(scales.data<float>());
        const auto groups = static_cast<std::size_t>(scales.size(1));
        std::vector<float> result(256);
        for (std::size_t index = 0; index < result.size(); ++index)
            result[index] = values[row * result.size() + index] * multipliers[row * groups + index / group_size_];
        return result;
    }

    auto contains(std::string_view name) const -> bool { return weights_.contains(name); }
    auto group_size() const noexcept -> std::int32_t { return group_size_; }

    auto validate(const YAML::Node& config) const -> void {
        if (group_size_ <= 0)
            throw ops::Failure({ErrorCode::INVALID_MANIFEST, "Kokoro weight group size must be positive"});
        const auto is_scale = [](std::string_view name) {
            return name.ends_with(".scale") || name.ends_with("_scale");
        };
        for (const auto& name : weights_.names()) {
            const auto value = tensor(name);
            if (is_scale(name)) {
                if (value.dtype() != DType::F32)
                    throw ops::Failure({ErrorCode::INVALID_MANIFEST, "Kokoro scale must be FP32: " + name});
                continue;
            }
            if (value.dtype() != DType::I8 || (value.dimensions() != 1 && value.dimensions() != 2))
                throw ops::Failure(
                    {ErrorCode::INVALID_MANIFEST, "Kokoro parameter must be rank-one or rank-two INT8: " + name});
            const auto expected_scale = scale_name(name);
            if (!weights_.contains(expected_scale))
                throw ops::Failure(
                    {ErrorCode::INVALID_MANIFEST, "Kokoro parameter lacks quantization scales: " + name});
            const auto scales = tensor(expected_scale);
            if (scales.dtype() != DType::F32)
                throw ops::Failure({ErrorCode::INVALID_MANIFEST, "Kokoro parameter scale must be FP32: " + name});
            if (value.dimensions() == 1) {
                if (scales.dimensions() != 1 || (scales.numel() != 1 && scales.numel() != value.numel()))
                    throw ops::Failure({ErrorCode::INVALID_MANIFEST, "Kokoro vector scale shape mismatch: " + name});
            } else {
                const auto groups = (value.size(1) + group_size_ - 1) / group_size_;
                if (scales.dimensions() != 2 || scales.size(0) != value.size(0) || scales.size(1) != groups)
                    throw ops::Failure({ErrorCode::INVALID_MANIFEST, "Kokoro matrix scale shape mismatch: " + name});
            }
        }
        for (const auto& item : config["voices"]) {
            const auto voice_name = item.as<std::string>();
            const auto name = "voices." + voice_name + ".weight";
            if (!weights_.contains(name))
                throw ops::Failure({ErrorCode::INVALID_MANIFEST, "Kokoro package lacks declared voice: " + voice_name});
            const auto voice = tensor(name);
            const auto scales = tensor(scale_name(name));
            const auto groups = (256 + group_size_ - 1) / group_size_;
            if (voice.dimensions() != 2 || voice.size(0) < 510 || voice.size(1) != 256 || scales.dimensions() != 2 ||
                scales.size(0) != voice.size(0) || scales.size(1) != groups)
                throw ops::Failure({ErrorCode::INVALID_MANIFEST, "Kokoro voice tensor shape mismatch: " + voice_name});
        }
    }

private:
    checkpoint::Weights weights_;
    std::int32_t group_size_;
    mutable std::map<std::string, Tensor, std::less<>> decoded_;
};

auto shape(const Tensor& input, std::int64_t time, std::int64_t channels) -> void {
    if (input.dtype() != DType::F32 || input.device() != tensor::Device::cpu() || input.dimensions() != 3 ||
        input.size(0) != 1 || input.size(1) != time || input.size(2) != channels)
        throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "Kokoro tensor shape mismatch"});
}

auto values(ops::Context& context, const Tensor& input) -> std::span<const float> {
    context.synchronize();
    return require(input.data<float>());
}

auto mutable_values(ops::Context& context, Tensor& input) -> std::span<float> {
    context.synchronize();
    return require(input.data<float>());
}

auto checked(ops::Context& context, Tensor input, std::string_view stage) -> Tensor {
    const auto data = values(context, input);
    float maximum = 0;
    for (const auto value : data) {
        if (!std::isfinite(value))
            throw ops::Failure(
                {ErrorCode::RUNTIME, "Kokoro produced a non-finite activation at " + std::string(stage)});
        maximum = std::max(maximum, std::abs(value));
    }
    if (maximum > 1e6F)
        throw ops::Failure({ErrorCode::RUNTIME, "Kokoro activation exceeded range at " + std::string(stage) + ": " +
                                                    std::to_string(maximum)});
    return input;
}

auto from_values(std::size_t time, std::size_t channels, std::vector<float> data) -> Tensor {
    return require(Tensor::from_host({1, static_cast<std::int64_t>(time), static_cast<std::int64_t>(channels)},
                                     std::span<const float>(data)));
}

auto add(ops::Context& context, const Tensor& left, const Tensor& right) -> Tensor { return context.add(left, right); }

auto concat(ops::Context& context, std::span<const Tensor> inputs) -> Tensor {
    if (inputs.empty()) throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "cannot concatenate no Kokoro tensors"});
    const auto time = inputs.front().size(1);
    std::size_t channels = 0;
    for (const auto& input : inputs) {
        if (input.dimensions() != 3 || input.size(0) != 1 || input.size(1) != time || input.dtype() != DType::F32)
            throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "Kokoro concatenation shape mismatch"});
        channels += input.size(2);
    }
    std::vector<float> output(static_cast<std::size_t>(time) * channels);
    std::size_t offset = 0;
    for (const auto& input : inputs) {
        const auto source = values(context, input);
        const auto width = static_cast<std::size_t>(input.size(2));
        for (std::size_t row = 0; row < static_cast<std::size_t>(time); ++row)
            std::ranges::copy(source.subspan(row * width, width), output.begin() + row * channels + offset);
        offset += width;
    }
    return from_values(time, channels, std::move(output));
}

auto activate(ops::Context& context, Tensor input, float slope) -> Tensor {
    auto data = mutable_values(context, input);
    for (auto& value : data)
        if (value < 0) value *= slope;
    return input;
}

auto sigmoid(float value) -> float { return 1.F / (1.F + std::exp(-value)); }

auto layer_norm(ops::Context& context, const Parameters& parameters, const Tensor& input, std::string_view weight,
                std::string_view bias, float epsilon) -> Tensor {
    const auto gamma = require(parameters.vector(weight).data<float>());
    const auto beta = require(parameters.vector(bias).data<float>());
    const auto source = values(context, input);
    const auto time = static_cast<std::size_t>(input.size(1));
    const auto channels = static_cast<std::size_t>(input.size(2));
    std::vector<float> output(source.size());
    for (std::size_t row = 0; row < time; ++row) {
        const auto begin = source.subspan(row * channels, channels);
        const auto mean = std::accumulate(begin.begin(), begin.end(), 0.F) / channels;
        float variance = 0;
        for (const auto value : begin) variance += (value - mean) * (value - mean);
        const auto inverse = 1.F / std::sqrt(variance / channels + epsilon);
        for (std::size_t channel = 0; channel < channels; ++channel)
            output[row * channels + channel] = (begin[channel] - mean) * inverse * gamma[channel] + beta[channel];
    }
    return from_values(time, channels, std::move(output));
}

auto layer_norm_plain(ops::Context& context, const Tensor& input, std::span<const float> gamma,
                      std::span<const float> beta, float epsilon) -> Tensor {
    const auto source = values(context, input);
    const auto time = static_cast<std::size_t>(input.size(1));
    const auto channels = static_cast<std::size_t>(input.size(2));
    std::vector<float> output(source.size());
    for (std::size_t row = 0; row < time; ++row) {
        const auto begin = source.subspan(row * channels, channels);
        const auto mean = std::accumulate(begin.begin(), begin.end(), 0.F) / channels;
        float variance = 0;
        for (const auto value : begin) variance += (value - mean) * (value - mean);
        const auto inverse = 1.F / std::sqrt(variance / channels + epsilon);
        for (std::size_t channel = 0; channel < channels; ++channel)
            output[row * channels + channel] = (begin[channel] - mean) * inverse * gamma[channel] + beta[channel];
    }
    return from_values(time, channels, std::move(output));
}

auto conv1d(ops::Context& context, const Parameters& parameters, std::string_view prefix, const Tensor& input,
            std::int32_t output_channels, std::int32_t kernel, std::int32_t stride = 1, std::int32_t padding = 0,
            std::int32_t dilation = 1) -> Tensor {
    const auto input_channels = static_cast<std::int32_t>(input.size(2));
    const auto input_length = static_cast<std::int64_t>(input.size(1));
    const auto receptive = dilation * (kernel - 1) + 1;
    const auto output_length = (input_length + 2 * padding - receptive) / stride + 1;
    if (output_length <= 0) throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "empty Kokoro convolution"});
    auto columns = require(Tensor::zeros({1, output_length, input_channels * kernel}, DType::F32));
    const auto source = values(context, input);
    auto unfolded = require(columns.data<float>());
    for (std::int64_t output = 0; output < output_length; ++output)
        for (std::int32_t channel = 0; channel < input_channels; ++channel)
            for (std::int32_t tap = 0; tap < kernel; ++tap) {
                const auto position = output * stride - padding + tap * dilation;
                if (position >= 0 && position < input_length)
                    unfolded[(output * input_channels + channel) * kernel + tap] =
                        source[position * input_channels + channel];
            }
    const auto result = parameters.linear(context, prefix, columns);
    if (result.size(2) != output_channels)
        throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "Kokoro convolution output mismatch"});
    return result;
}

auto conv_transpose1d(ops::Context& context, const Parameters& parameters, std::string_view prefix, const Tensor& input,
                      std::int32_t output_channels, std::int32_t kernel, std::int32_t stride, std::int32_t padding,
                      std::int32_t output_padding = 0) -> Tensor {
    const auto input_length = static_cast<std::int64_t>(input.size(1));
    const auto output_length = (input_length - 1) * stride - 2 * padding + kernel + output_padding;
    const auto weight_name = std::string(prefix) + ".weight";
    auto projected = parameters.project(context, weight_name, input);
    const auto source = values(context, projected);
    std::vector<float> bias(output_channels, 0.F);
    const auto bias_name = std::string(prefix) + ".bias";
    if (parameters.contains(bias_name)) {
        const auto values = require(parameters.vector(bias_name).data<float>());
        std::ranges::copy(values, bias.begin());
    }
    std::vector<float> result(static_cast<std::size_t>(output_length) * output_channels);
    for (std::int64_t position = 0; position < output_length; ++position)
        std::ranges::copy(bias, result.begin() + position * output_channels);
    for (std::int64_t position = 0; position < input_length; ++position)
        for (std::int32_t channel = 0; channel < output_channels; ++channel)
            for (std::int32_t tap = 0; tap < kernel; ++tap) {
                const auto target = position * stride - padding + tap;
                if (target >= 0 && target < output_length)
                    result[target * output_channels + channel] +=
                        source[(position * output_channels + channel) * kernel + tap];
            }
    return from_values(output_length, output_channels, std::move(result));
}

auto depthwise_conv_transpose1d(ops::Context& context, const Parameters& parameters, std::string_view prefix,
                                const Tensor& input, std::int32_t kernel, std::int32_t stride, std::int32_t padding,
                                std::int32_t output_padding) -> Tensor {
    const auto channels = static_cast<std::size_t>(input.size(2));
    const auto input_length = static_cast<std::size_t>(input.size(1));
    const auto output_length = (input_length - 1) * stride - 2 * padding + kernel + output_padding;
    const auto weight = parameters.tensor(std::string(prefix) + ".weight");
    const auto scales = parameters.tensor(std::string(prefix) + ".scale");
    const auto encoded = require(weight.data<std::int8_t>());
    const auto multipliers = require(scales.data<float>());
    std::vector<float> bias(channels, 0.F);
    const auto bias_name = std::string(prefix) + ".bias";
    if (parameters.contains(bias_name)) {
        const auto source = require(parameters.vector(bias_name).data<float>());
        std::ranges::copy(source, bias.begin());
    }
    const auto source = values(context, input);
    std::vector<float> output(output_length * channels);
    for (std::size_t time = 0; time < output_length; ++time) std::ranges::copy(bias, output.begin() + time * channels);
    for (std::size_t time = 0; time < input_length; ++time)
        for (std::size_t channel = 0; channel < channels; ++channel)
            for (std::int32_t tap = 0; tap < kernel; ++tap) {
                const auto target = static_cast<std::int64_t>(time * stride) - padding + tap;
                if (target >= 0 && static_cast<std::size_t>(target) < output_length)
                    output[target * channels + channel] +=
                        source[time * channels + channel] * encoded[channel * kernel + tap] *
                        multipliers[channel * scales.size(1) + tap / parameters.group_size()];
            }
    return from_values(output_length, channels, std::move(output));
}

auto nearest_2x(ops::Context& context, const Tensor& input) -> Tensor {
    const auto source = values(context, input);
    const auto time = static_cast<std::size_t>(input.size(1));
    const auto channels = static_cast<std::size_t>(input.size(2));
    std::vector<float> output(2 * source.size());
    for (std::size_t row = 0; row < time; ++row)
        for (std::size_t repeat = 0; repeat < 2; ++repeat)
            std::ranges::copy(source.subspan(row * channels, channels), output.begin() + (2 * row + repeat) * channels);
    return from_values(2 * time, channels, std::move(output));
}

auto linear_interpolate(ops::Context& context, const Tensor& input, std::size_t output_length) -> Tensor {
    const auto source = values(context, input);
    const auto input_length = static_cast<std::size_t>(input.size(1));
    const auto channels = static_cast<std::size_t>(input.size(2));
    std::vector<float> output(output_length * channels);
    const auto scale = static_cast<double>(input_length) / output_length;
    for (std::size_t out = 0; out < output_length; ++out) {
        const auto position = (out + 0.5) * scale - 0.5;
        const auto left = static_cast<std::size_t>(std::max(0.0, std::floor(position)));
        const auto right = std::min(input_length - 1, left + 1);
        const auto fraction = static_cast<float>(std::max(0.0, position - std::floor(position)));
        for (std::size_t channel = 0; channel < channels; ++channel)
            output[out * channels + channel] =
                source[left * channels + channel] * (1.F - fraction) + source[right * channels + channel] * fraction;
    }
    return from_values(output_length, channels, std::move(output));
}

auto bilstm(ops::Context& context, const Parameters& parameters, std::string_view prefix, const Tensor& input,
            std::int32_t hidden) -> Tensor {
    const auto time = static_cast<std::size_t>(input.size(1));
    const auto run = [&](bool reverse) {
        const auto suffix = reverse ? "_reverse" : "";
        const auto input_weight = std::string(prefix) + ".weight_ih_l0" + suffix;
        const auto input_bias = std::string(prefix) + ".bias_ih_l0" + suffix;
        auto projected = parameters.project(context, input_weight, input, input_bias);
        const auto input_values = values(context, projected);
        std::vector<float> output(time * hidden), h(hidden, 0.F), c(hidden, 0.F);
        for (std::size_t step = 0; step < time; ++step) {
            const auto row = reverse ? time - 1 - step : step;
            auto h_tensor = from_values(1, hidden, h);
            const auto recurrent_weight = std::string(prefix) + ".weight_hh_l0" + suffix;
            const auto recurrent_bias = std::string(prefix) + ".bias_hh_l0" + suffix;
            auto recurrent = parameters.project(context, recurrent_weight, h_tensor, recurrent_bias);
            const auto recurrent_values = values(context, recurrent);
            for (std::int32_t channel = 0; channel < hidden; ++channel) {
                const auto i = sigmoid(input_values[row * 4 * hidden + channel] + recurrent_values[channel]);
                const auto f =
                    sigmoid(input_values[row * 4 * hidden + hidden + channel] + recurrent_values[hidden + channel]);
                const auto g = std::tanh(input_values[row * 4 * hidden + 2 * hidden + channel] +
                                         recurrent_values[2 * hidden + channel]);
                const auto o = sigmoid(input_values[row * 4 * hidden + 3 * hidden + channel] +
                                       recurrent_values[3 * hidden + channel]);
                c[channel] = f * c[channel] + i * g;
                h[channel] = o * std::tanh(c[channel]);
                output[row * hidden + channel] = h[channel];
            }
        }
        return output;
    };
    const auto forward = run(false), backward = run(true);
    std::vector<float> output(time * 2 * hidden);
    for (std::size_t row = 0; row < time; ++row) {
        std::ranges::copy(std::span(forward).subspan(row * hidden, hidden), output.begin() + row * 2 * hidden);
        std::ranges::copy(std::span(backward).subspan(row * hidden, hidden),
                          output.begin() + row * 2 * hidden + hidden);
    }
    return from_values(time, 2 * hidden, std::move(output));
}

auto style_tensor(std::span<const float> style) -> Tensor {
    return require(Tensor::from_host({1, 1, static_cast<std::int64_t>(style.size())}, style));
}

auto style_affine(ops::Context& context, const Parameters& parameters, std::string_view prefix,
                  std::span<const float> style, std::size_t channels)
    -> std::pair<std::vector<float>, std::vector<float>> {
    auto projected = parameters.linear(context, std::string(prefix) + ".fc", style_tensor(style));
    const auto data = values(context, projected);
    if (data.size() != 2 * channels)
        throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "Kokoro style projection mismatch"});
    return {{data.begin(), data.begin() + channels}, {data.begin() + channels, data.end()}};
}

auto ada_layer_norm(ops::Context& context, const Parameters& parameters, std::string_view prefix, const Tensor& input,
                    std::span<const float> style) -> Tensor {
    const auto channels = static_cast<std::size_t>(input.size(2));
    auto [gamma, beta] = style_affine(context, parameters, prefix, style, channels);
    for (auto& value : gamma) value += 1.F;
    return layer_norm_plain(context, input, gamma, beta, 1e-5F);
}

auto ada_instance_norm(ops::Context& context, const Parameters& parameters, std::string_view prefix,
                       const Tensor& input, std::span<const float> style) -> Tensor {
    const auto time = static_cast<std::size_t>(input.size(1));
    const auto channels = static_cast<std::size_t>(input.size(2));
    auto [gamma, beta] = style_affine(context, parameters, prefix, style, channels);
    const auto source = values(context, input);
    std::vector<float> output(source.size());
    for (std::size_t channel = 0; channel < channels; ++channel) {
        float mean = 0;
        for (std::size_t row = 0; row < time; ++row) mean += source[row * channels + channel];
        mean /= time;
        float variance = 0;
        for (std::size_t row = 0; row < time; ++row) {
            const auto difference = source[row * channels + channel] - mean;
            variance += difference * difference;
        }
        const auto inverse = 1.F / std::sqrt(variance / time + 1e-5F);
        for (std::size_t row = 0; row < time; ++row)
            output[row * channels + channel] =
                (1.F + gamma[channel]) * (source[row * channels + channel] - mean) * inverse + beta[channel];
    }
    return from_values(time, channels, std::move(output));
}

auto snake(ops::Context& context, const Parameters& parameters, std::string_view name, Tensor input) -> Tensor {
    const auto alpha = require(parameters.vector(name).data<float>());
    auto data = mutable_values(context, input);
    const auto channels = static_cast<std::size_t>(input.size(2));
    for (std::size_t index = 0; index < data.size(); ++index) {
        const auto coefficient = alpha[index % channels];
        const auto sine = std::sin(coefficient * data[index]);
        data[index] += sine * sine / coefficient;
    }
    return input;
}

auto adain_block(ops::Context& context, const Parameters& parameters, std::string_view prefix, const Tensor& input,
                 std::span<const float> style, std::int32_t output_channels, bool upsample) -> Tensor {
    auto residual = ada_instance_norm(context, parameters, std::string(prefix) + ".norm1", input, style);
    residual = activate(context, std::move(residual), 0.2F);
    if (upsample)
        residual = depthwise_conv_transpose1d(context, parameters, std::string(prefix) + ".pool", residual, 3, 2, 1, 1);
    residual = conv1d(context, parameters, std::string(prefix) + ".conv1", residual, output_channels, 3, 1, 1);
    residual = ada_instance_norm(context, parameters, std::string(prefix) + ".norm2", residual, style);
    residual = activate(context, std::move(residual), 0.2F);
    residual = conv1d(context, parameters, std::string(prefix) + ".conv2", residual, output_channels, 3, 1, 1);

    auto shortcut = upsample ? nearest_2x(context, input) : input;
    if (input.size(2) != output_channels)
        shortcut = conv1d(context, parameters, std::string(prefix) + ".conv1x1", shortcut, output_channels, 1);
    auto output = add(context, residual, shortcut);
    auto data = mutable_values(context, output);
    for (auto& value : data) value *= std::numbers::sqrt2_v<float> / 2.F;
    return output;
}

auto adain_resblock(ops::Context& context, const Parameters& parameters, std::string_view prefix, Tensor input,
                    std::span<const float> style, std::int32_t kernel, std::span<const std::int32_t> dilations)
    -> Tensor {
    const auto channels = static_cast<std::int32_t>(input.size(2));
    for (std::size_t index = 0; index < dilations.size(); ++index) {
        auto hidden = ada_instance_norm(context, parameters, std::string(prefix) + ".adain1." + std::to_string(index),
                                        input, style);
        hidden = checked(context, std::move(hidden), std::string(prefix) + ".adain1." + std::to_string(index));
        hidden =
            snake(context, parameters, std::string(prefix) + ".alpha1." + std::to_string(index), std::move(hidden));
        hidden = checked(context, std::move(hidden), std::string(prefix) + ".snake1." + std::to_string(index));
        hidden = conv1d(context, parameters, std::string(prefix) + ".convs1." + std::to_string(index), hidden, channels,
                        kernel, 1, (kernel * dilations[index] - dilations[index]) / 2, dilations[index]);
        hidden = checked(context, std::move(hidden), std::string(prefix) + ".conv1." + std::to_string(index));
        hidden = ada_instance_norm(context, parameters, std::string(prefix) + ".adain2." + std::to_string(index),
                                   hidden, style);
        hidden = checked(context, std::move(hidden), std::string(prefix) + ".adain2." + std::to_string(index));
        hidden =
            snake(context, parameters, std::string(prefix) + ".alpha2." + std::to_string(index), std::move(hidden));
        hidden = checked(context, std::move(hidden), std::string(prefix) + ".snake2." + std::to_string(index));
        hidden = conv1d(context, parameters, std::string(prefix) + ".convs2." + std::to_string(index), hidden, channels,
                        kernel, 1, (kernel - 1) / 2);
        hidden = checked(context, std::move(hidden), std::string(prefix) + ".conv2." + std::to_string(index));
        input = add(context, input, hidden);
        input = checked(context, std::move(input), std::string(prefix) + ".residual." + std::to_string(index));
    }
    return input;
}

auto repeat_rows(ops::Context& context, const Tensor& input, std::span<const std::int32_t> durations) -> Tensor {
    const auto source = values(context, input);
    const auto channels = static_cast<std::size_t>(input.size(2));
    const auto total = std::accumulate(durations.begin(), durations.end(), std::size_t{0});
    std::vector<float> output(total * channels);
    std::size_t target = 0;
    for (std::size_t row = 0; row < durations.size(); ++row)
        for (std::int32_t repeat = 0; repeat < durations[row]; ++repeat) {
            std::ranges::copy(source.subspan(row * channels, channels), output.begin() + target * channels);
            ++target;
        }
    return from_values(total, channels, std::move(output));
}

auto stft(std::span<const float> signal, std::int32_t n_fft, std::int32_t hop) -> std::pair<Tensor, std::size_t> {
    const auto padding = n_fft / 2;
    std::vector<float> padded(signal.size() + 2 * padding);
    for (std::int32_t index = 0; index < padding; ++index) {
        padded[padding - 1 - index] = signal[std::min<std::size_t>(signal.size() - 1, index + 1)];
        padded[padding + signal.size() + index] =
            signal[signal.size() - 2 - std::min<std::size_t>(signal.size() - 2, index)];
    }
    std::ranges::copy(signal, padded.begin() + padding);
    const auto frames = (padded.size() - n_fft) / hop + 1;
    const auto bins = n_fft / 2 + 1;
    std::vector<float> output(frames * 2 * bins);
    for (std::size_t frame = 0; frame < frames; ++frame)
        for (std::int32_t bin = 0; bin < bins; ++bin) {
            std::complex<double> value{};
            for (std::int32_t sample = 0; sample < n_fft; ++sample) {
                const auto window = 0.5 - 0.5 * std::cos(2 * std::numbers::pi * sample / n_fft);
                const auto angle = -2 * std::numbers::pi * bin * sample / n_fft;
                value += padded[frame * hop + sample] * window * std::complex<double>(std::cos(angle), std::sin(angle));
            }
            output[frame * 2 * bins + bin] = static_cast<float>(std::abs(value));
            output[frame * 2 * bins + bins + bin] = static_cast<float>(std::arg(value));
        }
    return {from_values(frames, 2 * bins, std::move(output)), frames};
}

auto istft(ops::Context& context, const Tensor& spectrogram, std::int32_t n_fft, std::int32_t hop)
    -> std::vector<float> {
    const auto data = values(context, spectrogram);
    const auto frames = static_cast<std::size_t>(spectrogram.size(1));
    const auto bins = static_cast<std::size_t>(n_fft / 2 + 1);
    const auto length = (frames - 1) * hop + n_fft;
    std::vector<double> output(length), denominator(length);
    for (std::size_t frame = 0; frame < frames; ++frame) {
        std::vector<std::complex<double>> spectrum(n_fft);
        for (std::size_t bin = 0; bin < bins; ++bin) {
            const auto magnitude = data[frame * 2 * bins + bin];
            const auto phase = data[frame * 2 * bins + bins + bin];
            spectrum[bin] = std::polar(static_cast<double>(magnitude), static_cast<double>(phase));
            if (bin && bin + 1 < bins) spectrum[n_fft - bin] = std::conj(spectrum[bin]);
        }
        for (std::int32_t sample = 0; sample < n_fft; ++sample) {
            std::complex<double> value{};
            for (std::int32_t bin = 0; bin < n_fft; ++bin) {
                const auto angle = 2 * std::numbers::pi * bin * sample / n_fft;
                value += spectrum[bin] * std::complex<double>(std::cos(angle), std::sin(angle));
            }
            const auto window = 0.5 - 0.5 * std::cos(2 * std::numbers::pi * sample / n_fft);
            const auto index = frame * hop + sample;
            output[index] += value.real() / n_fft * window;
            denominator[index] += window * window;
        }
    }
    const auto trim = static_cast<std::size_t>(n_fft / 2);
    std::vector<float> result(length - 2 * trim);
    for (std::size_t index = trim; index < length - trim; ++index)
        result[index - trim] = denominator[index] > 1e-12 ? output[index] / denominator[index] : 0;
    return result;
}

} // namespace

struct KokoroImpl::State {
    YAML::Node config;
    ops::Context context{tensor::Device::cpu()};
    Parameters parameters;
    std::uint32_t sample_rate;
    std::optional<text::LexiconPhonemizer> phonemizer;
    std::string default_voice;

    State(YAML::Node value, checkpoint::Weights weights)
        : config(std::move(value)),
          parameters(std::move(weights), config["weight_group_size"].as<std::int32_t>()),
          sample_rate(config["sample_rate"].as<std::uint32_t>()) {
        parameters.validate(config);
    }

    auto albert(std::span<const std::int32_t> ids) -> Tensor {
        const auto length = ids.size();
        auto words = parameters.embedding("bert.module.embeddings.word_embeddings.weight", ids);
        std::vector<std::int32_t> positions(length), types(length, 0);
        std::iota(positions.begin(), positions.end(), 0);
        auto position = parameters.embedding("bert.module.embeddings.position_embeddings.weight", positions);
        auto type = parameters.embedding("bert.module.embeddings.token_type_embeddings.weight", types);
        auto hidden = add(context, add(context, words, position), type);
        hidden = layer_norm(context, parameters, hidden, "bert.module.embeddings.LayerNorm.weight",
                            "bert.module.embeddings.LayerNorm.bias", 1e-12F);
        hidden = parameters.linear(context, "bert.module.encoder.embedding_hidden_mapping_in", hidden);
        const auto prefix = "bert.module.encoder.albert_layer_groups.0.albert_layers.0";
        for (std::int32_t layer = 0; layer < 12; ++layer) {
            auto query = parameters.linear(context, std::string(prefix) + ".attention.query", hidden);
            auto key = parameters.linear(context, std::string(prefix) + ".attention.key", hidden);
            auto value = parameters.linear(context, std::string(prefix) + ".attention.value", hidden);
            auto attended = context.scaled_dot_product_attention(query, key, value, 12);
            attended = parameters.linear(context, std::string(prefix) + ".attention.dense", attended);
            hidden = layer_norm(context, parameters, add(context, hidden, attended),
                                std::string(prefix) + ".attention.LayerNorm.weight",
                                std::string(prefix) + ".attention.LayerNorm.bias", 1e-12F);
            auto feed_forward = parameters.linear(context, std::string(prefix) + ".ffn", hidden);
            feed_forward = context.gelu(feed_forward, true);
            feed_forward = parameters.linear(context, std::string(prefix) + ".ffn_output", feed_forward);
            hidden = layer_norm(context, parameters, add(context, hidden, feed_forward),
                                std::string(prefix) + ".full_layer_layer_norm.weight",
                                std::string(prefix) + ".full_layer_layer_norm.bias", 1e-12F);
            hidden = checked(context, std::move(hidden), "albert." + std::to_string(layer));
        }
        return hidden;
    }

    auto duration_encoder(const Tensor& input, std::span<const float> style) -> Tensor {
        auto hidden =
            concat(context, std::array{input, [&] {
                                           std::vector<float> repeated(input.size(1) * style.size());
                                           for (std::size_t row = 0; row < input.size(1); ++row)
                                               std::ranges::copy(style, repeated.begin() + row * style.size());
                                           return from_values(input.size(1), style.size(), std::move(repeated));
                                       }()});
        for (std::int32_t layer = 0; layer < 3; ++layer) {
            hidden = bilstm(context, parameters, "predictor.module.text_encoder.lstms." + std::to_string(2 * layer),
                            hidden, 256);
            hidden =
                ada_layer_norm(context, parameters,
                               "predictor.module.text_encoder.lstms." + std::to_string(2 * layer + 1), hidden, style);
            std::vector<float> repeated(hidden.size(1) * style.size());
            for (std::size_t row = 0; row < hidden.size(1); ++row)
                std::ranges::copy(style, repeated.begin() + row * style.size());
            hidden =
                concat(context, std::array{hidden, from_values(hidden.size(1), style.size(), std::move(repeated))});
        }
        return hidden;
    }

    auto text_encoder(std::span<const std::int32_t> ids) -> Tensor {
        auto hidden = parameters.embedding("text_encoder.module.embedding.weight", ids);
        for (std::int32_t layer = 0; layer < 3; ++layer) {
            const auto prefix = "text_encoder.module.cnn." + std::to_string(layer);
            hidden = conv1d(context, parameters, prefix + ".0", hidden, 512, 5, 1, 2);
            hidden = layer_norm(context, parameters, hidden, prefix + ".1.gamma", prefix + ".1.beta", 1e-5F);
            hidden = activate(context, std::move(hidden), 0.2F);
        }
        return bilstm(context, parameters, "text_encoder.module.lstm", hidden, 256);
    }

    auto prosody_branch(std::string_view name, const Tensor& shared, std::span<const float> style) -> Tensor {
        auto hidden = shared;
        hidden =
            adain_block(context, parameters, "predictor.module." + std::string(name) + ".0", hidden, style, 512, false);
        hidden =
            adain_block(context, parameters, "predictor.module." + std::string(name) + ".1", hidden, style, 256, true);
        hidden =
            adain_block(context, parameters, "predictor.module." + std::string(name) + ".2", hidden, style, 256, false);
        return conv1d(context, parameters, "predictor.module." + std::string(name) + "_proj", hidden, 1, 1);
    }

    auto decoder(const Tensor& asr, const Tensor& f0, const Tensor& noise, std::span<const float> style,
                 std::uint64_t seed) -> std::vector<float> {
        auto f0_down =
            checked(context, conv1d(context, parameters, "decoder.module.F0_conv", f0, 1, 3, 2, 1), "decoder.f0_down");
        auto noise_down = checked(context, conv1d(context, parameters, "decoder.module.N_conv", noise, 1, 3, 2, 1),
                                  "decoder.noise_down");
        auto hidden = concat(context, std::array{asr, f0_down, noise_down});
        hidden = checked(context, adain_block(context, parameters, "decoder.module.encode", hidden, style, 1024, false),
                         "decoder.encode");
        auto asr_res = conv1d(context, parameters, "decoder.module.asr_res.0", asr, 64, 1);
        for (std::int32_t block = 0; block < 4; ++block) {
            if (block < 4) hidden = concat(context, std::array{hidden, asr_res, f0_down, noise_down});
            hidden = adain_block(context, parameters, "decoder.module.decode." + std::to_string(block), hidden, style,
                                 block == 3 ? 512 : 1024, block == 3);
            hidden = checked(context, std::move(hidden), "decoder.block." + std::to_string(block));
        }
        return generator(hidden, f0, style, seed);
    }

    auto harmonic(const Tensor& f0, std::uint64_t seed) -> Tensor {
        auto upsampled = linear_interpolate(context, f0, static_cast<std::size_t>(f0.size(1)) * 300);
        const auto source = values(context, upsampled);
        const auto length = static_cast<std::size_t>(upsampled.size(1));
        std::mt19937_64 random(seed);
        std::normal_distribution<float> gaussian(0.F, 1.F);
        std::vector<float> harmonics(length * 9);
        std::array<double, 9> phase{};
        for (std::size_t time = 0; time < length; ++time)
            for (std::size_t harmonic = 0; harmonic < 9; ++harmonic) {
                const auto frequency = source[time] * static_cast<float>(harmonic + 1);
                phase[harmonic] += 2 * std::numbers::pi * frequency / sample_rate;
                const auto voiced = source[time] > 10.F;
                const auto noise_scale = voiced ? 0.003F : 0.1F / 3.F;
                harmonics[time * 9 + harmonic] =
                    (voiced ? 0.1F * std::sin(phase[harmonic]) : 0.F) + noise_scale * gaussian(random);
            }
        auto merged = parameters.linear(context, "decoder.module.generator.m_source.l_linear",
                                        from_values(length, 9, std::move(harmonics)));
        auto data = mutable_values(context, merged);
        for (auto& value : data) value = std::tanh(value);
        const auto waveform = values(context, merged);
        auto [result, frames] = stft(waveform, 20, 5);
        (void)frames;
        return result;
    }

    auto generator(Tensor hidden, const Tensor& f0, std::span<const float> style, std::uint64_t seed)
        -> std::vector<float> {
        auto source = checked(context, harmonic(f0, seed), "generator.harmonic");
        static constexpr std::array<std::int32_t, 2> rates{10, 6};
        static constexpr std::array<std::int32_t, 2> kernels{20, 12};
        static constexpr std::array<std::int32_t, 3> residual_kernels{3, 7, 11};
        static constexpr std::array<std::int32_t, 3> dilations{1, 3, 5};
        for (std::int32_t stage = 0; stage < 2; ++stage) {
            hidden = activate(context, std::move(hidden), 0.1F);
            Tensor source_hidden;
            if (stage == 0)
                source_hidden =
                    conv1d(context, parameters, "decoder.module.generator.noise_convs.0", source, 256, 12, 6, 3);
            else
                source_hidden = conv1d(context, parameters, "decoder.module.generator.noise_convs.1", source, 128, 1);
            source_hidden =
                adain_resblock(context, parameters, "decoder.module.generator.noise_res." + std::to_string(stage),
                               std::move(source_hidden), style, stage == 0 ? 7 : 11, dilations);
            source_hidden = checked(context, std::move(source_hidden), "generator.noise_res." + std::to_string(stage));
            hidden = conv_transpose1d(context, parameters, "decoder.module.generator.ups." + std::to_string(stage),
                                      hidden, stage == 0 ? 256 : 128, kernels[stage], rates[stage],
                                      (kernels[stage] - rates[stage]) / 2);
            hidden = checked(context, std::move(hidden), "generator.upsample." + std::to_string(stage));
            if (stage == 1) {
                const auto input = values(context, hidden);
                const auto time = static_cast<std::size_t>(hidden.size(1));
                const auto channels = static_cast<std::size_t>(hidden.size(2));
                std::vector<float> padded((time + 1) * channels);
                std::ranges::copy(input.subspan(channels, channels), padded.begin());
                std::ranges::copy(input, padded.begin() + channels);
                hidden = from_values(time + 1, channels, std::move(padded));
            }
            hidden = add(context, hidden, source_hidden);
            hidden = checked(context, std::move(hidden), "generator.source_add." + std::to_string(stage));
            std::array<Tensor, 3> residuals;
            for (std::int32_t block = 0; block < 3; ++block)
                residuals[block] = adain_resblock(
                    context, parameters, "decoder.module.generator.resblocks." + std::to_string(stage * 3 + block),
                    hidden, style, residual_kernels[block], dilations);
            hidden = add(context, add(context, residuals[0], residuals[1]), residuals[2]);
            auto data = mutable_values(context, hidden);
            for (auto& value : data) value /= 3.F;
            hidden = checked(context, std::move(hidden), "generator.resblocks." + std::to_string(stage));
        }
        hidden = activate(context, std::move(hidden), 0.1F);
        auto spectrum =
            checked(context, conv1d(context, parameters, "decoder.module.generator.conv_post", hidden, 22, 7, 1, 3),
                    "generator.conv_post");
        auto data = mutable_values(context, spectrum);
        float maximum_log_magnitude = 0;
        for (std::size_t row = 0; row < spectrum.size(1); ++row)
            for (std::size_t bin = 0; bin < 11; ++bin)
                maximum_log_magnitude = std::max(maximum_log_magnitude, data[row * 22 + bin]);
        if (!std::isfinite(maximum_log_magnitude) || maximum_log_magnitude > 80.F)
            throw ops::Failure({ErrorCode::RUNTIME,
                                "Kokoro spectrum log magnitude is invalid: " + std::to_string(maximum_log_magnitude)});
        for (std::size_t row = 0; row < spectrum.size(1); ++row)
            for (std::size_t bin = 0; bin < 11; ++bin) {
                data[row * 22 + bin] = std::exp(data[row * 22 + bin]);
                data[row * 22 + 11 + bin] = std::sin(data[row * 22 + 11 + bin]);
            }
        auto audio = istft(context, spectrum, 20, 5);
        float maximum = 0;
        for (const auto value : audio) {
            if (!std::isfinite(value))
                throw ops::Failure({ErrorCode::RUNTIME, "Kokoro iSTFT produced a non-finite sample"});
            maximum = std::max(maximum, std::abs(value));
        }
        if (maximum > 1e6F)
            throw ops::Failure({ErrorCode::RUNTIME, "Kokoro iSTFT exceeded range: " + std::to_string(maximum)});
        return audio;
    }

    auto run(std::span<const std::int32_t> raw_ids, std::string_view voice, float speed, std::uint64_t seed)
        -> KokoroOutput {
        if (raw_ids.empty() || raw_ids.size() > 510 || !std::isfinite(speed) || speed <= 0)
            throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "invalid Kokoro tokens or speed"});
        auto style = parameters.voice(voice, std::min<std::size_t>(raw_ids.size() - 1, 509));
        std::vector<std::int32_t> ids(raw_ids.size() + 2);
        std::ranges::copy(raw_ids, ids.begin() + 1);

        auto bert = checked(context, albert(ids), "bert");
        auto d_en = checked(context, parameters.linear(context, "bert_encoder.module", bert), "bert_encoder");
        const auto predictor_style = std::span<const float>(style).subspan(128, 128);
        auto duration_features = checked(context, duration_encoder(d_en, predictor_style), "duration_encoder");
        auto duration_hidden = checked(
            context, bilstm(context, parameters, "predictor.module.lstm", duration_features, 256), "duration_lstm");
        auto duration_logits =
            checked(context, parameters.linear(context, "predictor.module.duration_proj.linear_layer", duration_hidden),
                    "duration_projection");
        const auto logits = values(context, duration_logits);
        std::vector<std::int32_t> durations(ids.size());
        for (std::size_t row = 0; row < ids.size(); ++row) {
            float duration = 0;
            for (std::size_t value = 0; value < 50; ++value) duration += sigmoid(logits[row * 50 + value]);
            durations[row] = std::max<std::int32_t>(1, std::lround(duration / speed));
        }
        auto expanded_duration = repeat_rows(context, duration_features, durations);
        auto shared = checked(context, bilstm(context, parameters, "predictor.module.shared", expanded_duration, 256),
                              "shared_lstm");
        auto f0 = checked(context, prosody_branch("F0", shared, predictor_style), "f0");
        auto noise = checked(context, prosody_branch("N", shared, predictor_style), "noise");
        auto text = checked(context, text_encoder(ids), "text_encoder");
        auto asr = checked(context, repeat_rows(context, text, durations), "length_regulator");
        const auto decoder_style = std::span<const float>(style).first(128);
        auto samples = decoder(asr, f0, noise, decoder_style, seed);
        return {std::move(samples), std::move(durations)};
    }
};

auto KokoroImpl::validate_config(const YAML::Node& config) -> Result<void> {
    try {
        if (!config.IsMap() || config["type"].as<std::string>() != "kokoro" ||
            config["sample_rate"].as<std::uint32_t>() != 24000 || !config["config"]["vocab"].IsMap() ||
            !config["voices"].IsSequence() || !config["voices"].size() ||
            config["weight_group_size"].as<std::int32_t>() <= 0)
            return std::unexpected(Error{ErrorCode::INVALID_MANIFEST, "invalid Kokoro configuration"});
        return {};
    } catch (const std::exception& error) {
        return std::unexpected(Error{ErrorCode::INVALID_MANIFEST, error.what()});
    }
}

KokoroImpl::KokoroImpl(YAML::Node config, checkpoint::Weights weights) {
    auto valid = validate_config(config);
    if (!valid) throw ops::Failure(valid.error());
    impl_ = std::make_unique<State>(std::move(config), std::move(weights));
}

KokoroImpl::~KokoroImpl() = default;

auto KokoroImpl::load(YAML::Node config, checkpoint::Weights weights) -> Result<Kokoro> {
    auto valid = validate_config(config);
    if (!valid) return std::unexpected(valid.error());
    try {
        return Kokoro(std::move(config), std::move(weights));
    } catch (const ops::Failure& error) {
        return std::unexpected(error.error());
    } catch (const std::exception& error) {
        return std::unexpected(Error{ErrorCode::RUNTIME, error.what()});
    }
}

auto KokoroImpl::load(YAML::Node package, tensor::Device device) -> Result<Kokoro> {
    try {
        if (device != tensor::Device::cpu())
            throw ops::Failure({ErrorCode::UNSUPPORTED, "Kokoro currently supports the CPU only"});
        auto phonemizer = require(text::LexiconPhonemizer::load(package["lexicon_file"].as<std::string>(),
                                                                package["model"]["config"]["vocab"]));
        auto weights = require(checkpoint::Weights::load(package["weights_file"].as<std::string>()));
        auto model = require(load(package["model"], std::move(weights)));
        model->impl_->phonemizer.emplace(std::move(phonemizer));
        model->impl_->default_voice = package["synthesis"]["default_voice"].as<std::string>();
        return model;
    } catch (const ops::Failure& error) {
        return std::unexpected(error.error());
    } catch (const std::exception& error) {
        return std::unexpected(Error{ErrorCode::RUNTIME, error.what()});
    }
}

auto KokoroImpl::synthesize(std::span<const std::int32_t> phoneme_ids, std::string_view voice, float speed,
                            std::uint64_t seed) -> Result<KokoroOutput> {
    try {
        return impl_->run(phoneme_ids, voice, speed, seed);
    } catch (const ops::Failure& error) {
        return std::unexpected(error.error());
    } catch (const std::exception& error) {
        return std::unexpected(Error{ErrorCode::RUNTIME, error.what()});
    }
}

auto KokoroImpl::synthesize(std::string_view text, const inference::SynthesisOptions& options)
    -> Result<inference::Synthesis> {
    try {
        if (!impl_->phonemizer)
            throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "Kokoro was not loaded as a synthesis model"});
        if (options.duration_seconds)
            throw ops::Failure({ErrorCode::UNSUPPORTED, "Kokoro predicts duration; use --duration auto"});
        auto language = options.language;
        std::ranges::transform(language, language.begin(),
                               [](unsigned char value) { return static_cast<char>(std::tolower(value)); });
        if (!language.empty() && language != "auto" && language != "english" && language != "en" && language != "en-us")
            throw ops::Failure({ErrorCode::UNSUPPORTED, "this Kokoro package supports US English only"});

        auto voice = impl_->default_voice;
        float speed = 1.F;
        for (const auto& [key, value] : options.voice) {
            if (key == "name") {
                voice = value;
            } else if (key == "speed") {
                auto parsed = inference::parse_synthesis_number(value);
                if (!parsed || *parsed <= 0)
                    throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "Kokoro speed must be positive and finite"});
                speed = *parsed;
            } else {
                throw ops::Failure({ErrorCode::UNSUPPORTED, "Kokoro does not support voice attribute: " + key});
            }
        }

        auto phonemes = require(impl_->phonemizer->phonemize(text));
        const auto started = std::chrono::steady_clock::now();
        auto output = require(synthesize(phonemes.ids, voice, speed, options.seed));
        const auto generation_ns = inference::synthesis_elapsed_ns(started);
        const auto peak = std::ranges::max(output.samples, {}, [](float value) { return std::abs(value); });
        if (std::abs(peak) > 0.3F) {
            const auto scale = 0.3F / std::abs(peak);
            for (auto& sample : output.samples) sample *= scale;
        }
        const auto frames = std::accumulate(output.durations.begin(), output.durations.end(), std::size_t{0});
        return inference::Synthesis{
            .samples = std::move(output.samples),
            .sample_rate = sample_rate(),
            .tokens = std::move(phonemes.ids),
            .stats =
                {
                    .generation_ns = generation_ns,
                    .decode_ns = 0,
                    .preparation_ns = preparation_ns(),
                    .audio_tokens = frames,
                },
        };
    } catch (const ops::Failure& error) {
        return std::unexpected(error.error());
    } catch (const std::exception& error) {
        return std::unexpected(Error{ErrorCode::RUNTIME, error.what()});
    }
}

auto KokoroImpl::sample_rate() const noexcept -> std::uint32_t { return impl_->sample_rate; }
auto KokoroImpl::preparation_ns() const noexcept -> std::uint64_t { return impl_->context.preparation_ns(); }

} // namespace kidi::model
