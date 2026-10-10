#include "kidi/model/qwen3.h"

#include <cmath>

namespace kidi::model {
using ops::require;

auto Qwen3Impl::validate_config(const YAML::Node& config) -> Result<void> {
    try {
        if (!config.IsMap() || config["model_type"].as<std::string>() != "qwen3" ||
            config["hidden_act"].as<std::string>() != "silu")
            return std::unexpected(Error{ErrorCode::INVALID_MANIFEST, "expected a Qwen3 SiLU configuration"});
        const auto vocabulary = config["vocab_size"].as<std::int32_t>();
        const auto hidden = config["hidden_size"].as<std::int32_t>();
        const auto intermediate = config["intermediate_size"].as<std::int32_t>();
        const auto layers = config["num_hidden_layers"].as<std::int32_t>();
        const auto heads = config["num_attention_heads"].as<std::int32_t>();
        const auto key_heads = config["num_key_value_heads"].as<std::int32_t>();
        const auto head_width = config["head_dim"].as<std::int32_t>();
        const auto maximum = config["max_position_embeddings"].as<std::int32_t>();
        const auto epsilon = config["rms_norm_eps"].as<float>();
        const auto theta = config["rope_parameters"]["rope_theta"].as<float>();
        if (vocabulary <= 0 || hidden <= 0 || intermediate <= 0 || layers <= 0 || heads <= 0 || key_heads <= 0 ||
            heads % key_heads || head_width <= 0 || head_width % 2 || maximum <= 0 || !std::isfinite(epsilon) ||
            epsilon <= 0 || !std::isfinite(theta) || theta <= 0)
            return std::unexpected(Error{ErrorCode::INVALID_MANIFEST, "invalid Qwen3 dimensions"});
        for (const auto& layer : config["layer_types"])
            if (layer.as<std::string>() != "full_attention")
                return std::unexpected(
                    Error{ErrorCode::UNSUPPORTED, "Kidi's Qwen3 prototype supports full-attention layers only"});
        return {};
    } catch (const std::exception& error) {
        return std::unexpected(Error{ErrorCode::INVALID_MANIFEST, error.what()});
    }
}

Qwen3Impl::Qwen3Impl(const YAML::Node& config)
    : vocabulary_(config["vocab_size"].as<std::int32_t>()),
      hidden_(config["hidden_size"].as<std::int32_t>()),
      layers_(config["num_hidden_layers"].as<std::int32_t>()),
      heads_(config["num_attention_heads"].as<std::int32_t>()),
      key_value_heads_(config["num_key_value_heads"].as<std::int32_t>()),
      head_width_(config["head_dim"].as<std::int32_t>()),
      maximum_position_(config["max_position_embeddings"].as<std::int32_t>()),
      rope_theta_(config["rope_parameters"]["rope_theta"].as<float>()),
      tokens_(vocabulary_, hidden_, 1.F),
      norm_(hidden_, config["rms_norm_eps"].as<float>(), true, true) {
    const auto valid = validate_config(config);
    if (!valid) throw ops::Failure(valid.error());
    if (module_dtype != tensor::DType::I8 || module_device != tensor::Device::cpu())
        throw ops::Failure({ErrorCode::UNSUPPORTED, "Qwen3 prototype requires INT8 parameters on the CPU"});
    const layers::Qwen3Shape shape{
        .hidden = hidden_,
        .intermediate = config["intermediate_size"].as<std::int32_t>(),
        .heads = heads_,
        .key_value_heads = key_value_heads_,
        .head_width = head_width_,
        .epsilon = config["rms_norm_eps"].as<float>(),
    };
    for (std::int32_t index = 0; index < layers_; ++index) blocks_->push_back(layers::Qwen3Block(shape));
    register_module("embed_tokens", tokens_);
    register_module("layers", blocks_);
    register_module("norm", norm_);
}

auto Qwen3Impl::create(const YAML::Node& config) -> Result<Qwen3> {
    auto valid = validate_config(config);
    if (!valid) return std::unexpected(std::move(valid.error()));
    try {
        return Qwen3(config);
    } catch (const ops::Failure& error) {
        return std::unexpected(error.error());
    } catch (const std::exception& error) {
        return std::unexpected(Error{ErrorCode::RUNTIME, error.what()});
    }
}

auto Qwen3Impl::embed(ops::Context& context, std::span<const std::int32_t> tokens) const -> tensor::Tensor {
    return tokens_->forward(context, tokens);
}

auto Qwen3Impl::prepare_attention(std::size_t length) const -> void {
    if (length == attention_length_) return;
    if (!length || length > static_cast<std::size_t>(maximum_position_))
        throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "Qwen3 sequence exceeds positional capacity"});
    const auto half = static_cast<std::size_t>(head_width_ / 2);
    std::vector<float> cosine(length * half), sine(length * half);
    for (std::size_t position = 0; position < length; ++position)
        for (std::size_t index = 0; index < half; ++index) {
            const auto frequency =
                1.F / std::pow(rope_theta_, static_cast<float>(2 * index) / static_cast<float>(head_width_));
            const auto angle = static_cast<float>(position) * frequency;
            cosine[position * half + index] = std::cos(angle);
            sine[position * half + index] = std::sin(angle);
        }
    cosine_ = require(tensor::Tensor::from_host(
        {1, static_cast<std::int64_t>(length), 1, static_cast<std::int64_t>(half)}, std::span<const float>(cosine)));
    sine_ = require(tensor::Tensor::from_host(
        {1, static_cast<std::int64_t>(length), 1, static_cast<std::int64_t>(half)}, std::span<const float>(sine)));
    mask_ = require(tensor::Tensor::zeros({1, 1, static_cast<std::int64_t>(length), static_cast<std::int64_t>(length)},
                                          tensor::DType::F32));
    attention_length_ = length;
}

auto Qwen3Impl::forward(ops::Context& context, const tensor::Tensor& embeddings) const -> tensor::Tensor {
    if (embeddings.dtype() != tensor::DType::F32 || embeddings.device() != tensor::Device::cpu() ||
        embeddings.dimensions() != 3 || embeddings.size(0) != 1 || embeddings.size(2) != hidden_)
        throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "Qwen3 requires FP32 [1, sequence, hidden] embeddings"});
    prepare_attention(embeddings.size(1));
    auto hidden = embeddings;
    for (const auto& block : *blocks_) hidden = block->forward(context, hidden, mask_, cosine_, sine_);
    return norm_->forward(context, hidden);
}

} // namespace kidi::model
