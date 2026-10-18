#include "kidi/model/qwen3.h"

#include <cmath>
#include <iterator>
#include <numeric>

namespace kidi::model {
using ops::require;

namespace {
template <typename Layers, typename Step>
auto replay_regions(ops::Context& context, std::string_view key, std::span<const tensor::Tensor> inputs,
                    const Layers& layers, const Step& step) -> tensor::Tensor {
    constexpr std::size_t region_layers = 4;
    const auto count = static_cast<std::size_t>(std::distance(layers.begin(), layers.end()));
    if (context.device() != tensor::Device::web_gpu() || count <= region_layers)
        return context.replay(key, inputs, [&](std::span<const tensor::Tensor> operands) {
            return std::vector{step(operands, layers.begin(), layers.end(), true)};
        }, false).front();
    context.set_replay_capacity((count + region_layers - 1) / region_layers);
    auto hidden = inputs.front();
    std::size_t region = 0;
    for (auto first = layers.begin(); first != layers.end(); ++region) {
        auto last = first;
        for (std::size_t index = 0; index < region_layers && last != layers.end(); ++index) ++last;
        std::vector<tensor::Tensor> operands(inputs.begin(), inputs.end());
        operands.front() = hidden;
        hidden = context.replay(std::string(key) + ':' + std::to_string(region), operands,
            [&](std::span<const tensor::Tensor> values) {
                return std::vector{step(values, first, last, last == layers.end())};
            }, false).front();
        first = last;
    }
    return hidden;
}
} // namespace

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
        const auto packed_bits =
            config["activation_quantization"] ? config["activation_quantization"]["packed_bits"].as<std::int32_t>() : 0;
        if (vocabulary <= 0 || hidden <= 0 || intermediate <= 0 || layers <= 0 || heads <= 0 || key_heads <= 0 ||
            heads % key_heads || head_width <= 0 || head_width % 2 || maximum <= 0 || !std::isfinite(epsilon) ||
            epsilon <= 0 || !std::isfinite(theta) || theta <= 0 || (packed_bits != 0 && packed_bits != 8))
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
    if (module_dtype != tensor::DType::I8 ||
        (module_device != tensor::Device::cpu() && module_device != tensor::Device::vulkan() &&
         module_device != tensor::Device::apple_gpu() && module_device != tensor::Device::web_gpu()))
        throw ops::Failure({ErrorCode::UNSUPPORTED, "Qwen3 requires INT8 parameters on CPU, Metal, Vulkan, or WebGPU"});
    const layers::Qwen3Shape shape{
        .hidden = hidden_,
        .intermediate = config["intermediate_size"].as<std::int32_t>(),
        .heads = heads_,
        .key_value_heads = key_value_heads_,
        .head_width = head_width_,
        .packed_bits =
            config["activation_quantization"] ? config["activation_quantization"]["packed_bits"].as<std::int32_t>() : 0,
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

auto Qwen3Impl::prepare_attention(std::size_t length) const -> const AttentionState& {
    if (!length || length > static_cast<std::size_t>(maximum_position_))
        throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "Qwen3 sequence exceeds positional capacity"});
    if (const auto found = attention_.find(length); found != attention_.end()) {
        found->second.used = ++attention_clock_;
        return found->second;
    }
    constexpr std::size_t CAPACITY = 4;
    if (attention_.size() >= CAPACITY)
        attention_.erase(std::ranges::min_element(attention_, {}, [](const auto& item) { return item.second.used; }));
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
    AttentionState state{
        .cosine = require(
            tensor::Tensor::from_host({1, static_cast<std::int64_t>(length), 1, static_cast<std::int64_t>(half)},
                                      std::span<const float>(cosine), device())),
        .sine = require(
            tensor::Tensor::from_host({1, static_cast<std::int64_t>(length), 1, static_cast<std::int64_t>(half)},
                                      std::span<const float>(sine), device())),
        .mask =
            require(tensor::Tensor::zeros({1, 1, static_cast<std::int64_t>(length), static_cast<std::int64_t>(length)},
                                          tensor::DType::F32, device())),
        .used = ++attention_clock_,
    };
    return attention_.emplace(length, std::move(state)).first->second;
}

auto Qwen3Impl::forward(ops::Context& context, const tensor::Tensor& embeddings) const -> tensor::Tensor {
    if (embeddings.dtype() != tensor::DType::F32 || embeddings.device() != context.device() ||
        embeddings.dimensions() != 3 || !embeddings.size(0) || embeddings.size(2) != hidden_)
        throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "Qwen3 requires FP32 [batch, sequence, hidden] embeddings"});
    const auto& attention = prepare_attention(embeddings.size(1));
    return forward(context, embeddings, attention.mask);
}

auto Qwen3Impl::forward(ops::Context& context, const tensor::Tensor& embeddings, const tensor::Tensor& mask) const
    -> tensor::Tensor {
    if (embeddings.dtype() != tensor::DType::F32 || embeddings.device() != context.device() ||
        embeddings.dimensions() != 3 || !embeddings.size(0) || embeddings.size(2) != hidden_)
        throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "Qwen3 requires FP32 [batch, sequence, hidden] embeddings"});
    const auto& attention = prepare_attention(embeddings.size(1));
    if (mask.dtype() != tensor::DType::F32 || mask.device() != context.device() || mask.dimensions() != 4 ||
        (mask.size(0) != 1 && mask.size(0) != embeddings.size(0)) || mask.size(1) != 1 ||
        mask.size(2) != embeddings.size(1) || mask.size(3) != embeddings.size(1))
        throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "Qwen3 requires a broadcastable FP32 attention mask"});
    const auto key = "qwen3:" + std::to_string(embeddings.size(0)) + ':' + std::to_string(embeddings.size(1));
    const std::array inputs{embeddings, mask};
    return replay_regions(context, key, inputs, *blocks_,
        [&](std::span<const tensor::Tensor> operands, auto first, auto last, bool final) {
            auto hidden = operands[0];
            bool consume = false;
            for (auto block = first; block != last; ++block) {
                hidden =
                    (*block)->forward(context, std::move(hidden), operands[1], attention.cosine, attention.sine, consume);
                consume = true;
            }
            return final ? norm_->forward(context, hidden) : hidden;
        });
}

auto Qwen3Impl::forward_segmented(ops::Context& context, const tensor::Tensor& embeddings,
                                  std::span<const std::size_t> segment_lengths) const -> tensor::Tensor {
    if (embeddings.dtype() != tensor::DType::F32 || embeddings.device() != context.device() ||
        embeddings.dimensions() != 3 || embeddings.size(0) != 1 || embeddings.size(2) != hidden_ ||
        segment_lengths.empty() ||
        std::ranges::any_of(
            segment_lengths,
            [&](auto length) { return !length || length > static_cast<std::size_t>(maximum_position_); }) ||
        std::accumulate(segment_lengths.begin(), segment_lengths.end(), std::size_t{0}) != embeddings.size(1))
        throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "Qwen3 segmented input dimensions are invalid"});

    std::vector<layers::Qwen3AttentionSegment> segments;
    segments.reserve(segment_lengths.size());
    std::size_t offset = 0;
    std::string key = "qwen3-segmented:1";
    for (const auto length : segment_lengths) {
        const auto& attention = prepare_attention(length);
        segments.push_back({offset, length, &attention.mask, &attention.cosine, &attention.sine});
        offset += length;
        key += ':' + std::to_string(length);
    }
    const std::array inputs{embeddings};
    return replay_regions(context, key, inputs, *blocks_,
        [&](std::span<const tensor::Tensor> operands, auto first, auto last, bool final) {
            auto hidden = operands[0];
            bool consume = false;
            for (auto block = first; block != last; ++block) {
                hidden = (*block)->forward(context, std::move(hidden), segments, consume);
                consume = true;
            }
            return final ? norm_->forward(context, hidden) : hidden;
        });
}

} // namespace kidi::model
