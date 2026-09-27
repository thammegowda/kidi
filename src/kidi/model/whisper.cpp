#include "kidi/model/whisper.h"

#include <cmath>

namespace kidi::model {
using ops::require;
using tensor::DType;
using tensor::Tensor;

struct WhisperImpl::State {
    ops::Context context;
    std::int32_t hidden, encoder_layers, decoder_layers, maximum_target, vocabulary;
    layers::WhisperEncoder encoder;
    layers::WhisperDecoder decoder;
    layers::Linear output;

    explicit State(const YAML::Node& config)
        : context(module_device),
          hidden(config["d_model"].as<int>()),
          encoder_layers(config["encoder_layers"].as<int>()),
          decoder_layers(config["decoder_layers"].as<int>()),
          maximum_target(config["max_target_positions"].as<int>()),
          vocabulary(config["vocab_size"].as<int>()),
          encoder(config["num_mel_bins"].as<int>(), hidden, config["encoder_ffn_dim"].as<int>(),
                  config["encoder_attention_heads"].as<int>(), encoder_layers, config["max_source_positions"].as<int>(),
                  config["layer_norm_eps"].as<float>(1e-5F)),
          decoder(vocabulary, hidden, config["decoder_ffn_dim"].as<int>(), config["decoder_attention_heads"].as<int>(),
                  decoder_layers, maximum_target, config["layer_norm_eps"].as<float>(1e-5F)),
          output(hidden, vocabulary, true, false) {}
};

auto WhisperImpl::validate_config(const YAML::Node& config) -> Result<void> {
    try {
        if (!config.IsMap() || config["type"].as<std::string>() != "whisper" ||
            config["model_type"].as<std::string>() != "whisper" || config["architectures"].size() != 1 ||
            config["architectures"][0].as<std::string>() != "WhisperForConditionalGeneration" ||
            config["activation_function"].as<std::string>() != "gelu" || config["scale_embedding"].as<bool>() ||
            !config["is_encoder_decoder"].as<bool>() || !config["use_cache"].as<bool>())
            return std::unexpected(Error{ErrorCode::UNSUPPORTED, "unsupported Whisper architecture"});
        for (const auto* name :
             {"activation_dropout", "attention_dropout", "decoder_layerdrop", "dropout", "encoder_layerdrop"})
            if (config[name].as<float>() != 0.F)
                return std::unexpected(Error{ErrorCode::UNSUPPORTED, "Whisper dropout must be disabled"});
        for (const auto* name : {"d_model", "encoder_attention_heads", "encoder_ffn_dim", "encoder_layers",
                                 "decoder_attention_heads", "decoder_ffn_dim", "decoder_layers", "max_source_positions",
                                 "max_target_positions", "num_mel_bins", "vocab_size"})
            if (config[name].as<int>() <= 0)
                return std::unexpected(
                    Error{ErrorCode::INVALID_MANIFEST, std::string("invalid Whisper dimension: ") + name});
        const auto hidden = config["d_model"].as<int>();
        if (hidden % config["encoder_attention_heads"].as<int>() ||
            hidden % config["decoder_attention_heads"].as<int>())
            return std::unexpected(Error{ErrorCode::INVALID_MANIFEST, "invalid Whisper attention dimensions"});
        const auto encoder_layers = config["encoder_layers"].as<int>();
        const auto decoder_layers = config["decoder_layers"].as<int>();
        const auto encoder_heads = config["encoder_attention_heads"].as<int>();
        const auto decoder_heads = config["decoder_attention_heads"].as<int>();
        const auto encoder_feed_forward = config["encoder_ffn_dim"].as<int>();
        const auto decoder_feed_forward = config["decoder_ffn_dim"].as<int>();
        const bool supported_size =
            (hidden == 384 && encoder_layers == 4 && decoder_layers == 4 && encoder_heads == 6 && decoder_heads == 6 &&
             encoder_feed_forward == 1536 && decoder_feed_forward == 1536) ||
            (hidden == 512 && encoder_layers == 6 && decoder_layers == 6 && encoder_heads == 8 && decoder_heads == 8 &&
             encoder_feed_forward == 2048 && decoder_feed_forward == 2048) ||
            (hidden == 768 && encoder_layers == 12 && decoder_layers == 12 && encoder_heads == 12 &&
             decoder_heads == 12 && encoder_feed_forward == 3072 && decoder_feed_forward == 3072);
        if (!supported_size || config["num_mel_bins"].as<int>() != 80 ||
            config["max_source_positions"].as<int>() != 1500 || config["max_target_positions"].as<int>() != 448 ||
            config["vocab_size"].as<int>() != 51865)
            return std::unexpected(
                Error{ErrorCode::UNSUPPORTED, "supported multilingual Whisper sizes are Tiny, Base, and Small"});
        return {};
    } catch (const YAML::Exception& error) {
        return std::unexpected(
            Error{ErrorCode::INVALID_MANIFEST, "invalid Whisper config: " + std::string(error.what())});
    }
}

WhisperImpl::WhisperImpl(const YAML::Node& config) {
    require(validate_config(config));
    impl_ = std::make_unique<State>(config);
    register_module("encoder", impl_->encoder);
    register_module("decoder", impl_->decoder);
    register_module("proj_out", impl_->output);
    tie_parameter("proj_out.weight", "decoder.embed_tokens.weight");
}

WhisperImpl::~WhisperImpl() = default;

auto WhisperImpl::create(const YAML::Node& config) -> Result<Whisper> {
    try {
        return Whisper(config);
    } catch (const ops::Failure& error) {
        return std::unexpected(error.error());
    } catch (const std::exception& error) {
        return std::unexpected(Error{ErrorCode::RUNTIME, error.what()});
    }
}

auto WhisperImpl::set_checkpoint(const Weights& weights) -> Result<void> {
    try {
        const auto checkpoint = require(weights.state_dict());
        StateDict state;
        for (const auto& [key, value] : checkpoint) {
            constexpr std::string_view PREFIX = "model.";
            if (!key.starts_with(PREFIX))
                return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "unknown Whisper parameter: " + key});
            state.emplace(key.substr(PREFIX.size()), value);
        }
        require(set_state(state));
        return {};
    } catch (const ops::Failure& error) {
        return std::unexpected(error.error());
    }
}

auto WhisperImpl::encode(const audio::WhisperFeatures& features) -> Result<WhisperEncoderState> {
    try {
        if (device() != tensor::Device::cpu() || features.bins != 80 || features.frames != 3000 ||
            features.values.size() != features.bins * features.frames)
            throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "Whisper requires 80 x 3000 CPU features"});
        std::vector<float> time_major(features.values.size());
        for (std::size_t frame = 0; frame < features.frames; ++frame)
            for (std::size_t bin = 0; bin < features.bins; ++bin)
                time_major[frame * features.bins + bin] = features.values[bin * features.frames + frame];
        auto input = require(
            Tensor::from_host({1, static_cast<std::int64_t>(features.frames), static_cast<std::int64_t>(features.bins)},
                              std::span<const float>(time_major), device()));
        auto convolution = impl_->encoder->convolve(impl_->context, input);
        auto hidden = impl_->encoder->encode(impl_->context, convolution);
        WhisperEncoderState result{convolution, hidden, impl_->decoder->project_source(impl_->context, hidden)};
        impl_->context.synchronize();
        return result;
    } catch (const ops::Failure& error) {
        return std::unexpected(error.error());
    }
}

auto WhisperImpl::create_state(std::size_t capacity) -> Result<WhisperDecoderState> {
    try {
        if (!capacity || capacity > static_cast<std::size_t>(impl_->maximum_target))
            throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "invalid Whisper decoder capacity"});
        WhisperDecoderState result;
        result.capacity = capacity;
        result.mask = require(Tensor::empty({1, 1, 1, static_cast<std::int64_t>(capacity)}, DType::F32, device()));
        std::ranges::fill(require(result.mask.data<float>()), -1e9F);
        result.index = require(Tensor::empty({1}, DType::I32, device()));
        const std::vector<std::int64_t> shape{1, static_cast<std::int64_t>(capacity), impl_->hidden};
        for (std::int32_t layer = 0; layer < impl_->decoder_layers; ++layer)
            result.layers.push_back({require(Tensor::zeros(shape, DType::F32, device())),
                                     require(Tensor::zeros(shape, DType::F32, device()))});
        return result;
    } catch (const ops::Failure& error) {
        return std::unexpected(error.error());
    }
}

auto WhisperImpl::forward(const WhisperEncoderState& source, std::span<const std::int32_t> tokens,
                          WhisperDecoderState& state) -> Result<Tensor> {
    try {
        if (tokens.size() != 1 || state.position >= state.capacity ||
            source.layers.size() != static_cast<std::size_t>(impl_->decoder_layers) ||
            state.layers.size() != static_cast<std::size_t>(impl_->decoder_layers))
            throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "invalid Whisper decoder input"});
        require(state.mask.data<float>())[state.position] = 0.F;
        require(state.index.data<std::int32_t>())[0] = static_cast<std::int32_t>(state.position);
        auto hidden = impl_->decoder->forward(impl_->context, tokens, state.position, source.layers, state.mask,
                                              state.layers, state.index);
        auto logits = impl_->output->forward(impl_->context, hidden);
        impl_->context.synchronize();
        ++state.position;
        return logits;
    } catch (const ops::Failure& error) {
        return std::unexpected(error.error());
    }
}

auto WhisperImpl::preparation_ns() const -> std::uint64_t { return impl_->context.preparation_ns(); }

} // namespace kidi::model