#include "kidi/model/whisper.h"
#include "kidi/ops/quantization.h"

#include <cmath>
#include <utility>

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
    DType precision = module_dtype;
    std::vector<Tensor> step_inputs;
    std::string decode_key;

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
    if (impl_->precision == DType::I8) tie_parameter("proj_out.scale", "decoder.embed_tokens.scale");
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

auto WhisperImpl::checkpoint_config() -> checkpoint::ConfigAdapter {
    return {
        "config.json", [](const checkpoint::ConfigSource& source) -> Result<YAML::Node> {
            const auto& model = source.document;
            if (!model.IsMap() || model["model_type"].as<std::string>() != "whisper")
                return std::unexpected(Error{ErrorCode::UNSUPPORTED, "checkpoint is not a Hugging Face Whisper model"});
            YAML::Node config;
            config["model"] = YAML::Clone(model);
            config["model"]["type"] = "whisper";
            constexpr std::array<std::string_view, 2> candidates{"model.safetensors", "ggml-model.bin"};
            const auto weights = require(source.weights(candidates));
            config["model_file"] = weights["path"];
            if (weights["format"].as<std::string>() == "ggml") {
                constexpr std::array keys{
                    "vocab_size",     "max_source_positions", "d_model", "encoder_attention_heads",
                    "encoder_layers", "max_target_positions", "d_model", "decoder_attention_heads",
                    "decoder_layers", "num_mel_bins"};
                for (std::size_t index = 0; index < keys.size(); ++index)
                    if (model[keys[index]].as<std::int32_t>() != weights["ggml_header"][index].as<std::int32_t>())
                        return std::unexpected(
                            Error{ErrorCode::INVALID_MANIFEST,
                                  "Whisper GGML header disagrees with config: " + std::string(keys[index])});
                config["weights_format"] = "whisper_ggml";
            } else if (weights["format"].as<std::string>() == "gguf") {
                return std::unexpected(Error{ErrorCode::UNSUPPORTED,
                                             "Whisper GGUF model mapping is not supported; use legacy Whisper GGML"});
            }
            for (const auto* name : {"tokenizer.json", "preprocessor_config.json", "generation_config.json"})
                config[std::filesystem::path(name).stem().string() + "_file"] = require(source.file(name)).string();
            return config;
        }};
}

auto WhisperImpl::int8_preparation(const YAML::Node& config) -> Result<checkpoint::Preparation> {
    auto valid = validate_config(config["model"]);
    if (!valid) return std::unexpected(valid.error());
    const auto source = std::filesystem::path(config["model_file"].as<std::string>());
    const bool imported = config["weights_format"].as<std::string>("") == "whisper_ggml";
    return checkpoint::Preparation{
        .source = source,
        .cache_directory = imported ? source.filename().string() + ".kidi-int8-v1" : "kidi-int8-v2",
        .format = imported ? "kidi-whisper-ggml-int8-v1" : "kidi-whisper-int8-v2",
        .files = {"config.json", "tokenizer.json", "preprocessor_config.json", "generation_config.json"},
        .metadata = {{"weight_precision", "signed-int8-per-output-channel"},
                     {"fp32_parameters", "input convolution, normalization, positions, biases, quantization scales"}},
        .convert = int8_checkpoint,
    };
}

auto WhisperImpl::int8_checkpoint(const YAML::Node& config, const checkpoint::Weights& weights) -> Result<StateDict> {
    try {
        const ModuleScope construction(DType::I8, false, tensor::Device::cpu());
        auto model = require(create(config));
        require(model->set_checkpoint(weights));
        StateDict result;
        for (const auto& [name, value] : model->state_dict())
            if (!name.starts_with("proj_out.")) result.emplace("model." + name, value);
        return result;
    } catch (const ops::Failure& error) {
        return std::unexpected(error.error());
    }
}

auto WhisperImpl::set_checkpoint(const checkpoint::Weights& weights) -> Result<void> {
    try {
        StateDict state;
        for (const auto& key : weights.names()) {
            const auto value = require(weights.tensor(key));
            constexpr std::string_view PREFIX = "model.";
            if (!key.starts_with(PREFIX))
                return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "unknown Whisper parameter: " + key});
            const auto name = key.substr(PREFIX.size());
            const bool projection =
                name.ends_with(".weight") &&
                (name.starts_with("encoder.conv2") || name.ends_with("_proj.weight") || name.ends_with(".fc1.weight") ||
                 name.ends_with(".fc2.weight") || name == "decoder.embed_tokens.weight");
            if (impl_->precision == DType::I8 && projection && value.dtype() == DType::F32) {
                const auto rows = static_cast<std::int64_t>(value.size(0));
                const auto width = static_cast<std::int64_t>(value.numel() / value.size(0));
                const auto matrix = require(value.reshape({rows, width}));
                const auto packed = require(ops::pack_weight(matrix, 8, static_cast<std::int32_t>(width)));
                const auto bytes = require(packed.values.host_bytes());
                auto quantized = require(
                    Tensor::from_host(std::vector<std::int64_t>(value.shape().begin(), value.shape().end()),
                                      std::span(reinterpret_cast<const std::int8_t*>(bytes.data()), bytes.size()),
                                      tensor::Device::cpu()));
                state.emplace(name, std::move(quantized));
                state.emplace(name.substr(0, name.size() - 7) + ".scale", packed.scales);
            } else {
                state.emplace(name, value);
            }
        }
        for (const auto& [name, value] : state) {
            if (impl_->precision == DType::I8 && name.ends_with(".scale")) {
                if (value.dtype() != DType::F32)
                    throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "Whisper INT8 scale must be FP32: " + name});
                for (const auto scale : require(value.data<float>()))
                    if (!std::isfinite(scale) || scale <= 0)
                        throw ops::Failure(
                            {ErrorCode::INVALID_ARGUMENT, "Whisper INT8 scale must be positive and finite: " + name});
            }
        }
        require(set_state(state));
        return {};
    } catch (const ops::Failure& error) {
        return std::unexpected(error.error());
    }
}

auto WhisperImpl::encode(const audio::WhisperFeatures& features, std::size_t frames,
                         const std::function<bool()>& cancelled) -> Result<WhisperEncoderState> {
    try {
        if (!frames) frames = features.frames;
        if (device() != tensor::Device::cpu() || features.bins != 80 || features.frames != 3000 ||
            features.values.size() != features.bins * features.frames || frames % 2 || frames > features.frames)
            throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "Whisper requires 80 x 3000 CPU features"});
        auto input = require(Tensor::empty(
            {1, static_cast<std::int64_t>(frames), static_cast<std::int64_t>(features.bins)}, DType::F32, device()));
        auto time_major = require(input.data<float>());
        for (std::size_t frame = 0; frame < frames; ++frame)
            for (std::size_t bin = 0; bin < features.bins; ++bin)
                time_major[frame * features.bins + bin] = features.values[bin * features.frames + frame];
        auto convolution = impl_->encoder->convolve(impl_->context, input);
        auto hidden = impl_->encoder->encode(impl_->context, convolution, cancelled);
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
        result.tokens = require(Tensor::empty({1}, DType::I32, device()));
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
    if (tokens.size() != 1)
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "Whisper forward expects one token"});
    return decode(source, tokens.front(), state, true);
}

auto WhisperImpl::prefill(const WhisperEncoderState& source, std::span<const std::int32_t> tokens,
                          WhisperDecoderState& state) -> Result<void> {
    if (tokens.empty() || state.position > state.capacity || tokens.size() > state.capacity - state.position)
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "invalid Whisper prefix length"});
    for (const auto token : tokens) {
        auto hidden = decode(source, token, state, false);
        if (!hidden) return std::unexpected(hidden.error());
    }
    return {};
}

auto WhisperImpl::decode(const WhisperEncoderState& source, std::int32_t token, WhisperDecoderState& state,
                         bool project) -> Result<Tensor> {
    try {
        if (state.position >= state.capacity ||
            source.layers.size() != static_cast<std::size_t>(impl_->decoder_layers) ||
            state.layers.size() != static_cast<std::size_t>(impl_->decoder_layers))
            throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "invalid Whisper decoder input"});
        auto& context = impl_->context;
        require(state.mask.data<float>())[state.position] = 0.F;
        require(state.index.data<std::int32_t>())[0] = static_cast<std::int32_t>(state.position);
        require(state.tokens.data<std::int32_t>())[0] = token;
        Tensor output;
        if (project) {
            // Step inputs: tokens, mask, index, then source and self-attention key/value pairs per layer.
            auto& inputs = impl_->step_inputs;
            inputs.assign({state.tokens, state.mask, state.index});
            for (const auto* layers : {&source.layers, &std::as_const(state.layers)})
                for (const auto& layer : *layers) inputs.insert(inputs.end(), {layer.key, layer.value});
            const auto count = static_cast<std::size_t>(impl_->decoder_layers);
            const auto key = "whisper_decode_" + std::to_string(state.capacity) + '_' +
                             std::to_string(source.layers.front().key.size(1));
            // A captured step keeps its inputs, including the encoder keys and values, so only the current shape's
            // step is retained.
            if (key != impl_->decode_key) {
                context.clear_replays("whisper_decode_");
                impl_->decode_key = key;
            }
            output = context.replay(key, inputs, [&](std::span<const Tensor> step) {
                    std::vector<layers::KeyValue> memory, cache;
                    for (std::size_t layer = 0; layer < count; ++layer) {
                        memory.push_back({step[3 + 2 * layer], step[4 + 2 * layer]});
                        cache.push_back({step[3 + 2 * (count + layer)], step[4 + 2 * (count + layer)]});
                    }
                    const auto hidden = impl_->decoder->forward(context, step[0], memory, step[1], cache, step[2]);
                    return std::vector{impl_->output->forward(context, hidden)};
                })[0];
        } else {
            output =
                impl_->decoder->forward(context, state.tokens, source.layers, state.mask, state.layers, state.index);
        }
        context.synchronize();
        ++state.position;
        return output;
    } catch (const ops::Failure& error) {
        return std::unexpected(error.error());
    }
}

auto WhisperImpl::preparation_ns() const -> std::uint64_t { return impl_->context.preparation_ns(); }

} // namespace kidi::model