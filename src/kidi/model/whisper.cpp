#include "kidi/model/whisper.h"
#include "kidi/ops/quantization.h"

#include <algorithm>
#include <cmath>
#include <tuple>
#include <utility>

namespace kidi::model {
using ops::require;
using tensor::DType;
using tensor::Tensor;

struct WhisperImpl::State {
    ops::Context context;
    std::int32_t hidden, encoder_layers, decoder_layers, maximum_target, vocabulary, mel_bins;
    layers::WhisperEncoder encoder;
    layers::WhisperDecoder decoder;
    layers::Linear output;
    DType precision = module_dtype;
    std::vector<Tensor> step_inputs;

    explicit State(const YAML::Node& config)
        : context(module_device),
          hidden(config["d_model"].as<int>()),
          encoder_layers(config["encoder_layers"].as<int>()),
          decoder_layers(config["decoder_layers"].as<int>()),
          maximum_target(config["max_target_positions"].as<int>()),
          vocabulary(config["vocab_size"].as<int>()),
          mel_bins(config["num_mel_bins"].as<int>()),
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
        const std::tuple geometry{
            hidden,
            config["encoder_layers"].as<int>(),
            config["decoder_layers"].as<int>(),
            config["encoder_attention_heads"].as<int>(),
            config["decoder_attention_heads"].as<int>(),
            config["encoder_ffn_dim"].as<int>(),
            config["decoder_ffn_dim"].as<int>(),
            config["num_mel_bins"].as<int>(),
            config["vocab_size"].as<int>(),
        };
        constexpr std::array SUPPORTED{
            std::tuple{384, 4, 4, 6, 6, 1536, 1536, 80, 51865},
            std::tuple{512, 6, 6, 8, 8, 2048, 2048, 80, 51865},
            std::tuple{768, 12, 12, 12, 12, 3072, 3072, 80, 51865},
            std::tuple{1280, 32, 32, 20, 20, 5120, 5120, 128, 51866},
            std::tuple{1280, 32, 4, 20, 20, 5120, 5120, 128, 51866},
        };
        if (!std::ranges::contains(SUPPORTED, geometry) || config["max_source_positions"].as<int>() != 1500 ||
            config["max_target_positions"].as<int>() != 448)
            return std::unexpected(
                Error{ErrorCode::UNSUPPORTED,
                      "supported multilingual Whisper sizes are Tiny, Base, Small, Large-v3, and Large-v3-Turbo"});
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

auto WhisperImpl::int8_preparation(const YAML::Node& config) -> Result<checkpoint::Preparation> {
    auto valid = validate_config(config["model"]);
    if (!valid) return std::unexpected(valid.error());
    const auto source = std::filesystem::path(config["weights_file"].as<std::string>());
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

auto WhisperImpl::state_mapping_specs() const -> std::vector<checkpoint::StateMappingSpec> {
    using checkpoint::StateMappingOp;
    using checkpoint::StateMappingSpec;
    return {
        {.sources = {R"(^model\.(.+)$)"}, .destination = "$1"},
        {.sources = {R"(^proj_out\.(weight|scale)$)", "decoder.embed_tokens.$1"},
         .destination = "decoder.embed_tokens.$1",
         .op = StateMappingOp::IDENTICAL},
        {.sources = {R"(^(encoder|decoder)\.layers\.(\d+)\.self_attn\.q_proj\.(weight|scale)$)",
                     "$1.layers.$2.self_attn.k_proj.$3", "$1.layers.$2.self_attn.v_proj.$3"},
         .destination = "$1.layers.$2.self_attn.qkv_proj.$3"},
        {.sources = {R"(^(encoder|decoder)\.layers\.(\d+)\.self_attn\.q_proj\.bias$)",
                     "$1.layers.$2.self_attn.k_proj.bias", "$1.layers.$2.self_attn.v_proj.bias"},
         .destination = "$1.layers.$2.self_attn.qkv_proj.bias"},
        {.sources = {R"(^(encoder|decoder)\.layers\.(\d+)\.self_attn\.q_proj\.bias$)",
                     "0:$1.layers.$2.self_attn.q_proj.bias", "$1.layers.$2.self_attn.v_proj.bias"},
         .destination = "$1.layers.$2.self_attn.qkv_proj.bias"},
        {.sources = {R"(^decoder\.layers\.(\d+)\.encoder_attn\.k_proj\.(weight|scale)$)",
                     "decoder.layers.$1.encoder_attn.v_proj.$2"},
         .destination = "decoder.layers.$1.encoder_attn.kv_proj.$2"},
        {.sources = {R"(^decoder\.layers\.(\d+)\.encoder_attn\.k_proj\.bias$)",
                     "decoder.layers.$1.encoder_attn.v_proj.bias"},
         .destination = "decoder.layers.$1.encoder_attn.kv_proj.bias"},
        {.sources = {"0:decoder.layers.$1.encoder_attn.v_proj.bias",
                     R"(^decoder\.layers\.(\d+)\.encoder_attn\.v_proj\.bias$)"},
         .destination = "decoder.layers.$1.encoder_attn.kv_proj.bias"},
    };
}

auto WhisperImpl::hidden_size() const noexcept -> std::int32_t { return impl_->hidden; }
auto WhisperImpl::vocabulary_size() const noexcept -> std::int32_t { return impl_->vocabulary; }
auto WhisperImpl::max_positions() const noexcept -> std::size_t {
    return static_cast<std::size_t>(impl_->maximum_target);
}
auto WhisperImpl::mel_bins() const noexcept -> std::int32_t { return impl_->mel_bins; }
auto WhisperImpl::preparation_ns() const noexcept -> std::uint64_t { return impl_->context.preparation_ns(); }
auto WhisperImpl::release_workspaces() -> void { impl_->context.release_workspaces(); }

auto WhisperImpl::set_checkpoint(const checkpoint::Weights& weights) -> Result<void> {
    auto state = prepare_checkpoint(weights);
    if (!state) return std::unexpected(std::move(state.error()));
    if (impl_->precision == DType::I8) {
        try {
            const auto slots = parameter_specs();
            StateDict scales;
            for (auto& [name, value] : *state) {
                const auto slot = slots.find(name);
                if (slot == slots.end() || slot->second.dtype != DType::I8 || value.dtype() == DType::I8) continue;
                const auto rows = static_cast<std::int64_t>(value.size(0));
                const auto width = static_cast<std::int64_t>(value.numel() / value.size(0));
                const auto matrix = require(value.reshape({rows, width}));
                const auto packed = require(ops::pack_weight(matrix, 8, static_cast<std::int32_t>(width)));
                const auto bytes = require(packed.values.host_bytes());
                value = require(
                    Tensor::from_host(std::vector<std::int64_t>(value.shape().begin(), value.shape().end()),
                                      std::span(reinterpret_cast<const std::int8_t*>(bytes.data()), bytes.size()),
                                      tensor::Device::cpu()));
                scales.emplace(name.substr(0, name.size() - 7) + ".scale", packed.scales);
            }
            for (auto& [name, scale] : scales) state->emplace(name, std::move(scale));
            for (const auto& [name, value] : *state) {
                if (!name.ends_with(".scale")) continue;
                if (value.dtype() != DType::F32)
                    return std::unexpected(
                        Error{ErrorCode::INVALID_ARGUMENT, "Whisper INT8 scale must be FP32: " + name});
                for (const auto scale : require(std::as_const(value).data<float>()))
                    if (!std::isfinite(scale) || scale <= 0)
                        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT,
                                                     "Whisper INT8 scale must be positive and finite: " + name});
            }
        } catch (const ops::Failure& error) {
            return std::unexpected(error.error());
        }
    }
    return set_state(*state);
}

auto WhisperImpl::encode_audio(const audio::WhisperFeatures& features, std::size_t frames,
                               const std::function<bool()>& cancelled) -> Result<std::shared_ptr<EncoderMemory>> {
    try {
        if (!frames) frames = features.frames;
        if (!device().capabilities().supports_audio_transformer() ||
            features.bins != static_cast<std::size_t>(impl_->mel_bins) || features.frames != 3000 ||
            features.values.size() != features.bins * features.frames || frames % 2 || frames > features.frames)
            throw ops::Failure(
                {ErrorCode::INVALID_ARGUMENT, "Whisper features must match the model's mel bins and 3000 frames"});
        std::vector<float> values(frames * features.bins);
        for (std::size_t frame = 0; frame < frames; ++frame)
            for (std::size_t bin = 0; bin < features.bins; ++bin)
                values[frame * features.bins + bin] = features.values[bin * features.frames + frame];
        const auto input =
            require(Tensor::from_host({1, static_cast<std::int64_t>(frames), static_cast<std::int64_t>(features.bins)},
                                      std::span<const float>(values), device()));
        auto result = std::make_shared<WhisperEncoderState>();
        if (device().policy().replay.region_layers) {
            const auto outputs = impl_->context.replay(
                "whisper_encoder:" + std::to_string(frames), std::array{input},
                [&](std::span<const Tensor> inputs) {
                    const auto convolution = impl_->encoder->convolve(impl_->context, inputs[0]);
                    const auto hidden = impl_->encoder->encode(impl_->context, convolution, cancelled);
                    const auto memory = impl_->decoder->project_source(impl_->context, hidden);
                    std::vector<Tensor> replay_outputs{convolution, hidden};
                    for (const auto& layer : memory)
                        replay_outputs.insert(replay_outputs.end(), {layer.key, layer.value});
                    return replay_outputs;
                },
                false);
            result->convolution = outputs[0];
            result->hidden = outputs[1];
            for (std::size_t layer = 0; layer < static_cast<std::size_t>(impl_->decoder_layers); ++layer)
                result->layers.push_back({outputs[2 + 2 * layer], outputs[3 + 2 * layer]});
            impl_->context.synchronize();
            return std::shared_ptr<EncoderMemory>(std::move(result));
        }
        result->convolution = impl_->encoder->convolve(impl_->context, input);
        result->hidden = impl_->encoder->encode(impl_->context, result->convolution, cancelled);
        result->layers = impl_->decoder->project_source(impl_->context, result->hidden);
        impl_->context.synchronize();
        return std::shared_ptr<EncoderMemory>(std::move(result));
    } catch (const ops::Failure& error) {
        return std::unexpected(error.error());
    }
}

auto WhisperImpl::create_decoder_state(std::shared_ptr<const EncoderMemory> memory, std::size_t batch,
                                       std::size_t capacity) -> Result<std::unique_ptr<LanguageModelState>> {
    try {
        if (batch != 1 || !capacity || capacity > static_cast<std::size_t>(impl_->maximum_target))
            throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "invalid Whisper decoder capacity"});
        auto result = std::make_unique<WhisperDecoderState>();
        result->memory = std::move(memory);
        result->batch = batch;
        result->capacity = capacity;
        const std::vector<float> mask(capacity, -1e9F);
        result->mask = require(
            Tensor::from_host({1, 1, 1, static_cast<std::int64_t>(capacity)}, std::span<const float>(mask), device()));
        result->index = require(Tensor::empty({1}, DType::I32, device()));
        result->tokens = require(Tensor::empty({1}, DType::I32, device()));
        const std::vector<std::int64_t> shape{1, static_cast<std::int64_t>(capacity), impl_->hidden};
        for (std::int32_t layer = 0; layer < impl_->decoder_layers; ++layer)
            result->layers.push_back({require(Tensor::zeros(shape, DType::F32, device())),
                                      require(Tensor::zeros(shape, DType::F32, device()))});
        return std::unique_ptr<LanguageModelState>(std::move(result));
    } catch (const ops::Failure& error) {
        return std::unexpected(error.error());
    }
}

auto WhisperImpl::decode_step(const EncoderMemory& memory, std::span<const std::int32_t> tokens, std::size_t batch,
                              LanguageModelState* state, inference::InferenceStats*) -> Result<Tensor> {
    const auto* source = dynamic_cast<const WhisperEncoderState*>(&memory);
    auto* decoder = dynamic_cast<WhisperDecoderState*>(state);
    if (!source || !decoder || batch != 1 || tokens.size() != 1)
        return std::unexpected(
            Error{ErrorCode::INVALID_ARGUMENT, "Whisper decode requires one token and decoder state"});
    return decode(*source, tokens.front(), *decoder, true);
}

auto WhisperImpl::prefill(std::span<const std::int32_t> tokens, LanguageModelState& state) -> Result<void> {
    auto* decoder = dynamic_cast<WhisperDecoderState*>(&state);
    const auto* source = decoder ? dynamic_cast<const WhisperEncoderState*>(decoder->memory.get()) : nullptr;
    if (!decoder || !source || tokens.empty() || decoder->position > decoder->capacity ||
        tokens.size() > decoder->capacity - decoder->position)
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "invalid Whisper prefix length"});
    for (const auto token : tokens) {
        auto hidden = decode(*source, token, *decoder, false);
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
        const auto position = static_cast<std::int32_t>(state.position);
        const float visible = 0.F;
        require(state.mask.copy_from_host(std::as_bytes(std::span(&visible, 1)), state.position * sizeof(float)));
        require(state.index.copy_from_host(std::as_bytes(std::span(&position, 1))));
        require(state.tokens.copy_from_host(std::as_bytes(std::span(&token, 1))));
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

auto WhisperImpl::select_token(const Tensor& logits, const Tensor& mask) -> Tensor {
    return select_masked_token(impl_->context, logits, mask);
}

auto WhisperImpl::select_language(const Tensor& logits, const Tensor& language_ids) -> Tensor {
    return select_allowed_token(impl_->context, logits, language_ids);
}

} // namespace kidi::model