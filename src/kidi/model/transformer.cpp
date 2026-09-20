#include "kidi/model/transformer.h"
#include <algorithm>
#include <chrono>
#include <cmath>

namespace kidi::model {
using ops::require;
using tensor::DType;
using tensor::Tensor;
namespace {
using Clock = std::chrono::steady_clock;
auto elapsed(Clock::time_point start) -> std::uint64_t {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count();
}
} // namespace
struct TransformerImpl::State {
    ops::Context context;
    std::int32_t hidden_size, maximum_position, source_tokens, pad;
    layers::Embedding source_embedding{nullptr}, target_embedding{nullptr};
    ModuleList<layers::EncoderBlockImpl> encoder;
    ModuleList<layers::DecoderBlockImpl> decoder;
    layers::LayerNorm encoder_norm{nullptr}, decoder_norm{nullptr};
    layers::Linear generator{nullptr};
    explicit State(const YAML::Node& config)
        : context(module_device),
          hidden_size(config["hidden_size"].as<std::int32_t>()),
          maximum_position(config["maximum_position"].as<std::int32_t>()),
          source_tokens(config["source_tokens"].as<std::int32_t>()),
          pad(config["source_pad_id"].as<std::int32_t>()) {
        const auto epsilon = config["layer_norm_epsilon"].as<float>();
        source_embedding =
            layers::Embedding(config["source_vocabulary_size"].as<std::int32_t>(), hidden_size, maximum_position);
        target_embedding =
            layers::Embedding(config["target_vocabulary_size"].as<std::int32_t>(), hidden_size, maximum_position);
        encoder_norm = layers::LayerNorm(hidden_size, epsilon);
        decoder_norm = layers::LayerNorm(hidden_size, epsilon);
        const bool tied = module_dtype != DType::I8;
        generator = layers::Linear(hidden_size, config["target_vocabulary_size"].as<std::int32_t>(), tied);
        const layers::Shape shape{hidden_size, config["feed_forward_size"].as<std::int32_t>(),
                                  config["attention_heads"].as<std::int32_t>(), epsilon};
        for (std::int32_t index = 0; index < config["encoder_layers"].as<std::int32_t>(); ++index)
            encoder->push_back(layers::EncoderBlock(shape));
        for (std::int32_t index = 0; index < config["decoder_layers"].as<std::int32_t>(); ++index)
            decoder->push_back(layers::DecoderBlock(shape));
    }
};
auto TransformerImpl::validate_config(const YAML::Node& config) -> Result<void> {
    try {
        if (!config.IsMap() || config["type"].as<std::string>() != "rtg_transformer_nmt")
            return std::unexpected(Error{ErrorCode::INVALID_MANIFEST, "unsupported model.type"});
        for (const auto* field :
             {"encoder_layers", "decoder_layers", "hidden_size", "feed_forward_size", "attention_heads",
              "source_vocabulary_size", "target_vocabulary_size", "maximum_position", "source_tokens"})
            if (config[field].as<std::int32_t>() <= 0)
                return std::unexpected(
                    Error{ErrorCode::INVALID_MANIFEST, std::string("model.") + field + " must be positive"});
        if (config["hidden_size"].as<int>() % config["attention_heads"].as<int>() != 0)
            return std::unexpected(
                Error{ErrorCode::INVALID_MANIFEST, "hidden_size must be divisible by attention_heads"});
        if (config["activation"].as<std::string>() != "gelu" || !config["attention_bias"].as<bool>() ||
            config["tied_embeddings"].as<std::string>() != "one-way" ||
            config["position_encoding"].as<std::string>() != "sinusoidal")
            return std::unexpected(Error{ErrorCode::INVALID_MANIFEST, "unsupported Transformer architecture"});
        const auto epsilon = config["layer_norm_epsilon"].as<float>();
        if (!std::isfinite(epsilon) || epsilon <= 0)
            return std::unexpected(
                Error{ErrorCode::INVALID_MANIFEST, "layer_norm_epsilon must be finite and positive"});
        if (config["source_tokens"].as<int>() > config["maximum_position"].as<int>())
            return std::unexpected(Error{ErrorCode::INVALID_MANIFEST, "source_tokens exceeds maximum_position"});
        const auto pad = config["source_pad_id"].as<std::int32_t>();
        if (pad < 0 || pad >= config["source_vocabulary_size"].as<int>())
            return std::unexpected(Error{ErrorCode::INVALID_MANIFEST, "invalid source padding token ID"});
        return {};
    } catch (const YAML::Exception& error) {
        return std::unexpected(
            Error{ErrorCode::INVALID_MANIFEST, "invalid model config: " + std::string(error.what())});
    }
}
TransformerImpl::TransformerImpl(const YAML::Node& config) {
    require(validate_config(config));
    impl_ = std::make_unique<State>(config);
    register_module("source_embedding", impl_->source_embedding);
    register_module("target_embedding", impl_->target_embedding);
    register_module("encoder", impl_->encoder);
    register_module("decoder", impl_->decoder);
    register_module("encoder_norm", impl_->encoder_norm);
    register_module("decoder_norm", impl_->decoder_norm);
    register_module("generator", impl_->generator);
    if (module_dtype != DType::I8) tie_parameter("generator.weight", "target_embedding.weight");
}
TransformerImpl::~TransformerImpl() = default;
auto TransformerImpl::create(const YAML::Node& config) -> Result<Transformer> {
    try {
        return Transformer(config);
    } catch (const ops::Failure& error) {
        return std::unexpected(error.error());
    }
}
auto TransformerImpl::state_mapping_specs() -> std::span<const StateMappingSpec> {
    static const auto specs = [] {
        std::vector<StateMappingSpec> result;
        const auto add = [&](std::string source, std::string target) {
            for (const auto& [suffix, destination] :
                 std::array{std::pair{R"(\.weight)", ".weight"}, std::pair{R"(\.bias)", ".bias"},
                            std::pair{R"(\.weight\.scale)", ".scale"}})
                result.push_back({{source + suffix}, target + destination});
        };
        add(R"(src_embed\.0\.lut)", "source_embedding");
        add(R"(tgt_embed\.0\.lut)", "target_embedding");
        add(R"(generator\.proj)", "generator");
        add(R"((encoder|decoder)\.norm)", "$1_norm");
        add(R"(encoder\.layers\.(\d+)\.self_attn\.qkv)", "encoder.$1.qkv");
        add(R"(encoder\.layers\.(\d+)\.self_attn\.out)", "encoder.$1.attention.output");
        add(R"(decoder\.layers\.(\d+)\.self_attn\.qkv)", "decoder.$1.qkv");
        add(R"(decoder\.layers\.(\d+)\.self_attn\.out)", "decoder.$1.self_attention.output");
        add(R"(decoder\.layers\.(\d+)\.src_attn\.q)", "decoder.$1.query");
        add(R"(decoder\.layers\.(\d+)\.src_attn\.kv)", "decoder.$1.source");
        add(R"(decoder\.layers\.(\d+)\.src_attn\.out)", "decoder.$1.cross_attention.output");
        add(R"((encoder|decoder)\.layers\.(\d+)\.feed_forward\.w_1)", "$1.$2.feed_forward.first");
        add(R"((encoder|decoder)\.layers\.(\d+)\.feed_forward\.w_2)", "$1.$2.feed_forward.second");
        add(R"(encoder\.layers\.(\d+)\.sublayer\.0\.norm)", "encoder.$1.attention_norm");
        add(R"(encoder\.layers\.(\d+)\.sublayer\.1\.norm)", "encoder.$1.feed_forward_norm");
        add(R"(decoder\.layers\.(\d+)\.sublayer\.0\.norm)", "decoder.$1.self_norm");
        add(R"(decoder\.layers\.(\d+)\.sublayer\.1\.norm)", "decoder.$1.cross_norm");
        add(R"(decoder\.layers\.(\d+)\.sublayer\.2\.norm)", "decoder.$1.feed_forward_norm");
        return result;
    }();
    return specs;
}
auto TransformerImpl::encode(std::span<const std::vector<std::int32_t>> sources, inference::InferenceStats* stats)
    -> Result<EncoderState> {
    try {
        auto& context = impl_->context;
        const auto prep = context.preparation_ns();
        auto started = Clock::now();
        std::size_t length = 0;
        if (sources.empty()) throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "empty encoder batch"});
        for (const auto& source : sources) {
            if (source.empty() || source.size() > static_cast<std::size_t>(impl_->source_tokens))
                throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "source length outside configured limits"});
            length = std::max(length, source.size());
        }
        std::vector<std::int32_t> tokens(sources.size() * length, impl_->pad);
        std::vector<float> mask(sources.size() * length, -1.0e9F);
        for (std::size_t row = 0; row < sources.size(); ++row) {
            std::copy(sources[row].begin(), sources[row].end(), tokens.begin() + row * length);
            std::fill_n(mask.begin() + row * length, sources[row].size(), 0.F);
        }
        EncoderState result;
        result.mask = require(
            Tensor::from_host({static_cast<std::int64_t>(sources.size()), 1, 1, static_cast<std::int64_t>(length)},
                              std::span<const float>(mask), context.device()));
        auto hidden = impl_->source_embedding->forward(context, tokens, sources.size());
        if (stats) stats->source_embedding_ns += elapsed(started);
        context.profile_phase("encoder");
        for (const auto& layer : *impl_->encoder) hidden = layer->forward(context, hidden, result.mask);
        hidden = impl_->encoder_norm->forward(context, hidden);
        context.synchronize();
        if (stats) stats->encoder_ns += elapsed(started);
        started = Clock::now();
        context.profile_phase("source_projection");
        for (const auto& layer : *impl_->decoder) result.layers.push_back(layer->project_source(context, hidden));
        context.synchronize();
        if (stats) {
            stats->source_projection_ns += elapsed(started);
            stats->graph_compile_ns += context.preparation_ns() - prep;
        }
        return result;
    } catch (const ops::Failure& error) {
        return std::unexpected(error.error());
    }
}
auto TransformerImpl::create_state(std::size_t batch, std::size_t capacity) -> Result<DecoderState> {
    try {
        if (!batch || !capacity || capacity > static_cast<std::size_t>(impl_->maximum_position))
            throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "invalid decoder cache capacity"});
        DecoderState result;
        result.capacity = capacity;
        result.mask =
            require(Tensor::empty({1, 1, 1, static_cast<std::int64_t>(capacity)}, DType::F32, impl_->context.device()));
        auto mask = require(result.mask.data<float>());
        std::fill(mask.begin(), mask.end(), -1e9F);
        result.index = require(Tensor::empty({1}, DType::I32, impl_->context.device()));
        result.embedding = require(Tensor::empty({static_cast<std::int64_t>(batch), 1, impl_->hidden_size}, DType::F32,
                                                 impl_->context.device()));
        const std::vector<std::int64_t> shape{static_cast<std::int64_t>(batch), static_cast<std::int64_t>(capacity),
                                              impl_->hidden_size};
        for (std::size_t index = 0; index < impl_->decoder->size(); ++index)
            result.layers.push_back({require(Tensor::zeros(shape, DType::F32, impl_->context.device())),
                                     require(Tensor::zeros(shape, DType::F32, impl_->context.device()))});
        return result;
    } catch (const ops::Failure& error) {
        return std::unexpected(error.error());
    }
}
auto TransformerImpl::forward(const EncoderState& source, std::span<const std::int32_t> tokens, std::size_t batch,
                              DecoderState* state, inference::InferenceStats* stats) -> Result<Tensor> {
    try {
        if (!batch || tokens.empty() || tokens.size() % batch || source.layers.size() != impl_->decoder->size() ||
            (state && (state->position >= state->capacity || tokens.size() != batch ||
                       state->layers.size() != impl_->decoder->size())))
            throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "invalid decoder inputs/state"});
        auto& context = impl_->context;
        auto prep = context.preparation_ns();
        auto started = Clock::now();
        auto hidden = state
                          ? impl_->target_embedding->forward_(context, state->embedding, tokens, batch, state->position)
                          : impl_->target_embedding->forward(context, tokens, batch);
        if (stats) stats->target_embedding_ns += elapsed(started);
        const auto length = tokens.size() / batch;
        Tensor mask, index;
        if (state) {
            require(state->mask.data<float>())[state->position] = 0.F;
            require(state->index.data<std::int32_t>())[0] = static_cast<std::int32_t>(state->position);
            mask = state->mask;
            index = state->index;
        } else {
            std::vector<float> values(length * length, -1.0e9F);
            for (std::size_t row = 0; row < length; ++row) std::fill_n(values.begin() + row * length, row + 1, 0.F);
            mask =
                require(Tensor::from_host({1, 1, static_cast<std::int64_t>(length), static_cast<std::int64_t>(length)},
                                          std::span<const float>(values), context.device()));
        }
        context.profile_phase("decoder");
        for (std::size_t layer = 0; layer < impl_->decoder->size(); ++layer)
            hidden = impl_->decoder->at(layer)->forward(context, hidden, source.layers[layer], source.mask, mask,
                                                        state ? &state->layers[layer] : nullptr, index);
        hidden = impl_->decoder_norm->forward(context, hidden);
        if (!state) hidden = context.slice(hidden, 1, length - 1, 1);
        context.profile_phase("generator");
        auto logits = impl_->generator->forward(context, hidden);
        context.synchronize();
        if (state) ++state->position;
        if (stats) stats->graph_compile_ns += context.preparation_ns() - prep;
        return logits;
    } catch (const ops::Failure& error) {
        return std::unexpected(error.error());
    }
}
} // namespace kidi::model