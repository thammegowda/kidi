#include "kidi/model/transformer.h"
#include <algorithm>
#include <chrono>

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
    ModelManifest manifest;
    layers::Embedding source_embedding, target_embedding;
    ModuleList<layers::EncoderBlockImpl> encoder = std::make_shared<ModuleListImpl<layers::EncoderBlockImpl>>();
    ModuleList<layers::DecoderBlockImpl> decoder = std::make_shared<ModuleListImpl<layers::DecoderBlockImpl>>();
    layers::LayerNorm encoder_norm, decoder_norm;
    layers::Linear generator;
    State(const Package& package, tensor::Device device)
        : context(device),
          manifest(package.manifest()),
          source_embedding(std::make_shared<layers::EmbeddingImpl>(package.weights(), "src_embed.0.lut.weight",
                                                                   manifest.weights.encoding,
                                                                   manifest.architecture.maximum_position)),
          target_embedding(std::make_shared<layers::EmbeddingImpl>(package.weights(), "tgt_embed.0.lut.weight",
                                                                   manifest.weights.encoding,
                                                                   manifest.architecture.maximum_position)),
          encoder_norm(std::make_shared<layers::LayerNormImpl>(package.weights(), "encoder.norm",
                                                               manifest.architecture.layer_norm_epsilon)),
          decoder_norm(std::make_shared<layers::LayerNormImpl>(package.weights(), "decoder.norm",
                                                               manifest.architecture.layer_norm_epsilon)) {
        const bool tied =
            manifest.weights.encoding == WeightEncoding::F32 ||
            (manifest.weights.encoding == WeightEncoding::BF16 && !package.weights().contains("generator.proj.weight"));
        generator = std::make_shared<layers::LinearImpl>(package.weights(),
                                                         tied ? "tgt_embed.0.lut.weight" : "generator.proj.weight",
                                                         "generator.proj.bias", manifest.weights.encoding, tied);
        const auto& arch = manifest.architecture;
        const layers::Shape shape{arch.hidden_size, arch.feed_forward_size, arch.attention_heads,
                                  arch.layer_norm_epsilon};
        for (std::int32_t index = 0; index < arch.encoder_layers; ++index)
            encoder->push_back(std::make_shared<layers::EncoderBlockImpl>(
                package.weights(), "encoder.layers." + std::to_string(index), manifest.weights.encoding, shape));
        for (std::int32_t index = 0; index < arch.decoder_layers; ++index)
            decoder->push_back(std::make_shared<layers::DecoderBlockImpl>(
                package.weights(), "decoder.layers." + std::to_string(index), manifest.weights.encoding, shape));
    }
};
TransformerImpl::TransformerImpl(const Package& package, tensor::Device device)
    : impl_(std::make_unique<State>(package, device)) {
    register_module("source_embedding", impl_->source_embedding);
    register_module("target_embedding", impl_->target_embedding);
    register_module("encoder", impl_->encoder);
    register_module("decoder", impl_->decoder);
    register_module("encoder_norm", impl_->encoder_norm);
    register_module("decoder_norm", impl_->decoder_norm);
    register_module("generator", impl_->generator);
}
TransformerImpl::~TransformerImpl() = default;
auto TransformerImpl::create(const Package& package, tensor::Device device) -> Result<Transformer> {
    try {
        return std::make_shared<TransformerImpl>(package, device);
    } catch (const ops::Failure& error) {
        return std::unexpected(error.error());
    }
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
            if (source.empty() || source.size() > static_cast<std::size_t>(impl_->manifest.limits.source_tokens))
                throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "source length outside configured limits"});
            length = std::max(length, source.size());
        }
        std::vector<std::int32_t> tokens(sources.size() * length, impl_->manifest.special_tokens.pad);
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
        if (!batch || !capacity || capacity > static_cast<std::size_t>(impl_->manifest.architecture.maximum_position))
            throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "invalid decoder cache capacity"});
        DecoderState result;
        result.capacity = capacity;
        result.mask =
            require(Tensor::empty({1, 1, 1, static_cast<std::int64_t>(capacity)}, DType::F32, impl_->context.device()));
        auto mask = require(result.mask.data<float>());
        std::fill(mask.begin(), mask.end(), -1e9F);
        result.index = require(Tensor::empty({1}, DType::I32, impl_->context.device()));
        result.embedding =
            require(Tensor::empty({static_cast<std::int64_t>(batch), 1, impl_->manifest.architecture.hidden_size},
                                  DType::F32, impl_->context.device()));
        const std::vector<std::int64_t> shape{static_cast<std::int64_t>(batch), static_cast<std::int64_t>(capacity),
                                              impl_->manifest.architecture.hidden_size};
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