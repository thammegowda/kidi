#include "kidi/inference/translator.h"
#include "kidi/inference/decoder.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <utility>
#include <vector>

namespace kidi::inference {
namespace {

using Clock = std::chrono::steady_clock;

class ScopedTimer {
public:
    explicit ScopedTimer(std::uint64_t* elapsed_ns, const std::uint64_t* excluded_ns = nullptr)
        : elapsed_ns_(elapsed_ns), excluded_ns_(excluded_ns), excluded_start_(excluded_ns ? *excluded_ns : 0) {
        if (elapsed_ns_) started_ = Clock::now();
    }

    ~ScopedTimer() {
        if (elapsed_ns_) *elapsed_ns_ += elapsed_since(started_) - (excluded_ns_ ? *excluded_ns_ - excluded_start_ : 0);
    }

    static auto elapsed_since(Clock::time_point started) -> std::uint64_t {
        return static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - started).count());
    }

private:
    std::uint64_t* elapsed_ns_;
    Clock::time_point started_;
    const std::uint64_t* excluded_ns_;
    std::uint64_t excluded_start_;
};

template <typename Function>
auto measure(std::uint64_t* elapsed_ns, Function&& function) {
    ScopedTimer timer(elapsed_ns);
    return std::forward<Function>(function)();
}

auto resolve_options(const YAML::Node& config, const DecodeOptions& options) -> Result<YAML::Node> {
    try {
        auto result = YAML::Clone(config);
        if (options.beam_size) result["beam_size"] = *options.beam_size;
        if (options.maximum_extra_tokens) result["maximum_extra_tokens"] = *options.maximum_extra_tokens;
        if (options.length_penalty) result["length_penalty"] = *options.length_penalty;
        const auto beam = result["beam_size"].as<std::int32_t>();
        const auto extra = result["maximum_extra_tokens"].as<std::int32_t>();
        const auto penalty = result["length_penalty"].as<float>();
        if (beam <= 0) return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "beam size must be positive"});
        if (extra <= 0)
            return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "maximum extra tokens must be positive"});
        if (!std::isfinite(penalty) || penalty < 0)
            return std::unexpected(
                Error{ErrorCode::INVALID_ARGUMENT, "length penalty must be finite and non-negative"});
        return result;
    } catch (const YAML::Exception& error) {
        return std::unexpected(
            Error{ErrorCode::INVALID_MANIFEST, "invalid decode config: " + std::string(error.what())});
    }
}

} // namespace

Translator::Translator(model::Package package, model::Transformer model) noexcept
    : package_(std::move(package)), model_(std::move(model)) {}

auto Translator::load(const std::filesystem::path& model_directory, tensor::Device device, InferenceStats* stats,
                      std::size_t batch_size) -> Result<Translator> {
    if (device != tensor::Device::cpu() && device != tensor::Device::apple_gpu())
        return std::unexpected(
            Error{ErrorCode::UNSUPPORTED, "no translation backend for " + tensor::to_string(device)});
    if (batch_size == 0 || batch_size > 256)
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "batch size must be between 1 and 256"});
    auto package =
        measure(stats ? &stats->package_load_ns : nullptr, [&] { return model::Package::load(model_directory); });
    if (!package) return std::unexpected(std::move(package.error()));
    auto defaults = resolve_options(package->config()["decode"], {});
    if (!defaults) return std::unexpected(std::move(defaults.error()));
    auto embedding = package->weights().tensor("target_embedding.weight");
    if (!embedding) return std::unexpected(std::move(embedding.error()));
    auto model = [&] {
        const ModuleScope construction(embedding->dtype(), false, device);
        return model::TransformerImpl::create(package->config()["model"]);
    }();
    if (!model) return std::unexpected(std::move(model.error()));
    auto loaded = (*model)->set_state(package->weights());
    if (!loaded) return std::unexpected(std::move(loaded.error()));
    Translator translator(std::move(*package), std::move(*model));
    translator.batch_size_ = batch_size;
    return translator;
}

auto Translator::translate(std::string_view source, DecodeOptions options, InferenceStats* stats)
    -> Result<Translation> {
    ScopedTimer translate_timer(stats ? &stats->translate_ns : nullptr, stats ? &stats->graph_compile_ns : nullptr);
    const auto config = package_.config()["decode"];
    auto decode = resolve_options(config, options);
    if (!decode) return std::unexpected(std::move(decode.error()));
    auto source_ids = measure(stats ? &stats->source_tokenize_ns : nullptr,
                              [&] { return package_.source_tokenizer().encode(source); });
    if (!source_ids) return std::unexpected(std::move(source_ids.error()));
    if (source_ids->empty() || source_ids->back() != config["source_end_id"].as<std::int32_t>()) {
        source_ids->push_back(config["source_end_id"].as<std::int32_t>());
    }
    const auto maximum_steps = source_ids->size() + (*decode)["maximum_extra_tokens"].as<std::size_t>();
    auto projected_source = model_->encode(std::array{*source_ids}, stats);
    if (!projected_source) return std::unexpected(std::move(projected_source.error()));

    const auto beam_size = (*decode)["beam_size"].as<std::size_t>();
    const auto vocabulary_size = config["vocabulary_size"].as<std::size_t>();
    std::optional<model::DecoderState> self_cache;
    if (beam_size == 1) {
        auto cache = model_->create_state(1, maximum_steps);
        if (!cache) return std::unexpected(std::move(cache.error()));
        self_cache = std::move(*cache);
    }
    const inference::SearchOptions search{
        .vocabulary_size = vocabulary_size,
        .end_id = config["end_id"].as<std::int32_t>(),
        .pad_id = config["pad_id"].as<std::int32_t>(),
        .maximum_steps = maximum_steps,
        .beam_size = beam_size,
        .length_penalty = (*decode)["length_penalty"].as<float>(),
        .compute_score = options.compute_score,
        .unfinished_score_length = (*decode)["maximum_extra_tokens"].as<std::size_t>()};
    tensor::Tensor score_storage;
    auto generated = inference::Decoder::generate(
        std::array{config["begin_id"].as<std::int32_t>()}, search,
        [&](const inference::DecodeRequest& request) -> Result<inference::TokenScores> {
            auto scores = model_->forward(*projected_source, self_cache ? request.prefixes.last(1) : request.prefixes,
                                          request.beam_indices.size(), self_cache ? &*self_cache : nullptr, stats);
            if (!scores) return std::unexpected(std::move(scores.error()));
            score_storage = std::move(*scores);
            auto values = score_storage.data<float>();
            if (!values) return std::unexpected(std::move(values.error()));
            return inference::TokenScores{*values, inference::ScoreKind::LOGITS};
        },
        stats);
    if (!generated) return std::unexpected(std::move(generated.error()));
    auto output_ids = std::move(generated->token_ids);
    auto text = measure(stats ? &stats->target_decode_ns : nullptr,
                        [&] { return package_.target_tokenizer().decode(output_ids); });
    if (!text) return std::unexpected(std::move(text.error()));
    return Translation{
        .text = std::move(*text),
        .token_ids = std::move(output_ids),
        .score = generated->score,
    };
}

auto Translator::translate_batch(std::span<const std::string> sources, DecodeOptions options, InferenceStats* stats)
    -> Result<std::vector<Translation>> {
    if (sources.empty()) return std::vector<Translation>{};
    if (options.beam_size.value_or(package_.config()["decode"]["beam_size"].as<std::int32_t>()) != 1) {
        std::vector<Translation> results;
        for (const auto& source : sources) {
            auto result = translate(source, options, stats);
            if (!result) return std::unexpected(std::move(result.error()));
            results.push_back(std::move(*result));
        }
        return results;
    }
    ScopedTimer timer(stats ? &stats->translate_ns : nullptr, stats ? &stats->graph_compile_ns : nullptr);
    const auto config = package_.config()["decode"];
    auto decode = resolve_options(config, options);
    if (!decode) return std::unexpected(std::move(decode.error()));
    std::vector<std::vector<std::int32_t>> ids;
    std::vector<std::size_t> limits;
    for (const auto& source : sources) {
        auto encoded = measure(stats ? &stats->source_tokenize_ns : nullptr,
                               [&] { return package_.source_tokenizer().encode(source); });
        if (!encoded) return std::unexpected(std::move(encoded.error()));
        if (encoded->empty() || encoded->back() != config["source_end_id"].as<std::int32_t>())
            encoded->push_back(config["source_end_id"].as<std::int32_t>());
        limits.push_back(encoded->size() + (*decode)["maximum_extra_tokens"].as<std::size_t>());
        ids.push_back(std::move(*encoded));
    }
    std::vector<std::size_t> order(ids.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(),
                     [&](auto left, auto right) { return ids[left].size() < ids[right].size(); });
    std::vector<Generation> decoded(ids.size());
    for (std::size_t begin = 0; begin < order.size(); begin += batch_size_) {
        const auto count = std::min(batch_size_, order.size() - begin);
        std::vector<std::vector<std::int32_t>> batch_ids;
        std::vector<std::size_t> batch_limits;
        for (std::size_t offset = 0; offset < count; ++offset) {
            batch_ids.push_back(std::move(ids[order[begin + offset]]));
            batch_limits.push_back(limits[order[begin + offset]]);
        }
        auto source = model_->encode(batch_ids, stats);
        if (!source) return std::unexpected(std::move(source.error()));
        auto state = model_->create_state(count, *std::ranges::max_element(batch_limits));
        if (!state) return std::unexpected(std::move(state.error()));
        std::vector<SearchOptions> settings;
        for (auto limit : batch_limits)
            settings.push_back({.vocabulary_size = config["vocabulary_size"].as<std::size_t>(),
                                .end_id = config["end_id"].as<std::int32_t>(),
                                .pad_id = config["pad_id"].as<std::int32_t>(),
                                .maximum_steps = limit,
                                .length_penalty = (*decode)["length_penalty"].as<float>(),
                                .compute_score = options.compute_score,
                                .unfinished_score_length = (*decode)["maximum_extra_tokens"].as<std::size_t>()});
        const std::vector<std::int32_t> initial(count, config["begin_id"].as<std::int32_t>());
        tensor::Tensor score_storage;
        auto batch = Decoder::generate_batch(
            initial, settings,
            [&](const GreedyRequest& request) -> Result<TokenScores> {
                auto logits = model_->forward(*source, request.tokens, count, &*state, stats);
                if (!logits) return std::unexpected(std::move(logits.error()));
                score_storage = std::move(*logits);
                auto values = score_storage.data<float>();
                if (!values) return std::unexpected(std::move(values.error()));
                return TokenScores{*values, ScoreKind::LOGITS};
            },
            stats);
        if (!batch) return std::unexpected(std::move(batch.error()));
        for (std::size_t offset = 0; offset < count; ++offset)
            decoded[order[begin + offset]] = std::move((*batch)[offset]);
    }
    std::vector<Translation> results;
    for (auto& result : decoded) {
        auto output_ids = std::move(result.token_ids);
        auto text = measure(stats ? &stats->target_decode_ns : nullptr,
                            [&] { return package_.target_tokenizer().decode(output_ids); });
        if (!text) return std::unexpected(std::move(text.error()));
        results.push_back({std::move(*text), std::move(output_ids), result.score});
    }
    return results;
}

} // namespace kidi::inference