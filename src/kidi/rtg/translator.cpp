#include "kidi/rtg/translator.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <queue>
#include <utility>
#include <vector>

namespace kidi::rtg {
namespace {

struct Beam {
    std::vector<std::int32_t> token_ids;
    float score;
    std::int32_t length;
    bool active;
};

struct Candidate {
    float score;
    std::size_t beam;
    std::int32_t token_id;
};

struct CandidateGreater {
    bool operator()(const Candidate& left, const Candidate& right) const noexcept { return left.score > right.score; }
};

void offer(std::priority_queue<Candidate, std::vector<Candidate>, CandidateGreater>& candidates, Candidate candidate,
           std::size_t limit) {
    if (candidates.size() < limit) {
        candidates.push(candidate);
    } else if (candidate.score > candidates.top().score) {
        candidates.pop();
        candidates.push(candidate);
    }
}

float normalized_score(const Beam& beam, float alpha) {
    if (alpha <= 0.0F) return beam.score;
    const auto penalty = std::pow(5.0F + static_cast<float>(beam.length), alpha) / std::pow(6.0F, alpha);
    return beam.score / penalty;
}

Result<model::DecodeDefaults> resolve_options(const model::ModelManifest& manifest, const DecodeOptions& options) {
    model::DecodeDefaults result = {
        .beam_size = options.beam_size.value_or(manifest.decode_defaults.beam_size),
        .maximum_extra_tokens = options.maximum_extra_tokens.value_or(manifest.decode_defaults.maximum_extra_tokens),
        .length_penalty = options.length_penalty.value_or(manifest.decode_defaults.length_penalty),
    };
    if (result.beam_size <= 0 || result.beam_size > manifest.limits.maximum_beam_size) {
        return std::unexpected(
            Error{ErrorCode::INVALID_ARGUMENT,
                  "beam size must be between 1 and " + std::to_string(manifest.limits.maximum_beam_size)});
    }
    if (result.maximum_extra_tokens <= 0 || result.maximum_extra_tokens > manifest.limits.maximum_extra_tokens) {
        return std::unexpected(Error{
            ErrorCode::INVALID_ARGUMENT,
            "maximum extra tokens must be between 1 and " + std::to_string(manifest.limits.maximum_extra_tokens)});
    }
    if (result.length_penalty < 0.0F) {
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "length penalty cannot be negative"});
    }
    return result;
}

} // namespace

Translator::Translator(Package package, Encoder encoder, Decoder decoder) noexcept
    : package_(std::move(package)), encoder_(std::move(encoder)), decoder_(std::move(decoder)) {}

Result<Translator> Translator::load(const std::filesystem::path& model_directory) {
    auto package = Package::load(model_directory);
    if (!package) return std::unexpected(std::move(package.error()));
    auto encoder = Encoder::create(*package);
    if (!encoder) return std::unexpected(std::move(encoder.error()));
    auto decoder = Decoder::create(*package);
    if (!decoder) return std::unexpected(std::move(decoder.error()));
    return Translator(std::move(*package), std::move(*encoder), std::move(*decoder));
}

Result<Translation> Translator::translate(std::string_view source, DecodeOptions options) {
    const auto& manifest = package_.manifest();
    auto decode = resolve_options(manifest, options);
    if (!decode) return std::unexpected(std::move(decode.error()));
    auto source_ids = package_.source_tokenizer().encode(source);
    if (!source_ids) return std::unexpected(std::move(source_ids.error()));
    if (source_ids->empty() || source_ids->back() != manifest.special_tokens.end) {
        source_ids->push_back(manifest.special_tokens.end);
    }
    auto memory = encoder_.run(*source_ids);
    if (!memory) return std::unexpected(std::move(memory.error()));

    const auto beam_size = static_cast<std::size_t>(decode->beam_size);
    const auto vocabulary_size = static_cast<std::size_t>(manifest.architecture.target_vocabulary_size);
    std::vector<Beam> beams(beam_size, Beam{
                                           .token_ids = {manifest.special_tokens.begin},
                                           .score = 0.0F,
                                           .length = decode->maximum_extra_tokens,
                                           .active = true,
                                       });
    const auto maximum_steps = source_ids->size() + static_cast<std::size_t>(decode->maximum_extra_tokens);
    for (std::size_t step = 1; step <= maximum_steps; ++step) {
        std::vector<std::size_t> active_beams;
        active_beams.reserve(beam_size);
        for (std::size_t beam_index = 0; beam_index < beam_size; ++beam_index) {
            if (beams[beam_index].active && (step != 1 || beam_index == 0)) {
                active_beams.push_back(beam_index);
            }
        }
        if (active_beams.empty()) break;

        std::vector<std::int32_t> prefixes;
        prefixes.reserve(active_beams.size() * beams.front().token_ids.size());
        for (const auto beam_index : active_beams) {
            const auto& beam = beams[beam_index];
            prefixes.insert(prefixes.end(), beam.token_ids.begin(), beam.token_ids.end());
        }
        auto log_probabilities = decoder_.next(*memory, prefixes, active_beams.size());
        if (!log_probabilities) return std::unexpected(std::move(log_probabilities.error()));
        if (log_probabilities->size() != active_beams.size() * vocabulary_size) {
            return std::unexpected(Error{ErrorCode::RUNTIME, "decoder returned an incompatible probability shape"});
        }
        std::vector<std::size_t> probability_batches(beam_size, beam_size);
        for (std::size_t batch = 0; batch < active_beams.size(); ++batch) {
            probability_batches[active_beams[batch]] = batch;
        }

        std::priority_queue<Candidate, std::vector<Candidate>, CandidateGreater> candidates;
        for (std::size_t beam_index = 0; beam_index < beam_size; ++beam_index) {
            const auto& beam = beams[beam_index];
            if (step == 1 && beam_index != 0) continue;
            if (!beam.active) {
                offer(candidates,
                      Candidate{.score = beam.score, .beam = beam_index, .token_id = manifest.special_tokens.pad},
                      beam_size);
                continue;
            }
            const auto offset = probability_batches[beam_index] * vocabulary_size;
            for (std::size_t token = 0; token < vocabulary_size; ++token) {
                const auto probability = (*log_probabilities)[offset + token];
                if (probability == -std::numeric_limits<float>::infinity()) continue;
                offer(candidates,
                      Candidate{.score = beam.score + probability,
                                .beam = beam_index,
                                .token_id = static_cast<std::int32_t>(token)},
                      beam_size);
            }
        }
        if (candidates.size() != beam_size) {
            return std::unexpected(Error{ErrorCode::RUNTIME, "beam search found too few finite candidates"});
        }

        std::vector<Candidate> ranked;
        ranked.reserve(beam_size);
        while (!candidates.empty()) {
            ranked.push_back(candidates.top());
            candidates.pop();
        }
        std::ranges::sort(ranked,
                          [](const Candidate& left, const Candidate& right) { return left.score > right.score; });

        std::vector<Beam> next_beams;
        next_beams.reserve(beam_size);
        bool any_active = false;
        for (const auto& candidate : ranked) {
            const auto& previous = beams[candidate.beam];
            auto token_ids = previous.token_ids;
            token_ids.push_back(candidate.token_id);
            const bool ended = previous.active && candidate.token_id == manifest.special_tokens.end;
            const bool active = previous.active && !ended;
            next_beams.push_back(Beam{
                .token_ids = std::move(token_ids),
                .score = candidate.score,
                .length = ended ? static_cast<std::int32_t>(step) : previous.length,
                .active = active,
            });
            any_active = any_active || active;
        }
        beams = std::move(next_beams);
        if (!any_active) break;
    }

    const auto best = std::ranges::max_element(beams, [&](const Beam& left, const Beam& right) {
        return normalized_score(left, decode->length_penalty) < normalized_score(right, decode->length_penalty);
    });
    std::vector<std::int32_t> output_ids(best->token_ids.begin() + 1, best->token_ids.end());
    if (const auto end = std::ranges::find(output_ids, manifest.special_tokens.end); end != output_ids.end()) {
        output_ids.erase(end, output_ids.end());
    }
    std::erase(output_ids, manifest.special_tokens.pad);
    auto text = package_.target_tokenizer().decode(output_ids);
    if (!text) return std::unexpected(std::move(text.error()));
    return Translation{
        .text = std::move(*text),
        .token_ids = std::move(output_ids),
        .score = normalized_score(*best, decode->length_penalty),
    };
}

} // namespace kidi::rtg