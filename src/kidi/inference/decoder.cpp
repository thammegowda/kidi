#include "kidi/inference/decoder.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <queue>

namespace kidi::inference {
namespace {
using Clock = std::chrono::steady_clock;
auto is_end(std::int32_t token, const SearchOptions& options) -> bool {
    return options.stop_on_eos &&
           (token == options.end_id || std::ranges::find(options.stop_ids, token) != options.stop_ids.end());
}
auto elapsed(Clock::time_point start) -> std::uint64_t {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count();
}
struct Beam {
    std::vector<std::int32_t> tokens;
    float score;
    std::size_t length;
    bool active;
};
struct Candidate {
    float score;
    std::size_t beam;
    std::int32_t token;
};
struct Greater {
    auto operator()(const Candidate& left, const Candidate& right) const -> bool { return left.score > right.score; }
};
auto offer(std::priority_queue<Candidate, std::vector<Candidate>, Greater>& candidates, Candidate candidate,
           std::size_t limit) -> void {
    if (candidates.size() < limit)
        candidates.push(candidate);
    else if (candidate.score > candidates.top().score) {
        candidates.pop();
        candidates.push(candidate);
    }
}
} // namespace

auto normalized_score(float score, std::size_t length, float alpha) -> float {
    if (alpha <= 0.0F) return score;
    return score / (std::pow(5.0F + static_cast<float>(length), alpha) / std::pow(6.0F, alpha));
}

GreedyState::GreedyState(SearchOptions options)
    : options_(options), stop_ids_(options.stop_ids.begin(), options.stop_ids.end()), token_(options.pad_id) {
    options_.stop_ids = {};
    result_.token_ids.reserve(options.maximum_steps);
}
auto GreedyState::create(SearchOptions options) -> Result<GreedyState> {
    const auto valid = [&](auto token) {
        return token >= 0 && static_cast<std::size_t>(token) < options.vocabulary_size;
    };
    if (!options.vocabulary_size || options.vocabulary_size > INT32_MAX || options.beam_size != 1 ||
        !options.maximum_steps || !valid(options.end_id) || !valid(options.pad_id) ||
        !std::ranges::all_of(options.stop_ids, valid) || !std::isfinite(options.length_penalty) ||
        options.length_penalty < 0)
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "invalid greedy generation settings"});
    return GreedyState(options);
}
auto GreedyState::accept(TokenScores scores) -> Result<void> {
    if (finished_) return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "generation is already finished"});
    if (!scores.greedy_tokens.empty() ||
        (scores.greedy_token ? options_.compute_score || !scores.values.empty() || *scores.greedy_token < 0 ||
                                   static_cast<std::size_t>(*scores.greedy_token) >= options_.vocabulary_size
                             : scores.values.size() != options_.vocabulary_size))
        return std::unexpected(Error{ErrorCode::RUNTIME, "invalid greedy scores or selected token"});
    if (std::ranges::any_of(scores.values, [](float value) {
            return std::isnan(value) || value == std::numeric_limits<float>::infinity();
        }))
        return std::unexpected(Error{ErrorCode::RUNTIME, "decoder returned invalid scores"});
    const auto best = std::ranges::max_element(scores.values);
    if (!scores.greedy_token && !std::isfinite(*best))
        return std::unexpected(Error{ErrorCode::RUNTIME, "decoder has no finite token scores"});
    token_ = scores.greedy_token ? *scores.greedy_token : static_cast<std::int32_t>(best - scores.values.begin());
    if (options_.compute_score) {
        if (scores.kind == ScoreKind::LOGITS) {
            float sum = 0;
            for (auto value : scores.values) sum += std::exp(value - *best);
            result_.score -= std::log(sum);
        } else
            result_.score += *best;
    }
    ++result_.decoder_steps;
    auto settings = options_;
    settings.stop_ids = stop_ids_;
    const bool ended = is_end(token_, settings);
    if (!ended && (!options_.stop_on_eos || token_ != options_.pad_id)) result_.token_ids.push_back(token_);
    finished_ = ended || result_.decoder_steps == options_.maximum_steps;
    if (finished_) {
        const auto length =
            ended ? result_.decoder_steps : options_.unfinished_score_length.value_or(options_.maximum_steps);
        result_.score = normalized_score(result_.score, length, options_.length_penalty);
    }
    return {};
}

auto Decoder::generate(std::span<const std::int32_t> prompt, const SearchOptions& options, const ScoreFunction& score,
                       InferenceStats* stats) -> Result<Generation> {
    const auto valid_token = [&](auto token) {
        return token >= 0 && static_cast<std::size_t>(token) < options.vocabulary_size;
    };
    if (prompt.empty() || !score || options.vocabulary_size == 0 || options.vocabulary_size > INT32_MAX ||
        options.beam_size == 0 || options.beam_size > options.vocabulary_size || options.maximum_steps == 0 ||
        !valid_token(options.end_id) || !valid_token(options.pad_id) || !std::ranges::all_of(prompt, valid_token) ||
        !std::ranges::all_of(options.stop_ids, valid_token) || !std::isfinite(options.length_penalty) ||
        options.length_penalty < 0)
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "invalid generation request"});

    std::vector<Beam> beams(options.beam_size, Beam{{prompt.begin(), prompt.end()},
                                                    0.0F,
                                                    options.unfinished_score_length.value_or(options.maximum_steps),
                                                    true});
    std::size_t steps = 0;
    for (std::size_t step = 1; step <= options.maximum_steps; ++step) {
        std::vector<std::size_t> active;
        std::vector<std::int32_t> prefixes;
        for (std::size_t index = 0; index < beams.size(); ++index) {
            if (beams[index].active && (step != 1 || index == 0)) {
                active.push_back(index);
                prefixes.insert(prefixes.end(), beams[index].tokens.begin(), beams[index].tokens.end());
            }
        }
        if (active.empty()) break;
        auto started = Clock::now();
        auto scores = score({prefixes, active, beams.front().tokens.size(), step - 1});
        if (stats) stats->decoder_ns += elapsed(started);
        if (!scores) return std::unexpected(std::move(scores.error()));
        if (!scores->greedy_tokens.empty())
            return std::unexpected(Error{ErrorCode::RUNTIME, "single-request decoder cannot accept batched tokens"});
        if (scores->greedy_token && (options.beam_size != 1 || options.compute_score || !scores->values.empty() ||
                                     !valid_token(*scores->greedy_token)))
            return std::unexpected(Error{ErrorCode::RUNTIME, "invalid preselected greedy token or search policy"});
        if (!scores->greedy_token && scores->values.size() != active.size() * options.vocabulary_size)
            return std::unexpected(Error{ErrorCode::RUNTIME, "decoder returned an incompatible score shape"});
        if (std::ranges::any_of(scores->values, [](float value) {
                return std::isnan(value) || value == std::numeric_limits<float>::infinity();
            }))
            return std::unexpected(Error{ErrorCode::RUNTIME, "decoder returned invalid scores"});
        steps = step;
        if (stats) ++stats->decoder_steps;
        started = Clock::now();
        if (options.beam_size == 1) {
            const auto best = std::ranges::max_element(scores->values);
            if (!scores->greedy_token && !std::isfinite(*best))
                return std::unexpected(Error{ErrorCode::RUNTIME, "decoder has no finite token scores"});
            const auto token =
                scores->greedy_token ? *scores->greedy_token : static_cast<std::int32_t>(best - scores->values.begin());
            if (options.compute_score) {
                if (scores->kind == ScoreKind::LOGITS) {
                    float sum = 0;
                    for (auto value : scores->values) sum += std::exp(value - *best);
                    beams.front().score -= std::log(sum);
                } else
                    beams.front().score += *best;
            }
            beams.front().tokens.push_back(token);
            beams.front().active = !is_end(token, options);
            if (!beams.front().active) beams.front().length = step;
            if (stats) stats->host_search_ns += elapsed(started);
            if (!beams.front().active) break;
            continue;
        }
        std::vector<float> normalized;
        auto probabilities = scores->values;
        if (scores->kind == ScoreKind::LOGITS) {
            normalized.assign(probabilities.begin(), probabilities.end());
            for (std::size_t row = 0; row < active.size(); ++row) {
                auto values = std::span(normalized).subspan(row * options.vocabulary_size, options.vocabulary_size);
                const auto maximum = *std::ranges::max_element(values);
                if (!std::isfinite(maximum))
                    return std::unexpected(Error{ErrorCode::RUNTIME, "decoder has no finite token scores"});
                float sum = 0;
                for (auto value : values) sum += std::exp(value - maximum);
                const auto normalization = maximum + std::log(sum);
                for (auto& value : values) value -= normalization;
            }
            probabilities = normalized;
        }
        std::vector<std::size_t> batches(beams.size(), beams.size());
        for (std::size_t index = 0; index < active.size(); ++index) batches[active[index]] = index;
        std::priority_queue<Candidate, std::vector<Candidate>, Greater> candidates;
        for (std::size_t index = 0; index < beams.size(); ++index) {
            if (step == 1 && index != 0) continue;
            const auto& beam = beams[index];
            if (!beam.active) {
                offer(candidates, {beam.score, index, options.pad_id}, options.beam_size);
                continue;
            }
            const auto offset = batches[index] * options.vocabulary_size;
            for (std::size_t token = 0; token < options.vocabulary_size; ++token) {
                const auto probability = probabilities[offset + token];
                if (probability == -std::numeric_limits<float>::infinity()) continue;
                offer(candidates, {beam.score + probability, index, static_cast<std::int32_t>(token)},
                      options.beam_size);
            }
        }
        if (candidates.size() != options.beam_size)
            return std::unexpected(Error{ErrorCode::RUNTIME, "beam search found too few finite candidates"});
        std::vector<Candidate> ranked;
        while (!candidates.empty()) {
            ranked.push_back(candidates.top());
            candidates.pop();
        }
        std::ranges::sort(ranked, [](const auto& left, const auto& right) { return left.score > right.score; });
        std::vector<Beam> next;
        bool any_active = false;
        for (const auto& candidate : ranked) {
            const auto& previous = beams[candidate.beam];
            auto tokens = previous.tokens;
            tokens.push_back(candidate.token);
            const bool ended = previous.active && is_end(candidate.token, options);
            const bool is_active = previous.active && !ended;
            next.push_back({std::move(tokens), candidate.score, ended ? step : previous.length, is_active});
            any_active |= is_active;
        }
        beams = std::move(next);
        if (stats) stats->host_search_ns += elapsed(started);
        if (!any_active) break;
    }
    const auto best = std::ranges::max_element(beams, [&](const auto& left, const auto& right) {
        return normalized_score(left.score, left.length, options.length_penalty) <
               normalized_score(right.score, right.length, options.length_penalty);
    });
    std::vector<std::int32_t> result(best->tokens.begin() + prompt.size(), best->tokens.end());
    if (auto end = std::ranges::find_if(result, [&](auto token) { return is_end(token, options); });
        end != result.end())
        result.erase(end, result.end());
    if (options.stop_on_eos) std::erase(result, options.pad_id);
    return Generation{std::move(result), normalized_score(best->score, best->length, options.length_penalty), steps};
}
} // namespace kidi::inference

namespace kidi::inference {
auto Decoder::generate_batch(std::span<const std::int32_t> initial_tokens, std::span<const SearchOptions> options,
                             const BatchScoreFunction& score, InferenceStats* stats)
    -> Result<std::vector<Generation>> {
    if (options.empty() || initial_tokens.size() != options.size() || !score)
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "invalid batch generation request"});
    const auto vocabulary = options.front().vocabulary_size;
    std::vector<GreedyState> states;
    states.reserve(options.size());
    for (std::size_t row = 0; row < options.size(); ++row) {
        const auto& settings = options[row];
        if (settings.vocabulary_size != vocabulary || initial_tokens[row] < 0 ||
            static_cast<std::size_t>(initial_tokens[row]) >= vocabulary)
            return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "invalid batch generation settings"});
        auto state = GreedyState::create(settings);
        if (!state) return std::unexpected(std::move(state.error()));
        states.push_back(std::move(*state));
    }
    std::vector<std::int32_t> tokens(initial_tokens.begin(), initial_tokens.end());
    std::size_t remaining = options.size();
    std::vector<std::size_t> active;
    active.reserve(options.size());
    for (std::size_t step = 0; remaining; ++step) {
        active.clear();
        for (std::size_t row = 0; row < options.size(); ++row)
            if (!states[row].finished()) active.push_back(row);
        auto start = Clock::now();
        auto scores = score({tokens, step, active});
        if (stats) {
            stats->decoder_ns += elapsed(start);
            ++stats->decoder_steps;
        }
        if (!scores) return std::unexpected(std::move(scores.error()));
        const bool preselected = !scores->greedy_tokens.empty();
        if (scores->greedy_token ||
            (preselected ? scores->greedy_tokens.size() != options.size() || !scores->values.empty() ||
                               std::ranges::any_of(options, [](const auto& settings) { return settings.compute_score; })
                         : scores->values.size() != options.size() * vocabulary))
            return std::unexpected(Error{ErrorCode::RUNTIME, "batch decoder returned incompatible score shape"});
        start = Clock::now();
        for (std::size_t row = 0; row < options.size(); ++row) {
            auto& state = states[row];
            if (state.finished()) continue;
            const auto accepted = state.accept(
                preselected ? TokenScores{{}, scores->kind, scores->greedy_tokens[row]}
                            : TokenScores{scores->values.subspan(row * vocabulary, vocabulary), scores->kind});
            if (!accepted) return std::unexpected(accepted.error());
            tokens[row] = state.finished() ? options[row].pad_id : state.token();
            if (state.finished()) --remaining;
        }
        if (stats) stats->host_search_ns += elapsed(start);
    }
    std::vector<Generation> results;
    results.reserve(states.size());
    for (auto& state : states) results.push_back(std::move(state).result());
    return results;
}
} // namespace kidi::inference