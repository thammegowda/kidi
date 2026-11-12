#include "kidi/inference/decoder.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>

namespace {
using namespace kidi::inference;
auto settings(std::size_t maximum = 8, bool stop = false, bool score = false) -> SearchOptions {
    return {.vocabulary_size = 6,
            .end_id = 5,
            .pad_id = 0,
            .maximum_steps = maximum,
            .compute_score = score,
            .stop_on_eos = stop};
}
auto check() -> bool {
    const std::array<std::int32_t, 3> drafts{2, 3, 4};
    for (std::size_t matched = 0; matched <= drafts.size(); ++matched) {
        auto state = kidi::inference::GreedyState::create(settings());
        if (!state) return false;
        std::array<std::int32_t, 4> selected{2, 3, 4, 1};
        if (matched < drafts.size()) selected[matched] = 1;
        const auto accepted = state->accept_speculative(drafts, TokenScores{{}, ScoreKind::LOGITS, {}, selected});
        if (!accepted || accepted->accepted_drafts != matched || accepted->committed_inputs != matched + 1 ||
            state->result().decoder_steps != matched + 1 ||
            state->result().token_ids != std::vector<std::int32_t>(selected.begin(), selected.begin() + matched + 1))
            return false;
    }
    for (std::size_t maximum = 1; maximum <= 4; ++maximum) {
        auto state = GreedyState::create(settings(maximum));
        const std::array<std::int32_t, 4> selected{2, 3, 4, 1};
        const auto accepted = state->accept_speculative(drafts, TokenScores{{}, ScoreKind::LOGITS, {}, selected});
        if (!accepted || accepted->committed_inputs != maximum ||
            accepted->accepted_drafts != std::min(maximum, drafts.size()) || !state->finished())
            return false;
    }
    for (std::size_t stop = 0; stop < 4; ++stop) {
        auto state = GreedyState::create(settings(8, true));
        auto proposed = drafts;
        std::array<std::int32_t, 4> selected{2, 3, 4, 1};
        selected[stop] = 5;
        if (stop < drafts.size()) proposed[stop] = 5;
        const auto accepted = state->accept_speculative(proposed, TokenScores{{}, ScoreKind::LOGITS, {}, selected});
        if (!accepted || accepted->committed_inputs != stop + 1 || !state->finished() ||
            state->result().token_ids.size() != stop || state->result().truncated ||
            state->accept_speculative({}, TokenScores{{}, ScoreKind::LOGITS, {}, selected}))
            return false;
    }
    const std::array<std::int32_t, 1> extra_stop{3};
    auto stopped_options = settings(8, true);
    stopped_options.stop_ids = extra_stop;
    auto stopped = GreedyState::create(stopped_options);
    const std::array<std::int32_t, 4> selected{2, 3, 4, 1};
    const auto stopped_round = stopped->accept_speculative(drafts, TokenScores{{}, ScoreKind::LOGITS, {}, selected});
    if (!stopped_round || stopped_round->committed_inputs != 2 || stopped->result().token_ids != std::vector{2})
        return false;
    std::array<float, 24> logits{};
    for (std::size_t row = 0; row < 4; ++row) logits[row * 6 + selected[row]] = 2.F;
    auto scored = GreedyState::create(settings(4, true, true));
    auto serial = GreedyState::create(settings(4, true, true));
    const auto scored_round = scored->accept_speculative(drafts, TokenScores{logits});
    for (std::size_t row = 0; row < 4; ++row)
        if (!serial->accept(TokenScores{std::span(logits).subspan(row * 6, 6)})) return false;
    if (!scored_round || scored->result().token_ids != serial->result().token_ids ||
        std::abs(scored->result().score - serial->result().score) > 1e-6F)
        return false;
    const auto unchanged = [&](std::span<const std::int32_t> proposed, TokenScores scores) {
        auto state = GreedyState::create(settings());
        return !state->accept_speculative(proposed, scores) && state->result().decoder_steps == 0;
    };
    if (!unchanged(drafts, TokenScores{}) ||
        !unchanged(drafts, TokenScores{{}, ScoreKind::LOGITS, {}, std::span(selected).first(3)}) ||
        !unchanged(std::array<std::int32_t, 1>{6}, TokenScores{logits}) ||
        !unchanged(drafts, TokenScores{{}, ScoreKind::LOGITS, 2}))
        return false;
    auto invalid_tokens = selected;
    invalid_tokens.back() = -1;
    if (!unchanged(drafts, TokenScores{{}, ScoreKind::LOGITS, {}, invalid_tokens})) return false;
    for (const auto invalid : {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity()}) {
        auto invalid_logits = logits;
        invalid_logits.back() = invalid;
        if (!unchanged(drafts, TokenScores{invalid_logits})) return false;
    }
    auto invalid_logits = logits;
    std::ranges::fill(std::span(invalid_logits).last(6), -std::numeric_limits<float>::infinity());
    if (!unchanged(drafts, TokenScores{invalid_logits}) ||
        !unchanged(drafts, TokenScores{logits, ScoreKind::LOGITS, {}, selected}))
        return false;
    auto no_scores = GreedyState::create(settings(8, false, true));
    if (no_scores->accept_speculative(drafts, TokenScores{{}, ScoreKind::LOGITS, {}, selected})) return false;
    auto empty_draft = GreedyState::create(settings());
    const std::array<std::int32_t, 1> bonus{4};
    const auto bonus_round = empty_draft->accept_speculative({}, TokenScores{{}, ScoreKind::LOGITS, {}, bonus});
    return bonus_round && bonus_round->accepted_drafts == 0 && bonus_round->committed_inputs == 1 &&
           empty_draft->result().token_ids == std::vector{4};
}
} // namespace

auto main() -> int {
    if (!check()) {
        std::cerr << "speculative greedy verification failed\n";
        return 1;
    }
    std::cout << "speculative greedy verification passed\n";
    return 0;
}
