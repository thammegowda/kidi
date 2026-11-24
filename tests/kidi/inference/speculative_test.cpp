#include "kidi/inference/decoder.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

#include <gtest/gtest.h>

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

constexpr std::array<std::int32_t, 3> drafts{2, 3, 4};
constexpr std::array<std::int32_t, 4> selected{2, 3, 4, 1};
constexpr std::array<std::int32_t, 1> bonus{4};

auto selected_logits() -> std::array<float, 24> {
    std::array<float, 24> logits{};
    for (std::size_t row = 0; row < 4; ++row) logits[row * 6 + selected[row]] = 2.F;
    return logits;
}

auto unchanged(std::span<const std::int32_t> proposed, TokenScores scores) -> bool {
    auto state = GreedyState::create(settings());
    return !state->accept_speculative(proposed, scores) && state->result().decoder_steps == 0;
}

TEST(SpeculativeTest, AcceptsDraftPrefix) {
    for (std::size_t matched = 0; matched <= drafts.size(); ++matched) {
        SCOPED_TRACE(matched);
        auto state = kidi::inference::GreedyState::create(settings());
        ASSERT_TRUE(state);
        std::array<std::int32_t, 4> chosen = selected;
        if (matched < drafts.size()) chosen[matched] = 1;
        const auto accepted = state->accept_speculative(drafts, TokenScores{{}, ScoreKind::LOGITS, {}, chosen});
        ASSERT_TRUE(accepted);
        EXPECT_EQ(accepted->accepted_drafts, matched);
        EXPECT_EQ(accepted->committed_inputs, matched + 1);
        EXPECT_EQ(state->result().decoder_steps, matched + 1);
        EXPECT_EQ(state->result().token_ids, std::vector<std::int32_t>(chosen.begin(), chosen.begin() + matched + 1));
    }
}

TEST(SpeculativeTest, StopsAtMaximumSteps) {
    for (std::size_t maximum = 1; maximum <= 4; ++maximum) {
        SCOPED_TRACE(maximum);
        auto state = GreedyState::create(settings(maximum));
        const auto accepted = state->accept_speculative(drafts, TokenScores{{}, ScoreKind::LOGITS, {}, selected});
        ASSERT_TRUE(accepted);
        EXPECT_EQ(accepted->committed_inputs, maximum);
        EXPECT_EQ(accepted->accepted_drafts, std::min(maximum, drafts.size()));
        EXPECT_TRUE(state->finished());
    }
}

TEST(SpeculativeTest, StopsAtEndToken) {
    for (std::size_t stop = 0; stop < 4; ++stop) {
        SCOPED_TRACE(stop);
        auto state = GreedyState::create(settings(8, true));
        auto proposed = drafts;
        std::array<std::int32_t, 4> chosen = selected;
        chosen[stop] = 5;
        if (stop < drafts.size()) proposed[stop] = 5;
        const auto accepted = state->accept_speculative(proposed, TokenScores{{}, ScoreKind::LOGITS, {}, chosen});
        ASSERT_TRUE(accepted);
        EXPECT_EQ(accepted->committed_inputs, stop + 1);
        EXPECT_TRUE(state->finished());
        EXPECT_EQ(state->result().token_ids.size(), stop);
        EXPECT_FALSE(state->result().truncated);
        EXPECT_FALSE(state->accept_speculative({}, TokenScores{{}, ScoreKind::LOGITS, {}, chosen}));
    }
}

TEST(SpeculativeTest, StopsAtStopToken) {
    const std::array<std::int32_t, 1> extra_stop{3};
    auto stopped_options = settings(8, true);
    stopped_options.stop_ids = extra_stop;
    auto stopped = GreedyState::create(stopped_options);
    const auto stopped_round = stopped->accept_speculative(drafts, TokenScores{{}, ScoreKind::LOGITS, {}, selected});
    ASSERT_TRUE(stopped_round);
    EXPECT_EQ(stopped_round->committed_inputs, 2);
    EXPECT_EQ(stopped->result().token_ids, std::vector{2});
}

TEST(SpeculativeTest, ScoredMatchesSerial) {
    const auto logits = selected_logits();
    auto scored = GreedyState::create(settings(4, true, true));
    auto serial = GreedyState::create(settings(4, true, true));
    const auto scored_round = scored->accept_speculative(drafts, TokenScores{logits});
    for (std::size_t row = 0; row < 4; ++row)
        ASSERT_TRUE(serial->accept(TokenScores{std::span(logits).subspan(row * 6, 6)})) << "row " << row;
    ASSERT_TRUE(scored_round);
    EXPECT_EQ(scored->result().token_ids, serial->result().token_ids);
    EXPECT_NEAR(scored->result().score, serial->result().score, 1e-6F);
}

TEST(SpeculativeTest, RejectsInvalidScores) {
    const auto logits = selected_logits();
    EXPECT_TRUE(unchanged(drafts, TokenScores{}));
    EXPECT_TRUE(unchanged(drafts, TokenScores{{}, ScoreKind::LOGITS, {}, std::span(selected).first(3)}));
    EXPECT_TRUE(unchanged(std::array<std::int32_t, 1>{6}, TokenScores{logits}));
    EXPECT_TRUE(unchanged(drafts, TokenScores{{}, ScoreKind::LOGITS, 2}));
    auto invalid_tokens = selected;
    invalid_tokens.back() = -1;
    EXPECT_TRUE(unchanged(drafts, TokenScores{{}, ScoreKind::LOGITS, {}, invalid_tokens}));
    for (const auto invalid : {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity()}) {
        auto invalid_logits = logits;
        invalid_logits.back() = invalid;
        EXPECT_TRUE(unchanged(drafts, TokenScores{invalid_logits})) << "invalid logit " << invalid;
    }
    auto invalid_logits = logits;
    std::ranges::fill(std::span(invalid_logits).last(6), -std::numeric_limits<float>::infinity());
    EXPECT_TRUE(unchanged(drafts, TokenScores{invalid_logits}));
    EXPECT_TRUE(unchanged(drafts, TokenScores{logits, ScoreKind::LOGITS, {}, selected}));
    auto no_scores = GreedyState::create(settings(8, false, true));
    EXPECT_FALSE(no_scores->accept_speculative(drafts, TokenScores{{}, ScoreKind::LOGITS, {}, selected}));
}

TEST(SpeculativeTest, EmptyDraftBonusToken) {
    auto empty_draft = GreedyState::create(settings());
    const auto bonus_round = empty_draft->accept_speculative({}, TokenScores{{}, ScoreKind::LOGITS, {}, bonus});
    ASSERT_TRUE(bonus_round);
    EXPECT_EQ(bonus_round->accepted_drafts, 0);
    EXPECT_EQ(bonus_round->committed_inputs, 1);
    EXPECT_EQ(empty_draft->result().token_ids, std::vector{4});
}

// Standalone draft-LM callbacks (e.g. Qwen draft -> target with independent KV histories and logit verification).
TEST(SpeculativeTest, DraftMatchesTarget) {
    const std::array<std::int32_t, 9> target_rule{1, 2, 3, 4, 2, 1, 4, 3, 5};
    const std::array<std::int32_t, 9> imperfect_draft_rule{1, 2, 4, 4, 2, 1, 3, 3, 5};
    const std::array<std::int32_t, 3> prompt{1, 2, 3};
    const auto run_draft_target = [&](std::span<const std::int32_t> draft_rule, const SearchOptions& opts,
                                      std::size_t max_drafts, std::vector<std::int32_t>& out_history,
                                      SpeculativeStats& stats) {
        std::vector<std::int32_t> target_history(prompt.begin(), prompt.end() - 1);
        std::vector<std::int32_t> draft_history(prompt.begin(), prompt.end());
        std::vector<float> verify_logits;
        std::size_t verify_base = 0;
        const auto last_prompt = prompt.back();
        const auto eval_target = [&](std::span<const std::int32_t> inputs) -> TokenScores {
            verify_logits.assign(inputs.size() * opts.vocabulary_size, -1.F);
            for (std::size_t i = 0; i < inputs.size(); ++i) {
                target_history.push_back(inputs[i]);
                const auto winner =
                    target_rule[(static_cast<std::size_t>(inputs[i]) + target_history.size()) % target_rule.size()];
                verify_logits[i * opts.vocabulary_size + static_cast<std::size_t>(winner)] = 3.F;
            }
            return TokenScores{verify_logits, ScoreKind::LOGITS};
        };
        const auto initial = eval_target(std::span<const std::int32_t>(&last_prompt, 1));
        auto gen = Decoder::generate_speculative(
            initial, opts, max_drafts,
            [&](std::int32_t last_token, std::size_t limit, const GreedyState::StopPredicate& is_stop,
                std::vector<std::int32_t>& out) -> kidi::Result<void> {
                auto cur = last_token;
                for (std::size_t i = 0; i < limit; ++i) {
                    draft_history.push_back(cur);
                    const auto next =
                        draft_rule[(static_cast<std::size_t>(cur) + draft_history.size()) % draft_rule.size()];
                    out.push_back(next);
                    cur = next;
                    if (is_stop && is_stop(next)) break;
                }
                return {};
            },
            [&](std::span<const std::int32_t> inputs) -> kidi::Result<TokenScores> {
                verify_base = target_history.size();
                return eval_target(inputs);
            },
            [&](std::size_t committed_inputs) -> kidi::Result<void> {
                target_history.resize(verify_base + committed_inputs);
                draft_history.assign(target_history.begin(), target_history.end());
                return {};
            },
            &stats);
        out_history = std::move(target_history);
        return gen;
    };

    for (const bool stop_eos : {false, true}) {
        SCOPED_TRACE(stop_eos);
        const auto opts = settings(10, stop_eos, true);
        std::vector<std::int32_t> ref_hist, test_hist;
        SpeculativeStats ref_stats, test_stats;
        const auto ref_gen = run_draft_target(target_rule, opts, 1, ref_hist, ref_stats);
        const auto test_gen = run_draft_target(imperfect_draft_rule, opts, 4, test_hist, test_stats);
        ASSERT_TRUE(ref_gen);
        ASSERT_TRUE(test_gen);
        EXPECT_EQ(ref_gen->token_ids, test_gen->token_ids);
        EXPECT_EQ(ref_gen->decoder_steps, test_gen->decoder_steps);
        EXPECT_NEAR(ref_gen->score, test_gen->score, 1e-5F);
        EXPECT_EQ(ref_hist, test_hist);
        EXPECT_NE(test_stats.verification_steps, 0);
    }
}

TEST(SpeculativeTest, RejectsDraftLimits) {
    const auto noop_draft = [](std::int32_t, std::size_t, const GreedyState::StopPredicate&,
                               std::vector<std::int32_t>&) -> kidi::Result<void> { return {}; };
    const auto noop_verify = [&](std::span<const std::int32_t>) -> kidi::Result<TokenScores> {
        return TokenScores{{}, ScoreKind::LOGITS, {}, bonus};
    };
    EXPECT_FALSE(Decoder::generate_speculative(TokenScores{{}, ScoreKind::LOGITS, 1}, settings(), 0, noop_draft,
                                               noop_verify));
    EXPECT_FALSE(Decoder::generate_speculative(TokenScores{{}, ScoreKind::LOGITS, 1}, settings(),
                                               MAX_DRAFT_TOKENS + 1, noop_draft, noop_verify));
}

TEST(SpeculativeTest, LookupUsesLongestThenMostRecentMatch) {
    EXPECT_EQ(prompt_lookup(std::array{1, 2, 3, 8, 1, 2, 4, 9, 1, 2, 3}, 2), (std::vector{8, 1}));
    EXPECT_EQ(prompt_lookup(std::array{1, 2, 3, 1, 2, 4, 1, 2}, 1), (std::vector{4}));
    EXPECT_EQ(prompt_lookup(std::array{1, 2, 3, 1, 2}, 8), (std::vector{3, 1, 2}));
    EXPECT_TRUE(prompt_lookup(std::array{1, 2, 3}, 4).empty());
    EXPECT_TRUE(prompt_lookup(std::array{1, 2}, 4).empty());
    EXPECT_TRUE(prompt_lookup(std::array{1, 2, 1, 2}, 0).empty());
}

TEST(SpeculativeTest, DraftLengthAdaptsToAcceptance) {
    auto search = GreedyState::create(settings(100));
    ASSERT_TRUE(search);
    ASSERT_TRUE(search->accept(TokenScores{{}, ScoreKind::LOGITS, 1}));
    std::vector<std::int32_t> winners;
    std::vector<std::size_t> limits;
    std::size_t reject_after = MAX_DRAFT_TOKENS;
    bool no_match = false;
    const auto round = [&] {
        return search->step_speculative(
            4,
            [&](std::int32_t, std::size_t limit, const GreedyState::StopPredicate&,
                std::vector<std::int32_t>& out) -> kidi::Result<void> {
                limits.push_back(limit);
                if (!no_match) out.assign(limit, 2);
                return {};
            },
            [&](std::span<const std::int32_t> inputs) -> kidi::Result<TokenScores> {
                winners.assign(inputs.size(), 2);
                if (reject_after < winners.size()) winners[reject_after] = 3;
                return TokenScores{{}, ScoreKind::LOGITS, {}, winners};
            });
    };
    for (int i = 0; i < 5; ++i) ASSERT_TRUE(round());
    EXPECT_EQ(limits, (std::vector<std::size_t>{1, 2, 4, 4, 4}));
    reject_after = 2;
    ASSERT_TRUE(round());
    reject_after = 0;
    ASSERT_TRUE(round());
    reject_after = MAX_DRAFT_TOKENS;
    no_match = true;
    ASSERT_TRUE(round());
    ASSERT_TRUE(round());
    no_match = false;
    ASSERT_TRUE(round());
    ASSERT_TRUE(round());
    EXPECT_EQ(limits, (std::vector<std::size_t>{1, 2, 4, 4, 4, 4, 2, 1, 1, 1, 2}));
}
} // namespace
