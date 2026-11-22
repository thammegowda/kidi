#include <array>
#include <algorithm>
#include <cmath>
#include <limits>
#include "kidi/inference/decoder.h"

#include <gtest/gtest.h>

namespace {
using namespace kidi::inference;

const SearchOptions greedy{.vocabulary_size = 5, .end_id = 3, .pad_id = 0, .maximum_steps = 4};

// Prefers token 4 on the first step and the end token afterwards, recording whether the prefixes looked right.
struct LanguageModel {
    std::vector<float> values;
    bool prefixes_valid = true;
    auto operator()(const DecodeRequest& request) -> kidi::Result<TokenScores> {
        prefixes_valid &= request.prefixes[0] == 1 && request.prefixes[1] == 2;
        prefixes_valid &= request.prefix_length == request.generated_steps + 2;
        values.assign(5, -10.0F);
        values[request.generated_steps == 0 ? 4 : 3] = 10.0F;
        return TokenScores{values};
    }
};

const auto selected_model = [](const DecodeRequest& request) -> kidi::Result<TokenScores> {
    return TokenScores{{}, ScoreKind::LOGITS, request.generated_steps == 0 ? 4 : 3};
};

auto multi_stop_options() -> SearchOptions {
    static const std::array<std::int32_t, 1> additional_stops{4};
    auto multi_stop = greedy;
    multi_stop.stop_ids = additional_stops;
    return multi_stop;
}

auto unscored_greedy() -> SearchOptions {
    auto unscored = greedy;
    unscored.compute_score = false;
    return unscored;
}

TEST(DecoderTest, GeneratesGreedily) {
    LanguageModel model;
    auto generated = Decoder::generate(std::array<std::int32_t, 2>{1, 2}, greedy, std::ref(model));
    ASSERT_TRUE(generated);
    EXPECT_TRUE(model.prefixes_valid);
    EXPECT_EQ(generated->token_ids, std::vector<std::int32_t>{4});
    EXPECT_EQ(generated->decoder_steps, 2);
    EXPECT_FALSE(generated->truncated);
}

TEST(DecoderTest, HonoursAdditionalStopIds) {
    LanguageModel model;
    auto multi_stop = multi_stop_options();
    auto stopped = Decoder::generate(std::array<std::int32_t, 2>{1, 2}, multi_stop, std::ref(model));
    ASSERT_TRUE(stopped);
    EXPECT_TRUE(stopped->token_ids.empty());
    EXPECT_EQ(stopped->decoder_steps, 1);
    multi_stop.stop_on_eos = false;
    auto fixed = Decoder::generate(std::array<std::int32_t, 2>{1, 2}, multi_stop, std::ref(model));
    ASSERT_TRUE(fixed);
    EXPECT_EQ(fixed->token_ids, (std::vector<std::int32_t>{4, 3, 3, 3}));
    EXPECT_EQ(fixed->decoder_steps, 4);
}

TEST(DecoderTest, SelectedMatchHost) {
    LanguageModel model;
    auto multi_stop = multi_stop_options();
    multi_stop.stop_on_eos = false;
    auto selected_stop = multi_stop;
    selected_stop.stop_on_eos = true;
    for (const auto& policy : {greedy, selected_stop, multi_stop}) {
        auto unscored = policy;
        unscored.compute_score = false;
        const auto host = Decoder::generate(std::array<std::int32_t, 2>{1, 2}, unscored, std::ref(model));
        const auto selected = Decoder::generate(std::array<std::int32_t, 2>{1, 2}, unscored, selected_model);
        ASSERT_TRUE(host);
        ASSERT_TRUE(selected);
        EXPECT_EQ(selected->token_ids, host->token_ids);
        EXPECT_EQ(selected->decoder_steps, host->decoder_steps);
    }
}

TEST(DecoderTest, RejectsInvalidSelectedTokens) {
    EXPECT_FALSE(Decoder::generate(std::array<std::int32_t, 1>{2}, greedy, selected_model));
    const auto unscored = unscored_greedy();
    for (auto invalid : {-1, 5}) {
        EXPECT_FALSE(Decoder::generate(std::array<std::int32_t, 1>{2}, unscored,
                                       [invalid](const DecodeRequest&) -> kidi::Result<TokenScores> {
                                           return TokenScores{{}, ScoreKind::LOGITS, invalid};
                                       }))
            << "invalid token " << invalid;
    }
}

TEST(DecoderTest, BeamSearchConditioned) {
    auto beam = greedy;
    beam.beam_size = 2;
    auto selected_beam = beam;
    selected_beam.compute_score = false;
    EXPECT_FALSE(Decoder::generate(std::array<std::int32_t, 1>{2}, selected_beam, selected_model));
    std::vector<float> values;
    const auto conditioned_model = [&](const DecodeRequest& request) -> kidi::Result<TokenScores> {
        values.assign(request.beam_indices.size() * 5, -std::numeric_limits<float>::infinity());
        for (std::size_t row = 0; row < request.beam_indices.size(); ++row) {
            if (request.generated_steps == 0) {
                values[row * 5 + 4] = -0.1F;
                values[row * 5 + 3] = -0.2F;
            } else
                values[row * 5 + 3] = -0.5F;
        }
        return TokenScores{values, ScoreKind::LOG_PROBABILITIES};
    };
    auto conditioned = Decoder::generate(std::array<std::int32_t, 1>{2}, beam, conditioned_model);
    ASSERT_TRUE(conditioned);
    EXPECT_TRUE(conditioned->token_ids.empty());
    EXPECT_NEAR(conditioned->score, -0.2F, 1e-6F);
}

TEST(DecoderTest, ReportsTruncation) {
    LanguageModel model;
    auto limited = greedy;
    limited.maximum_steps = 1;
    auto truncated = Decoder::generate(std::array<std::int32_t, 2>{1, 2}, limited, std::ref(model));
    ASSERT_TRUE(truncated);
    EXPECT_EQ(truncated->token_ids, std::vector<std::int32_t>{4});
    EXPECT_EQ(truncated->decoder_steps, 1);
    EXPECT_TRUE(truncated->truncated);
    auto exact_end = greedy;
    exact_end.maximum_steps = 2;
    auto exact = Decoder::generate(std::array<std::int32_t, 2>{1, 2}, exact_end, std::ref(model));
    ASSERT_TRUE(exact);
    EXPECT_FALSE(exact->truncated);
}

TEST(DecoderTest, PropagatesModelErrors) {
    auto malformed = Decoder::generate(std::array<std::int32_t, 1>{2}, greedy,
                                       [](const DecodeRequest&) -> kidi::Result<TokenScores> { return TokenScores{}; });
    EXPECT_FALSE(malformed);
    auto failed = Decoder::generate(std::array<std::int32_t, 1>{2}, greedy,
                                    [](const DecodeRequest&) -> kidi::Result<TokenScores> {
                                        return std::unexpected(kidi::Error{kidi::ErrorCode::RUNTIME, "model failure"});
                                    });
    ASSERT_FALSE(failed);
    EXPECT_EQ(failed.error().message, "model failure");
}

class DecoderBatchTest : public ::testing::Test {
protected:
    DecoderBatchTest() {
        settings[0].maximum_steps = 1;
        settings[1].maximum_steps = 3;
        table[0 * 5 + 3] = 10;
        table[1 * 5 + 3] = 10;
        table[2 * 5 + 1] = 10;
        table[3 * 5 + 3] = 10;
        table[4 * 5 + 4] = 10;
    }

    const std::array<std::int32_t, 3> starts = {2, 2, 4};
    std::array<SearchOptions, 3> settings{greedy, greedy, greedy};
    std::vector<float> table = std::vector<float>(25, -10.0F);
};

TEST_F(DecoderBatchTest, FromScores) {
    std::vector<float> values;
    for (int repetition = 0; repetition < 2; ++repetition) {
        SCOPED_TRACE(repetition);
        auto result =
            Decoder::generate_batch(starts, settings, [&](const GreedyRequest& request) -> kidi::Result<TokenScores> {
                values.clear();
                for (auto token : request.tokens)
                    values.insert(values.end(), table.begin() + token * 5, table.begin() + token * 5 + 5);
                return TokenScores{values};
            });
        ASSERT_TRUE(result);
        EXPECT_EQ((*result)[0].token_ids, std::vector<std::int32_t>{1});
        EXPECT_EQ((*result)[1].token_ids, std::vector<std::int32_t>{1});
        EXPECT_EQ((*result)[2].token_ids, (std::vector<std::int32_t>{4, 4, 4, 4}));
        EXPECT_EQ((*result)[0].decoder_steps, 1);
        EXPECT_EQ((*result)[1].decoder_steps, 2);
        EXPECT_EQ((*result)[2].decoder_steps, 4);
    }
}

TEST_F(DecoderBatchTest, FromSelectedTokens) {
    for (auto& settings_row : settings) settings_row.compute_score = false;
    std::array<std::int32_t, 3> chosen{};
    auto selected_batch =
        Decoder::generate_batch(starts, settings, [&](const GreedyRequest& request) -> kidi::Result<TokenScores> {
            const std::vector<std::size_t> expected_active =
                request.generated_steps == 0   ? std::vector<std::size_t>{0, 1, 2}
                : request.generated_steps == 1 ? std::vector<std::size_t>{1, 2}
                                               : std::vector<std::size_t>{2};
            if (!std::ranges::equal(request.active_rows, expected_active))
                return std::unexpected(kidi::Error{kidi::ErrorCode::RUNTIME, "wrong active rows"});
            for (auto row : request.active_rows) {
                const auto values = std::span(table).subspan(request.tokens[row] * 5, 5);
                chosen[row] = std::ranges::max_element(values) - values.begin();
            }
            return TokenScores{{}, ScoreKind::LOGITS, {}, chosen};
        });
    ASSERT_TRUE(selected_batch) << selected_batch.error().message;
    EXPECT_EQ((*selected_batch)[0].token_ids, std::vector<std::int32_t>{1});
    EXPECT_EQ((*selected_batch)[1].token_ids, std::vector<std::int32_t>{1});
    EXPECT_EQ((*selected_batch)[2].token_ids, (std::vector<std::int32_t>{4, 4, 4, 4}));
}

TEST_F(DecoderBatchTest, RejectsSelectedTokens) {
    for (auto& settings_row : settings) settings_row.compute_score = false;
    std::array<std::int32_t, 3> chosen{};
    for (auto invalid : {-1, 5}) {
        chosen.fill(invalid);
        EXPECT_FALSE(Decoder::generate_batch(starts, settings, [&](const GreedyRequest&) -> kidi::Result<TokenScores> {
            return TokenScores{{}, ScoreKind::LOGITS, {}, chosen};
        })) << "invalid token " << invalid;
    }
}

TEST(DecoderTest, IncrementalGreedyState) {
    auto incremental = GreedyState::create(unscored_greedy());
    ASSERT_TRUE(incremental);
    EXPECT_FALSE(incremental->accept(TokenScores{{}, ScoreKind::LOGITS, -1}));
    EXPECT_EQ(incremental->result().decoder_steps, 0);
    EXPECT_TRUE(incremental->accept(TokenScores{{}, ScoreKind::LOGITS, 4}));
    EXPECT_FALSE(incremental->finished());
    EXPECT_TRUE(incremental->accept(TokenScores{{}, ScoreKind::LOGITS, 3}));
    EXPECT_TRUE(incremental->finished());
    EXPECT_EQ(incremental->result().token_ids, std::vector<std::int32_t>{4});
    EXPECT_FALSE(incremental->result().truncated);
    EXPECT_FALSE(incremental->accept(TokenScores{{}, ScoreKind::LOGITS, 4}));
}
} // namespace
