#include <array>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include "kidi/inference/decoder.h"

auto main() -> int {
    using namespace kidi::inference;
    const SearchOptions greedy{.vocabulary_size = 5, .end_id = 3, .pad_id = 0, .maximum_steps = 4};
    std::vector<float> values;
    bool prefixes_valid = true;
    const auto language_model = [&](const DecodeRequest& request) -> kidi::Result<TokenScores> {
        prefixes_valid &= request.prefixes[0] == 1 && request.prefixes[1] == 2;
        prefixes_valid &= request.prefix_length == request.generated_steps + 2;
        values.assign(5, -10.0F);
        values[request.generated_steps == 0 ? 4 : 3] = 10.0F;
        return TokenScores{values};
    };
    auto generated = Decoder::generate(std::array<std::int32_t, 2>{1, 2}, greedy, language_model);
    if (!generated || !prefixes_valid || generated->token_ids != std::vector<std::int32_t>{4} ||
        generated->decoder_steps != 2)
        return 1;
    const std::array<std::int32_t, 1> additional_stops{4};
    auto multi_stop = greedy;
    multi_stop.stop_ids = additional_stops;
    auto stopped = Decoder::generate(std::array<std::int32_t, 2>{1, 2}, multi_stop, language_model);
    if (!stopped || !stopped->token_ids.empty() || stopped->decoder_steps != 1) return 1;
    multi_stop.stop_on_eos = false;
    auto fixed = Decoder::generate(std::array<std::int32_t, 2>{1, 2}, multi_stop, language_model);
    if (!fixed || fixed->token_ids != std::vector<std::int32_t>{4, 3, 3, 3} || fixed->decoder_steps != 4) return 1;
    auto beam = greedy;
    beam.beam_size = 2;
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
    if (!conditioned || !conditioned->token_ids.empty() || std::abs(conditioned->score + 0.2F) > 1e-6F) return 1;
    auto limited = greedy;
    limited.maximum_steps = 1;
    auto truncated = Decoder::generate(std::array<std::int32_t, 2>{1, 2}, limited, language_model);
    if (!truncated || truncated->token_ids != std::vector<std::int32_t>{4} || truncated->decoder_steps != 1) return 1;
    auto malformed = Decoder::generate(std::array<std::int32_t, 1>{2}, greedy,
                                       [](const DecodeRequest&) -> kidi::Result<TokenScores> { return TokenScores{}; });
    if (malformed) return 1;
    auto failed = Decoder::generate(std::array<std::int32_t, 1>{2}, greedy,
                                    [](const DecodeRequest&) -> kidi::Result<TokenScores> {
                                        return std::unexpected(kidi::Error{kidi::ErrorCode::RUNTIME, "model failure"});
                                    });
    if (failed || failed.error().message != "model failure") return 1;
    const std::array<std::int32_t, 3> starts = {2, 2, 4};
    std::array settings{greedy, greedy, greedy};
    settings[0].maximum_steps = 1;
    settings[1].maximum_steps = 3;
    std::vector<float> table(25, -10.0F);
    table[0 * 5 + 3] = 10;
    table[1 * 5 + 3] = 10;
    table[2 * 5 + 1] = 10;
    table[3 * 5 + 3] = 10;
    table[4 * 5 + 4] = 10;
    for (int repetition = 0; repetition < 2; ++repetition) {
        auto result =
            Decoder::generate_batch(starts, settings, [&](const GreedyRequest& request) -> kidi::Result<TokenScores> {
                values.clear();
                for (auto token : request.tokens)
                    values.insert(values.end(), table.begin() + token * 5, table.begin() + token * 5 + 5);
                return TokenScores{values};
            });
        if (!result || (*result)[0].token_ids != std::vector<std::int32_t>{1} ||
            (*result)[1].token_ids != std::vector<std::int32_t>{1} ||
            (*result)[2].token_ids != std::vector<std::int32_t>{4, 4, 4, 4} || (*result)[0].decoder_steps != 1 ||
            (*result)[1].decoder_steps != 2 || (*result)[2].decoder_steps != 4)
            return 1;
    }
    std::cout << "generic conditioned/decoder-only search tests passed\n";
    return 0;
}