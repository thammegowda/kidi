#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <vector>

#include "kidi/core/error.h"
#include "kidi/inference/profile.h"

namespace kidi::inference {

enum class ScoreKind { LOGITS, LOG_PROBABILITIES };

struct TokenScores {
    std::span<const float> values;
    ScoreKind kind = ScoreKind::LOGITS;
};

struct DecodeRequest {
    std::span<const std::int32_t> prefixes;
    std::span<const std::size_t> beam_indices;
    std::size_t prefix_length;
    std::size_t generated_steps;
};

struct SearchOptions {
    std::size_t vocabulary_size;
    std::int32_t end_id;
    std::int32_t pad_id;
    std::size_t maximum_steps;
    std::size_t beam_size = 1;
    float length_penalty = 0.0F;
    bool compute_score = true;
    std::optional<std::size_t> unfinished_score_length;
};

struct Generation {
    std::vector<std::int32_t> token_ids;
    float score;
    std::size_t decoder_steps;
};

struct GreedyRequest {
    std::span<const std::int32_t> tokens;
    std::size_t generated_steps;
};

auto normalized_score(float score, std::size_t length, float alpha) -> float;

class Decoder {
public:
    using ScoreFunction = std::function<Result<TokenScores>(const DecodeRequest&)>;
    using BatchScoreFunction = std::function<Result<TokenScores>(const GreedyRequest&)>;
    static auto generate_batch(std::span<const std::int32_t> initial_tokens, std::span<const SearchOptions> options,
                               const BatchScoreFunction& score, InferenceStats* stats = nullptr)
        -> Result<std::vector<Generation>>;

    static auto generate(std::span<const std::int32_t> prompt, const SearchOptions& options, const ScoreFunction& score,
                         InferenceStats* stats = nullptr) -> Result<Generation>;
};

} // namespace kidi::inference