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
    std::optional<std::int32_t> greedy_token;
    std::span<const std::int32_t> greedy_tokens;
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
    std::span<const std::int32_t> stop_ids;
    bool stop_on_eos = true;
};

struct Generation {
    std::vector<std::int32_t> token_ids;
    float score;
    std::size_t decoder_steps;
    bool truncated = false;
};

inline constexpr std::size_t MAX_DRAFT_TOKENS = 16;

struct SpeculativeAcceptance {
    std::size_t accepted_drafts = 0, committed_inputs = 0;
};

struct SpeculativeStats {
    std::uint64_t draft_ns = 0, verify_ns = 0;
    std::size_t draft_tokens = 0, accepted_draft_tokens = 0, verification_steps = 0;
};

struct SpeculativeStep {
    SpeculativeAcceptance acceptance;
    std::vector<std::int32_t> tokens;
    std::size_t draft_tokens = 0;
};

struct GreedyRequest {
    std::span<const std::int32_t> tokens;
    std::size_t generated_steps;
    std::span<const std::size_t> active_rows;
};

auto normalized_score(float score, std::size_t length, float alpha) -> float;

class GreedyState {
public:
    using StopPredicate = std::function<bool(std::int32_t)>;
    using DraftFunction = std::function<Result<void>(std::int32_t last_token, std::size_t max_drafts,
                                                     const StopPredicate& is_stop, std::vector<std::int32_t>& drafts)>;
    using VerifyFunction = std::function<Result<TokenScores>(std::span<const std::int32_t> inputs)>;
    using CommitFunction = std::function<Result<void>(std::size_t committed_inputs)>;

    static auto create(SearchOptions options) -> Result<GreedyState>;
    auto accept(TokenScores scores) -> Result<void>;
    /// Verifies `drafts.size() + 1` target-score rows, accepting the matching prefix and one correction or bonus
    /// token. `committed_inputs` excludes the rejected verification suffix, including when a stop ends the round.
    auto accept_speculative(std::span<const std::int32_t> drafts, TokenScores scores,
                            std::vector<std::int32_t>* committed_tokens = nullptr) -> Result<SpeculativeAcceptance>;
    /// Runs one model-agnostic speculative draft + target verification + rewind round.
    auto step_speculative(std::size_t max_drafts, const DraftFunction& draft, const VerifyFunction& verify,
                          const CommitFunction& commit = {}, SpeculativeStats* stats = nullptr)
        -> Result<SpeculativeStep>;
    auto finished() const -> bool { return finished_; }
    auto remaining_steps() const noexcept -> std::size_t {
        return finished_ ? 0 : options_.maximum_steps - result_.decoder_steps;
    }
    auto is_stop_token(std::int32_t token) const noexcept -> bool;
    auto token() const -> std::int32_t { return token_; }
    auto result() const& -> const Generation& { return result_; }
    auto result() && -> Generation { return std::move(result_); }

private:
    explicit GreedyState(SearchOptions options);
    SearchOptions options_;
    std::vector<std::int32_t> stop_ids_;
    Generation result_{};
    std::int32_t token_;
    bool finished_ = false;
};

class Decoder {
public:
    using ScoreFunction = std::function<Result<TokenScores>(const DecodeRequest&)>;
    using BatchScoreFunction = std::function<Result<TokenScores>(const GreedyRequest&)>;
    static auto generate_batch(std::span<const std::int32_t> initial_tokens, std::span<const SearchOptions> options,
                               const BatchScoreFunction& score, InferenceStats* stats = nullptr)
        -> Result<std::vector<Generation>>;

    static auto generate(std::span<const std::int32_t> prompt, const SearchOptions& options, const ScoreFunction& score,
                         InferenceStats* stats = nullptr) -> Result<Generation>;

    static auto generate_speculative(TokenScores initial_scores, const SearchOptions& options, std::size_t max_drafts,
                                     const GreedyState::DraftFunction& draft, const GreedyState::VerifyFunction& verify,
                                     const GreedyState::CommitFunction& commit = {}, SpeculativeStats* stats = nullptr)
        -> Result<Generation>;
};

} // namespace kidi::inference