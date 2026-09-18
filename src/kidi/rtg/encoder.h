#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "kidi/core/error.h"
#include "kidi/rtg/embedding.h"
#include "kidi/rtg/package.h"
#include "kidi/runtime/ynn.h"

namespace kidi::rtg {

class EncoderGraph {
public:
    EncoderGraph(EncoderGraph&&) noexcept = default;
    EncoderGraph& operator=(EncoderGraph&&) noexcept = default;

    EncoderGraph(const EncoderGraph&) = delete;
    EncoderGraph& operator=(const EncoderGraph&) = delete;

    [[nodiscard]] static Result<EncoderGraph> create(const Package& package);
    [[nodiscard]] Result<std::vector<float>> run(std::span<const float> embeddings);

private:
    EncoderGraph(runtime::YnnExecutable executable, std::int32_t hidden_size,
                 std::int32_t maximum_source_tokens) noexcept;

    runtime::YnnExecutable executable_;
    std::int32_t hidden_size_;
    std::int32_t maximum_source_tokens_;
};

class Encoder {
public:
    Encoder(Encoder&&) noexcept = default;
    Encoder& operator=(Encoder&&) noexcept = default;

    Encoder(const Encoder&) = delete;
    Encoder& operator=(const Encoder&) = delete;

    [[nodiscard]] static Result<Encoder> create(const Package& package);
    [[nodiscard]] Result<std::vector<float>> run(std::span<const std::int32_t> token_ids);

private:
    Encoder(EmbeddingGraph embedding, EncoderGraph graph) noexcept;

    EmbeddingGraph embedding_;
    EncoderGraph graph_;
};

} // namespace kidi::rtg