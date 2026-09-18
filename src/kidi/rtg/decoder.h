#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "kidi/core/error.h"
#include "kidi/rtg/embedding.h"
#include "kidi/rtg/package.h"
#include "kidi/runtime/ynn.h"

namespace kidi::rtg {

class DecoderGraph {
public:
    DecoderGraph(DecoderGraph&&) noexcept = default;
    DecoderGraph& operator=(DecoderGraph&&) noexcept = default;

    DecoderGraph(const DecoderGraph&) = delete;
    DecoderGraph& operator=(const DecoderGraph&) = delete;

    [[nodiscard]] static Result<DecoderGraph> create(const Package& package);
    [[nodiscard]] Result<std::vector<float>> run(std::span<const float> embeddings, std::size_t batch_size,
                                                 std::span<const float> memory);

private:
    DecoderGraph(runtime::YnnExecutable executable, std::int32_t hidden_size, std::int32_t maximum_position) noexcept;

    runtime::YnnExecutable executable_;
    std::int32_t hidden_size_;
    std::int32_t maximum_position_;
};

class GeneratorGraph {
public:
    GeneratorGraph(GeneratorGraph&&) noexcept = default;
    GeneratorGraph& operator=(GeneratorGraph&&) noexcept = default;

    GeneratorGraph(const GeneratorGraph&) = delete;
    GeneratorGraph& operator=(const GeneratorGraph&) = delete;

    [[nodiscard]] static Result<GeneratorGraph> create(const Package& package);
    [[nodiscard]] Result<std::vector<float>> run(std::span<const float> hidden_states);

private:
    GeneratorGraph(runtime::YnnExecutable executable, std::int32_t hidden_size, std::int32_t vocabulary_size) noexcept;

    runtime::YnnExecutable executable_;
    std::int32_t hidden_size_;
    std::int32_t vocabulary_size_;
};

class Decoder {
public:
    Decoder(Decoder&&) noexcept = default;
    Decoder& operator=(Decoder&&) noexcept = default;

    Decoder(const Decoder&) = delete;
    Decoder& operator=(const Decoder&) = delete;

    [[nodiscard]] static Result<Decoder> create(const Package& package);
    [[nodiscard]] Result<std::vector<float>> next(std::span<const float> memory,
                                                  std::span<const std::int32_t> token_ids, std::size_t batch_size);

private:
    Decoder(EmbeddingGraph embedding, DecoderGraph graph, GeneratorGraph generator, std::int32_t hidden_size) noexcept;

    EmbeddingGraph embedding_;
    DecoderGraph graph_;
    GeneratorGraph generator_;
    std::int32_t hidden_size_;
};

} // namespace kidi::rtg