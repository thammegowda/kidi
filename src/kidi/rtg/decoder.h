#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "kidi/core/error.h"
#include "kidi/rtg/embedding.h"
#include "kidi/rtg/package.h"
#include "kidi/rtg/profile.h"
#include "kidi/runtime/ynn.h"

namespace kidi::rtg {

struct SourceKVCache {
    std::vector<float> values;
    std::size_t source_length;
    std::size_t hidden_size;
    std::size_t layer_count;

    [[nodiscard]] std::span<const float> key(std::size_t layer) const noexcept;
    [[nodiscard]] std::span<const float> value(std::size_t layer) const noexcept;
};

struct SelfKVCache {
    std::vector<float> values;
    std::vector<float> mask;
    std::size_t length;
    std::size_t capacity;
    std::size_t hidden_size;
    std::size_t layer_count;

    [[nodiscard]] std::span<const float> key(std::size_t layer) const noexcept;
    [[nodiscard]] std::span<const float> value(std::size_t layer) const noexcept;
    [[nodiscard]] std::span<float> key_slot(std::size_t layer, std::size_t position) noexcept;
    [[nodiscard]] std::span<float> value_slot(std::size_t layer, std::size_t position) noexcept;
};

class SourceProjectionGraph {
public:
    SourceProjectionGraph(SourceProjectionGraph&&) noexcept = default;
    SourceProjectionGraph& operator=(SourceProjectionGraph&&) noexcept = default;

    SourceProjectionGraph(const SourceProjectionGraph&) = delete;
    SourceProjectionGraph& operator=(const SourceProjectionGraph&) = delete;

    [[nodiscard]] static Result<SourceProjectionGraph> create(const Package& package);
    [[nodiscard]] Result<SourceKVCache> run(std::span<const float> memory, GraphRunStats* stats = nullptr);

private:
    SourceProjectionGraph(runtime::YnnExecutable executable, std::int32_t hidden_size, std::int32_t decoder_layers,
                          std::int32_t maximum_position) noexcept;

    runtime::YnnExecutable executable_;
    std::int32_t hidden_size_;
    std::int32_t decoder_layers_;
    std::int32_t maximum_position_;
};

class DecoderGraph {
public:
    DecoderGraph(DecoderGraph&&) noexcept = default;
    DecoderGraph& operator=(DecoderGraph&&) noexcept = default;

    DecoderGraph(const DecoderGraph&) = delete;
    DecoderGraph& operator=(const DecoderGraph&) = delete;

    [[nodiscard]] static Result<DecoderGraph> create(const Package& package);
    [[nodiscard]] Result<std::vector<float>> run(std::span<const float> embeddings, std::size_t batch_size,
                                                 const SourceKVCache& source, GraphRunStats* stats = nullptr);

private:
    DecoderGraph(runtime::YnnExecutable executable, std::int32_t hidden_size, std::int32_t decoder_layers,
                 std::int32_t maximum_position) noexcept;

    runtime::YnnExecutable executable_;
    std::int32_t hidden_size_;
    std::int32_t decoder_layers_;
    std::int32_t maximum_position_;
};

class IncrementalDecoderGraph {
public:
    IncrementalDecoderGraph(IncrementalDecoderGraph&&) noexcept = default;
    IncrementalDecoderGraph& operator=(IncrementalDecoderGraph&&) noexcept = default;

    IncrementalDecoderGraph(const IncrementalDecoderGraph&) = delete;
    IncrementalDecoderGraph& operator=(const IncrementalDecoderGraph&) = delete;

    [[nodiscard]] static Result<IncrementalDecoderGraph> create(const Package& package);
    [[nodiscard]] Result<std::span<const float>> run(std::span<const float> embedding, const SourceKVCache& source,
                                                     SelfKVCache& self, GraphRunStats* stats = nullptr);

private:
    IncrementalDecoderGraph(runtime::YnnExecutable executable, std::int32_t hidden_size, std::int32_t decoder_layers,
                            std::int32_t maximum_position) noexcept;

    runtime::YnnExecutable executable_;
    std::int32_t hidden_size_;
    std::int32_t decoder_layers_;
    std::int32_t maximum_position_;
    std::vector<float> output_;
};

class GeneratorGraph {
public:
    GeneratorGraph(GeneratorGraph&&) noexcept = default;
    GeneratorGraph& operator=(GeneratorGraph&&) noexcept = default;

    GeneratorGraph(const GeneratorGraph&) = delete;
    GeneratorGraph& operator=(const GeneratorGraph&) = delete;

    [[nodiscard]] static Result<GeneratorGraph> create(const Package& package);
    [[nodiscard]] Result<std::span<float>> run(std::span<const float> hidden_states, GraphRunStats* stats = nullptr);
    [[nodiscard]] std::int32_t vocabulary_size() const noexcept { return vocabulary_size_; }

private:
    GeneratorGraph(runtime::YnnExecutable executable, std::int32_t hidden_size, std::int32_t vocabulary_size,
                   std::int32_t maximum_batch_size) noexcept;

    runtime::YnnExecutable executable_;
    std::int32_t hidden_size_;
    std::int32_t vocabulary_size_;
    std::vector<float> output_;
};

class Decoder {
public:
    Decoder(Decoder&&) noexcept = default;
    Decoder& operator=(Decoder&&) noexcept = default;

    Decoder(const Decoder&) = delete;
    Decoder& operator=(const Decoder&) = delete;

    [[nodiscard]] static Result<Decoder> create(const Package& package);
    [[nodiscard]] Result<SourceKVCache> prepare_source(std::span<const float> memory, InferenceStats* stats = nullptr);
    [[nodiscard]] SelfKVCache create_self_cache(std::size_t maximum_steps) const;
    [[nodiscard]] Result<std::span<const float>> next_greedy(const SourceKVCache& source, std::int32_t token_id,
                                                             std::size_t position, SelfKVCache& self,
                                                             InferenceStats* stats = nullptr);
    [[nodiscard]] Result<std::span<const float>> next(const SourceKVCache& source,
                                                      std::span<const std::int32_t> token_ids, std::size_t batch_size,
                                                      InferenceStats* stats = nullptr);

private:
    Decoder(EmbeddingGraph embedding, SourceProjectionGraph source_projection, IncrementalDecoderGraph incremental,
            DecoderGraph graph, GeneratorGraph generator, std::int32_t hidden_size, std::int32_t decoder_layers,
            std::int32_t maximum_batch_size) noexcept;

    EmbeddingGraph embedding_;
    SourceProjectionGraph source_projection_;
    IncrementalDecoderGraph incremental_;
    DecoderGraph graph_;
    GeneratorGraph generator_;
    std::int32_t hidden_size_;
    std::int32_t decoder_layers_;
    std::vector<float> last_hidden_;
};

} // namespace kidi::rtg