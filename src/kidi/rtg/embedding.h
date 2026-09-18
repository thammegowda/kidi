#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

#include "kidi/core/error.h"
#include "kidi/model/weights.h"
#include "kidi/runtime/ynn.h"

namespace kidi::rtg {

class EmbeddingGraph {
public:
    EmbeddingGraph(EmbeddingGraph&&) noexcept = default;
    EmbeddingGraph& operator=(EmbeddingGraph&&) noexcept = default;

    EmbeddingGraph(const EmbeddingGraph&) = delete;
    EmbeddingGraph& operator=(const EmbeddingGraph&) = delete;

    [[nodiscard]] static std::expected<EmbeddingGraph, core::Error> create(const model::Weights& weights,
                                                                           std::string_view weight_name,
                                                                           std::int32_t vocabulary_size,
                                                                           std::int32_t hidden_size);
    [[nodiscard]] std::expected<std::vector<float>, core::Error> run(std::span<const std::int32_t> token_ids,
                                                                     std::size_t batch_size = 1);

private:
    EmbeddingGraph(runtime::YnnExecutable executable, std::int32_t vocabulary_size, std::int32_t hidden_size,
                   std::unique_ptr<float> scale);

    runtime::YnnExecutable executable_;
    std::int32_t vocabulary_size_;
    std::int32_t hidden_size_;
    std::unique_ptr<float> scale_;
    std::uint32_t token_ids_id_;
    std::uint32_t positions_id_;
    std::uint32_t output_id_;
};

} // namespace kidi::rtg