#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

#include <ynnpack.h>

#include "kidi/core/error.h"
#include "kidi/model/weights.h"

namespace kidi::rtg {

class LinearImpl {
public:
    LinearImpl(ynn_subgraph_t graph, const model::Weights& weights, model::WeightEncoding weight_encoding) noexcept;

    [[nodiscard]] Result<std::uint32_t> define(std::uint32_t input_id, std::string_view prefix, std::int32_t input_size,
                                               std::int32_t output_size,
                                               std::uint32_t output_id = YNN_INVALID_VALUE_ID) const;
    [[nodiscard]] Result<std::uint32_t> define(std::uint32_t input_id, std::string_view weight_name,
                                               std::string_view bias_name, std::int32_t input_size,
                                               std::int32_t output_size,
                                               std::uint32_t output_id = YNN_INVALID_VALUE_ID) const;
    [[nodiscard]] Result<std::vector<std::uint32_t>> define_split(std::uint32_t input_id, std::string_view prefix,
                                                                  std::int32_t input_size, std::int32_t output_size,
                                                                  std::size_t output_count,
                                                                  std::span<const std::uint32_t> output_ids = {}) const;
    [[nodiscard]] Result<std::uint32_t> define_tied(std::uint32_t input_id, std::string_view embedding_name,
                                                    std::string_view bias_name, std::int32_t hidden_size,
                                                    std::int32_t vocabulary_size,
                                                    std::uint32_t output_id = YNN_INVALID_VALUE_ID) const;

private:
    struct QState {
        std::uint32_t scale_id;
    };

    struct Parameters {
        std::uint32_t weight_id;
        std::uint32_t bias_id;
        std::optional<QState> qstate;
    };

    [[nodiscard]] Result<Parameters> parameters(std::string_view weight_name, std::string_view bias_name,
                                                std::int32_t input_size, std::int32_t output_size) const;
    [[nodiscard]] Result<std::uint32_t> weight(std::string_view name, std::int32_t first_extent,
                                               std::int32_t second_extent = 0,
                                               model::DataType data_type = model::DataType::F32) const;

    ynn_subgraph_t graph_;
    const model::Weights& weights_;
    model::WeightEncoding weight_encoding_;
};

} // namespace kidi::rtg
