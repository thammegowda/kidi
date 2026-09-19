#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

#include <ynnpack.h>

#include "kidi/core/error.h"
#include "kidi/model/weights.h"
#include "kidi/rtg/linear.h"

namespace kidi::rtg {

class TransformerBuilder {
public:
    TransformerBuilder(ynn_subgraph_t graph, const model::Weights& weights, std::int32_t hidden_size,
                       std::int32_t feed_forward_size, std::int32_t attention_heads, float layer_norm_epsilon,
                       model::WeightEncoding weight_encoding = model::WeightEncoding::F32) noexcept;

    [[nodiscard]] Result<std::uint32_t> linear(std::uint32_t input_id, std::string_view prefix, std::int32_t input_size,
                                               std::int32_t output_size,
                                               std::uint32_t output_id = YNN_INVALID_VALUE_ID) const;
    [[nodiscard]] Result<std::uint32_t> linear(std::uint32_t input_id, std::string_view weight_name,
                                               std::string_view bias_name, std::int32_t input_size,
                                               std::int32_t output_size,
                                               std::uint32_t output_id = YNN_INVALID_VALUE_ID) const;
    [[nodiscard]] Result<std::uint32_t> tied_projection(std::uint32_t input_id, std::string_view embedding_name,
                                                        std::string_view bias_name, std::int32_t hidden_size,
                                                        std::int32_t vocabulary_size,
                                                        std::uint32_t output_id = YNN_INVALID_VALUE_ID) const;
    [[nodiscard]] Result<std::vector<std::uint32_t>> linear_split(std::uint32_t input_id, std::string_view prefix,
                                                                  std::int32_t input_size, std::int32_t output_size,
                                                                  std::size_t output_count,
                                                                  std::span<const std::uint32_t> output_ids = {}) const;
    [[nodiscard]] Result<std::uint32_t> gelu(std::uint32_t input_id,
                                             std::uint32_t output_id = YNN_INVALID_VALUE_ID) const;
    [[nodiscard]] Result<std::uint32_t> layer_norm(std::uint32_t input_id, std::string_view prefix,
                                                   std::uint32_t output_id = YNN_INVALID_VALUE_ID) const;
    [[nodiscard]] Result<std::uint32_t> feed_forward(std::uint32_t input_id, std::string_view prefix,
                                                     std::uint32_t output_id = YNN_INVALID_VALUE_ID) const;
    [[nodiscard]] Result<std::uint32_t> self_attention(std::uint32_t input_id, std::uint32_t mask_id,
                                                       std::string_view prefix,
                                                       std::uint32_t output_id = YNN_INVALID_VALUE_ID) const;
    [[nodiscard]] Result<std::uint32_t> attention_with_projected_key_value(
        std::uint32_t query_id, std::uint32_t projected_key_id, std::uint32_t projected_value_id, std::uint32_t mask_id,
        std::string_view prefix, std::uint32_t output_id = YNN_INVALID_VALUE_ID) const;
    [[nodiscard]] Result<std::uint32_t> attention_from_projections(std::uint32_t projected_query_id,
                                                                   std::uint32_t projected_key_id,
                                                                   std::uint32_t projected_value_id,
                                                                   std::uint32_t mask_id, std::string_view prefix,
                                                                   std::uint32_t output_id) const;
    [[nodiscard]] Result<std::uint32_t> attention_decode_one_from_projections(
        std::uint32_t projected_query_id, std::uint32_t projected_key_id, std::uint32_t projected_value_id,
        std::uint32_t mask_id, std::string_view prefix, std::uint32_t output_id = YNN_INVALID_VALUE_ID) const;

private:
    [[nodiscard]] Result<std::uint32_t> weight(std::string_view name, std::int32_t first_extent,
                                               std::int32_t second_extent = 0,
                                               model::DataType data_type = model::DataType::F32) const;
    [[nodiscard]] Result<std::uint32_t> scalar(float value) const;

    ynn_subgraph_t graph_;
    const model::Weights& weights_;
    LinearImpl linear_;
    std::int32_t hidden_size_;
    std::int32_t feed_forward_size_;
    std::int32_t attention_heads_;
    float layer_norm_epsilon_;
};

} // namespace kidi::rtg