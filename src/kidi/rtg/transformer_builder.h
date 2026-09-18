#pragma once

#include <cstdint>
#include <string_view>

#include <ynnpack.h>

#include "kidi/core/error.h"
#include "kidi/model/weights.h"

namespace kidi::rtg {

class TransformerBuilder {
public:
    TransformerBuilder(ynn_subgraph_t graph, const model::Weights& weights, std::int32_t hidden_size,
                       std::int32_t feed_forward_size, std::int32_t attention_heads, float layer_norm_epsilon) noexcept;

    [[nodiscard]] Result<std::uint32_t> linear(std::uint32_t input_id, std::string_view prefix, std::int32_t input_size,
                                               std::int32_t output_size,
                                               std::uint32_t output_id = YNN_INVALID_VALUE_ID) const;
    [[nodiscard]] Result<std::uint32_t> linear(std::uint32_t input_id, std::string_view weight_name,
                                               std::string_view bias_name, std::int32_t input_size,
                                               std::int32_t output_size,
                                               std::uint32_t output_id = YNN_INVALID_VALUE_ID) const;
    [[nodiscard]] Result<std::uint32_t> gelu(std::uint32_t input_id,
                                             std::uint32_t output_id = YNN_INVALID_VALUE_ID) const;
    [[nodiscard]] Result<std::uint32_t> layer_norm(std::uint32_t input_id, std::string_view prefix,
                                                   std::uint32_t output_id = YNN_INVALID_VALUE_ID) const;
    [[nodiscard]] Result<std::uint32_t> feed_forward(std::uint32_t input_id, std::string_view prefix,
                                                     std::uint32_t output_id = YNN_INVALID_VALUE_ID) const;
    [[nodiscard]] Result<std::uint32_t> attention(std::uint32_t query_id, std::uint32_t key_id, std::uint32_t value_id,
                                                  std::uint32_t mask_id, std::string_view prefix,
                                                  std::uint32_t output_id = YNN_INVALID_VALUE_ID) const;

private:
    [[nodiscard]] Result<std::uint32_t> weight(std::string_view name, std::int32_t first_extent,
                                               std::int32_t second_extent = 0) const;
    [[nodiscard]] Result<std::uint32_t> scalar(float value) const;

    ynn_subgraph_t graph_;
    const model::Weights& weights_;
    std::int32_t hidden_size_;
    std::int32_t feed_forward_size_;
    std::int32_t attention_heads_;
    float layer_norm_epsilon_;
};

} // namespace kidi::rtg