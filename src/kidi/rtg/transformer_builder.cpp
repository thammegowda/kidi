#include "kidi/rtg/transformer_builder.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <string>
#include <utility>

#include "kidi/runtime/ynn.h"
#include "ynnpack/composites/composites.h"

namespace kidi::rtg {
namespace {

Result<ynn_type> to_ynn_type(model::DataType data_type) {
    switch (data_type) {
        case model::DataType::F32:
            return ynn_type_fp32;
        case model::DataType::BF16:
            return ynn_type_bf16;
        case model::DataType::I8:
            return ynn_type_int8;
        default:
            return std::unexpected(Error{ErrorCode::UNSUPPORTED, "unsupported YNNPACK weight data type"});
    }
}

} // namespace

TransformerBuilder::TransformerBuilder(ynn_subgraph_t graph, const model::Weights& weights, std::int32_t hidden_size,
                                       std::int32_t feed_forward_size, std::int32_t attention_heads,
                                       float layer_norm_epsilon, model::WeightEncoding weight_encoding) noexcept
    : graph_(graph),
      weights_(weights),
      hidden_size_(hidden_size),
      feed_forward_size_(feed_forward_size),
      attention_heads_(attention_heads),
      layer_norm_epsilon_(layer_norm_epsilon),
      weight_encoding_(weight_encoding) {}

Result<std::uint32_t> TransformerBuilder::weight(std::string_view name, std::int32_t first_extent,
                                                 std::int32_t second_extent, model::DataType data_type) const {
    auto tensor = weights_.tensor(name);
    if (!tensor) return std::unexpected(std::move(tensor.error()));
    const bool shape_matches = second_extent == 0 ? tensor->shape.size() == 1 && tensor->shape[0] == first_extent
                                                  : tensor->shape.size() == 2 && tensor->shape[0] == first_extent &&
                                                        tensor->shape[1] == second_extent;
    if (tensor->data_type != data_type || !shape_matches) {
        return std::unexpected(Error{
            ErrorCode::INVALID_ARGUMENT,
            "weight has incompatible dtype or shape: " + std::string(name),
        });
    }

    const std::array<std::size_t, 2> dimensions = {
        static_cast<std::size_t>(first_extent),
        static_cast<std::size_t>(second_extent),
    };
    auto type = to_ynn_type(data_type);
    if (!type) return std::unexpected(std::move(type.error()));
    std::uint32_t id = YNN_INVALID_VALUE_ID;
    auto status = runtime::check_ynn_status(
        ynn_define_tensor(graph_, *type, second_extent == 0 ? 1 : 2, dimensions.data(), tensor->data(), 0, &id),
        "define mapped weight");
    if (!status) return std::unexpected(std::move(status.error()));
    return id;
}

Result<std::uint32_t> TransformerBuilder::scalar(float value) const {
    std::uint32_t id = YNN_INVALID_VALUE_ID;
    auto status = runtime::check_ynn_status(
        ynn_define_tensor(graph_, ynn_type_fp32, 0, nullptr, &value, YNN_VALUE_FLAG_COPY_DATA, &id), "define scalar");
    if (!status) return std::unexpected(std::move(status.error()));
    return id;
}

Result<std::uint32_t> TransformerBuilder::linear(std::uint32_t input_id, std::string_view prefix,
                                                 std::int32_t input_size, std::int32_t output_size,
                                                 std::uint32_t output_id) const {
    return linear(input_id, std::string(prefix) + ".weight", std::string(prefix) + ".bias", input_size, output_size,
                  output_id);
}

Result<std::uint32_t> TransformerBuilder::linear(std::uint32_t input_id, std::string_view weight_name,
                                                 std::string_view bias_name, std::int32_t input_size,
                                                 std::int32_t output_size, std::uint32_t output_id) const {
    const auto matrix_type = model::matrix_data_type(weight_encoding_);
    auto weight_id = weight(weight_name, input_size, output_size, matrix_type);
    if (!weight_id) return std::unexpected(std::move(weight_id.error()));
    auto bias_id = weight(bias_name, output_size);
    if (!bias_id) return std::unexpected(std::move(bias_id.error()));

    if (weight_encoding_ == model::WeightEncoding::INT8_PER_CHANNEL) {
        auto scale_id = weight(model::quantization_scale_name(weight_name), output_size, 1);
        if (!scale_id) return std::unexpected(std::move(scale_id.error()));

        constexpr std::array<std::int32_t, 1> REDUCE_AXIS = {-1};
        std::uint32_t min_max_id = YNN_INVALID_VALUE_ID;
        std::uint32_t input_zero_point_id = YNN_INVALID_VALUE_ID;
        std::uint32_t input_scale_id = YNN_INVALID_VALUE_ID;
        std::uint32_t quantized_input_id = YNN_INVALID_VALUE_ID;
        auto status = runtime::check_ynn_status(
            ynn_define_reduce(graph_, ynn_reduce_min_max, REDUCE_AXIS.size(), REDUCE_AXIS.data(), input_id,
                              YNN_INVALID_VALUE_ID, &min_max_id, YNN_NODE_FLAG_KEEP_DIMS),
            "define INT8 input range");
        if (status)
            status =
                runtime::check_ynn_status(ynn_define_dynamic_quantization(graph_, min_max_id, ynn_type_int8,
                                                                          &input_zero_point_id, &input_scale_id, 0),
                                          "define INT8 input quantization");
        if (status)
            status = runtime::check_ynn_status(ynn_define_quantize(graph_, input_id, ynn_type_int8, input_zero_point_id,
                                                                   input_scale_id, &quantized_input_id, 0),
                                               "quantize linear input");
        if (status)
            status = runtime::check_ynn_status(
                ynn::define_blockwise_dot(graph_, quantized_input_id, input_zero_point_id, input_scale_id, *weight_id,
                                          YNN_INVALID_VALUE_ID, *scale_id, static_cast<std::size_t>(input_size),
                                          *bias_id, ynn_type_fp32, output_id, 0),
                "define INT8 linear");
        if (!status) return std::unexpected(std::move(status.error()));
        return output_id;
    }

    std::uint32_t dot_input_id = weight_encoding_ == model::WeightEncoding::BF16 ? YNN_INVALID_VALUE_ID : input_id;
    Result<void> status;
    if (weight_encoding_ == model::WeightEncoding::BF16) {
        status = runtime::check_ynn_status(
            ynn_define_convert(graph_, input_id, ynn_type_bf16, &dot_input_id, YNN_NODE_FLAG_NO_EXCESS_PRECISION),
            "convert linear input to BF16");
    }
    if (status) {
        status = runtime::check_ynn_status(ynn_define_dot(graph_, 1, dot_input_id, *weight_id, *bias_id, &output_id, 0),
                                           "define linear");
    }
    if (!status) return std::unexpected(std::move(status.error()));
    return output_id;
}

Result<std::uint32_t> TransformerBuilder::tied_projection(std::uint32_t input_id, std::string_view embedding_name,
                                                          std::string_view bias_name, std::int32_t hidden_size,
                                                          std::int32_t vocabulary_size, std::uint32_t output_id) const {
    if (weight_encoding_ != model::WeightEncoding::F32) {
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "tied projection requires FP32 embedding weights"});
    }
    auto embedding_id = weight(embedding_name, vocabulary_size, hidden_size);
    if (!embedding_id) return std::unexpected(std::move(embedding_id.error()));
    auto bias_id = weight(bias_name, vocabulary_size);
    if (!bias_id) return std::unexpected(std::move(bias_id.error()));

    constexpr std::array<std::int32_t, 2> TRANSPOSE_AXES = {1, 0};
    std::uint32_t transposed_id = YNN_INVALID_VALUE_ID;
    auto status =
        runtime::check_ynn_status(ynn_define_static_transpose(graph_, TRANSPOSE_AXES.size(), TRANSPOSE_AXES.data(),
                                                              *embedding_id, &transposed_id, 0),
                                  "transpose tied embedding");
    if (status)
        status = runtime::check_ynn_status(ynn_define_dot(graph_, 1, input_id, transposed_id, *bias_id, &output_id, 0),
                                           "define tied projection");
    if (!status) return std::unexpected(std::move(status.error()));
    return output_id;
}

Result<std::uint32_t> TransformerBuilder::gelu(std::uint32_t input_id, std::uint32_t output_id) const {
    auto status = runtime::check_ynn_status(ynn::define_gelu(graph_, input_id, output_id), "define exact GELU");
    if (!status) return std::unexpected(std::move(status.error()));
    return output_id;
}

Result<std::uint32_t> TransformerBuilder::layer_norm(std::uint32_t input_id, std::string_view prefix,
                                                     std::uint32_t output_id) const {
    auto scale_id = weight(std::string(prefix) + ".weight", hidden_size_);
    if (!scale_id) return std::unexpected(std::move(scale_id.error()));
    auto bias_id = weight(std::string(prefix) + ".bias", hidden_size_);
    if (!bias_id) return std::unexpected(std::move(bias_id.error()));
    auto reciprocal_size_id = scalar(1.0F / static_cast<float>(hidden_size_));
    if (!reciprocal_size_id) return std::unexpected(std::move(reciprocal_size_id.error()));
    auto epsilon_id = scalar(layer_norm_epsilon_);
    if (!epsilon_id) return std::unexpected(std::move(epsilon_id.error()));

    constexpr std::array<std::int32_t, 1> REDUCE_AXIS = {-1};
    std::uint32_t sum_id = YNN_INVALID_VALUE_ID;
    std::uint32_t mean_id = YNN_INVALID_VALUE_ID;
    std::uint32_t centered_id = YNN_INVALID_VALUE_ID;
    std::uint32_t squared_id = YNN_INVALID_VALUE_ID;
    std::uint32_t square_sum_id = YNN_INVALID_VALUE_ID;
    std::uint32_t variance_id = YNN_INVALID_VALUE_ID;
    std::uint32_t stabilized_id = YNN_INVALID_VALUE_ID;
    std::uint32_t reciprocal_standard_deviation_id = YNN_INVALID_VALUE_ID;
    std::uint32_t normalized_id = YNN_INVALID_VALUE_ID;
    std::uint32_t scaled_id = YNN_INVALID_VALUE_ID;

    auto status =
        runtime::check_ynn_status(ynn_define_reduce(graph_, ynn_reduce_sum, REDUCE_AXIS.size(), REDUCE_AXIS.data(),
                                                    input_id, YNN_INVALID_VALUE_ID, &sum_id, YNN_NODE_FLAG_KEEP_DIMS),
                                  "define layer norm sum");
    if (status)
        status = runtime::check_ynn_status(
            ynn_define_binary(graph_, ynn_binary_multiply, sum_id, *reciprocal_size_id, &mean_id, 0),
            "define layer norm mean");
    if (status)
        status = runtime::check_ynn_status(
            ynn_define_binary(graph_, ynn_binary_subtract, input_id, mean_id, &centered_id, 0),
            "center layer norm input");
    if (status)
        status = runtime::check_ynn_status(ynn_define_unary(graph_, ynn_unary_square, centered_id, &squared_id, 0),
                                           "square layer norm input");
    if (status)
        status = runtime::check_ynn_status(
            ynn_define_reduce(graph_, ynn_reduce_sum, REDUCE_AXIS.size(), REDUCE_AXIS.data(), squared_id,
                              YNN_INVALID_VALUE_ID, &square_sum_id, YNN_NODE_FLAG_KEEP_DIMS),
            "define layer norm square sum");
    if (status)
        status = runtime::check_ynn_status(
            ynn_define_binary(graph_, ynn_binary_multiply, square_sum_id, *reciprocal_size_id, &variance_id, 0),
            "define layer norm variance");
    if (status)
        status = runtime::check_ynn_status(
            ynn_define_binary(graph_, ynn_binary_add, variance_id, *epsilon_id, &stabilized_id, 0),
            "stabilize layer norm variance");
    if (status)
        status = runtime::check_ynn_status(
            ynn_define_unary(graph_, ynn_unary_rsqrt, stabilized_id, &reciprocal_standard_deviation_id, 0),
            "define layer norm reciprocal standard deviation");
    if (status)
        status = runtime::check_ynn_status(ynn_define_binary(graph_, ynn_binary_multiply, centered_id,
                                                             reciprocal_standard_deviation_id, &normalized_id, 0),
                                           "normalize layer norm input");
    if (status)
        status = runtime::check_ynn_status(
            ynn_define_binary(graph_, ynn_binary_multiply, normalized_id, *scale_id, &scaled_id, 0),
            "scale layer norm output");
    if (status)
        status = runtime::check_ynn_status(
            ynn_define_binary(graph_, ynn_binary_add, scaled_id, *bias_id, &output_id, 0), "bias layer norm output");
    if (!status) return std::unexpected(std::move(status.error()));
    return output_id;
}

Result<std::uint32_t> TransformerBuilder::feed_forward(std::uint32_t input_id, std::string_view prefix,
                                                       std::uint32_t output_id) const {
    auto hidden_id = linear(input_id, std::string(prefix) + ".w_1", hidden_size_, feed_forward_size_);
    if (!hidden_id) return std::unexpected(std::move(hidden_id.error()));
    auto activated_id = gelu(*hidden_id);
    if (!activated_id) return std::unexpected(std::move(activated_id.error()));
    return linear(*activated_id, std::string(prefix) + ".w_2", feed_forward_size_, hidden_size_, output_id);
}

Result<std::uint32_t> TransformerBuilder::attention(std::uint32_t query_id, std::uint32_t key_id,
                                                    std::uint32_t value_id, std::uint32_t mask_id,
                                                    std::string_view prefix, std::uint32_t output_id) const {
    if (attention_heads_ <= 0 || hidden_size_ % attention_heads_ != 0) {
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "invalid attention dimensions"});
    }

    auto query_projection_id = linear(query_id, std::string(prefix) + ".linears.0", hidden_size_, hidden_size_);
    if (!query_projection_id) return std::unexpected(std::move(query_projection_id.error()));
    auto key_projection_id = linear(key_id, std::string(prefix) + ".linears.1", hidden_size_, hidden_size_);
    if (!key_projection_id) return std::unexpected(std::move(key_projection_id.error()));
    auto value_projection_id = linear(value_id, std::string(prefix) + ".linears.2", hidden_size_, hidden_size_);
    if (!value_projection_id) return std::unexpected(std::move(value_projection_id.error()));

    const std::array<std::size_t, 2> HEAD_SPLIT = {
        static_cast<std::size_t>(attention_heads_),
        static_cast<std::size_t>(hidden_size_ / attention_heads_),
    };
    constexpr std::array<std::int32_t, 4> HEAD_MAJOR_AXES = {0, 2, 1, 3};
    const auto split_heads = [&](std::uint32_t input_id) -> Result<std::uint32_t> {
        std::uint32_t split_id = YNN_INVALID_VALUE_ID;
        auto status = runtime::check_ynn_status(
            ynn_define_split_dim(graph_, -1, HEAD_SPLIT.size(), HEAD_SPLIT.data(), input_id, &split_id, 0),
            "split attention heads");
        std::uint32_t transposed_id = YNN_INVALID_VALUE_ID;
        if (status)
            status = runtime::check_ynn_status(
                ynn_define_static_transpose(graph_, HEAD_MAJOR_AXES.size(), HEAD_MAJOR_AXES.data(), split_id,
                                            &transposed_id, 0),
                "transpose attention heads");
        if (!status) return std::unexpected(std::move(status.error()));
        return transposed_id;
    };

    auto head_query_id = split_heads(*query_projection_id);
    if (!head_query_id) return std::unexpected(std::move(head_query_id.error()));
    auto head_key_id = split_heads(*key_projection_id);
    if (!head_key_id) return std::unexpected(std::move(head_key_id.error()));
    auto head_value_id = split_heads(*value_projection_id);
    if (!head_value_id) return std::unexpected(std::move(head_value_id.error()));

    constexpr std::array<std::int32_t, 4> TRANSPOSE_LAST_AXES = {0, 1, 3, 2};
    std::uint32_t transposed_key_id = YNN_INVALID_VALUE_ID;
    std::uint32_t scores_id = YNN_INVALID_VALUE_ID;
    std::uint32_t scaled_scores_id = YNN_INVALID_VALUE_ID;
    std::uint32_t masked_scores_id = YNN_INVALID_VALUE_ID;
    std::uint32_t probabilities_id = YNN_INVALID_VALUE_ID;
    std::uint32_t context_id = YNN_INVALID_VALUE_ID;
    std::uint32_t sequence_major_context_id = YNN_INVALID_VALUE_ID;
    std::uint32_t concatenated_context_id = YNN_INVALID_VALUE_ID;
    auto scale_id = scalar(1.0F / std::sqrt(static_cast<float>(hidden_size_ / attention_heads_)));
    if (!scale_id) return std::unexpected(std::move(scale_id.error()));

    auto status = runtime::check_ynn_status(
        ynn_define_static_transpose(graph_, TRANSPOSE_LAST_AXES.size(), TRANSPOSE_LAST_AXES.data(), *head_key_id,
                                    &transposed_key_id, 0),
        "transpose attention key");
    if (status)
        status = runtime::check_ynn_status(
            ynn_define_dot(graph_, 1, *head_query_id, transposed_key_id, YNN_INVALID_VALUE_ID, &scores_id, 0),
            "define attention scores");
    if (status)
        status = runtime::check_ynn_status(
            ynn_define_binary(graph_, ynn_binary_multiply, scores_id, *scale_id, &scaled_scores_id, 0),
            "scale attention scores");
    if (status && mask_id != YNN_INVALID_VALUE_ID)
        status = runtime::check_ynn_status(
            ynn_define_binary(graph_, ynn_binary_add, scaled_scores_id, mask_id, &masked_scores_id, 0),
            "mask attention scores");
    if (mask_id == YNN_INVALID_VALUE_ID) masked_scores_id = scaled_scores_id;
    if (status)
        status = runtime::check_ynn_status(ynn::define_softmax(graph_, masked_scores_id, 1.0F, probabilities_id),
                                           "define attention softmax");
    if (status)
        status = runtime::check_ynn_status(
            ynn_define_dot(graph_, 1, probabilities_id, *head_value_id, YNN_INVALID_VALUE_ID, &context_id, 0),
            "define attention context");
    if (status)
        status = runtime::check_ynn_status(
            ynn_define_static_transpose(graph_, HEAD_MAJOR_AXES.size(), HEAD_MAJOR_AXES.data(), context_id,
                                        &sequence_major_context_id, 0),
            "transpose attention context");
    if (status)
        status = runtime::check_ynn_status(
            ynn_define_fuse_dim(graph_, -2, 2, sequence_major_context_id, &concatenated_context_id, 0),
            "concatenate attention heads");
    if (!status) return std::unexpected(std::move(status.error()));
    return linear(concatenated_context_id, std::string(prefix) + ".linears.3", hidden_size_, hidden_size_, output_id);
}

} // namespace kidi::rtg