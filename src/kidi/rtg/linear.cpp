#include "kidi/rtg/linear.h"

#include <algorithm>
#include <array>
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

LinearImpl::LinearImpl(ynn_subgraph_t graph, const model::Weights& weights,
                       model::WeightEncoding weight_encoding) noexcept
    : graph_(graph), weights_(weights), weight_encoding_(weight_encoding) {}

Result<std::uint32_t> LinearImpl::weight(std::string_view name, std::int32_t first_extent, std::int32_t second_extent,
                                         model::DataType data_type) const {
    auto tensor = weights_.tensor(name);
    if (!tensor) return std::unexpected(std::move(tensor.error()));
    const bool shape_matches = second_extent == 0 ? tensor->shape.size() == 1 && tensor->shape[0] == first_extent
                                                  : tensor->shape.size() == 2 && tensor->shape[0] == first_extent &&
                                                        tensor->shape[1] == second_extent;
    if (tensor->data_type != data_type || !shape_matches) {
        return std::unexpected(
            Error{ErrorCode::INVALID_ARGUMENT, "weight has incompatible dtype or shape: " + std::string(name)});
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
        "define mapped linear parameter");
    if (!status) return std::unexpected(std::move(status.error()));
    return id;
}

Result<LinearImpl::Parameters> LinearImpl::parameters(std::string_view weight_name, std::string_view bias_name,
                                                      std::int32_t input_size, std::int32_t output_size) const {
    auto weight_id = weight(weight_name, input_size, output_size, model::matrix_data_type(weight_encoding_));
    if (!weight_id) return std::unexpected(std::move(weight_id.error()));
    auto bias_id = weight(bias_name, output_size);
    if (!bias_id) return std::unexpected(std::move(bias_id.error()));

    std::optional<QState> qstate;
    if (weight_encoding_ == model::WeightEncoding::INT8_PER_CHANNEL) {
        auto scale_id = weight(model::quantization_scale_name(weight_name), output_size, 1);
        if (!scale_id) return std::unexpected(std::move(scale_id.error()));
        qstate = QState{.scale_id = *scale_id};
    }
    return Parameters{.weight_id = *weight_id, .bias_id = *bias_id, .qstate = qstate};
}

Result<std::uint32_t> LinearImpl::define(std::uint32_t input_id, std::string_view prefix, std::int32_t input_size,
                                         std::int32_t output_size, std::uint32_t output_id) const {
    return define(input_id, std::string(prefix) + ".weight", std::string(prefix) + ".bias", input_size, output_size,
                  output_id);
}

Result<std::uint32_t> LinearImpl::define(std::uint32_t input_id, std::string_view weight_name,
                                         std::string_view bias_name, std::int32_t input_size, std::int32_t output_size,
                                         std::uint32_t output_id) const {
    auto params = parameters(weight_name, bias_name, input_size, output_size);
    if (!params) return std::unexpected(std::move(params.error()));

    Result<void> status;
    if (params->qstate) {
        constexpr std::array<std::int32_t, 1> REDUCE_AXIS = {-1};
        std::uint32_t min_max_id = YNN_INVALID_VALUE_ID;
        std::uint32_t input_zero_point_id = YNN_INVALID_VALUE_ID;
        std::uint32_t input_scale_id = YNN_INVALID_VALUE_ID;
        std::uint32_t quantized_input_id = YNN_INVALID_VALUE_ID;
        status = runtime::check_ynn_status(
            ynn_define_reduce(graph_, ynn_reduce_min_max, REDUCE_AXIS.size(), REDUCE_AXIS.data(), input_id,
                              YNN_INVALID_VALUE_ID, &min_max_id, YNN_NODE_FLAG_KEEP_DIMS),
            "define INT8 linear input range");
        if (status) {
            status =
                runtime::check_ynn_status(ynn_define_dynamic_quantization(graph_, min_max_id, ynn_type_int8,
                                                                          &input_zero_point_id, &input_scale_id, 0),
                                          "define INT8 linear input quantization");
        }
        if (status) {
            status = runtime::check_ynn_status(ynn_define_quantize(graph_, input_id, ynn_type_int8, input_zero_point_id,
                                                                   input_scale_id, &quantized_input_id, 0),
                                               "quantize linear input");
        }
        if (status) {
            status = runtime::check_ynn_status(
                ynn::define_blockwise_dot(graph_, quantized_input_id, input_zero_point_id, input_scale_id,
                                          params->weight_id, YNN_INVALID_VALUE_ID, params->qstate->scale_id,
                                          static_cast<std::size_t>(input_size), params->bias_id, ynn_type_fp32,
                                          output_id, 0),
                "define INT8 linear");
        }
    } else {
        std::uint32_t dot_input_id = input_id;
        if (weight_encoding_ == model::WeightEncoding::BF16) {
            dot_input_id = YNN_INVALID_VALUE_ID;
            status = runtime::check_ynn_status(
                ynn_define_convert(graph_, input_id, ynn_type_bf16, &dot_input_id, YNN_NODE_FLAG_NO_EXCESS_PRECISION),
                "convert linear input to BF16");
        }
        if (status) {
            status = runtime::check_ynn_status(
                ynn_define_dot(graph_, 1, dot_input_id, params->weight_id, params->bias_id, &output_id, 0),
                "define linear");
        }
    }
    if (!status) return std::unexpected(std::move(status.error()));
    return output_id;
}

Result<std::vector<std::uint32_t>> LinearImpl::define_split(std::uint32_t input_id, std::string_view prefix,
                                                            std::int32_t input_size, std::int32_t output_size,
                                                            std::size_t output_count,
                                                            std::span<const std::uint32_t> output_ids) const {
    if (output_count < 2 || (!output_ids.empty() && output_ids.size() != output_count)) {
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "invalid split linear output count"});
    }
    auto fused = define(input_id, prefix, input_size, static_cast<std::int32_t>(output_count) * output_size);
    if (!fused) return std::unexpected(std::move(fused.error()));

    std::vector<std::uint32_t> result(output_count, YNN_INVALID_VALUE_ID);
    if (!output_ids.empty()) std::ranges::copy(output_ids, result.begin());
    auto status = runtime::check_ynn_status(ynn_define_even_split(graph_, -1, *fused, result.size(), result.data(), 0),
                                            "split linear output");
    if (!status) return std::unexpected(std::move(status.error()));
    return result;
}

Result<std::uint32_t> LinearImpl::define_tied(std::uint32_t input_id, std::string_view embedding_name,
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
    if (status) {
        status = runtime::check_ynn_status(ynn_define_dot(graph_, 1, input_id, transposed_id, *bias_id, &output_id, 0),
                                           "define tied projection");
    }
    if (!status) return std::unexpected(std::move(status.error()));
    return output_id;
}

} // namespace kidi::rtg
