#include "kidi/rtg/embedding.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <string>
#include <utility>

namespace kidi::rtg {
namespace {

constexpr std::uint32_t TOKEN_IDS_ID = 0;
constexpr std::uint32_t POSITIONS_ID = 1;
constexpr std::uint32_t OUTPUT_ID = 2;

std::vector<float> positional_encoding(std::size_t length, std::int32_t hidden_size) {
    std::vector<float> result(length * static_cast<std::size_t>(hidden_size));
    for (std::size_t position = 0; position < length; ++position) {
        for (std::int32_t index = 0; index < hidden_size; index += 2) {
            const auto exponent = static_cast<float>(index) * -(std::log(10000.0F) / hidden_size);
            const auto angle = static_cast<float>(position) * std::exp(exponent);
            result[position * hidden_size + index] = std::sin(angle);
            result[position * hidden_size + index + 1] = std::cos(angle);
        }
    }
    return result;
}

} // namespace

EmbeddingGraph::EmbeddingGraph(runtime::YnnExecutable executable, std::int32_t vocabulary_size,
                               std::int32_t hidden_size, std::unique_ptr<float> scale)
    : executable_(std::move(executable)),
      vocabulary_size_(vocabulary_size),
      hidden_size_(hidden_size),
      scale_(std::move(scale)),
      token_ids_id_(TOKEN_IDS_ID),
      positions_id_(POSITIONS_ID),
      output_id_(OUTPUT_ID) {}

Result<EmbeddingGraph> EmbeddingGraph::create(const model::Weights& weights, std::string_view weight_name,
                                              std::int32_t vocabulary_size, std::int32_t hidden_size,
                                              model::WeightEncoding weight_encoding) {
    if (vocabulary_size <= 0 || hidden_size <= 0 || hidden_size % 2 != 0) {
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "invalid embedding dimensions"});
    }
    auto weight = weights.tensor(weight_name);
    if (!weight) return std::unexpected(std::move(weight.error()));
    const auto data_type = model::matrix_data_type(weight_encoding);
    const std::array<std::int64_t, 2> expected_shape = {vocabulary_size, hidden_size};
    if (weight->data_type != data_type || !std::ranges::equal(weight->shape, expected_shape)) {
        return std::unexpected(Error{
            ErrorCode::INVALID_ARGUMENT,
            "embedding weight has incompatible dtype or shape: " + std::string(weight_name),
        });
    }
    std::optional<model::TensorView> quantization_scale;
    if (weight_encoding == model::WeightEncoding::INT8_PER_CHANNEL) {
        const auto scale_name = model::quantization_scale_name(weight_name);
        auto scale = weights.tensor(scale_name);
        if (!scale) return std::unexpected(std::move(scale.error()));
        const std::array<std::int64_t, 2> scale_shape = {vocabulary_size, 1};
        if (scale->data_type != model::DataType::F32 || !std::ranges::equal(scale->shape, scale_shape)) {
            return std::unexpected(Error{
                ErrorCode::INVALID_ARGUMENT,
                "embedding scale has incompatible dtype or shape: " + scale_name,
            });
        }
        quantization_scale = *scale;
    }

    const auto graph_flags = weight_encoding == model::WeightEncoding::BF16 ? YNN_FLAG_NO_EXCESS_PRECISION : 0;
    auto graph = runtime::YnnGraph::create(3, graph_flags);
    if (!graph) return std::unexpected(std::move(graph.error()));
    const std::array<std::size_t, 3> token_shape = {0, 0, 1};
    const std::array<std::size_t, 3> hidden_shape = {1, 0, static_cast<std::size_t>(hidden_size)};
    const std::array<std::size_t, 3> weight_shape = {
        static_cast<std::size_t>(vocabulary_size),
        1,
        static_cast<std::size_t>(hidden_size),
    };
    const std::array<std::int32_t, 1> gather_axes = {0};
    std::uint32_t token_id = TOKEN_IDS_ID;
    std::uint32_t positions_id = POSITIONS_ID;
    std::uint32_t output_id = OUTPUT_ID;
    std::uint32_t weight_id = YNN_INVALID_VALUE_ID;
    std::uint32_t gathered_id = YNN_INVALID_VALUE_ID;
    std::uint32_t quantization_scale_id = YNN_INVALID_VALUE_ID;
    std::uint32_t gathered_quantization_scale_id = YNN_INVALID_VALUE_ID;
    std::uint32_t dequantized_id = YNN_INVALID_VALUE_ID;
    std::uint32_t scaled_id = YNN_INVALID_VALUE_ID;
    std::uint32_t scale_id = YNN_INVALID_VALUE_ID;
    auto scale = std::make_unique<float>(std::sqrt(static_cast<float>(hidden_size)));
    const auto weight_type = weight_encoding == model::WeightEncoding::BF16               ? ynn_type_bf16
                             : weight_encoding == model::WeightEncoding::INT8_PER_CHANNEL ? ynn_type_int8
                                                                                          : ynn_type_fp32;

    auto status = runtime::check_ynn_status(
        ynn_define_tensor(graph->get(), ynn_type_int32, token_shape.size(), token_shape.data(), nullptr,
                          YNN_VALUE_FLAG_EXTERNAL_INPUT, &token_id),
        "define token IDs");
    if (status)
        status = runtime::check_ynn_status(
            ynn_define_tensor(graph->get(), ynn_type_fp32, hidden_shape.size(), hidden_shape.data(), nullptr,
                              YNN_VALUE_FLAG_EXTERNAL_INPUT, &positions_id),
            "define positions");
    if (status)
        status = runtime::check_ynn_status(
            ynn_define_tensor(graph->get(), ynn_type_fp32, hidden_shape.size(), hidden_shape.data(), nullptr,
                              YNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id),
            "define embedding output");
    if (status)
        status = runtime::check_ynn_status(ynn_define_tensor(graph->get(), weight_type, weight_shape.size(),
                                                             weight_shape.data(), weight->data(), 0, &weight_id),
                                           "define embedding weight");
    if (status && quantization_scale) {
        const std::array<std::size_t, 3> quantization_scale_shape = {static_cast<std::size_t>(vocabulary_size), 1, 1};
        status = runtime::check_ynn_status(
            ynn_define_tensor(graph->get(), ynn_type_fp32, quantization_scale_shape.size(),
                              quantization_scale_shape.data(), quantization_scale->data(), 0, &quantization_scale_id),
            "define embedding quantization scale");
    }
    if (status)
        status = runtime::check_ynn_status(
            ynn_define_tensor(graph->get(), ynn_type_fp32, 0, nullptr, scale.get(), 0, &scale_id),
            "define embedding scale");
    if (status)
        status = runtime::check_ynn_status(ynn_define_gather(graph->get(), gather_axes.size(), gather_axes.data(), 3,
                                                             weight_id, token_id, &gathered_id, 0),
                                           "define embedding gather");
    if (status && quantization_scale)
        status = runtime::check_ynn_status(
            ynn_define_gather(graph->get(), gather_axes.size(), gather_axes.data(), 3, quantization_scale_id, token_id,
                              &gathered_quantization_scale_id, 0),
            "define embedding scale gather");
    if (status && quantization_scale)
        status = runtime::check_ynn_status(
            ynn_define_dequantize(graph->get(), gathered_id, YNN_INVALID_VALUE_ID, gathered_quantization_scale_id,
                                  ynn_type_fp32, &dequantized_id, 0),
            "dequantize embedding");
    if (!quantization_scale) dequantized_id = gathered_id;
    if (status)
        status = runtime::check_ynn_status(
            ynn_define_binary(graph->get(), ynn_binary_multiply, dequantized_id, scale_id, &scaled_id, 0),
            "scale embeddings");
    if (status)
        status = runtime::check_ynn_status(
            ynn_define_binary(graph->get(), ynn_binary_add, scaled_id, positions_id, &output_id, 0),
            "add positional encoding");
    if (!status) return std::unexpected(std::move(status.error()));

    auto executable = std::move(*graph).compile();
    if (!executable) return std::unexpected(std::move(executable.error()));
    return EmbeddingGraph(std::move(*executable), vocabulary_size, hidden_size, std::move(scale));
}

Result<std::vector<float>> EmbeddingGraph::run(std::span<const std::int32_t> token_ids, std::size_t batch_size) {
    if (batch_size == 0 || token_ids.empty() || token_ids.size() % batch_size != 0) {
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "cannot embed an empty sequence"});
    }
    for (const auto token_id : token_ids) {
        if (token_id < 0 || token_id >= vocabulary_size_) {
            return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "token ID is outside the vocabulary"});
        }
    }

    const auto sequence_length = token_ids.size() / batch_size;
    const std::array<std::size_t, 3> token_shape = {batch_size, sequence_length, 1};
    const std::array<std::size_t, 3> hidden_shape = {1, sequence_length, static_cast<std::size_t>(hidden_size_)};
    auto positions = positional_encoding(sequence_length, hidden_size_);
    std::vector<std::int32_t> owned_token_ids(token_ids.begin(), token_ids.end());
    std::vector<float> output(token_ids.size() * static_cast<std::size_t>(hidden_size_));

    auto status = executable_.set_shape(token_ids_id_, token_shape);
    if (status) status = executable_.set_shape(positions_id_, hidden_shape);
    if (status) status = executable_.reshape();
    if (status) status = executable_.bind(token_ids_id_, owned_token_ids.data());
    if (status) status = executable_.bind(positions_id_, positions.data());
    if (status) status = executable_.bind(output_id_, output.data());
    if (status) status = executable_.invoke();
    if (!status) return std::unexpected(std::move(status.error()));
    return output;
}

} // namespace kidi::rtg