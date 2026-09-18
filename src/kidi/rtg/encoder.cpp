#include "kidi/rtg/encoder.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>

#include <ynnpack.h>

#include "kidi/rtg/transformer_builder.h"

namespace kidi::rtg {
namespace {

constexpr std::uint32_t INPUT_ID = 0;
constexpr std::uint32_t OUTPUT_ID = 1;

} // namespace

EncoderGraph::EncoderGraph(runtime::YnnExecutable executable, std::int32_t hidden_size,
                           std::int32_t maximum_source_tokens) noexcept
    : executable_(std::move(executable)), hidden_size_(hidden_size), maximum_source_tokens_(maximum_source_tokens) {}

std::expected<EncoderGraph, core::Error> EncoderGraph::create(const Package& package) {
    const auto& manifest = package.manifest();
    const auto& architecture = manifest.architecture;
    auto graph = runtime::YnnGraph::create(2);
    if (!graph) return std::unexpected(std::move(graph.error()));

    const std::array<std::size_t, 3> input_shape = {
        1,
        0,
        static_cast<std::size_t>(architecture.hidden_size),
    };
    std::uint32_t input_id = INPUT_ID;
    std::uint32_t output_id = OUTPUT_ID;
    auto status =
        runtime::check_ynn_status(ynn_define_tensor(graph->get(), ynn_type_fp32, input_shape.size(), input_shape.data(),
                                                    nullptr, YNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id),
                                  "define encoder input");
    if (status)
        status = runtime::check_ynn_status(ynn_define_tensor(graph->get(), ynn_type_fp32, input_shape.size(), nullptr,
                                                             nullptr, YNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id),
                                           "define encoder output");
    if (!status) return std::unexpected(std::move(status.error()));

    TransformerBuilder builder(graph->get(), package.weights(), architecture.hidden_size,
                               architecture.feed_forward_size, architecture.attention_heads,
                               architecture.layer_norm_epsilon);
    std::uint32_t hidden_id = input_id;
    for (std::int32_t layer = 0; layer < architecture.encoder_layers; ++layer) {
        const auto prefix = "encoder.layers." + std::to_string(layer);
        auto normalized_attention_id = builder.layer_norm(hidden_id, prefix + ".sublayer.0.norm");
        if (!normalized_attention_id) return std::unexpected(std::move(normalized_attention_id.error()));
        auto attention_id = builder.attention(*normalized_attention_id, *normalized_attention_id,
                                              *normalized_attention_id, YNN_INVALID_VALUE_ID, prefix + ".self_attn");
        if (!attention_id) return std::unexpected(std::move(attention_id.error()));

        std::uint32_t attention_residual_id = YNN_INVALID_VALUE_ID;
        status = runtime::check_ynn_status(
            ynn_define_binary(graph->get(), ynn_binary_add, hidden_id, *attention_id, &attention_residual_id, 0),
            "define encoder attention residual");
        if (!status) return std::unexpected(std::move(status.error()));

        auto normalized_feed_forward_id = builder.layer_norm(attention_residual_id, prefix + ".sublayer.1.norm");
        if (!normalized_feed_forward_id) return std::unexpected(std::move(normalized_feed_forward_id.error()));
        auto feed_forward_id = builder.feed_forward(*normalized_feed_forward_id, prefix + ".feed_forward");
        if (!feed_forward_id) return std::unexpected(std::move(feed_forward_id.error()));

        std::uint32_t residual_id = YNN_INVALID_VALUE_ID;
        status = runtime::check_ynn_status(
            ynn_define_binary(graph->get(), ynn_binary_add, attention_residual_id, *feed_forward_id, &residual_id, 0),
            "define encoder feed-forward residual");
        if (!status) return std::unexpected(std::move(status.error()));
        hidden_id = residual_id;
    }

    auto result_id = builder.layer_norm(hidden_id, "encoder.norm", output_id);
    if (!result_id) return std::unexpected(std::move(result_id.error()));
    auto executable = std::move(*graph).compile();
    if (!executable) return std::unexpected(std::move(executable.error()));
    return EncoderGraph(std::move(*executable), architecture.hidden_size, manifest.limits.source_tokens);
}

std::expected<std::vector<float>, core::Error> EncoderGraph::run(std::span<const float> embeddings) {
    if (embeddings.empty() || embeddings.size() % static_cast<std::size_t>(hidden_size_) != 0) {
        return std::unexpected(core::Error{core::ErrorCode::INVALID_ARGUMENT, "encoder input has incompatible shape"});
    }
    const auto token_count = embeddings.size() / static_cast<std::size_t>(hidden_size_);
    if (token_count > static_cast<std::size_t>(maximum_source_tokens_)) {
        return std::unexpected(
            core::Error{core::ErrorCode::INVALID_ARGUMENT, "encoder input exceeds the source token limit"});
    }

    const std::array<std::size_t, 3> shape = {
        1,
        token_count,
        static_cast<std::size_t>(hidden_size_),
    };
    std::vector<float> output(embeddings.size());
    auto status = executable_.set_shape(INPUT_ID, shape);
    if (status) status = executable_.reshape();
    if (status) status = executable_.bind(INPUT_ID, const_cast<float*>(embeddings.data()));
    if (status) status = executable_.bind(OUTPUT_ID, output.data());
    if (status) status = executable_.invoke();
    if (!status) return std::unexpected(std::move(status.error()));
    return output;
}

Encoder::Encoder(EmbeddingGraph embedding, EncoderGraph graph) noexcept
    : embedding_(std::move(embedding)), graph_(std::move(graph)) {}

std::expected<Encoder, core::Error> Encoder::create(const Package& package) {
    const auto& architecture = package.manifest().architecture;
    auto embedding = EmbeddingGraph::create(package.weights(), "src_embed.0.lut.weight",
                                            architecture.source_vocabulary_size, architecture.hidden_size);
    if (!embedding) return std::unexpected(std::move(embedding.error()));
    auto graph = EncoderGraph::create(package);
    if (!graph) return std::unexpected(std::move(graph.error()));
    return Encoder(std::move(*embedding), std::move(*graph));
}

std::expected<std::vector<float>, core::Error> Encoder::run(std::span<const std::int32_t> token_ids) {
    auto embeddings = embedding_.run(token_ids);
    if (!embeddings) return std::unexpected(std::move(embeddings.error()));
    return graph_.run(*embeddings);
}

} // namespace kidi::rtg