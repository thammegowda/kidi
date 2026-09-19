#include "kidi/rtg/encoder.h"

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>

#include <ynnpack.h>

#include "kidi/rtg/transformer_builder.h"

namespace kidi::rtg {
namespace {

using Clock = std::chrono::steady_clock;

std::uint64_t elapsed_since(Clock::time_point started) {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - started).count());
}

template <typename Function>
auto measure(std::uint64_t* elapsed_ns, Function&& function) {
    if (!elapsed_ns) return std::forward<Function>(function)();
    const auto started = Clock::now();
    auto result = std::forward<Function>(function)();
    *elapsed_ns += elapsed_since(started);
    return result;
}

constexpr std::uint32_t INPUT_ID = 0;
constexpr std::uint32_t OUTPUT_ID = 1;

} // namespace

EncoderGraph::EncoderGraph(runtime::YnnExecutable executable, std::int32_t hidden_size,
                           std::int32_t maximum_source_tokens) noexcept
    : executable_(std::move(executable)), hidden_size_(hidden_size), maximum_source_tokens_(maximum_source_tokens) {}

Result<EncoderGraph> EncoderGraph::create(const Package& package) {
    const auto& manifest = package.manifest();
    const auto& architecture = manifest.architecture;
    const auto graph_flags =
        manifest.weights.encoding == model::WeightEncoding::BF16 ? YNN_FLAG_NO_EXCESS_PRECISION : 0;
    auto graph = runtime::YnnGraph::create(2, graph_flags);
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
                               architecture.layer_norm_epsilon, manifest.weights.encoding);
    std::uint32_t hidden_id = input_id;
    for (std::int32_t layer = 0; layer < architecture.encoder_layers; ++layer) {
        const auto prefix = "encoder.layers." + std::to_string(layer);
        auto normalized_attention_id = builder.layer_norm(hidden_id, prefix + ".sublayer.0.norm");
        if (!normalized_attention_id) return std::unexpected(std::move(normalized_attention_id.error()));
        auto attention_id =
            builder.self_attention(*normalized_attention_id, YNN_INVALID_VALUE_ID, prefix + ".self_attn");
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

Result<std::vector<float>> EncoderGraph::run(std::span<const float> embeddings, GraphRunStats* stats) {
    if (embeddings.empty() || embeddings.size() % static_cast<std::size_t>(hidden_size_) != 0) {
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "encoder input has incompatible shape"});
    }
    const auto token_count = embeddings.size() / static_cast<std::size_t>(hidden_size_);
    if (token_count > static_cast<std::size_t>(maximum_source_tokens_)) {
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "encoder input exceeds the source token limit"});
    }

    const auto prepare_started = stats ? Clock::now() : Clock::time_point{};
    const std::array<std::size_t, 3> shape = {
        1,
        token_count,
        static_cast<std::size_t>(hidden_size_),
    };
    std::vector<float> output(embeddings.size());
    if (stats) stats->prepare_ns += elapsed_since(prepare_started);

    const auto reshape_started = stats ? Clock::now() : Clock::time_point{};
    auto status = executable_.set_shape(INPUT_ID, shape);
    if (status) status = executable_.reshape();
    if (status && stats && stats->max_concurrency == 0) {
        auto concurrency = executable_.concurrency();
        if (concurrency) stats->max_concurrency = *concurrency;
    }
    if (stats) stats->reshape_ns += elapsed_since(reshape_started);

    const auto bind_started = stats ? Clock::now() : Clock::time_point{};
    if (status) status = executable_.bind(INPUT_ID, const_cast<float*>(embeddings.data()));
    if (status) status = executable_.bind(OUTPUT_ID, output.data());
    if (stats) stats->bind_ns += elapsed_since(bind_started);

    const auto invoke_started = stats ? Clock::now() : Clock::time_point{};
    if (status) status = executable_.invoke();
    if (stats) stats->invoke_ns += elapsed_since(invoke_started);
    if (!status) return std::unexpected(std::move(status.error()));
    return output;
}

Encoder::Encoder(EmbeddingGraph embedding, EncoderGraph graph) noexcept
    : embedding_(std::move(embedding)), graph_(std::move(graph)) {}

Result<Encoder> Encoder::create(const Package& package) {
    const auto& manifest = package.manifest();
    const auto& architecture = manifest.architecture;
    auto embedding =
        EmbeddingGraph::create(package.weights(), "src_embed.0.lut.weight", architecture.source_vocabulary_size,
                               architecture.hidden_size, manifest.weights.encoding, architecture.maximum_position);
    if (!embedding) return std::unexpected(std::move(embedding.error()));
    auto graph = EncoderGraph::create(package);
    if (!graph) return std::unexpected(std::move(graph.error()));
    return Encoder(std::move(*embedding), std::move(*graph));
}

Result<std::vector<float>> Encoder::run(std::span<const std::int32_t> token_ids, InferenceStats* stats) {
    auto embeddings = measure(stats ? &stats->source_embedding_ns : nullptr, [&] { return embedding_.run(token_ids); });
    if (!embeddings) return std::unexpected(std::move(embeddings.error()));
    return graph_.run(*embeddings, stats ? &stats->encoder_graph : nullptr);
}

} // namespace kidi::rtg