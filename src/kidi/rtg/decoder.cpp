#include "kidi/rtg/decoder.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>

#include <ynnpack.h>

#include "kidi/rtg/transformer_builder.h"
#include "ynnpack/composites/composites.h"

namespace kidi::rtg {
namespace {

constexpr std::uint32_t DECODER_INPUT_ID = 0;
constexpr std::uint32_t MEMORY_ID = 1;
constexpr std::uint32_t CAUSAL_MASK_ID = 2;
constexpr std::uint32_t DECODER_OUTPUT_ID = 3;
constexpr std::uint32_t GENERATOR_INPUT_ID = 0;
constexpr std::uint32_t GENERATOR_OUTPUT_ID = 1;

} // namespace

DecoderGraph::DecoderGraph(runtime::YnnExecutable executable, std::int32_t hidden_size,
                           std::int32_t maximum_position) noexcept
    : executable_(std::move(executable)), hidden_size_(hidden_size), maximum_position_(maximum_position) {}

Result<DecoderGraph> DecoderGraph::create(const Package& package) {
    const auto& architecture = package.manifest().architecture;
    auto graph = runtime::YnnGraph::create(4);
    if (!graph) return std::unexpected(std::move(graph.error()));

    const std::array<std::size_t, 3> decoder_shape = {
        0,
        0,
        static_cast<std::size_t>(architecture.hidden_size),
    };
    const std::array<std::size_t, 3> memory_shape = {
        1,
        0,
        static_cast<std::size_t>(architecture.hidden_size),
    };
    constexpr std::array<std::size_t, 4> MASK_SHAPE = {1, 1, 0, 0};
    std::uint32_t input_id = DECODER_INPUT_ID;
    std::uint32_t memory_id = MEMORY_ID;
    std::uint32_t mask_id = CAUSAL_MASK_ID;
    std::uint32_t output_id = DECODER_OUTPUT_ID;
    auto status = runtime::check_ynn_status(
        ynn_define_tensor(graph->get(), ynn_type_fp32, decoder_shape.size(), decoder_shape.data(), nullptr,
                          YNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id),
        "define decoder input");
    if (status)
        status = runtime::check_ynn_status(
            ynn_define_tensor(graph->get(), ynn_type_fp32, memory_shape.size(), memory_shape.data(), nullptr,
                              YNN_VALUE_FLAG_EXTERNAL_INPUT, &memory_id),
            "define decoder memory");
    if (status)
        status = runtime::check_ynn_status(
            ynn_define_tensor(graph->get(), ynn_type_fp32, MASK_SHAPE.size(), MASK_SHAPE.data(), nullptr,
                              YNN_VALUE_FLAG_EXTERNAL_INPUT, &mask_id),
            "define decoder causal mask");
    if (status)
        status = runtime::check_ynn_status(ynn_define_tensor(graph->get(), ynn_type_fp32, decoder_shape.size(), nullptr,
                                                             nullptr, YNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id),
                                           "define decoder output");
    if (!status) return std::unexpected(std::move(status.error()));

    TransformerBuilder builder(graph->get(), package.weights(), architecture.hidden_size,
                               architecture.feed_forward_size, architecture.attention_heads,
                               architecture.layer_norm_epsilon);
    std::uint32_t hidden_id = input_id;
    for (std::int32_t layer = 0; layer < architecture.decoder_layers; ++layer) {
        const auto prefix = "decoder.layers." + std::to_string(layer);
        auto normalized_self_attention_id = builder.layer_norm(hidden_id, prefix + ".sublayer.0.norm");
        if (!normalized_self_attention_id) return std::unexpected(std::move(normalized_self_attention_id.error()));
        auto self_attention_id = builder.attention(*normalized_self_attention_id, *normalized_self_attention_id,
                                                   *normalized_self_attention_id, mask_id, prefix + ".self_attn");
        if (!self_attention_id) return std::unexpected(std::move(self_attention_id.error()));

        std::uint32_t self_residual_id = YNN_INVALID_VALUE_ID;
        status = runtime::check_ynn_status(
            ynn_define_binary(graph->get(), ynn_binary_add, hidden_id, *self_attention_id, &self_residual_id, 0),
            "define decoder self-attention residual");
        if (!status) return std::unexpected(std::move(status.error()));

        auto normalized_source_attention_id = builder.layer_norm(self_residual_id, prefix + ".sublayer.1.norm");
        if (!normalized_source_attention_id) return std::unexpected(std::move(normalized_source_attention_id.error()));
        auto source_attention_id = builder.attention(*normalized_source_attention_id, memory_id, memory_id,
                                                     YNN_INVALID_VALUE_ID, prefix + ".src_attn");
        if (!source_attention_id) return std::unexpected(std::move(source_attention_id.error()));

        std::uint32_t source_residual_id = YNN_INVALID_VALUE_ID;
        status = runtime::check_ynn_status(ynn_define_binary(graph->get(), ynn_binary_add, self_residual_id,
                                                             *source_attention_id, &source_residual_id, 0),
                                           "define decoder source-attention residual");
        if (!status) return std::unexpected(std::move(status.error()));

        auto normalized_feed_forward_id = builder.layer_norm(source_residual_id, prefix + ".sublayer.2.norm");
        if (!normalized_feed_forward_id) return std::unexpected(std::move(normalized_feed_forward_id.error()));
        auto feed_forward_id = builder.feed_forward(*normalized_feed_forward_id, prefix + ".feed_forward");
        if (!feed_forward_id) return std::unexpected(std::move(feed_forward_id.error()));

        std::uint32_t residual_id = YNN_INVALID_VALUE_ID;
        status = runtime::check_ynn_status(
            ynn_define_binary(graph->get(), ynn_binary_add, source_residual_id, *feed_forward_id, &residual_id, 0),
            "define decoder feed-forward residual");
        if (!status) return std::unexpected(std::move(status.error()));
        hidden_id = residual_id;
    }

    auto result_id = builder.layer_norm(hidden_id, "decoder.norm", output_id);
    if (!result_id) return std::unexpected(std::move(result_id.error()));
    auto executable = std::move(*graph).compile();
    if (!executable) return std::unexpected(std::move(executable.error()));
    return DecoderGraph(std::move(*executable), architecture.hidden_size, architecture.maximum_position);
}

Result<std::vector<float>> DecoderGraph::run(std::span<const float> embeddings, std::size_t batch_size,
                                             std::span<const float> memory) {
    const auto hidden_size = static_cast<std::size_t>(hidden_size_);
    if (batch_size == 0 || embeddings.empty() || embeddings.size() % (batch_size * hidden_size) != 0 ||
        memory.empty() || memory.size() % hidden_size != 0) {
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "decoder input has incompatible shape"});
    }
    const auto target_length = embeddings.size() / (batch_size * hidden_size);
    const auto source_length = memory.size() / hidden_size;
    if (target_length > static_cast<std::size_t>(maximum_position_) ||
        source_length > static_cast<std::size_t>(maximum_position_)) {
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "decoder input exceeds the position limit"});
    }

    const std::array<std::size_t, 3> decoder_shape = {batch_size, target_length, hidden_size};
    const std::array<std::size_t, 3> memory_shape = {1, source_length, hidden_size};
    const std::array<std::size_t, 4> mask_shape = {1, 1, target_length, target_length};
    std::vector<float> causal_mask(target_length * target_length);
    for (std::size_t row = 0; row < target_length; ++row) {
        std::fill(causal_mask.begin() + static_cast<std::ptrdiff_t>(row * target_length + row + 1),
                  causal_mask.begin() + static_cast<std::ptrdiff_t>((row + 1) * target_length), -1.0e9F);
    }
    std::vector<float> output(embeddings.size());
    auto status = executable_.set_shape(DECODER_INPUT_ID, decoder_shape);
    if (status) status = executable_.set_shape(MEMORY_ID, memory_shape);
    if (status) status = executable_.set_shape(CAUSAL_MASK_ID, mask_shape);
    if (status) status = executable_.reshape();
    if (status) status = executable_.bind(DECODER_INPUT_ID, const_cast<float*>(embeddings.data()));
    if (status) status = executable_.bind(MEMORY_ID, const_cast<float*>(memory.data()));
    if (status) status = executable_.bind(CAUSAL_MASK_ID, causal_mask.data());
    if (status) status = executable_.bind(DECODER_OUTPUT_ID, output.data());
    if (status) status = executable_.invoke();
    if (!status) return std::unexpected(std::move(status.error()));
    return output;
}

GeneratorGraph::GeneratorGraph(runtime::YnnExecutable executable, std::int32_t hidden_size,
                               std::int32_t vocabulary_size) noexcept
    : executable_(std::move(executable)), hidden_size_(hidden_size), vocabulary_size_(vocabulary_size) {}

Result<GeneratorGraph> GeneratorGraph::create(const Package& package) {
    const auto& architecture = package.manifest().architecture;
    auto graph = runtime::YnnGraph::create(2);
    if (!graph) return std::unexpected(std::move(graph.error()));
    const std::array<std::size_t, 3> input_shape = {
        0,
        1,
        static_cast<std::size_t>(architecture.hidden_size),
    };
    std::uint32_t input_id = GENERATOR_INPUT_ID;
    std::uint32_t output_id = GENERATOR_OUTPUT_ID;
    auto status =
        runtime::check_ynn_status(ynn_define_tensor(graph->get(), ynn_type_fp32, input_shape.size(), input_shape.data(),
                                                    nullptr, YNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id),
                                  "define generator input");
    if (status)
        status = runtime::check_ynn_status(ynn_define_tensor(graph->get(), ynn_type_fp32, 3, nullptr, nullptr,
                                                             YNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id),
                                           "define generator output");
    if (!status) return std::unexpected(std::move(status.error()));

    TransformerBuilder builder(graph->get(), package.weights(), architecture.hidden_size,
                               architecture.feed_forward_size, architecture.attention_heads,
                               architecture.layer_norm_epsilon);
    auto logits_id = builder.linear(input_id, "tgt_embed.0.lut.weight", "generator.proj.bias", architecture.hidden_size,
                                    architecture.target_vocabulary_size);
    if (!logits_id) return std::unexpected(std::move(logits_id.error()));
    status = runtime::check_ynn_status(ynn::define_log_softmax(graph->get(), *logits_id, output_id),
                                       "define generator log softmax");
    if (!status) return std::unexpected(std::move(status.error()));

    auto executable = std::move(*graph).compile();
    if (!executable) return std::unexpected(std::move(executable.error()));
    return GeneratorGraph(std::move(*executable), architecture.hidden_size, architecture.target_vocabulary_size);
}

Result<std::vector<float>> GeneratorGraph::run(std::span<const float> hidden_states) {
    const auto hidden_size = static_cast<std::size_t>(hidden_size_);
    if (hidden_states.empty() || hidden_states.size() % hidden_size != 0) {
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "generator input has incompatible shape"});
    }
    const auto batch_size = hidden_states.size() / hidden_size;
    const std::array<std::size_t, 3> input_shape = {batch_size, 1, hidden_size};
    std::vector<float> output(batch_size * static_cast<std::size_t>(vocabulary_size_));
    auto status = executable_.set_shape(GENERATOR_INPUT_ID, input_shape);
    if (status) status = executable_.reshape();
    if (status) status = executable_.bind(GENERATOR_INPUT_ID, const_cast<float*>(hidden_states.data()));
    if (status) status = executable_.bind(GENERATOR_OUTPUT_ID, output.data());
    if (status) status = executable_.invoke();
    if (!status) return std::unexpected(std::move(status.error()));
    return output;
}

Decoder::Decoder(EmbeddingGraph embedding, DecoderGraph graph, GeneratorGraph generator,
                 std::int32_t hidden_size) noexcept
    : embedding_(std::move(embedding)),
      graph_(std::move(graph)),
      generator_(std::move(generator)),
      hidden_size_(hidden_size) {}

Result<Decoder> Decoder::create(const Package& package) {
    const auto& architecture = package.manifest().architecture;
    auto embedding = EmbeddingGraph::create(package.weights(), "tgt_embed.0.lut.weight",
                                            architecture.target_vocabulary_size, architecture.hidden_size);
    if (!embedding) return std::unexpected(std::move(embedding.error()));
    auto graph = DecoderGraph::create(package);
    if (!graph) return std::unexpected(std::move(graph.error()));
    auto generator = GeneratorGraph::create(package);
    if (!generator) return std::unexpected(std::move(generator.error()));
    return Decoder(std::move(*embedding), std::move(*graph), std::move(*generator), architecture.hidden_size);
}

Result<std::vector<float>> Decoder::next(std::span<const float> memory, std::span<const std::int32_t> token_ids,
                                         std::size_t batch_size) {
    if (batch_size == 0 || token_ids.empty() || token_ids.size() % batch_size != 0) {
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "decoder token input has incompatible shape"});
    }
    auto embeddings = embedding_.run(token_ids, batch_size);
    if (!embeddings) return std::unexpected(std::move(embeddings.error()));
    auto hidden_states = graph_.run(*embeddings, batch_size, memory);
    if (!hidden_states) return std::unexpected(std::move(hidden_states.error()));

    const auto target_length = token_ids.size() / batch_size;
    const auto hidden_size = static_cast<std::size_t>(hidden_size_);
    std::vector<float> last_hidden(batch_size * hidden_size);
    for (std::size_t batch = 0; batch < batch_size; ++batch) {
        const auto source_offset = (batch * target_length + target_length - 1) * hidden_size;
        std::copy_n(hidden_states->begin() + static_cast<std::ptrdiff_t>(source_offset), hidden_size,
                    last_hidden.begin() + static_cast<std::ptrdiff_t>(batch * hidden_size));
    }
    return generator_.run(last_hidden);
}

} // namespace kidi::rtg