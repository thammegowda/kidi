#include "kidi/rtg/decoder.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>

#include <ynnpack.h>

#include "kidi/rtg/transformer_builder.h"
#include "ynnpack/composites/composites.h"

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

constexpr std::uint32_t SOURCE_MEMORY_ID = 0;
constexpr std::uint32_t SOURCE_OUTPUT_BASE_ID = 1;
constexpr std::uint32_t DECODER_INPUT_ID = 0;
constexpr std::uint32_t CAUSAL_MASK_ID = 1;
constexpr std::uint32_t DECODER_OUTPUT_ID = 2;
constexpr std::uint32_t DECODER_SOURCE_BASE_ID = 3;
constexpr std::uint32_t INCREMENTAL_INPUT_ID = 0;
constexpr std::uint32_t INCREMENTAL_MASK_ID = 1;
constexpr std::uint32_t INCREMENTAL_OUTPUT_ID = 2;
constexpr std::uint32_t INCREMENTAL_SOURCE_BASE_ID = 3;
constexpr std::uint32_t GENERATOR_INPUT_ID = 0;
constexpr std::uint32_t GENERATOR_OUTPUT_ID = 1;

constexpr std::uint32_t source_key_id(std::uint32_t base, std::int32_t layer) {
    return base + static_cast<std::uint32_t>(2 * layer);
}

constexpr std::uint32_t source_value_id(std::uint32_t base, std::int32_t layer) {
    return source_key_id(base, layer) + 1;
}

constexpr std::uint32_t incremental_self_base(std::int32_t decoder_layers) {
    return INCREMENTAL_SOURCE_BASE_ID + static_cast<std::uint32_t>(2 * decoder_layers);
}

constexpr std::uint32_t incremental_current_base(std::int32_t decoder_layers) {
    return incremental_self_base(decoder_layers) + static_cast<std::uint32_t>(2 * decoder_layers);
}

} // namespace

std::span<const float> SourceKVCache::key(std::size_t layer) const noexcept {
    const auto layer_size = source_length * hidden_size;
    return std::span<const float>(values).subspan(2 * layer * layer_size, layer_size);
}

std::span<const float> SourceKVCache::value(std::size_t layer) const noexcept {
    const auto layer_size = source_length * hidden_size;
    return std::span<const float>(values).subspan((2 * layer + 1) * layer_size, layer_size);
}

std::span<const float> SelfKVCache::key(std::size_t layer) const noexcept {
    const auto layer_capacity = capacity * hidden_size;
    return std::span<const float>(values).subspan(2 * layer * layer_capacity, length * hidden_size);
}

std::span<const float> SelfKVCache::value(std::size_t layer) const noexcept {
    const auto layer_capacity = capacity * hidden_size;
    return std::span<const float>(values).subspan((2 * layer + 1) * layer_capacity, length * hidden_size);
}

std::span<float> SelfKVCache::key_slot(std::size_t layer, std::size_t position) noexcept {
    const auto layer_capacity = capacity * hidden_size;
    return std::span<float>(values).subspan(2 * layer * layer_capacity + position * hidden_size, hidden_size);
}

std::span<float> SelfKVCache::value_slot(std::size_t layer, std::size_t position) noexcept {
    const auto layer_capacity = capacity * hidden_size;
    return std::span<float>(values).subspan((2 * layer + 1) * layer_capacity + position * hidden_size, hidden_size);
}

SourceProjectionGraph::SourceProjectionGraph(runtime::YnnExecutable executable, std::int32_t hidden_size,
                                             std::int32_t decoder_layers, std::int32_t maximum_position) noexcept
    : executable_(std::move(executable)),
      hidden_size_(hidden_size),
      decoder_layers_(decoder_layers),
      maximum_position_(maximum_position) {}

Result<SourceProjectionGraph> SourceProjectionGraph::create(const Package& package) {
    const auto& manifest = package.manifest();
    const auto& architecture = manifest.architecture;
    const auto graph_flags =
        manifest.weights.encoding == model::WeightEncoding::BF16 ? YNN_FLAG_NO_EXCESS_PRECISION : 0;
    const auto external_value_count =
        SOURCE_OUTPUT_BASE_ID + static_cast<std::uint32_t>(2 * architecture.decoder_layers);
    auto graph = runtime::YnnGraph::create(external_value_count, graph_flags);
    if (!graph) return std::unexpected(std::move(graph.error()));

    const std::array<std::size_t, 3> memory_shape = {
        1,
        0,
        static_cast<std::size_t>(architecture.hidden_size),
    };
    std::uint32_t memory_id = SOURCE_MEMORY_ID;
    auto status = runtime::check_ynn_status(
        ynn_define_tensor(graph->get(), ynn_type_fp32, memory_shape.size(), memory_shape.data(), nullptr,
                          YNN_VALUE_FLAG_EXTERNAL_INPUT, &memory_id),
        "define source projection input");
    if (!status) return std::unexpected(std::move(status.error()));

    TransformerBuilder builder(graph->get(), package.weights(), architecture.hidden_size,
                               architecture.feed_forward_size, architecture.attention_heads,
                               architecture.layer_norm_epsilon, manifest.weights.encoding);
    for (std::int32_t layer = 0; layer < architecture.decoder_layers; ++layer) {
        std::uint32_t key_id = source_key_id(SOURCE_OUTPUT_BASE_ID, layer);
        std::uint32_t value_id = source_value_id(SOURCE_OUTPUT_BASE_ID, layer);
        status = runtime::check_ynn_status(ynn_define_tensor(graph->get(), ynn_type_fp32, memory_shape.size(), nullptr,
                                                             nullptr, YNN_VALUE_FLAG_EXTERNAL_OUTPUT, &key_id),
                                           "define projected source key");
        if (status) {
            status =
                runtime::check_ynn_status(ynn_define_tensor(graph->get(), ynn_type_fp32, memory_shape.size(), nullptr,
                                                            nullptr, YNN_VALUE_FLAG_EXTERNAL_OUTPUT, &value_id),
                                          "define projected source value");
        }
        if (!status) return std::unexpected(std::move(status.error()));

        const std::array projection_ids = {key_id, value_id};
        auto projected = builder.linear_split(memory_id, "decoder.layers." + std::to_string(layer) + ".src_attn.kv",
                                              architecture.hidden_size, architecture.hidden_size, projection_ids.size(),
                                              projection_ids);
        if (!projected) return std::unexpected(std::move(projected.error()));
    }

    auto executable = std::move(*graph).compile();
    if (!executable) return std::unexpected(std::move(executable.error()));
    return SourceProjectionGraph(std::move(*executable), architecture.hidden_size, architecture.decoder_layers,
                                 architecture.maximum_position);
}

Result<SourceKVCache> SourceProjectionGraph::run(std::span<const float> memory, GraphRunStats* stats) {
    const auto hidden_size = static_cast<std::size_t>(hidden_size_);
    if (memory.empty() || memory.size() % hidden_size != 0) {
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "source projection input has incompatible shape"});
    }
    const auto source_length = memory.size() / hidden_size;
    if (source_length > static_cast<std::size_t>(maximum_position_)) {
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "source projection input exceeds position limit"});
    }

    const auto prepare_started = stats ? Clock::now() : Clock::time_point{};
    const std::array<std::size_t, 3> memory_shape = {1, source_length, hidden_size};
    const auto layer_size = memory.size();
    std::vector<float> values(2 * static_cast<std::size_t>(decoder_layers_) * layer_size);
    if (stats) stats->prepare_ns += elapsed_since(prepare_started);

    const auto reshape_started = stats ? Clock::now() : Clock::time_point{};
    auto status = executable_.set_shape(SOURCE_MEMORY_ID, memory_shape);
    if (status) status = executable_.reshape();
    if (status && stats && stats->max_concurrency == 0) {
        auto concurrency = executable_.concurrency();
        if (concurrency) stats->max_concurrency = *concurrency;
    }
    if (stats) stats->reshape_ns += elapsed_since(reshape_started);

    const auto bind_started = stats ? Clock::now() : Clock::time_point{};
    if (status) status = executable_.bind(SOURCE_MEMORY_ID, const_cast<float*>(memory.data()));
    for (std::int32_t layer = 0; status && layer < decoder_layers_; ++layer) {
        const auto offset = 2 * static_cast<std::size_t>(layer) * layer_size;
        status = executable_.bind(source_key_id(SOURCE_OUTPUT_BASE_ID, layer), values.data() + offset);
        if (status) {
            status =
                executable_.bind(source_value_id(SOURCE_OUTPUT_BASE_ID, layer), values.data() + offset + layer_size);
        }
    }
    if (stats) stats->bind_ns += elapsed_since(bind_started);

    const auto invoke_started = stats ? Clock::now() : Clock::time_point{};
    if (status) status = executable_.invoke();
    if (stats) stats->invoke_ns += elapsed_since(invoke_started);
    if (!status) return std::unexpected(std::move(status.error()));
    return SourceKVCache{
        .values = std::move(values),
        .source_length = source_length,
        .hidden_size = hidden_size,
        .layer_count = static_cast<std::size_t>(decoder_layers_),
    };
}

IncrementalDecoderGraph::IncrementalDecoderGraph(runtime::YnnExecutable executable, std::int32_t hidden_size,
                                                 std::int32_t decoder_layers, std::int32_t maximum_position) noexcept
    : executable_(std::move(executable)),
      hidden_size_(hidden_size),
      decoder_layers_(decoder_layers),
      maximum_position_(maximum_position),
      output_(static_cast<std::size_t>(hidden_size)) {}

Result<IncrementalDecoderGraph> IncrementalDecoderGraph::create(const Package& package) {
    const auto& manifest = package.manifest();
    const auto& architecture = manifest.architecture;
    const auto graph_flags =
        manifest.weights.encoding == model::WeightEncoding::BF16 ? YNN_FLAG_NO_EXCESS_PRECISION : 0;
    const auto self_base = incremental_self_base(architecture.decoder_layers);
    const auto current_base = incremental_current_base(architecture.decoder_layers);
    const auto external_value_count = current_base + static_cast<std::uint32_t>(2 * architecture.decoder_layers);
    auto graph = runtime::YnnGraph::create(external_value_count, graph_flags);
    if (!graph) return std::unexpected(std::move(graph.error()));

    const std::array<std::size_t, 3> token_shape = {
        1,
        1,
        static_cast<std::size_t>(architecture.hidden_size),
    };
    const std::array<std::size_t, 3> sequence_shape = {
        1,
        0,
        static_cast<std::size_t>(architecture.hidden_size),
    };
    constexpr std::array<std::size_t, 4> MASK_SHAPE = {1, 1, 1, 0};
    std::uint32_t input_id = INCREMENTAL_INPUT_ID;
    std::uint32_t mask_id = INCREMENTAL_MASK_ID;
    std::uint32_t output_id = INCREMENTAL_OUTPUT_ID;
    auto status =
        runtime::check_ynn_status(ynn_define_tensor(graph->get(), ynn_type_fp32, token_shape.size(), token_shape.data(),
                                                    nullptr, YNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id),
                                  "define incremental decoder input");
    if (status) {
        status = runtime::check_ynn_status(
            ynn_define_tensor(graph->get(), ynn_type_fp32, MASK_SHAPE.size(), MASK_SHAPE.data(), nullptr,
                              YNN_VALUE_FLAG_EXTERNAL_INPUT, &mask_id),
            "define incremental decoder mask");
    }
    if (status) {
        status = runtime::check_ynn_status(ynn_define_tensor(graph->get(), ynn_type_fp32, token_shape.size(), nullptr,
                                                             nullptr, YNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id),
                                           "define incremental decoder output");
    }
    if (!status) return std::unexpected(std::move(status.error()));

    for (std::int32_t layer = 0; layer < architecture.decoder_layers; ++layer) {
        for (const auto base : {INCREMENTAL_SOURCE_BASE_ID, self_base}) {
            std::uint32_t key_id = source_key_id(base, layer);
            std::uint32_t value_id = source_value_id(base, layer);
            status = runtime::check_ynn_status(
                ynn_define_tensor(graph->get(), ynn_type_fp32, sequence_shape.size(), sequence_shape.data(), nullptr,
                                  YNN_VALUE_FLAG_EXTERNAL_INPUT, &key_id),
                "define incremental key input");
            if (status) {
                status = runtime::check_ynn_status(
                    ynn_define_tensor(graph->get(), ynn_type_fp32, sequence_shape.size(), sequence_shape.data(),
                                      nullptr, YNN_VALUE_FLAG_EXTERNAL_INPUT, &value_id),
                    "define incremental value input");
            }
            if (!status) return std::unexpected(std::move(status.error()));
        }

        std::uint32_t current_key_id = source_key_id(current_base, layer);
        std::uint32_t current_value_id = source_value_id(current_base, layer);
        status = runtime::check_ynn_status(ynn_define_tensor(graph->get(), ynn_type_fp32, token_shape.size(), nullptr,
                                                             nullptr, YNN_VALUE_FLAG_EXTERNAL_OUTPUT, &current_key_id),
                                           "define current self-attention key");
        if (status) {
            status =
                runtime::check_ynn_status(ynn_define_tensor(graph->get(), ynn_type_fp32, token_shape.size(), nullptr,
                                                            nullptr, YNN_VALUE_FLAG_EXTERNAL_OUTPUT, &current_value_id),
                                          "define current self-attention value");
        }
        if (!status) return std::unexpected(std::move(status.error()));
    }

    TransformerBuilder builder(graph->get(), package.weights(), architecture.hidden_size,
                               architecture.feed_forward_size, architecture.attention_heads,
                               architecture.layer_norm_epsilon, manifest.weights.encoding);
    std::uint32_t hidden_id = input_id;
    for (std::int32_t layer = 0; layer < architecture.decoder_layers; ++layer) {
        const auto prefix = "decoder.layers." + std::to_string(layer);
        auto normalized_self_attention_id = builder.layer_norm(hidden_id, prefix + ".sublayer.0.norm");
        if (!normalized_self_attention_id) return std::unexpected(std::move(normalized_self_attention_id.error()));
        const std::array projection_ids = {YNN_INVALID_VALUE_ID, source_key_id(current_base, layer),
                                           source_value_id(current_base, layer)};
        auto projections =
            builder.linear_split(*normalized_self_attention_id, prefix + ".self_attn.qkv", architecture.hidden_size,
                                 architecture.hidden_size, projection_ids.size(), projection_ids);
        if (!projections) return std::unexpected(std::move(projections.error()));
        const auto query_id = (*projections)[0];
        const auto current_key_id = (*projections)[1];
        const auto current_value_id = (*projections)[2];

        const std::array key_inputs = {source_key_id(self_base, layer), current_key_id};
        const std::array value_inputs = {source_value_id(self_base, layer), current_value_id};
        std::uint32_t updated_key_id = YNN_INVALID_VALUE_ID;
        std::uint32_t updated_value_id = YNN_INVALID_VALUE_ID;
        status = runtime::check_ynn_status(
            ynn_define_concatenate(graph->get(), 1, key_inputs.size(), key_inputs.data(), &updated_key_id, 0),
            "append self-attention key");
        if (status) {
            status = runtime::check_ynn_status(
                ynn_define_concatenate(graph->get(), 1, value_inputs.size(), value_inputs.data(), &updated_value_id, 0),
                "append self-attention value");
        }
        if (!status) return std::unexpected(std::move(status.error()));

        auto self_attention_id = builder.attention_decode_one_from_projections(
            query_id, updated_key_id, updated_value_id, mask_id, prefix + ".self_attn", YNN_INVALID_VALUE_ID);
        if (!self_attention_id) return std::unexpected(std::move(self_attention_id.error()));

        std::uint32_t self_residual_id = YNN_INVALID_VALUE_ID;
        status = runtime::check_ynn_status(
            ynn_define_binary(graph->get(), ynn_binary_add, hidden_id, *self_attention_id, &self_residual_id, 0),
            "define incremental self-attention residual");
        if (!status) return std::unexpected(std::move(status.error()));

        auto normalized_source_attention_id = builder.layer_norm(self_residual_id, prefix + ".sublayer.1.norm");
        if (!normalized_source_attention_id) return std::unexpected(std::move(normalized_source_attention_id.error()));
        auto source_query_id = builder.linear(*normalized_source_attention_id, prefix + ".src_attn.q",
                                              architecture.hidden_size, architecture.hidden_size);
        if (!source_query_id) return std::unexpected(std::move(source_query_id.error()));
        auto source_attention_id = builder.attention_decode_one_from_projections(
            *source_query_id, source_key_id(INCREMENTAL_SOURCE_BASE_ID, layer),
            source_value_id(INCREMENTAL_SOURCE_BASE_ID, layer), YNN_INVALID_VALUE_ID, prefix + ".src_attn");
        if (!source_attention_id) return std::unexpected(std::move(source_attention_id.error()));

        std::uint32_t source_residual_id = YNN_INVALID_VALUE_ID;
        status = runtime::check_ynn_status(ynn_define_binary(graph->get(), ynn_binary_add, self_residual_id,
                                                             *source_attention_id, &source_residual_id, 0),
                                           "define incremental source-attention residual");
        if (!status) return std::unexpected(std::move(status.error()));

        auto normalized_feed_forward_id = builder.layer_norm(source_residual_id, prefix + ".sublayer.2.norm");
        if (!normalized_feed_forward_id) return std::unexpected(std::move(normalized_feed_forward_id.error()));
        auto feed_forward_id = builder.feed_forward(*normalized_feed_forward_id, prefix + ".feed_forward");
        if (!feed_forward_id) return std::unexpected(std::move(feed_forward_id.error()));

        std::uint32_t residual_id = YNN_INVALID_VALUE_ID;
        status = runtime::check_ynn_status(
            ynn_define_binary(graph->get(), ynn_binary_add, source_residual_id, *feed_forward_id, &residual_id, 0),
            "define incremental feed-forward residual");
        if (!status) return std::unexpected(std::move(status.error()));
        hidden_id = residual_id;
    }

    auto result_id = builder.layer_norm(hidden_id, "decoder.norm", output_id);
    if (!result_id) return std::unexpected(std::move(result_id.error()));
    auto executable = std::move(*graph).compile();
    if (!executable) return std::unexpected(std::move(executable.error()));
    return IncrementalDecoderGraph(std::move(*executable), architecture.hidden_size, architecture.decoder_layers,
                                   architecture.maximum_position);
}

Result<std::span<const float>> IncrementalDecoderGraph::run(std::span<const float> embedding,
                                                            const SourceKVCache& source, SelfKVCache& self,
                                                            GraphRunStats* stats) {
    const auto hidden_size = static_cast<std::size_t>(hidden_size_);
    const auto layer_count = static_cast<std::size_t>(decoder_layers_);
    if (embedding.size() != hidden_size || source.source_length == 0 || source.hidden_size != hidden_size ||
        source.layer_count != layer_count ||
        source.values.size() != 2 * layer_count * source.source_length * hidden_size || self.length == 0 ||
        self.length >= self.capacity || self.hidden_size != hidden_size || self.layer_count != layer_count ||
        self.mask.size() != self.capacity || self.values.size() != 2 * layer_count * self.capacity * hidden_size) {
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "incremental decoder input has incompatible shape"});
    }
    if (self.length > static_cast<std::size_t>(maximum_position_) ||
        source.source_length > static_cast<std::size_t>(maximum_position_)) {
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "incremental decoder input exceeds position limit"});
    }

    const auto prepare_started = stats ? Clock::now() : Clock::time_point{};
    const auto self_base = incremental_self_base(decoder_layers_);
    const auto current_base = incremental_current_base(decoder_layers_);
    const auto updated_length = self.length + 1;
    const std::array<std::size_t, 3> token_shape = {1, 1, hidden_size};
    const std::array<std::size_t, 3> source_shape = {1, source.source_length, hidden_size};
    const std::array<std::size_t, 3> self_shape = {1, self.length, hidden_size};
    const std::array<std::size_t, 4> mask_shape = {1, 1, 1, updated_length};
    if (stats) stats->prepare_ns += elapsed_since(prepare_started);

    const auto reshape_started = stats ? Clock::now() : Clock::time_point{};
    auto status = executable_.set_shape(INCREMENTAL_INPUT_ID, token_shape);
    if (status) status = executable_.set_shape(INCREMENTAL_MASK_ID, mask_shape);
    for (std::int32_t layer = 0; status && layer < decoder_layers_; ++layer) {
        status = executable_.set_shape(source_key_id(INCREMENTAL_SOURCE_BASE_ID, layer), source_shape);
        if (status) status = executable_.set_shape(source_value_id(INCREMENTAL_SOURCE_BASE_ID, layer), source_shape);
        if (status) status = executable_.set_shape(source_key_id(self_base, layer), self_shape);
        if (status) status = executable_.set_shape(source_value_id(self_base, layer), self_shape);
    }
    if (status) status = executable_.reshape();
    if (status && stats && stats->max_concurrency == 0) {
        auto concurrency = executable_.concurrency();
        if (concurrency) stats->max_concurrency = *concurrency;
    }
    if (stats) stats->reshape_ns += elapsed_since(reshape_started);

    const auto bind_started = stats ? Clock::now() : Clock::time_point{};
    if (status) status = executable_.bind(INCREMENTAL_INPUT_ID, const_cast<float*>(embedding.data()));
    if (status) status = executable_.bind(INCREMENTAL_MASK_ID, self.mask.data());
    if (status) status = executable_.bind(INCREMENTAL_OUTPUT_ID, output_.data());
    for (std::int32_t layer = 0; status && layer < decoder_layers_; ++layer) {
        const auto layer_index = static_cast<std::size_t>(layer);
        const auto source_key = source.key(layer_index);
        const auto source_value = source.value(layer_index);
        const auto self_key = self.key(layer_index);
        const auto self_value = self.value(layer_index);
        status =
            executable_.bind(source_key_id(INCREMENTAL_SOURCE_BASE_ID, layer), const_cast<float*>(source_key.data()));
        if (status) {
            status = executable_.bind(source_value_id(INCREMENTAL_SOURCE_BASE_ID, layer),
                                      const_cast<float*>(source_value.data()));
        }
        if (status) status = executable_.bind(source_key_id(self_base, layer), const_cast<float*>(self_key.data()));
        if (status) {
            status = executable_.bind(source_value_id(self_base, layer), const_cast<float*>(self_value.data()));
        }
        const auto current_key = self.key_slot(layer_index, self.length);
        const auto current_value = self.value_slot(layer_index, self.length);
        if (status) status = executable_.bind(source_key_id(current_base, layer), current_key.data());
        if (status) {
            status = executable_.bind(source_value_id(current_base, layer), current_value.data());
        }
    }
    if (stats) stats->bind_ns += elapsed_since(bind_started);

    const auto invoke_started = stats ? Clock::now() : Clock::time_point{};
    if (status) status = executable_.invoke();
    if (stats) stats->invoke_ns += elapsed_since(invoke_started);
    if (!status) return std::unexpected(std::move(status.error()));
    self.length = updated_length;
    return std::span<const float>(output_);
}

DecoderGraph::DecoderGraph(runtime::YnnExecutable executable, std::int32_t hidden_size, std::int32_t decoder_layers,
                           std::int32_t maximum_position) noexcept
    : executable_(std::move(executable)),
      hidden_size_(hidden_size),
      decoder_layers_(decoder_layers),
      maximum_position_(maximum_position) {}

Result<DecoderGraph> DecoderGraph::create(const Package& package) {
    const auto& manifest = package.manifest();
    const auto& architecture = manifest.architecture;
    const auto graph_flags =
        manifest.weights.encoding == model::WeightEncoding::BF16 ? YNN_FLAG_NO_EXCESS_PRECISION : 0;
    const auto external_value_count =
        DECODER_SOURCE_BASE_ID + static_cast<std::uint32_t>(2 * architecture.decoder_layers);
    auto graph = runtime::YnnGraph::create(external_value_count, graph_flags);
    if (!graph) return std::unexpected(std::move(graph.error()));

    const std::array<std::size_t, 3> decoder_shape = {
        0,
        0,
        static_cast<std::size_t>(architecture.hidden_size),
    };
    const std::array<std::size_t, 3> source_shape = {
        1,
        0,
        static_cast<std::size_t>(architecture.hidden_size),
    };
    constexpr std::array<std::size_t, 4> MASK_SHAPE = {1, 1, 0, 0};
    std::uint32_t input_id = DECODER_INPUT_ID;
    std::uint32_t mask_id = CAUSAL_MASK_ID;
    std::uint32_t output_id = DECODER_OUTPUT_ID;
    auto status = runtime::check_ynn_status(
        ynn_define_tensor(graph->get(), ynn_type_fp32, decoder_shape.size(), decoder_shape.data(), nullptr,
                          YNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id),
        "define decoder input");
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

    for (std::int32_t layer = 0; layer < architecture.decoder_layers; ++layer) {
        std::uint32_t key_id = source_key_id(DECODER_SOURCE_BASE_ID, layer);
        std::uint32_t value_id = source_value_id(DECODER_SOURCE_BASE_ID, layer);
        status = runtime::check_ynn_status(
            ynn_define_tensor(graph->get(), ynn_type_fp32, source_shape.size(), source_shape.data(), nullptr,
                              YNN_VALUE_FLAG_EXTERNAL_INPUT, &key_id),
            "define projected source key input");
        if (status) {
            status = runtime::check_ynn_status(
                ynn_define_tensor(graph->get(), ynn_type_fp32, source_shape.size(), source_shape.data(), nullptr,
                                  YNN_VALUE_FLAG_EXTERNAL_INPUT, &value_id),
                "define projected source value input");
        }
        if (!status) return std::unexpected(std::move(status.error()));
    }

    TransformerBuilder builder(graph->get(), package.weights(), architecture.hidden_size,
                               architecture.feed_forward_size, architecture.attention_heads,
                               architecture.layer_norm_epsilon, manifest.weights.encoding);
    std::uint32_t hidden_id = input_id;
    for (std::int32_t layer = 0; layer < architecture.decoder_layers; ++layer) {
        const auto prefix = "decoder.layers." + std::to_string(layer);
        auto normalized_self_attention_id = builder.layer_norm(hidden_id, prefix + ".sublayer.0.norm");
        if (!normalized_self_attention_id) return std::unexpected(std::move(normalized_self_attention_id.error()));
        auto self_attention_id = builder.self_attention(*normalized_self_attention_id, mask_id, prefix + ".self_attn");
        if (!self_attention_id) return std::unexpected(std::move(self_attention_id.error()));

        std::uint32_t self_residual_id = YNN_INVALID_VALUE_ID;
        status = runtime::check_ynn_status(
            ynn_define_binary(graph->get(), ynn_binary_add, hidden_id, *self_attention_id, &self_residual_id, 0),
            "define decoder self-attention residual");
        if (!status) return std::unexpected(std::move(status.error()));

        auto normalized_source_attention_id = builder.layer_norm(self_residual_id, prefix + ".sublayer.1.norm");
        if (!normalized_source_attention_id) return std::unexpected(std::move(normalized_source_attention_id.error()));
        auto source_attention_id = builder.attention_with_projected_key_value(
            *normalized_source_attention_id, source_key_id(DECODER_SOURCE_BASE_ID, layer),
            source_value_id(DECODER_SOURCE_BASE_ID, layer), YNN_INVALID_VALUE_ID, prefix + ".src_attn");
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
    return DecoderGraph(std::move(*executable), architecture.hidden_size, architecture.decoder_layers,
                        architecture.maximum_position);
}

Result<std::vector<float>> DecoderGraph::run(std::span<const float> embeddings, std::size_t batch_size,
                                             const SourceKVCache& source, GraphRunStats* stats) {
    const auto hidden_size = static_cast<std::size_t>(hidden_size_);
    if (batch_size == 0 || embeddings.empty() || embeddings.size() % (batch_size * hidden_size) != 0 ||
        source.source_length == 0 || source.hidden_size != hidden_size ||
        source.layer_count != static_cast<std::size_t>(decoder_layers_) ||
        source.values.size() != 2 * source.layer_count * source.source_length * source.hidden_size) {
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "decoder input has incompatible shape"});
    }
    const auto target_length = embeddings.size() / (batch_size * hidden_size);
    const auto source_length = source.source_length;
    if (target_length > static_cast<std::size_t>(maximum_position_) ||
        source_length > static_cast<std::size_t>(maximum_position_)) {
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "decoder input exceeds the position limit"});
    }

    const auto prepare_started = stats ? Clock::now() : Clock::time_point{};
    const std::array<std::size_t, 3> decoder_shape = {batch_size, target_length, hidden_size};
    const std::array<std::size_t, 3> source_shape = {1, source_length, hidden_size};
    const std::array<std::size_t, 4> mask_shape = {1, 1, target_length, target_length};
    std::vector<float> causal_mask(target_length * target_length);
    for (std::size_t row = 0; row < target_length; ++row) {
        std::fill(causal_mask.begin() + static_cast<std::ptrdiff_t>(row * target_length + row + 1),
                  causal_mask.begin() + static_cast<std::ptrdiff_t>((row + 1) * target_length), -1.0e9F);
    }
    std::vector<float> output(embeddings.size());
    if (stats) stats->prepare_ns += elapsed_since(prepare_started);

    const auto reshape_started = stats ? Clock::now() : Clock::time_point{};
    auto status = executable_.set_shape(DECODER_INPUT_ID, decoder_shape);
    if (status) status = executable_.set_shape(CAUSAL_MASK_ID, mask_shape);
    for (std::int32_t layer = 0; status && layer < decoder_layers_; ++layer) {
        status = executable_.set_shape(source_key_id(DECODER_SOURCE_BASE_ID, layer), source_shape);
        if (status) status = executable_.set_shape(source_value_id(DECODER_SOURCE_BASE_ID, layer), source_shape);
    }
    if (status) status = executable_.reshape();
    if (status && stats && stats->max_concurrency == 0) {
        auto concurrency = executable_.concurrency();
        if (concurrency) stats->max_concurrency = *concurrency;
    }
    if (stats) stats->reshape_ns += elapsed_since(reshape_started);

    const auto bind_started = stats ? Clock::now() : Clock::time_point{};
    if (status) status = executable_.bind(DECODER_INPUT_ID, const_cast<float*>(embeddings.data()));
    if (status) status = executable_.bind(CAUSAL_MASK_ID, causal_mask.data());
    if (status) status = executable_.bind(DECODER_OUTPUT_ID, output.data());
    for (std::int32_t layer = 0; status && layer < decoder_layers_; ++layer) {
        const auto key = source.key(static_cast<std::size_t>(layer));
        const auto value = source.value(static_cast<std::size_t>(layer));
        status = executable_.bind(source_key_id(DECODER_SOURCE_BASE_ID, layer), const_cast<float*>(key.data()));
        if (status) {
            status = executable_.bind(source_value_id(DECODER_SOURCE_BASE_ID, layer), const_cast<float*>(value.data()));
        }
    }
    if (stats) stats->bind_ns += elapsed_since(bind_started);

    const auto invoke_started = stats ? Clock::now() : Clock::time_point{};
    if (status) status = executable_.invoke();
    if (stats) stats->invoke_ns += elapsed_since(invoke_started);
    if (!status) return std::unexpected(std::move(status.error()));
    return output;
}

GeneratorGraph::GeneratorGraph(runtime::YnnExecutable executable, std::int32_t hidden_size,
                               std::int32_t vocabulary_size, std::int32_t maximum_batch_size) noexcept
    : executable_(std::move(executable)),
      hidden_size_(hidden_size),
      vocabulary_size_(vocabulary_size),
      output_(static_cast<std::size_t>(maximum_batch_size) * static_cast<std::size_t>(vocabulary_size)) {}

Result<GeneratorGraph> GeneratorGraph::create(const Package& package) {
    const auto& manifest = package.manifest();
    const auto& architecture = manifest.architecture;
    const auto graph_flags =
        manifest.weights.encoding == model::WeightEncoding::BF16 ? YNN_FLAG_NO_EXCESS_PRECISION : 0;
    auto graph = runtime::YnnGraph::create(2, graph_flags);
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
                               architecture.layer_norm_epsilon, manifest.weights.encoding);
    auto logits_id =
        manifest.weights.encoding == model::WeightEncoding::F32
            ? builder.tied_projection(input_id, "tgt_embed.0.lut.weight", "generator.proj.bias",
                                      architecture.hidden_size, architecture.target_vocabulary_size, output_id)
            : builder.linear(input_id, "generator.proj.weight", "generator.proj.bias", architecture.hidden_size,
                             architecture.target_vocabulary_size, output_id);
    if (!logits_id) return std::unexpected(std::move(logits_id.error()));

    auto executable = std::move(*graph).compile();
    if (!executable) return std::unexpected(std::move(executable.error()));
    return GeneratorGraph(std::move(*executable), architecture.hidden_size, architecture.target_vocabulary_size,
                          manifest.limits.maximum_beam_size);
}

Result<std::span<float>> GeneratorGraph::run(std::span<const float> hidden_states, GraphRunStats* stats) {
    const auto hidden_size = static_cast<std::size_t>(hidden_size_);
    if (hidden_states.empty() || hidden_states.size() % hidden_size != 0) {
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "generator input has incompatible shape"});
    }
    const auto prepare_started = stats ? Clock::now() : Clock::time_point{};
    const auto batch_size = hidden_states.size() / hidden_size;
    const std::array<std::size_t, 3> input_shape = {batch_size, 1, hidden_size};
    const auto output_size = batch_size * static_cast<std::size_t>(vocabulary_size_);
    if (output_size > output_.size()) {
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "generator batch exceeds configured capacity"});
    }
    if (stats) stats->prepare_ns += elapsed_since(prepare_started);

    const auto reshape_started = stats ? Clock::now() : Clock::time_point{};
    auto status = executable_.set_shape(GENERATOR_INPUT_ID, input_shape);
    if (status) status = executable_.reshape();
    if (status && stats && stats->max_concurrency == 0) {
        auto concurrency = executable_.concurrency();
        if (concurrency) stats->max_concurrency = *concurrency;
    }
    if (stats) stats->reshape_ns += elapsed_since(reshape_started);

    const auto bind_started = stats ? Clock::now() : Clock::time_point{};
    if (status) status = executable_.bind(GENERATOR_INPUT_ID, const_cast<float*>(hidden_states.data()));
    if (status) status = executable_.bind(GENERATOR_OUTPUT_ID, output_.data());
    if (stats) stats->bind_ns += elapsed_since(bind_started);

    const auto invoke_started = stats ? Clock::now() : Clock::time_point{};
    if (status) status = executable_.invoke();
    if (stats) stats->invoke_ns += elapsed_since(invoke_started);
    if (!status) return std::unexpected(std::move(status.error()));
    return std::span<float>(output_.data(), output_size);
}

Decoder::Decoder(EmbeddingGraph embedding, SourceProjectionGraph source_projection, IncrementalDecoderGraph incremental,
                 DecoderGraph graph, GeneratorGraph generator, std::int32_t hidden_size, std::int32_t decoder_layers,
                 std::int32_t maximum_batch_size) noexcept
    : embedding_(std::move(embedding)),
      source_projection_(std::move(source_projection)),
      incremental_(std::move(incremental)),
      graph_(std::move(graph)),
      generator_(std::move(generator)),
      hidden_size_(hidden_size),
      decoder_layers_(decoder_layers),
      last_hidden_(static_cast<std::size_t>(hidden_size) * static_cast<std::size_t>(maximum_batch_size)) {}

Result<Decoder> Decoder::create(const Package& package) {
    const auto& manifest = package.manifest();
    const auto& architecture = manifest.architecture;
    auto embedding =
        EmbeddingGraph::create(package.weights(), "tgt_embed.0.lut.weight", architecture.target_vocabulary_size,
                               architecture.hidden_size, manifest.weights.encoding, architecture.maximum_position);
    if (!embedding) return std::unexpected(std::move(embedding.error()));
    auto source_projection = SourceProjectionGraph::create(package);
    if (!source_projection) return std::unexpected(std::move(source_projection.error()));
    auto incremental = IncrementalDecoderGraph::create(package);
    if (!incremental) return std::unexpected(std::move(incremental.error()));
    auto graph = DecoderGraph::create(package);
    if (!graph) return std::unexpected(std::move(graph.error()));
    auto generator = GeneratorGraph::create(package);
    if (!generator) return std::unexpected(std::move(generator.error()));
    return Decoder(std::move(*embedding), std::move(*source_projection), std::move(*incremental), std::move(*graph),
                   std::move(*generator), architecture.hidden_size, architecture.decoder_layers,
                   manifest.limits.maximum_beam_size);
}

Result<SourceKVCache> Decoder::prepare_source(std::span<const float> memory, InferenceStats* stats) {
    return source_projection_.run(memory, stats ? &stats->source_projection_graph : nullptr);
}

SelfKVCache Decoder::create_self_cache(std::size_t maximum_steps) const {
    const auto hidden_size = static_cast<std::size_t>(hidden_size_);
    const auto layer_count = static_cast<std::size_t>(decoder_layers_);
    const auto capacity = maximum_steps + 1;
    std::vector<float> mask(capacity);
    mask.front() = -1.0e9F;
    return SelfKVCache{
        .values = std::vector<float>(2 * layer_count * capacity * hidden_size),
        .mask = std::move(mask),
        .length = 1,
        .capacity = capacity,
        .hidden_size = hidden_size,
        .layer_count = layer_count,
    };
}

Result<std::span<const float>> Decoder::next_greedy(const SourceKVCache& source, std::int32_t token_id,
                                                    std::size_t position, SelfKVCache& self, InferenceStats* stats) {
    if (self.length != position + 1) {
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "greedy decoder cache position is inconsistent"});
    }
    const std::array token_ids = {token_id};
    auto embedding =
        measure(stats ? &stats->target_embedding_ns : nullptr, [&] { return embedding_.run(token_ids, 1, position); });
    if (!embedding) return std::unexpected(std::move(embedding.error()));
    auto hidden = incremental_.run(*embedding, source, self, stats ? &stats->decoder_graph : nullptr);
    if (!hidden) return std::unexpected(std::move(hidden.error()));
    return generator_.run(*hidden, stats ? &stats->generator_graph : nullptr);
}

Result<std::span<const float>> Decoder::next(const SourceKVCache& source, std::span<const std::int32_t> token_ids,
                                             std::size_t batch_size, InferenceStats* stats) {
    if (batch_size == 0 || token_ids.empty() || token_ids.size() % batch_size != 0) {
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "decoder token input has incompatible shape"});
    }
    auto embeddings =
        measure(stats ? &stats->target_embedding_ns : nullptr, [&] { return embedding_.run(token_ids, batch_size); });
    if (!embeddings) return std::unexpected(std::move(embeddings.error()));
    auto hidden_states = graph_.run(*embeddings, batch_size, source, stats ? &stats->decoder_graph : nullptr);
    if (!hidden_states) return std::unexpected(std::move(hidden_states.error()));

    const auto last_hidden_started = stats ? Clock::now() : Clock::time_point{};
    const auto target_length = token_ids.size() / batch_size;
    const auto hidden_size = static_cast<std::size_t>(hidden_size_);
    if (batch_size * hidden_size > last_hidden_.size()) {
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "decoder batch exceeds configured capacity"});
    }
    auto last_hidden = std::span<float>(last_hidden_).first(batch_size * hidden_size);
    for (std::size_t batch = 0; batch < batch_size; ++batch) {
        const auto source_offset = (batch * target_length + target_length - 1) * hidden_size;
        std::copy_n(hidden_states->begin() + static_cast<std::ptrdiff_t>(source_offset), hidden_size,
                    last_hidden.begin() + static_cast<std::ptrdiff_t>(batch * hidden_size));
    }
    if (stats) stats->last_hidden_ns += elapsed_since(last_hidden_started);
    auto logits = generator_.run(last_hidden, stats ? &stats->generator_graph : nullptr);
    if (!logits) return std::unexpected(std::move(logits.error()));
    const auto vocabulary_size = static_cast<std::size_t>(generator_.vocabulary_size());
    for (std::size_t batch = 0; batch < batch_size; ++batch) {
        auto row = logits->subspan(batch * vocabulary_size, vocabulary_size);
        const auto maximum = *std::ranges::max_element(row);
        float sum = 0.0F;
        for (const auto value : row) sum += std::exp(value - maximum);
        const auto normalization = maximum + std::log(sum);
        for (auto& value : row) value -= normalization;
    }
    return std::span<const float>(*logits);
}

} // namespace kidi::rtg