#pragma once

#include <cstddef>
#include <cstdint>

namespace kidi::rtg {

struct GraphRunStats {
    std::int32_t max_concurrency = 0;
    std::uint64_t prepare_ns = 0;
    std::uint64_t reshape_ns = 0;
    std::uint64_t bind_ns = 0;
    std::uint64_t invoke_ns = 0;
};

struct InferenceStats {
    std::uint64_t package_load_ns = 0;
    std::uint64_t graph_compile_ns = 0;
    std::uint64_t source_tokenize_ns = 0;
    std::uint64_t encoder_ns = 0;
    std::uint64_t source_embedding_ns = 0;
    GraphRunStats encoder_graph;
    std::uint64_t source_projection_ns = 0;
    GraphRunStats source_projection_graph;
    std::uint64_t decoder_ns = 0;
    std::uint64_t target_embedding_ns = 0;
    GraphRunStats decoder_graph;
    std::uint64_t last_hidden_ns = 0;
    GraphRunStats generator_graph;
    std::uint64_t host_search_ns = 0;
    std::uint64_t target_decode_ns = 0;
    std::uint64_t translate_ns = 0;
    std::size_t decoder_steps = 0;
};

} // namespace kidi::rtg