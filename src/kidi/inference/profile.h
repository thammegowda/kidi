#pragma once

#include <cstddef>
#include <cstdint>

namespace kidi::inference {

struct InferenceStats {
    std::uint64_t package_load_ns = 0;
    std::uint64_t graph_compile_ns = 0;
    std::uint64_t source_tokenize_ns = 0;
    std::uint64_t encoder_ns = 0;
    std::uint64_t source_embedding_ns = 0;
    std::uint64_t source_projection_ns = 0;
    std::uint64_t decoder_ns = 0;
    std::uint64_t target_embedding_ns = 0;
    std::uint64_t host_search_ns = 0;
    std::uint64_t target_decode_ns = 0;
    std::uint64_t translate_ns = 0;
    std::size_t decoder_steps = 0;
};

} // namespace kidi::inference