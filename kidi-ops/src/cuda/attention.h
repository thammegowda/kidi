#pragma once

#include <cuda_runtime_api.h>
#include <cstdint>

#include "cuda/launch.h"

namespace kidi::cuda {

inline constexpr std::int64_t DECODE_ATTENTION_PARTITION_SIZE = 16;

using AttentionLauncher = void (*)(Launch, std::int64_t, std::int64_t, std::int64_t, cudaStream_t);

auto select_parallel_attention(const Descriptor& descriptor, std::int64_t keys, std::int64_t width, int device)
    -> AttentionLauncher;

auto decode_attention(const Launch& plan, float* partials, std::int64_t batches, std::int64_t heads,
                      std::int64_t key_heads, std::int64_t keys, std::int64_t width, int input_type,
                      cudaStream_t stream) -> void;

auto fallback_attention(const Launch& plan, float* scratch, std::int64_t batches, std::int64_t heads,
                        std::int64_t key_heads, std::int64_t queries, std::int64_t keys, cudaStream_t stream) -> void;

} // namespace kidi::cuda
