#pragma once

#include <cuda_runtime_api.h>
#include <cstddef>
#include <cstdint>

namespace kidi::cuda {

struct FlashView {
    const void* data = nullptr;
    void* const* binding = nullptr;
    std::size_t binding_offset = 0;
    int type = 0;
    int rank = 0;
    std::int64_t shape[4]{};
};

struct FlashAttentionArgs {
    FlashView query;
    FlashView key;
    FlashView value;
    FlashView mask;
    FlashView output;
    FlashView relative;
    FlashView sink;
    FlashView gate;
    /// INT8 key and value caches: F32 scales [rows, width / 32], one per 32 columns.
    FlashView key_scales;
    FlashView value_scales;
    bool has_mask = false;
    bool has_relative = false;
    std::int64_t batches = 1;
    std::int64_t queries = 1;
    std::int64_t heads = 1;
    std::int64_t key_heads = 1;
    std::int64_t key_offset = 0;
    std::int64_t mask_window = 0;
    std::int64_t keys = 0;
    /// Nonzero for ring caches: key `k` lives in row `k % ring_rows`, and `keys` bounds absolute key positions.
    std::int64_t ring_rows = 0;
    std::int64_t extent = 0;
    int head_dim = 0;
    float scale = 0.F;
    float log_alpha = 0.F;
    float log_floor = 0.F;
};

auto flash_attention_bf16(const FlashAttentionArgs& args, cudaStream_t stream) -> bool;
auto flash_attention_f32(const FlashAttentionArgs& args, cudaStream_t stream) -> bool;

} // namespace kidi::cuda
