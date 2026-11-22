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
    bool has_mask = false;
    std::int64_t batches = 1;
    std::int64_t queries = 1;
    std::int64_t heads = 1;
    std::int64_t key_heads = 1;
    std::int64_t key_offset = 0;
    std::int64_t mask_window = 0;
    std::int64_t keys = 0;
    int head_dim = 0;
    float scale = 0.F;
};

auto flash_attention_bf16(const FlashAttentionArgs& args, cudaStream_t stream) -> bool;
auto flash_attention_f32(const FlashAttentionArgs& args, cudaStream_t stream) -> bool;

} // namespace kidi::cuda
