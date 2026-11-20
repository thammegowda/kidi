#pragma once

#include <cuda_runtime_api.h>
#include <cstddef>
#include <cstdint>

#include "cuda/launch.h"

namespace kidi::cuda {

auto token_candidate_bytes() -> std::size_t;

auto pointwise(const Launch& plan, cudaStream_t stream) -> void;
auto gated_activation(const Launch& plan, cudaStream_t stream) -> void;
auto unfold_convolution(const Launch& plan, float* columns, std::size_t elements, cudaStream_t stream) -> void;
auto fold_convolution(const Launch& plan, cudaStream_t stream) -> void;
auto normalize(const Launch& plan, Op operation, cudaStream_t stream) -> void;
auto greedy_token(const Launch& plan, void* candidates, cudaStream_t stream) -> void;
auto layout(const Launch& plan, cudaStream_t stream) -> void;
auto scatter_updates(const Launch& plan, cudaStream_t stream) -> void;
auto embedding(const Launch& plan, cudaStream_t stream) -> void;
auto embedding_sum(const Launch& plan, cudaStream_t stream) -> void;
auto diffusion_candidates(const Launch& plan, cudaStream_t stream) -> void;
auto diffusion_update(const Launch& plan, float* scratch, cudaStream_t stream) -> void;

} // namespace kidi::cuda
