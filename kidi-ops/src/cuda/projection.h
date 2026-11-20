#pragma once

#include <cuda_runtime_api.h>

#include "cuda/launch.h"

namespace kidi::cuda {

auto prepare_projection_weights(const Launch& plan, int bits, void* expanded, void* sums, cudaStream_t stream) -> void;

auto quantize_rows(const Launch& plan, std::int64_t rows, void* scratch, void* scales, void* zeros, cudaStream_t stream)
    -> void;

auto execute_matrix(const Launch& plan, int device, bool optimized, void* workspace, int slices, cudaStream_t stream)
    -> void;

auto floating_projection(const Launch& plan, int device, bool reference_projection, void* shared_workspace,
                         void* scratch, int dense_slices, cudaStream_t stream) -> void;

auto quantized_projection(const Launch& plan, int device, bool reference_projection, bool vector_projection,
                          bool tensor_projection, bool fused_w8a8, void* scratch, void* scales, void* zeros,
                          const void* prepared_expanded, const void* prepared_sums, void* accumulators,
                          void* shared_workspace, int dense_slices, cudaStream_t stream) -> void;

auto w8a16_projection(const Launch& plan, int device, bool reference_projection, void* shared_workspace, void* scratch,
                      int dense_slices, cudaStream_t stream) -> void;

} // namespace kidi::cuda
