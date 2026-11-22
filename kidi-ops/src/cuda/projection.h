#pragma once

#include <cuda_runtime_api.h>

#include "cuda/launch.h"

namespace kidi::cuda {

auto prepare_projection_weights(const Launch& plan, int bits, void* expanded, void* sums, cudaStream_t stream) -> void;

auto quantize_rows(const Launch& plan, std::int64_t rows, void* scratch, void* scales, void* zeros, cudaStream_t stream)
    -> void;

// Widest row the fused projection prologue keeps in registers (256 threads x values per thread).
inline constexpr int prologue_values_per_thread = 40;
inline constexpr std::int64_t prologue_max_width = 256 * prologue_values_per_thread;

// Applies an RMS-norm or GELU-gate prologue to the F32 input rows and quantizes them for a W8A8 projection.
auto prologue_quantize(const Launch& plan, ProjectionPrologue kind, float rms_epsilon, std::int64_t rows,
                       std::int64_t width, void* scratch, void* scales, void* zeros, cudaStream_t stream) -> void;

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
