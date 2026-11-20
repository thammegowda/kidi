#pragma once

#include <cuda_runtime_api.h>

namespace kidi::cuda {

auto dense_gemm_slices(int rows, int columns, int width, int multiprocessors, bool bfloat) -> int;

auto launch_bfloat_gemm(int rows, int columns, int width, const void* input, const void* weights, void* output,
                        cudaStream_t stream, void* workspace, int slices) -> void;

auto launch_float_gemm(int rows, int columns, int width, const void* input, const void* weights, void* output,
                       cudaStream_t stream, bool pipelined, void* workspace, int slices) -> void;

} // namespace kidi::cuda