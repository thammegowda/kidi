#pragma once

#include <cuda_runtime_api.h>
#include <cstddef>

namespace kidi::cuda {

auto dense_gemm_slices(int rows, int columns, int width, int multiprocessors, bool bfloat) -> int;

auto w8a16_gemm_slices(int rows, int columns, int width, int group_size, int multiprocessors) -> int;

auto bfloat_gemm(int rows, int columns, int width, const void* input, const void* weights, void* output,
                 cudaStream_t stream, void* workspace, int slices) -> void;

auto w8a16_gemm(int rows, int columns, int width, int group_size, const void* input, const void* weights,
                const float* scales, void* output, cudaStream_t stream, void* workspace, int slices) -> void;

auto float_gemm(int rows, int columns, int width, const void* input, const void* weights, void* output,
                cudaStream_t stream, bool pipelined, void* workspace, int slices) -> void;

} // namespace kidi::cuda