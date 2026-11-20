#pragma once

#include <cuda_runtime_api.h>
#include <cstdint>

namespace kidi::cuda {

auto w8a8_gemm_slices(int rows, int columns, int width, int multiprocessors) -> int;

auto w8a8_gemm(int rows, int columns, int width, const std::int8_t* input, const std::int8_t* weights,
               const float* row_scales, const int* row_zeros, const int* weight_sums, const float* col_scales,
               const float* bias, float output_scale, void* output, bool bfloat_out, cudaStream_t stream,
               void* workspace, int slices) -> void;

auto int8_gemm(int rows, int columns, int width, const std::int8_t* input, const std::int8_t* weights,
               std::int32_t* output, cudaStream_t stream) -> void;

} // namespace kidi::cuda