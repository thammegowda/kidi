#pragma once

#include <cuda_runtime_api.h>
#include <cstdint>

namespace kidi::cuda {

auto launch_int8_gemm(int rows, int columns, int width, const std::int8_t* input, const std::int8_t* weights,
                      std::int32_t* output, cudaStream_t stream) -> void;

} // namespace kidi::cuda