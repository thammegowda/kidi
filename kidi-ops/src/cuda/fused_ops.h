#pragma once

#include <cuda_runtime_api.h>

#include "cuda/launch.h"

namespace kidi::cuda {

/// Launches specialized fused operations. Shapes are validated by `infer`.
auto fused_operation(const Launch& plan, Op operation, cudaStream_t stream) -> void;

} // namespace kidi::cuda
