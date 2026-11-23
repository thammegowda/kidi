#pragma once

#include <cuda_runtime_api.h>

#include "cuda/launch.h"

namespace kidi::cuda {

/// Launches specialized fused operations. Shapes are validated by `infer`.
auto fused_operation(const Launch& plan, Op operation, cudaStream_t stream) -> void;

/// Workspace bytes for a routed expert projection that groups rows by expert, or 0 when the per-row GEMV runs.
auto expert_workspace_bytes(std::int64_t tokens, std::int64_t slots, std::int64_t experts, std::int64_t rows,
                            bool gated) -> std::size_t;

/// Routed expert projection; `workspace` holds `expert_workspace_bytes` bytes, or is null for the GEMV.
auto expert_linear(const Launch& plan, void* workspace, cudaStream_t stream) -> void;

} // namespace kidi::cuda
