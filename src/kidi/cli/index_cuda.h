#pragma once

#include "kidi/core/hnsw.h"

namespace kidi::cli {
auto cuda_index_available() -> Result<bool>;
auto build_cuda_index(const tensor::Tensor& vectors, core::HnswBuildOptions options, std::size_t threads)
    -> Result<core::HnswIndex>;
} // namespace kidi::cli
