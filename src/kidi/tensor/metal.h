#pragma once

#include <cstddef>

#include "kidi/core/error.h"

namespace kidi::tensor {

class Tensor;

struct MetalBufferView {
    void* handle;
    std::size_t offset_bytes;
};

struct MetalMemoryStats {
    std::size_t allocated_bytes = 0;
    std::size_t recommended_working_set_bytes = 0;
};

auto metal_buffer(const Tensor& tensor) -> Result<MetalBufferView>;
auto metal_memory_stats() -> MetalMemoryStats;

} // namespace kidi::tensor