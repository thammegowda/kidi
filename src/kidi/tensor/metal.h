#pragma once

#include <cstddef>

#include "kidi/core/error.h"

namespace kidi::tensor {

class Tensor;

struct MetalBufferView {
    void* handle;
    std::size_t offset_bytes;
};

auto metal_buffer(const Tensor& tensor) -> Result<MetalBufferView>;

} // namespace kidi::tensor