#pragma once

#include "kidi/tensor/tensor.h"

namespace kidi::tensor {
struct WebGpuBufferView {
    std::uint32_t handle;
    std::size_t offset_bytes;
};
auto web_gpu_buffer(const Tensor& tensor) -> Result<WebGpuBufferView>;
auto web_gpu_synchronize() -> Result<void>;
auto web_gpu_error() -> Error;
} // namespace kidi::tensor