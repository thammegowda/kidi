#pragma once
#include "kidi/tensor/tensor.h"

namespace kidi::tensor {
struct CudaBufferView {
    void* data;
    std::size_t bytes;
};
auto cuda_buffer(const Tensor& tensor) -> Result<CudaBufferView>;
} // namespace kidi::tensor