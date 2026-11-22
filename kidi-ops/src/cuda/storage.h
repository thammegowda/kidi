#pragma once
#include "kidi/tensor/tensor.h"

namespace kidi::cuda {
struct BufferView {
    void* data;
    std::size_t bytes;
};
auto buffer_view(const tensor::Tensor& tensor) -> Result<BufferView>;
} // namespace kidi::cuda