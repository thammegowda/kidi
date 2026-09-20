#include "kidi/runtime/operator.h"

#include <algorithm>

#include "kidi/ops/context.h"

namespace kidi::runtime {

auto OutputPool::acquire(std::span<const std::int64_t> shape, tensor::DType dtype, tensor::Device device)
    -> tensor::Tensor {
    for (const auto& buffer : buffers_)
        if (buffer.owns_unique_storage() && buffer.dtype() == dtype && buffer.device() == device &&
            std::ranges::equal(buffer.shape(), shape))
            return buffer;
    auto output = ops::require(arena_ ? arena_->allocate({shape.begin(), shape.end()}, dtype)
                                      : tensor::Tensor::empty({shape.begin(), shape.end()}, dtype, device));
    ++allocations_.count;
    allocations_.bytes += output.nbytes();
    buffers_.push_back(output);
    return output;
}

} // namespace kidi::runtime