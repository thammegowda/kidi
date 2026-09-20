#pragma once

#include "kidi/tensor/tensor.h"

namespace kidi::tensor {

/// Grow-only aligned CPU storage for reusable tensor slots. Returned tensors keep their slab alive.
class Arena {
public:
    explicit Arena(std::size_t slab_bytes = 8 * 1024 * 1024) : slab_bytes_(slab_bytes) {}
    auto reserve(std::size_t bytes) -> Result<void>;
    auto allocate(std::vector<std::int64_t> shape, DType dtype) -> Result<Tensor>;
    auto reserved_bytes() const noexcept -> std::size_t { return reserved_bytes_; }

private:
    struct Slab {
        Tensor storage;
        std::size_t used = 0;
    };
    std::size_t slab_bytes_, reserved_bytes_ = 0;
    std::vector<Slab> slabs_;
};

} // namespace kidi::tensor