#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include "kidi/tensor/tensor.h"

namespace kidi::tensor {

/// Row-major table bytes held outside this process's linear memory, for example in browser JavaScript buffers.
/// Lookup tables such as per-layer token embeddings read only a few rows per step, so they need not be resident.
class RowSource {
public:
    virtual ~RowSource() = default;
    /// Copies each requested row, `destination.size() / rows.size()` bytes apiece, into `destination` in order.
    virtual auto gather(std::span<const std::int32_t> rows, std::span<std::byte> destination) const -> Result<void> = 0;
};

/// A CPU tensor of at least one dimension whose rows are read through `source`. It is not host-addressable: direct
/// byte access and device transfers fail, and lookups use `gather_rows`.
auto external_tensor(std::vector<std::int64_t> shape, DType dtype, std::shared_ptr<const RowSource> source)
    -> Result<Tensor>;
auto is_external(const Tensor& tensor) noexcept -> bool;
/// Copies the given first-axis rows of a contiguous host or external table into `destination`.
auto gather_rows(const Tensor& table, std::span<const std::int32_t> rows, std::span<std::byte> destination)
    -> Result<void>;

} // namespace kidi::tensor
