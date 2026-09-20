#include "kidi/tensor/tensor.h"
#include "kidi/tensor/arena.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <utility>

namespace kidi::tensor {
namespace {

auto checked_numel(std::span<const std::int64_t> shape) -> Result<std::size_t> {
    std::size_t result = 1;
    for (const auto extent : shape) {
        if (extent < 0) {
            return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "tensor shape cannot contain negative extents"});
        }
        const auto unsigned_extent = static_cast<std::size_t>(extent);
        if (unsigned_extent != 0 && result > std::numeric_limits<std::size_t>::max() / unsigned_extent) {
            return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "tensor element count overflows"});
        }
        result *= unsigned_extent;
    }
    return result;
}

auto checked_nbytes(std::span<const std::int64_t> shape, DType dtype) -> Result<std::size_t> {
    auto elements = checked_numel(shape);
    if (!elements) return std::unexpected(std::move(elements.error()));
    const auto width = element_size(dtype);
    if (*elements > std::numeric_limits<std::size_t>::max() / width) {
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "tensor byte count overflows"});
    }
    return *elements * width;
}

auto contiguous_strides(std::span<const std::int64_t> shape) -> detail::Dimensions {
    detail::Dimensions result(shape.size());
    std::int64_t stride = 1;
    for (std::size_t axis = shape.size(); axis-- > 0;) {
        result[axis] = stride;
        stride *= shape[axis];
    }
    return result;
}

} // namespace

auto Arena::reserve(std::size_t bytes) -> Result<void> {
    bytes = std::max<std::size_t>(bytes, 256);
    if (bytes > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()) ||
        bytes > std::numeric_limits<std::size_t>::max() - reserved_bytes_)
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "arena size overflow"});
    auto storage = Tensor::empty({static_cast<std::int64_t>(bytes)}, DType::U8, Device::cpu());
    if (!storage) return std::unexpected(std::move(storage.error()));
    slabs_.push_back({std::move(*storage), 0});
    reserved_bytes_ += bytes;
    return {};
}
auto Arena::allocate(std::vector<std::int64_t> shape, DType dtype) -> Result<Tensor> {
    auto bytes = checked_nbytes(shape, dtype);
    if (!bytes) return std::unexpected(std::move(bytes.error()));
    if (*bytes > std::numeric_limits<std::size_t>::max() - 255)
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "arena allocation overflow"});
    const auto aligned = std::max<std::size_t>(256, (*bytes + 255) & ~std::size_t{255});
    if (slabs_.empty() || aligned > slabs_.back().storage.nbytes() - slabs_.back().used) {
        auto status = reserve(std::max(slab_bytes_, aligned));
        if (!status) return std::unexpected(std::move(status.error()));
    }
    auto& slab = slabs_.back();
    auto owner = std::make_shared<std::shared_ptr<Storage>>(slab.storage.storage_);
    std::shared_ptr<Storage> slot(owner, slab.storage.storage_.get());
    auto strides = contiguous_strides(shape);
    Tensor result(slab.storage.backend_, std::move(slot), dtype, std::move(shape), std::move(strides),
                  static_cast<std::int64_t>(slab.used / element_size(dtype)));
    slab.used += aligned;
    return result;
}

Tensor::Tensor(std::shared_ptr<Backend> backend, std::shared_ptr<Storage> storage, DType dtype,
               detail::Dimensions shape, detail::Dimensions strides, std::int64_t storage_offset) noexcept
    : backend_(std::move(backend)),
      storage_(std::move(storage)),
      dtype_(dtype),
      shape_(std::move(shape)),
      strides_(std::move(strides)),
      storage_offset_(storage_offset) {}

auto Tensor::defined() const noexcept -> bool { return backend_ != nullptr && storage_ != nullptr; }

auto Tensor::dtype() const noexcept -> DType { return dtype_; }

auto Tensor::device() const noexcept -> Device { return defined() ? storage_->device() : Device::cpu(); }

auto Tensor::backend_name() const noexcept -> std::string_view {
    return defined() ? backend_->name() : std::string_view{};
}

auto Tensor::shape() const noexcept -> std::span<const std::int64_t> { return shape_; }

auto Tensor::strides() const noexcept -> std::span<const std::int64_t> { return strides_; }

auto Tensor::storage_offset() const noexcept -> std::int64_t { return storage_offset_; }

auto Tensor::dimensions() const noexcept -> std::size_t { return shape_.size(); }

auto Tensor::size(std::int64_t axis) const noexcept -> std::size_t {
    if (axis < 0) axis += static_cast<std::int64_t>(shape_.size());
    return static_cast<std::size_t>(shape_[static_cast<std::size_t>(axis)]);
}

auto Tensor::numel() const noexcept -> std::size_t {
    std::size_t result = 1;
    for (const auto extent : shape_) result *= static_cast<std::size_t>(extent);
    return result;
}

auto Tensor::nbytes() const noexcept -> std::size_t { return numel() * element_size(dtype_); }

auto Tensor::is_contiguous() const noexcept -> bool {
    if (strides_.size() != shape_.size()) return false;
    std::int64_t expected_stride = 1;
    for (std::size_t axis = shape_.size(); axis-- > 0;) {
        if (strides_[axis] != expected_stride) return false;
        expected_stride *= shape_[axis];
    }
    return true;
}

auto Tensor::is_host_accessible() const noexcept -> bool { return defined() && backend_->is_host_accessible(device()); }

auto Tensor::host_bytes() -> Result<std::span<std::byte>> {
    if (!defined()) {
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "cannot access an undefined tensor"});
    }
    if (!is_contiguous()) {
        return std::unexpected(Error{ErrorCode::UNSUPPORTED, "direct host access requires a contiguous tensor"});
    }
    auto bytes = backend_->host_view(*storage_);
    if (!bytes) return std::unexpected(std::move(bytes.error()));
    const auto offset = static_cast<std::size_t>(storage_offset_) * element_size(dtype_);
    return bytes->subspan(offset, nbytes());
}

auto Tensor::host_bytes() const -> Result<std::span<const std::byte>> {
    if (!defined()) {
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "cannot access an undefined tensor"});
    }
    if (!is_contiguous()) {
        return std::unexpected(Error{ErrorCode::UNSUPPORTED, "direct host access requires a contiguous tensor"});
    }
    auto bytes = backend_->host_view(std::as_const(*storage_));
    if (!bytes) return std::unexpected(std::move(bytes.error()));
    const auto offset = static_cast<std::size_t>(storage_offset_) * element_size(dtype_);
    return bytes->subspan(offset, nbytes());
}

auto Tensor::copy_to_host() const -> Result<std::vector<std::byte>> {
    if (!defined()) {
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "cannot copy an undefined tensor"});
    }
    if (!is_contiguous()) {
        return std::unexpected(Error{ErrorCode::UNSUPPORTED, "host copy requires a contiguous tensor"});
    }
    std::vector<std::byte> result(nbytes());
    const auto offset = static_cast<std::size_t>(storage_offset_) * element_size(dtype_);
    auto status = backend_->copy_to_host(*storage_, offset, result);
    if (!status) return std::unexpected(std::move(status.error()));
    return result;
}

auto Tensor::to(Device target) const -> Result<Tensor> {
    if (!defined()) {
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "cannot transfer an undefined tensor"});
    }
    if (target == device()) return *this;
    auto host = copy_to_host();
    if (!host) return std::unexpected(std::move(host.error()));
    return from_bytes({shape_.begin(), shape_.end()}, dtype_, *host, target);
}

auto Tensor::reshape(std::vector<std::int64_t> shape) const -> Result<Tensor> {
    if (!defined()) {
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "cannot reshape an undefined tensor"});
    }
    auto elements = checked_numel(shape);
    if (!elements) return std::unexpected(std::move(elements.error()));
    if (*elements != numel()) {
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "reshape changes tensor element count"});
    }
    if (!is_contiguous()) {
        return std::unexpected(Error{ErrorCode::UNSUPPORTED, "reshape requires a contiguous tensor"});
    }
    auto strides = contiguous_strides(shape);
    return Tensor(backend_, storage_, dtype_, std::move(shape), std::move(strides), storage_offset_);
}

auto Tensor::narrow(std::int64_t axis, std::int64_t start, std::int64_t length) const -> Result<Tensor> {
    if (!defined()) {
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "cannot narrow an undefined tensor"});
    }
    const auto rank = static_cast<std::int64_t>(shape_.size());
    if (axis < 0) axis += rank;
    if (axis < 0 || axis >= rank || length < 0) {
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "invalid tensor narrow geometry"});
    }
    const auto extent = shape_[static_cast<std::size_t>(axis)];
    if (start < 0) start += extent;
    if (start < 0 || start > extent || length > extent - start) {
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "tensor narrow is out of bounds"});
    }
    auto shape = shape_;
    shape[static_cast<std::size_t>(axis)] = length;
    const auto offset = storage_offset_ + start * strides_[static_cast<std::size_t>(axis)];
    return Tensor(backend_, storage_, dtype_, std::move(shape), strides_, offset);
}

auto Tensor::select(std::int64_t axis, std::int64_t index) const -> Result<Tensor> {
    if (!defined()) {
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "cannot select from an undefined tensor"});
    }
    const auto rank = static_cast<std::int64_t>(shape_.size());
    if (axis < 0) axis += rank;
    if (axis < 0 || axis >= rank) {
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "tensor select axis is out of bounds"});
    }
    const auto extent = shape_[static_cast<std::size_t>(axis)];
    if (index < 0) index += extent;
    if (index < 0 || index >= extent) {
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "tensor select index is out of bounds"});
    }
    const auto selected_axis = static_cast<std::size_t>(axis);
    auto shape = shape_;
    auto strides = strides_;
    const auto offset = storage_offset_ + index * strides[selected_axis];
    shape.erase(shape.begin() + static_cast<std::ptrdiff_t>(selected_axis));
    strides.erase(strides.begin() + static_cast<std::ptrdiff_t>(selected_axis));
    return Tensor(backend_, storage_, dtype_, std::move(shape), std::move(strides), offset);
}

auto Tensor::empty(std::vector<std::int64_t> shape, DType dtype, Device device) -> Result<Tensor> {
    auto bytes = checked_nbytes(shape, dtype);
    if (!bytes) return std::unexpected(std::move(bytes.error()));
    auto backend = BackendRegistry::instance().backend(device);
    if (!backend) return std::unexpected(std::move(backend.error()));
    if (!(*backend)->is_available(device)) {
        return std::unexpected(Error{ErrorCode::UNSUPPORTED, std::string((*backend)->name()) +
                                                                 " backend is unavailable for " + to_string(device) +
                                                                 ": " + (*backend)->unavailable_reason(device)});
    }
    auto storage = (*backend)->allocate(device, *bytes, 64);
    if (!storage) return std::unexpected(std::move(storage.error()));
    auto strides = contiguous_strides(shape);
    return Tensor(std::move(*backend), std::move(*storage), dtype, std::move(shape), std::move(strides), 0);
}

auto Tensor::zeros(std::vector<std::int64_t> shape, DType dtype, Device device) -> Result<Tensor> {
    auto tensor = empty(std::move(shape), dtype, device);
    if (!tensor) return std::unexpected(std::move(tensor.error()));
    std::vector<std::byte> zeros(tensor->nbytes());
    auto status = tensor->backend_->copy_from_host(*tensor->storage_, 0, zeros);
    if (!status) return std::unexpected(std::move(status.error()));
    return tensor;
}

auto Tensor::from_bytes(std::vector<std::int64_t> shape, DType dtype, std::span<const std::byte> bytes, Device device)
    -> Result<Tensor> {
    auto expected = checked_nbytes(shape, dtype);
    if (!expected) return std::unexpected(std::move(expected.error()));
    if (bytes.size() != *expected) {
        return std::unexpected(
            Error{ErrorCode::INVALID_ARGUMENT, "host byte count does not match tensor shape and dtype"});
    }
    auto tensor = empty(std::move(shape), dtype, device);
    if (!tensor) return std::unexpected(std::move(tensor.error()));
    auto status = tensor->backend_->copy_from_host(*tensor->storage_, 0, bytes);
    if (!status) return std::unexpected(std::move(status.error()));
    return tensor;
}

auto Tensor::from_blob(std::vector<std::int64_t> shape, DType dtype, std::span<const std::byte> bytes,
                       std::shared_ptr<const void> owner) -> Result<Tensor> {
    auto expected = checked_nbytes(shape, dtype);
    if (!expected) return std::unexpected(std::move(expected.error()));
    if (bytes.size() != *expected) {
        return std::unexpected(
            Error{ErrorCode::INVALID_ARGUMENT, "host blob byte count does not match tensor shape and dtype"});
    }
    auto backend = BackendRegistry::instance().backend(Device::cpu());
    if (!backend) return std::unexpected(std::move(backend.error()));
    auto storage = (*backend)->wrap_host(Device::cpu(), bytes, std::move(owner));
    if (!storage) return std::unexpected(std::move(storage.error()));
    auto strides = contiguous_strides(shape);
    return Tensor(std::move(*backend), std::move(*storage), dtype, std::move(shape), std::move(strides), 0);
}

} // namespace kidi::tensor