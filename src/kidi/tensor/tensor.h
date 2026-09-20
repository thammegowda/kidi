#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <type_traits>
#include <vector>
#include <array>
#include <algorithm>

#include "kidi/core/error.h"
#include "kidi/tensor/backend.h"
#include "kidi/tensor/dtype.h"

namespace kidi::tensor {

namespace detail {
/// Inline metadata for common tensor ranks, with an owning fallback for larger ranks.
class Dimensions {
public:
    Dimensions() = default;
    explicit Dimensions(std::size_t size) : size_(size) {
        if (size_ > inline_.size()) large_.resize(size_);
    }
    Dimensions(std::vector<std::int64_t> values) : size_(values.size()) {
        if (size_ > inline_.size())
            large_ = std::move(values);
        else
            std::copy(values.begin(), values.end(), inline_.begin());
    }
    auto size() const noexcept -> std::size_t { return size_; }
    auto data() noexcept -> std::int64_t* { return size_ > inline_.size() ? large_.data() : inline_.data(); }
    auto data() const noexcept -> const std::int64_t* {
        return size_ > inline_.size() ? large_.data() : inline_.data();
    }
    auto begin() noexcept -> std::int64_t* { return data(); }
    auto end() noexcept -> std::int64_t* { return data() + size_; }
    auto begin() const noexcept -> const std::int64_t* { return data(); }
    auto end() const noexcept -> const std::int64_t* { return data() + size_; }
    auto operator[](std::size_t index) noexcept -> std::int64_t& { return data()[index]; }
    auto operator[](std::size_t index) const noexcept -> std::int64_t { return data()[index]; }
    operator std::span<const std::int64_t>() const noexcept { return {data(), size_}; }
    auto erase(std::int64_t* position) -> void {
        const auto index = position - begin();
        if (size_ > inline_.size()) {
            large_.erase(large_.begin() + index);
            --size_;
            if (size_ == inline_.size()) std::copy(large_.begin(), large_.end(), inline_.begin());
        } else {
            std::move(position + 1, end(), position);
            --size_;
        }
    }

private:
    std::array<std::int64_t, 8> inline_{};
    std::vector<std::int64_t> large_;
    std::size_t size_ = 0;
};
} // namespace detail

struct MetalBufferView;

class Tensor {
public:
    Tensor() = default;

    auto defined() const noexcept -> bool;
    auto dtype() const noexcept -> DType;
    auto device() const noexcept -> Device;
    auto backend_name() const noexcept -> std::string_view;
    auto shape() const noexcept -> std::span<const std::int64_t>;
    auto strides() const noexcept -> std::span<const std::int64_t>;
    auto storage_offset() const noexcept -> std::int64_t;
    auto dimensions() const noexcept -> std::size_t;
    auto size(std::int64_t axis) const noexcept -> std::size_t;
    auto numel() const noexcept -> std::size_t;
    auto nbytes() const noexcept -> std::size_t;
    auto is_contiguous() const noexcept -> bool;
    auto is_host_accessible() const noexcept -> bool;
    auto owns_unique_storage() const noexcept -> bool { return storage_ && storage_.use_count() == 1; }

    auto host_bytes() -> Result<std::span<std::byte>>;
    auto host_bytes() const -> Result<std::span<const std::byte>>;
    auto copy_to_host() const -> Result<std::vector<std::byte>>;
    auto to(Device target) const -> Result<Tensor>;
    auto reshape(std::vector<std::int64_t> shape) const -> Result<Tensor>;
    auto narrow(std::int64_t axis, std::int64_t start, std::int64_t length) const -> Result<Tensor>;
    auto select(std::int64_t axis, std::int64_t index) const -> Result<Tensor>;

    template <typename Value>
    auto data() -> Result<std::span<Value>> {
        using Element = std::remove_cv_t<Value>;
        if (dtype_ != dtype_of<Element>) {
            return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "tensor dtype does not match requested type"});
        }
        auto bytes = host_bytes();
        if (!bytes) return std::unexpected(std::move(bytes.error()));
        return std::span<Value>(reinterpret_cast<Value*>(bytes->data()), numel());
    }

    template <typename Value>
    auto data() const -> Result<std::span<const Value>> {
        using Element = std::remove_cv_t<Value>;
        if (dtype_ != dtype_of<Element>) {
            return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "tensor dtype does not match requested type"});
        }
        auto bytes = host_bytes();
        if (!bytes) return std::unexpected(std::move(bytes.error()));
        return std::span<const Value>(reinterpret_cast<const Value*>(bytes->data()), numel());
    }

    static auto empty(std::vector<std::int64_t> shape, DType dtype, Device device = Device::cpu()) -> Result<Tensor>;
    static auto zeros(std::vector<std::int64_t> shape, DType dtype, Device device = Device::cpu()) -> Result<Tensor>;
    static auto from_bytes(std::vector<std::int64_t> shape, DType dtype, std::span<const std::byte> bytes,
                           Device device = Device::cpu()) -> Result<Tensor>;
    static auto from_blob(std::vector<std::int64_t> shape, DType dtype, std::span<const std::byte> bytes,
                          std::shared_ptr<const void> owner) -> Result<Tensor>;

    template <typename Value>
    static auto from_host(std::vector<std::int64_t> shape, std::span<const Value> values, Device device = Device::cpu())
        -> Result<Tensor> {
        return from_bytes(std::move(shape), dtype_of<std::remove_cv_t<Value>>, std::as_bytes(values), device);
    }

private:
    Tensor(std::shared_ptr<Backend> backend, std::shared_ptr<Storage> storage, DType dtype, detail::Dimensions shape,
           detail::Dimensions strides, std::int64_t storage_offset) noexcept;

    friend auto metal_buffer(const Tensor& tensor) -> Result<MetalBufferView>;
    friend class Arena;

    std::shared_ptr<Backend> backend_;
    std::shared_ptr<Storage> storage_;
    DType dtype_ = DType::F32;
    detail::Dimensions shape_;
    detail::Dimensions strides_;
    std::int64_t storage_offset_ = 0;
};

} // namespace kidi::tensor