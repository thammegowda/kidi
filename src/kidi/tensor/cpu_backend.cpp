#include "kidi/tensor/backend.h"

#include <algorithm>
#include <cstring>
#include <new>
#include <utility>

namespace kidi::tensor {
namespace {

constexpr std::size_t YNNPACK_ALIGNMENT = 64;

class CpuStorage final : public Storage {
public:
    CpuStorage(Device device, std::size_t size_bytes, std::size_t alignment)
        : device_(device), size_bytes_(size_bytes), writable_(true) {
        auto* allocation = ::operator new(std::max<std::size_t>(size_bytes, 1), std::align_val_t(alignment));
        owner_ = std::shared_ptr<const void>(allocation, [alignment](const void* pointer) {
            ::operator delete(const_cast<void*>(pointer), std::align_val_t(alignment));
        });
        data_ = static_cast<const std::byte*>(allocation);
    }

    CpuStorage(Device device, std::span<const std::byte> bytes, std::shared_ptr<const void> owner)
        : device_(device), size_bytes_(bytes.size()), owner_(std::move(owner)), data_(bytes.data()), writable_(false) {}

    CpuStorage(const CpuStorage&) = delete;
    auto operator=(const CpuStorage&) -> CpuStorage& = delete;

    auto device() const noexcept -> Device override { return device_; }
    auto size_bytes() const noexcept -> std::size_t override { return size_bytes_; }
    auto mutable_data() noexcept -> std::byte* { return writable_ ? const_cast<std::byte*>(data_) : nullptr; }
    auto data() const noexcept -> const std::byte* { return data_; }
    auto writable() const noexcept -> bool { return writable_; }

private:
    Device device_;
    std::size_t size_bytes_;
    std::shared_ptr<const void> owner_;
    const std::byte* data_;
    bool writable_;
};

auto cpu_storage(Storage& storage) -> Result<CpuStorage*> {
    auto* result = dynamic_cast<CpuStorage*>(&storage);
    if (result == nullptr) {
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "storage does not belong to YNNPACK backend"});
    }
    return result;
}

auto cpu_storage(const Storage& storage) -> Result<const CpuStorage*> {
    const auto* result = dynamic_cast<const CpuStorage*>(&storage);
    if (result == nullptr) {
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "storage does not belong to YNNPACK backend"});
    }
    return result;
}

auto check_copy_bounds(std::size_t storage_size, std::size_t offset, std::size_t copy_size) -> Result<void> {
    if (offset > storage_size || copy_size > storage_size - offset) {
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "tensor storage copy is out of bounds"});
    }
    return {};
}

class YnnpackBackend final : public Backend {
public:
    auto name() const noexcept -> std::string_view override { return "ynnpack"; }
    auto device_kind() const noexcept -> DeviceKind override { return DeviceKind::CPU; }
    auto is_available(Device device) const noexcept -> bool override {
        return device.kind == DeviceKind::CPU && device.index == 0;
    }
    auto is_host_accessible(Device device) const noexcept -> bool override { return is_available(device); }
    auto supports_execution() const noexcept -> bool override { return true; }
    auto unavailable_reason(Device) const -> std::string override { return "YNNPACK supports cpu:0 only"; }

    auto allocate(Device device, std::size_t size_bytes, std::size_t alignment) const
        -> Result<std::shared_ptr<Storage>> override {
        if (!is_available(device)) {
            return std::unexpected(Error{ErrorCode::UNSUPPORTED, unavailable_reason(device)});
        }
        alignment = std::max(alignment, YNNPACK_ALIGNMENT);
        if ((alignment & (alignment - 1)) != 0) {
            return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "tensor alignment must be a power of two"});
        }
        try {
            return std::shared_ptr<Storage>(new CpuStorage(device, size_bytes, alignment));
        } catch (const std::bad_alloc&) {
            return std::unexpected(Error{ErrorCode::RUNTIME, "failed to allocate CPU tensor storage"});
        }
    }

    auto wrap_host(Device device, std::span<const std::byte> bytes, std::shared_ptr<const void> owner) const
        -> Result<std::shared_ptr<Storage>> override {
        if (!is_available(device)) {
            return std::unexpected(Error{ErrorCode::UNSUPPORTED, unavailable_reason(device)});
        }
        if (!owner && !bytes.empty()) {
            return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "external tensor storage requires an owner"});
        }
        return std::shared_ptr<Storage>(new CpuStorage(device, bytes, std::move(owner)));
    }

    auto copy_from_host(Storage& destination, std::size_t destination_offset, std::span<const std::byte> source) const
        -> Result<void> override {
        auto storage = cpu_storage(destination);
        if (!storage) return std::unexpected(std::move(storage.error()));
        if (!(*storage)->writable()) {
            return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "cannot write to read-only tensor storage"});
        }
        auto bounds = check_copy_bounds((*storage)->size_bytes(), destination_offset, source.size());
        if (!bounds) return bounds;
        std::memcpy((*storage)->mutable_data() + destination_offset, source.data(), source.size());
        return {};
    }

    auto copy_to_host(const Storage& source, std::size_t source_offset, std::span<std::byte> destination) const
        -> Result<void> override {
        auto storage = cpu_storage(source);
        if (!storage) return std::unexpected(std::move(storage.error()));
        auto bounds = check_copy_bounds((*storage)->size_bytes(), source_offset, destination.size());
        if (!bounds) return bounds;
        std::memcpy(destination.data(), (*storage)->data() + source_offset, destination.size());
        return {};
    }

    auto host_view(Storage& storage) const -> Result<std::span<std::byte>> override {
        auto typed = cpu_storage(storage);
        if (!typed) return std::unexpected(std::move(typed.error()));
        if (!(*typed)->writable()) {
            return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "tensor storage is read-only"});
        }
        return std::span<std::byte>((*typed)->mutable_data(), (*typed)->size_bytes());
    }

    auto host_view(const Storage& storage) const -> Result<std::span<const std::byte>> override {
        auto typed = cpu_storage(storage);
        if (!typed) return std::unexpected(std::move(typed.error()));
        return std::span<const std::byte>((*typed)->data(), (*typed)->size_bytes());
    }

    auto synchronize(Device device) const -> Result<void> override {
        if (!is_available(device)) {
            return std::unexpected(Error{ErrorCode::UNSUPPORTED, unavailable_reason(device)});
        }
        return {};
    }
};

} // namespace

auto make_ynnpack_backend() -> std::shared_ptr<Backend> { return std::make_shared<YnnpackBackend>(); }

} // namespace kidi::tensor