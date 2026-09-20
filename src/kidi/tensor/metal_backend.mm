#include "kidi/tensor/backend.h"
#include "kidi/tensor/metal.h"
#include "kidi/tensor/tensor.h"

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include <algorithm>
#include <cstring>
#include <utility>

namespace kidi::tensor {
namespace {

class MetalStorage final : public Storage {
public:
    MetalStorage(Device device, id<MTLBuffer> buffer, std::size_t size_bytes)
        : device_(device), buffer_(buffer), size_bytes_(size_bytes) {}

    auto device() const noexcept -> Device override { return device_; }
    auto size_bytes() const noexcept -> std::size_t override { return size_bytes_; }
    auto buffer() const noexcept -> id<MTLBuffer> { return buffer_; }

private:
    Device device_;
    id<MTLBuffer> buffer_;
    std::size_t size_bytes_;
};

auto metal_storage(Storage& storage) -> Result<MetalStorage*> {
    auto* result = dynamic_cast<MetalStorage*>(&storage);
    if (result == nullptr) {
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "storage does not belong to Metal backend"});
    }
    return result;
}

auto metal_storage(const Storage& storage) -> Result<const MetalStorage*> {
    const auto* result = dynamic_cast<const MetalStorage*>(&storage);
    if (result == nullptr) {
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "storage does not belong to Metal backend"});
    }
    return result;
}

auto check_copy_bounds(std::size_t storage_size, std::size_t offset, std::size_t copy_size) -> Result<void> {
    if (offset > storage_size || copy_size > storage_size - offset) {
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "Metal tensor storage copy is out of bounds"});
    }
    return {};
}

class MetalBackend final : public Backend {
public:
    MetalBackend() : device_(MTLCreateSystemDefaultDevice()) {}

    auto name() const noexcept -> std::string_view override { return "metal-mps"; }
    auto device_kind() const noexcept -> DeviceKind override { return DeviceKind::A_GPU; }
    auto is_available(Device device) const noexcept -> bool override {
        return device.kind == DeviceKind::A_GPU && device.index == 0 && device_ != nil;
    }
    auto is_host_accessible(Device device) const noexcept -> bool override { return is_available(device); }
    auto supports_execution() const noexcept -> bool override { return true; }
    auto unavailable_reason(Device device) const -> std::string override {
        if (device.kind != DeviceKind::A_GPU || device.index != 0) return "Metal backend supports a_gpu:0 only";
        return device_ == nil ? "no Metal device is available" : std::string{};
    }

    auto allocate(Device device, std::size_t size_bytes, std::size_t) const
        -> Result<std::shared_ptr<Storage>> override {
        if (!is_available(device)) {
            return std::unexpected(Error{ErrorCode::UNSUPPORTED, unavailable_reason(device)});
        }
        const auto allocation_size = std::max<std::size_t>(size_bytes, 1);
        id<MTLBuffer> buffer = [device_ newBufferWithLength:allocation_size options:MTLResourceStorageModeShared];
        if (buffer == nil) {
            return std::unexpected(Error{ErrorCode::RUNTIME, "failed to allocate Metal tensor storage"});
        }
        return std::shared_ptr<Storage>(new MetalStorage(device, buffer, size_bytes));
    }

    auto wrap_host(Device, std::span<const std::byte>, std::shared_ptr<const void>) const
        -> Result<std::shared_ptr<Storage>> override {
        return std::unexpected(
            Error{ErrorCode::UNSUPPORTED, "zero-copy host blobs are supported only by the CPU backend"});
    }

    auto copy_from_host(Storage& destination, std::size_t destination_offset, std::span<const std::byte> source) const
        -> Result<void> override {
        auto storage = metal_storage(destination);
        if (!storage) return std::unexpected(std::move(storage.error()));
        auto bounds = check_copy_bounds((*storage)->size_bytes(), destination_offset, source.size());
        if (!bounds) return bounds;
        std::memcpy(static_cast<std::byte*>((*storage)->buffer().contents) + destination_offset, source.data(),
                    source.size());
        return {};
    }

    auto copy_to_host(const Storage& source, std::size_t source_offset, std::span<std::byte> destination) const
        -> Result<void> override {
        auto storage = metal_storage(source);
        if (!storage) return std::unexpected(std::move(storage.error()));
        auto bounds = check_copy_bounds((*storage)->size_bytes(), source_offset, destination.size());
        if (!bounds) return bounds;
        std::memcpy(destination.data(), static_cast<const std::byte*>((*storage)->buffer().contents) + source_offset,
                    destination.size());
        return {};
    }

    auto host_view(Storage& storage) const -> Result<std::span<std::byte>> override {
        auto typed = metal_storage(storage);
        if (!typed) return std::unexpected(std::move(typed.error()));
        return std::span<std::byte>(static_cast<std::byte*>((*typed)->buffer().contents), (*typed)->size_bytes());
    }

    auto host_view(const Storage& storage) const -> Result<std::span<const std::byte>> override {
        auto typed = metal_storage(storage);
        if (!typed) return std::unexpected(std::move(typed.error()));
        return std::span<const std::byte>(static_cast<const std::byte*>((*typed)->buffer().contents),
                                          (*typed)->size_bytes());
    }

    auto synchronize(Device device) const -> Result<void> override {
        if (!is_available(device)) {
            return std::unexpected(Error{ErrorCode::UNSUPPORTED, unavailable_reason(device)});
        }
        return {};
    }

private:
    id<MTLDevice> device_;
};

} // namespace

auto make_metal_backend() -> std::shared_ptr<Backend> { return std::make_shared<MetalBackend>(); }

auto metal_buffer(const Tensor& tensor) -> Result<MetalBufferView> {
    if (!tensor.defined() || tensor.device().kind != DeviceKind::A_GPU || !tensor.is_contiguous()) {
        return std::unexpected(
            Error{ErrorCode::INVALID_ARGUMENT, "Metal execution requires a contiguous Apple GPU tensor"});
    }
    auto storage = metal_storage(std::as_const(*tensor.storage_));
    if (!storage) return std::unexpected(std::move(storage.error()));
    return MetalBufferView{
        .handle = (__bridge void*)(*storage)->buffer(),
        .offset_bytes = static_cast<std::size_t>(tensor.storage_offset_) * element_size(tensor.dtype_),
    };
}

} // namespace kidi::tensor