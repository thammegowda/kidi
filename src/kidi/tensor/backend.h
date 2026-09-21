#pragma once

#include <cstddef>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "kidi/core/error.h"
#include "kidi/tensor/device.h"

namespace kidi::tensor {

class Storage {
public:
    virtual ~Storage() = default;

    virtual auto device() const noexcept -> Device = 0;
    virtual auto size_bytes() const noexcept -> std::size_t = 0;
};

class Backend {
public:
    virtual ~Backend() = default;

    virtual auto name() const noexcept -> std::string_view = 0;
    virtual auto device_kind() const noexcept -> DeviceKind = 0;
    virtual auto is_available(Device device) const noexcept -> bool = 0;
    virtual auto is_host_accessible(Device device) const noexcept -> bool = 0;
    virtual auto supports_execution() const noexcept -> bool = 0;
    virtual auto unavailable_reason(Device device) const -> std::string = 0;

    virtual auto allocate(Device device, std::size_t size_bytes, std::size_t alignment) const
        -> Result<std::shared_ptr<Storage>> = 0;
    virtual auto wrap_host(Device device, std::span<const std::byte> bytes, std::shared_ptr<const void> owner) const
        -> Result<std::shared_ptr<Storage>> = 0;
    virtual auto copy_from_host(Storage& destination, std::size_t destination_offset,
                                std::span<const std::byte> source) const -> Result<void> = 0;
    virtual auto copy_to_host(const Storage& source, std::size_t source_offset, std::span<std::byte> destination) const
        -> Result<void> = 0;
    virtual auto host_view(Storage& storage) const -> Result<std::span<std::byte>> = 0;
    virtual auto host_view(const Storage& storage) const -> Result<std::span<const std::byte>> = 0;
    virtual auto synchronize(Device device) const -> Result<void> = 0;
};

struct BackendInfo {
    DeviceKind device_kind;
    std::string name;
    bool storage_available;
    bool execution_available;
    std::string unavailable_reason;
};

class BackendRegistry {
public:
    static auto instance() -> BackendRegistry&;

    auto register_backend(std::shared_ptr<Backend> backend) -> Result<void>;
    auto backend(Device device) const -> Result<std::shared_ptr<Backend>>;
    auto backends(std::int32_t device_index = 0) const -> std::vector<BackendInfo>;

private:
    BackendRegistry();

    struct DeviceKindHash {
        auto operator()(DeviceKind kind) const noexcept -> std::size_t { return static_cast<std::size_t>(kind); }
    };

    mutable std::mutex mutex_;
    std::unordered_map<DeviceKind, std::shared_ptr<Backend>, DeviceKindHash> backends_;
};

auto make_ynnpack_backend() -> std::shared_ptr<Backend>;
auto make_metal_backend() -> std::shared_ptr<Backend>;
auto make_unavailable_backend(DeviceKind kind, std::string name, std::string reason) -> std::shared_ptr<Backend>;

} // namespace kidi::tensor