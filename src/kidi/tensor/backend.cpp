#include "kidi/tensor/backend.h"

#include <array>
#include <utility>

namespace kidi::tensor {
namespace {

class UnavailableBackend final : public Backend {
public:
    UnavailableBackend(DeviceKind kind, std::string name, std::string reason)
        : kind_(kind), name_(std::move(name)), reason_(std::move(reason)) {}

    auto name() const noexcept -> std::string_view override { return name_; }
    auto device_kind() const noexcept -> DeviceKind override { return kind_; }
    auto is_available(Device) const noexcept -> bool override { return false; }
    auto is_host_accessible(Device) const noexcept -> bool override { return false; }
    auto supports_execution() const noexcept -> bool override { return false; }
    auto unavailable_reason(Device) const -> std::string override { return reason_; }

    auto allocate(Device, std::size_t, std::size_t) const -> Result<std::shared_ptr<Storage>> override {
        return std::unexpected(unavailable());
    }

    auto wrap_host(Device, std::span<const std::byte>, std::shared_ptr<const void>) const
        -> Result<std::shared_ptr<Storage>> override {
        return std::unexpected(unavailable());
    }

    auto copy_from_host(Storage&, std::size_t, std::span<const std::byte>) const -> Result<void> override {
        return std::unexpected(unavailable());
    }

    auto copy_to_host(const Storage&, std::size_t, std::span<std::byte>) const -> Result<void> override {
        return std::unexpected(unavailable());
    }

    auto host_view(Storage&) const -> Result<std::span<std::byte>> override { return std::unexpected(unavailable()); }

    auto host_view(const Storage&) const -> Result<std::span<const std::byte>> override {
        return std::unexpected(unavailable());
    }

    auto synchronize(Device) const -> Result<void> override { return std::unexpected(unavailable()); }

private:
    auto unavailable() const -> Error {
        return Error{ErrorCode::UNSUPPORTED, name_ + " backend is unavailable: " + reason_};
    }

    DeviceKind kind_;
    std::string name_;
    std::string reason_;
};

} // namespace

auto make_unavailable_backend(DeviceKind kind, std::string name, std::string reason) -> std::shared_ptr<Backend> {
    return std::make_shared<UnavailableBackend>(kind, std::move(name), std::move(reason));
}

auto Backend::clear(Storage& storage) const -> Result<void> {
    const std::vector<std::byte> zeros(storage.size_bytes());
    return copy_from_host(storage, 0, zeros);
}

auto BackendRegistry::instance() -> BackendRegistry& {
    static BackendRegistry registry;
    return registry;
}

BackendRegistry::BackendRegistry() {
    backends_.emplace(DeviceKind::CPU, make_ynnpack_backend());
#if defined(KIDI_HAS_METAL)
    backends_.emplace(DeviceKind::A_GPU, make_metal_backend());
#else
    backends_.emplace(DeviceKind::A_GPU,
                      make_unavailable_backend(DeviceKind::A_GPU, "metal-mps", "Metal backend is not built"));
#endif
    backends_.emplace(DeviceKind::Q_NPU, make_unavailable_backend(DeviceKind::Q_NPU, "hexagon-qnn", "not implemented"));
    backends_.emplace(DeviceKind::CUDA, make_unavailable_backend(DeviceKind::CUDA, "cuda-cudnn", "not implemented"));
#if defined(KIDI_HAS_WEBGPU)
    backends_.emplace(DeviceKind::WEB_GPU, make_web_gpu_backend());
#else
    backends_.emplace(DeviceKind::WEB_GPU,
                      make_unavailable_backend(DeviceKind::WEB_GPU, "webgpu", "WebGPU backend is not built"));
#endif
}

auto BackendRegistry::register_backend(std::shared_ptr<Backend> backend) -> Result<void> {
    if (!backend || backend->name().empty()) {
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "cannot register an empty tensor backend"});
    }
    std::scoped_lock lock(mutex_);
    backends_.insert_or_assign(backend->device_kind(), std::move(backend));
    return {};
}

auto BackendRegistry::backend(Device device) const -> Result<std::shared_ptr<Backend>> {
    if (device.index < 0) {
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "tensor device index cannot be negative"});
    }
    std::scoped_lock lock(mutex_);
    const auto found = backends_.find(device.kind);
    if (found == backends_.end()) {
        return std::unexpected(Error{ErrorCode::UNSUPPORTED, "no tensor backend registered for " + to_string(device)});
    }
    return found->second;
}

auto BackendRegistry::backends(std::int32_t device_index) const -> std::vector<BackendInfo> {
    constexpr std::array KINDS = {DeviceKind::CPU, DeviceKind::A_GPU, DeviceKind::Q_NPU, DeviceKind::CUDA,
                                  DeviceKind::WEB_GPU};
    std::vector<BackendInfo> result;
    result.reserve(KINDS.size());
    std::scoped_lock lock(mutex_);
    for (const auto kind : KINDS) {
        const auto found = backends_.find(kind);
        if (found == backends_.end()) continue;
        const Device device{kind, device_index};
        const bool storage_available = found->second->is_available(device);
        result.push_back({kind, std::string(found->second->name()), storage_available,
                          storage_available && found->second->supports_execution(),
                          storage_available ? std::string{} : found->second->unavailable_reason(device)});
    }
    return result;
}

} // namespace kidi::tensor