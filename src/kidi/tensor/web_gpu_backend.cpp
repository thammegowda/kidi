#include "kidi/tensor/web_gpu.h"

#include <emscripten.h>

namespace kidi::tensor {
namespace {
// clang-format off
EM_JS(int, available, (), { return Module.kidiGpu && !Module.kidiGpu.failure ? 1 : 0; });
EM_JS(int, allocate_buffer, (std::size_t bytes), {
    try {
        return Module.kidiGpu.allocate(Number(bytes));
    } catch (error) {
        Module.kidiGpu.failure = String(error);
        return 0;
    }
});
EM_JS(void, release_buffer, (int handle), { Module.kidiGpu.release(handle); });
EM_JS(int, clear_buffer, (int handle), {
    try {
        Module.kidiGpu.clear(handle);
        return 0;
    } catch (error) {
        Module.kidiGpu.failure = String(error);
        return -1;
    }
});
EM_JS(int, upload_buffer, (int handle, std::size_t offset, const void* source, std::size_t bytes), {
    try {
        Module.kidiGpu.upload(handle, Number(offset), HEAPU8.subarray(Number(source), Number(source) + Number(bytes)));
        return 0;
    } catch (error) {
        Module.kidiGpu.failure = String(error);
        return -1;
    }
});
EM_JS(int, queue_read, (int handle, std::size_t offset, void* destination, std::size_t bytes), {
    try {
        Module.kidiGpu.readLater(handle, Number(offset), Number(destination), Number(bytes));
        return 0;
    } catch (error) {
        Module.kidiGpu.failure = Module.kidiGpu.failure || String(error);
        return -1;
    }
});
EM_JS(int, finish_work, (), {
    try {
        Module.kidiGpu.flush();
        return 0;
    } catch (error) {
        Module.kidiGpu.failure = Module.kidiGpu.failure || String(error);
        return -1;
    }
});
EM_JS(void, error_text, (char* destination, int capacity),
    { stringToUTF8(Module.kidiGpu?.failure || 'WebGPU operation failed', Number(destination), capacity); });
// clang-format on

WebGpuWait wait_for_gpu = nullptr;

class WebGpuStorage final : public Storage {
public:
    explicit WebGpuStorage(std::size_t bytes) : bytes_(bytes), handle_(allocate_buffer(bytes)) {}
    WebGpuStorage(std::size_t bytes, std::uint32_t handle, std::int32_t packed_layout)
        : bytes_(bytes), handle_(handle), packed_layout_(packed_layout) {}
    ~WebGpuStorage() override {
        if (handle_) release_buffer(handle_);
    }
    auto device() const noexcept -> Device override { return Device::web_gpu(); }
    auto size_bytes() const noexcept -> std::size_t override { return bytes_; }
    auto handle() const -> std::uint32_t { return handle_; }
    auto packed_layout() const -> std::int32_t { return packed_layout_; }

private:
    std::size_t bytes_;
    std::uint32_t handle_;
    std::int32_t packed_layout_ = 0;
};

class WebGpuBackend final : public Backend {
public:
    auto name() const noexcept -> std::string_view override { return "webgpu"; }
    auto device_kind() const noexcept -> DeviceKind override { return DeviceKind::WEB_GPU; }
    auto is_available(Device device) const noexcept -> bool override {
        return device == Device::web_gpu() && available();
    }
    auto is_host_accessible(Device) const noexcept -> bool override { return false; }
    auto supports_execution() const noexcept -> bool override { return true; }
    auto clear(Storage& storage) const -> Result<void> override {
        if (clear_buffer(static_cast<WebGpuStorage&>(storage).handle())) return std::unexpected(web_gpu_error());
        return {};
    }
    auto unavailable_reason(Device) const -> std::string override { return web_gpu_error().message; }
    auto allocate(Device device, std::size_t bytes, std::size_t) const -> Result<std::shared_ptr<Storage>> override {
        if (!is_available(device)) return std::unexpected(web_gpu_error());
        auto storage = std::make_shared<WebGpuStorage>(bytes);
        if (!storage->handle()) return std::unexpected(web_gpu_error());
        return storage;
    }
    auto wrap_host(Device, std::span<const std::byte>, std::shared_ptr<const void>) const
        -> Result<std::shared_ptr<Storage>> override {
        return std::unexpected(Error{ErrorCode::UNSUPPORTED, "WebGPU cannot alias host memory"});
    }
    auto copy_from_host(Storage& destination, std::size_t offset, std::span<const std::byte> source) const
        -> Result<void> override {
        if (offset > destination.size_bytes() || source.size() > destination.size_bytes() - offset)
            return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "WebGPU upload outside storage"});
        if (upload_buffer(static_cast<WebGpuStorage&>(destination).handle(), offset, source.data(), source.size()))
            return std::unexpected(web_gpu_error());
        return {};
    }
    auto copy_to_host(const Storage& source, std::size_t offset, std::span<std::byte> destination) const
        -> Result<void> override {
        if (offset > source.size_bytes() || destination.size() > source.size_bytes() - offset)
            return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "WebGPU read outside storage"});
        if (!wait_for_gpu)
            return std::unexpected(Error{ErrorCode::UNSUPPORTED,
                                         "WebGPU results reach the host only after JavaScript awaits submitted work"});
        if (queue_read(static_cast<int>(static_cast<const WebGpuStorage&>(source).handle()), offset,
                       destination.data(), destination.size()) ||
            finish_work() || wait_for_gpu())
            return std::unexpected(web_gpu_error());
        return {};
    }
    auto host_view(Storage&) const -> Result<std::span<std::byte>> override {
        return std::unexpected(Error{ErrorCode::UNSUPPORTED, "WebGPU storage requires an explicit host transfer"});
    }
    auto host_view(const Storage&) const -> Result<std::span<const std::byte>> override {
        return std::unexpected(Error{ErrorCode::UNSUPPORTED, "WebGPU storage requires an explicit host transfer"});
    }
    auto synchronize(Device) const -> Result<void> override { return web_gpu_synchronize(); }
};
} // namespace
auto web_gpu_error() -> Error {
    std::array<char, 1024> message{};
    error_text(message.data(), message.size());
    return {ErrorCode::RUNTIME, message.data()};
}
auto web_gpu_synchronize() -> Result<void> {
    if (finish_work() || (wait_for_gpu && wait_for_gpu())) return std::unexpected(web_gpu_error());
    return {};
}
auto web_gpu_buffer(const Tensor& tensor) -> Result<WebGpuBufferView> {
    if (!tensor.defined() || tensor.device() != Device::web_gpu() || !tensor.is_contiguous())
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "expected a contiguous WebGPU tensor"});
    const auto& storage = static_cast<const WebGpuStorage&>(*tensor.storage_);
    return WebGpuBufferView{storage.handle(),
                            static_cast<std::size_t>(tensor.storage_offset()) * element_size(tensor.dtype()),
                            storage.packed_layout()};
}
auto adopt_web_gpu_buffer(std::uint32_t handle, std::vector<std::int64_t> shape, DType dtype,
                          std::int32_t packed_layout) -> Result<Tensor> {
    if (!handle || shape.empty()) return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "invalid WebGPU buffer"});
    auto backend = BackendRegistry::instance().backend(Device::web_gpu());
    if (!backend) {
        release_buffer(static_cast<int>(handle));
        return std::unexpected(std::move(backend.error()));
    }
    std::size_t bytes = element_size(dtype);
    detail::Dimensions strides(shape.size());
    for (std::size_t axis = shape.size(); axis-- > 0;) {
        strides[axis] = static_cast<std::int64_t>(bytes / element_size(dtype));
        bytes *= static_cast<std::size_t>(shape[axis]);
    }
    return Tensor(std::move(*backend), std::make_shared<WebGpuStorage>(bytes, handle, packed_layout), dtype,
                  std::move(shape), std::move(strides), 0);
}
auto set_web_gpu_wait(WebGpuWait wait) -> void { wait_for_gpu = wait; }
auto make_web_gpu_backend() -> std::shared_ptr<Backend> { return std::make_shared<WebGpuBackend>(); }
} // namespace kidi::tensor