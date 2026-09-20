#include "kidi/tensor/backend.h"

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <utility>

#if defined(__unix__) || defined(__APPLE__)
#include <dlfcn.h>
#endif

namespace kidi::tensor {
namespace {

#if defined(__unix__) || defined(__APPLE__)

using CudaError = int;
constexpr CudaError CUDA_SUCCESS = 0;
constexpr int CUDA_MEMCPY_HOST_TO_DEVICE = 1;
constexpr int CUDA_MEMCPY_DEVICE_TO_HOST = 2;

template <std::size_t Size>
auto open_first(const std::array<const char*, Size>& names) -> void* {
    for (const auto* name : names) {
        if (auto* handle = dlopen(name, RTLD_LOCAL | RTLD_LAZY); handle != nullptr) return handle;
    }
    return nullptr;
}

template <typename Function>
auto load_symbol(void* library, const char* name, Function& function) -> bool {
    function = reinterpret_cast<Function>(dlsym(library, name));
    return function != nullptr;
}

class CudaRuntime {
public:
    using GetDeviceCount = CudaError (*)(int*);
    using SetDevice = CudaError (*)(int);
    using Malloc = CudaError (*)(void**, std::size_t);
    using Free = CudaError (*)(void*);
    using Memcpy = CudaError (*)(void*, const void*, std::size_t, int);
    using DeviceSynchronize = CudaError (*)();
    using GetErrorString = const char* (*)(CudaError);
    using CudnnGetVersion = std::size_t (*)();

    CudaRuntime() {
        constexpr std::array CUDART_NAMES = {"libcudart.so", "libcudart.so.13", "libcudart.so.12", "libcudart.so.11.0"};
        constexpr std::array CUDNN_NAMES = {"libcudnn.so", "libcudnn.so.9", "libcudnn.so.8"};
        cudart_ = open_first(CUDART_NAMES);
        cudnn_ = open_first(CUDNN_NAMES);
        if (cudart_ == nullptr || cudnn_ == nullptr) {
            reason_ = "CUDA Runtime and cuDNN shared libraries were not found";
            return;
        }
        const bool loaded = load_symbol(cudart_, "cudaGetDeviceCount", get_device_count) &&
                            load_symbol(cudart_, "cudaSetDevice", set_device) &&
                            load_symbol(cudart_, "cudaMalloc", malloc) && load_symbol(cudart_, "cudaFree", free) &&
                            load_symbol(cudart_, "cudaMemcpy", memcpy) &&
                            load_symbol(cudart_, "cudaDeviceSynchronize", device_synchronize) &&
                            load_symbol(cudart_, "cudaGetErrorString", get_error_string) &&
                            load_symbol(cudnn_, "cudnnGetVersion", cudnn_get_version);
        if (!loaded) {
            reason_ = "CUDA Runtime or cuDNN is missing required symbols";
            return;
        }
        const auto status = get_device_count(&device_count_);
        if (status != CUDA_SUCCESS) {
            reason_ = "CUDA device discovery failed: " + error_message(status);
            device_count_ = 0;
            return;
        }
        if (cudnn_get_version() == 0) {
            reason_ = "cuDNN reported an invalid version";
            device_count_ = 0;
        }
    }

    ~CudaRuntime() {
        if (cudnn_ != nullptr) dlclose(cudnn_);
        if (cudart_ != nullptr) dlclose(cudart_);
    }

    CudaRuntime(const CudaRuntime&) = delete;
    auto operator=(const CudaRuntime&) -> CudaRuntime& = delete;

    auto available(std::int32_t index) const noexcept -> bool {
        return reason_.empty() && index >= 0 && index < device_count_;
    }

    auto reason(std::int32_t index) const -> std::string {
        if (!reason_.empty()) return reason_;
        if (index < 0 || index >= device_count_) return "CUDA device index is out of range";
        return {};
    }

    auto error_message(CudaError error) const -> std::string {
        if (get_error_string == nullptr) return "error " + std::to_string(error);
        const auto* message = get_error_string(error);
        return message == nullptr ? "error " + std::to_string(error) : std::string(message);
    }

    GetDeviceCount get_device_count = nullptr;
    SetDevice set_device = nullptr;
    Malloc malloc = nullptr;
    Free free = nullptr;
    Memcpy memcpy = nullptr;
    DeviceSynchronize device_synchronize = nullptr;
    GetErrorString get_error_string = nullptr;
    CudnnGetVersion cudnn_get_version = nullptr;

private:
    void* cudart_ = nullptr;
    void* cudnn_ = nullptr;
    int device_count_ = 0;
    std::string reason_;
};

class CudaStorage final : public Storage {
public:
    CudaStorage(Device device, std::size_t size_bytes, void* data, std::shared_ptr<CudaRuntime> runtime)
        : device_(device), size_bytes_(size_bytes), data_(data), runtime_(std::move(runtime)) {}

    ~CudaStorage() override {
        if (data_ != nullptr) {
            runtime_->set_device(device_.index);
            runtime_->free(data_);
        }
    }

    auto device() const noexcept -> Device override { return device_; }
    auto size_bytes() const noexcept -> std::size_t override { return size_bytes_; }
    auto data() const noexcept -> void* { return data_; }

private:
    Device device_;
    std::size_t size_bytes_;
    void* data_;
    std::shared_ptr<CudaRuntime> runtime_;
};

auto cuda_storage(Storage& storage) -> Result<CudaStorage*> {
    auto* result = dynamic_cast<CudaStorage*>(&storage);
    if (result == nullptr) {
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "storage does not belong to CUDA backend"});
    }
    return result;
}

auto cuda_storage(const Storage& storage) -> Result<const CudaStorage*> {
    const auto* result = dynamic_cast<const CudaStorage*>(&storage);
    if (result == nullptr) {
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "storage does not belong to CUDA backend"});
    }
    return result;
}

auto check_copy_bounds(std::size_t storage_size, std::size_t offset, std::size_t copy_size) -> Result<void> {
    if (offset > storage_size || copy_size > storage_size - offset) {
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "CUDA tensor storage copy is out of bounds"});
    }
    return {};
}

class CudaBackend final : public Backend {
public:
    CudaBackend() : runtime_(std::make_shared<CudaRuntime>()) {}

    auto name() const noexcept -> std::string_view override { return "cuda-cudnn"; }
    auto device_kind() const noexcept -> DeviceKind override { return DeviceKind::CUDA; }
    auto is_available(Device device) const noexcept -> bool override {
        return device.kind == DeviceKind::CUDA && runtime_->available(device.index);
    }
    auto is_host_accessible(Device) const noexcept -> bool override { return false; }
    auto supports_execution() const noexcept -> bool override { return false; }
    auto unavailable_reason(Device device) const -> std::string override {
        if (device.kind != DeviceKind::CUDA) return "CUDA backend requires a CUDA device";
        return runtime_->reason(device.index);
    }

    auto allocate(Device device, std::size_t size_bytes, std::size_t) const
        -> Result<std::shared_ptr<Storage>> override {
        if (!is_available(device)) {
            return std::unexpected(Error{ErrorCode::UNSUPPORTED, unavailable_reason(device)});
        }
        auto status = runtime_->set_device(device.index);
        if (status != CUDA_SUCCESS) return std::unexpected(cuda_error("select CUDA device", status));
        void* data = nullptr;
        status = runtime_->malloc(&data, std::max<std::size_t>(size_bytes, 1));
        if (status != CUDA_SUCCESS) return std::unexpected(cuda_error("allocate CUDA tensor", status));
        return std::shared_ptr<Storage>(new CudaStorage(device, size_bytes, data, runtime_));
    }

    auto wrap_host(Device, std::span<const std::byte>, std::shared_ptr<const void>) const
        -> Result<std::shared_ptr<Storage>> override {
        return std::unexpected(
            Error{ErrorCode::UNSUPPORTED, "zero-copy host blobs are supported only by the CPU backend"});
    }

    auto copy_from_host(Storage& destination, std::size_t destination_offset, std::span<const std::byte> source) const
        -> Result<void> override {
        auto storage = cuda_storage(destination);
        if (!storage) return std::unexpected(std::move(storage.error()));
        auto bounds = check_copy_bounds((*storage)->size_bytes(), destination_offset, source.size());
        if (!bounds) return bounds;
        auto status = runtime_->set_device((*storage)->device().index);
        if (status != CUDA_SUCCESS) return std::unexpected(cuda_error("select CUDA device", status));
        status = runtime_->memcpy(static_cast<std::byte*>((*storage)->data()) + destination_offset, source.data(),
                                  source.size(), CUDA_MEMCPY_HOST_TO_DEVICE);
        if (status != CUDA_SUCCESS) return std::unexpected(cuda_error("copy tensor to CUDA", status));
        return {};
    }

    auto copy_to_host(const Storage& source, std::size_t source_offset, std::span<std::byte> destination) const
        -> Result<void> override {
        auto storage = cuda_storage(source);
        if (!storage) return std::unexpected(std::move(storage.error()));
        auto bounds = check_copy_bounds((*storage)->size_bytes(), source_offset, destination.size());
        if (!bounds) return bounds;
        auto status = runtime_->set_device((*storage)->device().index);
        if (status != CUDA_SUCCESS) return std::unexpected(cuda_error("select CUDA device", status));
        status = runtime_->memcpy(destination.data(), static_cast<const std::byte*>((*storage)->data()) + source_offset,
                                  destination.size(), CUDA_MEMCPY_DEVICE_TO_HOST);
        if (status != CUDA_SUCCESS) return std::unexpected(cuda_error("copy tensor from CUDA", status));
        return {};
    }

    auto host_view(Storage&) const -> Result<std::span<std::byte>> override {
        return std::unexpected(Error{ErrorCode::UNSUPPORTED, "CUDA tensor storage is not host-accessible"});
    }

    auto host_view(const Storage&) const -> Result<std::span<const std::byte>> override {
        return std::unexpected(Error{ErrorCode::UNSUPPORTED, "CUDA tensor storage is not host-accessible"});
    }

    auto synchronize(Device device) const -> Result<void> override {
        if (!is_available(device)) {
            return std::unexpected(Error{ErrorCode::UNSUPPORTED, unavailable_reason(device)});
        }
        auto status = runtime_->set_device(device.index);
        if (status != CUDA_SUCCESS) return std::unexpected(cuda_error("select CUDA device", status));
        status = runtime_->device_synchronize();
        if (status != CUDA_SUCCESS) return std::unexpected(cuda_error("synchronize CUDA device", status));
        return {};
    }

private:
    auto cuda_error(std::string_view operation, CudaError status) const -> Error {
        return Error{ErrorCode::RUNTIME, std::string(operation) + " failed: " + runtime_->error_message(status)};
    }

    std::shared_ptr<CudaRuntime> runtime_;
};

#endif

} // namespace

auto make_cuda_backend() -> std::shared_ptr<Backend> {
#if defined(__unix__) || defined(__APPLE__)
    return std::make_shared<CudaBackend>();
#else
    return make_unavailable_backend(DeviceKind::CUDA, "cuda-cudnn",
                                    "dynamic CUDA loading is not supported on this platform");
#endif
}

} // namespace kidi::tensor