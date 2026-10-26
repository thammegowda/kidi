#include "kidi/tensor/cuda.h"
#include "ynnpack/include/ynnpack_cuda.h"
#include <cuda_runtime_api.h>
#include <atomic>
#include <mutex>
#include <stdexcept>

namespace kidi::tensor {
namespace {
class CudaStorage final : public Storage {
public:
    CudaStorage(Device device, std::size_t bytes)
        : data_(ynn::cuda::allocate(bytes, device.index)), bytes_(bytes), device_(device) {}
    ~CudaStorage() override { ynn::cuda::release(data_); }
    auto device() const noexcept -> Device override { return device_; }
    auto size_bytes() const noexcept -> std::size_t override { return bytes_; }
    auto data() const noexcept -> std::byte* { return static_cast<std::byte*>(data_); }

private:
    void* data_;
    std::size_t bytes_;
    Device device_;
};
auto error(std::string message) { return std::unexpected(Error{ErrorCode::RUNTIME, std::move(message)}); }

auto check_cuda(cudaError_t status) -> void {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
class CudaDeviceScope {
public:
    explicit CudaDeviceScope(Device device) {
        check_cuda(cudaGetDevice(&previous_));
        changed_ = previous_ != device.index;
        if (changed_) check_cuda(cudaSetDevice(device.index));
    }
    ~CudaDeviceScope() {
        if (changed_) cudaSetDevice(previous_);
    }

private:
    int previous_ = 0;
    bool changed_ = false;
};

class CudaReadback final : public Readback {
public:
    CudaReadback(Device device, std::size_t bytes) : device_(device), capacity_(std::max(bytes, std::size_t{1})) {
        const CudaDeviceScope selected(device_);
        check_cuda(cudaHostAlloc(&data_, capacity_, cudaHostAllocDefault));
        const auto status = cudaEventCreateWithFlags(&done_, cudaEventDisableTiming);
        if (status != cudaSuccess) {
            cudaFreeHost(data_);
            check_cuda(status);
        }
    }
    ~CudaReadback() override {
        int previous = 0;
        if (cudaGetDevice(&previous) != cudaSuccess || cudaSetDevice(device_.index) != cudaSuccess) return;
        cudaEventSynchronize(done_);
        cudaEventDestroy(done_);
        cudaFreeHost(data_);
        cudaSetDevice(previous);
    }
    auto accepts(Device device, std::size_t bytes) const -> bool { return device == device_ && bytes <= capacity_; }
    auto submit(std::shared_ptr<const Storage> source, std::size_t offset, std::size_t bytes) -> void {
        const CudaDeviceScope selected(device_);
        const auto& storage = static_cast<const CudaStorage&>(*source);
        source_ = std::move(source);
        size_ = bytes;
        completed_.store(false, std::memory_order_relaxed);
        if (bytes) check_cuda(cudaMemcpyAsync(data_, storage.data() + offset, bytes, cudaMemcpyDeviceToHost, nullptr));
        check_cuda(cudaEventRecord(done_, nullptr));
    }
    auto ready() const -> Result<bool> override {
        if (completed_.load(std::memory_order_acquire)) return true;
        try {
            const CudaDeviceScope selected(device_);
            const auto status = cudaEventQuery(done_);
            if (status == cudaErrorNotReady) return false;
            check_cuda(status);
            if (!completed_.exchange(true, std::memory_order_acq_rel)) source_.reset();
            return true;
        } catch (const std::exception& exception) {
            return error(exception.what());
        }
    }
    auto bytes() const -> Result<std::span<const std::byte>> override {
        const auto completed = ready();
        if (!completed) return std::unexpected(completed.error());
        if (!*completed) return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "readback is not complete"});
        return std::span<const std::byte>(static_cast<const std::byte*>(data_), size_);
    }

private:
    Device device_;
    void* data_ = nullptr;
    cudaEvent_t done_ = nullptr;
    std::size_t capacity_, size_ = 0;
    mutable std::shared_ptr<const Storage> source_;
    mutable std::atomic<bool> completed_{false};
};

class CudaBackend final : public Backend {
public:
    auto name() const noexcept -> std::string_view override { return "cuda"; }
    auto device_name(Device device) const -> std::string override {
        return is_available(device) ? ynn::cuda::device_name(device.index) : "CUDA unavailable";
    }
    auto device_kind() const noexcept -> DeviceKind override { return DeviceKind::CUDA; }
    auto is_available(Device device) const noexcept -> bool override {
        return device.kind == DeviceKind::CUDA && ynn::cuda::available(device.index);
    }
    auto is_host_accessible(Device) const noexcept -> bool override { return false; }
    auto supports_execution() const noexcept -> bool override { return true; }
    auto unavailable_reason(Device device) const -> std::string override {
        return is_available(device) ? "" : "CUDA device " + to_string(device) + " is unavailable";
    }
    auto allocate(Device device, std::size_t bytes, std::size_t) const -> Result<std::shared_ptr<Storage>> override {
        if (!is_available(device)) return std::unexpected(Error{ErrorCode::UNSUPPORTED, unavailable_reason(device)});
        try {
            return std::shared_ptr<Storage>(std::make_shared<CudaStorage>(device, bytes));
        } catch (const std::exception& exception) {
            return error(exception.what());
        }
    }
    auto wrap_host(Device, std::span<const std::byte>, std::shared_ptr<const void>) const
        -> Result<std::shared_ptr<Storage>> override {
        return std::unexpected(Error{ErrorCode::UNSUPPORTED, "CUDA storage cannot wrap host memory"});
    }
    auto clear(Storage& storage) const -> Result<void> override {
        auto* cuda = dynamic_cast<CudaStorage*>(&storage);
        if (!cuda) return error("invalid CUDA storage");
        try {
            ynn::cuda::clear(cuda->data(), cuda->size_bytes(), cuda->device().index);
            return {};
        } catch (const std::exception& exception) {
            return error(exception.what());
        }
    }
    auto copy_from_host(Storage& destination, std::size_t offset, std::span<const std::byte> source) const
        -> Result<void> override {
        auto* cuda = dynamic_cast<CudaStorage*>(&destination);
        if (!cuda || offset > cuda->size_bytes() || source.size() > cuda->size_bytes() - offset)
            return error("CUDA copy is out of bounds");
        try {
            ynn::cuda::copy_to_device(cuda->data() + offset, source.data(), source.size(), cuda->device().index);
            return {};
        } catch (const std::exception& exception) {
            return error(exception.what());
        }
    }
    auto copy_to_host(const Storage& source, std::size_t offset, std::span<std::byte> destination) const
        -> Result<void> override {
        const auto* cuda = dynamic_cast<const CudaStorage*>(&source);
        if (!cuda || offset > cuda->size_bytes() || destination.size() > cuda->size_bytes() - offset)
            return error("CUDA copy is out of bounds");
        try {
            ynn::cuda::copy_to_host(destination.data(), cuda->data() + offset, destination.size(),
                                    cuda->device().index);
            return {};
        } catch (const std::exception& exception) {
            return error(exception.what());
        }
    }
    auto host_view(Storage&) const -> Result<std::span<std::byte>> override {
        return error("CUDA storage is not host accessible; copy it to CPU explicitly");
    }
    auto host_view(const Storage&) const -> Result<std::span<const std::byte>> override {
        return error("CUDA storage is not host accessible; copy it to CPU explicitly");
    }
    auto readback(std::shared_ptr<const Storage> source, std::size_t offset, std::size_t bytes) const
        -> Result<std::shared_ptr<Readback>> override {
        const auto* storage = dynamic_cast<const CudaStorage*>(source.get());
        if (!storage || offset > storage->size_bytes() || bytes > storage->size_bytes() - offset)
            return error("CUDA readback is out of bounds");
        try {
            std::scoped_lock lock(readback_mutex_);
            for (auto& cached : readbacks_) {
                if (cached.use_count() != 1 || !cached->accepts(storage->device(), bytes)) continue;
                const auto completed = cached->ready();
                if (!completed) return std::unexpected(completed.error());
                if (!*completed) continue;
                cached->submit(std::move(source), offset, bytes);
                return std::shared_ptr<Readback>(cached);
            }
            auto result = std::make_shared<CudaReadback>(storage->device(), bytes);
            result->submit(std::move(source), offset, bytes);
            if (readbacks_.size() < 64 && bytes <= 65536) readbacks_.push_back(result);
            return std::shared_ptr<Readback>(std::move(result));
        } catch (const std::exception& exception) {
            return error(exception.what());
        }
    }
    auto synchronize(Device device) const -> Result<void> override {
        if (!is_available(device)) return error(unavailable_reason(device));
        try {
            ynn::cuda::synchronize(device.index);
            return {};
        } catch (const std::exception& exception) {
            return error(exception.what());
        }
    }

private:
    mutable std::mutex readback_mutex_;
    mutable std::vector<std::shared_ptr<CudaReadback>> readbacks_;
};
} // namespace
auto make_cuda_backend() -> std::shared_ptr<Backend> { return std::make_shared<CudaBackend>(); }
auto cuda_buffer(const Tensor& tensor) -> Result<CudaBufferView> {
    const auto* storage = dynamic_cast<const CudaStorage*>(tensor.storage_.get());
    if (!storage || !tensor.is_contiguous()) return error("CUDA execution requires a contiguous CUDA tensor");
    const auto offset = static_cast<std::size_t>(tensor.storage_offset()) * element_size(tensor.dtype());
    if (offset > storage->size_bytes() || tensor.nbytes() > storage->size_bytes() - offset)
        return error("CUDA tensor view is out of bounds");
    return CudaBufferView{storage->data() + offset, tensor.nbytes()};
}
} // namespace kidi::tensor