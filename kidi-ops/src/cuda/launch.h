#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "cuda/runtime.h"

namespace kidi::cuda {

inline void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(std::string("CUDA: ") + cudaGetErrorString(status));
}

class DeviceScope {
public:
    explicit DeviceScope(int device) {
        check(cudaGetDevice(&previous_));
        changed_ = device != previous_;
        if (changed_) check(cudaSetDevice(device));
    }
    ~DeviceScope() {
        if (changed_) cudaSetDevice(previous_);
    }
    DeviceScope(const DeviceScope&) = delete;
    DeviceScope& operator=(const DeviceScope&) = delete;

private:
    int previous_ = 0;
    bool changed_ = false;
};

// Opts `Kernel` into `bytes` of dynamic shared memory (needed above 48 KiB) on the current device.
// The attribute is per device, so it is cached per device bit. It is skipped while `stream` is capturing.
// Requests above the device limit (about 99 KiB on sm_86/sm_89) fail without a runtime call: callers may fall back
// to smaller kernels, and a failed cudaFuncSetAttribute would leave its error for the next launch check.
template <auto Kernel>
cudaError_t allow_shared_memory(int bytes, cudaStream_t stream) {
    static thread_local std::uint64_t configured = 0, too_large = 0;
    int device = 0;
    if (const auto status = cudaGetDevice(&device); status != cudaSuccess) return status;
    const std::uint64_t bit = device >= 0 && device < 64 ? std::uint64_t{1} << device : 0;
    if (bit & configured) return cudaSuccess;
    if (bit & too_large) return cudaErrorInvalidValue;
    int limit = 0;
    if (const auto status = cudaDeviceGetAttribute(&limit, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
        status != cudaSuccess)
        return status;
    if (bytes > limit) {
        too_large |= bit;
        return cudaErrorInvalidValue;
    }
    cudaStreamCaptureStatus capturing = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(stream, &capturing) != cudaSuccess || capturing != cudaStreamCaptureStatusNone)
        return cudaSuccess;
    // The C API (rather than the nvcc-only template overload) keeps this usable from host-compiled code such as tests.
    const auto status =
        cudaFuncSetAttribute(reinterpret_cast<const void*>(Kernel), cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
    if (status == cudaSuccess) configured |= bit;
    return status;
}

struct DenseDevice {
    int major, multiprocessors;
};

const DenseDevice& dense_device(int device);

struct View {
    void* data;
    int type;
    int rank;
    std::int64_t shape[8];
    void* const* binding = nullptr;
    std::size_t binding_offset = 0;
};

struct Launch {
    int operation;
    View inputs[MAX_INPUTS];
    View output;
    int count;
    int attribute_count;
    std::int64_t attributes[32];
    float epsilon;
    std::size_t elements;
};

#if defined(__CUDACC__)
__device__ inline void* address(const View& tensor) {
    return tensor.binding ? static_cast<unsigned char*>(*tensor.binding) + tensor.binding_offset : tensor.data;
}

__device__ inline float read(const View& tensor, std::size_t index) {
    const auto* data = address(tensor);
    switch (static_cast<Type>(tensor.type)) {
        case Type::BOOL:
        case Type::U8:
            return static_cast<const unsigned char*>(data)[index];
        case Type::I8:
            return static_cast<const signed char*>(data)[index];
        case Type::I32:
            return static_cast<const int*>(data)[index];
        case Type::I64:
            return static_cast<float>(static_cast<const long long*>(data)[index]);
        case Type::F16:
            return __half2float(static_cast<const __half*>(data)[index]);
        case Type::BF16:
            return __bfloat162float(static_cast<const __nv_bfloat16*>(data)[index]);
        case Type::F32:
            return static_cast<const float*>(data)[index];
        case Type::F64:
            return static_cast<float>(static_cast<const double*>(data)[index]);
        default:
            return nanf("");
    }
}

__device__ inline void write(const View& tensor, std::size_t index, float value) {
    auto* data = address(tensor);
    switch (static_cast<Type>(tensor.type)) {
        case Type::BOOL:
            static_cast<unsigned char*>(data)[index] = value != 0;
            break;
        case Type::U8:
            static_cast<unsigned char*>(data)[index] =
                static_cast<unsigned char>(fminf(255, fmaxf(0, nearbyintf(value))));
            break;
        case Type::I8:
            static_cast<signed char*>(data)[index] =
                static_cast<signed char>(fminf(127, fmaxf(-128, nearbyintf(value))));
            break;
        case Type::I32:
            static_cast<int*>(data)[index] = static_cast<int>(value);
            break;
        case Type::I64:
            static_cast<long long*>(data)[index] = static_cast<long long>(value);
            break;
        case Type::F16:
            static_cast<__half*>(data)[index] = __float2half(value);
            break;
        case Type::BF16:
            static_cast<__nv_bfloat16*>(data)[index] = __float2bfloat16(value);
            break;
        case Type::F32:
            static_cast<float*>(data)[index] = value;
            break;
        case Type::F64:
            static_cast<double*>(data)[index] = value;
            break;
        default:
            break;
    }
}

__device__ inline std::size_t broadcast_index(std::size_t index, const View& output, const View& input) {
    std::size_t offset = 0, stride = 1;
    for (int axis = output.rank - 1; axis >= 0; --axis) {
        const auto coordinate = index % output.shape[axis];
        index /= output.shape[axis];
        const int input_axis = axis - output.rank + input.rank;
        if (input_axis >= 0) {
            if (input.shape[input_axis] != 1) offset += coordinate * stride;
            stride *= input.shape[input_axis];
        }
    }
    return offset;
}
#endif

} // namespace kidi::cuda
