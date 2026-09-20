#pragma once

#include <cstdint>
#include <string>
#include <string_view>

namespace kidi::tensor {

enum class DeviceKind {
    CPU,
    Q_NPU,
    CUDA,
    A_GPU,
};

struct Device {
    DeviceKind kind = DeviceKind::CPU;
    std::int32_t index = 0;

    static constexpr auto cpu(std::int32_t index = 0) noexcept -> Device { return {DeviceKind::CPU, index}; }

    static constexpr auto qualcomm_npu(std::int32_t index = 0) noexcept -> Device { return {DeviceKind::Q_NPU, index}; }

    static constexpr auto cuda(std::int32_t index = 0) noexcept -> Device { return {DeviceKind::CUDA, index}; }

    static constexpr auto apple_gpu(std::int32_t index = 0) noexcept -> Device { return {DeviceKind::A_GPU, index}; }

    friend auto operator==(const Device&, const Device&) -> bool = default;
};

constexpr auto to_string(DeviceKind kind) noexcept -> std::string_view {
    switch (kind) {
        case DeviceKind::CPU:
            return "cpu";
        case DeviceKind::Q_NPU:
            return "q_npu";
        case DeviceKind::CUDA:
            return "cuda";
        case DeviceKind::A_GPU:
            return "a_gpu";
    }
}

inline auto to_string(Device device) -> std::string {
    return std::string(to_string(device.kind)) + ':' + std::to_string(device.index);
}

} // namespace kidi::tensor