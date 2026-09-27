#pragma once

#include <array>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <utility>

namespace kidi::tensor {

enum class DeviceKind {
    CPU,
    Q_NPU,
    CUDA,
    A_GPU,
    WEB_GPU,
    COUNT,
};

struct DeviceCapabilities {
    bool calibrated_int8_cast = false;
    bool blockwise_int8_attention = false;
    std::size_t blockwise_int8_attention_max_tokens = 0;
};

template <typename Key, typename Value, std::size_t Size>
class EnumMap {
public:
    consteval explicit EnumMap(std::array<std::pair<Key, Value>, Size> entries) : entries_(std::move(entries)) {
        std::array<bool, Size> seen{};
        for (const auto& [key, value] : entries_) {
            const auto index = static_cast<std::size_t>(key);
            if (index >= Size || seen[index]) throw "invalid enum map";
            seen[index] = true;
        }
    }

    constexpr auto operator[](Key key) const -> const Value& {
        for (const auto& [candidate, value] : entries_)
            if (candidate == key) return value;
        throw "enum map key not found";
    }

private:
    std::array<std::pair<Key, Value>, Size> entries_;
};

inline constexpr EnumMap<DeviceKind, DeviceCapabilities, static_cast<std::size_t>(DeviceKind::COUNT)>
    DEVICE_CAPABILITIES{std::array{
        std::pair{DeviceKind::CPU,
              DeviceCapabilities{.calibrated_int8_cast = true,
                         .blockwise_int8_attention = true,
                         .blockwise_int8_attention_max_tokens = 512}},
        std::pair{DeviceKind::Q_NPU, DeviceCapabilities{}},
        std::pair{DeviceKind::CUDA, DeviceCapabilities{}},
        std::pair{DeviceKind::A_GPU, DeviceCapabilities{.calibrated_int8_cast = true}},
        std::pair{DeviceKind::WEB_GPU,
                  DeviceCapabilities{.calibrated_int8_cast = true,
                                     .blockwise_int8_attention = true,
                                     .blockwise_int8_attention_max_tokens = std::numeric_limits<std::size_t>::max()}},
    }};

struct Device {
    DeviceKind kind = DeviceKind::CPU;
    std::int32_t index = 0;

    static constexpr auto cpu(std::int32_t index = 0) noexcept -> Device { return {DeviceKind::CPU, index}; }

    static constexpr auto qualcomm_npu(std::int32_t index = 0) noexcept -> Device { return {DeviceKind::Q_NPU, index}; }

    static constexpr auto cuda(std::int32_t index = 0) noexcept -> Device { return {DeviceKind::CUDA, index}; }

    static constexpr auto apple_gpu(std::int32_t index = 0) noexcept -> Device { return {DeviceKind::A_GPU, index}; }

    static constexpr auto web_gpu(std::int32_t index = 0) noexcept -> Device { return {DeviceKind::WEB_GPU, index}; }

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
        case DeviceKind::WEB_GPU:
            return "web_gpu";
        case DeviceKind::COUNT:
            return "unknown";
    }
}

inline auto to_string(Device device) -> std::string {
    return std::string(to_string(device.kind)) + ':' + std::to_string(device.index);
}

} // namespace kidi::tensor