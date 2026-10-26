#pragma once

#include <array>
#include <charconv>
#include <cstdint>
#include <limits>
#include <optional>
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
    VULKAN,
    COUNT,
};

enum class TokenSelection { HOST, DEVICE, BATCHED_DEVICE, CHAINED_DEVICE };

struct SamplingCapabilities {
    TokenSelection tokens = TokenSelection::HOST;
    bool diffusion = false;

    constexpr auto supports(TokenSelection selection) const noexcept -> bool {
        if (selection == TokenSelection::HOST) return true;
        if (selection == TokenSelection::DEVICE) return tokens != TokenSelection::HOST;
        return tokens == selection;
    }
};

enum class ExecutionDomain { HOST, DEVICE };
enum class TransformerSupport { NONE, QUANTIZED, GENERAL };
enum class AudioExecution { NONE, LOWERED, NATIVE };
enum class HostAccess { DIRECT, EXPLICIT_TRANSFER, DEFERRED_TRANSFER };
enum class AttentionExecution { STANDARD, FUSED };

struct ComputeCapabilities {
    ExecutionDomain execution = ExecutionDomain::DEVICE;
    TransformerSupport transformer = TransformerSupport::NONE;
    AudioExecution audio = AudioExecution::NONE;
    bool embedding = false;

    constexpr auto host() const noexcept -> bool { return execution == ExecutionDomain::HOST; }
    constexpr auto supports(TransformerSupport required) const noexcept -> bool { return transformer >= required; }
    constexpr auto native_audio() const noexcept -> bool { return audio == AudioExecution::NATIVE; }
};

struct AttentionCapabilities {
    std::size_t int8_cache_tokens = 0;
    AttentionExecution execution = AttentionExecution::STANDARD;

    constexpr auto fused() const noexcept -> bool { return execution == AttentionExecution::FUSED; }
};

struct MemoryCapabilities {
    HostAccess access = HostAccess::EXPLICIT_TRANSFER;
    bool offset_views = false;

    constexpr auto deferred() const noexcept -> bool { return access == HostAccess::DEFERRED_TRANSFER; }
    constexpr auto requires_staging() const noexcept -> bool { return access == HostAccess::EXPLICIT_TRANSFER; }
};

struct DeviceCapabilities {
    ComputeCapabilities compute{};
    SamplingCapabilities sampling{};
    AttentionCapabilities attention{};
    MemoryCapabilities memory{};

    constexpr auto supports_audio_transformer() const noexcept -> bool {
        return compute.supports(TransformerSupport::QUANTIZED) && compute.embedding &&
               compute.audio != AudioExecution::NONE;
    }
};

enum class ParameterResidency { DEVICE, HOST, HOST_QUANTIZED };
enum class ParameterRole { PROJECTION, EMBEDDING, NORMALIZATION, AUDIO };
enum class ParameterKind { FLOAT, QUANTIZED, PREPARED_VECTOR };

struct ParameterPolicy {
    ParameterResidency projections = ParameterResidency::DEVICE;
    ParameterResidency embeddings = ParameterResidency::DEVICE;
    ParameterResidency normalization = ParameterResidency::DEVICE;
    ParameterResidency audio = ParameterResidency::DEVICE;
    ParameterResidency vectors = ParameterResidency::DEVICE;

    constexpr auto on_host(ParameterRole role, ParameterKind kind) const noexcept -> bool {
        auto residency = [&] {
            switch (role) {
                case ParameterRole::PROJECTION:
                    return projections;
                case ParameterRole::EMBEDDING:
                    return embeddings;
                case ParameterRole::NORMALIZATION:
                    return normalization;
                case ParameterRole::AUDIO:
                    return audio;
            }
            std::unreachable();
        }();
        if (kind == ParameterKind::PREPARED_VECTOR && vectors != ParameterResidency::DEVICE) residency = vectors;
        return residency == ParameterResidency::HOST ||
               (residency == ParameterResidency::HOST_QUANTIZED && kind != ParameterKind::FLOAT);
    }
};

enum class FeedForwardKernel { COMPOSED, PACKED_GATE, PACKED_SMALL, PACKED_LARGE, PACKED_INT8 };
enum class PrefillStrategy { FULL, LAST_TOKEN, SHARED_KV_TAIL };
enum class ParameterBinding { PREPARED, DYNAMIC_LINEAR, DYNAMIC };
enum class ProjectionDispatch { DEFAULT, VECTOR_DECODE };

struct TransformerPolicy {
    FeedForwardKernel gelu = FeedForwardKernel::COMPOSED;
    FeedForwardKernel silu = FeedForwardKernel::COMPOSED;
    PrefillStrategy prefill = PrefillStrategy::FULL;
    bool crop_history = false;
    std::size_t attention_query_rows = 512;
    std::size_t decode_ahead_tokens = 1;
};

struct ReplayPolicy {
    std::size_t region_layers = 0;
    std::size_t cache_capacity = 8;
    bool generator_opt_in = false;
};

struct DispatchPolicy {
    ParameterBinding parameters = ParameterBinding::PREPARED;
    ProjectionDispatch projection = ProjectionDispatch::DEFAULT;
    std::size_t cache_capacity = 4096;
};

struct DevicePolicy {
    ParameterPolicy parameters{};
    TransformerPolicy transformer{};
    ReplayPolicy replay{};
    DispatchPolicy dispatch{};
};

struct DeviceProfile {
    DeviceCapabilities capabilities{};
    DevicePolicy policy{};
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

#if defined(__EMSCRIPTEN__)
inline constexpr auto CPU_GELU_KERNEL = FeedForwardKernel::COMPOSED;
#else
inline constexpr auto CPU_GELU_KERNEL = FeedForwardKernel::PACKED_LARGE;
#endif

inline constexpr EnumMap<DeviceKind, DeviceProfile, static_cast<std::size_t>(DeviceKind::COUNT)> DEVICE_PROFILES{
    std::array{
        std::pair{DeviceKind::CPU,
                  DeviceProfile{.capabilities = {.compute = {ExecutionDomain::HOST, TransformerSupport::GENERAL,
                                                             AudioExecution::LOWERED, true},
                                                 .attention = {.int8_cache_tokens = 512},
                                                 .memory = {HostAccess::DIRECT, true}},
                                .policy = {.transformer = {.gelu = CPU_GELU_KERNEL,
                                                           .silu = FeedForwardKernel::PACKED_GATE,
                                                           .prefill = PrefillStrategy::LAST_TOKEN,
                                                           .crop_history = true},
                                           .replay = {.generator_opt_in = true}}}},
        std::pair{DeviceKind::Q_NPU, DeviceProfile{}},
        std::pair{DeviceKind::CUDA,
                  DeviceProfile{.capabilities = {.compute = {.transformer = TransformerSupport::QUANTIZED,
                                                             .audio = AudioExecution::NATIVE,
                                                             .embedding = true},
                                                 .sampling = {TokenSelection::CHAINED_DEVICE, true}},
                                .policy = {.parameters = {.audio = ParameterResidency::HOST},
                                           .transformer = {.gelu = FeedForwardKernel::PACKED_GATE,
                                                           .silu = FeedForwardKernel::PACKED_GATE,
                                                           .crop_history = true,
                                                           .decode_ahead_tokens = 16}}}},
        std::pair{DeviceKind::A_GPU,
                  DeviceProfile{.capabilities = {.compute = {.transformer = TransformerSupport::GENERAL,
                                                             .audio = AudioExecution::NATIVE},
                                                 .sampling = {TokenSelection::BATCHED_DEVICE},
                                                 .attention = {.execution = AttentionExecution::FUSED},
                                                 .memory = {.access = HostAccess::DIRECT}},
                                .policy = {.parameters = {.projections = ParameterResidency::HOST_QUANTIZED,
                                                          .embeddings = ParameterResidency::HOST,
                                                          .audio = ParameterResidency::HOST},
                                           .transformer = {.gelu = FeedForwardKernel::PACKED_INT8,
                                                           .silu = FeedForwardKernel::PACKED_GATE,
                                                           .prefill = PrefillStrategy::SHARED_KV_TAIL},
                                           .dispatch = {.parameters = ParameterBinding::DYNAMIC_LINEAR,
                                                        .projection = ProjectionDispatch::VECTOR_DECODE}}}},
        std::pair{
            DeviceKind::WEB_GPU,
            DeviceProfile{.capabilities = {.compute = {.transformer = TransformerSupport::QUANTIZED,
                                                       .audio = AudioExecution::NATIVE,
                                                       .embedding = true},
                                           .sampling = {TokenSelection::DEVICE, true},
                                           .attention = {.int8_cache_tokens = std::numeric_limits<std::size_t>::max()},
                                           .memory = {HostAccess::DEFERRED_TRANSFER, true}},
                          .policy = {.parameters = {.vectors = ParameterResidency::HOST},
                                     .transformer = {.gelu = FeedForwardKernel::PACKED_SMALL,
                                                     .crop_history = true,
                                                     .attention_query_rows = std::numeric_limits<std::int64_t>::max()},
                                     .replay = {.region_layers = 4, .cache_capacity = 4},
                                     .dispatch = {.parameters = ParameterBinding::DYNAMIC, .cache_capacity = 2048}}}},
        std::pair{
            DeviceKind::VULKAN,
            DeviceProfile{.capabilities = {.compute = {.transformer = TransformerSupport::QUANTIZED, .embedding = true},
                                           .sampling = {TokenSelection::DEVICE},
                                           .attention = {.int8_cache_tokens = 512},
                                           .memory = {HostAccess::DIRECT, true}},
                          .policy = {.parameters = {.projections = ParameterResidency::HOST,
                                                    .embeddings = ParameterResidency::HOST,
                                                    .normalization = ParameterResidency::HOST},
                                     .transformer = {.gelu = FeedForwardKernel::PACKED_LARGE,
                                                     .silu = FeedForwardKernel::PACKED_LARGE,
                                                     .prefill = PrefillStrategy::LAST_TOKEN,
                                                     .crop_history = true},
                                     .replay = {.generator_opt_in = true}}}},
    }};

struct Device {
    DeviceKind kind = DeviceKind::CPU;
    std::int32_t index = 0;

    constexpr auto capabilities() const -> const DeviceCapabilities& { return DEVICE_PROFILES[kind].capabilities; }
    constexpr auto policy() const -> const DevicePolicy& { return DEVICE_PROFILES[kind].policy; }
    constexpr auto parameter_device(ParameterRole role, ParameterKind parameter_kind = ParameterKind::FLOAT) const
        -> Device {
        return policy().parameters.on_host(role, parameter_kind) ? Device::cpu() : *this;
    }

    static constexpr auto cpu(std::int32_t index = 0) noexcept -> Device { return {DeviceKind::CPU, index}; }

    static constexpr auto qualcomm_npu(std::int32_t index = 0) noexcept -> Device { return {DeviceKind::Q_NPU, index}; }

    static constexpr auto cuda(std::int32_t index = 0) noexcept -> Device { return {DeviceKind::CUDA, index}; }

    static constexpr auto apple_gpu(std::int32_t index = 0) noexcept -> Device { return {DeviceKind::A_GPU, index}; }

    static constexpr auto web_gpu(std::int32_t index = 0) noexcept -> Device { return {DeviceKind::WEB_GPU, index}; }

    static constexpr auto vulkan(std::int32_t index = 0) noexcept -> Device { return {DeviceKind::VULKAN, index}; }

    friend auto operator==(const Device&, const Device&) -> bool = default;
};

inline auto parse_cuda_device(std::string_view value) -> std::optional<Device> {
    if (value == "cuda") return Device::cuda();
    if (!value.starts_with("cuda:")) return std::nullopt;
    value.remove_prefix(5);
    if (value.empty() || value.front() < '0' || value.front() > '9') return std::nullopt;
    std::int32_t index = 0;
    const auto [end, status] = std::from_chars(value.data(), value.data() + value.size(), index);
    if (status != std::errc{} || end != value.data() + value.size() || index < 0) return std::nullopt;
    return Device::cuda(index);
}

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
        case DeviceKind::VULKAN:
            return "vulkan";
        case DeviceKind::COUNT:
            return "unknown";
    }
    std::unreachable();
}

inline auto to_string(Device device) -> std::string {
    return std::string(to_string(device.kind)) + ':' + std::to_string(device.index);
}

} // namespace kidi::tensor