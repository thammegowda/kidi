#pragma once

#include <cstdint>
#include <optional>
#include <string_view>

namespace kidi::core {

/// Compute policy for packed checkpoint projections. CHECKPOINT preserves each layer's trained weight bit width.
enum class InferencePrecision {
    CHECKPOINT,
    LOWBIT_PARITY,
    FP32,
    BF16,
    Q2A16,
    Q4A16,
    Q8A16,
    Q2AE4M3,
    Q4AE4M3,
    Q8AE4M3,
    Q2AE5M2,
    Q4AE5M2,
    Q8AE5M2,
    QAT_FP32,
};

/// Storage policy for language-model keys and values. AUTO selects a calibrated INT8 cache when supported.
enum class KVCachePrecision {
    AUTO,
    FP32,
    BF16,
    INT8,
    E4M3,
    E5M2,
};

constexpr auto to_string(InferencePrecision precision) noexcept -> std::string_view {
    switch (precision) {
        case InferencePrecision::CHECKPOINT:
            return "checkpoint";
        case InferencePrecision::LOWBIT_PARITY:
            return "lowbit-parity";
        case InferencePrecision::FP32:
            return "fp32";
        case InferencePrecision::BF16:
            return "bf16";
        case InferencePrecision::Q2A16:
            return "q2a16";
        case InferencePrecision::Q4A16:
            return "q4a16";
        case InferencePrecision::Q8A16:
            return "q8a16";
        case InferencePrecision::Q2AE4M3:
            return "q2ae4m3";
        case InferencePrecision::Q4AE4M3:
            return "q4ae4m3";
        case InferencePrecision::Q8AE4M3:
            return "q8ae4m3";
        case InferencePrecision::Q2AE5M2:
            return "q2ae5m2";
        case InferencePrecision::Q4AE5M2:
            return "q4ae5m2";
        case InferencePrecision::Q8AE5M2:
            return "q8ae5m2";
        case InferencePrecision::QAT_FP32:
            return "qat-fp32";
    }
}

constexpr auto parse_precision(std::string_view name) noexcept -> std::optional<InferencePrecision> {
    if (name == "checkpoint" || name == "native") return InferencePrecision::CHECKPOINT;
    if (name == "lowbit-parity" || name == "checkpoint-parity") return InferencePrecision::LOWBIT_PARITY;
    if (name == "fp32") return InferencePrecision::FP32;
    if (name == "bf16") return InferencePrecision::BF16;
    if (name == "q2" || name == "q2a16" || name == "i2a16") return InferencePrecision::Q2A16;
    if (name == "q4" || name == "q4a16" || name == "i4a16") return InferencePrecision::Q4A16;
    if (name == "q8" || name == "q8a16" || name == "i8a16" || name == "w8a16")
        return InferencePrecision::Q8A16;
    if (name == "q2ae4m3" || name == "i2a8") return InferencePrecision::Q2AE4M3;
    if (name == "q4ae4m3" || name == "i4a8") return InferencePrecision::Q4AE4M3;
    if (name == "q8ae4m3" || name == "i8a8" || name == "w8a8" || name == "w8afp8" || name == "i8afp8")
        return InferencePrecision::Q8AE4M3;
    if (name == "q2ae5m2") return InferencePrecision::Q2AE5M2;
    if (name == "q4ae5m2") return InferencePrecision::Q4AE5M2;
    if (name == "q8ae5m2" || name == "w8ae5m2") return InferencePrecision::Q8AE5M2;
    if (name == "qat-fp32") return InferencePrecision::QAT_FP32;
    return std::nullopt;
}

constexpr auto requested_weight_bits(InferencePrecision precision) noexcept -> std::optional<std::int32_t> {
    switch (precision) {
        case InferencePrecision::Q2A16:
        case InferencePrecision::Q2AE4M3:
        case InferencePrecision::Q2AE5M2:
            return 2;
        case InferencePrecision::Q4A16:
        case InferencePrecision::Q4AE4M3:
        case InferencePrecision::Q4AE5M2:
            return 4;
        case InferencePrecision::Q8A16:
        case InferencePrecision::Q8AE4M3:
        case InferencePrecision::Q8AE5M2:
            return 8;
        default:
            return std::nullopt;
    }
}

constexpr auto uses_e4m3_activations(InferencePrecision precision) noexcept -> bool {
    return precision == InferencePrecision::Q2AE4M3 || precision == InferencePrecision::Q4AE4M3 ||
           precision == InferencePrecision::Q8AE4M3;
}

constexpr auto uses_e5m2_activations(InferencePrecision precision) noexcept -> bool {
    return precision == InferencePrecision::Q2AE5M2 || precision == InferencePrecision::Q4AE5M2 ||
           precision == InferencePrecision::Q8AE5M2;
}

constexpr auto to_string(KVCachePrecision precision) noexcept -> std::string_view {
    switch (precision) {
        case KVCachePrecision::AUTO:
            return "auto";
        case KVCachePrecision::FP32:
            return "fp32";
        case KVCachePrecision::BF16:
            return "bf16";
        case KVCachePrecision::INT8:
            return "int8";
        case KVCachePrecision::E4M3:
            return "e4m3";
        case KVCachePrecision::E5M2:
            return "e5m2";
    }
}

constexpr auto parse_kv_cache_precision(std::string_view name) noexcept -> std::optional<KVCachePrecision> {
    if (name == "auto" || name == "checkpoint") return KVCachePrecision::AUTO;
    if (name == "fp32") return KVCachePrecision::FP32;
    if (name == "bf16") return KVCachePrecision::BF16;
    if (name == "int8" || name == "i8") return KVCachePrecision::INT8;
    if (name == "e4m3" || name == "fp8-e4m3") return KVCachePrecision::E4M3;
    if (name == "e5m2" || name == "fp8-e5m2") return KVCachePrecision::E5M2;
    return std::nullopt;
}

} // namespace kidi::core
