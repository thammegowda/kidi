#pragma once

#include <optional>
#include <string_view>
#include <utility>

#include "kidi/tensor/dtype.h"

namespace kidi::core {

/// Weight and compute policy. CHECKPOINT runs each layer in its stored format with native kernels; BF16 is the
/// reference. Q8 packs BF16 weights to per-channel INT8 at load (INT8 experts for Inkling-2; CUDA keeps BF16
/// activations), and Q4 packs them to INT4 in groups of 128. QAT_FP32 is a developer oracle for QAT checkpoints that
/// dequantizes to FP32 and keeps trained activation rounding; VQ3D_HYBRID stores Inkling-2 experts as VQ3D/VQ4D codes.
enum class InferencePrecision {
    CHECKPOINT,
    BF16,
    Q8,
    Q4,
    QAT_FP32,
    VQ3D_HYBRID,
};

/// Storage policy for language-model keys and values. AUTO picks the model and device default.
enum class KVCachePrecision {
    AUTO,
    FP32,
    BF16,
    INT8,
};

constexpr auto to_string(InferencePrecision precision) noexcept -> std::string_view {
    switch (precision) {
        case InferencePrecision::CHECKPOINT:
            return "checkpoint";
        case InferencePrecision::BF16:
            return "bf16";
        case InferencePrecision::Q8:
            return "q8";
        case InferencePrecision::Q4:
            return "q4";
        case InferencePrecision::QAT_FP32:
            return "qat-fp32";
        case InferencePrecision::VQ3D_HYBRID:
            return "vq3d-hybrid";
    }
    std::unreachable();
}

constexpr auto parse_precision(std::string_view name) noexcept -> std::optional<InferencePrecision> {
    if (name == "checkpoint") return InferencePrecision::CHECKPOINT;
    if (name == "bf16") return InferencePrecision::BF16;
    if (name == "q8") return InferencePrecision::Q8;
    if (name == "q4") return InferencePrecision::Q4;
    if (name == "qat-fp32") return InferencePrecision::QAT_FP32;
    if (name == "vq3d-hybrid") return InferencePrecision::VQ3D_HYBRID;
    return std::nullopt;
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
    }
    std::unreachable();
}

constexpr auto parse_kv_cache_precision(std::string_view name) noexcept -> std::optional<KVCachePrecision> {
    if (name == "auto") return KVCachePrecision::AUTO;
    if (name == "fp32") return KVCachePrecision::FP32;
    if (name == "bf16") return KVCachePrecision::BF16;
    if (name == "int8") return KVCachePrecision::INT8;
    return std::nullopt;
}

/// Storage dtype for an explicit cache precision; AUTO lets the model pick from backend capabilities.
constexpr auto cache_dtype(KVCachePrecision precision) noexcept -> std::optional<tensor::DType> {
    switch (precision) {
        case KVCachePrecision::AUTO:
            return std::nullopt;
        case KVCachePrecision::FP32:
            return tensor::DType::F32;
        case KVCachePrecision::BF16:
            return tensor::DType::BF16;
        case KVCachePrecision::INT8:
            return tensor::DType::I8;
    }
    std::unreachable();
}

} // namespace kidi::core
