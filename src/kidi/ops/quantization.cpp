#include "kidi/ops/quantization.h"
#include "kidi/checkpoint/ggml/dequantize.h"
#include "kidi/ops/context.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <limits>

namespace kidi::ops {
auto pack_weight(const tensor::Tensor& weight, std::int32_t bits, std::int32_t group_size, bool asymmetric_clip)
    -> Result<PackedWeight> {
    try {
        if (!weight.defined() || weight.dimensions() != 2 || !weight.is_contiguous() ||
            (weight.dtype() != tensor::DType::F16 && weight.dtype() != tensor::DType::BF16 &&
             weight.dtype() != tensor::DType::F32) ||
            (bits != 2 && bits != 4 && bits != 8) || group_size <= 0 || group_size % (8 / bits) ||
            weight.size(1) % group_size || !weight.size(0) || !weight.size(1))
            throw Failure({ErrorCode::INVALID_ARGUMENT, "invalid packed weight dimensions, precision or group size"});
        const auto rows = weight.size(0), width = weight.size(1), groups = width / group_size;
        auto values = require(tensor::Tensor::zeros(
            {static_cast<std::int64_t>(rows), static_cast<std::int64_t>(width / (8 / bits))}, tensor::DType::U8));
        auto scales = require(tensor::Tensor::empty(
            {static_cast<std::int64_t>(rows), static_cast<std::int64_t>(groups)}, tensor::DType::F32));
        tensor::Tensor zeros;
        float* zero_values = nullptr;
        if (asymmetric_clip && bits == 4) {
            zeros = require(tensor::Tensor::empty(
                {static_cast<std::int64_t>(rows), static_cast<std::int64_t>(groups)}, tensor::DType::F32));
            zero_values = require(zeros.data<float>()).data();
        }
        const auto bytes = require(weight.host_bytes());
        auto packed = require(values.data<std::uint8_t>());
        auto scale_values = require(scales.data<float>());
        const auto limit = (1 << (bits - 1)) - 1;
        // Same results as std::round(value / scale) clamped, without a libm call per element: truncation is exact, and
        // the remainder decides rounding half away from zero.
        const auto quantize = [&](float value, float scale) {
            const auto quotient = value / scale;
            auto rounded = std::trunc(quotient);
            if (std::abs(quotient - rounded) >= 0.5F) rounded += std::copysign(1.F, quotient);
            return static_cast<int>(std::clamp(rounded, -float(limit), float(limit)));
        };
        const auto pack = [&](auto read) {
            for (std::size_t row = 0; row < rows; ++row)
                for (std::size_t group = 0; group < groups; ++group) {
                    const auto begin = row * width + group * group_size;
                    if (asymmetric_clip && bits == 4) {
                        float mn = std::numeric_limits<float>::infinity();
                        float mx = -std::numeric_limits<float>::infinity();
                        bool finite = true;
                        for (std::int32_t channel = 0; channel < group_size; ++channel) {
                            const auto value = read(begin + channel);
                            finite &= std::isfinite(value);
                            mn = std::min(mn, value);
                            mx = std::max(mx, value);
                        }
                        if (!finite)
                            throw Failure({ErrorCode::INVALID_ARGUMENT, "non-finite weight cannot be quantized"});
                        const float center = 0.5F * (mx + mn);
                        const float half = 0.5F * (mx - mn);
                        float best_scale = 1.F, best_zp = 8.F;
                        float best_mse = std::numeric_limits<float>::infinity();
                        constexpr std::array<float, 11> FRACTIONS{
                            1.00F, 0.95F, 0.90F, 0.85F, 0.80F, 0.75F, 0.70F, 0.65F, 0.60F, 0.55F, 0.50F};
                        for (const float frac : FRACTIONS) {
                            const float lo = center - frac * half;
                            const float sc = std::max((2.0F * frac * half) / 15.0F, 1e-8F);
                            const float zp = std::clamp(std::nearbyint(-lo / sc), 0.0F, 15.0F);
                            float mse = 0.0F;
                            for (std::int32_t channel = 0; channel < group_size; ++channel) {
                                const float w = read(begin + channel);
                                const float q = std::clamp(std::nearbyint(w / sc + zp), 0.0F, 15.0F);
                                const float diff = w - (q - zp) * sc;
                                mse += diff * diff;
                            }
                            if (mse < best_mse) {
                                best_mse = mse;
                                best_scale = sc;
                                best_zp = zp;
                            }
                        }
                        scale_values[row * groups + group] = best_scale;
                        zero_values[row * groups + group] = (8.0F - best_zp) * best_scale;
                        for (std::int32_t channel = 0; channel < group_size; ++channel) {
                            const auto offset = begin + channel;
                            const float w = read(offset);
                            const int q = static_cast<int>(std::clamp(std::nearbyint(w / best_scale + best_zp), 0.0F, 15.0F));
                            const int signed_code = q - 8;
                            packed[offset / 2] |= static_cast<std::uint8_t>((signed_code & 0xF) << ((offset % 2) * 4));
                        }
                        continue;
                    }
                    float maximum = 0.F;
                    bool finite = true;
                    for (std::int32_t channel = 0; channel < group_size; ++channel) {
                        const auto value = read(begin + channel);
                        finite &= std::isfinite(value);
                        maximum = std::max(maximum, std::abs(value));
                    }
                    if (!finite) throw Failure({ErrorCode::INVALID_ARGUMENT, "non-finite weight cannot be quantized"});
                    const auto scale = maximum == 0.F ? 1.F : maximum / limit;
                    if (scale == 0.F) throw Failure({ErrorCode::INVALID_ARGUMENT, "quantization scale underflow"});
                    scale_values[row * groups + group] = scale;
                    if (bits == 8) {
                        for (std::int32_t channel = 0; channel < group_size; ++channel)
                            packed[begin + channel] = static_cast<std::uint8_t>(quantize(read(begin + channel), scale));
                        continue;
                    }
                    for (std::int32_t channel = 0; channel < group_size; ++channel) {
                        const auto offset = begin + channel;
                        packed[offset / (8 / bits)] |= (quantize(read(offset), scale) & ((1 << bits) - 1))
                                                       << ((offset % (8 / bits)) * bits);
                    }
                }
        };
        if (weight.dtype() == tensor::DType::F32) {
            const auto* source = reinterpret_cast<const float*>(bytes.data());
            pack([source](std::size_t offset) { return source[offset]; });
        } else {
            const auto* source = reinterpret_cast<const std::uint16_t*>(bytes.data());
            const bool fp16 = weight.dtype() == tensor::DType::F16;
            pack([source, fp16](std::size_t offset) {
                return fp16 ? checkpoint::ggml::detail::fp16_to_fp32(source[offset])
                            : std::bit_cast<float>(static_cast<std::uint32_t>(source[offset]) << 16);
            });
        }
        return PackedWeight{std::move(values), std::move(scales), std::move(zeros), bits, group_size};
    } catch (const Failure& error) {
        return std::unexpected(error.error());
    }
}
} // namespace kidi::ops