#include "kidi/ops/quantization.h"
#include "kidi/checkpoint/ggml/dequantize.h"
#include "kidi/ops/context.h"
#include "kidi/ops/cpu_kernels.h"
#include "kidi/runtime/ynn/graph.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

namespace kidi::ops {
auto pack_expert_weights(const tensor::Tensor& values, const tensor::Tensor& scales, const tensor::Tensor& global,
                         bool interleaved) -> Result<PackedWeight> {
    using tensor::DType;
    using tensor::Tensor;
    try {
        const bool nvfp4 = values.dtype() == DType::U8;
        if (values.dimensions() != 3 || scales.dimensions() != 3 || !values.is_contiguous() ||
            !scales.is_contiguous() || (!nvfp4 && values.dtype() != DType::E4M3) ||
            !values.size(0) || !values.size(1) || !values.size(2) ||
            scales.size(0) != values.size(0) || scales.size(1) != values.size(1) ||
            (scales.dtype() != DType::U8 && !(nvfp4 && scales.dtype() == DType::E4M3)) ||
            (interleaved && values.size(1) % 2))
            throw Failure({ErrorCode::INVALID_ARGUMENT, "invalid expert weight format or dimensions"});
        const auto experts = static_cast<std::int64_t>(values.size(0));
        const auto rows = static_cast<std::int64_t>(values.size(1));
        const auto width = static_cast<std::int64_t>(values.size(2)) * (nvfp4 ? 2 : 1);
        const auto block = nvfp4 ? 16 : 32;
        if (width > INT32_MAX || width % block || scales.size(2) != width / block ||
            (nvfp4 && (!global.defined() || global.dtype() != DType::F32 || global.dimensions() != 1 ||
                       global.numel() != values.size(0))) ||
            (!nvfp4 && global.defined()))
            throw Failure({ErrorCode::INVALID_ARGUMENT, "invalid expert block scales or global scales"});
        const auto source = require(values.host_bytes());
        const auto blocks = require(scales.host_bytes());
        const auto globals = nvfp4 ? require(global.data<float>()) : std::span<float>{};
        const auto e4m3 = [](std::uint8_t byte) {
            const int exponent = (byte >> 3) & 15, mantissa = byte & 7;
            if (exponent == 15 && mantissa == 7) return std::numeric_limits<float>::quiet_NaN();
            const auto value = exponent ? std::ldexp(1.F + mantissa / 8.F, exponent - 7) : mantissa / 512.F;
            return byte & 128 ? -value : value;
        };
        std::array<float, 256> decoded{}, block_scales{};
        for (std::size_t byte = 0; byte < decoded.size(); ++byte) {
            decoded[byte] = e4m3(static_cast<std::uint8_t>(byte));
            block_scales[byte] = nvfp4 ? decoded[byte]
                                      : (byte == 255 ? std::numeric_limits<float>::quiet_NaN()
                                                     : std::ldexp(1.F, static_cast<int>(byte) - 127));
        }
        constexpr std::array<float, 16> E2M1{0, .5F, 1, 1.5F, 2, 3, 4, 6, -0.F, -.5F, -1, -1.5F, -2, -3, -4, -6};
        auto packed = require(Tensor::empty({experts, rows, width}, DType::U8));
        auto row_scales = require(Tensor::empty({experts, rows, 1}, DType::F32));
        const auto target = require(packed.data<std::uint8_t>());
        const auto steps = require(row_scales.data<float>());
        require(runtime::ynn::parallel_ranges(experts * rows, 32, [&](std::size_t begin, std::size_t end) {
            std::vector<float> expanded(static_cast<std::size_t>(width));
            for (auto item = begin; item < end; ++item) {
                const auto expert = item / rows;
                const auto row = item % rows;
                if (nvfp4 && !std::isfinite(globals[expert]))
                    throw Failure({ErrorCode::INVALID_ARGUMENT, "non-finite expert global scale"});
                const auto source_row = interleaved ? (row % (rows / 2)) * 2 + row / (rows / 2) : row;
                const auto base = (expert * rows + source_row) * width;
                for (std::int64_t column = 0; column < width; column += block) {
                    const auto scale = block_scales[std::to_integer<std::uint8_t>(blocks[(base + column) / block])];
                    for (std::int64_t channel = 0; channel < block; ++channel) {
                        const auto c = column + channel;
                        const auto byte = std::to_integer<std::uint8_t>(source[(base + c) / (nvfp4 ? 2 : 1)]);
                        expanded[c] = nvfp4 ? E2M1[c % 2 ? byte >> 4 : byte & 15] * scale * globals[expert]
                                            : decoded[byte] * scale;
                    }
                }
                if (!cpu::pack_int8_rows({expanded.data(), cpu::Rows::Storage::F32, width}, 1,
                                         target.data() + item * width, steps.data() + item))
                    throw Failure({ErrorCode::INVALID_ARGUMENT, "non-finite weight or quantization scale underflow"});
            }
        }));
        return PackedWeight{std::move(packed), std::move(row_scales), {}, 8, static_cast<std::int32_t>(width)};
    } catch (const Failure& error) {
        return std::unexpected(error.error());
    }
}

auto pack_rows_int8(const tensor::Tensor& weight) -> Result<PackedWeight> {
    try {
        if (!weight.defined() || weight.dimensions() != 2 || !weight.is_contiguous() || !weight.is_host_accessible())
            throw Failure({ErrorCode::INVALID_ARGUMENT, "row packing needs a contiguous host matrix"});
        const auto rows = static_cast<std::int64_t>(weight.size(0)), width = static_cast<std::int64_t>(weight.size(1));
        auto values = require(tensor::Tensor::empty({rows, width}, tensor::DType::U8));
        auto scales = require(tensor::Tensor::empty({rows, 1}, tensor::DType::F32));
        const auto target_values = require(values.data<std::uint8_t>());
        const auto target_scales = require(scales.data<float>());
        if (weight.dtype() == tensor::DType::F32 || weight.dtype() == tensor::DType::BF16) {
            const auto source = require(weight.host_bytes());
            const auto storage = weight.dtype() == tensor::DType::F32 ? cpu::Rows::Storage::F32
                                                                     : cpu::Rows::Storage::BF16;
            const auto element_bytes = tensor::element_size(weight.dtype());
            require(runtime::ynn::parallel_ranges(rows, 32, [&](std::size_t begin, std::size_t end) {
                if (!cpu::pack_int8_rows({source.data() + begin * width * element_bytes, storage, width},
                                         static_cast<std::int64_t>(end - begin),
                                         target_values.data() + begin * width, target_scales.data() + begin))
                    throw Failure({ErrorCode::INVALID_ARGUMENT, "non-finite weight or quantization scale underflow"});
            }));
            return PackedWeight{std::move(values), std::move(scales), {}, 8, static_cast<std::int32_t>(width)};
        }
        constexpr std::int64_t CHUNK = 256;
        const auto chunks = static_cast<std::size_t>((rows + CHUNK - 1) / CHUNK);
        require(runtime::ynn::parallel_ranges(chunks, 1, [&](std::size_t begin, std::size_t end) {
            for (auto chunk = begin; chunk < end; ++chunk) {
                const auto first = static_cast<std::int64_t>(chunk) * CHUNK;
                const auto count = std::min(CHUNK, rows - first);
                const auto part = require(pack_weight(require(weight.narrow(0, first, count)), 8,
                                                      static_cast<std::int32_t>(width)));
                const auto part_values = require(part.values.host_bytes());
                const auto part_scales = require(part.scales.host_bytes());
                std::memcpy(target_values.data() + first * width, part_values.data(), part_values.size());
                std::memcpy(target_scales.data() + first, part_scales.data(), part_scales.size());
            }
        }));
        return PackedWeight{std::move(values), std::move(scales), {}, 8, static_cast<std::int32_t>(width)};
    } catch (const Failure& error) {
        return std::unexpected(error.error());
    }
}

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