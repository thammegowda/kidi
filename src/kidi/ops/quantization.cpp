#include "kidi/ops/quantization.h"
#include "kidi/ops/context.h"

#include <algorithm>
#include <bit>
#include <cmath>

namespace kidi::ops {
auto pack_weight(const tensor::Tensor& weight, std::int32_t bits, std::int32_t group_size) -> Result<PackedWeight> {
    try {
        if (!weight.defined() || weight.dimensions() != 2 || !weight.is_contiguous() ||
            (weight.dtype() != tensor::DType::BF16 && weight.dtype() != tensor::DType::F32) ||
            (bits != 2 && bits != 4 && bits != 8) || group_size <= 0 || group_size % (8 / bits) ||
            weight.size(1) % group_size || !weight.size(0) || !weight.size(1))
            throw Failure({ErrorCode::INVALID_ARGUMENT, "invalid packed weight dimensions, precision or group size"});
        const auto rows = weight.size(0), width = weight.size(1), groups = width / group_size;
        auto values = require(tensor::Tensor::zeros(
            {static_cast<std::int64_t>(rows), static_cast<std::int64_t>(width / (8 / bits))}, tensor::DType::U8));
        auto scales = require(tensor::Tensor::empty(
            {static_cast<std::int64_t>(rows), static_cast<std::int64_t>(groups)}, tensor::DType::F32));
        const auto bytes = require(weight.host_bytes());
        const auto read = [&](std::size_t offset) {
            return weight.dtype() == tensor::DType::F32
                       ? reinterpret_cast<const float*>(bytes.data())[offset]
                       : std::bit_cast<float>(
                             static_cast<std::uint32_t>(reinterpret_cast<const std::uint16_t*>(bytes.data())[offset])
                             << 16);
        };
        auto packed = require(values.data<std::uint8_t>());
        auto scale_values = require(scales.data<float>());
        const auto limit = (1 << (bits - 1)) - 1;
        for (std::size_t row = 0; row < rows; ++row)
            for (std::size_t group = 0; group < groups; ++group) {
                const auto begin = row * width + group * group_size;
                float maximum = 0.F;
                for (std::int32_t channel = 0; channel < group_size; ++channel) {
                    const auto value = read(begin + channel);
                    if (!std::isfinite(value))
                        throw Failure({ErrorCode::INVALID_ARGUMENT, "non-finite weight cannot be quantized"});
                    maximum = std::max(maximum, std::abs(value));
                }
                const auto scale = maximum == 0.F ? 1.F : maximum / limit;
                if (scale == 0.F) throw Failure({ErrorCode::INVALID_ARGUMENT, "quantization scale underflow"});
                scale_values[row * groups + group] = scale;
                for (std::int32_t channel = 0; channel < group_size; ++channel) {
                    const auto offset = begin + channel;
                    const auto quantized =
                        static_cast<int>(std::clamp(std::round(read(offset) / scale), -float(limit), float(limit)));
                    packed[offset / (8 / bits)] |= (quantized & ((1 << bits) - 1)) << ((offset % (8 / bits)) * bits);
                }
            }
        return PackedWeight{std::move(values), std::move(scales), bits, group_size};
    } catch (const Failure& error) {
        return std::unexpected(error.error());
    }
}
} // namespace kidi::ops