#include "kidi/ops/quantization.h"
#include "kidi/checkpoint/ggml/dequantize.h"
#include "kidi/ops/context.h"
#include "kidi/ops/cpu_kernels.h"
#include "kidi/runtime/ynn/graph.h"
#include "ynnpack/composites/vq_linear.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

namespace kidi::ops {
namespace {
auto pack_storage(tensor::DType dtype) -> cpu::Rows::Storage {
    if (dtype == tensor::DType::F32) return cpu::Rows::Storage::F32;
    if (dtype == tensor::DType::BF16) return cpu::Rows::Storage::BF16;
    if (dtype == tensor::DType::F16) return cpu::Rows::Storage::F16;
    throw Failure({ErrorCode::INVALID_ARGUMENT, "packed weights require F16, BF16 or F32 input"});
}
} // namespace

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
        if (!weight.defined() || weight.dimensions() != 2 || !weight.is_contiguous() || !weight.is_host_accessible() ||
            !weight.size(0) || !weight.size(1))
            throw Failure({ErrorCode::INVALID_ARGUMENT, "row packing needs a nonempty contiguous host matrix"});
        const auto storage = pack_storage(weight.dtype());
        const auto rows = static_cast<std::int64_t>(weight.size(0));
        const auto width = static_cast<std::int64_t>(weight.size(1));
        auto values = require(tensor::Tensor::empty({rows, width}, tensor::DType::U8));
        auto scales = require(tensor::Tensor::empty({rows, 1}, tensor::DType::F32));
        const auto target_values = require(values.data<std::uint8_t>());
        const auto target_scales = require(scales.data<float>());
        const auto source = require(weight.host_bytes());
        const auto element_bytes = tensor::element_size(weight.dtype());
        require(runtime::ynn::parallel_ranges(rows, 32, [&](std::size_t begin, std::size_t end) {
            if (!cpu::pack_int8_rows({source.data() + begin * width * element_bytes, storage, width},
                                     static_cast<std::int64_t>(end - begin), target_values.data() + begin * width,
                                     target_scales.data() + begin))
                throw Failure({ErrorCode::INVALID_ARGUMENT, "non-finite weight or quantization scale underflow"});
        }));
        return PackedWeight{std::move(values), std::move(scales), {}, 8, static_cast<std::int32_t>(width)};
    } catch (const Failure& error) {
        return std::unexpected(error.error());
    }
}

auto pack_weight(const tensor::Tensor& weight, std::int32_t bits, std::int32_t group_size) -> Result<PackedWeight> {
    try {
        if (!weight.defined() || weight.dimensions() != 2 || !weight.is_contiguous() ||
            (weight.dtype() != tensor::DType::F16 && weight.dtype() != tensor::DType::BF16 &&
             weight.dtype() != tensor::DType::F32) ||
            (bits != 2 && bits != 4 && bits != 8) || group_size <= 0 || group_size % (8 / bits) ||
            weight.size(1) % group_size || !weight.size(0) || !weight.size(1))
            throw Failure({ErrorCode::INVALID_ARGUMENT, "invalid packed weight dimensions, precision or group size"});
        const auto rows = weight.size(0), width = weight.size(1), groups = width / group_size;
        if (bits == 8) {
            auto values = require(tensor::Tensor::empty(
                {static_cast<std::int64_t>(rows), static_cast<std::int64_t>(width)}, tensor::DType::U8));
            auto scales = require(tensor::Tensor::empty(
                {static_cast<std::int64_t>(rows), static_cast<std::int64_t>(groups)}, tensor::DType::F32));
            const auto bytes = require(weight.host_bytes());
            auto packed = require(values.data<std::uint8_t>());
            auto scale_values = require(scales.data<float>());
            if (!cpu::pack_int8_rows({bytes.data(), pack_storage(weight.dtype()), group_size},
                                     static_cast<std::int64_t>(rows * groups), packed.data(), scale_values.data()))
                throw Failure({ErrorCode::INVALID_ARGUMENT, "non-finite weight or quantization scale underflow"});
            return PackedWeight{std::move(values), std::move(scales), {}, 8, group_size};
        }
        auto values = require(tensor::Tensor::zeros(
            {static_cast<std::int64_t>(rows), static_cast<std::int64_t>(width / (8 / bits))}, tensor::DType::U8));
        auto scales = require(tensor::Tensor::empty(
            {static_cast<std::int64_t>(rows), static_cast<std::int64_t>(groups)}, tensor::DType::F32));
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
        return PackedWeight{std::move(values), std::move(scales), {}, bits, group_size};
    } catch (const Failure& error) {
        return std::unexpected(error.error());
    }
}

auto pack_vq_expert_weights(const tensor::Tensor& values, const tensor::Tensor& scales, const tensor::Tensor& global,
                            bool interleaved, BlockScaledFormat format) -> Result<BlockScaledWeight> {
    using tensor::DType;
    using tensor::Tensor;
    try {
        if (format != BlockScaledFormat::VQ3D && format != BlockScaledFormat::VQ4D)
            throw Failure({ErrorCode::INVALID_ARGUMENT, "pack_vq_expert_weights requires VQ3D or VQ4D format"});
        const bool nvfp4 = values.dtype() == DType::U8;
        if (values.dimensions() != 3 || scales.dimensions() != 3 || !values.is_contiguous() ||
            !scales.is_contiguous() || (!nvfp4 && values.dtype() != DType::E4M3) || !values.size(0) ||
            !values.size(1) || !values.size(2) || scales.size(0) != values.size(0) ||
            scales.size(1) != values.size(1) ||
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

        const std::int32_t dim = format == BlockScaledFormat::VQ3D ? 3 : 4;
        const auto group = static_cast<std::int32_t>(ynn::vq_group_size(dim));
        const auto k_pad = static_cast<std::int64_t>(ynn::vq_padded_cols(dim, static_cast<std::size_t>(width)));
        const auto n_groups = k_pad / group;
        const auto row_codes = k_pad / dim;
        auto out_values = require(Tensor::empty({experts, rows, row_codes}, DType::U8));
        auto out_scales = require(Tensor::empty({experts, rows, n_groups}, DType::U8));
        auto out_global = require(Tensor::empty({experts, rows}, DType::F32));

        const auto source = require(values.host_bytes());
        const auto blocks = require(scales.host_bytes());
        const auto globals = nvfp4 ? require(global.data<float>()) : std::span<float>{};
        const auto dst_vals = require(out_values.data<std::uint8_t>());
        const auto dst_scales = require(out_scales.data<std::uint8_t>());
        const auto dst_global = require(out_global.data<float>());

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

        // Decode 64-row chunks to F32 and pack them directly, without expanding the whole bank.
        const auto row_chunk = static_cast<std::size_t>(rows % 64 == 0 ? 64 : 32);
        const auto chunks_per_expert = (static_cast<std::size_t>(rows) + row_chunk - 1) / row_chunk;
        require(runtime::ynn::parallel_ranges(
            static_cast<std::size_t>(experts) * chunks_per_expert, 1, [&](std::size_t c_begin, std::size_t c_end) {
                std::vector<float> f32_slice;
                for (std::size_t c = c_begin; c < c_end; ++c) {
                    const auto expert = c / chunks_per_expert;
                    const auto r_begin = (c % chunks_per_expert) * row_chunk;
                    const auto r_end = std::min(static_cast<std::size_t>(rows), r_begin + row_chunk);
                    if (nvfp4 && !std::isfinite(globals[expert]))
                        throw Failure({ErrorCode::INVALID_ARGUMENT, "non-finite expert global scale"});
                    f32_slice.resize((r_end - r_begin) * static_cast<std::size_t>(width));
                    for (std::size_t lr = 0; lr < r_end - r_begin; ++lr) {
                        const auto row = static_cast<std::int64_t>(r_begin + lr);
                        const auto source_row = interleaved ? (row % (rows / 2)) * 2 + row / (rows / 2) : row;
                        const auto base = (static_cast<std::int64_t>(expert) * rows + source_row) * width;
                        float* dst_row = f32_slice.data() + lr * static_cast<std::size_t>(width);
                        for (std::int64_t column = 0; column < width; column += block) {
                            const auto scale =
                                block_scales[std::to_integer<std::uint8_t>(blocks[(base + column) / block])];
                            const float eff_scale = nvfp4 ? scale * globals[expert] : scale;
                            for (std::int64_t channel = 0; channel < block; ++channel) {
                                const auto col = column + channel;
                                const auto byte = std::to_integer<std::uint8_t>(source[(base + col) / (nvfp4 ? 2 : 1)]);
                                dst_row[col] = nvfp4 ? E2M1[col % 2 ? byte >> 4 : byte & 15] * eff_scale
                                                     : decoded[byte] * eff_scale;
                            }
                        }
                    }
                    const auto first_row = expert * static_cast<std::size_t>(rows);
                    ynn::vq_pack_row_block_f32(dim, static_cast<std::size_t>(rows), static_cast<std::size_t>(width),
                                               r_begin, r_end, f32_slice.data(),
                                               dst_vals.data() + first_row * static_cast<std::size_t>(row_codes),
                                               dst_scales.data() + first_row * static_cast<std::size_t>(n_groups),
                                               dst_global.data() + first_row);
                }
            }));
        return BlockScaledWeight{format, std::move(out_values), std::move(out_scales), std::move(out_global), group};
    } catch (const Failure& error) {
        return std::unexpected(error.error());
    }
}

auto pack_vq_weight(const tensor::Tensor& weight, BlockScaledFormat format) -> Result<BlockScaledWeight> {
    using tensor::DType;
    using tensor::Tensor;
    try {
        if (format != BlockScaledFormat::VQ3D && format != BlockScaledFormat::VQ4D)
            throw Failure({ErrorCode::INVALID_ARGUMENT, "pack_vq_weight requires VQ3D or VQ4D format"});
        if (!weight.defined() || (weight.dimensions() != 2 && weight.dimensions() != 3) || !weight.is_contiguous() ||
            !weight.is_host_accessible() ||
            (weight.dtype() != DType::F32 && weight.dtype() != DType::BF16 && weight.dtype() != DType::F16))
            throw Failure({ErrorCode::INVALID_ARGUMENT, "VQ packing requires contiguous host F16/BF16/F32 weights"});
        const bool is_3d = weight.dimensions() == 3;
        const auto experts = is_3d ? static_cast<std::int64_t>(weight.size(0)) : std::int64_t{1};
        const auto rows = static_cast<std::int64_t>(weight.size(is_3d ? 1 : 0));
        const auto cols = static_cast<std::int64_t>(weight.size(is_3d ? 2 : 1));
        if (!experts || !rows || !cols)
            throw Failure({ErrorCode::INVALID_ARGUMENT, "VQ packing requires nonempty weight dimensions"});

        const std::int32_t dim = format == BlockScaledFormat::VQ3D ? 3 : 4;
        const auto group = static_cast<std::int32_t>(ynn::vq_group_size(dim));
        const auto k_pad = static_cast<std::int64_t>(ynn::vq_padded_cols(dim, static_cast<std::size_t>(cols)));
        const auto n_groups = k_pad / group;
        const auto row_codes = k_pad / dim;
        const auto shape = [&](std::int64_t width) {
            return is_3d ? std::vector<std::int64_t>{experts, rows, width} : std::vector<std::int64_t>{rows, width};
        };
        auto values = require(Tensor::empty(shape(row_codes), DType::U8));
        auto scales = require(Tensor::empty(shape(n_groups), DType::U8));
        auto global = require(is_3d ? Tensor::empty({experts, rows}, DType::F32) : Tensor::empty({rows}, DType::F32));

        const auto src_bytes = require(weight.host_bytes());
        const auto dst_vals = require(values.data<std::uint8_t>());
        const auto dst_scales = require(scales.data<std::uint8_t>());
        const auto dst_global = require(global.data<float>());
        const auto row_chunk = static_cast<std::size_t>(rows % 64 == 0 ? 64 : 32);
        const auto chunks_per_expert = (static_cast<std::size_t>(rows) + row_chunk - 1) / row_chunk;
        require(runtime::ynn::parallel_ranges(
            static_cast<std::size_t>(experts) * chunks_per_expert, 1, [&](std::size_t c_begin, std::size_t c_end) {
                std::vector<float> f32_slice;
                for (std::size_t c = c_begin; c < c_end; ++c) {
                    const auto expert = c / chunks_per_expert;
                    const auto r_begin = (c % chunks_per_expert) * row_chunk;
                    const auto r_end = std::min(static_cast<std::size_t>(rows), r_begin + row_chunk);
                    const auto first_row = expert * static_cast<std::size_t>(rows);
                    const auto elem_offset = (first_row + r_begin) * static_cast<std::size_t>(cols);
                    const auto count = (r_end - r_begin) * static_cast<std::size_t>(cols);
                    const float* f32_ptr = nullptr;
                    if (weight.dtype() == DType::F32) {
                        f32_ptr = reinterpret_cast<const float*>(src_bytes.data()) + elem_offset;
                    } else {
                        f32_slice.resize(count);
                        const auto* raw16 = reinterpret_cast<const std::uint16_t*>(src_bytes.data()) + elem_offset;
                        for (std::size_t i = 0; i < count; ++i)
                            f32_slice[i] = weight.dtype() == DType::BF16
                                               ? std::bit_cast<float>(static_cast<std::uint32_t>(raw16[i]) << 16)
                                               : checkpoint::ggml::detail::fp16_to_fp32(raw16[i]);
                        f32_ptr = f32_slice.data();
                    }
                    ynn::vq_pack_row_block_f32(dim, static_cast<std::size_t>(rows), static_cast<std::size_t>(cols),
                                               r_begin, r_end, f32_ptr,
                                               dst_vals.data() + first_row * static_cast<std::size_t>(row_codes),
                                               dst_scales.data() + first_row * static_cast<std::size_t>(n_groups),
                                               dst_global.data() + first_row);
                }
            }));
        return BlockScaledWeight{format, std::move(values), std::move(scales), std::move(global), group};
    } catch (const Failure& error) {
        return std::unexpected(error.error());
    }
}
} // namespace kidi::ops