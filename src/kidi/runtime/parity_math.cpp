#include "kidi/runtime/parity_math.h"
#include "kidi/ops/context.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <limits>
#include <vector>

#if defined(__APPLE__)
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#endif

// Numerical compatibility helpers for Kidi's opt-in parity modes; there is no PyTorch or Python runtime dependency.
// Reduction ordering is adapted from PyTorch (BSD-3-Clause): third_party/PYTORCH_LICENSE.txt.
// The sin/cos/exp approximations are adapted from SLEEF by Naoki Shibata and contributors (BSL-1.0):
// third_party/SLEEF_LICENSE.txt.
namespace kidi::runtime::parity {
namespace {
using ops::require;

// SLEEF-style two-float arithmetic retains the low component during trigonometric range reduction.
struct DoubleFloat {
    float high, low;
};

auto add_exact(float left, float right) -> DoubleFloat {
#pragma clang fp contract(off)
    const auto sum = left + right;
    const auto virtual_right = sum - left;
    return {sum, (left - (sum - virtual_right)) + (right - virtual_right)};
}

auto add(float left, float right) -> DoubleFloat {
#pragma clang fp contract(off)
    const auto sum = left + right;
    return {sum, (left - sum) + right};
}

auto add_exact(DoubleFloat left, float right) -> DoubleFloat {
#pragma clang fp contract(off)
    const auto sum = left.high + right;
    const auto virtual_right = sum - left.high;
    const auto error = (left.high - (sum - virtual_right)) + (right - virtual_right);
    return {sum, error + left.low};
}

auto add(DoubleFloat left, float right) -> DoubleFloat {
#pragma clang fp contract(off)
    const auto sum = left.high + right;
    return {sum, (left.high - sum) + right + left.low};
}

auto add(float left, DoubleFloat right) -> DoubleFloat {
#pragma clang fp contract(off)
    const auto sum = left + right.high;
    return {sum, (left - sum) + right.high + right.low};
}

auto square(DoubleFloat value) -> DoubleFloat {
#pragma clang fp contract(off)
    const auto high = value.high * value.high;
    return {high, std::fma(2.F * value.high, value.low, std::fma(value.high, value.high, -high))};
}

auto multiply(DoubleFloat left, DoubleFloat right) -> DoubleFloat {
#pragma clang fp contract(off)
    const auto high = left.high * right.high;
    return {high, std::fma(left.high, right.low,
                           std::fma(left.low, right.high, std::fma(left.high, right.high, -high)))};
}

auto multiply_value(DoubleFloat left, DoubleFloat right) -> float {
#pragma clang fp contract(off)
    return std::fma(left.high, right.high,
                    std::fma(left.low, right.high, left.high * right.low));
}

auto sine_polynomial(DoubleFloat reduced) -> float {
#pragma clang fp contract(off)
    const auto squared = square(reduced);
    auto polynomial = 2.6083159809786594e-06F;
    polynomial = std::fma(polynomial, squared.high, -0.00019810690719168633F);
    polynomial = std::fma(polynomial, squared.high, 0.00833307858556509F);
    const auto correction = add(-0.16666659712791443F, polynomial * squared.high);
    return multiply_value(reduced, add(1.F, multiply(correction, squared)));
}
} // namespace

/// SLEEF-compatible sine for inputs in the small-range reduction interval.
auto sine(float value) -> float {
#pragma clang fp contract(off)
    constexpr float INVERSE_PI = 0.31830988618379067F;
    constexpr float PI_HIGH = 3.1414794921875F;
    constexpr float PI_MIDDLE = 0.0001131594181060791F;
    constexpr float PI_LOW = 1.984187258941006e-09F;
    const auto quadrant_float = std::nearbyint(value * INVERSE_PI);
    const auto quadrant = static_cast<std::int32_t>(quadrant_float);
    const auto first = std::fma(quadrant_float, -PI_HIGH, value);
    auto reduced = add_exact(first, quadrant_float * -PI_MIDDLE);
    reduced = add(reduced, quadrant_float * -PI_LOW);
    auto result = sine_polynomial(reduced);
    if (quadrant & 1) result = -result;
    return value == 0.F ? value : result;
}

/// SLEEF-compatible cosine for inputs in the small-range reduction interval.
auto cosine(float value) -> float {
#pragma clang fp contract(off)
    constexpr float INVERSE_PI = 0.31830988618379067F;
    constexpr float HALF_PI_HIGH = 3.1414794921875F * 0.5F;
    constexpr float HALF_PI_MIDDLE = 0.0001131594181060791F * 0.5F;
    constexpr float HALF_PI_LOW = 1.984187258941006e-09F * 0.5F;
    const auto quadrant_float =
        std::fma(std::nearbyint(std::fma(value, INVERSE_PI, -0.5F)), 2.F, 1.F);
    const auto quadrant = static_cast<std::int32_t>(quadrant_float);
    auto reduced = add_exact(value, quadrant_float * -HALF_PI_HIGH);
    reduced = add_exact(reduced, quadrant_float * -HALF_PI_MIDDLE);
    reduced = add_exact(reduced, quadrant_float * -HALF_PI_LOW);
    auto result = sine_polynomial(reduced);
    if ((quadrant & 2) == 0) result = -result;
    return result;
}

/// SLEEF-compatible single-precision exponential used by parity softmax.
auto exponential(float value) -> float {
#pragma clang fp contract(off)
    if (value < -104.F) return 0.F;
    if (value > 100.F) return std::numeric_limits<float>::infinity();
    constexpr float R_LN2 = 1.4426950408889634074F;
    constexpr float LN2_HIGH = 0.693145751953125F;
    constexpr float LN2_LOW = 1.428606765330187045e-06F;
    const auto exponent = static_cast<std::int32_t>(std::nearbyint(value * R_LN2));
    auto reduced = std::fma(static_cast<float>(exponent), -LN2_HIGH, value);
    reduced = std::fma(static_cast<float>(exponent), -LN2_LOW, reduced);
    auto polynomial = 0.0001985276176128536463F;
    polynomial = std::fma(polynomial, reduced, 0.0013930435525253415F);
    polynomial = std::fma(polynomial, reduced, 0.008333360776305199F);
    polynomial = std::fma(polynomial, reduced, 0.041666485369205475F);
    polynomial = std::fma(polynomial, reduced, 0.1666666716337204F);
    polynomial = std::fma(polynomial, reduced, 0.5F);
    auto result = 1.F + std::fma(reduced * reduced, polynomial, reduced);
    const auto power_of_two = [](std::int32_t power) {
        return std::bit_cast<float>(static_cast<std::uint32_t>(power + 127) << 23);
    };
    const auto half = exponent >> 1;
    result *= power_of_two(half);
    return result * power_of_two(exponent - half);
}

/// Applies the reference mask, exponential, four-lane sum, and normalization in place.
auto softmax(std::span<float> values, std::span<const float> mask) -> void {
    auto maximum = -std::numeric_limits<float>::infinity();
    for (std::size_t index = 0; index < values.size(); ++index) {
        values[index] += mask[index];
        maximum = std::max(maximum, values[index]);
    }
    std::array<float, 4> partial{};
    for (std::size_t index = 0; index < values.size(); ++index) {
        values[index] = exponential(values[index] - maximum);
        partial[index % partial.size()] += values[index];
    }
    const auto even = partial[0] + partial[2];
    const auto odd = partial[1] + partial[3];
    const auto reciprocal = 1.F / (even + odd);
    for (auto& value : values) value *= reciprocal;
}

/// PyTorch-compatible four-level cascade reduction for RMS normalization.
auto sum_squares(std::span<const float> values) -> float {
#pragma clang fp contract(off)
    constexpr std::size_t VECTOR_WIDTH = 4, ILP = 4, LEVELS = 4;
    using Vector = std::array<float, VECTOR_WIDTH>;
    std::array<std::array<Vector, ILP>, LEVELS> accumulators{};
    const auto vector_count = values.size() / VECTOR_WIDTH;
    const auto grouped_count = vector_count / ILP;
    const auto ceil_log2 = [](std::size_t value) {
        return value <= 1 ? std::size_t{0} : static_cast<std::size_t>(std::bit_width(value - 1));
    };
    const auto level_power = std::max<std::size_t>(4, ceil_log2(grouped_count) / LEVELS);
    const auto level_step = std::size_t{1} << level_power;
    const auto level_mask = level_step - 1;
    const auto add_vector = [&](Vector& destination, std::size_t vector) {
        const auto offset = vector * VECTOR_WIDTH;
        for (std::size_t lane = 0; lane < VECTOR_WIDTH; ++lane) {
            const auto square = values[offset + lane] * values[offset + lane];
            destination[lane] += square;
        }
    };
    std::size_t group = 0;
    for (; group + level_step <= grouped_count;) {
        for (std::size_t step = 0; step < level_step; ++step, ++group)
            for (std::size_t lane_group = 0; lane_group < ILP; ++lane_group)
                add_vector(accumulators[0][lane_group], group * ILP + lane_group);
        for (std::size_t level = 1; level < LEVELS; ++level) {
            for (std::size_t lane_group = 0; lane_group < ILP; ++lane_group)
                for (std::size_t lane = 0; lane < VECTOR_WIDTH; ++lane) {
                    accumulators[level][lane_group][lane] += accumulators[level - 1][lane_group][lane];
                    accumulators[level - 1][lane_group][lane] = 0.F;
                }
            if (group & (level_mask << (level * level_power))) break;
        }
    }
    for (; group < grouped_count; ++group)
        for (std::size_t lane_group = 0; lane_group < ILP; ++lane_group)
            add_vector(accumulators[0][lane_group], group * ILP + lane_group);
    for (std::size_t level = 1; level < LEVELS; ++level)
        for (std::size_t lane_group = 0; lane_group < ILP; ++lane_group)
            for (std::size_t lane = 0; lane < VECTOR_WIDTH; ++lane)
                accumulators[0][lane_group][lane] += accumulators[level][lane_group][lane];
    for (std::size_t vector = grouped_count * ILP; vector < vector_count; ++vector)
        add_vector(accumulators[0][0], vector);
    for (std::size_t lane_group = 1; lane_group < ILP; ++lane_group)
        for (std::size_t lane = 0; lane < VECTOR_WIDTH; ++lane)
            accumulators[0][0][lane] += accumulators[0][lane_group][lane];
    auto result = 0.F;
    for (std::size_t index = vector_count * VECTOR_WIDTH; index < values.size(); ++index)
        result += values[index] * values[index];
    for (const auto value : accumulators[0][0]) result += value;
    return result;
}

/// Dequantizes one output tile at a time and immediately consumes it with SGEMM.
auto packed_linear(const tensor::Tensor& input, const tensor::Tensor& weight, const tensor::Tensor& scales,
                   std::int32_t bits, std::int32_t group_size, float input_scale, float output_scale)
    -> Result<tensor::Tensor> {
#if !defined(__APPLE__)
    return std::unexpected(Error{ErrorCode::UNSUPPORTED, "lowbit-parity currently requires Apple Accelerate"});
#else
    try {
        constexpr std::size_t OUTPUT_TILE = 192;
        const auto values_per_byte = 8 / bits;
        const auto width = weight.size(1) * values_per_byte;
        const auto outputs = weight.size(0), groups = width / group_size;
        const auto rows = input.numel() / width;
        const auto input_values = require(input.data<float>());
        const auto weight_bytes = require(weight.host_bytes());
        const auto scale_values = require(scales.data<float>());
        if (!std::ranges::all_of(scale_values, [](float scale) { return std::isfinite(scale) && scale > 0.F; }))
            throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "packed weight scales must be finite and positive"});
        std::vector<float> rounded_input(input_values.size()), output(rows * outputs);
        for (std::size_t index = 0; index < input_values.size(); ++index) {
            const auto code = std::clamp(std::nearbyint(input_values[index] / input_scale), -128.F, 127.F);
            rounded_input[index] = code * input_scale;
        }
        const auto* packed = reinterpret_cast<const std::uint8_t*>(weight_bytes.data());
        const auto mask = (1 << bits) - 1;
        const auto sign = 1 << (bits - 1);
        std::vector<float> tile(OUTPUT_TILE * width);
        for (std::size_t begin = 0; begin < outputs; begin += OUTPUT_TILE) {
            const auto count = std::min(OUTPUT_TILE, outputs - begin);
            for (std::size_t row = 0; row < count; ++row)
                for (std::size_t column = 0; column < width; ++column) {
                    const auto packed_column = column / values_per_byte;
                    auto code = (packed[(begin + row) * weight.size(1) + packed_column] >>
                                 ((column % values_per_byte) * bits)) &
                                mask;
                    if (code & sign) code -= 1 << bits;
                    tile[row * width + column] =
                        static_cast<float>(code) * scale_values[(begin + row) * groups + column / group_size];
                }
            if (rows >= 8) {
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, static_cast<int>(rows), static_cast<int>(count),
                            static_cast<int>(width), 1.F, rounded_input.data(), static_cast<int>(width), tile.data(),
                            static_cast<int>(width), 0.F, output.data() + begin, static_cast<int>(outputs));
            } else {
#pragma clang fp contract(off)
                for (std::size_t input_row = 0; input_row < rows; ++input_row)
                    for (std::size_t output_row = 0; output_row < count; ++output_row) {
                        auto value = 0.F;
                        for (std::size_t column = 0; column < width; ++column)
                            value = std::fma(rounded_input[input_row * width + column],
                                             tile[output_row * width + column], value);
                        output[input_row * outputs + begin + output_row] = value;
                    }
            }
        }
        for (auto& value : output) {
            const auto code = std::clamp(std::nearbyint(value / output_scale), -128.F, 127.F);
            value = code * output_scale;
        }
        std::vector<std::int64_t> shape(input.shape().begin(), input.shape().end());
        shape.back() = static_cast<std::int64_t>(outputs);
        return tensor::Tensor::from_host(std::move(shape), std::span<const float>(output));
    } catch (const ops::Failure& error) {
        return std::unexpected(error.error());
    }
#endif
}

} // namespace kidi::runtime::parity
