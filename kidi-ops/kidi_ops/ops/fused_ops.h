#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

#ifdef __hexagon__
#include "HTP/core/intrinsics.h"
#endif

namespace kidi::qnn::htp {

inline constexpr float GELU_SIGMOID_LINEAR = 1.5957691216057308F; // 2 * sqrt(2 / pi)
inline constexpr float GELU_SIGMOID_CUBIC = GELU_SIGMOID_LINEAR * 0.044715F;

// gelu_tanh(x) = x * sigmoid(2 * sqrt(2/pi) * (x + 0.044715 x^3)), identical to the CPU tanh form.
inline auto gelu_tanh(float x) -> float {
    const float y = std::clamp(x * (GELU_SIGMOID_LINEAR + GELU_SIGMOID_CUBIC * x * x), -40.0F, 40.0F);
    return x / (1.0F + std::exp(-y));
}

inline auto quantize_code(float value, std::int32_t zero_point) -> std::uint8_t {
    const auto rounded = static_cast<std::int32_t>(std::lround(value));
    return static_cast<std::uint8_t>(std::clamp<std::int32_t>(rounded, -128, 127) + zero_point);
}

/// Gate and up halves share one quantized FC output; the product is re-quantized for the down projection.
inline void gelu_multiply_scalar(const std::uint8_t* gate, const std::uint8_t* up, std::uint8_t* output,
                                 std::size_t count, std::int32_t input_zero, float input_scale,
                                 std::int32_t output_zero, float output_scale) {
    for (std::size_t index = 0; index < count; ++index) {
        const float x = static_cast<float>(static_cast<std::int32_t>(gate[index]) - input_zero) * input_scale;
        const float u = static_cast<float>(static_cast<std::int32_t>(up[index]) - input_zero) * input_scale;
        output[index] = quantize_code(gelu_tanh(x) * u / output_scale, output_zero);
    }
}

template <typename Half>
inline void rms_norm_scalar(const Half* input, const Half* weight, const Half* residual, float scale, Half* output,
                            std::size_t width, float epsilon) {
    float sum = 0.0F;
    for (std::size_t index = 0; index < width; ++index) {
        const float value = static_cast<float>(input[index]);
        sum += value * value;
    }
    const float inverse = 1.0F / std::sqrt(sum / static_cast<float>(width) + epsilon);
    for (std::size_t index = 0; index < width; ++index) {
        float value = static_cast<float>(input[index]) * inverse * static_cast<float>(weight[index]);
        if (residual) value = (static_cast<float>(residual[index]) + value) * scale;
        output[index] = Half(value);
    }
}

#ifdef __hexagon__

inline auto sf_bits(float value) -> std::int32_t {
    union {
        float value;
        std::int32_t bits;
    } cast{value};
    return cast.bits;
}
inline auto splat_sf(float value) -> HVX_Vector { return Q6_V_vsplat_R(sf_bits(value)); }
inline auto mul_sf(HVX_Vector left, HVX_Vector right) -> HVX_Vector {
    return Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(left, right));
}
inline auto add_sf(HVX_Vector left, HVX_Vector right) -> HVX_Vector {
    return Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(left, right));
}
inline auto sub_sf(HVX_Vector left, HVX_Vector right) -> HVX_Vector {
    return Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(left, right));
}

/// 1 / (1 + exp(-y)) for 32 FP32 lanes: exp2 range reduction, degree-7 polynomial, Newton reciprocal.
inline auto sigmoid_sf(HVX_Vector y) -> HVX_Vector {
    y = Q6_Vsf_vmin_VsfVsf(Q6_Vsf_vmax_VsfVsf(y, splat_sf(-40.0F)), splat_sf(40.0F));
    const auto t = mul_sf(y, splat_sf(-1.4426950408889634F));
    const auto whole = Q6_Vw_equals_Vsf(t);
    const auto z = mul_sf(sub_sf(t, Q6_Vsf_equals_Vw(whole)), splat_sf(0.6931471805599453F));
    auto p = splat_sf(1.0F / 5040.0F);
    for (const float coefficient : {1.0F / 720.0F, 1.0F / 120.0F, 1.0F / 24.0F, 1.0F / 6.0F, 0.5F, 1.0F, 1.0F})
        p = add_sf(mul_sf(p, z), splat_sf(coefficient));
    const auto exponential = Q6_Vw_vadd_VwVw(p, Q6_Vw_vasl_VwR(whole, 23));
    const auto denominator = add_sf(exponential, splat_sf(1.0F));
    auto reciprocal = Q6_Vw_vsub_VwVw(Q6_V_vsplat_R(0x7EF311C3), denominator);
    const auto two = splat_sf(2.0F);
    for (int iteration = 0; iteration < 3; ++iteration)
        reciprocal = mul_sf(reciprocal, sub_sf(two, mul_sf(denominator, reciprocal)));
    return reciprocal;
}

/// Quantized code for 32 lanes: gelu(gate) * up * scale, rounded half away from zero and offset by `zero`.
inline auto gelu_multiply_lanes(HVX_Vector gate_codes, HVX_Vector up_codes, HVX_Vector input_scale,
                                HVX_Vector product_scale, HVX_Vector zero) -> HVX_Vector {
    const auto x = mul_sf(Q6_Vsf_equals_Vw(gate_codes), input_scale);
    const auto y = mul_sf(x, add_sf(mul_sf(mul_sf(x, x), splat_sf(GELU_SIGMOID_CUBIC)), splat_sf(GELU_SIGMOID_LINEAR)));
    const auto value = mul_sf(mul_sf(x, sigmoid_sf(y)), mul_sf(Q6_Vsf_equals_Vw(up_codes), product_scale));
    const auto sign = Q6_V_vand_VV(value, Q6_V_vsplat_R(static_cast<std::int32_t>(0x80000000)));
    const auto rounded = Q6_Vw_equals_Vsf(add_sf(value, Q6_V_vor_VV(sign, splat_sf(0.5F))));
    const auto clamped = Q6_Vw_vmin_VwVw(Q6_Vw_vmax_VwVw(rounded, Q6_V_vsplat_R(-128)), Q6_V_vsplat_R(127));
    return Q6_Vw_vadd_VwVw(clamped, zero);
}

/// Processes `count` codes (a multiple of 128) with one HVX pass; no intermediate tensor is materialized.
inline void gelu_multiply_hvx(const std::uint8_t* gate, const std::uint8_t* up, std::uint8_t* output, std::size_t count,
                              std::int32_t input_zero, float input_scale, std::int32_t output_zero,
                              float output_scale) {
    const auto zero_in = Q6_Vh_vsplat_R(input_zero);
    const auto zero_out = Q6_V_vsplat_R(output_zero);
    const auto scale = splat_sf(input_scale);
    const auto product_scale = splat_sf(input_scale / output_scale);
    for (std::size_t offset = 0; offset < count; offset += 128) {
        const auto gate_half = Q6_Wuh_vunpack_Vub(q6op_V_vldu_A(gate + offset));
        const auto up_half = Q6_Wuh_vunpack_Vub(q6op_V_vldu_A(up + offset));
        HVX_Vector halves[2];
        for (int part = 0; part < 2; ++part) {
            const auto gate_signed = Q6_Vh_vsub_VhVh(part ? Q6_V_hi_W(gate_half) : Q6_V_lo_W(gate_half), zero_in);
            const auto up_signed = Q6_Vh_vsub_VhVh(part ? Q6_V_hi_W(up_half) : Q6_V_lo_W(up_half), zero_in);
            const auto gate_words = Q6_Ww_vunpack_Vh(gate_signed);
            const auto up_words = Q6_Ww_vunpack_Vh(up_signed);
            const auto low =
                gelu_multiply_lanes(Q6_V_lo_W(gate_words), Q6_V_lo_W(up_words), scale, product_scale, zero_out);
            const auto high =
                gelu_multiply_lanes(Q6_V_hi_W(gate_words), Q6_V_hi_W(up_words), scale, product_scale, zero_out);
            halves[part] = Q6_Vh_vpacke_VwVw(high, low);
        }
        q6op_vstu_AV(output + offset, Q6_Vb_vpacke_VhVh(halves[1], halves[0]));
    }
}

inline auto horizontal_sum_sf_qf32(HVX_Vector value) -> float {
    const auto zero = Q6_V_vzero();
    for (int shift = 64; shift >= 4; shift >>= 1)
        value = Q6_Vqf32_vadd_Vqf32Vqf32(value, Q6_V_vlalign_VVR(value, zero, shift));
    union {
        std::int32_t bits;
        float value;
    } cast{Q6_R_vextract_VR(Q6_Vsf_equals_Vqf32(value), 124)};
    return cast.value;
}

/// One row of RMSNorm (with RESIDUAL, `(residual + norm) * scale`) for widths that are multiples of 64.
template <bool RESIDUAL>
inline void rms_norm_hvx_row(const Float16* input, const Float16* weight, const Float16* residual, float scale,
                             Float16* output, std::size_t width, float epsilon) {
    auto sum_low = Q6_V_vzero();
    auto sum_high = Q6_V_vzero();
    for (std::size_t offset = 0; offset < width; offset += 64) {
        const auto value = q6op_V_vldu_A(input + offset);
        const auto square = Q6_Wqf32_vmpy_VhfVhf(value, value);
        sum_low = Q6_Vqf32_vadd_Vqf32Vqf32(sum_low, Q6_V_lo_W(square));
        sum_high = Q6_Vqf32_vadd_Vqf32Vqf32(sum_high, Q6_V_hi_W(square));
    }
    const float sum = horizontal_sum_sf_qf32(Q6_Vqf32_vadd_Vqf32Vqf32(sum_low, sum_high));
    const auto inverse = splat_sf(1.0F / std::sqrt(sum / static_cast<float>(width) + epsilon));
    const auto output_scale = splat_sf(scale);
    const auto one = q6op_V_vsplat_float16(Float16(1.0F));
    for (std::size_t offset = 0; offset < width; offset += 64) {
        const auto product = Q6_Wqf32_vmpy_VhfVhf(q6op_V_vldu_A(input + offset), q6op_V_vldu_A(weight + offset));
        HVX_Vector parts[2] = {Q6_V_lo_W(product), Q6_V_hi_W(product)};
        HVX_VectorPair skip{};
        if constexpr (RESIDUAL) skip = Q6_Wqf32_vmpy_VhfVhf(q6op_V_vldu_A(residual + offset), one);
        for (int part = 0; part < 2; ++part) {
            auto value = mul_sf(Q6_Vsf_equals_Vqf32(parts[part]), inverse);
            if constexpr (RESIDUAL) {
                value = add_sf(value, Q6_Vsf_equals_Vqf32(part ? Q6_V_hi_W(skip) : Q6_V_lo_W(skip)));
                value = mul_sf(value, output_scale);
            }
            parts[part] = Q6_Vqf32_vadd_VsfVsf(value, Q6_V_vzero());
        }
        q6op_vstu_AV(output + offset, Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(parts[1], parts[0])));
    }
}

inline void rms_norm_hvx(const Float16* input, const Float16* weight, const Float16* residual, float scale,
                         Float16* output, std::size_t width, float epsilon) {
    if (residual)
        rms_norm_hvx_row<true>(input, weight, residual, scale, output, width, epsilon);
    else
        rms_norm_hvx_row<false>(input, weight, nullptr, scale, output, width, epsilon);
}

#endif

} // namespace kidi::qnn::htp
