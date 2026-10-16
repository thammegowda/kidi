#pragma once

#include <cstdint>

namespace kidi::qnn::htp {

inline float fast_exp_negative(float value) {
    if (value >= 0.0f) return 1.0f;
    if (value <= -16.0f) return 0.0f;

    constexpr float LOG2_E = 1.4426950408889634f;
    constexpr float LN_2 = 0.6931471805599453f;
    const float exponent_value = value * LOG2_E;
    int exponent = static_cast<int>(exponent_value);
    if (static_cast<float>(exponent) > exponent_value) --exponent;

    const float remainder = value - static_cast<float>(exponent) * LN_2;
    const float polynomial =
        1.0f +
        remainder *
            (1.0f +
             remainder * (0.5f + remainder * (0.1666666716f +
                                              remainder * (0.0416666679f +
                                                           remainder * (0.0083333338f + remainder * 0.0013888889f)))));

    union {
        std::uint32_t bits;
        float value;
    } power_of_two{static_cast<std::uint32_t>(exponent + 127) << 23};
    return polynomial * power_of_two.value;
}

struct OnlineSoftmaxWeights {
    float previous;
    float current;
    float maximum;
};

inline auto online_softmax_weights(float maximum, float score, bool empty) -> OnlineSoftmaxWeights {
    if (empty) return {0.0f, 1.0f, score};
    if (score <= maximum) return {1.0f, fast_exp_negative(score - maximum), maximum};
    return {fast_exp_negative(maximum - score), 1.0f, score};
}

} // namespace kidi::qnn::htp
