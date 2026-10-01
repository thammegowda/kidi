#include "AttentionMath.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>

using kidi::qnn::htp::fast_exp_negative;
using kidi::qnn::htp::online_softmax_weights;

int main() {
    for (int step = 0; step <= 1590; ++step) {
        const float input = -static_cast<float>(step) * 0.01f;
        const float expected = std::exp(input);
        const float actual = fast_exp_negative(input);
        assert(std::abs(actual - expected) <= std::max(2.0e-7f, expected * 1.0e-5f));
    }
    assert(fast_exp_negative(-16.0f) == 0.0f);
    assert(fast_exp_negative(0.25f) == 1.0f);

    constexpr std::array<float, 8> scores{{-4.0f, 1.0f, -0.5f, 3.0f, 2.0f, -8.0f, 3.5f, 0.25f}};
    constexpr std::array<float, 8> values{{2.0f, -1.0f, 4.0f, 0.5f, -3.0f, 1.0f, 2.5f, -0.25f}};

    float maximum = 0.0f;
    float denominator = 0.0f;
    float accumulator = 0.0f;
    for (std::size_t i = 0; i < scores.size(); ++i) {
        const auto weights = online_softmax_weights(maximum, scores[i], denominator == 0.0f);
        denominator = denominator * weights.previous + weights.current;
        accumulator = accumulator * weights.previous + values[i] * weights.current;
        maximum = weights.maximum;
    }

    const float reference_maximum = *std::max_element(scores.begin(), scores.end());
    float reference_denominator = 0.0f;
    float reference_accumulator = 0.0f;
    for (std::size_t i = 0; i < scores.size(); ++i) {
        const float weight = std::exp(scores[i] - reference_maximum);
        reference_denominator += weight;
        reference_accumulator += values[i] * weight;
    }

    assert(std::abs(accumulator / denominator - reference_accumulator / reference_denominator) < 1.0e-4f);
}
