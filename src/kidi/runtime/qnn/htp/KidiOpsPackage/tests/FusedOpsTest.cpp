#include "FusedOps.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace {

using namespace kidi::qnn::htp;

constexpr std::size_t GELU_COUNT = 2048;
alignas(128) std::array<std::uint8_t, GELU_COUNT> gate;
alignas(128) std::array<std::uint8_t, GELU_COUNT> up;
alignas(128) std::array<std::uint8_t, GELU_COUNT> expected_codes;
alignas(128) std::array<std::uint8_t, GELU_COUNT> actual_codes;

constexpr std::size_t MAX_WIDTH = 1536;
alignas(128) std::array<Float16, MAX_WIDTH> input;
alignas(128) std::array<Float16, MAX_WIDTH> weight;
alignas(128) std::array<Float16, MAX_WIDTH> residual;
alignas(128) std::array<Float16, MAX_WIDTH> expected;
alignas(128) std::array<Float16, MAX_WIDTH> actual;

auto next(std::uint32_t& state) -> std::uint32_t {
    state = state * 1664525U + 1013904223U;
    return state >> 8;
}

auto test_gelu(float input_scale, float output_scale) -> bool {
    std::uint32_t state = 17;
    for (std::size_t index = 0; index < GELU_COUNT; ++index) {
        gate[index] = static_cast<std::uint8_t>(next(state));
        up[index] = static_cast<std::uint8_t>(next(state));
    }
    gate[0] = 0;
    gate[1] = 255;
    up[0] = 255;
    up[1] = 0;
    gelu_multiply_scalar(gate.data(), up.data(), expected_codes.data(), GELU_COUNT, 128, input_scale, 128,
                         output_scale);
    gelu_multiply_hvx(gate.data(), up.data(), actual_codes.data(), GELU_COUNT, 128, input_scale, 128, output_scale);
    int worst = 0;
    std::size_t different = 0;
    for (std::size_t index = 0; index < GELU_COUNT; ++index) {
        const int delta = std::abs(static_cast<int>(expected_codes[index]) - static_cast<int>(actual_codes[index]));
        if (delta > worst) worst = delta;
        different += delta != 0;
        if (delta > 1) {
            std::printf("gelu mismatch index=%zu gate=%u up=%u expected=%u actual=%u\n", index, gate[index], up[index],
                        expected_codes[index], actual_codes[index]);
            return false;
        }
    }
    std::printf("gelu input_scale=%g output_scale=%g max_code_delta=%d off_by_one=%zu/%zu\n", input_scale, output_scale,
                worst, different, GELU_COUNT);
    return different * 100 < GELU_COUNT;
}

auto test_norm(std::size_t width, bool with_residual, float scale) -> bool {
    std::uint32_t state = 29 + static_cast<std::uint32_t>(width);
    for (std::size_t index = 0; index < width; ++index) {
        input[index] = Float16(static_cast<float>(static_cast<int>(next(state) % 4001) - 2000) / 7.0F);
        weight[index] = Float16(static_cast<float>(static_cast<int>(next(state) % 301)) / 150.0F);
        residual[index] = Float16(static_cast<float>(static_cast<int>(next(state) % 2001) - 1000) / 9.0F);
    }
    const auto* skip = with_residual ? residual.data() : nullptr;
    rms_norm_scalar(input.data(), weight.data(), skip, scale, expected.data(), width, 1e-6F);
    rms_norm_hvx(input.data(), weight.data(), skip, scale, actual.data(), width, 1e-6F);
    double squared = 0, reference = 0;
    for (std::size_t index = 0; index < width; ++index) {
        const double delta = static_cast<float>(actual[index]) - static_cast<float>(expected[index]);
        squared += delta * delta;
        reference += static_cast<double>(static_cast<float>(expected[index])) * static_cast<float>(expected[index]);
    }
    const double relative = std::sqrt(squared / reference);
    std::printf("rms_norm width=%zu residual=%d scale=%g relative_rms=%.3g\n", width, with_residual, scale, relative);
    return relative < 2e-3;
}

} // namespace

int main() {
    bool ok = test_gelu(0.05F, 0.02F);
    ok = test_gelu(0.011F, 0.004F) && ok;
    ok = test_gelu(0.2F, 0.5F) && ok;
    ok = test_norm(1536, false, 1.0F) && ok;
    ok = test_norm(1536, true, 1.0F) && ok;
    ok = test_norm(256, true, 0.75F) && ok;
    ok = test_norm(512, false, 1.0F) && ok;
    return ok ? 0 : 1;
}
