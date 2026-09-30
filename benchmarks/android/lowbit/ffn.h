#pragma once

#include "lowbit.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace kidi::bench {

// One Gemma 4 E2B QAT MLP: gate/up projection -> requantize -> gelu_tanh(gate) * up -> requantize -> down projection
// -> requantize. Gate and up share the static output scale and run as one [2 * intermediate] projection.
struct FfnLayer {
    int bits = 4;
    std::uint32_t intermediate = 0;
    float input_scale = 0;
    float gate_up_scale = 0;
    float hidden_scale = 0;
    float output_scale = 0;
    std::vector<float> gate_up_weight_scale; // [2 * intermediate]; gate channels first
    std::vector<float> down_weight_scale;    // [hidden]
    std::vector<float> gate_up_factor;       // input_scale * weight_scale / gate_up_scale
    std::vector<float> down_factor;          // hidden_scale * weight_scale / output_scale
};

// A decode-step FFN stack (no attention, normalization, or residuals) with statically calibrated activation scales.
// Without normalization the chain is a numerical fixture only: it amplifies rounding differences and, beyond a few
// layers, input B decays to zero. Compute and launch costs match the real MLPs exactly. Two inputs verify that replay
// reads in-place updated inputs while both remain live (shallow stacks).
struct FfnStack {
    std::uint32_t m = 1;
    std::uint32_t hidden = 1536;
    std::vector<FfnLayer> layers;
    std::array<std::vector<std::int8_t>, 2> inputs;   // [m][hidden]
    std::array<std::vector<std::int8_t>, 2> expected; // integer reference outputs
    auto max_intermediate() const -> std::uint32_t;
};

// Real Gemma 4 E2B MLP schedule scaled to `layers`: W4 with intermediate 6144 for the first 15/35 of layers, then W2
// with intermediate 12288.
auto make_ffn_stack(std::uint32_t m, std::uint32_t layers) -> FfnStack;

// Deterministic signed weights for one layer: matrix 0 is gate/up [2 * intermediate][hidden] (gate rows first) and
// matrix 1 is down [hidden][intermediate]. Regenerated on demand so all layers are never held unpacked at once.
auto ffn_weights(const FfnStack& stack, std::size_t layer, int matrix) -> std::vector<std::int8_t>;

auto gelu_multiply(std::int8_t gate, std::int8_t up, float gate_up_scale, float inverse_hidden_scale) -> std::int8_t;

struct FfnMeasurement {
    std::string device;
    double prepare_ms = 0;
    std::size_t launches_per_step = 0;
    std::vector<double> step_ms;
    std::vector<double> device_ms;
    std::array<std::vector<std::int8_t>, 2> outputs;
    nlohmann::json details = nlohmann::json::object();
};

auto run_ffn_cpu(const FfnStack& stack, const Options& options) -> FfnMeasurement;
auto run_ffn_gpu(const FfnStack& stack, const Options& options) -> FfnMeasurement;
auto run_ffn_npu(const FfnStack& stack, const Options& options) -> FfnMeasurement;

} // namespace kidi::bench
