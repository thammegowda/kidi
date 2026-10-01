#include "ffn.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <span>
#include <stdexcept>

namespace kidi::bench {
namespace {

constexpr std::uint32_t GEMMA_LAYERS = 35;
constexpr std::uint32_t GEMMA_W4_LAYERS = 15;

class Random {
public:
    explicit Random(std::uint64_t seed) : state_(seed * 0x9E3779B97F4A7C15ULL + 0xD1B54A32D192ED03ULL) {}
    auto next() -> std::uint64_t {
        state_ ^= state_ << 13;
        state_ ^= state_ >> 7;
        state_ ^= state_ << 17;
        return state_;
    }

private:
    std::uint64_t state_;
};

auto rms(const std::vector<float>& values) -> double {
    double sum = 0;
    for (auto value : values) sum += static_cast<double>(value) * value;
    return std::sqrt(sum / static_cast<double>(values.size()));
}

// Scale mapping about four standard deviations to the INT8 range.
auto calibrated_scale(const std::vector<float>& values) -> float { return static_cast<float>(4 * rms(values) / 127); }

// Rescales per-channel weight scales (and the real values they produced) to unit RMS, like the RMSNorm between real
// Gemma MLPs, so activation scales stay stable across layers.
auto normalize(std::vector<float>& values, std::vector<float>& weight_scales) -> void {
    const auto factor = static_cast<float>(1 / rms(values));
    for (auto& value : values) value *= factor;
    for (auto& scale : weight_scales) scale *= factor;
}

auto matmul(const std::vector<std::int8_t>& input, const std::vector<std::int8_t>& weight, std::uint32_t m,
            std::uint32_t k, std::uint32_t n) -> std::vector<std::int32_t> {
    std::vector<std::int32_t> result(static_cast<std::size_t>(m) * n);
    for (std::uint32_t row = 0; row < m; ++row)
        for (std::uint32_t column = 0; column < n; ++column) {
            const auto* left = input.data() + static_cast<std::size_t>(row) * k;
            const auto* right = weight.data() + static_cast<std::size_t>(column) * k;
            std::int32_t sum = 0;
            for (std::uint32_t index = 0; index < k; ++index) sum += left[index] * right[index];
            result[static_cast<std::size_t>(row) * n + column] = sum;
        }
    return result;
}

} // namespace

auto FfnStack::max_intermediate() const -> std::uint32_t {
    std::uint32_t result = 0;
    for (const auto& layer : layers) result = std::max(result, layer.intermediate);
    return result;
}

auto gelu_multiply(std::int8_t gate, std::int8_t up, float gate_up_scale, float inverse_hidden_scale) -> std::int8_t {
    const float g = static_cast<float>(gate) * gate_up_scale;
    const float u = static_cast<float>(up) * gate_up_scale;
    const float activated = 0.5F * g * (1.F + std::tanh(0.7978845608F * (g + 0.044715F * g * g * g)));
    return static_cast<std::int8_t>(std::clamp(std::nearbyint(activated * u * inverse_hidden_scale), -128.F, 127.F));
}

auto ffn_weights(const FfnStack& stack, std::size_t layer, int matrix) -> std::vector<std::int8_t> {
    // Signed code frequencies measured in the Gemma 4 E2B QAT checkpoint (layer 0 and layer 15 gate projections).
    // Trained codes are bell-shaped and nearly zero-mean; uniform codes would add an unrealistic DC component that
    // amplifies small activation biases through every 2-bit fan-in.
    static constexpr std::array<double, 16> W4 = {0.0048, 0.0058, 0.0120, 0.0233, 0.0424, 0.0716, 0.1092, 0.1466,
                                                  0.1640, 0.1478, 0.1107, 0.0726, 0.0430, 0.0235, 0.0121, 0.0104};
    static constexpr std::array<double, 4> W2 = {0.0723, 0.2518, 0.3935, 0.2825};
    const auto& spec = stack.layers.at(layer);
    const auto count = static_cast<std::size_t>(matrix == 0 ? 2 * spec.intermediate : stack.hidden) *
                       (matrix == 0 ? stack.hidden : spec.intermediate);
    constexpr std::size_t TABLE = 1024;
    std::array<std::int8_t, TABLE> table{};
    const std::span<const double> frequencies =
        spec.bits == 4 ? std::span<const double>(W4) : std::span<const double>(W2);
    const int low = -(1 << (spec.bits - 1));
    double cumulative = 0;
    std::size_t filled = 0;
    for (std::size_t code = 0; code < frequencies.size(); ++code) {
        cumulative += frequencies[code];
        const auto end =
            code + 1 == frequencies.size() ? TABLE : static_cast<std::size_t>(std::lround(cumulative * TABLE));
        for (; filled < end && filled < TABLE; ++filled) table[filled] = static_cast<std::int8_t>(low + code);
    }
    std::vector<std::int8_t> values(count);
    Random random(layer * 2 + static_cast<std::uint64_t>(matrix) + 1);
    for (std::size_t index = 0; index < count; index += 6) {
        auto bits = random.next();
        for (std::size_t lane = 0; lane < 6 && index + lane < count; ++lane, bits >>= 10)
            values[index + lane] = table[bits % TABLE];
    }
    return values;
}

auto make_ffn_stack(std::uint32_t m, std::uint32_t layers) -> FfnStack {
    if (m == 0 || layers == 0 || layers > GEMMA_LAYERS)
        throw std::runtime_error("FFN stack needs M > 0 and 1-35 layers");
    FfnStack stack;
    stack.m = m;
    const auto w4_layers = static_cast<std::uint32_t>(std::lround(layers * GEMMA_W4_LAYERS / double(GEMMA_LAYERS)));
    for (std::uint32_t index = 0; index < layers; ++index) {
        FfnLayer layer;
        layer.bits = index < w4_layers ? 4 : 2;
        layer.intermediate = layer.bits == 4 ? 6144 : 12288;
        layer.gate_up_weight_scale.resize(2 * layer.intermediate);
        for (std::size_t channel = 0; channel < layer.gate_up_weight_scale.size(); ++channel)
            layer.gate_up_weight_scale[channel] = 0.002F * (1.F + static_cast<float>(channel % 13) / 13.F);
        layer.down_weight_scale.resize(stack.hidden);
        for (std::size_t channel = 0; channel < layer.down_weight_scale.size(); ++channel)
            layer.down_weight_scale[channel] = 0.002F * (1.F + static_cast<float>(channel % 11) / 11.F);
        stack.layers.push_back(std::move(layer));
    }

    Random random(0xF00D);
    for (auto& input : stack.inputs) {
        input.resize(static_cast<std::size_t>(m) * stack.hidden);
        for (auto& value : input) value = static_cast<std::int8_t>(static_cast<int>(random.next() % 255) - 127);
    }
    // Calibrate static scales on input A (like QAT activation scales), then apply the same scales to both inputs.
    auto activations = stack.inputs;
    float input_scale = 0.05F;
    for (std::size_t index = 0; index < stack.layers.size(); ++index) {
        auto& layer = stack.layers[index];
        const auto intermediate = layer.intermediate;
        layer.input_scale = input_scale;
        const auto gate_up = ffn_weights(stack, index, 0);
        std::array<std::vector<std::int32_t>, 2> accumulators;
        for (int input = 0; input < 2; ++input)
            accumulators[input] = matmul(activations[input], gate_up, m, stack.hidden, 2 * intermediate);
        std::vector<float> real(accumulators[0].size());
        for (std::size_t element = 0; element < real.size(); ++element)
            real[element] = static_cast<float>(accumulators[0][element]) * input_scale *
                            layer.gate_up_weight_scale[element % (2 * intermediate)];
        normalize(real, layer.gate_up_weight_scale);
        layer.gate_up_scale = calibrated_scale(real);
        layer.gate_up_factor.resize(2 * intermediate);
        for (std::uint32_t channel = 0; channel < 2 * intermediate; ++channel)
            layer.gate_up_factor[channel] = input_scale * layer.gate_up_weight_scale[channel] / layer.gate_up_scale;

        std::array<std::vector<std::int8_t>, 2> quantized_gate_up;
        for (int input = 0; input < 2; ++input) {
            quantized_gate_up[input].resize(accumulators[input].size());
            for (std::size_t element = 0; element < accumulators[input].size(); ++element)
                quantized_gate_up[input][element] =
                    requantize(accumulators[input][element], layer.gate_up_factor[element % (2 * intermediate)]);
        }
        std::vector<float> hidden_real(static_cast<std::size_t>(m) * intermediate);
        for (std::uint32_t row = 0; row < m; ++row)
            for (std::uint32_t channel = 0; channel < intermediate; ++channel) {
                const auto base = static_cast<std::size_t>(row) * 2 * intermediate;
                const float g = quantized_gate_up[0][base + channel] * layer.gate_up_scale;
                const float u = quantized_gate_up[0][base + intermediate + channel] * layer.gate_up_scale;
                hidden_real[static_cast<std::size_t>(row) * intermediate + channel] =
                    0.5F * g * (1.F + std::tanh(0.7978845608F * (g + 0.044715F * g * g * g))) * u;
            }
        layer.hidden_scale = calibrated_scale(hidden_real);
        const float inverse_hidden = 1.F / layer.hidden_scale;
        std::array<std::vector<std::int8_t>, 2> hidden;
        for (int input = 0; input < 2; ++input) {
            hidden[input].resize(static_cast<std::size_t>(m) * intermediate);
            for (std::uint32_t row = 0; row < m; ++row)
                for (std::uint32_t channel = 0; channel < intermediate; ++channel) {
                    const auto base = static_cast<std::size_t>(row) * 2 * intermediate;
                    hidden[input][static_cast<std::size_t>(row) * intermediate + channel] = gelu_multiply(
                        quantized_gate_up[input][base + channel],
                        quantized_gate_up[input][base + intermediate + channel], layer.gate_up_scale, inverse_hidden);
                }
        }

        const auto down = ffn_weights(stack, index, 1);
        for (int input = 0; input < 2; ++input)
            accumulators[input] = matmul(hidden[input], down, m, intermediate, stack.hidden);
        std::vector<float> output_real(accumulators[0].size());
        for (std::size_t element = 0; element < output_real.size(); ++element)
            output_real[element] = static_cast<float>(accumulators[0][element]) * layer.hidden_scale *
                                   layer.down_weight_scale[element % stack.hidden];
        normalize(output_real, layer.down_weight_scale);
        layer.output_scale = calibrated_scale(output_real);
        layer.down_factor.resize(stack.hidden);
        for (std::uint32_t channel = 0; channel < stack.hidden; ++channel)
            layer.down_factor[channel] = layer.hidden_scale * layer.down_weight_scale[channel] / layer.output_scale;
        for (int input = 0; input < 2; ++input)
            for (std::size_t element = 0; element < accumulators[input].size(); ++element)
                activations[input][element] =
                    requantize(accumulators[input][element], layer.down_factor[element % stack.hidden]);
        input_scale = layer.output_scale;
    }
    stack.expected = activations;
    return stack;
}

} // namespace kidi::bench
