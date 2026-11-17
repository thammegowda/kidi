#include "ffn.h"

#include "kidi/ops/context.h"
#include "kidi/runtime/ynn/graph.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <span>
#include <stdexcept>
#include <string>

namespace kidi::bench {
namespace {

using Clock = std::chrono::steady_clock;

auto pack_subbyte(const std::int8_t* rows, std::uint32_t n, std::uint32_t k, int bits) -> tensor::Tensor {
    const auto per_byte = static_cast<std::uint32_t>(8 / bits);
    const auto row_bytes = k / per_byte;
    const auto mask = static_cast<std::uint8_t>((1U << bits) - 1);
    std::vector<std::uint8_t> packed(static_cast<std::size_t>(n) * row_bytes, 0);
    for (std::uint32_t row = 0; row < n; ++row) {
        for (std::uint32_t col = 0; col < k; ++col) {
            const auto weight = static_cast<std::uint8_t>(rows[static_cast<std::size_t>(row) * k + col]) & mask;
            packed[static_cast<std::size_t>(row) * row_bytes + col / per_byte] |=
                static_cast<std::uint8_t>(weight << ((col % per_byte) * bits));
        }
    }
    return ops::require(tensor::Tensor::from_host({static_cast<std::int64_t>(n), static_cast<std::int64_t>(row_bytes)},
                                                  std::span<const std::uint8_t>(packed)));
}

struct PreparedLayer {
    tensor::Tensor gate_up_weight;
    tensor::Tensor gate_up_scale;
    tensor::Tensor down_weight;
    tensor::Tensor down_scale;
    std::uint32_t intermediate = 0;
    int bits = 4;
    float input_scale = 0;
    float gate_up_scale_val = 0;
    float hidden_scale = 0;
    float output_scale = 0;
};

} // namespace

auto run_ffn_cpu(const FfnStack& stack, const Options& options) -> FfnMeasurement {
    if (options.mode != "eager" && options.mode != "replay")
        throw std::runtime_error("CPU FFN modes are eager and replay");
    FfnMeasurement measurement;
    measurement.device = "cpu-ynnpack";
    const auto m = stack.m, hidden = stack.hidden;
    const auto layers = stack.layers.size();
    runtime::ynn::set_thread_count(options.threads);
    const bool replay_mode = options.mode == "replay";
    ops::Context context(tensor::Device::cpu(), true, replay_mode);

    const auto start = Clock::now();
    std::vector<PreparedLayer> prepared;
    prepared.reserve(layers);
    for (std::size_t index = 0; index < layers; ++index) {
        const auto& layer = stack.layers[index];
        const auto gate_up = ffn_weights(stack, index, 0);
        const auto down = ffn_weights(stack, index, 1);
        prepared.push_back({
            pack_subbyte(gate_up.data(), 2 * layer.intermediate, hidden, layer.bits),
            ops::require(tensor::Tensor::from_host({static_cast<std::int64_t>(2 * layer.intermediate), 1},
                                                   std::span<const float>(layer.gate_up_weight_scale))),
            pack_subbyte(down.data(), hidden, layer.intermediate, layer.bits),
            ops::require(tensor::Tensor::from_host({static_cast<std::int64_t>(hidden), 1},
                                                   std::span<const float>(layer.down_weight_scale))),
            layer.intermediate,
            layer.bits,
            layer.input_scale,
            layer.gate_up_scale,
            layer.hidden_scale,
            layer.output_scale,
        });
    }

    std::array<std::vector<float>, 2> inputs_f32;
    const float first_input_scale = stack.layers.front().input_scale;
    for (std::size_t idx = 0; idx < 2; ++idx) {
        inputs_f32[idx].resize(stack.inputs[idx].size());
        for (std::size_t i = 0; i < stack.inputs[idx].size(); ++i)
            inputs_f32[idx][i] = static_cast<float>(stack.inputs[idx][i]) * first_input_scale;
    }
    auto x_input = ops::require(tensor::Tensor::zeros(
        {1, static_cast<std::int64_t>(m), static_cast<std::int64_t>(hidden)}, tensor::DType::F32));
    auto* x_ptr = ops::require(x_input.data<float>()).data();
    const std::size_t x_bytes = inputs_f32[0].size() * sizeof(float);

    const auto exec_stack = [&](const tensor::Tensor& initial) {
        auto x = initial;
        for (const auto& layer : prepared) {
            const auto proj =
                context.packed_linear(x, layer.gate_up_weight, layer.gate_up_scale, layer.bits,
                                      static_cast<std::int32_t>(hidden), layer.input_scale, layer.gate_up_scale_val);
            const auto g = context.slice(proj, -1, 0, layer.intermediate);
            const auto u = context.slice(proj, -1, layer.intermediate, layer.intermediate);
            const auto h = context.static_round(context.gelu_multiply(g, u), layer.hidden_scale);
            x = context.packed_linear(h, layer.down_weight, layer.down_scale, layer.bits,
                                      static_cast<std::int32_t>(layer.intermediate), layer.hidden_scale,
                                      layer.output_scale);
        }
        return x;
    };
    const auto run_stack = [&]() {
        tensor::Tensor out;
        if (replay_mode) {
            const auto outputs = context.replay(
                "ffn_cpu_stack", std::span<const tensor::Tensor>(&x_input, 1),
                [&](std::span<const tensor::Tensor> in) { return std::vector<tensor::Tensor>{exec_stack(in[0])}; },
                false);
            out = outputs[0];
        } else {
            out = exec_stack(x_input);
        }
        context.synchronize();
        return out;
    };

    // Warm one step so packed tile caches and replay capture are included in prepare_ms.
    std::memcpy(x_ptr, inputs_f32[0].data(), x_bytes);
    (void)run_stack();
    measurement.prepare_ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    measurement.launches_per_step = replay_mode ? 1 : layers * 6;

    for (int iteration = 0; iteration < options.warmups + options.runs; ++iteration) {
        std::memcpy(x_ptr, inputs_f32[iteration % 2].data(), x_bytes);
        const auto begin = Clock::now();
        (void)run_stack();
        const auto elapsed = std::chrono::duration<double, std::milli>(Clock::now() - begin).count();
        if (iteration >= options.warmups) measurement.step_ms.push_back(elapsed);
    }
    const float inv_out = 1.0F / stack.layers.back().output_scale;
    for (int input = 0; input < 2; ++input) {
        std::memcpy(x_ptr, inputs_f32[input].data(), x_bytes);
        const auto out = run_stack();
        const auto vals = ops::require(out.data<float>());
        measurement.outputs[input].resize(vals.size());
        for (std::size_t i = 0; i < vals.size(); ++i) {
            measurement.outputs[input][i] =
                static_cast<std::int8_t>(std::clamp(std::nearbyint(vals[i] * inv_out), -128.0F, 127.0F));
        }
    }
    measurement.details = {{"threads", options.threads}, {"arch", runtime::ynn::supported_arch_names()}};
    return measurement;
}

} // namespace kidi::bench
