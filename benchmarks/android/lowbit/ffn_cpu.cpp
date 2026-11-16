#include "ffn.h"

#include "kidi/ops/context.h"
#include "kidi/runtime/ynn/graph.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
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
    ops::Context context(tensor::Device::cpu());

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

    auto make_input_f32 = [&](const std::vector<std::int8_t>& input_q, float scale) {
        std::vector<float> vals(input_q.size());
        for (std::size_t i = 0; i < input_q.size(); ++i) vals[i] = static_cast<float>(input_q[i]) * scale;
        return ops::require(tensor::Tensor::from_host(
            {1, static_cast<std::int64_t>(m), static_cast<std::int64_t>(hidden)}, std::span<const float>(vals)));
    };
    const auto run_stack = [&](const std::vector<std::int8_t>& input_q) {
        auto x = make_input_f32(input_q, stack.layers.front().input_scale);
        for (const auto& layer : prepared) {
            if (options.mode == "replay") {
                x = context.gated_feed_forward(x, layer.gate_up_weight, layer.gate_up_scale, layer.down_weight,
                                               layer.down_scale, layer.bits, static_cast<std::int32_t>(hidden),
                                               static_cast<std::int32_t>(layer.intermediate), layer.input_scale,
                                               layer.gate_up_scale_val, layer.hidden_scale, layer.output_scale);
            } else {
                const auto proj = context.packed_linear(x, layer.gate_up_weight, layer.gate_up_scale, layer.bits,
                                                        static_cast<std::int32_t>(hidden), layer.input_scale,
                                                        layer.gate_up_scale_val);
                const auto g = context.slice(proj, -1, 0, layer.intermediate);
                const auto u = context.slice(proj, -1, layer.intermediate, layer.intermediate);
                const auto h = context.static_round(context.gelu_multiply(g, u), layer.hidden_scale);
                x = context.packed_linear(h, layer.down_weight, layer.down_scale, layer.bits,
                                          static_cast<std::int32_t>(layer.intermediate), layer.hidden_scale,
                                          layer.output_scale);
            }
        }
        context.synchronize();
        return x;
    };

    // Warm one step so packed tile caches are included in prepare_ms.
    (void)run_stack(stack.inputs[0]);
    measurement.prepare_ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    measurement.launches_per_step = options.mode == "replay" ? layers : layers * 5;

    for (int iteration = 0; iteration < options.warmups + options.runs; ++iteration) {
        const auto begin = Clock::now();
        (void)run_stack(stack.inputs[iteration % 2]);
        const auto elapsed = std::chrono::duration<double, std::milli>(Clock::now() - begin).count();
        if (iteration >= options.warmups) measurement.step_ms.push_back(elapsed);
    }
    const float inv_out = 1.0F / stack.layers.back().output_scale;
    for (int input = 0; input < 2; ++input) {
        const auto out = run_stack(stack.inputs[input]);
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
