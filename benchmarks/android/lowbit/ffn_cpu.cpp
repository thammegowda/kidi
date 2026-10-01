#include "ffn.h"

#include "kidi/runtime/ynn/graph.h"

#include <ynnpack.h>

#include <array>
#include <chrono>
#include <cstring>
#include <span>
#include <stdexcept>
#include <string>

namespace kidi::bench {
namespace {

using Clock = std::chrono::steady_clock;

auto check(ynn_status status, const char* operation) -> void {
    if (status != ynn_status_success)
        throw std::runtime_error(std::string(operation) + ": YNNPACK status " + std::to_string(status));
}

template <typename T>
auto value(Result<T> result, const char* operation) -> T {
    if (!result) throw std::runtime_error(std::string(operation) + ": " + result.error().message);
    return std::move(*result);
}

auto value(Result<void> result, const char* operation) -> void {
    if (!result) throw std::runtime_error(std::string(operation) + ": " + result.error().message);
}

// YNNPACK dot expects B as [K, N]; sub-byte types pack consecutive N values into each byte, low bits first.
auto pack_kn(const std::int8_t* rows, std::uint32_t n, std::uint32_t k, int bits) -> std::vector<std::uint8_t> {
    const auto per_byte = static_cast<std::uint32_t>(8 / bits);
    const auto row_bytes = n / per_byte;
    const auto mask = static_cast<std::uint8_t>((1U << bits) - 1);
    std::vector<std::uint8_t> packed(static_cast<std::size_t>(k) * row_bytes);
    for (std::uint32_t column = 0; column < n; ++column)
        for (std::uint32_t index = 0; index < k; ++index) {
            const auto weight = static_cast<std::uint8_t>(rows[static_cast<std::size_t>(column) * k + index]) & mask;
            packed[static_cast<std::size_t>(index) * row_bytes + column / per_byte] |=
                static_cast<std::uint8_t>(weight << ((column % per_byte) * bits));
        }
    return packed;
}

class Builder {
public:
    explicit Builder(ynn_subgraph_t graph) : graph_(graph) {}

    auto external(std::size_t rows, std::size_t columns, std::uint32_t flags) -> std::uint32_t {
        const std::array<std::size_t, 2> shape{rows, columns};
        std::uint32_t id = YNN_INVALID_VALUE_ID;
        check(ynn_define_tensor(graph_, ynn_type_int8, shape.size(), shape.data(), nullptr, flags, &id),
              "define external");
        return id;
    }

    auto scalar(float number) -> std::uint32_t {
        std::uint32_t id = YNN_INVALID_VALUE_ID;
        check(ynn_define_tensor(graph_, ynn_type_fp32, 0, nullptr, &number, YNN_VALUE_FLAG_COPY_DATA, &id),
              "define scalar");
        return id;
    }

    auto binary(ynn_binary_operator operation, std::uint32_t left, std::uint32_t right) -> std::uint32_t {
        std::uint32_t id = YNN_INVALID_VALUE_ID;
        check(ynn_define_binary(graph_, operation, left, right, &id, 0), "define binary");
        return id;
    }

    // INT8 x packed weights -> INT32 -> per-channel factor -> INT8 (round to nearest even, saturating).
    auto projection(std::uint32_t input, const std::int8_t* rows, std::uint32_t k, std::uint32_t n, int bits,
                    std::span<const float> factor, std::uint32_t output = YNN_INVALID_VALUE_ID) -> std::uint32_t {
        const auto packed = pack_kn(rows, n, k, bits);
        const std::array<std::size_t, 2> weight_shape{k, n};
        const std::array<std::size_t, 1> factor_shape{n};
        const auto type = bits == 2 ? ynn_type_int2 : bits == 4 ? ynn_type_int4 : ynn_type_int8;
        std::uint32_t weight = YNN_INVALID_VALUE_ID, scale = YNN_INVALID_VALUE_ID, accumulator = YNN_INVALID_VALUE_ID;
        check(ynn_define_tensor(graph_, type, weight_shape.size(), weight_shape.data(), packed.data(),
                                YNN_VALUE_FLAG_COPY_DATA, &weight),
              "define weights");
        check(ynn_define_tensor(graph_, ynn_type_fp32, factor_shape.size(), factor_shape.data(), factor.data(),
                                YNN_VALUE_FLAG_COPY_DATA, &scale),
              "define factors");
        check(ynn_define_dot(graph_, 1, input, weight, YNN_INVALID_VALUE_ID, &accumulator, 0), "define dot");
        return quantize(binary(ynn_binary_multiply, accumulator, scale), output);
    }

    // Same float evaluation order as the host reference: ((0.5 * g) * (1 + tanh(c * (g + 0.044715 * g^3)))) * u / s.
    auto gelu_multiply(std::uint32_t gate, std::uint32_t up, float scale, float inverse_hidden,
                       std::uint32_t output = YNN_INVALID_VALUE_ID) -> std::uint32_t {
        const auto g = dequantize(gate, scale), u = dequantize(up, scale);
        const auto cube = binary(ynn_binary_multiply, binary(ynn_binary_multiply, g, g), g);
        const auto inner = binary(ynn_binary_add, g, binary(ynn_binary_multiply, cube, scalar(0.044715F)));
        std::uint32_t tanh = YNN_INVALID_VALUE_ID;
        check(ynn_define_unary(graph_, ynn_unary_tanh, binary(ynn_binary_multiply, inner, scalar(0.7978845608F)), &tanh,
                               0),
              "define tanh");
        const auto activated = binary(ynn_binary_multiply, binary(ynn_binary_multiply, g, scalar(0.5F)),
                                      binary(ynn_binary_add, scalar(1.F), tanh));
        const auto product = binary(ynn_binary_multiply, activated, u);
        return quantize(binary(ynn_binary_multiply, product, scalar(inverse_hidden)), output);
    }

private:
    auto zero() -> std::uint32_t {
        if (zero_ == YNN_INVALID_VALUE_ID) {
            const std::int32_t origin = 0;
            check(ynn_define_tensor(graph_, ynn_type_int32, 0, nullptr, &origin, YNN_VALUE_FLAG_COPY_DATA, &zero_),
                  "define zero point");
        }
        return zero_;
    }

    auto quantize(std::uint32_t input, std::uint32_t output) -> std::uint32_t {
        check(ynn_define_quantize(graph_, input, ynn_type_int8, zero(), scalar(1.F), &output, 0), "define quantize");
        return output;
    }

    auto dequantize(std::uint32_t input, float scale) -> std::uint32_t {
        std::uint32_t id = YNN_INVALID_VALUE_ID;
        check(ynn_define_dequantize(graph_, input, zero(), scalar(scale), ynn_type_fp32, &id, 0), "define dequantize");
        return id;
    }

    ynn_subgraph_t graph_;
    std::uint32_t zero_ = YNN_INVALID_VALUE_ID;
};

struct Buffers {
    std::vector<std::int8_t> input, gate, up, hidden;
    std::array<std::vector<std::int8_t>, 2> outputs;
};

} // namespace

auto run_ffn_cpu(const FfnStack& stack, const Options& options) -> FfnMeasurement {
    if (options.mode != "eager" && options.mode != "replay")
        throw std::runtime_error("CPU FFN modes are eager and replay");
    FfnMeasurement measurement;
    measurement.device = "cpu-ynnpack";
    const auto m = stack.m, hidden = stack.hidden;
    const auto rows = [&](std::size_t columns) { return static_cast<std::size_t>(m) * columns; };
    Buffers buffers{std::vector<std::int8_t>(rows(hidden)),
                    std::vector<std::int8_t>(rows(stack.max_intermediate())),
                    std::vector<std::int8_t>(rows(stack.max_intermediate())),
                    std::vector<std::int8_t>(rows(stack.max_intermediate())),
                    {std::vector<std::int8_t>(rows(hidden)), std::vector<std::int8_t>(rows(hidden))}};
    const auto layers = stack.layers.size();
    auto* final_output = buffers.outputs[(layers - 1) % 2].data();

    const auto start = Clock::now();
    std::vector<runtime::ynn::Executable> executables;
    const auto compile = [&](runtime::ynn::Graph graph, std::initializer_list<std::pair<std::uint32_t, void*>> binds) {
        auto executable = value(std::move(graph).compile(options.threads), "compile YNNPACK graph");
        value(executable.reshape(), "reshape YNNPACK graph");
        for (const auto& [id, data] : binds) value(executable.bind(id, data), "bind buffer");
        executables.push_back(std::move(executable));
    };
    if (options.mode == "replay") {
        auto graph = value(runtime::ynn::Graph::create(2), "create graph");
        Builder builder(graph.get());
        const auto input = builder.external(m, hidden, YNN_VALUE_FLAG_EXTERNAL_INPUT);
        const auto output = builder.external(m, hidden, YNN_VALUE_FLAG_EXTERNAL_OUTPUT);
        auto x = input;
        for (std::size_t index = 0; index < layers; ++index) {
            const auto& layer = stack.layers[index];
            const auto intermediate = layer.intermediate;
            const auto gate_up = ffn_weights(stack, index, 0);
            const std::span<const float> factors(layer.gate_up_factor);
            const auto gate =
                builder.projection(x, gate_up.data(), hidden, intermediate, layer.bits, factors.first(intermediate));
            const auto up = builder.projection(x, gate_up.data() + static_cast<std::size_t>(intermediate) * hidden,
                                               hidden, intermediate, layer.bits, factors.subspan(intermediate));
            const auto activation = builder.gelu_multiply(gate, up, layer.gate_up_scale, 1.F / layer.hidden_scale);
            const auto down = ffn_weights(stack, index, 1);
            x = builder.projection(activation, down.data(), intermediate, hidden, layer.bits, layer.down_factor,
                                   index + 1 == layers ? output : YNN_INVALID_VALUE_ID);
        }
        compile(std::move(graph), {{input, buffers.input.data()}, {output, final_output}});
    } else {
        for (std::size_t index = 0; index < layers; ++index) {
            const auto& layer = stack.layers[index];
            const auto intermediate = layer.intermediate;
            auto* layer_input = index == 0 ? buffers.input.data() : buffers.outputs[(index - 1) % 2].data();
            {
                auto graph = value(runtime::ynn::Graph::create(3), "create graph");
                Builder builder(graph.get());
                const auto x = builder.external(m, hidden, YNN_VALUE_FLAG_EXTERNAL_INPUT);
                const auto gate = builder.external(m, intermediate, YNN_VALUE_FLAG_EXTERNAL_OUTPUT);
                const auto up = builder.external(m, intermediate, YNN_VALUE_FLAG_EXTERNAL_OUTPUT);
                const auto gate_up = ffn_weights(stack, index, 0);
                const std::span<const float> factors(layer.gate_up_factor);
                builder.projection(x, gate_up.data(), hidden, intermediate, layer.bits, factors.first(intermediate),
                                   gate);
                builder.projection(x, gate_up.data() + static_cast<std::size_t>(intermediate) * hidden, hidden,
                                   intermediate, layer.bits, factors.subspan(intermediate), up);
                compile(std::move(graph), {{x, layer_input}, {gate, buffers.gate.data()}, {up, buffers.up.data()}});
            }
            {
                auto graph = value(runtime::ynn::Graph::create(3), "create graph");
                Builder builder(graph.get());
                const auto gate = builder.external(m, intermediate, YNN_VALUE_FLAG_EXTERNAL_INPUT);
                const auto up = builder.external(m, intermediate, YNN_VALUE_FLAG_EXTERNAL_INPUT);
                const auto activation = builder.external(m, intermediate, YNN_VALUE_FLAG_EXTERNAL_OUTPUT);
                builder.gelu_multiply(gate, up, layer.gate_up_scale, 1.F / layer.hidden_scale, activation);
                compile(std::move(graph),
                        {{gate, buffers.gate.data()}, {up, buffers.up.data()}, {activation, buffers.hidden.data()}});
            }
            {
                auto graph = value(runtime::ynn::Graph::create(2), "create graph");
                Builder builder(graph.get());
                const auto activation = builder.external(m, intermediate, YNN_VALUE_FLAG_EXTERNAL_INPUT);
                const auto output = builder.external(m, hidden, YNN_VALUE_FLAG_EXTERNAL_OUTPUT);
                const auto down = ffn_weights(stack, index, 1);
                builder.projection(activation, down.data(), intermediate, hidden, layer.bits, layer.down_factor,
                                   output);
                compile(std::move(graph),
                        {{activation, buffers.hidden.data()}, {output, buffers.outputs[index % 2].data()}});
            }
        }
    }
    measurement.prepare_ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    measurement.launches_per_step = executables.size();

    const auto step = [&](const std::vector<std::int8_t>& input) {
        std::memcpy(buffers.input.data(), input.data(), input.size());
        const auto begin = Clock::now();
        for (auto& executable : executables) value(executable.invoke(), "run YNNPACK graph");
        return std::chrono::duration<double, std::milli>(Clock::now() - begin).count();
    };
    for (int iteration = 0; iteration < options.warmups + options.runs; ++iteration) {
        const auto elapsed = step(stack.inputs[iteration % 2]);
        if (iteration >= options.warmups) measurement.step_ms.push_back(elapsed);
    }
    for (int input = 0; input < 2; ++input) {
        step(stack.inputs[input]);
        measurement.outputs[input].assign(final_output, final_output + rows(hidden));
    }
    measurement.details = {{"threads", options.threads}, {"arch", runtime::ynn::supported_arch_names()}};
    return measurement;
}

} // namespace kidi::bench
