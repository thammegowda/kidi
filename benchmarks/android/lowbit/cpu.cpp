#include "lowbit.h"

#include "kidi/runtime/ynn/graph.h"

#include <ynnpack.h>

#include <array>
#include <chrono>
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
auto pack_weights(const Problem& problem) -> std::vector<std::uint8_t> {
    const auto per_byte = static_cast<std::uint32_t>(8 / problem.bits);
    const auto row_bytes = problem.n / per_byte;
    const auto mask = static_cast<std::uint8_t>((1U << problem.bits) - 1);
    std::vector<std::uint8_t> packed(static_cast<std::size_t>(problem.k) * row_bytes);
    for (std::uint32_t column = 0; column < problem.n; ++column)
        for (std::uint32_t index = 0; index < problem.k; ++index) {
            const auto value =
                static_cast<std::uint8_t>(problem.weight[static_cast<std::size_t>(column) * problem.k + index]) & mask;
            packed[static_cast<std::size_t>(index) * row_bytes + column / per_byte] |=
                static_cast<std::uint8_t>(value << ((column % per_byte) * problem.bits));
        }
    return packed;
}

} // namespace

auto run_cpu(const Problem& problem, const Options& options) -> Measurement {
    Measurement measurement;
    measurement.device = "cpu-ynnpack";
    measurement.output.resize(static_cast<std::size_t>(problem.m) * problem.n);
    const auto start = Clock::now();
    const auto packed = pack_weights(problem);
    auto graph = value(runtime::ynn::Graph::create(2), "create YNNPACK graph");
    auto* subgraph = graph.get();
    const std::array<std::size_t, 2> input_shape{problem.m, problem.k};
    const std::array<std::size_t, 2> weight_shape{problem.k, problem.n};
    const std::array<std::size_t, 2> output_shape{problem.m, problem.n};
    const std::array<std::size_t, 1> factor_shape{problem.n};
    const auto weight_type = problem.bits == 2 ? ynn_type_int2 : problem.bits == 4 ? ynn_type_int4 : ynn_type_int8;
    std::uint32_t input_id = YNN_INVALID_VALUE_ID, output_id = YNN_INVALID_VALUE_ID;
    std::uint32_t weight_id = YNN_INVALID_VALUE_ID, factor_id = YNN_INVALID_VALUE_ID;
    std::uint32_t accumulator_id = YNN_INVALID_VALUE_ID, scaled_id = YNN_INVALID_VALUE_ID;
    std::uint32_t zero_id = YNN_INVALID_VALUE_ID, unit_id = YNN_INVALID_VALUE_ID;
    const std::int32_t zero = 0;
    const float unit = 1.F;
    check(ynn_define_tensor(subgraph, ynn_type_int8, input_shape.size(), input_shape.data(), nullptr,
                            YNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id),
          "define input");
    check(ynn_define_tensor(subgraph, ynn_type_int8, output_shape.size(), output_shape.data(), nullptr,
                            YNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id),
          "define output");
    check(ynn_define_tensor(subgraph, weight_type, weight_shape.size(), weight_shape.data(), packed.data(),
                            YNN_VALUE_FLAG_COPY_DATA, &weight_id),
          "define weights");
    check(ynn_define_tensor(subgraph, ynn_type_fp32, factor_shape.size(), factor_shape.data(), problem.factor.data(),
                            YNN_VALUE_FLAG_COPY_DATA, &factor_id),
          "define requantization factors");
    check(ynn_define_tensor(subgraph, ynn_type_int32, 0, nullptr, &zero, YNN_VALUE_FLAG_COPY_DATA, &zero_id),
          "define zero point");
    check(ynn_define_tensor(subgraph, ynn_type_fp32, 0, nullptr, &unit, YNN_VALUE_FLAG_COPY_DATA, &unit_id),
          "define unit scale");
    check(ynn_define_dot(subgraph, 1, input_id, weight_id, YNN_INVALID_VALUE_ID, &accumulator_id, 0),
          "define integer dot");
    check(ynn_define_binary(subgraph, ynn_binary_multiply, accumulator_id, factor_id, &scaled_id, 0),
          "define channel scale");
    check(ynn_define_quantize(subgraph, scaled_id, ynn_type_int8, zero_id, unit_id, &output_id, 0),
          "define output quantization");
    auto executable = value(std::move(graph).compile(options.threads), "compile YNNPACK graph");
    value(executable.reshape(), "reshape YNNPACK graph");
    value(executable.bind(input_id, const_cast<std::int8_t*>(problem.input.data())), "bind input");
    value(executable.bind(output_id, measurement.output.data()), "bind output");
    measurement.prepare_ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();

    for (int iteration = 0; iteration < options.warmups + options.runs; ++iteration) {
        const auto begin = Clock::now();
        value(executable.invoke(), "run YNNPACK graph");
        const auto elapsed = std::chrono::duration<double, std::milli>(Clock::now() - begin).count();
        if (iteration >= options.warmups) measurement.wall_ms.push_back(elapsed);
    }
    measurement.details = {{"threads", options.threads}, {"arch", runtime::ynn::supported_arch_names()}};
    return measurement;
}

} // namespace kidi::bench
