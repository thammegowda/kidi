#include "lowbit.h"
#include "ffn.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cmath>
#include <iostream>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

namespace kidi::bench {

auto requantize(std::int32_t accumulator, float factor) -> std::int8_t {
    const auto value = std::nearbyint(static_cast<float>(accumulator) * factor);
    return static_cast<std::int8_t>(std::clamp(value, -128.F, 127.F));
}

auto make_problem(int bits, std::uint32_t m, std::uint32_t k, std::uint32_t n) -> Problem {
    if (bits != 2 && bits != 4 && bits != 8) throw std::runtime_error("bits must be 2, 4, or 8");
    if (m == 0 || k == 0 || n == 0 || k % 32 != 0 || n % 64 != 0)
        throw std::runtime_error("shape requires M > 0, K divisible by 32, and N divisible by 64");
    Problem problem{bits, m, k, n};
    std::uint64_t state = 0x9E3779B97F4A7C15ULL ^ (static_cast<std::uint64_t>(bits) << 56) ^
                          (static_cast<std::uint64_t>(m) << 36) ^ (static_cast<std::uint64_t>(k) << 18) ^ n;
    const auto next = [&] {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        return state;
    };
    problem.input.resize(static_cast<std::size_t>(m) * k);
    for (auto& value : problem.input) value = static_cast<std::int8_t>(static_cast<int>(next() % 255) - 127);
    // Whisper INT8 weights are symmetric; Gemma QAT W4/W2 weights use the full signed range.
    const int low = bits == 8 ? -127 : -(1 << (bits - 1));
    const int span = bits == 8 ? 255 : 1 << bits;
    problem.weight.resize(static_cast<std::size_t>(n) * k);
    for (auto& value : problem.weight) value = static_cast<std::int8_t>(low + static_cast<int>(next() % span));
    problem.weight_scale.resize(n);
    for (std::uint32_t column = 0; column < n; ++column)
        problem.weight_scale[column] = 0.002F * (1.F + static_cast<float>(column % 13) / 13.F);

    const auto rms = [](const std::vector<std::int8_t>& values) {
        double sum = 0;
        for (auto value : values) sum += static_cast<double>(value) * value;
        return std::sqrt(sum / static_cast<double>(values.size()));
    };
    const auto mean_scale = std::accumulate(problem.weight_scale.begin(), problem.weight_scale.end(), 0.0) / n;
    const auto sigma =
        std::sqrt(static_cast<double>(k)) * rms(problem.input) * rms(problem.weight) * problem.input_scale * mean_scale;
    // Map about four standard deviations of the output to the INT8 range.
    problem.output_scale = static_cast<float>(4 * sigma / 127);
    problem.factor.resize(n);
    for (std::uint32_t column = 0; column < n; ++column)
        problem.factor[column] = problem.input_scale * problem.weight_scale[column] / problem.output_scale;
    return problem;
}

auto reference_row(const Problem& problem, std::uint32_t row) -> std::vector<std::int8_t> {
    std::vector<std::int8_t> output(problem.n);
    const auto* input = problem.input.data() + static_cast<std::size_t>(row) * problem.k;
    for (std::uint32_t column = 0; column < problem.n; ++column) {
        const auto* weight = problem.weight.data() + static_cast<std::size_t>(column) * problem.k;
        std::int32_t accumulator = 0;
        for (std::uint32_t index = 0; index < problem.k; ++index) accumulator += input[index] * weight[index];
        output[column] = requantize(accumulator, problem.factor[column]);
    }
    return output;
}

} // namespace kidi::bench

namespace {

using kidi::bench::Measurement;
using kidi::bench::Options;
using kidi::bench::Problem;

auto median(std::vector<double> values) -> double {
    if (values.empty()) return 0;
    std::sort(values.begin(), values.end());
    const auto middle = values.size() / 2;
    return values.size() % 2 ? values[middle] : (values[middle - 1] + values[middle]) / 2;
}

auto verify(const Problem& problem, const Measurement& measurement) -> nlohmann::json {
    if (measurement.output.size() != static_cast<std::size_t>(problem.m) * problem.n)
        throw std::runtime_error("backend returned an output with the wrong size");
    std::vector<std::uint32_t> rows;
    constexpr std::uint32_t SAMPLES = 32;
    if (problem.m <= SAMPLES) {
        rows.resize(problem.m);
        std::iota(rows.begin(), rows.end(), 0);
    } else {
        for (std::uint32_t sample = 0; sample < SAMPLES; ++sample)
            rows.push_back(
                static_cast<std::uint32_t>(static_cast<std::uint64_t>(sample) * (problem.m - 1) / (SAMPLES - 1)));
    }
    std::size_t checked = 0, exact = 0, off_by_one = 0;
    int max_difference = 0;
    auto mismatches = nlohmann::json::array();
    for (auto row : rows) {
        const auto expected = kidi::bench::reference_row(problem, row);
        for (std::uint32_t column = 0; column < problem.n; ++column) {
            const auto actual =
                static_cast<int>(measurement.output[static_cast<std::size_t>(row) * problem.n + column]);
            const auto difference = std::abs(actual - expected[column]);
            ++checked;
            exact += difference == 0;
            off_by_one += difference == 1;
            max_difference = std::max(max_difference, difference);
            if (difference > 1 && (mismatches.empty() || mismatches.back()["row"] != row) && mismatches.size() < 32)
                mismatches.push_back(
                    {{"row", row}, {"column", column}, {"actual", actual}, {"expected", expected[column]}});
        }
    }
    return {{"checked_rows", rows.size()},
            {"checked_values", checked},
            {"exact_fraction", static_cast<double>(exact) / static_cast<double>(checked)},
            {"off_by_one", off_by_one},
            {"max_difference", max_difference},
            {"first_mismatches", mismatches},
            {"passed", max_difference <= 1}};
}

auto usage() -> void {
    std::cerr << "usage: kidi_lowbit_bench --backend cpu|gpu|npu --bits 8|4|2 --m M --k K --n N\n"
                 "       kidi_lowbit_bench --workload ffn --backend cpu|gpu|npu --mode MODE [--layers 35] [--m 1]\n"
                 "  [--label NAME] [--warmups N] [--runs N] [--threads N] [--queue-depth N]\n"
                 "  [--qnn-lib libQnnHtp.so] [--qnn-context FILE] [--qnn-prepare] [--qnn-memory shared|client]\n"
                 "  [--qnn-graph-layers N] (NPU replay: capture the step as chunks of N layers)\n"
                 "  [--qnn-weights container|packed] [--qnn-op fc|conv]\n"
                 "FFN modes: eager, replay (all); eager-async, encode, replay-queued (gpu)\n";
}

auto compare(const std::vector<std::int8_t>& actual, const std::vector<std::int8_t>& expected) -> nlohmann::json {
    if (actual.size() != expected.size()) throw std::runtime_error("backend returned an output with the wrong size");
    std::size_t exact = 0;
    int max_difference = 0;
    double squared = 0, signed_sum = 0;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        const auto signed_difference = static_cast<int>(actual[index]) - expected[index];
        const auto difference = std::abs(signed_difference);
        exact += difference == 0;
        max_difference = std::max(max_difference, difference);
        squared += static_cast<double>(difference) * difference;
        signed_sum += signed_difference;
    }
    const auto rms = std::sqrt(squared / static_cast<double>(actual.size()));
    // Gate fixed before measurement: the RMS error of the final INT8 output stays within one quantum.
    return {{"exact_fraction", static_cast<double>(exact) / static_cast<double>(actual.size())},
            {"max_difference", max_difference},
            {"mean_signed_quanta", signed_sum / static_cast<double>(actual.size())},
            {"rms_quanta", rms},
            {"passed", rms <= 1.0}};
}

auto fingerprint(const std::vector<std::int8_t>& values) -> std::string {
    std::uint64_t hash = 0xCBF29CE484222325ULL;
    for (auto value : values) hash = (hash ^ static_cast<std::uint8_t>(value)) * 0x100000001B3ULL;
    char text[17];
    std::snprintf(text, sizeof(text), "%016llx", static_cast<unsigned long long>(hash));
    return text;
}

auto run_ffn(const std::string& backend, std::uint32_t m, std::uint32_t layers, const Options& options,
             const std::string& label) -> int {
    const auto started = std::chrono::steady_clock::now();
    const auto stack = kidi::bench::make_ffn_stack(m, layers);
    const auto reference_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
    kidi::bench::FfnMeasurement measurement;
    if (backend == "cpu")
        measurement = kidi::bench::run_ffn_cpu(stack, options);
    else if (backend == "gpu")
        measurement = kidi::bench::run_ffn_gpu(stack, options);
    else if (backend == "npu")
        measurement = kidi::bench::run_ffn_npu(stack, options);
    else
        throw std::runtime_error("unknown backend " + backend);
    const std::array checks{compare(measurement.outputs[0], stack.expected[0]),
                            compare(measurement.outputs[1], stack.expected[1])};
    // Replay must observe in-place input updates: the two inputs must produce their own, different outputs. This is
    // only checkable while both reference outputs are live; the unnormalized fixture decays input B to zero in deep
    // stacks.
    const bool distinct = measurement.outputs[0] != measurement.outputs[1];
    const auto live = [](const std::vector<std::int8_t>& values) {
        return std::ranges::any_of(values, [](std::int8_t value) { return value != 0; });
    };
    const bool in_place_checkable =
        live(stack.expected[0]) && live(stack.expected[1]) && stack.expected[0] != stack.expected[1];
    const bool end_to_end = checks[0]["passed"].get<bool>() && checks[1]["passed"].get<bool>();
    // The unnormalized stack amplifies any rounding difference exponentially, so HTP (whose rounding differs from the
    // reference) is gated per stage: each graph against the reference applied to its own HTP inputs.
    std::optional<double> stage_rms;
    if (measurement.details.contains("diagnostics"))
        for (const auto& record : measurement.details["diagnostics"])
            for (const auto& [key, value] : record.items())
                if (value.is_object() && !key.starts_with("chain_"))
                    stage_rms = std::max(stage_rms.value_or(0.0), value["rms_quanta"].get<double>());
    const bool passed =
        (!in_place_checkable || distinct) && (backend == "npu" ? stage_rms.value_or(0.0) <= 1.0 : end_to_end);
    double weight_bytes = 0;
    std::size_t w4 = 0;
    for (const auto& layer : stack.layers) {
        weight_bytes += 3.0 * layer.intermediate * stack.hidden * layer.bits / 8;
        w4 += layer.bits == 4;
    }
    const auto step = median(measurement.step_ms);
    nlohmann::json report = {
        {"workload", "gemma4-e2b-ffn-decode"},
        {"label", label},
        {"backend", backend},
        {"mode", options.mode},
        {"device", measurement.device},
        {"layers", layers},
        {"w4_layers", w4},
        {"w2_layers", layers - w4},
        {"m", m},
        {"weight_bytes", weight_bytes},
        {"launches_per_step", measurement.launches_per_step},
        {"reference_ms", reference_ms},
        {"prepare_ms", measurement.prepare_ms},
        {"median_step_ms", step},
        {"min_step_ms",
         measurement.step_ms.empty() ? 0 : *std::min_element(measurement.step_ms.begin(), measurement.step_ms.end())},
        {"steps_per_second", 1000.0 / step},
        {"weight_gbps", weight_bytes / step / 1e6},
        {"step_ms", measurement.step_ms},
        {"checks", checks},
        {"end_to_end_passed", end_to_end},
        {"max_stage_rms_quanta", stage_rms ? nlohmann::json(*stage_rms) : nlohmann::json(nullptr)},
        {"inputs_distinguished", distinct},
        {"in_place_checked", in_place_checkable},
        {"outputs_live", {live(measurement.outputs[0]), live(measurement.outputs[1])}},
        {"output_fingerprints", {fingerprint(measurement.outputs[0]), fingerprint(measurement.outputs[1])}},
        {"passed", passed},
        {"details", measurement.details},
    };
    if (!measurement.device_ms.empty()) report["device_median_step_ms"] = median(measurement.device_ms);
    std::cout << report.dump() << std::endl;
    return passed ? 0 : 3;
}

} // namespace

auto main(int argc, char** argv) -> int {
    try {
        std::string backend, label, workload = "projection";
        int bits = 0;
        std::uint32_t m = 0, k = 0, n = 0, layers = 35;
        Options options;
        for (int index = 1; index < argc; ++index) {
            const std::string_view argument = argv[index];
            const auto next = [&]() -> std::string {
                if (index + 1 >= argc) throw std::runtime_error("missing value for " + std::string(argument));
                return argv[++index];
            };
            if (argument == "--backend")
                backend = next();
            else if (argument == "--label")
                label = next();
            else if (argument == "--bits")
                bits = std::stoi(next());
            else if (argument == "--m")
                m = static_cast<std::uint32_t>(std::stoul(next()));
            else if (argument == "--k")
                k = static_cast<std::uint32_t>(std::stoul(next()));
            else if (argument == "--n")
                n = static_cast<std::uint32_t>(std::stoul(next()));
            else if (argument == "--warmups")
                options.warmups = std::stoi(next());
            else if (argument == "--runs")
                options.runs = std::stoi(next());
            else if (argument == "--threads")
                options.threads = std::stoul(next());
            else if (argument == "--qnn-lib")
                options.qnn_library = next();
            else if (argument == "--qnn-context")
                options.qnn_context = next();
            else if (argument == "--qnn-prepare")
                options.qnn_prepare = true;
            else if (argument == "--qnn-weights")
                options.qnn_weights = next();
            else if (argument == "--qnn-op")
                options.qnn_op = next();
            else if (argument == "--qnn-memory")
                options.qnn_memory = next();
            else if (argument == "--qnn-graph-layers")
                options.qnn_graph_layers = std::stoi(next());
            else if (argument == "--workload")
                workload = next();
            else if (argument == "--mode")
                options.mode = next();
            else if (argument == "--layers")
                layers = static_cast<std::uint32_t>(std::stoul(next()));
            else if (argument == "--queue-depth")
                options.queue_depth = std::stoi(next());
            else {
                usage();
                return 2;
            }
        }
        if (workload == "ffn") {
            if (backend.empty() || options.runs < 1 || options.warmups < 0 || options.queue_depth < 1) {
                usage();
                return 2;
            }
            return run_ffn(backend, m == 0 ? 1 : m, layers, options, label);
        }
        if (workload != "projection" || backend.empty() || bits == 0 || m == 0 || k == 0 || n == 0 ||
            options.runs < 1 || options.warmups < 0) {
            usage();
            return 2;
        }
        const auto problem = kidi::bench::make_problem(bits, m, k, n);
        Measurement measurement;
        if (backend == "cpu")
            measurement = kidi::bench::run_cpu(problem, options);
        else if (backend == "gpu")
            measurement = kidi::bench::run_gpu(problem, options);
        else if (backend == "npu")
            measurement = kidi::bench::run_npu(problem, options);
        else
            throw std::runtime_error("unknown backend " + backend);

        const auto check = verify(problem, measurement);
        const auto wall = median(measurement.wall_ms);
        const auto operations = 2.0 * m * k * n;
        const auto weight_bytes = static_cast<double>(n) * k * bits / 8;
        nlohmann::json report = {
            {"label", label},
            {"backend", backend},
            {"device", measurement.device},
            {"bits", bits},
            {"precision", "W" + std::to_string(bits) + "A8"},
            {"m", m},
            {"k", k},
            {"n", n},
            {"weight_bytes", weight_bytes},
            {"prepare_ms", measurement.prepare_ms},
            {"median_ms", wall},
            {"min_ms", measurement.wall_ms.empty()
                           ? 0
                           : *std::min_element(measurement.wall_ms.begin(), measurement.wall_ms.end())},
            {"gops", operations / wall / 1e6},
            {"weight_gbps", weight_bytes / wall / 1e6},
            {"wall_ms", measurement.wall_ms},
            {"check", check},
            {"details", measurement.details},
        };
        if (!measurement.device_ms.empty()) report["device_median_ms"] = median(measurement.device_ms);
        std::cout << report.dump() << std::endl;
        return check["passed"].get<bool>() ? 0 : 3;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
