#pragma once

#include <nlohmann/json.hpp>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace kidi::bench {

// One quantized projection: INT8 activations times signed `bits`-bit weights with per-output-channel scales,
// accumulated in INT32 and requantized to INT8 with a static output scale (the Gemma QAT contract).
struct Problem {
    int bits = 8;
    std::uint32_t m = 1;
    std::uint32_t k = 0;
    std::uint32_t n = 0;
    float input_scale = 0.05F;
    float output_scale = 0.F;
    std::vector<std::int8_t> input;  // [m][k]
    std::vector<std::int8_t> weight; // [n][k], one signed value per byte
    std::vector<float> weight_scale; // [n]
    std::vector<float> factor;       // [n] input_scale * weight_scale / output_scale
};

struct Options {
    int warmups = 3;
    int runs = 20;
    std::size_t threads = 4;
    std::string qnn_library;
    std::string qnn_context;
    bool qnn_prepare = false;
    std::string qnn_weights = "container";
    std::string qnn_op = "fc";
    std::string qnn_memory = "shared";
    int qnn_graph_layers = 0;
    std::string mode = "replay";
    int queue_depth = 16;
};

struct Measurement {
    std::string device;
    double prepare_ms = 0;
    std::vector<double> wall_ms;
    std::vector<double> device_ms;
    std::vector<std::int8_t> output; // [m][n]
    nlohmann::json details = nlohmann::json::object();
};

auto make_problem(int bits, std::uint32_t m, std::uint32_t k, std::uint32_t n) -> Problem;
auto requantize(std::int32_t accumulator, float factor) -> std::int8_t;
auto reference_row(const Problem& problem, std::uint32_t row) -> std::vector<std::int8_t>;

auto run_cpu(const Problem& problem, const Options& options) -> Measurement;
auto run_gpu(const Problem& problem, const Options& options) -> Measurement;
auto run_npu(const Problem& problem, const Options& options) -> Measurement;

} // namespace kidi::bench
