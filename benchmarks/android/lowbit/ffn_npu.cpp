#include "ffn.h"

#include <stdexcept>

#if defined(KIDI_BENCH_QNN)
#include "qnn.h"

#include <QnnOpDef.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <optional>
#include <string>
#include <vector>
#endif

namespace kidi::bench {

#if !defined(KIDI_BENCH_QNN)

auto run_ffn_npu(const FfnStack&, const Options&) -> FfnMeasurement {
    throw std::runtime_error("built without KIDI_QNN_SDK; the Qualcomm NPU backend is unavailable");
}

#else

namespace {

using Clock = std::chrono::steady_clock;

auto elapsed_ms(Clock::time_point start) -> double {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

// Graph boundary buffers. Gate, up, and hidden buffers exist per intermediate width so each has one registered shape.
// Captured chunks exchange activations through dedicated BOUNDARY + chunk buffers, so no graph aliases input and
// output.
enum class Slot : int { INPUT, OUTPUT0, OUTPUT1, GATE, UP, HIDDEN, BOUNDARY = 100 };

struct Io {
    std::string name;
    Slot slot;
    std::uint32_t width;
    float scale;
};

struct GraphPlan {
    std::string name;
    std::size_t layer;
    int stage; // 0 gate/up, 1 activation, 2 down, -1 captured layers [layer, layer + layers)
    std::vector<Io> inputs, outputs;
    std::size_t layers = 1;
};

auto output_slot(std::size_t layer) -> Slot { return layer % 2 ? Slot::OUTPUT1 : Slot::OUTPUT0; }

auto boundary_slot(std::size_t chunk) -> Slot { return static_cast<Slot>(static_cast<int>(Slot::BOUNDARY) + chunk); }

// Replay captures the step as consecutive chunks of `graph_layers` layers (one chunk when zero or at least the depth).
auto plan_graphs(const FfnStack& stack, bool replay, std::size_t graph_layers) -> std::vector<GraphPlan> {
    const auto layers = stack.layers.size();
    const auto hidden = stack.hidden;
    std::vector<GraphPlan> plans;
    if (replay) {
        const auto chunk_layers = graph_layers == 0 ? layers : std::min(graph_layers, layers);
        for (std::size_t first = 0, chunk = 0; first < layers; first += chunk_layers, ++chunk) {
            const auto count = std::min(chunk_layers, layers - first);
            const Io input{"x", chunk == 0 ? Slot::INPUT : boundary_slot(chunk - 1), hidden,
                           stack.layers[first].input_scale};
            const Io output{"y", boundary_slot(chunk), hidden, stack.layers[first + count - 1].output_scale};
            plans.push_back({"step" + std::to_string(chunk), first, -1, {input}, {output}, count});
        }
        return plans;
    }
    for (std::size_t index = 0; index < layers; ++index) {
        const auto& layer = stack.layers[index];
        const auto prefix = "l" + std::to_string(index) + "_";
        const Io x{"x", index == 0 ? Slot::INPUT : output_slot(index - 1), hidden, layer.input_scale};
        const Io gate{"gate", Slot::GATE, layer.intermediate, layer.gate_up_scale};
        const Io up{"up", Slot::UP, layer.intermediate, layer.gate_up_scale};
        const Io activation{"hidden", Slot::HIDDEN, layer.intermediate, layer.hidden_scale};
        plans.push_back({prefix + "gate_up", index, 0, {x}, {gate, up}});
        plans.push_back({prefix + "activation", index, 1, {gate, up}, {activation}});
        plans.push_back(
            {prefix + "down", index, 2, {activation}, {{"y", output_slot(index), hidden, layer.output_scale}}});
    }
    return plans;
}

// Registers graph tensors and adds nodes. Names and dimension arrays live until the graph is finalized.
class GraphBuilder {
public:
    GraphBuilder(qnn::Session& session, const FfnStack& stack, bool packed)
        : session_(session), stack_(stack), packed_(packed) {}

    auto tensor(Qnn_GraphHandle_t graph, const std::string& name, Qnn_TensorType_t type, std::uint32_t width,
                float scale) -> Qnn_Tensor_t {
        auto& dims = dims_.emplace_back(std::vector<std::uint32_t>{stack_.m, width});
        auto result = qnn::make_tensor(names_.emplace_back(name).c_str(), type, QNN_DATATYPE_UFIXED_POINT_8, dims);
        result.v1.quantizeParams = qnn::per_tensor(scale, -128);
        qnn::check(session_.api.tensorCreateGraphTensor(graph, &result), "register " + name);
        return result;
    }

    auto gate_up(Qnn_GraphHandle_t graph, std::size_t index, const Qnn_Tensor_t& x, const Qnn_Tensor_t& gate,
                 const Qnn_Tensor_t& up) -> void {
        const auto& layer = stack_.layers[index];
        const auto values = ffn_weights(stack_, index, 0);
        const auto count = static_cast<std::size_t>(layer.intermediate) * stack_.hidden;
        const std::span<const float> scales(layer.gate_up_weight_scale);
        const auto prefix = "l" + std::to_string(index) + "_";
        projection(graph, prefix + "gate", x, {values.data(), count}, scales.first(layer.intermediate), layer, gate);
        projection(graph, prefix + "up", x, {values.data() + count, count}, scales.subspan(layer.intermediate), layer,
                   up);
    }

    auto activation(Qnn_GraphHandle_t graph, std::size_t index, const Qnn_Tensor_t& gate, const Qnn_Tensor_t& up,
                    const Qnn_Tensor_t& output) -> void {
        const auto& layer = stack_.layers[index];
        const auto prefix = "l" + std::to_string(index) + "_";
        auto activated =
            tensor(graph, prefix + "gelu", QNN_TENSOR_TYPE_NATIVE, layer.intermediate, layer.gate_up_scale);
        node(graph, prefix + "gelu_op", QNN_OP_GELU, {gate}, activated);
        node(graph, prefix + "multiply_op", QNN_OP_ELEMENT_WISE_MULTIPLY, {activated, up}, output);
    }

    auto down(Qnn_GraphHandle_t graph, std::size_t index, const Qnn_Tensor_t& input, const Qnn_Tensor_t& output)
        -> void {
        const auto& layer = stack_.layers[index];
        const auto values = ffn_weights(stack_, index, 1);
        projection(graph, "l" + std::to_string(index) + "_down", input, values, layer.down_weight_scale, layer, output);
    }

    std::string encoding;

private:
    auto projection(Qnn_GraphHandle_t graph, const std::string& name, const Qnn_Tensor_t& input,
                    std::span<const std::int8_t> values, std::span<const float> scales, const FfnLayer& layer,
                    const Qnn_Tensor_t& output) -> void {
        auto weights = qnn::make_weights(values, scales, 0, layer.bits, packed_);
        encoding = weights.encoding;
        auto& dims = dims_.emplace_back(std::vector<std::uint32_t>{
            static_cast<std::uint32_t>(scales.size()), static_cast<std::uint32_t>(values.size() / scales.size())});
        auto weight = qnn::make_tensor(names_.emplace_back(name + "_weight").c_str(), QNN_TENSOR_TYPE_STATIC,
                                       weights.data_type, dims);
        weight.v1.quantizeParams = weights.params;
        weight.v1.clientBuf = {weights.data.data(), static_cast<std::uint32_t>(weights.data.size())};
        qnn::check(session_.api.tensorCreateGraphTensor(graph, &weight), "register " + name + " weights");
        node(graph, name + "_op", QNN_OP_FULLY_CONNECTED, {input, weight}, output);
    }

    auto node(Qnn_GraphHandle_t graph, const std::string& name, const char* type, std::vector<Qnn_Tensor_t> inputs,
              Qnn_Tensor_t output) -> void {
        Qnn_OpConfig_t op = QNN_OPCONFIG_INIT;
        op.version = QNN_OPCONFIG_VERSION_1;
        op.v1.name = names_.emplace_back(name).c_str();
        op.v1.packageName = QNN_OP_PACKAGE_NAME_QTI_AISW;
        op.v1.typeName = type;
        op.v1.numOfInputs = static_cast<std::uint32_t>(inputs.size());
        op.v1.inputTensors = inputs.data();
        op.v1.numOfOutputs = 1;
        op.v1.outputTensors = &output;
        qnn::check(session_.api.graphAddNode(graph, op), "add node " + name);
    }

    qnn::Session& session_;
    const FfnStack& stack_;
    bool packed_;
    std::deque<std::string> names_;
    std::deque<std::vector<std::uint32_t> > dims_;
};

// Builds and finalizes one planned graph, returning its boundary tensor IDs (inputs then outputs).
auto build_graph(qnn::Session& session, GraphBuilder& builder, const FfnStack& stack, const GraphPlan& plan)
    -> std::vector<std::uint32_t> {
    const auto graph = session.create_graph(plan.name);
    std::vector<Qnn_Tensor_t> inputs, outputs;
    for (const auto& io : plan.inputs)
        inputs.push_back(builder.tensor(graph, io.name, QNN_TENSOR_TYPE_APP_WRITE, io.width, io.scale));
    const auto boundary_output = [&](const Io& io) {
        return builder.tensor(graph, io.name, QNN_TENSOR_TYPE_APP_READ, io.width, io.scale);
    };
    if (plan.stage == -1) {
        auto x = inputs.front();
        for (std::size_t index = plan.layer; index < plan.layer + plan.layers; ++index) {
            const auto& layer = stack.layers[index];
            const auto prefix = "l" + std::to_string(index) + "_";
            const auto gate = builder.tensor(graph, prefix + "gate_out", QNN_TENSOR_TYPE_NATIVE, layer.intermediate,
                                             layer.gate_up_scale);
            const auto up = builder.tensor(graph, prefix + "up_out", QNN_TENSOR_TYPE_NATIVE, layer.intermediate,
                                           layer.gate_up_scale);
            builder.gate_up(graph, index, x, gate, up);
            const auto activation = builder.tensor(graph, prefix + "hidden_out", QNN_TENSOR_TYPE_NATIVE,
                                                   layer.intermediate, layer.hidden_scale);
            builder.activation(graph, index, gate, up, activation);
            const bool last = index + 1 == plan.layer + plan.layers;
            const auto y = last ? boundary_output(plan.outputs.front())
                                : builder.tensor(graph, prefix + "y_out", QNN_TENSOR_TYPE_NATIVE, stack.hidden,
                                                 layer.output_scale);
            builder.down(graph, index, activation, y);
            if (last) outputs.push_back(y);
            x = y;
        }
    } else {
        for (const auto& io : plan.outputs) outputs.push_back(boundary_output(io));
        if (plan.stage == 0) builder.gate_up(graph, plan.layer, inputs[0], outputs[0], outputs[1]);
        if (plan.stage == 1) builder.activation(graph, plan.layer, inputs[0], inputs[1], outputs[0]);
        if (plan.stage == 2) builder.down(graph, plan.layer, inputs[0], outputs[0]);
    }
    qnn::check(session.api.graphFinalize(graph, nullptr, nullptr), "finalize " + plan.name);
    std::vector<std::uint32_t> ids;
    for (const auto& tensor : inputs) ids.push_back(tensor.v1.id);
    for (const auto& tensor : outputs) ids.push_back(tensor.v1.id);
    return ids;
}

struct Execution {
    Qnn_GraphHandle_t graph{};
    std::vector<Qnn_Tensor_t> inputs, outputs;
};

} // namespace

auto run_ffn_npu(const FfnStack& stack, const Options& options) -> FfnMeasurement {
    if (options.mode != "eager" && options.mode != "replay")
        throw std::runtime_error("NPU FFN modes are eager and replay");
    if (options.qnn_library.empty()) throw std::runtime_error("--qnn-lib is required for the NPU backend");
    if (!options.qnn_prepare && options.qnn_context.empty())
        throw std::runtime_error("NPU runs need --qnn-prepare or --qnn-context");
    if (options.qnn_memory != "shared" && options.qnn_memory != "client")
        throw std::runtime_error("--qnn-memory must be shared or client");
    const bool replay = options.mode == "replay";
    const bool shared = options.qnn_memory == "shared";
    FfnMeasurement measurement;
    measurement.device = "qualcomm-htp";
    const auto graph_layers = replay ? static_cast<std::size_t>(options.qnn_graph_layers) : 0;
    const auto plans = plan_graphs(stack, replay, graph_layers);
    const auto layers = stack.layers.size();

    const auto start = Clock::now();
    // Declared before the session so shared buffers outlive the context that registers them.
    std::optional<qnn::SharedMemory> rpc;
    if (shared) rpc.emplace();
    qnn::Session session(options.qnn_library);
    std::vector<std::vector<std::uint32_t> > ids;
    std::size_t context_bytes = 0;
    double context_load_ms = 0;
    std::string encoding;
    if (options.qnn_prepare) {
        session.create_context();
        GraphBuilder builder(session, stack, options.qnn_weights == "packed");
        for (const auto& plan : plans) ids.push_back(build_graph(session, builder, stack, plan));
        encoding = builder.encoding;
        const auto binary = session.save_context();
        context_bytes = binary.size();
        if (!options.qnn_context.empty()) {
            qnn::write_file(options.qnn_context, binary);
            const auto text = nlohmann::json{{"mode", options.mode}, {"layers", layers}, {"graph_layers", graph_layers},
                                             {"m", stack.m},         {"ids", ids},       {"weights", encoding}}
                                  .dump();
            qnn::write_file(options.qnn_context + ".json",
                            {reinterpret_cast<const std::uint8_t*>(text.data()), text.size()});
        }
    } else {
        const auto metadata = nlohmann::json::parse(qnn::read_file(options.qnn_context + ".json"));
        if (metadata["mode"] != options.mode || metadata["layers"] != layers || metadata["m"] != stack.m ||
            metadata.value("graph_layers", std::size_t{0}) != graph_layers)
            throw std::runtime_error("context binary was prepared for a different FFN stack or mode");
        ids = metadata["ids"].get<std::vector<std::vector<std::uint32_t> > >();
        encoding = metadata["weights"].get<std::string>();
        const auto binary = qnn::read_file(options.qnn_context);
        context_bytes = binary.size();
        const auto load_start = Clock::now();
        session.load_context(binary);
        context_load_ms = elapsed_ms(load_start);
    }

    // Preallocated boundary buffers, bound once. With shared memory the host and HTP use the same pages.
    struct Storage {
        std::uint8_t* data{};
        int fd = -1;
        std::vector<std::uint8_t> owned;
        std::map<std::uint32_t, Qnn_MemHandle_t> handles; // registered per width
    };
    std::map<std::pair<Slot, std::uint32_t>, Storage> storage;
    std::deque<std::vector<std::uint32_t> > dims;
    const auto buffer = [&](const Io& io) -> Storage& {
        // Gate/up/hidden buffers are separate per width; input and output ping-pong buffers are always hidden-wide.
        const auto key = std::make_pair(
            io.slot, io.slot == Slot::GATE || io.slot == Slot::UP || io.slot == Slot::HIDDEN ? io.width : stack.hidden);
        auto [found, inserted] = storage.try_emplace(key);
        auto& entry = found->second;
        if (inserted) {
            const auto bytes = static_cast<std::size_t>(stack.m) * io.width;
            if (shared) {
                const auto block = rpc->allocate(bytes);
                entry.data = static_cast<std::uint8_t*>(block.data);
                entry.fd = block.fd;
            } else {
                entry.owned.assign(bytes, 0);
                entry.data = entry.owned.data();
            }
        }
        return entry;
    };
    std::vector<Execution> executions;
    for (std::size_t index = 0; index < plans.size(); ++index) {
        const auto& plan = plans[index];
        Execution execution;
        execution.graph = session.retrieve_graph(plan.name);
        std::size_t position = 0;
        const auto bind = [&](const Io& io, Qnn_TensorType_t type) {
            auto& shape = dims.emplace_back(std::vector<std::uint32_t>{stack.m, io.width});
            auto tensor = qnn::make_tensor(io.name.c_str(), type, QNN_DATATYPE_UFIXED_POINT_8, shape);
            tensor.v1.quantizeParams = qnn::per_tensor(io.scale, -128);
            tensor.v1.id = ids.at(index).at(position++);
            auto& entry = buffer(io);
            if (shared) {
                auto [handle, inserted] = entry.handles.try_emplace(io.width);
                if (inserted) handle->second = session.register_memory(entry.fd, shape, QNN_DATATYPE_UFIXED_POINT_8);
                tensor.v1.memType = QNN_TENSORMEMTYPE_MEMHANDLE;
                tensor.v1.memHandle = handle->second;
            } else {
                tensor.v1.clientBuf = {entry.data, stack.m * io.width};
            }
            return tensor;
        };
        for (const auto& io : plan.inputs) execution.inputs.push_back(bind(io, QNN_TENSOR_TYPE_APP_WRITE));
        for (const auto& io : plan.outputs) execution.outputs.push_back(bind(io, QNN_TENSOR_TYPE_APP_READ));
        executions.push_back(std::move(execution));
    }
    measurement.prepare_ms = elapsed_ms(start);
    measurement.launches_per_step = executions.size();
    auto* input = storage.at({Slot::INPUT, stack.hidden}).data;
    const auto* output = storage.at({plans.back().outputs.front().slot, stack.hidden}).data;
    const auto values = static_cast<std::size_t>(stack.m) * stack.hidden;

    const auto step = [&](const std::vector<std::int8_t>& activations) {
        for (std::size_t index = 0; index < values; ++index)
            input[index] = static_cast<std::uint8_t>(activations[index]) ^ 0x80U;
        const auto begin = Clock::now();
        for (auto& execution : executions)
            qnn::check(
                session.api.graphExecute(execution.graph, execution.inputs.data(),
                                         static_cast<std::uint32_t>(execution.inputs.size()), execution.outputs.data(),
                                         static_cast<std::uint32_t>(execution.outputs.size()), nullptr, nullptr),
                "execute graph");
        return elapsed_ms(begin);
    };
    for (int iteration = 0; iteration < options.warmups + options.runs; ++iteration) {
        const auto elapsed = step(stack.inputs[iteration % 2]);
        if (iteration >= options.warmups) measurement.step_ms.push_back(elapsed);
    }
    for (int index = 0; index < 2; ++index) {
        step(stack.inputs[index]);
        measurement.outputs[index].resize(values);
        for (std::size_t element = 0; element < values; ++element)
            measurement.outputs[index][element] = static_cast<std::int8_t>(output[element] ^ 0x80U);
    }
    // Per-graph local error: each stage is compared with the reference applied to that stage's own HTP inputs, which
    // separates operator error from error propagated by earlier stages.
    auto diagnostics = nlohmann::json::array();
    if (!replay && std::getenv("KIDI_FFN_DIAGNOSE")) {
        const auto read = [&](const Io& io) {
            const auto& entry = buffer(io);
            std::vector<std::int8_t> result(static_cast<std::size_t>(stack.m) * io.width);
            for (std::size_t element = 0; element < result.size(); ++element)
                result[element] = static_cast<std::int8_t>(entry.data[element] ^ 0x80U);
            return result;
        };
        const auto project = [&](const std::vector<std::int8_t>& x, const std::int8_t* weights, std::uint32_t k,
                                 std::uint32_t n, std::span<const float> factor) {
            std::vector<std::int8_t> result(static_cast<std::size_t>(stack.m) * n);
            for (std::uint32_t row = 0; row < stack.m; ++row)
                for (std::uint32_t column = 0; column < n; ++column) {
                    std::int32_t sum = 0;
                    for (std::uint32_t index = 0; index < k; ++index)
                        sum += x[static_cast<std::size_t>(row) * k + index] *
                               weights[static_cast<std::size_t>(column) * k + index];
                    result[static_cast<std::size_t>(row) * n + column] = requantize(sum, factor[column]);
                }
            return result;
        };
        const auto stats = [](const std::vector<std::int8_t>& actual, const std::vector<std::int8_t>& expected) {
            double squared = 0, signed_sum = 0;
            int max_difference = 0;
            for (std::size_t element = 0; element < actual.size(); ++element) {
                const auto difference = static_cast<int>(actual[element]) - expected[element];
                squared += static_cast<double>(difference) * difference;
                signed_sum += difference;
                max_difference = std::max(max_difference, std::abs(difference));
            }
            const auto count = static_cast<double>(actual.size());
            return nlohmann::json{{"rms_quanta", std::sqrt(squared / count)},
                                  {"mean_signed_quanta", signed_sum / count},
                                  {"max_difference", max_difference}};
        };
        for (int input_index = 0; input_index < 2; ++input_index) {
            for (std::size_t element = 0; element < values; ++element)
                input[element] = static_cast<std::uint8_t>(stack.inputs[input_index][element]) ^ 0x80U;
            // Reference chain from the original input, to compare accumulated error at each boundary.
            auto chain_x = stack.inputs[input_index];
            std::vector<std::int8_t> chain_gate, chain_up, chain_hidden;
            for (std::size_t index = 0; index < executions.size(); ++index) {
                auto& execution = executions[index];
                qnn::check(session.api.graphExecute(
                               execution.graph, execution.inputs.data(),
                               static_cast<std::uint32_t>(execution.inputs.size()), execution.outputs.data(),
                               static_cast<std::uint32_t>(execution.outputs.size()), nullptr, nullptr),
                           "execute graph");
                const auto& plan = plans[index];
                const auto& layer = stack.layers[plan.layer];
                const auto intermediate = layer.intermediate;
                nlohmann::json record = {
                    {"input", input_index ? "B" : "A"}, {"layer", plan.layer}, {"stage", plan.name}};
                if (plan.stage == 0) {
                    const auto x = read(plan.inputs[0]);
                    const auto weights = ffn_weights(stack, plan.layer, 0);
                    const std::span<const float> factors(layer.gate_up_factor);
                    const auto* up_weights = weights.data() + static_cast<std::size_t>(intermediate) * stack.hidden;
                    const auto gate = read(plan.outputs[0]), up = read(plan.outputs[1]);
                    record["gate"] = stats(
                        gate, project(x, weights.data(), stack.hidden, intermediate, factors.first(intermediate)));
                    record["up"] =
                        stats(up, project(x, up_weights, stack.hidden, intermediate, factors.subspan(intermediate)));
                    chain_gate =
                        project(chain_x, weights.data(), stack.hidden, intermediate, factors.first(intermediate));
                    chain_up = project(chain_x, up_weights, stack.hidden, intermediate, factors.subspan(intermediate));
                    record["chain_input"] = stats(x, chain_x);
                    record["chain_gate"] = stats(gate, chain_gate);
                    record["chain_up"] = stats(up, chain_up);
                } else if (plan.stage == 1) {
                    const auto gate = read(plan.inputs[0]), up = read(plan.inputs[1]);
                    std::vector<std::int8_t> expected(gate.size());
                    chain_hidden.resize(gate.size());
                    for (std::size_t element = 0; element < gate.size(); ++element) {
                        expected[element] =
                            gelu_multiply(gate[element], up[element], layer.gate_up_scale, 1.F / layer.hidden_scale);
                        chain_hidden[element] = gelu_multiply(chain_gate[element], chain_up[element],
                                                              layer.gate_up_scale, 1.F / layer.hidden_scale);
                    }
                    const auto hidden = read(plan.outputs[0]);
                    record["hidden"] = stats(hidden, expected);
                    record["chain_hidden"] = stats(hidden, chain_hidden);
                } else {
                    const auto weights = ffn_weights(stack, plan.layer, 1);
                    const auto y = read(plan.outputs[0]);
                    record["output"] = stats(y, project(read(plan.inputs[0]), weights.data(), intermediate,
                                                        stack.hidden, layer.down_factor));
                    chain_x = project(chain_hidden, weights.data(), intermediate, stack.hidden, layer.down_factor);
                    record["chain_output"] = stats(y, chain_x);
                }
                diagnostics.push_back(record);
            }
        }
    }
    measurement.details = {{"mode", options.qnn_prepare ? "prepare" : "context-binary"},
                           {"memory", options.qnn_memory},
                           {"graph_layers", graph_layers},
                           {"weights", encoding},
                           {"graphs", executions.size()},
                           {"context_bytes", context_bytes},
                           {"context_load_ms", context_load_ms},
                           {"qnn_api", session.api_version},
                           {"build_id", session.build_id},
                           {"burst_power", session.burst},
                           {"prepare_library_loaded", qnn::library_loaded("libQnnHtpPrepare.so")}};
    if (!diagnostics.empty()) measurement.details["diagnostics"] = diagnostics;
    return measurement;
}

#endif

} // namespace kidi::bench
