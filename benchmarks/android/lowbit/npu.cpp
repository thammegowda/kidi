#include "lowbit.h"

#include <stdexcept>

#if defined(KIDI_BENCH_QNN)
#include "qnn.h"

#include <QnnOpDef.h>

#include <chrono>
#include <string>
#include <vector>
#endif

namespace kidi::bench {

#if !defined(KIDI_BENCH_QNN)

auto run_npu(const Problem&, const Options&) -> Measurement {
    throw std::runtime_error("built without KIDI_QNN_SDK; the Qualcomm NPU backend is unavailable");
}

#else

namespace {

using Clock = std::chrono::steady_clock;
constexpr const char* GRAPH_NAME = "kidi_lowbit";

auto elapsed_ms(Clock::time_point start) -> double {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

// FullyConnected uses [N][K] weights; Conv2d 1x1 uses NHWC activations and HWIO ([K][N]) weights.
struct Layout {
    bool conv = false;
    std::vector<std::uint32_t> input_dims;
    std::vector<std::uint32_t> weight_dims;
    std::vector<std::uint32_t> output_dims;
    std::int32_t weight_axis = 0;

    Layout(const Problem& problem, bool use_conv) : conv(use_conv) {
        if (conv) {
            input_dims = {1, 1, problem.m, problem.k};
            weight_dims = {1, 1, problem.k, problem.n};
            output_dims = {1, 1, problem.m, problem.n};
            weight_axis = 3;
        } else {
            input_dims = {problem.m, problem.k};
            weight_dims = {problem.n, problem.k};
            output_dims = {problem.m, problem.n};
        }
    }
};

} // namespace

auto run_npu(const Problem& problem, const Options& options) -> Measurement {
    if (options.qnn_library.empty()) throw std::runtime_error("--qnn-lib is required for the NPU backend");
    if (!options.qnn_prepare && options.qnn_context.empty())
        throw std::runtime_error("NPU runs need --qnn-prepare or --qnn-context");
    if (options.qnn_op != "fc" && options.qnn_op != "conv") throw std::runtime_error("--qnn-op must be fc or conv");
    if (options.qnn_weights != "container" && options.qnn_weights != "packed")
        throw std::runtime_error("--qnn-weights must be container or packed");
    Measurement measurement;
    measurement.device = "qualcomm-htp";
    const Layout layout(problem, options.qnn_op == "conv");
    auto input_dims = layout.input_dims;
    auto output_dims = layout.output_dims;
    // Activations are symmetric INT8 carried as UINT8 with offset -128, so real = scale * (u8 - 128).
    auto input = qnn::make_tensor("input", QNN_TENSOR_TYPE_APP_WRITE, QNN_DATATYPE_UFIXED_POINT_8, input_dims);
    input.v1.quantizeParams = qnn::per_tensor(problem.input_scale, -128);
    auto output = qnn::make_tensor("output", QNN_TENSOR_TYPE_APP_READ, QNN_DATATYPE_UFIXED_POINT_8, output_dims);
    output.v1.quantizeParams = qnn::per_tensor(problem.output_scale, -128);

    const auto start = Clock::now();
    qnn::Session session(options.qnn_library);
    std::size_t context_bytes = 0;
    double context_load_ms = 0;
    std::string weight_encoding;
    Qnn_GraphHandle_t graph{};
    if (options.qnn_prepare) {
        session.create_context();
        graph = session.create_graph(GRAPH_NAME);
        std::vector<std::int8_t> ordered(problem.weight.size());
        for (std::uint32_t column = 0; column < problem.n; ++column)
            for (std::uint32_t index = 0; index < problem.k; ++index)
                ordered[layout.conv ? static_cast<std::size_t>(index) * problem.n + column
                                    : static_cast<std::size_t>(column) * problem.k + index] =
                    problem.weight[static_cast<std::size_t>(column) * problem.k + index];
        auto weights = qnn::make_weights(ordered, problem.weight_scale, layout.weight_axis, problem.bits,
                                         options.qnn_weights == "packed");
        weight_encoding = weights.encoding;
        auto weight_dims = layout.weight_dims;
        auto weight = qnn::make_tensor("weight", QNN_TENSOR_TYPE_STATIC, weights.data_type, weight_dims);
        weight.v1.quantizeParams = weights.params;
        weight.v1.clientBuf = {weights.data.data(), static_cast<std::uint32_t>(weights.data.size())};
        qnn::check(session.api.tensorCreateGraphTensor(graph, &input), "register input");
        qnn::check(session.api.tensorCreateGraphTensor(graph, &weight), "register weights");
        qnn::check(session.api.tensorCreateGraphTensor(graph, &output), "register output");

        std::vector<std::uint32_t> pair_dims{2}, pad_dims{2, 2};
        std::vector<std::uint32_t> stride_values{1, 1}, dilation_values{1, 1}, pad_values{0, 0, 0, 0};
        auto stride = qnn::make_tensor("stride", QNN_TENSOR_TYPE_STATIC, QNN_DATATYPE_UINT_32, pair_dims);
        auto dilation = qnn::make_tensor("dilation", QNN_TENSOR_TYPE_STATIC, QNN_DATATYPE_UINT_32, pair_dims);
        auto pad = qnn::make_tensor("pad_amount", QNN_TENSOR_TYPE_STATIC, QNN_DATATYPE_UINT_32, pad_dims);
        std::vector<Qnn_Param_t> params;
        if (layout.conv) {
            const auto bind = [&](Qnn_Tensor_t& tensor, std::vector<std::uint32_t>& values, const char* name) {
                tensor.v1.clientBuf = {values.data(), static_cast<std::uint32_t>(values.size() * 4)};
                qnn::check(session.api.tensorCreateGraphTensor(graph, &tensor), std::string("register ") + name);
                // Registration clears client pointers; parameters must carry their values at node creation.
                tensor.v1.clientBuf = {values.data(), static_cast<std::uint32_t>(values.size() * 4)};
                Qnn_Param_t param = QNN_PARAM_INIT;
                param.paramType = QNN_PARAMTYPE_TENSOR;
                param.name = name;
                param.tensorParam = tensor;
                params.push_back(param);
            };
            bind(stride, stride_values, QNN_OP_CONV_2D_PARAM_STRIDE);
            bind(pad, pad_values, QNN_OP_CONV_2D_PARAM_PAD_AMOUNT);
            bind(dilation, dilation_values, QNN_OP_CONV_2D_PARAM_DILATION);
        }
        std::vector<Qnn_Tensor_t> inputs{input, weight};
        Qnn_OpConfig_t op = QNN_OPCONFIG_INIT;
        op.version = QNN_OPCONFIG_VERSION_1;
        op.v1.name = "projection";
        op.v1.packageName = QNN_OP_PACKAGE_NAME_QTI_AISW;
        op.v1.typeName = layout.conv ? QNN_OP_CONV_2D : QNN_OP_FULLY_CONNECTED;
        op.v1.numOfParams = static_cast<std::uint32_t>(params.size());
        op.v1.params = params.empty() ? nullptr : params.data();
        op.v1.numOfInputs = static_cast<std::uint32_t>(inputs.size());
        op.v1.inputTensors = inputs.data();
        op.v1.numOfOutputs = 1;
        op.v1.outputTensors = &output;
        qnn::check(session.api.graphAddNode(graph, op), "add projection node");
        qnn::check(session.api.graphFinalize(graph, nullptr, nullptr), "finalize graph");
        const auto binary = session.save_context();
        context_bytes = binary.size();
        if (!options.qnn_context.empty()) {
            qnn::write_file(options.qnn_context, binary);
            const nlohmann::json metadata = {{"input_id", input.v1.id},
                                             {"output_id", output.v1.id},
                                             {"bits", problem.bits},
                                             {"m", problem.m},
                                             {"k", problem.k},
                                             {"n", problem.n},
                                             {"op", options.qnn_op},
                                             {"weights", weight_encoding},
                                             {"build_id", session.build_id}};
            const auto text = metadata.dump();
            qnn::write_file(options.qnn_context + ".json",
                            {reinterpret_cast<const std::uint8_t*>(text.data()), text.size()});
        }
    } else {
        const auto metadata = nlohmann::json::parse(qnn::read_file(options.qnn_context + ".json"));
        if (metadata["bits"] != problem.bits || metadata["m"] != problem.m || metadata["k"] != problem.k ||
            metadata["n"] != problem.n || metadata["op"] != options.qnn_op)
            throw std::runtime_error("context binary was prepared for a different problem");
        input.v1.id = metadata["input_id"].get<std::uint32_t>();
        output.v1.id = metadata["output_id"].get<std::uint32_t>();
        weight_encoding = metadata["weights"].get<std::string>();
        const auto binary = qnn::read_file(options.qnn_context);
        context_bytes = binary.size();
        const auto load_start = Clock::now();
        session.load_context(binary);
        graph = session.retrieve_graph(GRAPH_NAME);
        context_load_ms = elapsed_ms(load_start);
    }
    measurement.prepare_ms = elapsed_ms(start);

    std::vector<std::uint8_t> input_data(problem.input.size());
    for (std::size_t index = 0; index < input_data.size(); ++index)
        input_data[index] = static_cast<std::uint8_t>(problem.input[index]) ^ 0x80U;
    std::vector<std::uint8_t> output_data(static_cast<std::size_t>(problem.m) * problem.n);
    input.v1.clientBuf = {input_data.data(), static_cast<std::uint32_t>(input_data.size())};
    output.v1.clientBuf = {output_data.data(), static_cast<std::uint32_t>(output_data.size())};
    for (int iteration = 0; iteration < options.warmups + options.runs; ++iteration) {
        const auto begin = Clock::now();
        qnn::check(session.api.graphExecute(graph, &input, 1, &output, 1, nullptr, nullptr), "execute graph");
        const auto wall = elapsed_ms(begin);
        if (iteration >= options.warmups) measurement.wall_ms.push_back(wall);
    }
    measurement.output.resize(output_data.size());
    for (std::size_t index = 0; index < output_data.size(); ++index)
        measurement.output[index] = static_cast<std::int8_t>(output_data[index] ^ 0x80U);
    measurement.details = {{"mode", options.qnn_prepare ? "prepare" : "context-binary"},
                           {"op", options.qnn_op},
                           {"weights", weight_encoding},
                           {"context_bytes", context_bytes},
                           {"context_load_ms", context_load_ms},
                           {"qnn_api", session.api_version},
                           {"build_id", session.build_id},
                           {"burst_power", session.burst},
                           {"library", options.qnn_library},
                           {"prepare_library_loaded", qnn::library_loaded("libQnnHtpPrepare.so")}};
    return measurement;
}

#endif

} // namespace kidi::bench
