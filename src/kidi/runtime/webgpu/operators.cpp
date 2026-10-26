#include "kidi/runtime/operator.h"
#include "kidi/ops/context.h"
#include "kidi/tensor/web_gpu.h"
#include "kidi/graph/graph.h"
#include "kidi/graph/compiler.h"

#include <emscripten.h>
#include <nlohmann/json.hpp>

#include <array>
#include <bit>
#include <cstring>
#include <tuple>

namespace kidi::runtime {
namespace {
using ops::require;
// clang-format off
EM_JS(int, prepare_gpu, (const char* specification), {
    try {
        return Module.kidiGpu.prepare(JSON.parse(UTF8ToString(Number(specification))));
    } catch (error) {
        Module.kidiGpu.failure = Module.kidiGpu.failure || String(error);
        return 0;
    }
});
EM_JS(int, output_dimension, (int program, int axis), {
    const plan = Module.kidiGpu.programs.get(program);
    return axis === -1 ? plan.shape.length : axis === -2 ? plan.dtype : plan.shape[axis];
});
EM_JS(int, run_gpu, (int program, const std::uint32_t* bindings, int count), {
    try {
        Module.kidiGpu.run(program, HEAPU32.subarray(Number(bindings) / 4, Number(bindings) / 4 + count));
        return 0;
    } catch (error) {
        Module.kidiGpu.failure = String(error);
        return -1;
    }
});
EM_JS(void, release_program, (int program), { Module.kidiGpu.releaseProgram(program); });
EM_JS(int, prepare_region, (const char* specification), {
    try {
        return Module.kidiGpu.prepareRegion(JSON.parse(UTF8ToString(Number(specification))));
    } catch (error) {
        Module.kidiGpu.failure = String(error);
        return 0;
    }
});
EM_JS(int, run_region, (int region, const std::uint32_t* bindings, int count), {
    try {
        Module.kidiGpu.runRegion(region, HEAPU32.subarray(Number(bindings) / 4, Number(bindings) / 4 + count));
        return 0;
    } catch (error) {
        Module.kidiGpu.failure = String(error);
        return -1;
    }
});
EM_JS(void, release_region, (int region), { Module.kidiGpu.releaseRegion(region); });
EM_JS(int, lower_regions, (), { return Module.kidiGpu.lowerGraphs ? 1 : 0; });
EM_JS(int, copy_gpu,
      (int source, std::size_t source_offset, int destination, std::size_t destination_offset, std::size_t bytes), {
          try {
              Module.kidiGpu.copy(source, Number(source_offset), destination, Number(destination_offset),
                                  Number(bytes));
              return 0;
          } catch (error) {
              Module.kidiGpu.failure = String(error);
              return -1;
          }
      });
EM_JS(int, read_later, (int source, std::size_t offset, void* destination, std::size_t bytes), {
    try {
        Module.kidiGpu.readLater(source, Number(offset), Number(destination), Number(bytes));
        return 0;
    } catch (error) {
        Module.kidiGpu.failure = String(error);
        return -1;
    }
});
// clang-format on

// Interleaving permits four signed-byte extractions with one shift and mask: within each 32-bit word, value `slot`
// moves to byte `slot % 4` at bit `bits * (slot / 4)`. The mapping is linear in the input bytes, so each word is the OR
// of four byte lookups.
auto interleave_packed(const tensor::Tensor& weight, std::int32_t bits) -> tensor::Tensor {
    std::vector<std::byte> copied;
    std::span<const std::byte> source;
    if (weight.is_host_accessible()) {
        source = require(weight.host_bytes());
    } else {
        copied = require(weight.copy_to_host());
        source = copied;
    }
    if (source.size() % 4 != 0)
        throw ops::Failure({ErrorCode::UNSUPPORTED, "packed weights must occupy whole 32-bit words"});
    std::array<std::uint32_t, 4 * 256> table{};
    const auto slots = static_cast<std::uint32_t>(32 / bits);
    const auto mask = (1u << bits) - 1u;
    for (std::uint32_t position = 0; position < 4; ++position)
        for (std::uint32_t value = 0; value < 256; ++value) {
            const auto input = value << (8 * position);
            std::uint32_t output = 0;
            for (std::uint32_t slot = 0; slot < slots; ++slot)
                output |= ((input >> (slot * static_cast<std::uint32_t>(bits))) & mask)
                          << (8 * (slot % 4) + static_cast<std::uint32_t>(bits) * (slot / 4));
            table[position * 256 + value] = output;
        }
    auto host = require(tensor::Tensor::empty({weight.shape().begin(), weight.shape().end()}, weight.dtype()));
    auto destination = require(host.host_bytes());
    for (std::size_t word = 0; word < source.size() / 4; ++word) {
        std::uint32_t input = 0;
        std::memcpy(&input, source.data() + word * 4, 4);
        const auto output = table[input & 255] | table[256 + ((input >> 8) & 255)] |
                            table[512 + ((input >> 16) & 255)] | table[768 + (input >> 24)];
        std::memcpy(destination.data() + word * 4, &output, 4);
    }
    return require(host.to(tensor::Device::web_gpu()));
}

class GpuOperator final : public Operator {
public:
    GpuOperator(int program, std::vector<tensor::Tensor> constants, bool selected, bool direct_inplace)
        : program_(program), constants_(std::move(constants)), selected_(selected), direct_inplace_(direct_inplace) {
        dtype_ = static_cast<tensor::DType>(output_dimension(program_, -2));
        for (int axis = 0; axis < output_dimension(program_, -1); ++axis)
            shape_.push_back(output_dimension(program_, axis));
    }
    ~GpuOperator() override { release_program(program_); }
    auto run(TensorInputs inputs) -> tensor::Tensor override {
        auto output = selected_ ? host_pool_.acquire(shape_, dtype_, tensor::Device::cpu())
                                : pool_.acquire(shape_, dtype_, tensor::Device::web_gpu());
        run_into(inputs, {&output, 1});
        return output;
    }
    auto run_into(TensorInputs inputs, std::span<tensor::Tensor> outputs) -> void override {
        auto output = selected_ ? pool_.acquire(shape_, dtype_, tensor::Device::web_gpu()) : outputs.front();
        std::array<std::uint32_t, 32> bindings{};
        if (inputs.size() > 8) throw ops::Failure({ErrorCode::UNSUPPORTED, "too many WebGPU operands"});
        for (std::size_t index = 0; index < inputs.size(); ++index) {
            const auto view =
                require(tensor::web_gpu_buffer(constants_[index].defined() ? constants_[index] : inputs[index]));
            bindings[index * 2] = view.handle;
            bindings[index * 2 + 1] = view.offset_bytes;
        }
        const auto destination = require(tensor::web_gpu_buffer(output));
        bindings[inputs.size() * 2] = destination.handle;
        bindings[inputs.size() * 2 + 1] = destination.offset_bytes;
        if (run_gpu(program_, bindings.data(), (inputs.size() + 1) * 2)) throw ops::Failure(tensor::web_gpu_error());
        if (!selected_) return;
        auto bytes = require(outputs.front().host_bytes());
        if (read_later(destination.handle, destination.offset_bytes, bytes.data(), bytes.size()))
            throw ops::Failure(tensor::web_gpu_error());
    }
    auto run_(TensorInputs inputs, tensor::Tensor& destination) -> tensor::Tensor override {
        if (direct_inplace_) {
            std::vector<tensor::Tensor> rebound;
            for (std::size_t index = 0; index < inputs.size(); ++index) rebound.push_back(inputs[index]);
            for (std::size_t index = 1; index < rebound.size(); ++index) {
                if (rebound[index].storage_identity() != destination.storage_identity()) continue;
                const auto source = require(tensor::web_gpu_buffer(rebound[index]));
                auto snapshot =
                    require(tensor::Tensor::empty({rebound[index].shape().begin(), rebound[index].shape().end()},
                                                  rebound[index].dtype(), tensor::Device::web_gpu()));
                const auto target = require(tensor::web_gpu_buffer(snapshot));
                if (copy_gpu(source.handle, source.offset_bytes, target.handle, target.offset_bytes,
                             (snapshot.nbytes() + 3) / 4 * 4))
                    throw ops::Failure(tensor::web_gpu_error());
                rebound[index] = std::move(snapshot);
            }
            run_into(rebound, {&destination, 1});
            return destination;
        }
        auto result = run(inputs);
        if (result.dtype() != destination.dtype() || !std::ranges::equal(result.shape(), destination.shape()))
            throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "in-place WebGPU operation changes shape or dtype"});
        const auto source = require(tensor::web_gpu_buffer(result)),
                   target = require(tensor::web_gpu_buffer(destination));
        if (copy_gpu(source.handle, source.offset_bytes, target.handle, target.offset_bytes, destination.nbytes()))
            throw ops::Failure(tensor::web_gpu_error());
        return destination;
    }
    auto allocations() const -> AllocationStats override { return pool_.allocations(); }
    auto release_workspaces() -> void override {
        pool_.clear();
        host_pool_.clear();
    }

    auto record(const graph::Node& node, std::vector<tensor::Tensor>& retained) -> std::optional<nlohmann::json> {
        using Source = graph::Source;
        const auto binding = [&](const tensor::Tensor& value, const Source* source = nullptr) -> nlohmann::json {
            if (source && source->kind == Source::Kind::INPUT)
                return {{"input", source->index}, {"offset", source->offset}};
            const auto view = require(tensor::web_gpu_buffer(value));
            return {{"handle", view.handle}, {"offset", view.offset_bytes}};
        };
        nlohmann::json command{{"program", program_}, {"bindings", nlohmann::json::array()}};
        for (std::size_t index = 0; index < node.input_count; ++index) {
            if (constants_[index].defined())
                command["bindings"].push_back(binding(constants_[index]));
            else
                command["bindings"].push_back(binding(node.operands[index], &node.sources[index]));
        }
        const auto& destination = node.outputs().front();
        if (direct_inplace_ && node.mode == graph::Mode::IN_PLACE) {
            for (std::size_t index = 1; index < node.input_count; ++index)
                if (node.operands[index].storage_identity() == destination.storage_identity()) return std::nullopt;
        }
        if (selected_ || (node.mode == graph::Mode::IN_PLACE && !direct_inplace_)) {
            auto temporary = pool_.acquire(shape_, dtype_, tensor::Device::web_gpu());
            command["bindings"].push_back(binding(temporary));
            retained.push_back(std::move(temporary));
            if (selected_) {
                const auto bytes = require(destination.host_bytes());
                command["read"] = {{"destination", reinterpret_cast<std::uintptr_t>(bytes.data())},
                                   {"bytes", bytes.size()}};
            } else {
                command["copy"] = {{"destination", binding(destination, &node.sources.back())},
                                   {"bytes", destination.nbytes()}};
            }
        } else {
            command["bindings"].push_back(binding(destination, &node.sources.back()));
        }
        return command;
    }

private:
    int program_;
    std::vector<tensor::Tensor> constants_;
    std::vector<std::int64_t> shape_;
    tensor::DType dtype_;
    bool selected_, direct_inplace_;
    OutputPool pool_, host_pool_;
};

class GpuRegion final : public graph::Executable {
public:
    GpuRegion(int program, graph::Graph graph, std::vector<tensor::Tensor> retained)
        : program_(program), graph_(std::move(graph)), retained_(std::move(retained)) {}
    ~GpuRegion() override { release_region(program_); }
    auto run(std::span<const tensor::Tensor> inputs, std::span<tensor::Tensor> outputs) -> void override {
        graph_.rebind(inputs);
        std::vector<std::uint32_t> bindings;
        bindings.reserve(inputs.size() * 2);
        for (const auto& input : inputs) {
            const auto view = require(tensor::web_gpu_buffer(input));
            bindings.push_back(view.handle);
            bindings.push_back(view.offset_bytes);
        }
        if (run_region(program_, bindings.data(), bindings.size())) throw ops::Failure(tensor::web_gpu_error());
        std::ranges::copy(graph_.outputs(), outputs.begin());
    }

private:
    int program_;
    graph::Graph graph_;
    std::vector<tensor::Tensor> retained_;
};

class GpuCompiler final : public graph::Compiler {
public:
    auto name() const -> std::string_view override { return "webgpu"; }
    auto compile(const graph::Graph& graph, std::string_view key) -> std::unique_ptr<graph::Executable> override {
        if (!lower_regions()) return nullptr;
        for (const auto& input : graph.inputs())
            if (input.device() != tensor::Device::web_gpu()) return nullptr;
        auto planned = graph;
        const auto memory = planned.plan_intermediates(
            [&](std::size_t slot, std::size_t elements, tensor::DType dtype, tensor::Device device) {
                auto& storage = scratch_[{static_cast<int>(dtype), slot}];
                if (!storage.defined() || storage.numel() < elements) {
                    const auto capacity = std::bit_ceil(std::max<std::size_t>(elements, 1));
                    storage = require(tensor::Tensor::empty({static_cast<std::int64_t>(capacity)}, dtype, device));
                }
                return storage;
            });
        nlohmann::json description{{"key", key},
                                   {"commands", nlohmann::json::array()},
                                   {"originalBytes", memory.original_bytes},
                                   {"plannedBytes", memory.planned_bytes}};
        std::vector<tensor::Tensor> retained;
        for (const auto& node : planned.nodes()) {
            if (node.outputs().size() != 1) return nullptr;
            auto* operation = dynamic_cast<GpuOperator*>(node.prepared.get());
            if (!operation) return nullptr;
            auto command = operation->record(node, retained);
            if (!command) return nullptr;
            description["commands"].push_back(std::move(*command));
        }
        const auto region = prepare_region(description.dump().c_str());
        if (!region) return nullptr;
        for (const auto& node : planned.nodes()) static_cast<GpuOperator*>(node.prepared.get())->release_workspaces();
        return std::make_unique<GpuRegion>(region, std::move(planned), std::move(retained));
    }

private:
    std::map<std::pair<int, std::size_t>, tensor::Tensor> scratch_;
};

class GpuBackend final : public OperatorBackend {
public:
    auto supports_replay() const noexcept -> bool override { return true; }
    auto graph_compiler() -> std::shared_ptr<graph::Compiler> override { return compiler_; }
    auto prepare(const OperatorSpec& spec, TensorInputs inputs) -> std::unique_ptr<Operator> override {
        static const std::map<Operation, std::string> names{{Operation::ADD, "add"},
                                                            {Operation::MULTIPLY, "multiply"},
                                                            {Operation::CAST, "cast"},
                                                            {Operation::MATMUL, "matmul"},
                                                            {Operation::LINEAR, "linear"},
                                                            {Operation::QUANTIZED_LINEAR, "quantized_linear"},
                                                            {Operation::GELU, "gelu"},
                                                            {Operation::LAYER_NORM, "layer_norm"},
                                                            {Operation::SOFTMAX, "softmax"},
                                                            {Operation::LOG_SOFTMAX, "log_softmax"},
                                                            {Operation::TRANSPOSE, "transpose"},
                                                            {Operation::SLICE, "slice"},
                                                            {Operation::GATHER, "gather"},
                                                            {Operation::CONCAT, "concat"},
                                                            {Operation::SCATTER, "scatter"},
                                                            {Operation::ATTENTION, "attention"},
                                                            {Operation::RESIDUAL_NORM, "residual_norm"},
                                                            {Operation::RMS_NORM, "rms_norm"},
                                                            {Operation::TANH, "tanh"},
                                                            {Operation::ROTARY, "rotary"},
                                                            {Operation::PACKED_LINEAR, "packed_linear"},
                                                            {Operation::RMS_NORM_RESIDUAL, "rms_norm_residual"},
                                                            {Operation::STATIC_ROUND, "static_round"},
                                                            {Operation::GREEDY_TOKEN, "greedy_token"},
                                                            {Operation::RMS_ROTARY, "rms_rotary"},
                                                            {Operation::GELU_MULTIPLY, "gelu_multiply"},
                                                            {Operation::SILU_MULTIPLY, "silu_multiply"},
                                                            {Operation::SILU_GATE, "silu_gate"},
                                                            {Operation::EMBEDDING, "embedding"},
                                                            {Operation::GATED_FEED_FORWARD, "gated_feed_forward"},
                                                            {Operation::DIFFUSION_CANDIDATES, "diffusion_candidates"},
                                                            {Operation::DIFFUSION_UPDATE, "diffusion_update"},
                                                            {Operation::EMBEDDING_SUM, "embedding_sum"},
                                                            {Operation::CONV1D, "conv1d"},
                                                            {Operation::SNAKE_CONV1D, "snake_conv1d"},
                                                            {Operation::SNAKE_QUANTIZED_LINEAR,
                                                             "snake_quantized_linear"},
                                                            {Operation::CONV_TRANSPOSE_FOLD, "conv_transpose_fold"},
                                                            {Operation::SNAKE, "snake"}};
        const bool direct_inplace =
            ops::is_inplace && (spec.operation == Operation::SCATTER || spec.operation == Operation::ADD ||
                                spec.operation == Operation::MULTIPLY);
        nlohmann::json description{
            {"operation", names.at(spec.operation)},
            {"inplace", direct_inplace},
            {"dtype", static_cast<int>(spec.dtype)},
            {"epsilon", spec.epsilon},
            {"attributes", std::vector<std::int64_t>(spec.attributes.begin(), spec.attributes.end())}};
        description["quantization"] = nlohmann::json::array();
        for (const auto& quantization : spec.quantization)
            description["quantization"].push_back(
                {{"scales", std::vector<float>(quantization.scales.begin(), quantization.scales.end())},
                 {"zero_points",
                  std::vector<std::int32_t>(quantization.zero_points.begin(), quantization.zero_points.end())},
                 {"block_size", quantization.block_size}});
        description["inputs"] = nlohmann::json::array();
        std::vector<tensor::Tensor> constants(inputs.size());
        for (std::size_t index = 0; index < inputs.size(); ++index) {
            const auto& input = inputs[index];
            description["inputs"].push_back(
                {{"shape", std::vector<std::int64_t>(input.shape().begin(), input.shape().end())},
                 {"dtype", static_cast<int>(input.dtype())}});
            const auto packed_bits =
                ((spec.operation == Operation::PACKED_LINEAR && index == 1) ||
                 (spec.operation == Operation::GATED_FEED_FORWARD && (index == 1 || index == 3))) &&
                        !spec.attributes.empty()
                    ? static_cast<std::int32_t>(spec.attributes[0])
                    : 0;
            const auto interleaved = packed_bits == 2 || packed_bits == 4 ? packed_bits : 0;
            // Weights the loader uploaded in kernel layout are used in place; others get an interleaved device copy.
            const bool kernel_layout = interleaved && input.device() == tensor::Device::web_gpu() &&
                                       require(tensor::web_gpu_buffer(input)).packed_layout == interleaved;
            if (!kernel_layout && (interleaved || input.device() != tensor::Device::web_gpu())) {
                const auto key = std::tuple{input.storage_identity(), input.storage_offset(), interleaved};
                auto found = constants_.find(key);
                if (found == constants_.end())
                    found =
                        constants_
                            .emplace(key, std::pair{input, interleaved ? interleave_packed(input, interleaved)
                                                                       : require(input.to(tensor::Device::web_gpu()))})
                            .first;
                constants[index] = found->second.second;
            }
        }
        const auto program = prepare_gpu(description.dump().c_str());
        if (!program) throw ops::Failure(tensor::web_gpu_error());
        return std::make_unique<GpuOperator>(program, std::move(constants), spec.operation == Operation::GREEDY_TOKEN,
                                             direct_inplace);
    }
    auto synchronize() -> void override { require(tensor::web_gpu_synchronize()); }
    auto copy_slice_(tensor::Tensor& destination, const tensor::Tensor& source, std::size_t outer,
                     std::size_t source_bytes, std::size_t destination_bytes, std::size_t offset_bytes)
        -> void override {
        auto input = require(tensor::web_gpu_buffer(source)), output = require(tensor::web_gpu_buffer(destination));
        tensor::Tensor snapshot;
        if (input.handle == output.handle) {
            snapshot = require(tensor::Tensor::empty({static_cast<std::int64_t>(source.nbytes())}, tensor::DType::U8,
                                                     tensor::Device::web_gpu()));
            auto temporary = require(tensor::web_gpu_buffer(snapshot));
            if (copy_gpu(input.handle, input.offset_bytes, temporary.handle, 0, source.nbytes()))
                throw ops::Failure(tensor::web_gpu_error());
            input = temporary;
        }
        for (std::size_t row = 0; row < outer; ++row)
            if (copy_gpu(input.handle, input.offset_bytes + row * source_bytes, output.handle,
                         output.offset_bytes + row * destination_bytes + offset_bytes, source_bytes))
                throw ops::Failure(tensor::web_gpu_error());
    }

private:
    std::shared_ptr<GpuCompiler> compiler_ = std::make_shared<GpuCompiler>();
    std::map<std::tuple<const void*, std::int64_t, std::int32_t>, std::pair<tensor::Tensor, tensor::Tensor>> constants_;
};
} // namespace
auto web_gpu_operators() -> std::unique_ptr<OperatorBackend> { return std::make_unique<GpuBackend>(); }
} // namespace kidi::runtime