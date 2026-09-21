#include "kidi/runtime/operator.h"
#include "kidi/runtime/mps/graph.h"
#include "kidi/runtime/mps/quantized_linear.h"
#include "kidi/runtime/mps/eager_kernels.h"
#include "kidi/ops/context.h"
#include "kidi/tensor/metal.h"
#include <array>
#include <bit>
#include <map>
#include <tuple>
#include <cmath>
#include <optional>
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <unordered_set>
#include <limits>

namespace kidi::runtime {
namespace {
using ops::require;
using tensor::DType;
using tensor::Tensor;
struct Stream {
    std::optional<mps::CommandBatch> batch;
    std::vector<Tensor> pending;
    OutputPool outputs;
    AllocationStats output_allocations;
    std::unordered_set<void*> output_buffers;
    std::size_t prefill_cache_limit = std::numeric_limits<std::size_t>::max();
    bool profile_memory = std::getenv("KIDI_PROFILE_MEMORY") != nullptr;
    std::size_t observed_device_bytes = 0, recommended_working_set_bytes = 0;
    std::size_t peak_pending = 0, expanded_bytes = 0;
    Stream() {
        pending.reserve(512);
        if (const auto value = std::getenv("KIDI_PREFILL_CACHE_BYTES")) {
            char* end = nullptr;
            const auto bytes = std::strtoull(value, &end, 10);
            if (end == value || *end || bytes > (std::uint64_t{16} << 30))
                throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "prefill cache bytes must be 0..16 GiB"});
            prefill_cache_limit = bytes;
        }
    }
    ~Stream() {
        if (profile_memory)
            std::cerr << "kidi_metal_memory|prefill_cache_limit=" << prefill_cache_limit
                      << "|output_allocations=" << output_allocations.count
                      << "|output_allocated_bytes=" << output_allocations.bytes
                      << "|expanded_weight_bytes=" << expanded_bytes << "|peak_external_owner_slots=" << peak_pending
                      << "|observed_peak_device_bytes=" << observed_device_bytes
                      << "|recommended_working_set_bytes=" << recommended_working_set_bytes << '\n';
    }
    auto acquire(std::span<const std::int64_t> shape, DType dtype) -> Tensor {
        const auto before = outputs.allocations();
        auto output = outputs.acquire(shape, dtype, tensor::Device::apple_gpu());
        output_allocations.count += outputs.allocations().count - before.count;
        output_allocations.bytes += outputs.allocations().bytes - before.bytes;
        output_buffers.insert(require(tensor::metal_buffer(output)).handle);
        return output;
    }
    auto retain(const Tensor& tensor) -> void {
        if (output_buffers.contains(require(tensor::metal_buffer(tensor)).handle)) return;
        pending.push_back(tensor);
        peak_pending = std::max(peak_pending, pending.size());
    }
    auto commands() -> mps::CommandBatch& {
        if (!batch)
            batch = require(mps::CommandBatch::create());
        else
            require(batch->restart());
        return *batch;
    }
    auto synchronize() -> void {
        if (batch) require(batch->finish());
        if (profile_memory) {
            const auto memory = tensor::metal_memory_stats();
            observed_device_bytes = std::max(observed_device_bytes, memory.allocated_bytes);
            recommended_working_set_bytes = memory.recommended_working_set_bytes;
        }
        pending.clear();
    }
};
class MetalOperator final : public Operator {
public:
    Stream& stream;
    std::optional<mps::Executable> executable;
    std::optional<mps::QuantizedLinear> quantized;
    Tensor packed_weight, packed_scales;
    Tensor prefill_weight;
    std::shared_ptr<const mps::Executable> prefill_executable;
    std::int32_t packed_bits = 0, packed_group = 0;
    float packed_input_scale = 0, packed_output_scale = 0;
    std::vector<std::int64_t> shape;
    DType dtype;
    std::size_t dynamic_count;
    bool scatter = false;
    bool greedy = false;
    bool vector_projection = false;
    bool eager = false;
    Operation operation;
    float epsilon = 0;
    explicit MetalOperator(Stream& owner) : stream(owner) {}
    auto run(TensorInputs inputs) -> Tensor override {
        auto output = stream.acquire(shape, dtype);
        if (greedy) {
            const std::array temporary_shape{static_cast<std::int64_t>(output.numel()),
                                             static_cast<std::int64_t>((inputs[0].size(-1) + 1023) / 1024),
                                             std::int64_t{2}};
            auto scratch = stream.acquire(temporary_shape, DType::F32);
            require(mps::encode_greedy_token(stream.commands(), inputs[0], scratch, output));
        } else if (eager)
            require(mps::encode_eager(stream.commands(), operation, epsilon, inputs, output));
        else if (quantized)
            require(quantized->encode(stream.commands(), inputs[0], output));
        else if (packed_bits) {
            Tensor calibrated_input;
            if (packed_input_scale > 0) {
                calibrated_input = stream.acquire(inputs[0].shape(), prefill_executable ? DType::F16 : DType::F32);
                require(mps::encode_eager(stream.commands(), Operation::STATIC_ROUND, packed_input_scale,
                                          inputs.first(1), calibrated_input));
            }
            if (prefill_executable) {
                auto weight = prefill_weight;
                if (!weight.defined()) {
                    const std::array<std::int64_t, 2> matrix_shape{static_cast<std::int64_t>(packed_weight.size(0)),
                                                                   static_cast<std::int64_t>(inputs[0].size(-1))};
                    weight = stream.acquire(matrix_shape, DType::F16);
                    require(mps::encode_expand_packed_weight(stream.commands(), packed_weight, packed_scales, weight,
                                                             packed_bits, packed_group));
                }
                const std::array feeds{calibrated_input.defined() ? calibrated_input : inputs[0], weight};
                auto projected = packed_output_scale > 0 ? stream.acquire(shape, DType::F32) : output;
                std::array outputs{projected};
                require(prefill_executable->encode(stream.commands(), feeds, outputs));
                if (packed_output_scale > 0)
                    require(mps::encode_eager(stream.commands(), Operation::STATIC_ROUND, packed_output_scale,
                                              {&projected}, output));
            } else {
                require(mps::encode_packed_linear(
                    stream.commands(), calibrated_input.defined() ? calibrated_input : inputs[0], packed_weight,
                    packed_scales, output, packed_bits, packed_group, 0.F, packed_output_scale, vector_projection));
            }
        } else {
            std::array outputs{output};
            require(executable->encode(stream.commands(), inputs.first(dynamic_count), outputs));
        }
        for (std::size_t index = 0; index < dynamic_count; ++index) stream.retain(inputs[index]);
        stream.retain(output);
        return output;
    }
    auto run_(TensorInputs inputs, Tensor& destination) -> Tensor override {
        if (destination.dtype() != dtype || !std::ranges::equal(destination.shape(), shape))
            throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "in-place operation cannot change shape or dtype"});
        if (scatter) {
            require(stream.commands().scatter_(destination, inputs[1], inputs[2]));
            for (std::size_t index = 0; index < inputs.size(); ++index) stream.retain(inputs[index]);
            return destination;
        }
        const auto output = run(inputs);
        require(stream.commands().copy_(output, destination));
        return destination;
    }
    auto run_pair(TensorInputs inputs) -> std::array<Tensor, 2> override {
        auto normalized = stream.acquire(shape, dtype);
        auto residual = stream.acquire(shape, dtype);
        std::array outputs{normalized, residual};
        require(executable->encode(stream.commands(), inputs.first(dynamic_count), outputs));
        for (std::size_t index = 0; index < dynamic_count; ++index) stream.retain(inputs[index]);
        for (const auto& output : outputs) stream.retain(output);
        return {std::move(residual), std::move(normalized)};
    }
    auto allocations() const -> AllocationStats override { return stream.output_allocations; }
};
class MetalBackend final : public OperatorBackend {
public:
    auto synchronize() -> void override { stream_.synchronize(); }
    auto release_cached_buffers() -> void override {
        stream_.outputs = OutputPool{};
        stream_.output_buffers.clear();
    }
    auto copy_slice_(Tensor& destination, const Tensor& source, std::size_t outer, std::size_t source_bytes,
                     std::size_t destination_bytes, std::size_t offset_bytes) -> void override {
        require(
            stream_.commands().copy_slice_(source, destination, outer, source_bytes, destination_bytes, offset_bytes));
        stream_.retain(source);
        stream_.retain(destination);
    }
    auto prepare(const OperatorSpec& spec, TensorInputs inputs) -> std::unique_ptr<Operator> override {
        auto result = std::make_unique<MetalOperator>(stream_);
        result->scatter = spec.operation == Operation::SCATTER;
        result->dtype = spec.dtype;
        if (spec.operation == Operation::GREEDY_TOKEN) {
            result->greedy = true;
            result->dynamic_count = 1;
            result->shape = {static_cast<std::int64_t>(inputs[0].numel() / inputs[0].size(-1))};
            return result;
        }
        bool device_fp32 = true;
        for (std::size_t index = 0; index < inputs.size(); ++index)
            device_fp32 &= inputs[index].dtype() == DType::F32 && inputs[index].device() == tensor::Device::apple_gpu();
        const bool simple_binary =
            (spec.operation == Operation::ADD || spec.operation == Operation::MULTIPLY) &&
            (std::ranges::equal(inputs[0].shape(), inputs[1].shape()) || inputs[1].numel() == 1 ||
             (inputs[0].dimensions() > 0 && inputs[1].dimensions() == 1 && inputs[1].size(0) == inputs[0].size(-1)));
        if (device_fp32 && (spec.operation == Operation::RMS_NORM || spec.operation == Operation::RMS_NORM_RESIDUAL ||
                            spec.operation == Operation::RMS_ROTARY || spec.operation == Operation::ROTARY ||
                            spec.operation == Operation::GELU_MULTIPLY || spec.operation == Operation::TANH ||
                            spec.operation == Operation::STATIC_ROUND ||
                            (spec.operation == Operation::GELU && spec.attributes[0]) || simple_binary)) {
            result->eager = true;
            result->operation = spec.operation;
            result->epsilon = spec.epsilon;
            result->dynamic_count = inputs.size();
            result->shape.assign(inputs[0].shape().begin(), inputs[0].shape().end());
            return result;
        }
        if (result->scatter && ops::is_inplace) {
            result->shape.assign(inputs[0].shape().begin(), inputs[0].shape().end());
            return result;
        }
        if (spec.operation == Operation::QUANTIZED_LINEAR) {
            result->dynamic_count = 1;
            result->quantized = require(mps::QuantizedLinear::create(inputs[1], inputs[2], inputs[3]));
            require(result->quantized->prepare(inputs[0]));
            result->shape.assign(inputs[0].shape().begin(), inputs[0].shape().end());
            result->shape.back() = inputs[1].size(1);
            return result;
        }
        if (spec.operation == Operation::PACKED_LINEAR) {
            result->vector_projection = spec.vector_projection;
            result->dynamic_count = 1;
            result->packed_bits = spec.attributes[0];
            result->packed_group = spec.attributes[1];
            result->packed_input_scale = spec.epsilon;
            result->packed_output_scale = std::bit_cast<float>(static_cast<std::int32_t>(spec.attributes[2]));
            result->packed_weight = require(inputs[1].to(tensor::Device::apple_gpu()));
            result->packed_scales = require(inputs[2].to(tensor::Device::apple_gpu()));
            result->shape.assign(inputs[0].shape().begin(), inputs[0].shape().end());
            result->shape.back() = inputs[1].size(0);
            if (!spec.packed_prefill && !spec.vector_projection && inputs[0].numel() / inputs[0].size(-1) >= 4 &&
                (result->packed_input_scale > 0 || result->packed_output_scale > 0)) {
                const std::array<std::int64_t, 2> matrix_shape{static_cast<std::int64_t>(inputs[1].size(0)),
                                                               static_cast<std::int64_t>(inputs[0].size(-1))};
                const auto key =
                    std::tuple{require(inputs[1].host_bytes()).data(), require(inputs[2].host_bytes()).data(),
                               result->packed_bits, result->packed_group};
                auto found = prefill_weights_.find(key);
                const auto matrix_bytes =
                    static_cast<std::size_t>(matrix_shape[0]) * matrix_shape[1] * sizeof(_Float16);
                if (found != prefill_weights_.end() ||
                    matrix_bytes <= stream_.prefill_cache_limit - stream_.expanded_bytes) {
                    if (found == prefill_weights_.end()) {
                        const auto columns = inputs[1].size(0), width = inputs[0].size(-1);
                        auto expanded = require(
                            Tensor::empty({static_cast<std::int64_t>(columns), static_cast<std::int64_t>(width)},
                                          DType::F16, tensor::Device::apple_gpu()));
                        stream_.retain(result->packed_weight);
                        stream_.retain(result->packed_scales);
                        stream_.retain(expanded);
                        require(mps::encode_expand_packed_weight(stream_.commands(), result->packed_weight,
                                                                 result->packed_scales, expanded, result->packed_bits,
                                                                 result->packed_group));
                        found = prefill_weights_.emplace(key, PrefillWeight{inputs[1], inputs[2], std::move(expanded)})
                                    .first;
                        stream_.expanded_bytes += found->second.value.nbytes();
                    }
                    result->prefill_weight = found->second.value;
                }
                std::vector<std::int64_t> program_key(inputs[0].shape().begin(), inputs[0].shape().end());
                program_key.push_back(matrix_shape[0]);
                program_key.push_back(spec.epsilon > 0);
                auto program = prefill_programs_.find(program_key);
                if (program == prefill_programs_.end()) {
                    auto graph = require(mps::Graph::create());
                    const std::array feeds{
                        graph.placeholder(inputs[0].shape(), spec.epsilon > 0 ? DType::F16 : DType::F32),
                        graph.placeholder(matrix_shape, DType::F16)};
                    const auto hidden = graph.matmul(graph.cast(feeds[0], DType::F16), feeds[1], false, true);
                    const std::array outputs{graph.cast(hidden, DType::F32)};
                    auto executable = std::make_shared<mps::Executable>(require(graph.compile(feeds, outputs)));
                    if (prefill_programs_.size() >= 128) prefill_programs_.clear();
                    program = prefill_programs_.emplace(std::move(program_key), std::move(executable)).first;
                }
                result->prefill_executable = program->second;
            }
            return result;
        }
        auto graph = require(mps::Graph::create());
        std::vector<mps::Value> operands, feeds;
        const bool paired = spec.operation == Operation::RESIDUAL_NORM;
        const auto constant = !spec.dynamic_parameters &&
                              (spec.operation == Operation::LINEAR || spec.operation == Operation::LAYER_NORM ||
                               spec.operation == Operation::RMS_NORM || paired);
        const std::size_t parameter_start = paired ? 2 : 1;
        for (std::size_t index = 0; index < inputs.size(); ++index) {
            if (constant && index >= parameter_start)
                operands.push_back(graph.constant(inputs[index]));
            else {
                auto value = graph.placeholder(inputs[index].shape(), inputs[index].dtype());
                operands.push_back(value);
                feeds.push_back(value);
            }
        }
        result->dynamic_count = feeds.size();
        mps::Value output;
        mps::Value residual;
        const std::array<std::int64_t, 1> axis{static_cast<std::int64_t>(inputs[0].dimensions()) - 1};
        switch (spec.operation) {
            case Operation::GELU_MULTIPLY:
                throw ops::Failure({ErrorCode::UNSUPPORTED, "GELU multiplication requires direct kernel"});
            case Operation::RMS_ROTARY:
                throw ops::Failure({ErrorCode::UNSUPPORTED, "RMS rotary requires direct kernel"});
            case Operation::GREEDY_TOKEN:
                throw ops::Failure({ErrorCode::UNSUPPORTED, "greedy selection must use the direct kernel"});
            case Operation::ADD:
                output = graph.add(operands[0], operands[1]);
                break;
            case Operation::MULTIPLY:
                output = graph.multiply(operands[0], operands[1]);
                break;
            case Operation::ATTENTION: {
                const auto heads = spec.attributes[0];
                const auto head_width = static_cast<std::int64_t>(inputs[0].size(2)) / heads;
                if (spec.attributes.size() >= 2) {
                    const auto key_heads = spec.attributes[1];
                    const std::array<std::int64_t, 5> axes{0, 2, 3, 1, 4};
                    const auto split = [&](std::size_t index) {
                        auto operand = operands[index];
                        auto length = static_cast<std::int64_t>(inputs[index].size(1));
                        if (index != 0 && spec.attributes.size() == 4) {
                            length = spec.attributes[3];
                            if (spec.attributes[2] || length != static_cast<std::int64_t>(inputs[index].size(1)))
                                operand = graph.slice(operand, 1, spec.attributes[2], length);
                        }
                        return graph.transpose(
                            graph.reshape(operand, {static_cast<std::int64_t>(inputs[index].size(0)), length, key_heads,
                                                    index == 0 ? heads / key_heads : 1, head_width}),
                            axes);
                    };
                    const auto query = split(0), key = split(1), value = split(2);
                    std::vector<std::int64_t> mask_shape(5 - inputs[3].dimensions(), 1);
                    mask_shape.insert(mask_shape.end(), inputs[3].shape().begin(), inputs[3].shape().end());
                    if (inputs[3].dimensions() == 4) {
                        mask_shape[0] = inputs[3].size(0);
                        mask_shape[1] = 1;
                    }
                    const auto mask = graph.reshape(operands[3], mask_shape);
                    auto scores =
                        graph.multiply(graph.matmul(query, key, false, true), graph.scalar(spec.epsilon, DType::F32));
                    scores = graph.add(scores, mask);
                    auto hidden = graph.matmul(graph.softmax(scores, 4), value);
                    const std::array<std::int64_t, 5> merge{0, 3, 1, 2, 4};
                    output = graph.reshape(graph.transpose(hidden, merge), inputs[0].shape());
                    break;
                }
                const std::array<std::int64_t, 4> axes{0, 2, 1, 3};
                const auto split = [&](std::size_t index) {
                    return graph.transpose(
                        graph.reshape(operands[index],
                                      {static_cast<std::int64_t>(inputs[index].size(0)),
                                       static_cast<std::int64_t>(inputs[index].size(1)), heads, head_width}),
                        axes);
                };
                auto scores = graph.matmul(split(0), split(1), false, true);
                const auto scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(head_width)));
                scores = graph.multiply(scores, graph.scalar(scale, DType::F32));
                if (inputs.size() == 4) scores = graph.add(scores, operands[3]);
                auto hidden = graph.matmul(graph.softmax(scores, 3), split(2));
                output = graph.reshape(graph.transpose(hidden, axes), inputs[0].shape());
                break;
            }
            case Operation::CAST:
                output = graph.cast(operands[0], spec.dtype);
                break;
            case Operation::MATMUL:
            case Operation::LINEAR:
                output = graph.matmul(operands[0], operands[1], false, spec.attributes[0]);
                output = graph.cast(output, DType::F32);
                if (spec.operation == Operation::LINEAR && operands.size() == 3)
                    output = graph.add(output, operands[2]);
                break;
            case Operation::GELU: {
                if (spec.attributes[0]) {
                    auto square = graph.multiply(operands[0], operands[0]);
                    auto cubic = graph.multiply(square, operands[0]);
                    auto inner = graph.add(operands[0], graph.multiply(cubic, graph.scalar(0.044715F, DType::F32)));
                    auto probability =
                        graph.add(graph.scalar(1.F, DType::F32),
                                  graph.tanh(graph.multiply(
                                      inner, graph.scalar(std::sqrt(2.F / 3.14159265358979323846F), DType::F32))));
                    output = graph.multiply(graph.multiply(operands[0], graph.scalar(0.5F, DType::F32)), probability);
                    break;
                }
                auto normalized = graph.multiply(operands[0], graph.scalar(std::sqrt(2.0F) / 2.0F, DType::F32));
                auto half = graph.scalar(0.5F, DType::F32);
                output = graph.multiply(operands[0], graph.add(graph.multiply(graph.erf(normalized), half), half));
                break;
            }
            case Operation::ROTARY: {
                const auto width = static_cast<std::int64_t>(inputs[0].size(3) / 2);
                auto first = graph.slice(operands[0], 3, 0, width);
                auto second = graph.slice(operands[0], 3, width, width);
                const std::array parts{
                    graph.subtract(graph.multiply(first, operands[1]), graph.multiply(second, operands[2])),
                    graph.add(graph.multiply(second, operands[1]), graph.multiply(first, operands[2]))};
                output = graph.concat(parts, 3);
                break;
            }
            case Operation::TANH:
                output = graph.tanh(operands[0]);
                break;
            case Operation::RMS_NORM: {
                auto square = graph.multiply(operands[0], operands[0]);
                auto mean_square =
                    graph.multiply(graph.reduce_sum(square, axis), graph.scalar(1.F / inputs[0].size(-1), DType::F32));
                auto zero = graph.scalar(0.F, DType::F32);
                output = graph.normalize(operands[0], zero, mean_square, operands[1], zero, spec.epsilon);
                break;
            }
            case Operation::RESIDUAL_NORM:
            case Operation::LAYER_NORM: {
                auto input = paired ? graph.add(operands[0], operands[1]) : operands[0];
                if (paired) residual = input;
                auto reciprocal = graph.scalar(1.0F / inputs[0].size(-1), DType::F32);
                auto mean = graph.multiply(graph.reduce_sum(input, axis), reciprocal);
                auto centered = graph.subtract(input, mean);
                auto variance = graph.multiply(graph.reduce_sum(graph.multiply(centered, centered), axis), reciprocal);
                output = graph.normalize(input, mean, variance, operands[parameter_start],
                                         operands[parameter_start + 1], spec.epsilon);
                break;
            }
            case Operation::SOFTMAX:
                output = graph.softmax(operands[0], axis[0]);
                break;
            case Operation::LOG_SOFTMAX: {
                auto shifted = graph.subtract(operands[0], graph.reduce_max(operands[0], axis));
                output = graph.subtract(shifted, graph.log(graph.reduce_sum(graph.exp(shifted), axis)));
                break;
            }
            case Operation::TRANSPOSE:
                output = graph.transpose(operands[0], spec.attributes);
                break;
            case Operation::SLICE:
                output = graph.slice(operands[0], spec.attributes[0], spec.attributes[1], spec.attributes[2]);
                break;
            case Operation::GATHER:
                output = graph.gather(operands[0], operands[1], spec.attributes[0]);
                break;
            case Operation::CONCAT:
                output = graph.concat(operands, spec.attributes[0]);
                break;
            case Operation::SCATTER:
                output = graph.scatter(operands[0], operands[1], operands[2], 1);
                break;
            default:
                throw ops::Failure({ErrorCode::UNSUPPORTED, "unsupported eager Metal operation"});
        }
        const std::array outputs{output, residual};
        result->executable = require(graph.compile(feeds, std::span(outputs).first(paired ? 2 : 1)));
        result->shape = require(result->executable->output_shapes()).front();
        return result;
    }

private:
    struct PrefillWeight {
        Tensor source, scale, value;
    };
    std::map<std::tuple<const std::byte*, const std::byte*, std::int32_t, std::int32_t>, PrefillWeight>
        prefill_weights_;
    std::map<std::vector<std::int64_t>, std::shared_ptr<const mps::Executable>> prefill_programs_;
    Stream stream_;
};
} // namespace
auto metal_operators() -> std::unique_ptr<OperatorBackend> { return std::make_unique<MetalBackend>(); }
} // namespace kidi::runtime