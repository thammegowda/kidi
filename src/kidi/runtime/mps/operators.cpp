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

namespace kidi::runtime {
namespace {
using ops::require;
using tensor::DType;
using tensor::Tensor;
struct Stream {
    std::optional<mps::CommandBatch> batch;
    std::vector<Tensor> pending;
    std::size_t pending_count = 0;
    Stream() { pending.reserve(512); }
    auto retain(const Tensor& tensor) -> void {
        if (pending_count == pending.size())
            pending.push_back(tensor);
        else {
            const auto previous = tensor::metal_buffer(pending[pending_count]);
            const auto current = tensor::metal_buffer(tensor);
            if (!previous || !current || previous->handle != current->handle) pending[pending_count] = tensor;
        }
        ++pending_count;
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
        pending_count = 0;
    }
};
class MetalOperator final : public Operator {
public:
    Stream& stream;
    std::optional<mps::Executable> executable;
    std::optional<mps::QuantizedLinear> quantized;
    Tensor packed_weight, packed_scales, packed_input;
    Tensor prefill_weight, prefill_output;
    std::optional<mps::Executable> prefill_executable;
    std::int32_t packed_bits = 0, packed_group = 0;
    float packed_input_scale = 0, packed_output_scale = 0;
    std::vector<std::int64_t> shape;
    DType dtype;
    std::size_t dynamic_count;
    bool scatter = false;
    bool eager = false;
    Operation operation;
    float epsilon = 0;
    OutputPool pool;
    explicit MetalOperator(Stream& owner) : stream(owner) {}
    auto run(TensorInputs inputs) -> Tensor override {
        auto output = pool.acquire(shape, dtype, tensor::Device::apple_gpu());
        if (eager)
            require(mps::encode_eager(stream.commands(), operation, epsilon, inputs, output));
        else if (quantized)
            require(quantized->encode(stream.commands(), inputs[0], output));
        else if (packed_bits) {
            if (packed_input.defined())
                require(mps::encode_eager(stream.commands(), Operation::STATIC_ROUND, packed_input_scale,
                                          inputs.first(1), packed_input));
            if (prefill_executable) {
                const std::array feeds{packed_input.defined() ? packed_input : inputs[0], prefill_weight};
                std::array outputs{prefill_output.defined() ? prefill_output : output};
                require(prefill_executable->encode(stream.commands(), feeds, outputs));
                if (prefill_output.defined())
                    require(mps::encode_eager(stream.commands(), Operation::STATIC_ROUND, packed_output_scale,
                                              {&prefill_output}, output));
            } else {
                require(mps::encode_packed_linear(stream.commands(), packed_input.defined() ? packed_input : inputs[0],
                                                  packed_weight, packed_scales, output, packed_bits, packed_group, 0.F,
                                                  packed_output_scale));
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
        auto normalized = pool.acquire(shape, dtype, tensor::Device::apple_gpu());
        auto residual = pool.acquire(shape, dtype, tensor::Device::apple_gpu());
        std::array outputs{normalized, residual};
        require(executable->encode(stream.commands(), inputs.first(dynamic_count), outputs));
        for (std::size_t index = 0; index < dynamic_count; ++index) stream.retain(inputs[index]);
        for (const auto& output : outputs) stream.retain(output);
        return {std::move(residual), std::move(normalized)};
    }
    auto allocations() const -> AllocationStats override { return pool.allocations(); }
};
class MetalBackend final : public OperatorBackend {
public:
    auto synchronize() -> void override { stream_.synchronize(); }
    auto prepare(const OperatorSpec& spec, TensorInputs inputs) -> std::unique_ptr<Operator> override {
        auto result = std::make_unique<MetalOperator>(stream_);
        result->scatter = spec.operation == Operation::SCATTER;
        result->dtype = spec.dtype;
        bool device_fp32 = true;
        for (std::size_t index = 0; index < inputs.size(); ++index)
            device_fp32 &= inputs[index].dtype() == DType::F32 && inputs[index].device() == tensor::Device::apple_gpu();
        const bool simple_binary =
            (spec.operation == Operation::ADD || spec.operation == Operation::MULTIPLY) &&
            (std::ranges::equal(inputs[0].shape(), inputs[1].shape()) || inputs[1].numel() == 1 ||
             (inputs[0].dimensions() > 0 && inputs[1].dimensions() == 1 && inputs[1].size(0) == inputs[0].size(-1)));
        if (device_fp32 && (spec.operation == Operation::RMS_NORM || spec.operation == Operation::RMS_NORM_RESIDUAL ||
                            spec.operation == Operation::ROTARY || spec.operation == Operation::TANH ||
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
            result->dynamic_count = 1;
            result->packed_bits = spec.attributes[0];
            result->packed_group = spec.attributes[1];
            result->packed_input_scale = spec.epsilon;
            result->packed_output_scale = std::bit_cast<float>(static_cast<std::int32_t>(spec.attributes[2]));
            if (spec.epsilon > 0)
                result->packed_input = require(Tensor::empty({inputs[0].shape().begin(), inputs[0].shape().end()},
                                                             DType::F32, tensor::Device::apple_gpu()));
            result->packed_weight = require(inputs[1].to(tensor::Device::apple_gpu()));
            result->packed_scales = require(inputs[2].to(tensor::Device::apple_gpu()));
            result->shape.assign(inputs[0].shape().begin(), inputs[0].shape().end());
            result->shape.back() = inputs[1].size(0);
            if (!spec.packed_prefill && inputs[0].numel() / inputs[0].size(-1) >= 4 &&
                (result->packed_input_scale > 0 || result->packed_output_scale > 0)) {
                const auto key =
                    std::tuple{require(inputs[1].host_bytes()).data(), require(inputs[2].host_bytes()).data(),
                               result->packed_bits, result->packed_group};
                auto found = prefill_weights_.find(key);
                if (found == prefill_weights_.end()) {
                    const auto columns = inputs[1].size(0), width = inputs[0].size(-1);
                    auto expanded =
                        require(Tensor::empty({static_cast<std::int64_t>(columns), static_cast<std::int64_t>(width)},
                                              DType::F16, tensor::Device::apple_gpu()));
                    auto destination = reinterpret_cast<_Float16*>(require(expanded.host_bytes()).data());
                    const auto source = require(inputs[1].data<std::uint8_t>());
                    const auto scales = require(inputs[2].data<float>());
                    const auto bits = result->packed_bits, per_byte = 8 / bits;
                    const auto groups = width / result->packed_group;
                    for (std::size_t column = 0; column < columns; ++column)
                        for (std::size_t channel = 0; channel < width; ++channel) {
                            const auto offset = column * width + channel;
                            const auto raw =
                                (source[offset / per_byte] >> ((offset % per_byte) * bits)) & ((1 << bits) - 1);
                            const auto integer = (raw ^ (1 << (bits - 1))) - (1 << (bits - 1));
                            destination[offset] = static_cast<_Float16>(
                                integer * scales[column * groups + channel / result->packed_group]);
                        }
                    found =
                        prefill_weights_.emplace(key, PrefillWeight{inputs[1], inputs[2], std::move(expanded)}).first;
                }
                result->prefill_weight = found->second.value;
                auto graph = require(mps::Graph::create());
                const std::array feeds{graph.placeholder(inputs[0].shape(), DType::F32),
                                       graph.placeholder(result->prefill_weight.shape(), DType::F16)};
                const auto hidden = graph.matmul(graph.cast(feeds[0], DType::F16), feeds[1], false, true);
                const std::array outputs{graph.cast(hidden, DType::F32)};
                result->prefill_executable = require(graph.compile(feeds, outputs));
                if (result->packed_output_scale > 0)
                    result->prefill_output =
                        require(Tensor::empty(result->shape, DType::F32, tensor::Device::apple_gpu()));
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
                    auto scores = graph.matmul(split(0), split(1), false, true);
                    scores = graph.multiply(scores, graph.scalar(spec.epsilon, DType::F32));
                    std::vector<std::int64_t> mask_shape(5 - inputs[3].dimensions(), 1);
                    mask_shape.insert(mask_shape.end(), inputs[3].shape().begin(), inputs[3].shape().end());
                    if (inputs[3].dimensions() == 4) {
                        mask_shape[0] = inputs[3].size(0);
                        mask_shape[1] = 1;
                    }
                    scores = graph.add(scores, graph.reshape(operands[3], mask_shape));
                    auto hidden = graph.matmul(graph.softmax(scores, 4), split(2));
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
    Stream stream_;
};
} // namespace
auto metal_operators() -> std::unique_ptr<OperatorBackend> { return std::make_unique<MetalBackend>(); }
} // namespace kidi::runtime