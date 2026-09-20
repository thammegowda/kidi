#include "kidi/runtime/operator.h"
#include "kidi/runtime/ynn/graph.h"
#include "kidi/ops/context.h"
#include "ynnpack/composites/composites.h"
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <numeric>
#include <algorithm>

namespace kidi::runtime {
namespace {
using ops::require;
using tensor::DType;
using tensor::Tensor;
auto type(DType dtype) -> ynn_type {
    switch (dtype) {
        case DType::F32:
            return ynn_type_fp32;
        case DType::BF16:
            return ynn_type_bf16;
        case DType::I8:
            return ynn_type_int8;
        case DType::I32:
            return ynn_type_int32;
        default:
            throw ops::Failure({ErrorCode::UNSUPPORTED, "unsupported eager CPU dtype"});
    }
}
auto check(ynn_status status) -> void { require(ynn::check_status(status, "eager CPU operator")); }
class Scatter final : public Operator {
public:
    explicit Scatter(tensor::Arena& arena) : pool_(&arena) {}
    auto run(TensorInputs inputs) -> Tensor override {
        const auto& input = inputs[0];
        auto output = pool_.acquire(input.shape(), input.dtype(), tensor::Device::cpu());
        auto target = require(output.host_bytes());
        const auto original = require(input.host_bytes());
        std::memcpy(target.data(), original.data(), original.size());
        write_updates(inputs, output);
        return output;
    }
    auto run_(TensorInputs inputs, Tensor& destination) -> Tensor override {
        write_updates(inputs, destination);
        return destination;
    }
    auto allocations() const -> AllocationStats override { return pool_.allocations(); }

private:
    static auto write_updates(TensorInputs inputs, Tensor& destination) -> void {
        const auto& update = inputs[1];
        auto indices = require(inputs[2].data<std::int32_t>());
        for (auto index : indices)
            if (index < 0 || static_cast<std::size_t>(index) >= destination.size(1))
                throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "cache scatter index out of bounds"});
        auto target = require(destination.host_bytes());
        auto source = require(update.host_bytes());
        const auto overlaps = [&](std::span<const std::byte> bytes) {
            const auto first = reinterpret_cast<std::uintptr_t>(bytes.data());
            const auto second = reinterpret_cast<std::uintptr_t>(target.data());
            return first < second + target.size() && second < first + bytes.size();
        };
        std::vector<std::byte> saved_updates;
        std::vector<std::int32_t> saved_indices;
        if (overlaps(source)) {
            saved_updates.assign(source.begin(), source.end());
            source = saved_updates;
        }
        if (overlaps(std::as_bytes(indices))) {
            saved_indices.assign(indices.begin(), indices.end());
            indices = saved_indices;
        }
        const auto row_bytes = update.size(2) * tensor::element_size(update.dtype());
        for (std::size_t batch = 0; batch < destination.size(0); ++batch)
            for (std::size_t index = 0; index < indices.size(); ++index)
                std::memcpy(target.data() + (batch * destination.size(1) + indices[index]) * row_bytes,
                            source.data() + (batch * indices.size() + index) * row_bytes, row_bytes);
    }

    OutputPool pool_;
};
class CpuOperator final : public Operator {
public:
    CpuOperator(ynn::Executable executable, std::size_t count, std::size_t dynamic_count, DType dtype,
                std::vector<std::int64_t> shape, tensor::Arena& arena)
        : executable_(std::move(executable)),
          count_(count),
          dynamic_count_(dynamic_count),
          dtype_(dtype),
          shape_(std::move(shape)),
          pool_(&arena) {}
    auto run(TensorInputs inputs) -> Tensor override {
        for (std::size_t index = 0; index < dynamic_count_; ++index) require(executable_.bind(index, inputs[index]));
        auto output = pool_.acquire(shape_, dtype_, tensor::Device::cpu());
        require(executable_.bind(count_, output));
        require(executable_.invoke());
        return output;
    }
    auto run_(TensorInputs inputs, Tensor& destination) -> Tensor override {
        if (destination.dtype() != dtype_ || !std::ranges::equal(destination.shape(), shape_))
            throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "in-place operation cannot change shape or dtype"});
        auto target = require(destination.host_bytes());
        const auto output = run(inputs);
        const auto source = require(output.host_bytes());
        std::memcpy(target.data(), source.data(), source.size());
        return destination;
    }
    auto run_pair(TensorInputs inputs) -> std::array<Tensor, 2> override {
        auto residual = pool_.acquire(shape_, dtype_, tensor::Device::cpu());
        require(executable_.bind(count_ + 1, residual));
        auto normalized = run(inputs);
        return {std::move(residual), std::move(normalized)};
    }
    auto allocations() const -> AllocationStats override { return pool_.allocations(); }

private:
    ynn::Executable executable_;
    std::size_t count_, dynamic_count_;
    DType dtype_;
    std::vector<std::int64_t> shape_;
    OutputPool pool_;
};
class CpuBackend final : public OperatorBackend {
public:
    CpuBackend() { require(arena_.reserve(8 * 1024 * 1024)); }
    auto synchronize() -> void override {}
    auto prepare(const OperatorSpec& spec, TensorInputs inputs) -> std::unique_ptr<Operator> override {
        if (spec.operation == Operation::SCATTER) return std::make_unique<Scatter>(arena_);
        const auto flags = inputs[0].dtype() == DType::BF16 ? YNN_FLAG_NO_EXCESS_PRECISION : 0;
        const bool paired = spec.operation == Operation::RESIDUAL_NORM;
        auto graph = require(ynn::Graph::create(inputs.size() + (paired ? 2 : 1), flags));
        auto native = graph.get();
        const auto constant = !spec.dynamic_parameters &&
                              (spec.operation == Operation::LINEAR || spec.operation == Operation::QUANTIZED_LINEAR ||
                               spec.operation == Operation::LAYER_NORM || spec.operation == Operation::RMS_NORM ||
                               spec.operation == Operation::PACKED_LINEAR || paired);
        const std::size_t dynamic_count = constant ? (paired ? 2 : 1) : inputs.size();
        const std::size_t column_alignment = spec.operation == Operation::PACKED_LINEAR ? 8 / spec.attributes[0] : 1;
        const auto padded_columns =
            spec.operation == Operation::PACKED_LINEAR
                ? ((inputs[1].size(0) + column_alignment - 1) / column_alignment) * column_alignment
                : 0;
        const bool pad_columns = spec.operation == Operation::PACKED_LINEAR && padded_columns != inputs[1].size(0);
        std::vector<std::uint32_t> operands;
        for (std::size_t index = 0; index < inputs.size(); ++index) {
            auto id = static_cast<std::uint32_t>(index);
            std::vector<std::size_t> shape(inputs[index].shape().begin(), inputs[index].shape().end());
            const bool parameter = index >= dynamic_count;
            if (parameter) id = YNN_INVALID_VALUE_ID;
            const bool packed = spec.operation == Operation::PACKED_LINEAR && index == 1;
            if (packed) shape[1] *= 8 / spec.attributes[0];
            const auto dtype = packed ? (spec.attributes[0] == 2   ? ynn_type_int2
                                         : spec.attributes[0] == 4 ? ynn_type_int4
                                                                   : ynn_type_int8)
                                      : type(inputs[index].dtype());
            std::vector<std::byte> padded;
            const auto* data = parameter ? require(inputs[index].host_bytes()).data() : nullptr;
            if (pad_columns && index > 0) {
                shape[0] = padded_columns;
                const auto bytes = require(inputs[index].host_bytes());
                padded.assign(bytes.begin(), bytes.end());
                padded.resize(bytes.size() / inputs[index].size(0) * padded_columns, std::byte{});
                data = padded.data();
            }
            check(ynn_define_tensor(
                native, dtype, shape.size(), shape.data(), data,
                parameter ? (padded.empty() ? 0 : YNN_VALUE_FLAG_COPY_DATA) : YNN_VALUE_FLAG_EXTERNAL_INPUT, &id));
            operands.push_back(id);
        }
        const auto scalar = [&](float value) {
            auto id = YNN_INVALID_VALUE_ID;
            check(ynn_define_tensor(native, ynn_type_fp32, 0, nullptr, &value, YNN_VALUE_FLAG_COPY_DATA, &id));
            return id;
        };
        const auto unary = [&](ynn_unary_operator operation, std::uint32_t input) {
            auto id = YNN_INVALID_VALUE_ID;
            check(ynn_define_unary(native, operation, input, &id, 0));
            return id;
        };
        const auto binary = [&](ynn_binary_operator operation, std::uint32_t left, std::uint32_t right) {
            auto id = YNN_INVALID_VALUE_ID;
            check(ynn_define_binary(native, operation, left, right, &id, 0));
            return id;
        };
        const std::int32_t last_axis = -1;
        const auto reduce = [&](ynn_reduce_operator operation, std::uint32_t input) {
            auto id = YNN_INVALID_VALUE_ID;
            check(ynn_define_reduce(native, operation, 1, &last_axis, input, YNN_INVALID_VALUE_ID, &id,
                                    YNN_NODE_FLAG_KEEP_DIMS));
            return id;
        };
        auto result = YNN_INVALID_VALUE_ID;
        auto residual = YNN_INVALID_VALUE_ID;
        auto dtype = spec.dtype;
        switch (spec.operation) {
            case Operation::ADD:
                result = binary(ynn_binary_add, operands[0], operands[1]);
                break;
            case Operation::MULTIPLY:
                result = binary(ynn_binary_multiply, operands[0], operands[1]);
                break;
            case Operation::ATTENTION: {
                const auto heads = static_cast<std::size_t>(spec.attributes[0]);
                const auto head_width = inputs[0].size(2) / heads;
                if (spec.attributes.size() >= 2) {
                    const auto key_heads = static_cast<std::size_t>(spec.attributes[1]);
                    const auto reshape = [&](std::uint32_t input, std::span<const std::size_t> shape) {
                        auto output = YNN_INVALID_VALUE_ID;
                        check(ynn_define_static_reshape(native, shape.size(), shape.data(), input, &output, 0));
                        return output;
                    };
                    const auto transpose = [&](std::uint32_t input, std::array<std::int32_t, 5> axes) {
                        auto output = YNN_INVALID_VALUE_ID;
                        check(ynn_define_static_transpose(native, axes.size(), axes.data(), input, &output, 0));
                        return output;
                    };
                    const auto split = [&](std::size_t index, bool key) {
                        auto operand = operands[index];
                        auto length = inputs[index].size(1);
                        if (index != 0 && spec.attributes.size() == 4) {
                            length = spec.attributes[3];
                            if (spec.attributes[2] || length != inputs[index].size(1)) {
                                const std::int32_t axis = 1;
                                const std::int64_t start = spec.attributes[2], end = start + length, stride = 1;
                                operand = YNN_INVALID_VALUE_ID;
                                check(ynn_define_static_slice(native, 1, &axis, &start, &end, &stride, operands[index],
                                                              &operand, 0));
                            }
                        }
                        const std::array shape{inputs[index].size(0), length, key_heads,
                                               index == 0 ? heads / key_heads : std::size_t{1}, head_width};
                        return transpose(reshape(operand, shape), key ? std::array<std::int32_t, 5>{0, 2, 3, 4, 1}
                                                                      : std::array<std::int32_t, 5>{0, 2, 3, 1, 4});
                    };
                    auto scores = YNN_INVALID_VALUE_ID, probability = YNN_INVALID_VALUE_ID,
                         hidden = YNN_INVALID_VALUE_ID;
                    check(ynn_define_dot(native, 1, split(0, false), split(1, true), YNN_INVALID_VALUE_ID, &scores, 0));
                    scores = binary(ynn_binary_multiply, scores, scalar(spec.epsilon));
                    std::vector<std::size_t> mask_shape(5 - inputs[3].dimensions(), 1);
                    mask_shape.insert(mask_shape.end(), inputs[3].shape().begin(), inputs[3].shape().end());
                    if (inputs[3].dimensions() == 4) {
                        mask_shape[0] = inputs[3].size(0);
                        mask_shape[1] = 1;
                    }
                    scores = binary(ynn_binary_add, scores, reshape(operands[3], mask_shape));
                    check(::ynn::define_softmax(native, scores, 1.F, probability));
                    check(ynn_define_dot(native, 1, probability, split(2, false), YNN_INVALID_VALUE_ID, &hidden, 0));
                    const std::array shape{inputs[0].size(0), inputs[0].size(1), inputs[0].size(2)};
                    result = reshape(transpose(hidden, {0, 3, 1, 2, 4}), shape);
                    break;
                }
                const auto reshape = [&](std::uint32_t input, std::span<const std::size_t> shape) {
                    auto output = YNN_INVALID_VALUE_ID;
                    check(ynn_define_static_reshape(native, shape.size(), shape.data(), input, &output, 0));
                    return output;
                };
                const auto transpose = [&](std::uint32_t input, std::array<std::int32_t, 4> axes) {
                    auto output = YNN_INVALID_VALUE_ID;
                    check(ynn_define_static_transpose(native, axes.size(), axes.data(), input, &output, 0));
                    return output;
                };
                const auto split = [&](std::size_t index) {
                    const std::array shape{inputs[index].size(0), inputs[index].size(1), heads, head_width};
                    return transpose(reshape(operands[index], shape), {0, 2, 1, 3});
                };
                auto query = split(0), key = transpose(split(1), {0, 1, 3, 2}), value = split(2);
                auto scores = YNN_INVALID_VALUE_ID;
                check(ynn_define_dot(native, 1, query, key, YNN_INVALID_VALUE_ID, &scores, 0));
                scores = binary(ynn_binary_multiply, scores,
                                scalar(static_cast<float>(1.0 / std::sqrt(static_cast<double>(head_width)))));
                if (inputs.size() == 4) scores = binary(ynn_binary_add, scores, operands[3]);
                auto probability = YNN_INVALID_VALUE_ID, hidden = YNN_INVALID_VALUE_ID;
                check(::ynn::define_softmax(native, scores, 1.F, probability));
                check(ynn_define_dot(native, 1, probability, value, YNN_INVALID_VALUE_ID, &hidden, 0));
                const std::array shape{inputs[0].size(0), inputs[0].size(1), inputs[0].size(2)};
                result = reshape(transpose(hidden, {0, 2, 1, 3}), shape);
                break;
            }
            case Operation::CAST:
                check(ynn_define_convert(native, operands[0], type(dtype), &result, YNN_NODE_FLAG_NO_EXCESS_PRECISION));
                break;
            case Operation::MATMUL:
            case Operation::LINEAR: {
                auto right = operands[1];
                if (spec.attributes[0]) {
                    std::vector<std::int32_t> axes(inputs[1].dimensions());
                    std::iota(axes.begin(), axes.end(), 0);
                    std::swap(axes[axes.size() - 1], axes[axes.size() - 2]);
                    right = YNN_INVALID_VALUE_ID;
                    check(ynn_define_static_transpose(native, axes.size(), axes.data(), operands[1], &right, 0));
                }
                check(ynn_define_dot(native, 1, operands[0], right, YNN_INVALID_VALUE_ID, &result, 0));
                if (spec.operation == Operation::LINEAR && operands.size() == 3)
                    result = binary(ynn_binary_add, result, operands[2]);
                break;
            }
            case Operation::PACKED_LINEAR:
            case Operation::QUANTIZED_LINEAR: {
                auto zero = YNN_INVALID_VALUE_ID, scale = YNN_INVALID_VALUE_ID, quantized = YNN_INVALID_VALUE_ID;
                if (spec.operation == Operation::PACKED_LINEAR && spec.epsilon > 0) {
                    const std::int32_t origin = 0;
                    check(ynn_define_tensor(native, ynn_type_int32, 0, nullptr, &origin, YNN_VALUE_FLAG_COPY_DATA,
                                            &zero));
                    scale = scalar(spec.epsilon);
                } else {
                    const auto range = reduce(ynn_reduce_min_max, operands[0]);
                    check(ynn_define_dynamic_quantization(native, range, ynn_type_int8, &zero, &scale, 0));
                }
                check(ynn_define_quantize(native, operands[0], ynn_type_int8, zero, scale, &quantized, 0));
                auto weight = operands[1];
                const bool packed = spec.operation == Operation::PACKED_LINEAR;
                if (packed) {
                    const std::array<std::int32_t, 2> axes{1, 0};
                    weight = YNN_INVALID_VALUE_ID;
                    check(ynn_define_static_transpose(native, 2, axes.data(), operands[1], &weight, 0));
                }
                check(::ynn::define_blockwise_dot(native, quantized, zero, scale, weight, YNN_INVALID_VALUE_ID,
                                                  operands[2], packed ? spec.attributes[1] : inputs[1].size(0),
                                                  packed ? YNN_INVALID_VALUE_ID : operands[3], ynn_type_fp32, result,
                                                  0));
                if (packed) {
                    const auto output_scale = std::bit_cast<float>(static_cast<std::int32_t>(spec.attributes[2]));
                    if (output_scale > 0) {
                        result = unary(ynn_unary_round, binary(ynn_binary_divide, result, scalar(output_scale)));
                        result = binary(
                            ynn_binary_multiply,
                            binary(ynn_binary_min, binary(ynn_binary_max, result, scalar(-128.F)), scalar(127.F)),
                            scalar(output_scale));
                    }
                }
                break;
            }
            case Operation::GELU: {
                if (spec.attributes[0]) {
                    auto square = binary(ynn_binary_multiply, operands[0], operands[0]);
                    auto cubic = binary(ynn_binary_multiply, square, operands[0]);
                    auto inner =
                        binary(ynn_binary_add, operands[0], binary(ynn_binary_multiply, cubic, scalar(0.044715F)));
                    auto probability = binary(
                        ynn_binary_add, scalar(1.F),
                        unary(ynn_unary_tanh,
                              binary(ynn_binary_multiply, inner, scalar(std::sqrt(2.F / 3.14159265358979323846F)))));
                    result = binary(ynn_binary_multiply, binary(ynn_binary_multiply, operands[0], scalar(0.5F)),
                                    probability);
                    break;
                }
                auto normalized = binary(ynn_binary_multiply, operands[0], scalar(std::sqrt(2.0F) / 2.0F));
                auto probability =
                    binary(ynn_binary_add, binary(ynn_binary_multiply, unary(ynn_unary_erf, normalized), scalar(0.5F)),
                           scalar(0.5F));
                result = binary(ynn_binary_multiply, operands[0], probability);
                break;
            }
            case Operation::ROTARY: {
                const std::int32_t axis = 3;
                const std::int64_t begin = 0, middle = inputs[0].size(3) / 2, end = inputs[0].size(3), stride = 1;
                auto first = YNN_INVALID_VALUE_ID, second = YNN_INVALID_VALUE_ID;
                check(ynn_define_static_slice(native, 1, &axis, &begin, &middle, &stride, operands[0], &first, 0));
                check(ynn_define_static_slice(native, 1, &axis, &middle, &end, &stride, operands[0], &second, 0));
                const std::array parts{binary(ynn_binary_subtract, binary(ynn_binary_multiply, first, operands[1]),
                                              binary(ynn_binary_multiply, second, operands[2])),
                                       binary(ynn_binary_add, binary(ynn_binary_multiply, second, operands[1]),
                                              binary(ynn_binary_multiply, first, operands[2]))};
                check(ynn_define_concatenate(native, axis, parts.size(), parts.data(), &result, 0));
                break;
            }
            case Operation::TANH:
                result = unary(ynn_unary_tanh, operands[0]);
                break;
            case Operation::STATIC_ROUND:
                result = unary(ynn_unary_round, binary(ynn_binary_divide, operands[0], scalar(spec.epsilon)));
                result = binary(ynn_binary_multiply,
                                binary(ynn_binary_min, binary(ynn_binary_max, result, scalar(-128.F)), scalar(127.F)),
                                scalar(spec.epsilon));
                break;
            case Operation::RMS_NORM_RESIDUAL:
            case Operation::RMS_NORM: {
                auto square = binary(ynn_binary_multiply, operands[0], operands[0]);
                auto mean_square =
                    binary(ynn_binary_multiply, reduce(ynn_reduce_sum, square), scalar(1.F / inputs[0].size(-1)));
                auto inverse =
                    unary(ynn_unary_reciprocal_square_root, binary(ynn_binary_add, mean_square, scalar(spec.epsilon)));
                result = binary(ynn_binary_multiply, binary(ynn_binary_multiply, operands[0], inverse), operands[1]);
                if (spec.operation == Operation::RMS_NORM_RESIDUAL) {
                    result = binary(ynn_binary_add, operands[2], result);
                    if (operands.size() == 4) result = binary(ynn_binary_multiply, result, operands[3]);
                }
                break;
            }
            case Operation::RESIDUAL_NORM:
            case Operation::LAYER_NORM: {
                const auto input = paired ? binary(ynn_binary_add, operands[0], operands[1]) : operands[0];
                if (paired) residual = input;
                auto reciprocal = scalar(1.0F / inputs[0].size(-1));
                auto mean = binary(ynn_binary_multiply, reduce(ynn_reduce_sum, input), reciprocal);
                auto centered = binary(ynn_binary_subtract, input, mean);
                auto variance =
                    binary(ynn_binary_multiply, reduce(ynn_reduce_sum, binary(ynn_binary_multiply, centered, centered)),
                           reciprocal);
                auto normalized = binary(
                    ynn_binary_multiply, centered,
                    unary(ynn_unary_reciprocal_square_root, binary(ynn_binary_add, variance, scalar(spec.epsilon))));
                result = binary(ynn_binary_add, binary(ynn_binary_multiply, normalized, operands[dynamic_count]),
                                operands[dynamic_count + 1]);
                break;
            }
            case Operation::SOFTMAX:
                check(::ynn::define_softmax(native, operands[0], 1.0F, result));
                break;
            case Operation::LOG_SOFTMAX: {
                auto shifted = binary(ynn_binary_subtract, operands[0], reduce(ynn_reduce_max, operands[0]));
                result = binary(ynn_binary_subtract, shifted,
                                unary(ynn_unary_log, reduce(ynn_reduce_sum, unary(ynn_unary_exp, shifted))));
                break;
            }
            case Operation::TRANSPOSE: {
                std::vector<std::int32_t> axes(spec.attributes.begin(), spec.attributes.end());
                check(ynn_define_static_transpose(native, axes.size(), axes.data(), operands[0], &result, 0));
                break;
            }
            case Operation::SLICE: {
                const auto axis = static_cast<std::int32_t>(spec.attributes[0]);
                const std::int64_t end = spec.attributes[1] + spec.attributes[2], stride = 1;
                check(ynn_define_static_slice(native, 1, &axis, &spec.attributes[1], &end, &stride, operands[0],
                                              &result, 0));
                break;
            }
            case Operation::GATHER: {
                const auto axis = static_cast<std::int32_t>(spec.attributes[0]);
                check(ynn_define_gather(native, 1, &axis, inputs[0].dimensions() + inputs[1].dimensions() - 1,
                                        operands[0], operands[1], &result, 0));
                break;
            }
            case Operation::CONCAT:
                check(ynn_define_concatenate(native, spec.attributes[0], operands.size(), operands.data(), &result, 0));
                break;
            case Operation::SCATTER:
                break;
        }
        auto output_id = static_cast<std::uint32_t>(inputs.size());
        if (pad_columns) {
            const std::int32_t axis = -1;
            const std::int64_t start = 0, end = inputs[1].size(0), stride = 1;
            auto sliced = YNN_INVALID_VALUE_ID;
            check(ynn_define_static_slice(native, 1, &axis, &start, &end, &stride, result, &sliced, 0));
            result = sliced;
        }
        check(ynn_define_tensor(native, type(dtype), 0, nullptr, nullptr, YNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
        check(ynn_define_copy(native, result, &output_id, 0));
        if (paired) {
            auto residual_id = output_id + 1;
            check(ynn_define_tensor(native, type(dtype), 0, nullptr, nullptr, YNN_VALUE_FLAG_EXTERNAL_OUTPUT,
                                    &residual_id));
            check(ynn_define_copy(native, residual, &residual_id, 0));
        }
        std::size_t operator_threads = 0;
#if defined(__APPLE__) && (defined(__aarch64__) || defined(__arm64__))
        const bool small_projection = spec.operation == Operation::QUANTIZED_LINEAR &&
                                      inputs[0].numel() == inputs[0].size(-1) && inputs[1].numel() <= 4 * 1024 * 1024;
        const bool small_attention = spec.operation == Operation::ATTENTION && inputs[0].size(0) == 1 &&
                                     inputs[0].size(1) == 1 && inputs[1].numel() <= 256 * 1024;
        if (small_projection || small_attention) operator_threads = 1;
#endif
        auto executable = require(std::move(graph).compile(operator_threads));
        require(executable.reshape());
        auto shape = require(executable.shape(output_id));
        return std::make_unique<CpuOperator>(std::move(executable), inputs.size(), dynamic_count, dtype,
                                             std::vector<std::int64_t>(shape.begin(), shape.end()), arena_);
    }

private:
    tensor::Arena arena_;
};
} // namespace
auto cpu_operators() -> std::unique_ptr<OperatorBackend> { return std::make_unique<CpuBackend>(); }
} // namespace kidi::runtime