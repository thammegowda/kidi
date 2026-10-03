#include "kidi/runtime/operator.h"
#include "kidi/runtime/ynn/graph.h"
#include "kidi/ops/context.h"
#include "ynnpack/composites/composites.h"
#include <array>
#include <functional>
#include <bit>
#include <cmath>
#include <cstring>
#include <numeric>
#include <algorithm>
#include <limits>
#include <cstdlib>

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
#if defined(__EMSCRIPTEN__)
// Single-row calibrated projections use the same INT8 dot as multi-row ones, so YNNPACK's constant cache shares one
// packed copy of each weight between prefill and decode. WebAssembly decode measured no slower this way.
constexpr bool SHARED_DECODE_PACKING = true;
#else
constexpr bool SHARED_DECODE_PACKING = false;
#endif
auto transpose_packed_tile(const Tensor& input, std::int32_t bits, std::size_t row_start, std::size_t row_count,
                           std::size_t column_start, std::size_t column_count) -> std::vector<std::uint8_t> {
    const auto values_per_byte = static_cast<std::size_t>(8 / bits);
    const auto logical_columns = input.size(1) * values_per_byte;
    if (input.dtype() != DType::U8 || input.dimensions() != 2 || row_start + row_count > input.size(0) ||
        column_start + column_count > logical_columns || row_count % values_per_byte)
        throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "invalid packed feed-forward tile"});
    const auto& source_tensor = input;
    const auto source = require(source_tensor.data<std::uint8_t>());
    const auto destination_row_bytes = row_count / values_per_byte;
    std::vector<std::uint8_t> destination(column_count * destination_row_bytes);
    const auto mask = static_cast<std::uint8_t>((1 << bits) - 1);
    for (std::size_t row = 0; row < row_count; ++row)
        for (std::size_t column = 0; column < column_count; ++column) {
            const auto source_column = column_start + column;
            const auto value = static_cast<std::uint8_t>(
                (source[(row_start + row) * input.size(1) + source_column / values_per_byte] >>
                 ((source_column % values_per_byte) * bits)) &
                mask);
            destination[column * destination_row_bytes + row / values_per_byte] |=
                static_cast<std::uint8_t>(value << ((row % values_per_byte) * bits));
        }
    return destination;
}
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
    auto run_into(TensorInputs inputs, std::span<Tensor> outputs) -> void override {
        auto target = require(outputs[0].host_bytes());
        const auto original = require(inputs[0].host_bytes());
        std::memcpy(target.data(), original.data(), original.size());
        write_updates(inputs, outputs[0]);
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
class GreedyToken final : public Operator {
public:
    explicit GreedyToken(tensor::Arena& arena) : pool_(&arena) {}
    auto run(TensorInputs inputs) -> Tensor override {
        const std::array shape{static_cast<std::int64_t>(inputs[0].numel() / inputs[0].size(-1))};
        auto output = pool_.acquire(shape, DType::I32, tensor::Device::cpu());
        select(inputs[0], output);
        return output;
    }
    auto run_(TensorInputs, Tensor&) -> Tensor override {
        throw ops::Failure({ErrorCode::UNSUPPORTED, "greedy selection is not an in-place operation"});
    }
    auto run_into(TensorInputs inputs, std::span<Tensor> outputs) -> void override { select(inputs[0], outputs[0]); }
    auto allocations() const -> AllocationStats override { return pool_.allocations(); }

private:
    static auto select(const Tensor& logits, Tensor& output) -> void {
        const auto width = logits.size(-1), rows = logits.numel() / width;
        const auto values = require(logits.data<float>());
        auto tokens = require(output.data<std::int32_t>());
        for (std::size_t row = 0; row < rows; ++row) {
            float best = -std::numeric_limits<float>::infinity();
            std::int32_t token = -1;
            bool invalid = false;
            for (std::size_t column = 0; column < width; ++column) {
                const auto value = values[row * width + column];
                invalid |= std::isnan(value) || value == std::numeric_limits<float>::infinity();
                if (value > best) {
                    best = value;
                    token = column;
                }
            }
            tokens[row] = invalid ? -1 : token;
        }
    }

    OutputPool pool_;
};
class QuantizeInt8 final : public Operator {
public:
    QuantizeInt8(float scale, tensor::Arena& arena) : scale_(scale), pool_(&arena) {}
    auto run(TensorInputs inputs) -> Tensor override {
        auto output = pool_.acquire(inputs[0].shape(), DType::I8, tensor::Device::cpu());
        quantize(inputs[0], output);
        return output;
    }
    auto run_(TensorInputs, Tensor&) -> Tensor override {
        throw ops::Failure({ErrorCode::UNSUPPORTED, "quantization cannot run in place"});
    }
    auto run_into(TensorInputs inputs, std::span<Tensor> outputs) -> void override { quantize(inputs[0], outputs[0]); }
    auto allocations() const -> AllocationStats override { return pool_.allocations(); }

private:
    auto quantize(const Tensor& input, Tensor& output) const -> void {
        const auto source = require(input.data<float>());
        auto destination = require(output.data<std::int8_t>());
        for (std::size_t index = 0; index < source.size(); ++index) {
            const auto rounded = std::nearbyint(source[index] / scale_);
            destination[index] = static_cast<std::int8_t>(std::clamp(rounded, -128.F, 127.F));
        }
    }

    float scale_;
    OutputPool pool_;
};
/// Token rows gathered from FP32/BF16 tables, or signed 2/4/8-bit tables with per-row group scales, times a multiplier.
/// Splits `rows` across the thread pool in contiguous blocks; small problems run on the calling thread.
auto for_rows(std::size_t rows, std::size_t row_work,
              const std::function<void(std::size_t, std::size_t)>& body) -> void {
    const auto tasks = rows * row_work < 32768 ? std::size_t{1} : std::min(ynn::thread_count(), rows);
    if (tasks <= 1) return body(0, rows);
    const auto block = (rows + tasks - 1) / tasks;
    require(ynn::parallel_for(tasks, [&](std::size_t task) {
        const auto begin = task * block;
        if (begin < rows) body(begin, std::min(rows, begin + block));
    }));
}

/// Rotary embedding of `[batch, rows, heads, width]` by `[batch or 1, rows, 1, width / 2]` angles in one pass.
class Rotary final : public Operator {
public:
    explicit Rotary(tensor::Arena& arena) : pool_(&arena) {}
    static auto supports(TensorInputs inputs) -> bool {
        if (inputs.size() != 3) return false;
        const auto &input = inputs[0], &cosine = inputs[1], &sine = inputs[2];
        return input.dtype() == DType::F32 && cosine.dtype() == DType::F32 && sine.dtype() == DType::F32 &&
               input.dimensions() == 4 && cosine.dimensions() == 4 && input.is_contiguous() && cosine.is_contiguous() &&
               sine.is_contiguous() && std::ranges::equal(cosine.shape(), sine.shape()) && input.size(3) % 2 == 0 &&
               cosine.size(3) == input.size(3) / 2 && cosine.size(2) == 1 && cosine.size(1) == input.size(1) &&
               (cosine.size(0) == input.size(0) || cosine.size(0) == 1);
    }
    auto run(TensorInputs inputs) -> Tensor override {
        auto output = pool_.acquire(inputs[0].shape(), DType::F32, tensor::Device::cpu());
        apply(inputs, output);
        return output;
    }
    auto run_(TensorInputs inputs, Tensor& destination) -> Tensor override {
        apply(inputs, destination);
        return destination;
    }
    auto run_into(TensorInputs inputs, std::span<Tensor> outputs) -> void override { apply(inputs, outputs[0]); }
    auto allocations() const -> AllocationStats override { return pool_.allocations(); }

private:
    static auto apply(TensorInputs inputs, Tensor& output) -> void {
        const auto &input = inputs[0], &cosine = inputs[1];
        const auto rows = input.size(1), heads = input.size(2), width = input.size(3), half = width / 2;
        const auto shared_angles = cosine.size(0) == 1;
        const auto* source = require(input.data<float>()).data();
        const auto* cosines = require(cosine.data<float>()).data();
        const auto* sines = require(inputs[2].data<float>()).data();
        auto* target = require(output.data<float>()).data();
        for_rows(input.size(0) * rows, heads * width, [&](std::size_t begin, std::size_t end) {
        // Matches the separate multiply, subtract, and add of the graph path: no fused multiply-add.
#pragma clang fp contract(off)
            for (auto row = begin; row < end; ++row) {
                const auto angle = (shared_angles ? row % rows : row) * half;
                const auto* c = cosines + angle;
                const auto* s = sines + angle;
                for (std::size_t head = 0; head < heads; ++head) {
                    const auto* x = source + (row * heads + head) * width;
                    auto* y = target + (row * heads + head) * width;
                    for (std::size_t index = 0; index < half; ++index) {
                        const auto first = x[index], second = x[index + half];
                        y[index] = first * c[index] - second * s[index];
                        y[index + half] = second * c[index] + first * s[index];
                    }
                }
            }
        });
    }
    OutputPool pool_;
};

/// RMS normalization over the last axis with a per-channel weight; the residual form adds `inputs[2]` and optionally
/// scales by the scalar `inputs[3]`.
class RmsNorm final : public Operator {
public:
    RmsNorm(const OperatorSpec& spec, tensor::Arena& arena)
        : epsilon_(spec.epsilon), residual_(spec.operation == Operation::RMS_NORM_RESIDUAL), pool_(&arena) {}
    static auto supports(const OperatorSpec& spec, TensorInputs inputs) -> bool {
        const bool residual = spec.operation == Operation::RMS_NORM_RESIDUAL;
        if (inputs.size() < 2 || inputs.size() > (residual ? 4U : 2U) || (residual && inputs.size() < 3)) return false;
        const auto &input = inputs[0], &weight = inputs[1];
        if (input.dtype() != DType::F32 || weight.dtype() != DType::F32 || !input.is_contiguous() ||
            !weight.is_contiguous() || input.dimensions() == 0 || weight.numel() != input.size(-1) ||
            weight.numel() != weight.size(-1))
            return false;
        if (!residual) return true;
        return inputs[2].dtype() == DType::F32 && inputs[2].is_contiguous() &&
               std::ranges::equal(inputs[2].shape(), input.shape()) &&
               (inputs.size() == 3 || (inputs[3].dtype() == DType::F32 && inputs[3].numel() == 1));
    }
    auto run(TensorInputs inputs) -> Tensor override {
        auto output = pool_.acquire(inputs[0].shape(), DType::F32, tensor::Device::cpu());
        apply(inputs, output);
        return output;
    }
    auto run_(TensorInputs inputs, Tensor& destination) -> Tensor override {
        apply(inputs, destination);
        return destination;
    }
    auto run_into(TensorInputs inputs, std::span<Tensor> outputs) -> void override { apply(inputs, outputs[0]); }
    auto allocations() const -> AllocationStats override { return pool_.allocations(); }

private:
    auto apply(TensorInputs inputs, Tensor& output) const -> void {
        const auto width = inputs[0].size(-1), rows = inputs[0].numel() / width;
        const auto* source = require(inputs[0].data<float>()).data();
        const auto* weight = require(inputs[1].data<float>()).data();
        const auto* residual = residual_ ? require(inputs[2].data<float>()).data() : nullptr;
        const auto scale = residual_ && inputs.size() == 4 ? require(inputs[3].data<float>())[0] : 1.F;
        const bool scaled = residual_ && inputs.size() == 4;
        auto* target = require(output.data<float>()).data();
        const auto reciprocal = 1.F / static_cast<float>(width);
        for_rows(rows, width, [&](std::size_t begin, std::size_t end) {
#pragma clang fp contract(off)
            for (auto row = begin; row < end; ++row) {
                const auto* x = source + row * width;
                auto* y = target + row * width;
                std::array<float, 8> partial{};
                std::size_t index = 0;
                for (; index + 8 <= width; index += 8)
                    for (std::size_t lane = 0; lane < 8; ++lane) partial[lane] += x[index + lane] * x[index + lane];
                auto sum = 0.F;
                for (; index < width; ++index) sum += x[index] * x[index];
                for (const auto value : partial) sum += value;
                const auto inverse = 1.F / std::sqrt(sum * reciprocal + epsilon_);
                if (!residual_) {
                    for (index = 0; index < width; ++index) y[index] = x[index] * inverse * weight[index];
                    continue;
                }
                const auto* r = residual + row * width;
                for (index = 0; index < width; ++index) {
                    const auto value = r[index] + x[index] * inverse * weight[index];
                    y[index] = scaled ? value * scale : value;
                }
            }
        });
    }
    float epsilon_;
    bool residual_;
    OutputPool pool_;
};

class Embedding final : public Operator {
public:
    Embedding(const OperatorSpec& spec, TensorInputs inputs, tensor::Arena& arena)
        : width_(static_cast<std::size_t>(spec.attributes[0])),
          bits_(static_cast<int>(spec.attributes[1])),
          groups_(static_cast<std::size_t>(spec.attributes[2])),
          multiplier_(spec.epsilon),
          pool_(&arena) {
        const auto& weight = inputs[1];
        const bool packed = bits_ != 0;
        const auto stored = packed ? width_ / (8 / bits_) : width_;
        if (weight.size(1) != stored || (packed && width_ % (8 / bits_)) ||
            (packed ? (weight.dtype() != DType::U8 && !(bits_ == 8 && weight.dtype() == DType::I8)) ||
                          inputs.size() != 3 || inputs[2].dtype() != DType::F32 || inputs[2].dimensions() != 2 ||
                          inputs[2].size(0) != weight.size(0) || inputs[2].size(1) != groups_ || !groups_ ||
                          width_ % groups_
                    : weight.dtype() != DType::F32 && weight.dtype() != DType::BF16))
            throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "invalid embedding table layout"});
    }
    auto run(TensorInputs inputs) -> Tensor override {
        const std::array shape{std::int64_t{1}, static_cast<std::int64_t>(inputs[0].numel()),
                               static_cast<std::int64_t>(width_)};
        auto output = pool_.acquire(shape, DType::F32, tensor::Device::cpu());
        lookup(inputs, output);
        return output;
    }
    auto run_(TensorInputs, Tensor&) -> Tensor override {
        throw ops::Failure({ErrorCode::UNSUPPORTED, "embedding is not an in-place operation"});
    }
    auto run_into(TensorInputs inputs, std::span<Tensor> outputs) -> void override { lookup(inputs, outputs[0]); }
    auto allocations() const -> AllocationStats override { return pool_.allocations(); }

private:
    auto lookup(TensorInputs inputs, Tensor& output) const -> void {
        const auto tokens = require(inputs[0].data<std::int32_t>());
        const auto& weight = inputs[1];
        const auto bytes = require(weight.host_bytes());
        auto values = require(output.data<float>());
        for (const auto token : tokens)
            if (token < 0 || static_cast<std::size_t>(token) >= weight.size(0))
                throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "embedding token outside vocabulary"});
        if (bits_) {
            const auto scales = require(inputs[2].data<float>());
            const auto group_width = width_ / groups_, per_byte = static_cast<std::size_t>(8 / bits_);
            const auto data = reinterpret_cast<const std::uint8_t*>(bytes.data());
            const auto sign = 1 << (bits_ - 1);
            for (std::size_t row = 0; row < tokens.size(); ++row)
                for (std::size_t channel = 0; channel < width_; ++channel) {
                    const auto offset = static_cast<std::size_t>(tokens[row]) * width_ + channel;
                    const auto raw = (data[offset / per_byte] >> ((offset % per_byte) * bits_)) & ((1 << bits_) - 1);
                    const auto integer = (raw ^ sign) - sign;
                    values[row * width_ + channel] =
                        integer * scales[tokens[row] * groups_ + channel / group_width] * multiplier_;
                }
            return;
        }
        for (std::size_t row = 0; row < tokens.size(); ++row)
            for (std::size_t channel = 0; channel < width_; ++channel) {
                const auto offset = static_cast<std::size_t>(tokens[row]) * width_ + channel;
                const auto value =
                    weight.dtype() == DType::BF16
                        ? std::bit_cast<float>(
                              static_cast<std::uint32_t>(reinterpret_cast<const std::uint16_t*>(bytes.data())[offset])
                              << 16)
                        : reinterpret_cast<const float*>(bytes.data())[offset];
                values[row * width_ + channel] = value * multiplier_;
            }
    }

    std::size_t width_;
    int bits_;
    std::size_t groups_;
    float multiplier_;
    OutputPool pool_;
};
class QuantizedAttention final : public Operator {
public:
    QuantizedAttention(const OperatorSpec& spec, TensorInputs inputs, tensor::Arena& arena)
        : heads_(static_cast<std::size_t>(spec.attributes[0])),
          key_heads_(static_cast<std::size_t>(spec.attributes[1])),
          key_start_(static_cast<std::size_t>(spec.attributes[2])),
          key_length_(static_cast<std::size_t>(spec.attributes[3])),
          scale_(spec.epsilon),
          key_scales_(spec.quantization[0].scales.begin(), spec.quantization[0].scales.end()),
          key_zero_points_(spec.quantization[0].zero_points.begin(), spec.quantization[0].zero_points.end()),
          key_block_size_(spec.quantization[0].block_size),
          value_scales_(spec.quantization[1].scales.begin(), spec.quantization[1].scales.end()),
          value_zero_points_(spec.quantization[1].zero_points.begin(), spec.quantization[1].zero_points.end()),
          value_block_size_(spec.quantization[1].block_size),
          batch_size_(inputs[0].size(0)),
          query_length_(inputs[0].size(1)),
          head_dim_(inputs[0].size(2) / heads_),
          key_batch_stride_(inputs[1].size(0) == 1 ? 0 : inputs[1].size(1) * inputs[1].size(2)),
          pool_(&arena) {
        const auto& mask = inputs[3];
        std::array<std::size_t, 4> strides{};
        std::size_t stride = 1;
        for (std::size_t source_axis = mask.dimensions(); source_axis-- > 0;) {
            const auto target_axis = 4 - mask.dimensions() + source_axis;
            strides[target_axis] = mask.size(source_axis) == 1 ? 0 : stride;
            stride *= mask.size(source_axis);
        }
        mask_batch_stride_ = strides[0];
        mask_head_stride_ = strides[1];
        mask_query_stride_ = strides[2];
        mask_key_stride_ = strides[3];
        const auto params = parameters();
        work_items_ = ::ynn::quantized_attention_f32_work_items(params);
        tasks_ = std::min(ynn::thread_count(), work_items_);
        task_scratch_bytes_ = ::ynn::quantized_attention_f32_scratch_size(params);
        scratch_.resize(tasks_ * task_scratch_bytes_);
    }
    auto run(TensorInputs inputs) -> Tensor override {
        auto output = pool_.acquire(inputs[0].shape(), DType::F32, tensor::Device::cpu());
        attend(inputs, output);
        return output;
    }
    auto run_(TensorInputs, Tensor&) -> Tensor override {
        throw ops::Failure({ErrorCode::UNSUPPORTED, "attention is not an in-place operation"});
    }
    auto run_into(TensorInputs inputs, std::span<Tensor> outputs) -> void override { attend(inputs, outputs[0]); }
    auto allocations() const -> AllocationStats override { return pool_.allocations(); }

private:
    auto attend(TensorInputs inputs, Tensor& output) -> void {
        const auto& query = inputs[0];
        const auto params = parameters();
        const auto query_data = require(query.data<float>());
        const auto key_data = require(inputs[1].data<std::int8_t>());
        const auto value_data = require(inputs[2].data<std::int8_t>());
        const auto mask_data = require(inputs[3].data<float>());
        auto output_data = require(output.data<float>());
        const auto rows_per_task = (work_items_ + tasks_ - 1) / tasks_;
        require(ynn::parallel_for(tasks_, [&](std::size_t task) {
            const auto row_start = task * rows_per_task;
            const auto row_end = std::min(row_start + rows_per_task, work_items_);
            ::ynn::quantized_attention_f32(params, query_data.data(), key_data.data(), value_data.data(),
                                           mask_data.data(), output_data.data(), row_start, row_end,
                                           scratch_.data() + task * task_scratch_bytes_);
        }));
    }
    auto parameters() const -> ::ynn::quantized_attention_f32_params {
        return {
            .batch_size = batch_size_,
            .query_length = query_length_,
            .query_heads = heads_,
            .key_value_heads = key_heads_,
            .key_start = key_start_,
            .key_length = key_length_,
            .head_dim = head_dim_,
            .key_batch_stride = key_batch_stride_,
            .mask_batch_stride = mask_batch_stride_,
            .mask_head_stride = mask_head_stride_,
            .mask_query_stride = mask_query_stride_,
            .mask_key_stride = mask_key_stride_,
            .scale = scale_,
            .key_quantization = {key_scales_.data(), key_zero_points_.data(), key_block_size_, key_scales_.size()},
            .value_quantization = {value_scales_.data(), value_zero_points_.data(), value_block_size_,
                                   value_scales_.size()},
        };
    }

    std::size_t heads_, key_heads_, key_start_, key_length_;
    std::size_t mask_batch_stride_, mask_head_stride_, mask_query_stride_, mask_key_stride_;
    std::size_t batch_size_, query_length_, head_dim_, key_batch_stride_;
    float scale_;
    std::vector<float> key_scales_, value_scales_;
    std::vector<std::int32_t> key_zero_points_, value_zero_points_;
    std::size_t key_block_size_, value_block_size_;
    std::size_t work_items_, tasks_, task_scratch_bytes_;
    std::vector<std::byte> scratch_;
    OutputPool pool_;
};
/// Attention for a few FP32 query rows, as in an autoregressive decoding step. The generic graph copies every key and
/// value into head-major order on each call, which dominates a one-token step over a long encoder memory.
class FewQueryAttention final : public Operator {
public:
    static constexpr std::int64_t MAXIMUM_QUERIES = 8;
    static auto supports(const OperatorSpec& spec, TensorInputs inputs) -> bool {
        if (spec.attributes.size() != 1 || inputs[0].size(1) > MAXIMUM_QUERIES) return false;
        for (std::size_t index = 0; index < inputs.size(); ++index)
            if (inputs[index].dtype() != DType::F32) return false;
        return true;
    }
    FewQueryAttention(const OperatorSpec& spec, tensor::Arena& arena)
        : heads_(static_cast<std::size_t>(spec.attributes[0])), pool_(&arena) {}
    auto run(TensorInputs inputs) -> Tensor override {
        auto output = pool_.acquire(inputs[0].shape(), DType::F32, tensor::Device::cpu());
        attend(inputs, output);
        return output;
    }
    auto run_(TensorInputs, Tensor&) -> Tensor override {
        throw ops::Failure({ErrorCode::UNSUPPORTED, "attention is not an in-place operation"});
    }
    auto run_into(TensorInputs inputs, std::span<Tensor> outputs) -> void override { attend(inputs, outputs[0]); }
    auto allocations() const -> AllocationStats override { return pool_.allocations(); }

private:
    static auto dot(const float* left, const float* right, std::size_t count) -> float {
        // Independent lanes let the compiler use vector multiply-adds.
        std::array<float, 8> lanes{};
        std::size_t index = 0;
        for (; index + lanes.size() <= count; index += lanes.size())
            for (std::size_t lane = 0; lane < lanes.size(); ++lane)
                lanes[lane] += left[index + lane] * right[index + lane];
        float sum = 0;
        for (const auto lane : lanes) sum += lane;
        for (; index < count; ++index) sum += left[index] * right[index];
        return sum;
    }
    auto attend(TensorInputs inputs, Tensor& output) -> void {
        const auto& query = inputs[0];
        const auto batch = static_cast<std::size_t>(query.size(0)), rows = static_cast<std::size_t>(query.size(1));
        const auto width = static_cast<std::size_t>(query.size(2)), head_width = width / heads_;
        const auto keys = static_cast<std::size_t>(inputs[1].size(1));
        const auto key_batch_stride = inputs[1].size(0) == 1 ? 0 : keys * width;
        std::array<std::size_t, 4> mask_strides{};
        const float* mask = nullptr;
        if (inputs.size() == 4) {
            mask = require(inputs[3].data<float>()).data();
            std::size_t stride = 1;
            for (std::size_t axis = inputs[3].dimensions(); axis-- > 0;) {
                mask_strides[4 - inputs[3].dimensions() + axis] = inputs[3].size(axis) == 1 ? 0 : stride;
                stride *= inputs[3].size(axis);
            }
        }
        const auto* query_data = require(query.data<float>()).data();
        const auto* key_data = require(inputs[1].data<float>()).data();
        const auto* value_data = require(inputs[2].data<float>()).data();
        auto* output_data = require(output.data<float>()).data();
        const auto scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(head_width)));
        const auto items = batch * heads_ * rows;
        if (!items) return;
        const auto tasks = std::min(ynn::thread_count(), items);
        scratch_.resize(tasks * keys);
        const auto per_task = (items + tasks - 1) / tasks;
        require(ynn::parallel_for(tasks, [&](std::size_t task) {
            auto* scores = scratch_.data() + task * keys;
            for (auto item = task * per_task; item < std::min(items, (task + 1) * per_task); ++item) {
                const auto row = item % rows, head = item / rows % heads_, sample = item / rows / heads_;
                const auto* q = query_data + (sample * rows + row) * width + head * head_width;
                const auto* k = key_data + sample * key_batch_stride + head * head_width;
                const auto* v = value_data + sample * key_batch_stride + head * head_width;
                const auto* bias = mask ? mask + sample * mask_strides[0] + head * mask_strides[1] +
                                              row * mask_strides[2]
                                        : nullptr;
                auto largest = -std::numeric_limits<float>::infinity();
                for (std::size_t key = 0; key < keys; ++key) {
                    scores[key] = dot(q, k + key * width, head_width) * scale;
                    if (bias) scores[key] += bias[key * mask_strides[3]];
                    largest = std::max(largest, scores[key]);
                }
                float total = 0;
                for (std::size_t key = 0; key < keys; ++key) total += scores[key] = std::exp(scores[key] - largest);
                auto* out = output_data + (sample * rows + row) * width + head * head_width;
                std::fill_n(out, head_width, 0.F);
                for (std::size_t key = 0; key < keys; ++key) {
                    const auto weight = scores[key] / total;
                    const auto* source = v + key * width;
                    for (std::size_t index = 0; index < head_width; ++index) out[index] += weight * source[index];
                }
            }
        }));
    }

    std::size_t heads_;
    std::vector<float> scratch_;
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
    auto run_into(TensorInputs inputs, std::span<Tensor> outputs) -> void override {
        for (std::size_t index = 0; index < dynamic_count_; ++index) require(executable_.bind(index, inputs[index]));
        require(executable_.bind(count_, outputs.back()));
        if (outputs.size() == 2) require(executable_.bind(count_ + 1, outputs.front()));
        require(executable_.invoke());
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
    auto release_cached_buffers() -> void override { arena_ = tensor::Arena{}; }
    auto supports_replay() const noexcept -> bool override { return true; }
    auto copy_slice_(Tensor& destination, const Tensor& source, std::size_t outer, std::size_t source_bytes,
                     std::size_t destination_bytes, std::size_t offset_bytes) -> void override {
        auto target = require(destination.host_bytes());
        auto input = require(source.host_bytes());
        std::vector<std::byte> snapshot;
        const auto from = reinterpret_cast<std::uintptr_t>(input.data());
        const auto to = reinterpret_cast<std::uintptr_t>(target.data());
        if (from < to + target.size() && to < from + input.size()) {
            snapshot.assign(input.begin(), input.end());
            input = snapshot;
        }
        for (std::size_t block = 0; block < outer; ++block)
            std::memcpy(target.data() + block * destination_bytes + offset_bytes, input.data() + block * source_bytes,
                        source_bytes);
    }
    auto prepare(const OperatorSpec& spec, TensorInputs inputs) -> std::unique_ptr<Operator> override {
        if (spec.operation == Operation::SCATTER) return std::make_unique<Scatter>(arena_);
        if (spec.operation == Operation::GREEDY_TOKEN) return std::make_unique<GreedyToken>(arena_);
        if (spec.operation == Operation::EMBEDDING) return std::make_unique<Embedding>(spec, inputs, arena_);
        if (spec.operation == Operation::ROTARY && Rotary::supports(inputs)) return std::make_unique<Rotary>(arena_);
        if ((spec.operation == Operation::RMS_NORM || spec.operation == Operation::RMS_NORM_RESIDUAL) &&
            RmsNorm::supports(spec, inputs))
            return std::make_unique<RmsNorm>(spec, arena_);
        if (spec.operation == Operation::CAST && spec.dtype == DType::I8 && spec.epsilon > 0)
            return std::make_unique<QuantizeInt8>(spec.epsilon, arena_);
        if (spec.operation == Operation::ATTENTION) {
            if (FewQueryAttention::supports(spec, inputs)) return std::make_unique<FewQueryAttention>(spec, arena_);
            const bool byte_cache = inputs[1].dtype() == DType::I8 && !spec.quantization[0].scales.empty() &&
                                    !spec.quantization[1].scales.empty();
            if (byte_cache && inputs[0].dtype() == DType::F32 && inputs[2].dtype() == inputs[1].dtype())
                return std::make_unique<QuantizedAttention>(spec, inputs, arena_);
        }
        const auto flags = inputs[0].dtype() == DType::BF16 ? YNN_FLAG_NO_EXCESS_PRECISION : 0;
        const bool paired = spec.operation == Operation::RESIDUAL_NORM;
        auto graph = require(ynn::Graph::create(inputs.size() + (paired ? 2 : 1), flags));
        auto native = graph.get();
        const auto constant =
            !spec.dynamic_parameters &&
            (spec.operation == Operation::LINEAR || spec.operation == Operation::QUANTIZED_LINEAR ||
             spec.operation == Operation::LAYER_NORM || spec.operation == Operation::RMS_NORM ||
             spec.operation == Operation::PACKED_LINEAR || spec.operation == Operation::GATED_FEED_FORWARD || paired);
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
            const bool packed = (spec.operation == Operation::PACKED_LINEAR && index == 1) ||
                                (spec.operation == Operation::GATED_FEED_FORWARD && (index == 1 || index == 3));
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
                if (packed || (!spec.attributes.empty() && spec.attributes[0])) {
                    const std::array<std::int32_t, 2> axes{1, 0};
                    weight = YNN_INVALID_VALUE_ID;
                    check(ynn_define_static_transpose(native, 2, axes.data(), operands[1], &weight, 0));
                }
                if (packed && spec.epsilon > 0 && spec.attributes[1] == inputs[0].size(-1) &&
                    (SHARED_DECODE_PACKING || inputs[0].numel() > inputs[0].size(-1))) {
                    const std::size_t columns = padded_columns;
                    auto channel_scale = YNN_INVALID_VALUE_ID;
                    check(ynn_define_static_reshape(native, 1, &columns, operands[2], &channel_scale, 0));
                    check(ynn_define_dot(native, 1, quantized, weight, YNN_INVALID_VALUE_ID, &result, 0));
                    result = binary(ynn_binary_multiply, binary(ynn_binary_multiply, result, channel_scale), scale);
                } else
                    check(::ynn::define_blockwise_dot(native, quantized, zero, scale, weight, YNN_INVALID_VALUE_ID,
                                                      operands[2], packed ? spec.attributes[1] : inputs[0].size(-1),
                                                      packed ? YNN_INVALID_VALUE_ID : operands[3], ynn_type_fp32,
                                                      result, 0));
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
            case Operation::GATED_FEED_FORWARD: {
                const auto bits = static_cast<std::int32_t>(spec.attributes[0]);
                const auto input_size = static_cast<std::size_t>(spec.attributes[1]);
                const auto intermediate = static_cast<std::size_t>(spec.attributes[2]);
                const auto gate_output_scale = std::bit_cast<float>(static_cast<std::int32_t>(spec.attributes[3]));
                const auto down_input_scale = std::bit_cast<float>(static_cast<std::int32_t>(spec.attributes[4]));
                const auto down_output_scale = std::bit_cast<float>(static_cast<std::int32_t>(spec.attributes[5]));
                constexpr std::size_t TILE_SIZE = 2048;
                const auto tiles = (intermediate + TILE_SIZE - 1) / TILE_SIZE;
                std::vector<std::uint32_t> gate_weight_ids(tiles), gate_scale_ids(tiles), up_weight_ids(tiles),
                    up_scale_ids(tiles), down_weight_ids(tiles);
                const auto define_weight = [&](const Tensor& weight, std::size_t row_start, std::size_t row_count,
                                               std::size_t column_start, std::size_t column_count) {
                    const auto bytes =
                        transpose_packed_tile(weight, bits, row_start, row_count, column_start, column_count);
                    const std::array<std::size_t, 2> shape{column_count, row_count};
                    auto id = YNN_INVALID_VALUE_ID;
                    const auto dtype = bits == 2 ? ynn_type_int2 : bits == 4 ? ynn_type_int4 : ynn_type_int8;
                    check(ynn_define_tensor(native, dtype, shape.size(), shape.data(), bytes.data(),
                                            YNN_VALUE_FLAG_COPY_DATA, &id));
                    return id;
                };
                const auto& scales_tensor = inputs[2];
                const auto scales = require(scales_tensor.data<float>());
                const auto define_scales = [&](std::size_t start, std::size_t length) {
                    const std::array<std::size_t, 2> shape{length, 1};
                    auto id = YNN_INVALID_VALUE_ID;
                    check(ynn_define_tensor(native, ynn_type_fp32, shape.size(), shape.data(), scales.data() + start,
                                            YNN_VALUE_FLAG_COPY_DATA, &id));
                    return id;
                };
                for (std::size_t tile = 0; tile < tiles; ++tile) {
                    const auto begin = tile * TILE_SIZE;
                    const auto length = std::min(TILE_SIZE, intermediate - begin);
                    gate_weight_ids[tile] = define_weight(inputs[1], begin, length, 0, input_size);
                    gate_scale_ids[tile] = define_scales(begin, length);
                    up_weight_ids[tile] = define_weight(inputs[1], intermediate + begin, length, 0, input_size);
                    up_scale_ids[tile] = define_scales(intermediate + begin, length);
                    down_weight_ids[tile] = define_weight(inputs[3], 0, input_size, begin, length);
                }
                check(::ynn::define_packed_feed_forward(
                    native, operands[0], scalar(spec.epsilon), gate_weight_ids.data(), gate_scale_ids.data(),
                    up_weight_ids.data(), up_scale_ids.data(), scalar(gate_output_scale), down_weight_ids.data(),
                    operands[4], scalar(down_input_scale), scalar(down_output_scale), input_size, intermediate,
                    TILE_SIZE, true, result));
                break;
            }
            case Operation::GELU_MULTIPLY:
            case Operation::GELU: {
                if (spec.operation == Operation::GELU_MULTIPLY || spec.attributes[0]) {
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
                    if (spec.operation == Operation::GELU_MULTIPLY)
                        result = binary(ynn_binary_multiply, result, operands[1]);
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
                auto indices = operands[1];
                if (inputs[1].dimensions() == 1) {
                    const auto normalized = axis < 0 ? axis + static_cast<std::int32_t>(inputs[0].dimensions()) : axis;
                    if (normalized < 0 || static_cast<std::size_t>(normalized) >= inputs[0].dimensions())
                        throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "gather axis outside tensor rank"});
                    std::vector<std::size_t> index_shape(inputs[0].dimensions(), 1);
                    index_shape[normalized] = inputs[1].numel();
                    indices = YNN_INVALID_VALUE_ID;
                    check(ynn_define_static_reshape(native, index_shape.size(), index_shape.data(), operands[1],
                                                    &indices, 0));
                }
                check(ynn_define_gather(native, 1, &axis, inputs[0].dimensions() + inputs[1].dimensions() - 1,
                                        operands[0], indices, &result, 0));
                break;
            }
            case Operation::CONCAT:
                check(ynn_define_concatenate(native, spec.attributes[0], operands.size(), operands.data(), &result, 0));
                break;
            case Operation::SCATTER:
            case Operation::GREEDY_TOKEN:
            case Operation::RMS_ROTARY:
            case Operation::EMBEDDING:
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
#if defined(__EMSCRIPTEN__)
        const bool small_packed_projection = spec.operation == Operation::PACKED_LINEAR &&
                                             inputs[0].numel() == inputs[0].size(-1) &&
                                             inputs[1].nbytes() <= 512 * 1024;
        if (small_packed_projection) operator_threads = 1;
#endif
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