#include "kidi/runtime/operator.h"
#include "kidi/runtime/parity_math.h"
#include "kidi/runtime/ynn/feed_forward_cache.h"
#include "kidi/runtime/ynn/graph.h"
#include "kidi/ops/context.h"
#include "ynnpack/composites/composites.h"
#include "ynnpack/kernels/unary/unary.h"
#include <array>
#include <compare>
#include <functional>
#include <bit>
#include <cmath>
#include <cstring>
#include <map>
#include <numeric>
#include <algorithm>
#include <limits>
#include <cstdlib>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#if defined(__APPLE__)
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#include <malloc/malloc.h>
#endif

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
        case DType::E4M3:
            return ynn_type_fp8_e4m3;
        case DType::E5M2:
            return ynn_type_fp8_e5m2;
        default:
            throw ops::Failure({ErrorCode::UNSUPPORTED, "unsupported eager CPU dtype"});
    }
}
auto check(ynn_status status) -> void { require(ynn::check_status(status, "eager CPU operator")); }
#if defined(__EMSCRIPTEN__) || defined(__x86_64__)
// Single-row calibrated projections use the same INT8 dot as multi-row ones, so YNNPACK's constant cache shares one
// packed copy of each weight between prefill and decode. WebAssembly decode measured no slower this way, and x86
// decode measured faster.
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
auto hash_tensor(const Tensor& tensor, std::uint64_t hash = 0x9E3779B97F4A7C15ULL) -> std::uint64_t {
    const auto bytes = require(tensor.host_bytes());
    hash ^= bytes.size() + 0x9E3779B97F4A7C15ULL + (hash << 6) + (hash >> 2);
    std::size_t offset = 0;
    while (offset + sizeof(std::uint64_t) <= bytes.size()) {
        std::uint64_t word = 0;
        std::memcpy(&word, bytes.data() + offset, sizeof(word));
        hash ^= word + 0x9E3779B97F4A7C15ULL + (hash << 6) + (hash >> 2);
        offset += sizeof(word);
    }
    if (offset < bytes.size()) {
        std::uint64_t word = 0;
        std::memcpy(&word, bytes.data() + offset, bytes.size() - offset);
        hash ^= word + 0x9E3779B97F4A7C15ULL + (hash << 6) + (hash >> 2);
    }
    return hash;
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
#if defined(__APPLE__)
        for (std::size_t row = 0; row < rows; ++row) {
            const float* row_values = values.data() + row * width;
            float best = -std::numeric_limits<float>::infinity();
            vDSP_Length best_index = 0;
            vDSP_maxvi(row_values, 1, &best, &best_index, width);
            bool invalid = std::isnan(best) || best == std::numeric_limits<float>::infinity();
            for (std::size_t column = 0; !invalid && column < width; ++column) invalid = std::isnan(row_values[column]);
            tokens[row] =
                invalid || best == -std::numeric_limits<float>::infinity() ? -1 : static_cast<std::int32_t>(best_index);
        }
#else
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
#endif
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
auto for_rows(std::size_t rows, std::size_t row_work, const std::function<void(std::size_t, std::size_t)>& body)
    -> void {
    const auto tasks = rows * row_work < 32768 ? std::size_t{1} : std::min(ynn::thread_count(), rows);
    if (tasks <= 1) return body(0, rows);
    const auto block = (rows + tasks - 1) / tasks;
    require(ynn::parallel_for(tasks, [&](std::size_t task) {
        const auto begin = task * block;
        if (begin < rows) body(begin, std::min(rows, begin + block));
    }));
}

class SiluGate final : public Operator {
public:
    SiluGate(const Tensor& input, tensor::Arena& arena)
        : rows_(input.numel() / input.size(-1)),
          width_(input.size(-1) / 2),
          tasks_(rows_ * input.size(-1) < 32768 ? 1 : std::min(ynn::thread_count(), rows_)),
          scratch_(tasks_ * width_),
          pool_(&arena) {
        shape_.assign(input.shape().begin(), input.shape().end());
        shape_.back() = static_cast<std::int64_t>(width_);
    }

    static auto supports(TensorInputs inputs) -> bool {
        return inputs.size() == 1 && inputs[0].dtype() == DType::F32 && inputs[0].dimensions() > 0 &&
               inputs[0].is_contiguous() && inputs[0].size(-1) > 0 && inputs[0].size(-1) % 2 == 0;
    }

    auto run(TensorInputs inputs) -> Tensor override {
        auto output = pool_.acquire(shape_, DType::F32, tensor::Device::cpu());
        apply(inputs[0], output);
        return output;
    }

    auto run_(TensorInputs, Tensor&) -> Tensor override {
        throw ops::Failure({ErrorCode::UNSUPPORTED, "SiLU gate is not an in-place operation"});
    }

    auto run_into(TensorInputs inputs, std::span<Tensor> outputs) -> void override { apply(inputs[0], outputs[0]); }

    auto allocations() const -> AllocationStats override { return pool_.allocations(); }

private:
    auto apply(const Tensor& input, Tensor& output) -> void {
        const auto* source = require(input.data<float>()).data();
        auto* target = require(output.data<float>()).data();
        const auto rows_per_task = (rows_ + tasks_ - 1) / tasks_;
        require(ynn::parallel_for(tasks_, [&](std::size_t task) {
            const auto begin = task * rows_per_task, end = std::min(rows_, begin + rows_per_task);
#if defined(__APPLE__)
            auto* temporary = scratch_.data() + task * width_;
            const auto count = static_cast<int>(width_);
            const float one = 1.F;
#else
            static const auto sigmoid =
                ::ynn::get_unary_kernel(ynn_unary_sigmoid, ynn_type_fp32, ynn_type_fp32);
            static const auto sigmoid_params = ::ynn::get_unary_params(ynn_unary_sigmoid);
            if (!sigmoid) throw ops::Failure({ErrorCode::UNSUPPORTED, "FP32 sigmoid kernel is unavailable"});
#endif
            for (auto row = begin; row < end; ++row) {
                const auto* gate = source + row * 2 * width_;
                const auto* up = gate + width_;
                auto* hidden = target + row * width_;
#if defined(__APPLE__)
                vDSP_vneg(gate, 1, temporary, 1, width_);
                vvexpf(hidden, temporary, &count);
                vDSP_vsadd(hidden, 1, &one, hidden, 1, width_);
                vDSP_vdiv(hidden, 1, gate, 1, temporary, 1, width_);
                vDSP_vmul(temporary, 1, up, 1, hidden, 1, width_);
#else
                sigmoid(1, width_, width_ * sizeof(float), gate, width_ * sizeof(float), hidden, &sigmoid_params);
                std::size_t channel = 0;
#if defined(__ARM_NEON)
                for (; channel + 8 <= width_; channel += 8) {
                    const auto first = vmulq_f32(vld1q_f32(hidden + channel), vld1q_f32(gate + channel));
                    const auto second = vmulq_f32(vld1q_f32(hidden + channel + 4), vld1q_f32(gate + channel + 4));
                    vst1q_f32(hidden + channel, vmulq_f32(first, vld1q_f32(up + channel)));
                    vst1q_f32(hidden + channel + 4, vmulq_f32(second, vld1q_f32(up + channel + 4)));
                }
#endif
                for (; channel < width_; ++channel) hidden[channel] *= gate[channel] * up[channel];
#endif
            }
        }));
    }

    std::size_t rows_, width_, tasks_;
    std::vector<float> scratch_;
    std::vector<std::int64_t> shape_;
    OutputPool pool_;
};

class GeluGate final : public Operator {
public:
    GeluGate(const Tensor& input, tensor::Arena& arena)
        : rows_(input.numel() / input.size(-1)),
          width_(input.size(-1) / 2),
          tasks_(rows_ * input.size(-1) < 32768 ? 1 : std::min(ynn::thread_count(), rows_)),
          pool_(&arena) {
        shape_.assign(input.shape().begin(), input.shape().end());
        shape_.back() = static_cast<std::int64_t>(width_);
    }

    static auto supports(TensorInputs inputs) -> bool {
        return inputs.size() == 1 && inputs[0].dtype() == DType::F32 && inputs[0].dimensions() > 0 &&
               inputs[0].is_contiguous() && inputs[0].size(-1) > 0 && inputs[0].size(-1) % 2 == 0;
    }

    auto run(TensorInputs inputs) -> Tensor override {
        auto output = pool_.acquire(shape_, DType::F32, tensor::Device::cpu());
        apply(inputs[0], output);
        return output;
    }

    auto run_(TensorInputs, Tensor&) -> Tensor override {
        throw ops::Failure({ErrorCode::UNSUPPORTED, "GELU gate is not an in-place operation"});
    }

    auto run_into(TensorInputs inputs, std::span<Tensor> outputs) -> void override { apply(inputs[0], outputs[0]); }

    auto allocations() const -> AllocationStats override { return pool_.allocations(); }

private:
    auto apply(const Tensor& input, Tensor& output) -> void {
        const auto* source = require(input.data<float>()).data();
        auto* target = require(output.data<float>()).data();
        static const auto tanh_kernel = ::ynn::get_unary_kernel(ynn_unary_tanh, ynn_type_fp32, ynn_type_fp32);
        static const auto tanh_params = ::ynn::get_unary_params(ynn_unary_tanh);
        if (!tanh_kernel) throw ops::Failure({ErrorCode::UNSUPPORTED, "FP32 tanh kernel is unavailable"});
        const auto rows_per_task = (rows_ + tasks_ - 1) / tasks_;
        const auto run_task = [&](std::size_t task) {
            const auto begin = task * rows_per_task, end = std::min(rows_, begin + rows_per_task);
            constexpr float SCALE = 0.79788456080286535588F; // sqrt(2 / pi)
            for (auto row = begin; row < end; ++row) {
                const auto* gate = source + row * 2 * width_;
                const auto* up = gate + width_;
                auto* hidden = target + row * width_;
                constexpr std::size_t CHUNK = 1024;
                for (std::size_t offset = 0; offset < width_; offset += CHUNK) {
                    const auto count = std::min(CHUNK, width_ - offset);
                    {
#pragma clang fp contract(off)
                        for (std::size_t i = 0; i < count; ++i) {
                            const auto g = gate[offset + i];
                            hidden[offset + i] = (g + (g * g * g) * 0.044715F) * SCALE;
                        }
                    }
                    tanh_kernel(1, count, count * sizeof(float), hidden + offset, count * sizeof(float),
                                hidden + offset, &tanh_params);
                    {
#pragma clang fp contract(off)
                        for (std::size_t i = 0; i < count; ++i) {
                            hidden[offset + i] =
                                ((gate[offset + i] * 0.5F) * (1.F + hidden[offset + i])) * up[offset + i];
                        }
                    }
                }
            }
        };
        if (tasks_ <= 1) {
            run_task(0);
            return;
        }
        require(ynn::parallel_for(tasks_, run_task));
    }

    std::size_t rows_, width_, tasks_;
    std::vector<std::int64_t> shape_;
    OutputPool pool_;
};

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

#if defined(__APPLE__)
// PyTorch-compatible attention reduction order; see third_party/PYTORCH_LICENSE.txt.
class PytorchParityAttention final : public Operator {
public:
    static auto supports(const OperatorSpec& spec, TensorInputs inputs) -> bool {
        if (spec.operation != Operation::ATTENTION || spec.attributes.size() != 5 || !spec.attributes[4] ||
            inputs.size() != 4 || spec.attributes[0] != spec.attributes[1] || spec.attributes[2] != 0 ||
            inputs[0].dimensions() != 3 || inputs[1].dimensions() != 3 || inputs[2].dimensions() != 3 ||
            inputs[3].dtype() != DType::F32 || inputs[3].numel() != inputs[3].size(-1) ||
            static_cast<std::size_t>(spec.attributes[3]) != inputs[3].size(-1))
            return false;
        for (std::size_t index = 0; index < 3; ++index)
            if (inputs[index].dtype() != DType::F32 || !inputs[index].is_contiguous()) return false;
        return inputs[0].size(0) == inputs[1].size(0) && inputs[1].size(0) == inputs[2].size(0) &&
               inputs[1].size(1) == inputs[2].size(1) && inputs[0].size(2) == inputs[1].size(2) &&
               inputs[1].size(2) == inputs[2].size(2);
    }
    PytorchParityAttention(const OperatorSpec& spec, tensor::Arena& arena)
        : heads_(static_cast<std::size_t>(spec.attributes[0])),
          key_length_(static_cast<std::size_t>(spec.attributes[3])),
          scale_(spec.epsilon),
          pool_(&arena) {}
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
        const auto batch_size = inputs[0].size(0), query_length = inputs[0].size(1);
        const auto hidden = inputs[0].size(2), head_width = hidden / heads_;
        const auto* query = require(inputs[0].data<float>()).data();
        const auto* key = require(inputs[1].data<float>()).data();
        const auto* value = require(inputs[2].data<float>()).data();
        const auto* mask = require(inputs[3].data<float>()).data();
        auto* target = require(output.data<float>()).data();
        std::vector<float> scores(query_length * key_length_);
        for (std::size_t batch = 0; batch < batch_size; ++batch) {
            const auto query_batch = query + batch * query_length * hidden;
            const auto key_batch = key + batch * inputs[1].size(1) * hidden;
            const auto value_batch = value + batch * inputs[2].size(1) * hidden;
            auto output_batch = target + batch * query_length * hidden;
            for (std::size_t head = 0; head < heads_; ++head) {
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, static_cast<int>(query_length),
                            static_cast<int>(key_length_), static_cast<int>(head_width), scale_,
                            query_batch + head * head_width, static_cast<int>(hidden), key_batch + head * head_width,
                            static_cast<int>(hidden), 0.F, scores.data(), static_cast<int>(key_length_));
                for_rows(query_length, key_length_, [&](std::size_t begin, std::size_t end) {
                    for (std::size_t row = begin; row < end; ++row)
                        parity::softmax(std::span(scores).subspan(row * key_length_, key_length_),
                                        std::span(mask, key_length_));
                });
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, static_cast<int>(query_length),
                            static_cast<int>(head_width), static_cast<int>(key_length_), 1.F, scores.data(),
                            static_cast<int>(key_length_), value_batch + head * head_width, static_cast<int>(hidden),
                            0.F, output_batch + head * head_width, static_cast<int>(hidden));
            }
        }
    }

    std::size_t heads_, key_length_;
    float scale_;
    OutputPool pool_;
};
#endif

/// RMS normalization over the last axis with a per-channel weight; the residual form adds `inputs[2]` and optionally
/// scales by the scalar `inputs[3]`.
class RmsNorm final : public Operator {
public:
    RmsNorm(const OperatorSpec& spec, tensor::Arena& arena)
        : epsilon_(spec.epsilon),
          residual_(spec.operation == Operation::RMS_NORM_RESIDUAL),
          pytorch_parity_(!spec.attributes.empty() && spec.attributes[0]),
          pool_(&arena) {}
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
                std::size_t index = 0;
                auto sum = 0.F;
                if (pytorch_parity_) {
                    sum = parity::sum_squares(std::span(x, width));
                } else {
                    std::array<float, 8> partial{};
                    for (; index + 8 <= width; index += 8)
                        for (std::size_t lane = 0; lane < 8; ++lane) partial[lane] += x[index + lane] * x[index + lane];
                    for (; index < width; ++index) sum += x[index] * x[index];
                    for (const auto value : partial) sum += value;
                }
                const auto mean = pytorch_parity_ ? sum / static_cast<float>(width) : sum * reciprocal;
                const auto inverse = 1.F / std::sqrt(mean + epsilon_);
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
    bool pytorch_parity_;
    OutputPool pool_;
};

class RmsRotary final : public Operator {
public:
    RmsRotary(const OperatorSpec& spec, tensor::Arena& arena)
        : epsilon_(spec.epsilon),
          sliced_(!spec.attributes.empty()),
          heads_(sliced_ ? static_cast<std::size_t>(spec.attributes[0]) : 0),
          width_(sliced_ ? static_cast<std::size_t>(spec.attributes[1]) : 0),
          channel_offset_(sliced_ ? static_cast<std::size_t>(spec.attributes[2]) : 0),
          sequence_offset_(sliced_ ? static_cast<std::size_t>(spec.attributes[3]) : 0),
          sequence_length_(sliced_ ? static_cast<std::size_t>(spec.attributes[4]) : 0),
          pool_(&arena) {}

    static auto supports(const OperatorSpec& spec, TensorInputs inputs) -> bool {
        if (inputs.size() != 4) return false;
        const auto &input = inputs[0], &weight = inputs[1], &cosine = inputs[2], &sine = inputs[3];
        if (!spec.attributes.empty())
            return spec.attributes.size() == 5 && input.dtype() == DType::F32 && input.dimensions() == 3 &&
                   input.is_contiguous() && weight.dtype() == DType::F32 && weight.is_contiguous() &&
                   cosine.dtype() == DType::F32 && cosine.is_contiguous() && sine.dtype() == DType::F32 &&
                   sine.is_contiguous() && std::ranges::equal(cosine.shape(), sine.shape());
        return input.dtype() == DType::F32 && weight.dtype() == DType::F32 && cosine.dtype() == DType::F32 &&
               sine.dtype() == DType::F32 && input.dimensions() == 4 && input.is_contiguous() &&
               weight.is_contiguous() && cosine.is_contiguous() && sine.is_contiguous() && input.size(3) % 2 == 0 &&
               weight.numel() == input.size(3) && cosine.size(3) == input.size(3) / 2 && cosine.size(2) == 1 &&
               cosine.size(1) == input.size(1) && std::ranges::equal(cosine.shape(), sine.shape()) &&
               (cosine.size(0) == input.size(0) || cosine.size(0) == 1);
    }

    auto run(TensorInputs inputs) -> Tensor override {
        const auto shape =
            sliced_ ? std::vector<std::int64_t>{static_cast<std::int64_t>(inputs[0].size(0)),
                                                static_cast<std::int64_t>(sequence_length_),
                                                static_cast<std::int64_t>(heads_), static_cast<std::int64_t>(width_)}
                    : std::vector<std::int64_t>{inputs[0].shape().begin(), inputs[0].shape().end()};
        auto output = pool_.acquire(shape, DType::F32, tensor::Device::cpu());
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
        const auto &input = inputs[0], &cosine = inputs[2];
        const auto rows = sliced_ ? sequence_length_ : input.size(1);
        const auto heads = sliced_ ? heads_ : input.size(2);
        const auto width = sliced_ ? width_ : input.size(3);
        const auto half = width / 2;
        const auto vectors = input.size(0) * rows * heads;
        const auto* source = require(input.data<float>()).data();
        const auto* weight = require(inputs[1].data<float>()).data();
        const auto* cosines = require(cosine.data<float>()).data();
        const auto* sines = require(inputs[3].data<float>()).data();
        auto* target = require(output.data<float>()).data();
        const auto shared_angles = cosine.size(0) == 1;
        const auto reciprocal = 1.F / static_cast<float>(width);
        for_rows(vectors, width, [&](std::size_t begin, std::size_t end) {
#pragma clang fp contract(off)
            for (auto vector = begin; vector < end; ++vector) {
                const auto sample = vector / (rows * heads);
                const auto position = vector / heads % rows;
                const auto head = vector % heads;
                const auto* x = sliced_
                                    ? source + (sample * input.size(1) + sequence_offset_ + position) * input.size(2) +
                                          channel_offset_ + head * width
                                    : source + vector * width;
                auto* y = target + vector * width;
                std::array<float, 8> partial{};
                std::size_t index = 0;
                for (; index + 8 <= width; index += 8)
                    for (std::size_t lane = 0; lane < 8; ++lane) partial[lane] += x[index + lane] * x[index + lane];
                auto sum = 0.F;
                for (; index < width; ++index) sum += x[index] * x[index];
                for (const auto value : partial) sum += value;
                const auto inverse = 1.F / std::sqrt(sum * reciprocal + epsilon_);
                const auto angle = (shared_angles ? position : sample * rows + position) * half;
                const auto* c = cosines + angle;
                const auto* s = sines + angle;
                for (index = 0; index < half; ++index) {
                    const auto first = x[index] * inverse * weight[index];
                    const auto second = x[index + half] * inverse * weight[index + half];
                    y[index] = first * c[index] - second * s[index];
                    y[index + half] = second * c[index] + first * s[index];
                }
            }
        });
    }

    float epsilon_;
    bool sliced_;
    std::size_t heads_, width_, channel_offset_, sequence_offset_, sequence_length_;
    OutputPool pool_;
};

class Embedding final : public Operator {
public:
    Embedding(const OperatorSpec& spec, TensorInputs inputs, tensor::Arena& arena)
        : width_(static_cast<std::size_t>(spec.attributes[0])),
          bits_(static_cast<int>(spec.attributes[1])),
          groups_(static_cast<std::size_t>(spec.attributes[2])),
          unsigned_packed_(spec.attributes[3] != 0),
          multiplier_(spec.epsilon),
          pool_(&arena) {
        const auto& weight = inputs[1];
        const bool packed = bits_ != 0;
        const auto stored = packed ? width_ / (8 / bits_) : width_;
        const auto valid_scale_dtype = [](DType dt) { return dt == DType::F32 || dt == DType::F16; };
        if (weight.size(1) != stored || (packed && width_ % (8 / bits_)) ||
            (packed ? (weight.dtype() != DType::U8 && !(bits_ == 8 && weight.dtype() == DType::I8)) ||
                          (inputs.size() != 3 && inputs.size() != 4) || !valid_scale_dtype(inputs[2].dtype()) ||
                          inputs[2].dimensions() != 2 || inputs[2].size(0) != weight.size(0) ||
                          inputs[2].size(1) != groups_ || !groups_ || width_ % groups_ ||
                          (inputs.size() == 4 && (!valid_scale_dtype(inputs[3].dtype()) || inputs[3].dimensions() != 2 ||
                                                  inputs[3].size(0) != weight.size(0) || inputs[3].size(1) != groups_))
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
            const auto s_bytes = require(inputs[2].host_bytes());
            const auto z_bytes = inputs.size() == 4 ? require(inputs[3].host_bytes()) : std::span<const std::byte>{};
            const bool s_f16 = inputs[2].dtype() == DType::F16;
            const bool z_f16 = inputs.size() == 4 && inputs[3].dtype() == DType::F16;
            const auto* s_f32 = s_f16 ? nullptr : reinterpret_cast<const float*>(s_bytes.data());
            const auto* z_f32 = (inputs.size() == 4 && !z_f16) ? reinterpret_cast<const float*>(z_bytes.data()) : nullptr;
            const auto read_val = [](const std::byte* base, bool is_f16, const float* f32_ptr,
                                     std::size_t idx) -> float {
                if (!is_f16) return f32_ptr[idx];
#if defined(__aarch64__) || defined(__arm64__)
                return static_cast<float>(reinterpret_cast<const __fp16*>(base)[idx]);
#else
                std::uint16_t h = reinterpret_cast<const std::uint16_t*>(base)[idx];
                std::uint32_t sign = (h & 0x8000U) << 16;
                std::uint32_t exp = (h >> 10) & 0x1FU;
                std::uint32_t mant = h & 0x03FFU;
                std::uint32_t bits = exp == 0 ? sign : (exp == 31 ? (sign | 0x7F800000U | (mant << 13))
                                                                  : (sign | ((exp + 112) << 23) | (mant << 13)));
                return std::bit_cast<float>(bits);
#endif
            };
            const bool has_zeros = inputs.size() == 4;
            const auto group_width = width_ / groups_, per_byte = static_cast<std::size_t>(8 / bits_);
            const auto data = reinterpret_cast<const std::uint8_t*>(bytes.data());
            const auto sign = 1 << (bits_ - 1);
            const std::uint8_t sign_mask = unsigned_packed_ ? (bits_ == 2 ? 0xaaU : bits_ == 4 ? 0x88U : 0U) : 0U;
            for (std::size_t row = 0; row < tokens.size(); ++row)
                for (std::size_t channel = 0; channel < width_; ++channel) {
                    const auto offset = static_cast<std::size_t>(tokens[row]) * width_ + channel;
                    const auto byte = static_cast<std::uint8_t>(data[offset / per_byte] ^ sign_mask);
                    const auto raw = (byte >> ((offset % per_byte) * bits_)) & ((1 << bits_) - 1);
                    const auto integer = (raw ^ sign) - sign;
                    const auto group_index = static_cast<std::size_t>(tokens[row]) * groups_ + channel / group_width;
                    const auto sc = read_val(s_bytes.data(), s_f16, s_f32, group_index);
                    const auto zr = has_zeros ? read_val(z_bytes.data(), z_f16, z_f32, group_index) : 0.F;
                    const auto dequantized = integer * sc + zr;
                    values[row * width_ + channel] = dequantized * multiplier_;
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
    bool unsigned_packed_ = false;
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
        if (spec.attributes.size() == 1) {
            if (inputs[0].size(1) > MAXIMUM_QUERIES) return false;
        } else if (spec.attributes.size() == 4) {
            if (inputs[0].size(1) != 1) return false;
        } else {
            return false;
        }
        if (!spec.quantization[0].scales.empty() || !spec.quantization[1].scales.empty()) return false;
        const auto heads = static_cast<std::size_t>(spec.attributes[0]);
        const auto key_heads = spec.attributes.size() >= 2 ? static_cast<std::size_t>(spec.attributes[1]) : heads;
        if (!heads || !key_heads || heads % key_heads != 0) return false;
        for (std::size_t index = 0; index < inputs.size(); ++index)
            if (inputs[index].dtype() != DType::F32 || !inputs[index].is_contiguous()) return false;
        return true;
    }
    FewQueryAttention(const OperatorSpec& spec, tensor::Arena& arena)
        : heads_(static_cast<std::size_t>(spec.attributes[0])),
          key_heads_(spec.attributes.size() >= 2 ? static_cast<std::size_t>(spec.attributes[1]) : heads_),
          key_start_(spec.attributes.size() >= 4 ? static_cast<std::size_t>(spec.attributes[2]) : 0),
          key_length_(spec.attributes.size() >= 4 ? static_cast<std::size_t>(spec.attributes[3]) : 0),
          explicit_scale_(spec.attributes.size() >= 2),
          scale_(spec.epsilon),
          pool_(&arena) {}
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
        const auto kv_width = key_heads_ * head_width;
        const auto group_size = heads_ / key_heads_;
        const auto total_keys = static_cast<std::size_t>(inputs[1].size(1));
        const auto keys = key_length_ ? key_length_ : total_keys;
        const auto key_batch_stride = inputs[1].size(0) == 1 ? 0 : total_keys * kv_width;
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
        const auto scale =
            explicit_scale_ ? scale_ : static_cast<float>(1.0 / std::sqrt(static_cast<double>(head_width)));
        const auto items = batch * heads_ * rows;
        if (!items) return;
        if (batch == 1 && rows == 1 && key_heads_ == 1 && heads_ == 8 && mask_strides[1] == 0) {
            const auto* k_base = key_data + key_start_ * head_width;
            const auto* v_base = value_data + key_start_ * head_width;
            const auto* bias = mask;
            std::size_t active_keys = keys;
            if (bias) {
                while (active_keys > 0 && bias[(active_keys - 1) * mask_strides[3]] <= -1e8F) --active_keys;
            }
            if (!active_keys) {
                std::fill_n(output_data, width, 0.F);
                return;
            }
            scratch_.resize(8 * active_keys);
            std::array<float, 8> largest;
            largest.fill(-std::numeric_limits<float>::infinity());
            for (std::size_t key = 0; key < active_keys; ++key) {
                const auto b = bias ? bias[key * mask_strides[3]] : 0.F;
                if (b <= -1e8F) {
                    for (std::size_t h = 0; h < 8; ++h)
                        scratch_[h * active_keys + key] = -std::numeric_limits<float>::infinity();
                    continue;
                }
                const auto* k_row = k_base + key * head_width;
                for (std::size_t h = 0; h < 8; ++h) {
                    const auto s = dot(query_data + h * head_width, k_row, head_width) * scale + b;
                    scratch_[h * active_keys + key] = s;
                    largest[h] = std::max(largest[h], s);
                }
            }
            std::array<float, 8> inv_total{};
            for (std::size_t h = 0; h < 8; ++h) {
                float total = 0;
                auto* scores = scratch_.data() + h * active_keys;
                for (std::size_t key = 0; key < active_keys; ++key) {
                    scores[key] = scores[key] == -std::numeric_limits<float>::infinity()
                                      ? 0.F
                                      : std::exp(scores[key] - largest[h]);
                    total += scores[key];
                }
                inv_total[h] = 1.F / total;
            }
            std::fill_n(output_data, width, 0.F);
            for (std::size_t key = 0; key < active_keys; ++key) {
                const auto b = bias ? bias[key * mask_strides[3]] : 0.F;
                if (b <= -1e8F) continue;
                const auto* v_row = v_base + key * head_width;
                for (std::size_t h = 0; h < 8; ++h) {
                    const float w = scratch_[h * active_keys + key] * inv_total[h];
                    auto* out_h = output_data + h * head_width;
                    for (std::size_t index = 0; index < head_width; ++index) out_h[index] += w * v_row[index];
                }
            }
            return;
        }
        const auto tasks = items * keys * head_width <= 524288 ? std::size_t{1} : std::min(ynn::thread_count(), items);
        scratch_.resize(tasks * keys);
        const auto per_task = (items + tasks - 1) / tasks;
        const auto run_task = [&](std::size_t task) {
            auto* scores = scratch_.data() + task * keys;
            for (auto item = task * per_task; item < std::min(items, (task + 1) * per_task); ++item) {
                const auto row = item % rows, head = item / rows % heads_, sample = item / rows / heads_;
                const auto kv_head = head / group_size;
                const auto* q = query_data + (sample * rows + row) * width + head * head_width;
                const auto* k = key_data + sample * key_batch_stride + (key_start_ * key_heads_ + kv_head) * head_width;
                const auto* v =
                    value_data + sample * key_batch_stride + (key_start_ * key_heads_ + kv_head) * head_width;
                const auto* bias =
                    mask ? mask + sample * mask_strides[0] + head * mask_strides[1] + row * mask_strides[2] : nullptr;
                auto largest = -std::numeric_limits<float>::infinity();
                for (std::size_t key = 0; key < keys; ++key) {
                    const auto b = bias ? bias[key * mask_strides[3]] : 0.F;
                    if (b <= -1e8F) {
                        scores[key] = -std::numeric_limits<float>::infinity();
                        continue;
                    }
                    scores[key] = dot(q, k + key * kv_width, head_width) * scale + b;
                    largest = std::max(largest, scores[key]);
                }
                float total = 0;
                for (std::size_t key = 0; key < keys; ++key) {
                    scores[key] =
                        scores[key] == -std::numeric_limits<float>::infinity() ? 0.F : std::exp(scores[key] - largest);
                    total += scores[key];
                }
                const auto inv_total = 1.F / total;
                auto* out = output_data + (sample * rows + row) * width + head * head_width;
                std::fill_n(out, head_width, 0.F);
                for (std::size_t key = 0; key < keys; ++key) {
                    if (scores[key] == 0.F) continue;
                    const auto weight = scores[key] * inv_total;
                    const auto* source = v + key * kv_width;
                    for (std::size_t index = 0; index < head_width; ++index) out[index] += weight * source[index];
                }
            }
        };
        if (tasks <= 1) {
            run_task(0);
            return;
        }
        require(ynn::parallel_for(tasks, run_task));
    }

    std::size_t heads_, key_heads_, key_start_, key_length_;
    bool explicit_scale_;
    float scale_;
    std::vector<float> scratch_;
    OutputPool pool_;
};
#if defined(__ARM_NEON) && (defined(__aarch64__) || defined(__arm64__))
class BlockwiseInt4Linear final : public Operator {
public:
    static auto supports(const OperatorSpec& spec, TensorInputs inputs) noexcept -> bool {
        if (spec.operation != Operation::PACKED_LINEAR || (inputs.size() != 3 && inputs.size() != 4) ||
            spec.attributes.size() < 3)
            return false;
        const auto bits = spec.attributes[0];
        const auto block_size = spec.attributes[1];
        const auto valid_meta = [](DType dt) { return dt == DType::F32 || dt == DType::F16; };
        return bits == 4 && block_size >= 32 && (block_size % 32 == 0) &&
               (inputs.size() == 4 || block_size < inputs[0].size(-1)) && inputs[0].dtype() == DType::F32 &&
               inputs[1].dtype() == DType::U8 && valid_meta(inputs[2].dtype()) &&
               (inputs.size() == 3 || valid_meta(inputs[3].dtype())) && spec.epsilon <= 0.F;
    }

    static auto pack_scales_and_zeros(const Tensor& scales, const Tensor& zeros)
        -> std::shared_ptr<const std::vector<__fp16>> {
        const auto N = static_cast<std::size_t>(scales.size(0));
        const auto B = static_cast<std::size_t>(scales.size(1));
        const auto tiles_4 = (N + 3) / 4;
        const bool has_zero = zeros.defined();
        const std::size_t stride = has_zero ? 8 : 4;
        const auto s_bytes = require(std::as_const(scales).host_bytes());
        const bool s_f16 = scales.dtype() == DType::F16;
        const auto* s_f32 = s_f16 ? nullptr : reinterpret_cast<const float*>(s_bytes.data());
        const auto* s_h16 = s_f16 ? reinterpret_cast<const __fp16*>(s_bytes.data()) : nullptr;
        const auto z_bytes = has_zero ? require(std::as_const(zeros).host_bytes()) : std::span<const std::byte>{};
        const bool z_f16 = has_zero && zeros.dtype() == DType::F16;
        const auto* z_f32 = (has_zero && !z_f16) ? reinterpret_cast<const float*>(z_bytes.data()) : nullptr;
        const auto* z_h16 = (has_zero && z_f16) ? reinterpret_cast<const __fp16*>(z_bytes.data()) : nullptr;
        auto packed = std::make_shared<std::vector<__fp16>>(tiles_4 * B * stride);
        for (std::size_t tile = 0; tile < tiles_4; ++tile) {
            const auto n0 = tile * 4;
            for (std::size_t b = 0; b < B; ++b) {
                auto* dst = packed->data() + (tile * B + b) * stride;
                for (std::size_t r = 0; r < 4; ++r) {
                    const auto n = n0 + r;
                    const auto idx = n * B + b;
                    dst[r] = n < N ? (s_f16 ? s_h16[idx] : static_cast<__fp16>(s_f32[idx])) : static_cast<__fp16>(0.0F);
                    if (has_zero)
                        dst[4 + r] =
                            n < N ? (z_f16 ? z_h16[idx] : static_cast<__fp16>(z_f32[idx])) : static_cast<__fp16>(0.0F);
                }
            }
        }
        return packed;
    }

    BlockwiseInt4Linear(const OperatorSpec& spec, const Tensor& input, const Tensor& weight,
                        std::shared_ptr<const std::vector<__fp16>> scale_zero_f16, bool has_zero, tensor::Arena& arena)
        : weight_(weight),
          weight_ptr_(reinterpret_cast<const std::uint8_t*>(require(std::as_const(weight_).host_bytes()).data())),
          scale_zero_f16_(std::move(scale_zero_f16)),
          N_(static_cast<std::size_t>(weight.size(0))),
          K_(static_cast<std::size_t>(input.size(-1))),
          block_size_(static_cast<std::size_t>(spec.attributes[1])),
          B_(K_ / block_size_),
          has_zero_(has_zero),
          output_scale_(std::bit_cast<float>(static_cast<std::int32_t>(spec.attributes[2]))),
          shape_(input.shape().begin(), input.shape().end()),
          pool_(&arena) {
        shape_.back() = static_cast<std::int64_t>(N_);
    }

    auto run(TensorInputs inputs) -> Tensor override {
        std::vector<std::int64_t> out_shape(inputs[0].shape().begin(), inputs[0].shape().end());
        out_shape.back() = static_cast<std::int64_t>(N_);
        auto output = pool_.acquire(out_shape, DType::F32, tensor::Device::cpu());
        compute(inputs[0], output);
        return output;
    }
    auto run_(TensorInputs, Tensor&) -> Tensor override {
        throw ops::Failure({ErrorCode::UNSUPPORTED, "blockwise int4 linear is not in-place"});
    }
    auto run_into(TensorInputs inputs, std::span<Tensor> outputs) -> void override { compute(inputs[0], outputs.back()); }
    auto allocations() const -> AllocationStats override { return pool_.allocations(); }

private:
    template <std::size_t CHUNKS, std::size_t MR, bool HAS_ZERO>
    static void eval_tile_4xM(std::size_t B, const std::uint8_t* w0, const std::uint8_t* w1, const std::uint8_t* w2,
                              const std::uint8_t* w3, const __fp16* sz_ptr, const std::int8_t* q_base,
                              const float* a_scale_base, const float* a_bsum_base, float* y_base, std::size_t N,
                              std::size_t n_valid, float output_scale) noexcept {
        constexpr std::size_t G = CHUNKS * 32;
        constexpr std::size_t W_STEP = CHUNKS * 16;
        float32x4_t acc[MR];
        for (std::size_t r = 0; r < MR; ++r) acc[r] = vdupq_n_f32(0.0F);
        float32x4_t acc_zero_m1 = vdupq_n_f32(0.0F);

        for (std::size_t b = 0; b < B; ++b) {
            float32x4_t s4, z4;
            if constexpr (HAS_ZERO) {
                const float16x8_t sz = vld1q_f16(sz_ptr + b * 8);
                s4 = vcvt_f32_f16(vget_low_f16(sz));
                z4 = vcvt_f32_f16(vget_high_f16(sz));
            } else {
                s4 = vcvt_f32_f16(vld1_f16(sz_ptr + b * 4));
                z4 = vdupq_n_f32(0.0F);
            }

            if constexpr (CHUNKS == 1) {
                const uint8x16_t raw0 = vld1q_u8(w0 + b * 16);
                const uint8x16_t raw1 = vld1q_u8(w1 + b * 16);
                const uint8x16_t raw2 = vld1q_u8(w2 + b * 16);
                const uint8x16_t raw3 = vld1q_u8(w3 + b * 16);

                const int8x16_t w0_e = vshrq_n_s8(vreinterpretq_s8_u8(vshlq_n_u8(raw0, 4)), 4);
                const int8x16_t w0_o = vshrq_n_s8(vreinterpretq_s8_u8(raw0), 4);
                const int8x16_t w1_e = vshrq_n_s8(vreinterpretq_s8_u8(vshlq_n_u8(raw1, 4)), 4);
                const int8x16_t w1_o = vshrq_n_s8(vreinterpretq_s8_u8(raw1), 4);
                const int8x16_t w2_e = vshrq_n_s8(vreinterpretq_s8_u8(vshlq_n_u8(raw2, 4)), 4);
                const int8x16_t w2_o = vshrq_n_s8(vreinterpretq_s8_u8(raw2), 4);
                const int8x16_t w3_e = vshrq_n_s8(vreinterpretq_s8_u8(vshlq_n_u8(raw3, 4)), 4);
                const int8x16_t w3_o = vshrq_n_s8(vreinterpretq_s8_u8(raw3), 4);

                for (std::size_t r = 0; r < MR; ++r) {
                    const std::int8_t* qp = q_base + (r * B + b) * 32;
                    const int8x16_t qe = vld1q_s8(qp + 0);
                    const int8x16_t qo = vld1q_s8(qp + 16);
                    const int32x4_t d0 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), w0_e, qe), w0_o, qo);
                    const int32x4_t d1 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), w1_e, qe), w1_o, qo);
                    const int32x4_t d2 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), w2_e, qe), w2_o, qo);
                    const int32x4_t d3 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), w3_e, qe), w3_o, qo);
                    const int32x4_t dots = vpaddq_s32(vpaddq_s32(d0, d1), vpaddq_s32(d2, d3));
                    const float32x4_t fdots = vcvtq_f32_s32(dots);
                    const float a_s = a_scale_base[r * B + b];
                    acc[r] = vfmaq_f32(acc[r], fdots, vmulq_n_f32(s4, a_s));
                    if constexpr (HAS_ZERO) {
                        const float a_z = a_bsum_base[r * B + b];
                        if constexpr (MR == 1) {
                            acc_zero_m1 = vfmaq_n_f32(acc_zero_m1, z4, a_z);
                        } else {
                            acc[r] = vfmaq_n_f32(acc[r], z4, a_z);
                        }
                    }
                }
            } else {
                int32x4_t d0[MR], d1[MR], d2[MR], d3[MR];
                for (std::size_t r = 0; r < MR; ++r) {
                    d0[r] = vdupq_n_s32(0);
                    d1[r] = vdupq_n_s32(0);
                    d2[r] = vdupq_n_s32(0);
                    d3[r] = vdupq_n_s32(0);
                }

                for (std::size_t c = 0; c < CHUNKS; ++c) {
                    const uint8x16_t raw0 = vld1q_u8(w0 + b * W_STEP + c * 16);
                    const uint8x16_t raw1 = vld1q_u8(w1 + b * W_STEP + c * 16);
                    const uint8x16_t raw2 = vld1q_u8(w2 + b * W_STEP + c * 16);
                    const uint8x16_t raw3 = vld1q_u8(w3 + b * W_STEP + c * 16);

                    const int8x16_t w0_e = vshrq_n_s8(vreinterpretq_s8_u8(vshlq_n_u8(raw0, 4)), 4);
                    const int8x16_t w0_o = vshrq_n_s8(vreinterpretq_s8_u8(raw0), 4);
                    const int8x16_t w1_e = vshrq_n_s8(vreinterpretq_s8_u8(vshlq_n_u8(raw1, 4)), 4);
                    const int8x16_t w1_o = vshrq_n_s8(vreinterpretq_s8_u8(raw1), 4);
                    const int8x16_t w2_e = vshrq_n_s8(vreinterpretq_s8_u8(vshlq_n_u8(raw2, 4)), 4);
                    const int8x16_t w2_o = vshrq_n_s8(vreinterpretq_s8_u8(raw2), 4);
                    const int8x16_t w3_e = vshrq_n_s8(vreinterpretq_s8_u8(vshlq_n_u8(raw3, 4)), 4);
                    const int8x16_t w3_o = vshrq_n_s8(vreinterpretq_s8_u8(raw3), 4);

                    for (std::size_t r = 0; r < MR; ++r) {
                        const std::int8_t* qp = q_base + (r * B + b) * G + c * 32;
                        const int8x16_t qe = vld1q_s8(qp + 0);
                        const int8x16_t qo = vld1q_s8(qp + 16);
                        d0[r] = vdotq_s32(vdotq_s32(d0[r], w0_e, qe), w0_o, qo);
                        d1[r] = vdotq_s32(vdotq_s32(d1[r], w1_e, qe), w1_o, qo);
                        d2[r] = vdotq_s32(vdotq_s32(d2[r], w2_e, qe), w2_o, qo);
                        d3[r] = vdotq_s32(vdotq_s32(d3[r], w3_e, qe), w3_o, qo);
                    }
                }

                for (std::size_t r = 0; r < MR; ++r) {
                    const int32x4_t dots = vpaddq_s32(vpaddq_s32(d0[r], d1[r]), vpaddq_s32(d2[r], d3[r]));
                    const float32x4_t fdots = vcvtq_f32_s32(dots);
                    const float a_s = a_scale_base[r * B + b];
                    acc[r] = vfmaq_f32(acc[r], fdots, vmulq_n_f32(s4, a_s));
                    if constexpr (HAS_ZERO) {
                        const float a_z = a_bsum_base[r * B + b];
                        if constexpr (MR == 1) {
                            acc_zero_m1 = vfmaq_n_f32(acc_zero_m1, z4, a_z);
                        } else {
                            acc[r] = vfmaq_n_f32(acc[r], z4, a_z);
                        }
                    }
                }
            }
        }

        if constexpr (HAS_ZERO && MR == 1) acc[0] = vaddq_f32(acc[0], acc_zero_m1);
        for (std::size_t r = 0; r < MR; ++r) {
            float32x4_t res = acc[r];
            if (output_scale > 0.0F) {
                const float32x4_t vinv = vdupq_n_f32(1.0F / output_scale);
                const float32x4_t vscale = vdupq_n_f32(output_scale);
                res = vmulq_f32(vminq_f32(vmaxq_f32(vrndnq_f32(vmulq_f32(res, vinv)), vdupq_n_f32(-128.0F)),
                                          vdupq_n_f32(127.0F)),
                                vscale);
            }
            if (n_valid == 4) {
                vst1q_f32(y_base + r * N, res);
            } else {
                alignas(16) float tmp[4];
                vst1q_f32(tmp, res);
                for (std::size_t i = 0; i < n_valid; ++i) y_base[r * N + i] = tmp[i];
            }
        }
    }

    template <std::size_t CHUNKS, bool HAS_ZERO>
    void dispatch_tiles(std::size_t t_begin, std::size_t t_end, std::size_t M, float* y) const noexcept {
        const auto row_bytes = K_ / 2;
        const auto* sz_all = scale_zero_f16_->data();
        constexpr std::size_t SZ_STRIDE = HAS_ZERO ? 8 : 4;
        for (std::size_t tile = t_begin; tile < t_end; ++tile) {
            const auto n0 = tile * 4;
            const auto n_valid = std::min<std::size_t>(4, N_ - n0);
            const auto* w0 = weight_ptr_ + n0 * row_bytes;
            const auto* w1 = n_valid > 1 ? w0 + row_bytes : w0;
            const auto* w2 = n_valid > 2 ? w0 + 2 * row_bytes : w0;
            const auto* w3 = n_valid > 3 ? w0 + 3 * row_bytes : w0;
            const auto* sz_ptr = sz_all + tile * B_ * SZ_STRIDE;

            std::size_t m = 0;
            if constexpr (CHUNKS == 1) {
                for (; m + 8 <= M; m += 8) {
                    eval_tile_4xM<CHUNKS, 8, HAS_ZERO>(B_, w0, w1, w2, w3, sz_ptr, act_q_.data() + m * K_,
                                                       act_scale_.data() + m * B_, act_bsum_.data() + m * B_,
                                                       y + m * N_ + n0, N_, n_valid, output_scale_);
                }
            }
            for (; m + 4 <= M; m += 4) {
                eval_tile_4xM<CHUNKS, 4, HAS_ZERO>(B_, w0, w1, w2, w3, sz_ptr, act_q_.data() + m * K_,
                                                   act_scale_.data() + m * B_, act_bsum_.data() + m * B_,
                                                   y + m * N_ + n0, N_, n_valid, output_scale_);
            }
            if (m + 3 <= M) {
                eval_tile_4xM<CHUNKS, 3, HAS_ZERO>(B_, w0, w1, w2, w3, sz_ptr, act_q_.data() + m * K_,
                                                   act_scale_.data() + m * B_, act_bsum_.data() + m * B_,
                                                   y + m * N_ + n0, N_, n_valid, output_scale_);
                m += 3;
            }
            if (m + 2 <= M) {
                eval_tile_4xM<CHUNKS, 2, HAS_ZERO>(B_, w0, w1, w2, w3, sz_ptr, act_q_.data() + m * K_,
                                                   act_scale_.data() + m * B_, act_bsum_.data() + m * B_,
                                                   y + m * N_ + n0, N_, n_valid, output_scale_);
                m += 2;
            }
            if (m < M) {
                eval_tile_4xM<CHUNKS, 1, HAS_ZERO>(B_, w0, w1, w2, w3, sz_ptr, act_q_.data() + m * K_,
                                                   act_scale_.data() + m * B_, act_bsum_.data() + m * B_,
                                                   y + m * N_ + n0, N_, n_valid, output_scale_);
            }
        }
    }

    auto compute(const Tensor& input, Tensor& output) -> void {
        const auto* x = require(input.data<float>()).data();
        auto* y = require(output.data<float>()).data();
        const auto M = input.numel() / K_;
        act_q_.resize(M * K_);
        act_scale_.resize(M * B_);
        if (has_zero_) act_bsum_.resize(M * B_);

        const auto chunks_per_block = block_size_ / 32;
        for (std::size_t m = 0; m < M; ++m) {
            for (std::size_t b = 0; b < B_; ++b) {
                const float* x_block = x + (m * B_ + b) * block_size_;
                float32x4_t vsum0 = vdupq_n_f32(0.0F), vsum1 = vdupq_n_f32(0.0F);
                float32x4_t vmax0 = vdupq_n_f32(0.0F), vmax1 = vdupq_n_f32(0.0F);
                for (std::size_t c = 0; c < chunks_per_block; ++c) {
                    const float* xp = x_block + c * 32;
                    const float32x4_t x0 = vld1q_f32(xp + 0), x1 = vld1q_f32(xp + 4);
                    const float32x4_t x2 = vld1q_f32(xp + 8), x3 = vld1q_f32(xp + 12);
                    const float32x4_t x4 = vld1q_f32(xp + 16), x5 = vld1q_f32(xp + 20);
                    const float32x4_t x6 = vld1q_f32(xp + 24), x7 = vld1q_f32(xp + 28);
                    if (has_zero_) {
                        vsum0 = vaddq_f32(vsum0, vaddq_f32(vaddq_f32(x0, x1), vaddq_f32(x2, x3)));
                        vsum1 = vaddq_f32(vsum1, vaddq_f32(vaddq_f32(x4, x5), vaddq_f32(x6, x7)));
                    }
                    vmax0 = vmaxq_f32(vmax0, vmaxq_f32(vmaxq_f32(vabsq_f32(x0), vabsq_f32(x1)),
                                                       vmaxq_f32(vabsq_f32(x2), vabsq_f32(x3))));
                    vmax1 = vmaxq_f32(vmax1, vmaxq_f32(vmaxq_f32(vabsq_f32(x4), vabsq_f32(x5)),
                                                       vmaxq_f32(vabsq_f32(x6), vabsq_f32(x7))));
                }
                const float amax = vmaxvq_f32(vmaxq_f32(vmax0, vmax1));
                const float scale = amax / 127.0F;
                const float inv_scale = amax > 0.0F ? 127.0F / amax : 0.0F;
                act_scale_[m * B_ + b] = scale;
                if (has_zero_) act_bsum_[m * B_ + b] = vaddvq_f32(vaddq_f32(vsum0, vsum1));

                const float32x4_t vinv = vdupq_n_f32(inv_scale);
                for (std::size_t c = 0; c < chunks_per_block; ++c) {
                    const float* xp = x_block + c * 32;
                    const int32x4_t i0 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(xp + 0), vinv));
                    const int32x4_t i1 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(xp + 4), vinv));
                    const int32x4_t i2 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(xp + 8), vinv));
                    const int32x4_t i3 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(xp + 12), vinv));
                    const int32x4_t i4 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(xp + 16), vinv));
                    const int32x4_t i5 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(xp + 20), vinv));
                    const int32x4_t i6 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(xp + 24), vinv));
                    const int32x4_t i7 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(xp + 28), vinv));
                    const int16x8_t s0 = vcombine_s16(vqmovn_s32(i0), vqmovn_s32(i1));
                    const int16x8_t s1 = vcombine_s16(vqmovn_s32(i2), vqmovn_s32(i3));
                    const int16x8_t s2 = vcombine_s16(vqmovn_s32(i4), vqmovn_s32(i5));
                    const int16x8_t s3 = vcombine_s16(vqmovn_s32(i6), vqmovn_s32(i7));
                    const int8x16_t q0 = vcombine_s8(vqmovn_s16(s0), vqmovn_s16(s1));
                    const int8x16_t q1 = vcombine_s8(vqmovn_s16(s2), vqmovn_s16(s3));
                    std::int8_t* qp = act_q_.data() + (m * B_ + b) * block_size_ + c * 32;
                    vst1q_s8(qp + 0, vuzp1q_s8(q0, q1));
                    vst1q_s8(qp + 16, vuzp2q_s8(q0, q1));
                }
            }
        }

        const auto tiles_4 = (N_ + 3) / 4;
        const auto tasks = (M == 1 && K_ * N_ <= 384 * 1024) ? std::size_t{1} : std::min(ynn::thread_count(), tiles_4);
        const auto tiles_per_task = (tiles_4 + tasks - 1) / tasks;
        const auto run_task = [&](std::size_t task) {
            const auto t_begin = task * tiles_per_task;
            const auto t_end = std::min(tiles_4, t_begin + tiles_per_task);
            if (t_begin >= t_end) return;
            if (has_zero_) {
                if (chunks_per_block == 1) {
                    dispatch_tiles<1, true>(t_begin, t_end, M, y);
                } else if (chunks_per_block == 2) {
                    dispatch_tiles<2, true>(t_begin, t_end, M, y);
                } else if (chunks_per_block == 4) {
                    dispatch_tiles<4, true>(t_begin, t_end, M, y);
                } else {
                    dispatch_tiles<8, true>(t_begin, t_end, M, y);
                }
            } else {
                if (chunks_per_block == 1) {
                    dispatch_tiles<1, false>(t_begin, t_end, M, y);
                } else if (chunks_per_block == 2) {
                    dispatch_tiles<2, false>(t_begin, t_end, M, y);
                } else if (chunks_per_block == 4) {
                    dispatch_tiles<4, false>(t_begin, t_end, M, y);
                } else {
                    dispatch_tiles<8, false>(t_begin, t_end, M, y);
                }
            }
        };
        if (tasks <= 1) {
            run_task(0);
            return;
        }
        require(ynn::parallel_for(tasks, run_task));
    }

    Tensor weight_;
    const std::uint8_t* weight_ptr_;
    std::shared_ptr<const std::vector<__fp16>> scale_zero_f16_;
    std::size_t N_, K_, block_size_, B_;
    bool has_zero_;
    float output_scale_;
    std::vector<std::int64_t> shape_;
    OutputPool pool_;
    std::vector<std::int8_t> act_q_;
    std::vector<float> act_scale_;
    std::vector<float> act_bsum_;
};
#endif

class CpuOperator final : public Operator {
public:
    CpuOperator(ynn::Executable executable, std::size_t count, std::size_t dynamic_count, DType dtype,
                std::vector<std::int64_t> shape, tensor::Arena& arena,
                std::vector<std::shared_ptr<const void>> constant_owners = {}, const Tensor& block_zeros = {},
                std::size_t block_size = 0)
        : executable_(std::move(executable)),
          count_(count),
          dynamic_count_(dynamic_count),
          dtype_(dtype),
          shape_(std::move(shape)),
          pool_(&arena),
          constant_owners_(std::move(constant_owners)),
          block_zeros_(block_zeros),
          block_size_(block_size) {}
    auto run(TensorInputs inputs) -> Tensor override {
        for (std::size_t index = 0; index < dynamic_count_; ++index) require(executable_.bind(index, inputs[index]));
        auto output = pool_.acquire(shape_, dtype_, tensor::Device::cpu());
        require(executable_.bind(count_, output));
        require(executable_.invoke());
        if (block_zeros_.defined() && block_size_ > 0) apply_block_zeros(inputs[0], output);
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
        if (block_zeros_.defined() && block_size_ > 0) apply_block_zeros(inputs[0], outputs.back());
    }
    auto allocations() const -> AllocationStats override { return pool_.allocations(); }

private:
    auto apply_block_zeros(const Tensor& input, Tensor& output) const -> void {
        const auto* x = require(input.data<float>()).data();
        auto* y = require(output.data<float>()).data();
        const auto* z = require(std::as_const(block_zeros_).data<float>()).data();
        const auto N = static_cast<std::size_t>(block_zeros_.size(0));
        const auto B = static_cast<std::size_t>(block_zeros_.size(1));
        const auto K = B * block_size_;
        const auto M = input.numel() / K;
        std::vector<float> sums(B);
        for (std::size_t m = 0; m < M; ++m) {
            for (std::size_t b = 0; b < B; ++b) {
                const auto* xb = x + (m * B + b) * block_size_;
                float s = 0.F;
                for (std::size_t k = 0; k < block_size_; ++k) s += xb[k];
                sums[b] = s;
            }
            for (std::size_t n = 0; n < N; ++n) {
                float corr = 0.F;
                for (std::size_t b = 0; b < B; ++b) corr += sums[b] * z[n * B + b];
                y[m * N + n] += corr;
            }
        }
    }

    ynn::Executable executable_;
    std::size_t count_, dynamic_count_;
    DType dtype_;
    std::vector<std::int64_t> shape_;
    OutputPool pool_;
    std::vector<std::shared_ptr<const void>> constant_owners_;
    Tensor block_zeros_;
    std::size_t block_size_ = 0;
};
class CpuBackend final : public OperatorBackend {
public:
    CpuBackend() { require(arena_.reserve(8 * 1024 * 1024)); }
    auto synchronize() -> void override {}
    auto release_cached_buffers() -> void override { arena_ = tensor::Arena{}; }
    auto configure_cache(const std::filesystem::path& directory, const std::filesystem::path& source) -> void override {
        feed_forward_cache_.configure(directory, source);
    }
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
        if (spec.operation == Operation::SILU_GATE && SiluGate::supports(inputs))
            return std::make_unique<SiluGate>(inputs[0], arena_);
        if (spec.operation == Operation::GELU_GATE && GeluGate::supports(inputs))
            return std::make_unique<GeluGate>(inputs[0], arena_);
        if (spec.operation == Operation::ROTARY && Rotary::supports(inputs)) return std::make_unique<Rotary>(arena_);
        if ((spec.operation == Operation::RMS_NORM || spec.operation == Operation::RMS_NORM_RESIDUAL) &&
            RmsNorm::supports(spec, inputs))
            return std::make_unique<RmsNorm>(spec, arena_);
        if (spec.operation == Operation::RMS_ROTARY && RmsRotary::supports(spec, inputs))
            return std::make_unique<RmsRotary>(spec, arena_);
        if (spec.operation == Operation::CAST && spec.dtype == DType::I8 && spec.epsilon > 0)
            return std::make_unique<QuantizeInt8>(spec.epsilon, arena_);
#if defined(__ARM_NEON) && (defined(__aarch64__) || defined(__arm64__))
        if (BlockwiseInt4Linear::supports(spec, inputs)) {
            const bool has_zero = inputs.size() == 4;
            const std::pair<const void*, std::size_t> sz_key{inputs[2].storage_identity(), inputs[2].storage_offset()};
            auto& cached_sz = scale_zero_cache_[sz_key];
            if (!cached_sz) {
                cached_sz = BlockwiseInt4Linear::pack_scales_and_zeros(inputs[2], has_zero ? inputs[3] : Tensor{});
                if (inputs[1].size(0) < 65536) {
                    inputs[2].release_pages(true);
                    if (has_zero) inputs[3].release_pages(true);
                }
            }
            return std::make_unique<BlockwiseInt4Linear>(spec, inputs[0], inputs[1], cached_sz, has_zero, arena_);
        }
#endif
        if (spec.operation == Operation::ATTENTION) {
#if defined(__APPLE__)
            if (PytorchParityAttention::supports(spec, inputs))
                return std::make_unique<PytorchParityAttention>(spec, arena_);
#endif
            if (FewQueryAttention::supports(spec, inputs)) return std::make_unique<FewQueryAttention>(spec, arena_);
            const bool byte_cache = inputs[1].dtype() == DType::I8 && !spec.quantization[0].scales.empty() &&
                                    !spec.quantization[1].scales.empty();
            if (byte_cache && inputs[0].dtype() == DType::F32 && inputs[2].dtype() == inputs[1].dtype())
                return std::make_unique<QuantizedAttention>(spec, inputs, arena_);
        }
        const auto flags = inputs[0].dtype() == DType::BF16 ? YNN_FLAG_NO_EXCESS_PRECISION : 0;
        const bool paired = spec.operation == Operation::RESIDUAL_NORM;
        const bool has_packed_zero = spec.operation == Operation::PACKED_LINEAR && inputs.size() == 4;
        const auto graph_input_count = has_packed_zero ? 3U : inputs.size();
        auto graph = require(ynn::Graph::create(graph_input_count + (paired ? 2 : 1), flags));
        auto native = graph.get();
        const auto constant =
            !spec.dynamic_parameters &&
            (spec.operation == Operation::LINEAR || spec.operation == Operation::QUANTIZED_LINEAR ||
             spec.operation == Operation::LAYER_NORM || spec.operation == Operation::RMS_NORM ||
             spec.operation == Operation::PACKED_LINEAR || spec.operation == Operation::GATED_FEED_FORWARD || paired);
        const std::size_t dynamic_count = constant ? (paired ? 2 : 1) : graph_input_count;
        const std::size_t column_alignment = spec.operation == Operation::PACKED_LINEAR ? 8 / spec.attributes[0] : 1;
        const auto padded_columns =
            spec.operation == Operation::PACKED_LINEAR
                ? ((inputs[1].size(0) + column_alignment - 1) / column_alignment) * column_alignment
                : 0;
        const bool pad_columns = spec.operation == Operation::PACKED_LINEAR && padded_columns != inputs[1].size(0);
        std::vector<std::uint32_t> operands;
        for (std::size_t index = 0; index < graph_input_count; ++index) {
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
        std::vector<std::shared_ptr<const void>> constant_owners;
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
                        if (index != 0 && spec.attributes.size() >= 4) {
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
                const bool packed = spec.operation == Operation::PACKED_LINEAR;
                const auto block_size = packed ? static_cast<std::size_t>(spec.attributes[1]) : inputs[0].size(-1);
                const bool transposed = packed || (!spec.attributes.empty() && spec.attributes[0]);
                auto zero = YNN_INVALID_VALUE_ID, scale = YNN_INVALID_VALUE_ID, quantized = YNN_INVALID_VALUE_ID;
                if (packed && spec.epsilon > 0) {
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
                if (transposed) {
                    const std::array<std::int32_t, 2> axes{1, 0};
                    weight = YNN_INVALID_VALUE_ID;
                    check(ynn_define_static_transpose(native, 2, axes.data(), operands[1], &weight, 0));
                }
                if (packed && spec.epsilon > 0 && block_size == inputs[0].size(-1) &&
                    (SHARED_DECODE_PACKING || inputs[0].numel() > inputs[0].size(-1))) {
                    const std::size_t columns = padded_columns;
                    auto channel_scale = YNN_INVALID_VALUE_ID;
                    check(ynn_define_static_reshape(native, 1, &columns, operands[2], &channel_scale, 0));
                    check(ynn_define_dot(native, 1, quantized, weight, YNN_INVALID_VALUE_ID, &result, 0));
                    result = binary(ynn_binary_multiply, binary(ynn_binary_multiply, result, channel_scale), scale);
                } else
                    check(::ynn::define_blockwise_dot(
                        native, quantized, zero, scale, weight, YNN_INVALID_VALUE_ID, operands[2], block_size,
                        packed ? YNN_INVALID_VALUE_ID : operands[3], ynn_type_fp32, result, 0));
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
                const bool lookup_cache = feed_forward_cache_.enabled();
                const auto source_hash = lookup_cache ? hash_tensor(inputs[3], hash_tensor(inputs[1])) : 0;
                ynn::FeedForwardCacheSpec cache_spec{.bits = static_cast<std::uint32_t>(bits),
                                                     .input_size = input_size,
                                                     .intermediate_size = intermediate,
                                                     .source_hash = source_hash};
                const auto payload_size = 3 * input_size * intermediate / (8 / bits);
                auto cached =
                    lookup_cache ? feed_forward_cache_.load(cache_spec, payload_size) : ynn::FeedForwardCacheEntry{};
                const bool store_cache = !cached && feed_forward_cache_.writable();
                if (cached) constant_owners.push_back(cached.owner);
                std::vector<std::uint8_t> payload;
                if (store_cache) payload.reserve(payload_size);
                std::size_t payload_offset = 0;
                std::vector<std::uint32_t> gate_weight_ids(tiles), gate_scale_ids(tiles), up_weight_ids(tiles),
                    up_scale_ids(tiles), down_weight_ids(tiles);
                const auto define_weight = [&](const Tensor& weight, std::size_t row_start, std::size_t row_count,
                                               std::size_t column_start, std::size_t column_count) {
                    const auto size = column_count * row_count / (8 / bits);
                    std::vector<std::uint8_t> generated;
                    std::span<const std::uint8_t> bytes;
                    if (cached) {
                        if (payload_offset > cached.bytes.size() || size > cached.bytes.size() - payload_offset)
                            throw ops::Failure({ErrorCode::RUNTIME, "invalid cached feed-forward payload"});
                        bytes = cached.bytes.subspan(payload_offset, size);
                    } else {
                        generated =
                            transpose_packed_tile(weight, bits, row_start, row_count, column_start, column_count);
                        bytes = generated;
                        if (store_cache) payload.insert(payload.end(), bytes.begin(), bytes.end());
                    }
                    payload_offset += size;
                    const std::array<std::size_t, 2> shape{column_count, row_count};
                    auto id = YNN_INVALID_VALUE_ID;
                    const auto dtype = bits == 2 ? ynn_type_int2 : bits == 4 ? ynn_type_int4 : ynn_type_int8;
                    check(ynn_define_tensor(native, dtype, shape.size(), shape.data(), bytes.data(),
                                            cached ? 0 : YNN_VALUE_FLAG_COPY_DATA, &id));
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
                if (store_cache) feed_forward_cache_.store(cache_spec, payload);
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
            case Operation::SILU_MULTIPLY:
                result = binary(ynn_binary_multiply,
                                binary(ynn_binary_multiply, operands[0], unary(ynn_unary_sigmoid, operands[0])),
                                operands[1]);
                break;
            case Operation::GELU_GATE:
            case Operation::SILU_GATE: {
                const std::int32_t axis = static_cast<std::int32_t>(inputs[0].dimensions() - 1);
                const std::int64_t begin = 0, middle = inputs[0].size(-1) / 2, end = inputs[0].size(-1), stride = 1;
                auto gate = YNN_INVALID_VALUE_ID, up = YNN_INVALID_VALUE_ID;
                check(ynn_define_static_slice(native, 1, &axis, &begin, &middle, &stride, operands[0], &gate, 0));
                check(ynn_define_static_slice(native, 1, &axis, &middle, &end, &stride, operands[0], &up, 0));
                if (spec.operation == Operation::GELU_GATE) {
                    auto square = binary(ynn_binary_multiply, gate, gate);
                    auto cubic = binary(ynn_binary_multiply, square, gate);
                    auto inner = binary(ynn_binary_add, gate, binary(ynn_binary_multiply, cubic, scalar(0.044715F)));
                    auto probability = binary(
                        ynn_binary_add, scalar(1.F),
                        unary(ynn_unary_tanh,
                              binary(ynn_binary_multiply, inner, scalar(std::sqrt(2.F / 3.14159265358979323846F)))));
                    result = binary(
                        ynn_binary_multiply,
                        binary(ynn_binary_multiply, binary(ynn_binary_multiply, gate, scalar(0.5F)), probability), up);
                    break;
                }
                result =
                    binary(ynn_binary_multiply, binary(ynn_binary_multiply, gate, unary(ynn_unary_sigmoid, gate)), up);
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
            case Operation::SNAKE_CONV1D:
            case Operation::SNAKE_QUANTIZED_LINEAR:
                break;
        }
        auto output_id = static_cast<std::uint32_t>(graph_input_count);
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
        const bool small_packed_projection =
            spec.operation == Operation::PACKED_LINEAR && inputs[0].numel() == inputs[0].size(-1) &&
            static_cast<std::size_t>(inputs[0].size(-1)) * static_cast<std::size_t>(inputs[1].size(0)) <= 384 * 1024;
        const bool small_elementwise =
            (spec.operation == Operation::GELU_GATE || spec.operation == Operation::SILU_GATE ||
             spec.operation == Operation::GELU_MULTIPLY || spec.operation == Operation::SILU_MULTIPLY ||
             spec.operation == Operation::GELU || spec.operation == Operation::ADD ||
             spec.operation == Operation::MULTIPLY || spec.operation == Operation::TANH ||
             spec.operation == Operation::STATIC_ROUND) &&
            inputs[0].numel() <= 65536;
        if (small_packed_projection || small_elementwise) operator_threads = 1;
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
        auto op = std::make_unique<CpuOperator>(
            std::move(executable), graph_input_count, dynamic_count, dtype,
            std::vector<std::int64_t>(shape.begin(), shape.end()), arena_, std::move(constant_owners),
            has_packed_zero ? inputs[3] : Tensor{},
            has_packed_zero ? static_cast<std::size_t>(spec.attributes[1]) : 0);
        if (constant) {
            if (inputs[1].size(0) < 65536) {
                for (std::size_t index = dynamic_count; index < inputs.size(); ++index) inputs[index].release_pages();
            }
#if defined(__APPLE__)
            if (inputs[1].nbytes() >= (std::size_t{8} << 20)) ::malloc_zone_pressure_relief(nullptr, 0);
#endif
        }
        return op;
    }

private:
    tensor::Arena arena_;
    ynn::FeedForwardCache feed_forward_cache_;
#if defined(__ARM_NEON) && (defined(__aarch64__) || defined(__arm64__))
    std::map<std::pair<const void*, std::size_t>, std::shared_ptr<const std::vector<__fp16>>> scale_zero_cache_;
#endif
};
} // namespace
auto cpu_operators() -> std::unique_ptr<OperatorBackend> { return std::make_unique<CpuBackend>(); }
} // namespace kidi::runtime