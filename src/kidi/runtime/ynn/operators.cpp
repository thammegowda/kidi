#include "kidi/runtime/operator.h"
#include "kidi/runtime/parity_math.h"
#include "kidi/runtime/ynn/feed_forward_cache.h"
#include "kidi/runtime/ynn/graph.h"
#include "kidi/ops/context.h"
#include "ynnpack/composites/composites.h"
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
    const auto source = require(input.data<std::uint8_t>());
    std::vector<std::uint8_t> destination(column_count * (row_count / values_per_byte));
    ::ynn::transpose_packed_subbyte_tile(bits, source.data(), static_cast<std::size_t>(input.size(1)), row_start,
                                         row_count, column_start, column_count, destination.data());
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
        ::ynn::argmax_rows_f32(rows, width, require(logits.data<float>()).data(),
                               require(output.data<std::int32_t>()).data());
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
        ::ynn::quantize_symmetric_int8_f32(source.size(), scale_, source.data(),
                                           require(output.data<std::int8_t>()).data());
    }

    float scale_;
    OutputPool pool_;
};
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

class GatedActivationOp final : public Operator {
public:
    GatedActivationOp(const Tensor& input, bool silu, tensor::Arena& arena)
        : rows_(input.numel() / input.size(-1)), width_(input.size(-1) / 2), silu_(silu), pool_(&arena) {
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
        throw ops::Failure({ErrorCode::UNSUPPORTED, "gated activation is not an in-place operation"});
    }
    auto run_into(TensorInputs inputs, std::span<Tensor> outputs) -> void override { apply(inputs[0], outputs[0]); }
    auto allocations() const -> AllocationStats override { return pool_.allocations(); }

private:
    auto apply(const Tensor& input, Tensor& output) const -> void {
        const auto* source = require(input.data<float>()).data();
        auto* target = require(output.data<float>()).data();
        for_rows(rows_, 2 * width_, [&](std::size_t begin, std::size_t end) {
            check(silu_ ? ::ynn::silu_gate_f32(rows_, width_, source, target, begin, end)
                        : ::ynn::gelu_gate_f32(rows_, width_, source, target, begin, end));
        });
    }

    std::size_t rows_, width_;
    bool silu_;
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
        const auto rows = input.size(1), heads = input.size(2), width = input.size(3);
        const auto shared_angles = cosine.size(0) == 1;
        const auto* source = require(input.data<float>()).data();
        const auto* cosines = require(cosine.data<float>()).data();
        const auto* sines = require(inputs[2].data<float>()).data();
        auto* target = require(output.data<float>()).data();
        for_rows(input.size(0) * rows, heads * width, [&](std::size_t begin, std::size_t end) {
            ::ynn::rotary_f32(rows, heads, width, shared_angles, source, cosines, sines, target, begin, end);
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
        for_rows(rows, width, [&](std::size_t begin, std::size_t end) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
            if (!pytorch_parity_) {
                ::ynn::rms_norm_f32(rows, width, epsilon_, source, weight, residual, scaled, scale, target, begin, end);
                return;
            }
            for (auto row = begin; row < end; ++row) {
                const auto* x = source + row * width;
                auto* y = target + row * width;
                const auto sum = parity::sum_squares(std::span(x, width));
                const auto inverse = 1.F / std::sqrt(sum / static_cast<float>(width) + epsilon_);
                if (!residual_) {
                    for (std::size_t index = 0; index < width; ++index) y[index] = x[index] * inverse * weight[index];
                    continue;
                }
                const auto* r = residual + row * width;
                for (std::size_t index = 0; index < width; ++index) {
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
        const auto vectors = input.size(0) * rows * heads;
        const auto* source = require(input.data<float>()).data();
        const auto* weight = require(inputs[1].data<float>()).data();
        const auto* cosines = require(cosine.data<float>()).data();
        const auto* sines = require(inputs[3].data<float>()).data();
        auto* target = require(output.data<float>()).data();
        for_rows(vectors, width, [&](std::size_t begin, std::size_t end) {
            ::ynn::rms_rotary_f32(rows, heads, width, sliced_, input.size(1), input.size(2), sequence_offset_,
                                  channel_offset_, epsilon_, cosine.size(0) == 1, source, weight, cosines, sines,
                                  target, begin, end);
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
            (packed
                 ? (weight.dtype() != DType::U8 && !(bits_ == 8 && weight.dtype() == DType::I8)) ||
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
        for (const auto token : tokens)
            if (token < 0 || static_cast<std::size_t>(token) >= weight.size(0))
                throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "embedding token outside vocabulary"});
        const auto* w_bytes = require(weight.host_bytes()).data();
        const void* s_bytes = bits_ ? require(inputs[2].host_bytes()).data() : nullptr;
        const void* z_bytes = (bits_ && inputs.size() == 4) ? require(inputs[3].host_bytes()).data() : nullptr;
        const bool s_f16 = bits_ && inputs[2].dtype() == DType::F16;
        const bool z_f16 = bits_ && inputs.size() == 4 && inputs[3].dtype() == DType::F16;
        auto* out = require(output.data<float>()).data();
        for_rows(tokens.size(), width_, [&](std::size_t begin, std::size_t end) {
            ::ynn::dequantize_embedding_f32(tokens.data(), tokens.size(), w_bytes, weight.dtype() == DType::BF16, bits_,
                                            unsigned_packed_, width_, groups_, s_bytes, s_f16, z_bytes, z_f16,
                                            multiplier_, out, begin, end);
        });
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
    auto attend(TensorInputs inputs, Tensor& output) -> void {
        const auto& query = inputs[0];
        const auto batch = static_cast<std::size_t>(query.size(0)), rows = static_cast<std::size_t>(query.size(1));
        const auto width = static_cast<std::size_t>(query.size(2)), head_width = width / heads_;
        const auto kv_width = key_heads_ * head_width;
        const auto total_keys = static_cast<std::size_t>(inputs[1].size(1));
        const auto keys = key_length_ ? key_length_ : total_keys;
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
        const ::ynn::attention_f32_params params{
            batch,
            rows,
            heads_,
            key_heads_,
            key_start_,
            key_length_,
            total_keys,
            head_width,
            inputs[1].size(0) == 1 ? 0 : total_keys * kv_width,
            mask_strides[0],
            mask_strides[1],
            mask_strides[2],
            mask_strides[3],
            explicit_scale_ ? scale_ : static_cast<float>(1.0 / std::sqrt(static_cast<double>(head_width))),
        };
        const auto items = ::ynn::attention_f32_work_items(params);
        if (!items) return;
        const auto* query_data = require(query.data<float>()).data();
        const auto* key_data = require(inputs[1].data<float>()).data();
        const auto* value_data = require(inputs[2].data<float>()).data();
        auto* output_data = require(output.data<float>()).data();
        const auto tasks =
            (items == 1 || items * keys * head_width <= 524288) ? std::size_t{1} : std::min(ynn::thread_count(), items);
        const auto scratch_per_task = ::ynn::attention_f32_scratch_size(params);
        scratch_.resize(tasks * scratch_per_task);
        const auto per_task = (items + tasks - 1) / tasks;
        const auto run_task = [&](std::size_t task) {
            const auto begin = task * per_task, end = std::min(items, begin + per_task);
            if (begin >= end) return;
            ::ynn::attention_f32(params, query_data, key_data, value_data, mask, output_data, begin, end,
                                 scratch_.data() + task * scratch_per_task);
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

struct PackedTileKey {
    const void* weight_id = nullptr;
    std::int64_t weight_offset = 0;
    const void* scale_id = nullptr;
    std::int64_t scale_offset = 0;
    const void* zero_id = nullptr;
    std::int64_t zero_offset = 0;
    std::size_t block_size = 0;
    std::int32_t bits = 0;
    auto operator<=>(const PackedTileKey&) const = default;
};

class BlockwiseLinear final : public Operator {
public:
    static auto supports(const OperatorSpec& spec, TensorInputs inputs) noexcept -> bool {
        if (spec.operation != Operation::PACKED_LINEAR || (inputs.size() != 3 && inputs.size() != 4) ||
            spec.attributes.size() < 3 || spec.epsilon > 0.F)
            return false;
        const auto bits = static_cast<std::int32_t>(spec.attributes[0]);
        if (!::ynn::is_blockwise_linear_supported(bits)) return false;
        const auto block_size = static_cast<std::size_t>(spec.attributes[1]);
        if (block_size < 32 || (inputs[0].size(-1) % block_size != 0)) return false;
        const auto valid_meta = [](DType dt) { return dt == DType::F32 || dt == DType::F16; };
        if (inputs[0].dtype() != DType::F32 || !valid_meta(inputs[2].dtype())) return false;
        if (bits == 4) {
            return (block_size % 32 == 0) && inputs[1].dtype() == DType::U8 &&
                   (inputs.size() == 4 ||
                    (block_size < static_cast<std::size_t>(inputs[0].size(-1)) && inputs[1].size(0) >= 256)) &&
                   (inputs.size() == 3 || valid_meta(inputs[3].dtype()));
        }
        if (bits == 8) {
            return (block_size % 16 == 0) && inputs.size() == 3 && inputs[1].size(0) >= 64 &&
                   (inputs[1].dtype() == DType::I8 || inputs[1].dtype() == DType::U8);
        }
        return false;
    }

    static auto pack_tiles(std::int32_t bits, const Tensor& weight, const Tensor& scales, const Tensor& zeros,
                           std::size_t block_size) -> Tensor {
        const auto N = static_cast<std::size_t>(weight.size(0));
        const auto K = static_cast<std::size_t>(weight.size(1)) * (bits == 4 ? 2U : 1U);
        const bool has_zero = zeros.defined();
        const auto total_bytes = ::ynn::packed_blockwise_linear_size(bits, N, K, block_size, has_zero);
        auto packed = require(Tensor::empty({static_cast<std::int64_t>(total_bytes)}, DType::U8));
        const auto* w_bytes = require(std::as_const(weight).host_bytes()).data();
        const auto* s_bytes = require(std::as_const(scales).host_bytes()).data();
        const bool s_f16 = scales.dtype() == DType::F16;
        const void* z_bytes = has_zero ? require(std::as_const(zeros).host_bytes()).data() : nullptr;
        const bool z_f16 = has_zero && zeros.dtype() == DType::F16;
        auto* dst = require(packed.host_bytes()).data();
        const auto tiles_32 = ::ynn::blockwise_linear_tile_count(N);
        const auto workers = std::min(ynn::thread_count(), tiles_32);
        if (workers > 1 && tiles_32 >= 8) {
            const auto per_worker = (tiles_32 + workers - 1) / workers;
            require(ynn::parallel_for(workers, [&](std::size_t task) {
                const auto t_begin = task * per_worker, t_end = std::min(tiles_32, t_begin + per_worker);
                if (t_begin < t_end)
                    ::ynn::pack_blockwise_linear_tiles(bits, N, K, block_size, w_bytes, s_bytes, s_f16, z_bytes, z_f16,
                                                       t_begin, t_end, dst);
            }));
        } else {
            ::ynn::pack_blockwise_linear_tiles(bits, N, K, block_size, w_bytes, s_bytes, s_f16, z_bytes, z_f16, 0,
                                               tiles_32, dst);
        }
        return packed;
    }

    BlockwiseLinear(const OperatorSpec& spec, const Tensor& sample_input, const Tensor& weight, Tensor packed_tiles,
                    bool has_zero, tensor::Arena& arena)
        : packed_(std::move(packed_tiles)),
          packed_ptr_(require(std::as_const(packed_).host_bytes()).data()),
          bits_(static_cast<std::int32_t>(spec.attributes[0])),
          N_(static_cast<std::size_t>(weight.size(0))),
          K_(static_cast<std::size_t>(sample_input.size(-1))),
          block_size_(static_cast<std::size_t>(spec.attributes[1])),
          B_(K_ / block_size_),
          has_zero_(has_zero),
          output_scale_(std::bit_cast<float>(static_cast<std::int32_t>(spec.attributes[2]))),
          pool_(&arena) {}

    auto run(TensorInputs inputs) -> Tensor override {
        std::vector<std::int64_t> out_shape(inputs[0].shape().begin(), inputs[0].shape().end());
        out_shape.back() = static_cast<std::int64_t>(N_);
        auto output = pool_.acquire(out_shape, DType::F32, tensor::Device::cpu());
        apply(inputs[0], output);
        return output;
    }
    auto run_(TensorInputs, Tensor&) -> Tensor override {
        throw ops::Failure({ErrorCode::UNSUPPORTED, "blockwise linear cannot run in place"});
    }
    auto run_into(TensorInputs inputs, std::span<Tensor> outputs) -> void override { apply(inputs[0], outputs.back()); }
    auto allocations() const -> AllocationStats override { return pool_.allocations(); }

private:
    auto apply(const Tensor& input, Tensor& output) -> void {
        const auto M = input.numel() / K_;
        const auto* x = require(input.data<float>()).data();
        auto* y = require(output.data<float>()).data();
        act_q_.resize(M * K_);
        act_scale_.resize(M * B_);
        const bool use_bsum = has_zero_ || bits_ == 8;
        if (use_bsum) act_bsum_.resize(M * B_);
        float* bsum_ptr = use_bsum ? act_bsum_.data() : nullptr;

        const auto q_workers = (M >= 8 && M * K_ >= 65536) ? std::min(ynn::thread_count(), M) : std::size_t{1};
        if (q_workers > 1) {
            const auto per_worker = (M + q_workers - 1) / q_workers;
            require(ynn::parallel_for(q_workers, [&](std::size_t task) {
                const auto m_begin = task * per_worker, m_end = std::min(M, m_begin + per_worker);
                if (m_begin < m_end)
                    ::ynn::quantize_blockwise_activations_f32(bits_, M, K_, block_size_, use_bsum, x, act_q_.data(),
                                                              act_scale_.data(), bsum_ptr, m_begin, m_end);
            }));
        } else {
            ::ynn::quantize_blockwise_activations_f32(bits_, M, K_, block_size_, use_bsum, x, act_q_.data(),
                                                      act_scale_.data(), bsum_ptr, 0, M);
        }

        const auto tiles = ::ynn::blockwise_linear_tile_count(N_);
        const auto tasks = (M == 1 && K_ * N_ <= 64 * 1024) ? std::size_t{1} : std::min(ynn::thread_count(), tiles);
        const auto per_task = (tiles + tasks - 1) / tasks;
        const auto run_tiles = [&](std::size_t task) {
            const auto t_begin = task * per_task, t_end = std::min(tiles, t_begin + per_task);
            if (t_begin < t_end)
                ::ynn::blockwise_linear_f32(bits_, M, N_, K_, block_size_, has_zero_, output_scale_, packed_ptr_,
                                            act_q_.data(), act_scale_.data(), bsum_ptr, y, t_begin, t_end);
        };
        if (tasks <= 1) {
            run_tiles(0);
            return;
        }
        require(ynn::parallel_for(tasks, run_tiles));
    }

    Tensor packed_;
    const void* packed_ptr_;
    std::int32_t bits_;
    std::size_t N_, K_, block_size_, B_;
    bool has_zero_;
    float output_scale_;
    OutputPool pool_;
    std::vector<std::int8_t> act_q_;
    std::vector<float> act_scale_;
    std::vector<float> act_bsum_;
};
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
        if ((spec.operation == Operation::SILU_GATE || spec.operation == Operation::GELU_GATE) &&
            GatedActivationOp::supports(inputs))
            return std::make_unique<GatedActivationOp>(inputs[0], spec.operation == Operation::SILU_GATE, arena_);
        if (spec.operation == Operation::ROTARY && Rotary::supports(inputs)) return std::make_unique<Rotary>(arena_);
        if ((spec.operation == Operation::RMS_NORM || spec.operation == Operation::RMS_NORM_RESIDUAL) &&
            RmsNorm::supports(spec, inputs))
            return std::make_unique<RmsNorm>(spec, arena_);
        if (spec.operation == Operation::RMS_ROTARY && RmsRotary::supports(spec, inputs))
            return std::make_unique<RmsRotary>(spec, arena_);
        if (spec.operation == Operation::CAST && spec.dtype == DType::I8 && spec.epsilon > 0)
            return std::make_unique<QuantizeInt8>(spec.epsilon, arena_);
        if (BlockwiseLinear::supports(spec, inputs)) {
            const auto bits = static_cast<std::int32_t>(spec.attributes[0]);
            const bool has_zero = inputs.size() == 4;
            const auto block_size = static_cast<std::size_t>(spec.attributes[1]);
            const PackedTileKey key{inputs[1].storage_identity(),
                                    inputs[1].storage_offset(),
                                    inputs[2].storage_identity(),
                                    inputs[2].storage_offset(),
                                    has_zero ? inputs[3].storage_identity() : nullptr,
                                    has_zero ? inputs[3].storage_offset() : 0,
                                    block_size,
                                    bits};
            auto& cached_tiles = packed_tile_cache_[key];
            if (!cached_tiles.defined()) {
                cached_tiles = BlockwiseLinear::pack_tiles(bits, inputs[1], inputs[2], has_zero ? inputs[3] : Tensor{},
                                                           block_size);
                if (inputs[1].size(0) < 65536) {
                    inputs[1].release_pages();
                    inputs[2].release_pages();
                    if (has_zero) inputs[3].release_pages();
                }
            }
            return std::make_unique<BlockwiseLinear>(spec, inputs[0], inputs[1], cached_tiles, has_zero, arena_);
        }
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
        auto op = std::make_unique<CpuOperator>(std::move(executable), graph_input_count, dynamic_count, dtype,
                                                std::vector<std::int64_t>(shape.begin(), shape.end()), arena_,
                                                std::move(constant_owners), has_packed_zero ? inputs[3] : Tensor{},
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
    std::map<PackedTileKey, Tensor> packed_tile_cache_;
};
} // namespace
auto cpu_operators() -> std::unique_ptr<OperatorBackend> { return std::make_unique<CpuBackend>(); }
} // namespace kidi::runtime