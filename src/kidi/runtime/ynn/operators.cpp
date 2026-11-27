#include "kidi/runtime/operator.h"
#include "kidi/runtime/parity_math.h"
#include "kidi/runtime/ynn/feed_forward_cache.h"
#include "kidi/runtime/ynn/graph.h"
#include "kidi/ops/context.h"
#include "kidi/checkpoint/ggml/dequantize.h"
#include "ynnpack/composites/composites.h"
#include <atomic>
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
auto check(ynn_status status) -> void { require(ynn::check_status(status, "eager CPU operator")); }
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
        ::ynn::scatter_rows(destination.size(0), destination.size(1), indices.size(), row_bytes, indices.data(),
                            source.data(), target.data());
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
        const auto width = static_cast<std::size_t>(logits.size(-1));
        const auto rows = logits.numel() / width;
        const auto* data = require(logits.data<float>()).data();
        auto* tokens = require(output.data<std::int32_t>()).data();
        const auto threads = ynn::thread_count();
        if (threads <= 1 || rows * width < 65536) {
            ::ynn::argmax_rows_f32(rows, width, data, tokens);
            return;
        }
        if (rows == 1) {
            constexpr std::size_t MAX_CHUNKS = 32;
            const auto chunks = std::min({threads * 2, MAX_CHUNKS, (width + 8191) / 8192});
            if (chunks <= 1) {
                ::ynn::argmax_rows_f32(1, width, data, tokens);
                return;
            }
            const auto step = (width + chunks - 1) / chunks;
            std::array<float, MAX_CHUNKS> best_vals{};
            std::array<std::int32_t, MAX_CHUNKS> best_ids{};
            std::array<bool, MAX_CHUNKS> invalid_flags{};
            require(ynn::parallel_tasks(chunks, [&](std::size_t c) {
                const auto begin = c * step;
                const auto count = begin < width ? std::min(step, width - begin) : 0;
                ::ynn::argmax_chunk_f32(count, data + begin, begin, &best_vals[c], &best_ids[c], &invalid_flags[c]);
            }));
            float best = -std::numeric_limits<float>::infinity();
            std::int32_t winner = -1;
            bool invalid = false;
            for (std::size_t c = 0; c < chunks; ++c) {
                invalid |= invalid_flags[c];
                if (best_vals[c] > best) {
                    best = best_vals[c];
                    winner = best_ids[c];
                }
            }
            tokens[0] = invalid ? -1 : winner;
            return;
        }
        require(ynn::parallel_ranges(rows, 1, [&](std::size_t r_begin, std::size_t r_end) {
            ::ynn::argmax_rows_f32(r_end - r_begin, width, data + r_begin * width, tokens + r_begin);
        }));
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
        auto* dst = require(output.data<std::int8_t>()).data();
        const auto count = source.size();
        if (count < 32768 || ynn::thread_count() <= 1) {
            ::ynn::quantize_symmetric_int8_f32(count, scale_, source.data(), dst);
            return;
        }
        require(ynn::parallel_ranges(count, 4096, [&](std::size_t begin, std::size_t end) {
            ::ynn::quantize_symmetric_int8_f32(end - begin, scale_, source.data() + begin, dst + begin);
        }));
    }

    float scale_;
    OutputPool pool_;
};
/// Splits `rows` across the thread pool using dynamic ranges; small problems run on the calling thread.
template <typename Body>
auto for_rows(std::size_t rows, std::size_t row_work, Body&& body) -> void {
    if (rows == 0) return;
    if (rows == 1 || rows * row_work < 32768 || ynn::thread_count() <= 1) {
        body(0, rows);
        return;
    }
    require(ynn::parallel_ranges(rows, 1, std::forward<Body>(body)));
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
        const auto tiles = ::ynn::gated_activation_tile_count(rows_, width_);
        if (tiles <= 1 || rows_ * 2 * width_ < 8192 || ynn::thread_count() <= 1) {
            check(silu_ ? ::ynn::silu_gate_tiles_f32(rows_, width_, source, target, 0, tiles)
                        : ::ynn::gelu_gate_tiles_f32(rows_, width_, source, target, 0, tiles));
            return;
        }
        require(ynn::parallel_ranges(tiles, 1, [&](std::size_t begin, std::size_t end) {
            check(silu_ ? ::ynn::silu_gate_tiles_f32(rows_, width_, source, target, begin, end)
                        : ::ynn::gelu_gate_tiles_f32(rows_, width_, source, target, begin, end));
        }));
    }

    std::size_t rows_, width_;
    bool silu_;
    std::vector<std::int64_t> shape_;
    OutputPool pool_;
};

class GatedMultiplyOp final : public Operator {
public:
    GatedMultiplyOp(const Tensor& gate, bool silu, tensor::Arena& arena)
        : elements_(gate.numel()), silu_(silu), shape_(gate.shape().begin(), gate.shape().end()), pool_(&arena) {}

    static auto supports(TensorInputs inputs) noexcept -> bool {
        return inputs.size() == 2 && inputs[0].dtype() == DType::F32 && inputs[1].dtype() == DType::F32 &&
               inputs[0].is_contiguous() && inputs[1].is_contiguous() &&
               std::ranges::equal(inputs[0].shape(), inputs[1].shape());
    }

    auto run(TensorInputs inputs) -> Tensor override {
        auto output = pool_.acquire(shape_, DType::F32, tensor::Device::cpu());
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
        const auto* gate = require(inputs[0].data<float>()).data();
        const auto* up = require(inputs[1].data<float>()).data();
        auto* target = require(output.data<float>()).data();
        const auto tiles = ::ynn::gated_multiply_tile_count(elements_);
        if (tiles <= 1 || elements_ < 4096 || ynn::thread_count() <= 1) {
            check(silu_ ? ::ynn::silu_multiply_tiles_f32(elements_, gate, up, target, 0, tiles)
                        : ::ynn::gelu_multiply_tiles_f32(elements_, gate, up, target, 0, tiles));
            return;
        }
        require(ynn::parallel_ranges(tiles, 1, [&](std::size_t begin, std::size_t end) {
            check(silu_ ? ::ynn::silu_multiply_tiles_f32(elements_, gate, up, target, begin, end)
                        : ::ynn::gelu_multiply_tiles_f32(elements_, gate, up, target, begin, end));
        }));
    }

    std::size_t elements_;
    bool silu_;
    std::vector<std::int64_t> shape_;
    OutputPool pool_;
};

class ElementwiseBinaryF32 final : public Operator {
public:
    ElementwiseBinaryF32(TensorInputs inputs, bool is_multiply, tensor::Arena& arena)
        : is_multiply_(is_multiply), pool_(&arena) {
        swap_operands_ = inputs[0].numel() == 1 && inputs[1].numel() > 1;
        const auto& primary = swap_operands_ ? inputs[1] : inputs[0];
        const auto& secondary = swap_operands_ ? inputs[0] : inputs[1];
        shape_.assign(primary.shape().begin(), primary.shape().end());
        elements_ = primary.numel();
        rhs_stride_ = secondary.numel() == 1 ? 0 : secondary.numel();
    }

    static auto supports(TensorInputs inputs) noexcept -> bool {
        if (inputs.size() != 2 || inputs[0].dtype() != DType::F32 || inputs[1].dtype() != DType::F32 ||
            !inputs[0].is_contiguous() || !inputs[1].is_contiguous())
            return false;
        if (std::ranges::equal(inputs[0].shape(), inputs[1].shape())) return true;
        if (inputs[1].numel() == 1 && inputs[0].dimensions() >= inputs[1].dimensions()) return true;
        if (inputs[0].numel() == 1 && inputs[1].dimensions() >= inputs[0].dimensions()) return true;
        if (inputs[1].dimensions() == 1 && inputs[0].dimensions() >= 1 && inputs[0].size(-1) == inputs[1].size(0))
            return true;
        return false;
    }

    auto run(TensorInputs inputs) -> Tensor override {
        auto output = pool_.acquire(shape_, DType::F32, tensor::Device::cpu());
        apply(inputs, output);
        return output;
    }
    auto run_(TensorInputs inputs, Tensor& destination) -> Tensor override {
        if (destination.dtype() != DType::F32 || !std::ranges::equal(destination.shape(), shape_))
            throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "in-place operation cannot change shape or dtype"});
        apply(inputs, destination);
        return destination;
    }
    auto run_into(TensorInputs inputs, std::span<Tensor> outputs) -> void override { apply(inputs, outputs[0]); }
    auto allocations() const -> AllocationStats override { return pool_.allocations(); }

private:
    auto apply(TensorInputs inputs, Tensor& output) const -> void {
        if (elements_ == 0) return;
        const auto* lhs = require(inputs[swap_operands_ ? 1 : 0].data<float>()).data();
        const auto* rhs = require(inputs[swap_operands_ ? 0 : 1].data<float>()).data();
        auto* dst = require(output.data<float>()).data();
        if (elements_ < 32768 || ynn::thread_count() <= 1) {
            ::ynn::elementwise_binary_f32(is_multiply_, elements_, rhs_stride_, lhs, rhs, dst, 0, elements_);
            return;
        }
        require(ynn::parallel_ranges(elements_, 8192, [&](std::size_t begin, std::size_t end) {
            ::ynn::elementwise_binary_f32(is_multiply_, elements_, rhs_stride_, lhs, rhs, dst, begin, end);
        }));
    }

    bool is_multiply_;
    bool swap_operands_ = false;
    std::size_t elements_ = 0;
    std::size_t rhs_stride_ = 0;
    std::vector<std::int64_t> shape_;
    OutputPool pool_;
};

class SliceOp final : public Operator {
public:
    SliceOp(const OperatorSpec& spec, const Tensor& input, tensor::Arena& arena)
        : dtype_(input.dtype()), pool_(&arena) {
        const auto rank = input.dimensions();
        const auto raw_axis = spec.attributes[0];
        const auto axis =
            static_cast<std::size_t>(raw_axis < 0 ? raw_axis + static_cast<std::int64_t>(rank) : raw_axis);
        const auto start = static_cast<std::size_t>(spec.attributes[1]);
        const auto length = static_cast<std::size_t>(spec.attributes[2]);
        shape_.assign(input.shape().begin(), input.shape().end());
        shape_[axis] = static_cast<std::int64_t>(length);
        outer_ = 1;
        for (std::size_t d = 0; d < axis; ++d) outer_ *= static_cast<std::size_t>(input.shape()[d]);
        std::size_t inner = 1;
        for (std::size_t d = axis + 1; d < rank; ++d) inner *= static_cast<std::size_t>(input.shape()[d]);
        const auto elem_bytes = tensor::element_size(dtype_);
        src_stride_bytes_ = static_cast<std::size_t>(input.shape()[axis]) * inner * elem_bytes;
        dst_stride_bytes_ = length * inner * elem_bytes;
        offset_bytes_ = start * inner * elem_bytes;
    }

    static auto supports(const OperatorSpec& spec, TensorInputs inputs) noexcept -> bool {
        if (spec.operation != Operation::SLICE || inputs.size() != 1 || spec.attributes.size() < 3 ||
            !inputs[0].is_contiguous() || inputs[0].dimensions() == 0)
            return false;
        const auto rank = static_cast<std::int64_t>(inputs[0].dimensions());
        const auto axis = spec.attributes[0] < 0 ? spec.attributes[0] + rank : spec.attributes[0];
        if (axis < 0 || axis >= rank) return false;
        const auto start = spec.attributes[1], length = spec.attributes[2];
        const auto extent = static_cast<std::int64_t>(inputs[0].size(axis));
        return start >= 0 && length >= 0 && start <= extent && length <= extent - start;
    }

    auto run(TensorInputs inputs) -> Tensor override {
        auto output = pool_.acquire(shape_, dtype_, tensor::Device::cpu());
        apply(inputs[0], output);
        return output;
    }
    auto run_(TensorInputs, Tensor&) -> Tensor override {
        throw ops::Failure({ErrorCode::UNSUPPORTED, "slice cannot run in place"});
    }
    auto run_into(TensorInputs inputs, std::span<Tensor> outputs) -> void override { apply(inputs[0], outputs[0]); }
    auto allocations() const -> AllocationStats override { return pool_.allocations(); }

private:
    auto apply(const Tensor& input, Tensor& output) const -> void {
        if (outer_ == 0 || dst_stride_bytes_ == 0) return;
        const auto* src = require(input.host_bytes()).data();
        auto* dst = require(output.host_bytes()).data();
        if (outer_ <= 1 || outer_ * dst_stride_bytes_ < 65536 || ynn::thread_count() <= 1) {
            ::ynn::slice_blocks(outer_, src_stride_bytes_, dst_stride_bytes_, offset_bytes_, src, dst, 0, outer_);
            return;
        }
        require(ynn::parallel_ranges(outer_, 1, [&](std::size_t begin, std::size_t end) {
            ::ynn::slice_blocks(outer_, src_stride_bytes_, dst_stride_bytes_, offset_bytes_, src, dst, begin, end);
        }));
    }

    DType dtype_;
    std::size_t outer_ = 0, src_stride_bytes_ = 0, dst_stride_bytes_ = 0, offset_bytes_ = 0;
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
          decode_rows_(spec.vector_projection),
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
        auto params = parameters();
        if (decode_rows_) {
            params.batch_size = 1;
            params.query_length = 1;
            params.key_batch_stride = 0;
            params.mask_batch_stride = 0;
            params.mask_query_stride = 0;
        }
        work_items_ =
            decode_rows_ ? batch_size_ * query_length_ * heads_ : ::ynn::quantized_attention_f32_work_items(params);
        tasks_ = std::min(ynn::thread_count(), work_items_);
        task_scratch_bytes_ = ::ynn::quantized_attention_f32_scratch_size(params);
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
        const auto active_threads =
            (work_items_ <= 1 || work_items_ * key_length_ * head_dim_ <= 16384) ? std::size_t{1} : tasks_;
        auto decode = params;
        decode.batch_size = 1;
        decode.query_length = 1;
        decode.key_batch_stride = 0;
        decode.mask_batch_stride = 0;
        decode.mask_query_stride = 0;
        const auto query_stride = heads_ * head_dim_;
        const auto run_range = [&](void* scratch, std::size_t begin, std::size_t end) {
            if (!decode_rows_ || query_length_ == 1) {
                ::ynn::quantized_attention_f32(params, query_data.data(), key_data.data(), value_data.data(),
                                               mask_data.data(), output_data.data(), begin, end, scratch);
                return;
            }
            for (auto work = begin; work < end; ++work) {
                const auto query_row = work / heads_;
                const auto query_head = work % heads_;
                const auto batch = query_row / query_length_;
                const auto query_index = query_row % query_length_;
                const auto key_offset = key_batch_stride_ ? batch * key_batch_stride_ : 0;
                const auto mask_offset = batch * mask_batch_stride_ + query_index * mask_query_stride_;
                ::ynn::quantized_attention_f32(
                    decode, query_data.data() + query_row * query_stride, key_data.data() + key_offset,
                    value_data.data() + key_offset, mask_data.data() + mask_offset,
                    output_data.data() + query_row * query_stride, query_head, query_head + 1, scratch);
            }
        };
        if (active_threads <= 1) {
            if (fallback_scratch_.size() < task_scratch_bytes_) fallback_scratch_.resize(task_scratch_bytes_);
            run_range(fallback_scratch_.data(), 0, work_items_);
            return;
        }
        require(ynn::parallel_ranges_with_worker(work_items_, 1, task_scratch_bytes_,
                                                 [&](const ynn::WorkerContext& worker, std::size_t begin,
                                                     std::size_t end) { run_range(worker.scratch, begin, end); }));
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
    bool decode_rows_;
    std::vector<float> key_scales_, value_scales_;
    std::vector<std::int32_t> key_zero_points_, value_zero_points_;
    std::size_t key_block_size_, value_block_size_;
    std::size_t work_items_, tasks_, task_scratch_bytes_;
    std::vector<std::byte> fallback_scratch_;
    OutputPool pool_;
};
/// Direct FP32/BF16 attention over `[batch, queries, heads * head_dim]` and `[batch_or_1, total_keys, kv_heads *
/// head_dim]`, supporting MHA and GQA with optional slice `[key_start, key_length)` and broadcast mask.
class DirectAttentionOp final : public Operator {
public:
    static auto supports(const OperatorSpec& spec, TensorInputs inputs) -> bool {
        if (spec.operation != Operation::ATTENTION || (inputs.size() != 3 && inputs.size() != 4)) return false;
        if (spec.attributes.size() != 1 && spec.attributes.size() < 2) return false;
        if (!spec.quantization[0].scales.empty() || !spec.quantization[1].scales.empty()) return false;
        const auto heads = static_cast<std::size_t>(spec.attributes[0]);
        const auto key_heads = spec.attributes.size() >= 2 ? static_cast<std::size_t>(spec.attributes[1]) : heads;
        if (!heads || !key_heads || heads % key_heads != 0) return false;
        for (std::size_t index = 0; index < inputs.size(); ++index) {
            const auto dt = inputs[index].dtype();
            if ((dt != DType::F32 && dt != DType::BF16) || !inputs[index].is_contiguous()) return false;
        }
        return true;
    }
    DirectAttentionOp(const OperatorSpec& spec, tensor::Arena& arena)
        : heads_(static_cast<std::size_t>(spec.attributes[0])),
          key_heads_(spec.attributes.size() >= 2 ? static_cast<std::size_t>(spec.attributes[1]) : heads_),
          key_start_(spec.attributes.size() >= 4 ? static_cast<std::size_t>(spec.attributes[2]) : 0),
          key_length_(spec.attributes.size() >= 4 ? static_cast<std::size_t>(spec.attributes[3]) : 0),
          explicit_scale_(spec.attributes.size() >= 2),
          scale_(spec.epsilon),
          out_dtype_(spec.dtype == DType::BF16 ? DType::BF16 : DType::F32),
          pool_(&arena) {}
    auto run(TensorInputs inputs) -> Tensor override {
        auto output = pool_.acquire(inputs[0].shape(), out_dtype_, tensor::Device::cpu());
        attend(inputs, output);
        return output;
    }
    auto run_(TensorInputs, Tensor&) -> Tensor override {
        throw ops::Failure({ErrorCode::UNSUPPORTED, "attention is not an in-place operation"});
    }
    auto run_into(TensorInputs inputs, std::span<Tensor> outputs) -> void override { attend(inputs, outputs[0]); }
    auto allocations() const -> AllocationStats override { return pool_.allocations(); }

private:
    static auto ensure_f32(const Tensor& tensor, std::vector<float>& scratch) -> const float* {
        if (tensor.dtype() == DType::F32) return require(tensor.data<float>()).data();
        const auto count = tensor.numel();
        if (scratch.size() < count) scratch.resize(count);
        const auto raw = require(tensor.host_bytes());
        ::ynn::convert_bf16_to_f32_rows(1, count, reinterpret_cast<const std::uint16_t*>(raw.data()), count,
                                        scratch.data(), count, 0, 1);
        return scratch.data();
    }

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
            mask = ensure_f32(inputs[3], mask_f32_);
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
        const auto* query_data = ensure_f32(query, q_f32_);
        const auto* key_data = ensure_f32(inputs[1], k_f32_);
        const auto* value_data = ensure_f32(inputs[2], v_f32_);
        float* output_data = nullptr;
        if (out_dtype_ == DType::F32) {
            output_data = require(output.data<float>()).data();
        } else {
            if (out_f32_.size() < query.numel()) out_f32_.resize(query.numel());
            output_data = out_f32_.data();
        }
        const auto total_q_heads = batch * rows * heads_;
        if (rows > 17 && keys >= 8) {
            ::ynn::attention_prefill_f32(params, query_data, key_data, value_data, mask, output_data);
        } else {
            const auto scratch_floats = ::ynn::attention_f32_scratch_size(params);
            const auto scratch_bytes = scratch_floats * sizeof(float);
            if (items == 1 || total_q_heads * keys * head_width <= 32768 || ynn::thread_count() <= 1) {
                if (fallback_scratch_.size() < scratch_floats) fallback_scratch_.resize(scratch_floats);
                ::ynn::attention_f32(params, query_data, key_data, value_data, mask, output_data, 0, items,
                                     fallback_scratch_.data());
            } else {
                require(ynn::parallel_ranges_with_worker(
                    items, 1, scratch_bytes, [&](const ynn::WorkerContext& worker, std::size_t begin, std::size_t end) {
                        ::ynn::attention_f32(params, query_data, key_data, value_data, mask, output_data, begin, end,
                                             reinterpret_cast<float*>(worker.scratch));
                    }));
            }
        }
        if (out_dtype_ == DType::BF16) {
            auto raw = require(output.host_bytes());
            ::ynn::convert_f32_to_bf16_rows(1, query.numel(), output_data, query.numel(),
                                            reinterpret_cast<std::uint16_t*>(raw.data()), query.numel(), 0, 1);
        }
    }

    std::size_t heads_, key_heads_, key_start_, key_length_;
    bool explicit_scale_;
    float scale_;
    DType out_dtype_;
    std::vector<float> q_f32_, k_f32_, v_f32_, mask_f32_, out_f32_, fallback_scratch_;
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
    bool fp32_symmetric = false;
    auto operator<=>(const PackedTileKey&) const = default;
};

class BlockwiseLinear final : public Operator {
public:
    static auto supports(const OperatorSpec& spec, TensorInputs inputs) noexcept -> bool {
        if (spec.operation != Operation::PACKED_LINEAR || spec.dynamic_parameters ||
            (inputs.size() != 3 && inputs.size() != 4) || spec.attributes.size() < 3 || spec.epsilon > 0.F)
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
            require(ynn::parallel_tasks(workers, [&](std::size_t task) {
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
        if (act_q_.size() < M * K_) act_q_.resize(M * K_);
        if (act_scale_.size() < M * B_) act_scale_.resize(M * B_);
        const bool use_bsum = has_zero_ || bits_ == 8;
        if (use_bsum && act_bsum_.size() < M * B_) act_bsum_.resize(M * B_);
        float* bsum_ptr = use_bsum ? act_bsum_.data() : nullptr;

        const auto total_blocks = M * B_;
        const auto tiles = ::ynn::blockwise_linear_tile_count(N_);
        const auto threads = ynn::thread_count();
        if (threads <= 1 || tiles <= 1 || (M == 1 && K_ * N_ <= 64 * 1024)) {
            ::ynn::quantize_blockwise_blocks_f32(bits_, M, K_, block_size_, use_bsum, x, act_q_.data(),
                                                 act_scale_.data(), bsum_ptr, 0, total_blocks);
            ::ynn::blockwise_linear_f32(bits_, M, N_, K_, block_size_, has_zero_, output_scale_, packed_ptr_,
                                        act_q_.data(), act_scale_.data(), bsum_ptr, y, 0, tiles);
            return;
        }

        const auto min_tile_grain =
            tiles >= threads * 16 ? std::size_t{4} : (tiles >= threads * 8 ? std::size_t{2} : std::size_t{1});
        if (M * K_ < 8192 || total_blocks < 4) {
            ::ynn::quantize_blockwise_blocks_f32(bits_, M, K_, block_size_, use_bsum, x, act_q_.data(),
                                                 act_scale_.data(), bsum_ptr, 0, total_blocks);
            require(ynn::parallel_ranges(tiles, min_tile_grain, [&](std::size_t t_begin, std::size_t t_end) {
                ::ynn::blockwise_linear_f32(bits_, M, N_, K_, block_size_, has_zero_, output_scale_, packed_ptr_,
                                            act_q_.data(), act_scale_.data(), bsum_ptr, y, t_begin, t_end);
            }));
            return;
        }

        // Single-launch team execution with PhaseBarrier and cache-aligned home slots + stealing:
        // Phase 1: each worker quantizes its deterministic block range, then steals stragglers.
        // Barrier: single sense-reversing PhaseBarrier (no second thread-pool dispatch).
        // Phase 2: each worker computes its home output-tile range (preserving L2 weight affinity), then steals.
        const auto workers = std::min(threads, tiles);
        const auto q_grain = std::max<std::size_t>(1, total_blocks / (workers * 4));
        const auto t_grain = std::max<std::size_t>(min_tile_grain, tiles / (workers * 8));
        ynn::ParallelRangeSlots q_slots(total_blocks, workers), t_slots(tiles, workers);
        require(ynn::parallel_team(workers, 0, [&](const ynn::WorkerContext& worker, ynn::PhaseBarrier& barrier) {
            drain_range_slots(q_slots, worker.tid, q_grain, [&](std::size_t b_begin, std::size_t b_end) {
                ::ynn::quantize_blockwise_blocks_f32(bits_, M, K_, block_size_, use_bsum, x, act_q_.data(),
                                                     act_scale_.data(), bsum_ptr, b_begin, b_end);
            });
            if (!barrier.arrive_and_wait()) return;
            drain_range_slots(t_slots, worker.tid, t_grain, [&](std::size_t t_begin, std::size_t t_end) {
                ::ynn::blockwise_linear_f32(bits_, M, N_, K_, block_size_, has_zero_, output_scale_, packed_ptr_,
                                            act_q_.data(), act_scale_.data(), bsum_ptr, y, t_begin, t_end);
            });
        }));
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

class SymmetricPackedLinear final : public Operator {
public:
    static auto supports(const OperatorSpec& spec, TensorInputs inputs) noexcept -> bool {
        if (spec.operation != Operation::PACKED_LINEAR || spec.dynamic_parameters || inputs.size() != 3 ||
            spec.attributes.size() < 3 || spec.epsilon <= 0.F)
            return false;
        const auto bits = static_cast<std::int32_t>(spec.attributes[0]);
        if (!::ynn::is_symmetric_linear_supported(bits)) return false;
        const auto block_size = static_cast<std::size_t>(spec.attributes[1]);
        const auto values_per_byte = static_cast<std::size_t>(8 / bits);
        const auto in_features = static_cast<std::size_t>(inputs[0].size(-1));
        if (block_size == 0 || in_features == 0 || (in_features % block_size != 0) ||
            (block_size % values_per_byte != 0))
            return false;
        if (inputs[0].dtype() != DType::F32 || !inputs[0].is_contiguous() || !inputs[1].is_contiguous() ||
            !inputs[2].is_contiguous())
            return false;
        if (inputs[1].dimensions() != 2 || inputs[1].size(1) * values_per_byte != in_features) return false;
        const bool valid_weight = inputs[1].dtype() == DType::U8 || (bits == 8 && inputs[1].dtype() == DType::I8);
        const bool valid_scale = inputs[2].dtype() == DType::F32 || inputs[2].dtype() == DType::F16;
        return valid_weight && valid_scale && inputs[2].numel() == inputs[1].size(0) * (in_features / block_size);
    }

    static auto pack_tiles(std::int32_t bits, const Tensor& weight, const Tensor& scales, std::size_t block_size)
        -> Tensor {
        const auto values_per_byte = static_cast<std::size_t>(8 / bits);
        const auto out_features = static_cast<std::size_t>(weight.size(0));
        const auto in_features = static_cast<std::size_t>(weight.size(1)) * values_per_byte;
        const auto total_bytes = ::ynn::packed_symmetric_linear_size(bits, out_features, in_features, block_size);
        auto packed = require(Tensor::empty({static_cast<std::int64_t>(total_bytes)}, DType::U8));
        const auto* w_bytes = require(std::as_const(weight).host_bytes()).data();
        const auto* s_bytes = require(std::as_const(scales).host_bytes()).data();
        const bool s_f16 = scales.dtype() == DType::F16;
        auto* dst = require(packed.host_bytes()).data();
        const auto tiles = ::ynn::blockwise_linear_tile_count(out_features);
        const auto workers = std::min(ynn::thread_count(), tiles);
        if (workers > 1 && tiles >= 8) {
            const auto per_worker = (tiles + workers - 1) / workers;
            require(ynn::parallel_tasks(workers, [&](std::size_t task) {
                const auto t_begin = task * per_worker, t_end = std::min(tiles, t_begin + per_worker);
                if (t_begin < t_end)
                    ::ynn::pack_symmetric_linear_tiles(bits, out_features, in_features, 0, block_size, w_bytes, s_bytes,
                                                       s_f16, t_begin, t_end, dst);
            }));
        } else {
            ::ynn::pack_symmetric_linear_tiles(bits, out_features, in_features, 0, block_size, w_bytes, s_bytes, s_f16,
                                               0, tiles, dst);
        }
        return packed;
    }

    SymmetricPackedLinear(const OperatorSpec& spec, const Tensor& sample_input, const Tensor& weight,
                          Tensor packed_tiles, tensor::Arena& arena)
        : packed_(std::move(packed_tiles)),
          packed_ptr_(require(std::as_const(packed_).host_bytes()).data()),
          bits_(static_cast<std::int32_t>(spec.attributes[0])),
          out_features_(static_cast<std::size_t>(weight.size(0))),
          in_features_(static_cast<std::size_t>(sample_input.size(-1))),
          block_size_(static_cast<std::size_t>(spec.attributes[1])),
          blocks_(in_features_ / block_size_),
          padded_block_(::ynn::symmetric_linear_padded_block(block_size_)),
          input_scale_(spec.epsilon > 0.F ? spec.epsilon : 0.F),
          output_scale_(std::bit_cast<float>(static_cast<std::int32_t>(spec.attributes[2]))),
          pool_(&arena) {}

    auto run(TensorInputs inputs) -> Tensor override {
        std::vector<std::int64_t> out_shape(inputs[0].shape().begin(), inputs[0].shape().end());
        out_shape.back() = static_cast<std::int64_t>(out_features_);
        auto output = pool_.acquire(out_shape, DType::F32, tensor::Device::cpu());
        apply(inputs[0], output);
        return output;
    }
    auto run_(TensorInputs, Tensor&) -> Tensor override {
        throw ops::Failure({ErrorCode::UNSUPPORTED, "packed linear cannot run in place"});
    }
    auto run_into(TensorInputs inputs, std::span<Tensor> outputs) -> void override { apply(inputs[0], outputs.back()); }
    auto allocations() const -> AllocationStats override { return pool_.allocations(); }

private:
    auto apply(const Tensor& input, Tensor& output) -> void {
        const auto rows = input.numel() / in_features_;
        if (rows == 0) return;
        const auto* x = require(input.data<float>()).data();
        auto* y = require(output.data<float>()).data();
        const auto total_blocks = rows * blocks_;
        if (act_q_.size() < total_blocks * padded_block_) act_q_.resize(total_blocks * padded_block_);
        if (act_scale_.size() < total_blocks) act_scale_.resize(total_blocks);

        const auto tiles = ::ynn::blockwise_linear_tile_count(out_features_);
        const auto threads = ynn::thread_count();
        if (threads <= 1 || tiles <= 1 || (rows == 1 && in_features_ * out_features_ <= 64 * 1024)) {
            ::ynn::quantize_symmetric_blocks_f32(bits_, rows, in_features_, block_size_, input_scale_, x, act_q_.data(),
                                                 act_scale_.data(), 0, total_blocks);
            ::ynn::symmetric_linear_f32(bits_, rows, out_features_, in_features_, block_size_, output_scale_,
                                        packed_ptr_, act_q_.data(), act_scale_.data(), y, out_features_, 0, tiles);
            return;
        }

        const auto min_tile_grain = tiles >= threads * 8 ? std::size_t{2} : std::size_t{1};
        if (rows * in_features_ < 8192 || total_blocks < 4) {
            ::ynn::quantize_symmetric_blocks_f32(bits_, rows, in_features_, block_size_, input_scale_, x, act_q_.data(),
                                                 act_scale_.data(), 0, total_blocks);
            require(ynn::parallel_ranges(tiles, min_tile_grain, [&](std::size_t t_begin, std::size_t t_end) {
                ::ynn::symmetric_linear_f32(bits_, rows, out_features_, in_features_, block_size_, output_scale_,
                                            packed_ptr_, act_q_.data(), act_scale_.data(), y, out_features_, t_begin,
                                            t_end);
            }));
            return;
        }

        const auto workers = std::min(threads, tiles);
        const auto q_grain = std::max<std::size_t>(1, total_blocks / (workers * 4));
        const auto t_grain = std::max<std::size_t>(min_tile_grain, tiles / (workers * 8));
        ynn::ParallelRangeSlots q_slots(total_blocks, workers), t_slots(tiles, workers);
        require(ynn::parallel_team(workers, 0, [&](const ynn::WorkerContext& worker, ynn::PhaseBarrier& barrier) {
            drain_range_slots(q_slots, worker.tid, q_grain, [&](std::size_t b_begin, std::size_t b_end) {
                ::ynn::quantize_symmetric_blocks_f32(bits_, rows, in_features_, block_size_, input_scale_, x,
                                                     act_q_.data(), act_scale_.data(), b_begin, b_end);
            });
            if (!barrier.arrive_and_wait()) return;
            drain_range_slots(t_slots, worker.tid, t_grain, [&](std::size_t t_begin, std::size_t t_end) {
                ::ynn::symmetric_linear_f32(bits_, rows, out_features_, in_features_, block_size_, output_scale_,
                                            packed_ptr_, act_q_.data(), act_scale_.data(), y, out_features_, t_begin,
                                            t_end);
            });
        }));
    }

    Tensor packed_;
    const void* packed_ptr_;
    std::int32_t bits_;
    std::size_t out_features_, in_features_, block_size_, blocks_, padded_block_;
    float input_scale_, output_scale_;
    OutputPool pool_;
    std::vector<std::int8_t> act_q_;
    std::vector<float> act_scale_;
};

class PackedFeedForward final : public Operator {
public:
    static auto supports(const OperatorSpec& spec, TensorInputs inputs) noexcept -> bool {
        if (spec.operation != Operation::GATED_FEED_FORWARD || spec.dynamic_parameters || inputs.size() != 5 ||
            spec.attributes.size() < 6 || spec.epsilon <= 0.F)
            return false;
        const auto bits = static_cast<std::int32_t>(spec.attributes[0]);
        if (!::ynn::is_symmetric_linear_supported(bits)) return false;
        const auto input_size = static_cast<std::size_t>(spec.attributes[1]);
        const auto intermediate = static_cast<std::size_t>(spec.attributes[2]);
        const auto values_per_byte = static_cast<std::size_t>(8 / bits);
        return input_size > 0 && intermediate > 0 && (input_size % values_per_byte == 0) &&
               (intermediate % values_per_byte == 0) && inputs[0].dtype() == DType::F32 && inputs[0].is_contiguous() &&
               inputs[1].is_contiguous() && inputs[2].is_contiguous() && inputs[3].is_contiguous() &&
               inputs[4].is_contiguous() && inputs[2].dtype() == DType::F32 && inputs[4].dtype() == DType::F32;
    }

    PackedFeedForward(const OperatorSpec& spec, TensorInputs inputs, ynn::FeedForwardCache& cache, tensor::Arena& arena)
        : bits_(static_cast<std::int32_t>(spec.attributes[0])),
          input_size_(static_cast<std::size_t>(spec.attributes[1])),
          intermediate_(static_cast<std::size_t>(spec.attributes[2])),
          in_padded_(::ynn::symmetric_linear_padded_block(input_size_)),
          mid_padded_(::ynn::symmetric_linear_padded_block(intermediate_)),
          gate_input_scale_(spec.epsilon),
          gate_output_scale_(std::bit_cast<float>(static_cast<std::int32_t>(spec.attributes[3]))),
          down_input_scale_(std::bit_cast<float>(static_cast<std::int32_t>(spec.attributes[4]))),
          down_output_scale_(std::bit_cast<float>(static_cast<std::int32_t>(spec.attributes[5]))),
          silu_(spec.attributes.size() >= 7 && spec.attributes[6] != 0),
          pool_(&arena) {
        const auto gate_bytes = ::ynn::packed_symmetric_linear_size(bits_, intermediate_, input_size_, input_size_);
        const auto up_bytes = gate_bytes;
        const auto down_bytes = ::ynn::packed_symmetric_linear_size(bits_, input_size_, intermediate_, intermediate_);
        const auto payload_size = gate_bytes + up_bytes + down_bytes;
        const bool lookup_cache = cache.enabled();
        const auto source_hash = lookup_cache ? hash_tensor(inputs[3], hash_tensor(inputs[1])) : 0;
        const ynn::FeedForwardCacheSpec cache_spec{.bits = static_cast<std::uint32_t>(bits_),
                                                   .input_size = input_size_,
                                                   .intermediate_size = intermediate_,
                                                   .source_hash = source_hash};
        if (lookup_cache) {
            if (auto cached = cache.load(cache_spec, payload_size)) {
                owner_ = std::move(cached.owner);
                gate_ptr_ = cached.bytes.data();
                up_ptr_ = gate_ptr_ + gate_bytes;
                down_ptr_ = up_ptr_ + up_bytes;
                return;
            }
        }
        packed_storage_ = require(Tensor::empty({static_cast<std::int64_t>(payload_size)}, DType::U8));
        auto* base = reinterpret_cast<std::uint8_t*>(require(packed_storage_.host_bytes()).data());
        auto* gate_dst = base;
        auto* up_dst = gate_dst + gate_bytes;
        auto* down_dst = up_dst + up_bytes;
        const auto* gu_w = require(inputs[1].host_bytes()).data();
        const auto* gu_s = require(inputs[2].host_bytes()).data();
        const auto* d_w = require(inputs[3].host_bytes()).data();
        const auto* d_s = require(inputs[4].host_bytes()).data();
        const auto mid_tiles = ::ynn::blockwise_linear_tile_count(intermediate_);
        const auto out_tiles = ::ynn::blockwise_linear_tile_count(input_size_);
        const auto workers = std::min(ynn::thread_count(), std::max(mid_tiles, out_tiles));
        if (workers > 1 && (mid_tiles + out_tiles) >= 8) {
            require(ynn::parallel_tasks(workers, [&](std::size_t task) {
                const auto m_begin = (mid_tiles * task) / workers;
                const auto m_end = (mid_tiles * (task + 1)) / workers;
                if (m_begin < m_end) {
                    ::ynn::pack_symmetric_linear_tiles(bits_, intermediate_, input_size_, 0, input_size_, gu_w, gu_s,
                                                       false, m_begin, m_end, gate_dst);
                    ::ynn::pack_symmetric_linear_tiles(bits_, intermediate_, input_size_, intermediate_, input_size_,
                                                       gu_w, gu_s, false, m_begin, m_end, up_dst);
                }
                const auto o_begin = (out_tiles * task) / workers;
                const auto o_end = (out_tiles * (task + 1)) / workers;
                if (o_begin < o_end) {
                    ::ynn::pack_symmetric_linear_tiles(bits_, input_size_, intermediate_, 0, intermediate_, d_w, d_s,
                                                       false, o_begin, o_end, down_dst);
                }
            }));
        } else {
            ::ynn::pack_symmetric_linear_tiles(bits_, intermediate_, input_size_, 0, input_size_, gu_w, gu_s, false, 0,
                                               mid_tiles, gate_dst);
            ::ynn::pack_symmetric_linear_tiles(bits_, intermediate_, input_size_, intermediate_, input_size_, gu_w,
                                               gu_s, false, 0, mid_tiles, up_dst);
            ::ynn::pack_symmetric_linear_tiles(bits_, input_size_, intermediate_, 0, intermediate_, d_w, d_s, false, 0,
                                               out_tiles, down_dst);
        }
        if (cache.writable()) cache.store(cache_spec, {base, payload_size});
        gate_ptr_ = gate_dst;
        up_ptr_ = up_dst;
        down_ptr_ = down_dst;
    }

    auto run(TensorInputs inputs) -> Tensor override {
        auto output = pool_.acquire(inputs[0].shape(), DType::F32, tensor::Device::cpu());
        apply(inputs[0], output);
        return output;
    }
    auto run_(TensorInputs, Tensor&) -> Tensor override {
        throw ops::Failure({ErrorCode::UNSUPPORTED, "gated feed-forward cannot run in place"});
    }
    auto run_into(TensorInputs inputs, std::span<Tensor> outputs) -> void override { apply(inputs[0], outputs.back()); }
    auto allocations() const -> AllocationStats override { return pool_.allocations(); }

private:
    auto apply(const Tensor& input, Tensor& output) -> void {
        const auto rows = input.numel() / input_size_;
        if (rows == 0) return;
        const auto* x = require(input.data<float>()).data();
        auto* y = require(output.data<float>()).data();
        if (in_q_.size() < rows * in_padded_) in_q_.resize(rows * in_padded_);
        if (in_scale_.size() < rows) in_scale_.resize(rows, gate_input_scale_);
        std::ranges::fill_n(in_scale_.begin(), rows, gate_input_scale_);
        if (mid_q_.size() < rows * mid_padded_) mid_q_.resize(rows * mid_padded_);
        if (mid_scale_.size() < rows) mid_scale_.resize(rows, down_input_scale_);
        std::ranges::fill_n(mid_scale_.begin(), rows, down_input_scale_);

        const auto mid_tiles = ::ynn::blockwise_linear_tile_count(intermediate_);
        const auto out_tiles = ::ynn::blockwise_linear_tile_count(input_size_);
        const auto threads = ynn::thread_count();
        if (threads <= 1 || (mid_tiles <= 1 && out_tiles <= 1)) {
            ::ynn::quantize_symmetric_blocks_f32(bits_, rows, input_size_, input_size_, gate_input_scale_, x,
                                                 in_q_.data(), in_scale_.data(), 0, rows);
            check(::ynn::packed_ffn_mid_tiles_f32(bits_, rows, input_size_, intermediate_, gate_output_scale_,
                                                  down_input_scale_, silu_, gate_ptr_, up_ptr_, in_q_.data(),
                                                  in_scale_.data(), mid_q_.data(), 0, mid_tiles));
            ::ynn::symmetric_linear_f32(bits_, rows, input_size_, intermediate_, intermediate_, down_output_scale_,
                                        down_ptr_, mid_q_.data(), mid_scale_.data(), y, input_size_, 0, out_tiles);
            return;
        }

        const auto workers = std::min(threads, std::max(mid_tiles, out_tiles));
        const auto m_grain = std::max<std::size_t>(1, mid_tiles / (workers * 8));
        const auto o_grain = std::max<std::size_t>(1, out_tiles / (workers * 8));
        ynn::ParallelRangeSlots m_slots(mid_tiles, workers), o_slots(out_tiles, workers);
        std::atomic<ynn_status> ffn_status = ynn_status_success;

        if (rows * input_size_ < 8192 || rows < 2) {
            // Single-token decode / small input: quantize inline on the calling thread so the team pays only a single
            // PhaseBarrier between intermediate tiles and down-projection tiles.
            ::ynn::quantize_symmetric_blocks_f32(bits_, rows, input_size_, input_size_, gate_input_scale_, x,
                                                 in_q_.data(), in_scale_.data(), 0, rows);
            require(ynn::parallel_team(workers, 0, [&](const ynn::WorkerContext& worker, ynn::PhaseBarrier& barrier) {
                drain_range_slots(m_slots, worker.tid, m_grain, [&](std::size_t t_begin, std::size_t t_end) {
                    const auto st = ::ynn::packed_ffn_mid_tiles_f32(
                        bits_, rows, input_size_, intermediate_, gate_output_scale_, down_input_scale_, silu_,
                        gate_ptr_, up_ptr_, in_q_.data(), in_scale_.data(), mid_q_.data(), t_begin, t_end);
                    if (st != ynn_status_success) ffn_status.store(st, std::memory_order_relaxed);
                });
                if (!barrier.arrive_and_wait()) return;
                if (ffn_status.load(std::memory_order_relaxed) != ynn_status_success) return;
                drain_range_slots(o_slots, worker.tid, o_grain, [&](std::size_t t_begin, std::size_t t_end) {
                    ::ynn::symmetric_linear_f32(bits_, rows, input_size_, intermediate_, intermediate_,
                                                down_output_scale_, down_ptr_, mid_q_.data(), mid_scale_.data(), y,
                                                input_size_, t_begin, t_end);
                });
            }));
            check(ffn_status.load(std::memory_order_relaxed));
            return;
        }

        const auto q_grain = std::max<std::size_t>(1, rows / (workers * 4));
        ynn::ParallelRangeSlots q_slots(rows, workers);
        require(ynn::parallel_team(workers, 0, [&](const ynn::WorkerContext& worker, ynn::PhaseBarrier& barrier) {
            drain_range_slots(q_slots, worker.tid, q_grain, [&](std::size_t r_begin, std::size_t r_end) {
                ::ynn::quantize_symmetric_blocks_f32(bits_, rows, input_size_, input_size_, gate_input_scale_, x,
                                                     in_q_.data(), in_scale_.data(), r_begin, r_end);
            });
            if (!barrier.arrive_and_wait()) return;
            drain_range_slots(m_slots, worker.tid, m_grain, [&](std::size_t t_begin, std::size_t t_end) {
                const auto st = ::ynn::packed_ffn_mid_tiles_f32(
                    bits_, rows, input_size_, intermediate_, gate_output_scale_, down_input_scale_, silu_, gate_ptr_,
                    up_ptr_, in_q_.data(), in_scale_.data(), mid_q_.data(), t_begin, t_end);
                if (st != ynn_status_success) ffn_status.store(st, std::memory_order_relaxed);
            });
            if (!barrier.arrive_and_wait()) return;
            if (ffn_status.load(std::memory_order_relaxed) != ynn_status_success) return;
            drain_range_slots(o_slots, worker.tid, o_grain, [&](std::size_t t_begin, std::size_t t_end) {
                ::ynn::symmetric_linear_f32(bits_, rows, input_size_, intermediate_, intermediate_, down_output_scale_,
                                            down_ptr_, mid_q_.data(), mid_scale_.data(), y, input_size_, t_begin,
                                            t_end);
            });
        }));
        check(ffn_status.load(std::memory_order_relaxed));
    }

    Tensor packed_storage_;
    std::shared_ptr<const void> owner_;
    const std::uint8_t* gate_ptr_ = nullptr;
    const std::uint8_t* up_ptr_ = nullptr;
    const std::uint8_t* down_ptr_ = nullptr;
    std::int32_t bits_;
    std::size_t input_size_, intermediate_, in_padded_, mid_padded_;
    float gate_input_scale_, gate_output_scale_, down_input_scale_, down_output_scale_;
    bool silu_;
    OutputPool pool_;
    std::vector<std::int8_t> in_q_, mid_q_;
    std::vector<float> in_scale_, mid_scale_;
};
struct DenseTileKey {
    const void* weight_id = nullptr;
    std::int64_t weight_offset = 0;
    std::size_t n = 0;
    std::size_t k = 0;
    bool transpose_b = false;
    DType b_dtype = DType::F32;
    bool quantized_i8 = false;
    auto operator<=>(const DenseTileKey&) const = default;
};

struct PackedDenseEntry {
    ::ynn::dense_linear_plan plan;
    Tensor packed;
    Tensor col_sums;
};

auto prepare_f32_bias(const Tensor& bias, std::size_t n, std::vector<float>& dst) -> void {
    dst.resize(n);
    if (bias.dtype() == DType::F32) {
        const auto src = require(std::as_const(bias).data<float>());
        if (src.size() == 1)
            std::ranges::fill(dst, src[0]);
        else
            std::ranges::copy(src, dst.begin());
    } else {
        const auto raw = require(std::as_const(bias).host_bytes());
        const auto* src = reinterpret_cast<const std::uint16_t*>(raw.data());
        if (bias.numel() == 1) {
            float val = 0.F;
            ::ynn::convert_bf16_to_f32_rows(1, 1, src, 1, &val, 1, 0, 1);
            std::ranges::fill(dst, val);
        } else {
            ::ynn::convert_bf16_to_f32_rows(1, n, src, n, dst.data(), n, 0, 1);
        }
    }
}

class DenseLinearOp final : public Operator {
public:
    static auto supports(const OperatorSpec& spec, TensorInputs inputs) noexcept -> bool {
        if ((spec.operation != Operation::LINEAR && spec.operation != Operation::MATMUL) ||
            (inputs.size() != 2 && inputs.size() != 3) || spec.attributes.empty())
            return false;
        if (spec.operation == Operation::MATMUL && inputs.size() != 2) return false;
        if (spec.dtype != DType::F32 && spec.dtype != DType::BF16) return false;
        const auto& a = inputs[0];
        const auto& b = inputs[1];
        if ((a.dtype() != DType::F32 && a.dtype() != DType::BF16) ||
            (b.dtype() != DType::F32 && b.dtype() != DType::BF16))
            return false;
        if (!a.is_contiguous() || !b.is_contiguous() || a.dimensions() < 2 || b.dimensions() < 2) return false;
        const bool transpose_b = spec.attributes[0] != 0;
        const auto k = static_cast<std::size_t>(a.size(-1));
        const auto b_k = static_cast<std::size_t>(transpose_b ? b.size(-1) : b.size(-2));
        const auto n = static_cast<std::size_t>(transpose_b ? b.size(-2) : b.size(-1));
        if (k == 0 || n == 0 || k != b_k) return false;
        const auto a_batch_dims = a.dimensions() - 2;
        const auto b_batch_dims = b.dimensions() - 2;
        const auto min_batch = std::min(a_batch_dims, b_batch_dims);
        for (std::size_t i = 0; i < min_batch; ++i) {
            const auto ad = a.shape()[a_batch_dims - 1 - i];
            const auto bd = b.shape()[b_batch_dims - 1 - i];
            if (ad != bd && ad != 1 && bd != 1) return false;
        }
        if (inputs.size() == 3) {
            const auto& bias = inputs[2];
            if ((bias.dtype() != DType::F32 && bias.dtype() != DType::BF16) || !bias.is_contiguous() ||
                (bias.numel() != n && bias.numel() != 1))
                return false;
        }
        return true;
    }

    static auto pack_static_b(const Tensor& b, bool transpose_b, const ::ynn::dense_linear_plan& plan) -> Tensor {
        auto packed = require(Tensor::empty({static_cast<std::int64_t>(plan.total_bytes)}, DType::U8));
        const auto* b_bytes = require(std::as_const(b).host_bytes()).data();
        auto* dst = require(packed.host_bytes()).data();
        const std::size_t elem_bytes = tensor::element_size(b.dtype());
        const std::size_t b_row_stride = (transpose_b ? plan.k : plan.n) * elem_bytes;
        const auto workers = std::min(ynn::thread_count(), plan.blocks_n);
        if (workers > 1 && plan.blocks_n >= 4) {
            const auto per_worker = (plan.blocks_n + workers - 1) / workers;
            require(ynn::parallel_tasks(workers, [&](std::size_t task) {
                const auto t_begin = task * per_worker;
                const auto t_end = std::min(plan.blocks_n, t_begin + per_worker);
                if (t_begin < t_end)
                    ::ynn::pack_dense_linear_tiles(plan, transpose_b, b_bytes, b_row_stride, t_begin, t_end, dst);
            }));
        } else {
            ::ynn::pack_dense_linear_tiles(plan, transpose_b, b_bytes, b_row_stride, 0, plan.blocks_n, dst);
        }
        return packed;
    }

    DenseLinearOp(const OperatorSpec& spec, TensorInputs inputs, PackedDenseEntry static_b, tensor::Arena& arena)
        : out_dtype_(spec.dtype),
          b_dtype_(inputs[1].dtype()),
          transpose_b_(spec.attributes[0] != 0),
          static_b_(static_b.packed.defined()),
          plan_(static_b.plan),
          packed_b_(std::move(static_b.packed)),
          pool_(&arena) {
        const auto& a = inputs[0];
        const auto& b = inputs[1];
        k_ = static_cast<std::size_t>(a.size(-1));
        n_ = static_cast<std::size_t>(transpose_b_ ? b.size(-2) : b.size(-1));
        m_ = static_cast<std::size_t>(a.size(-2));
        if (!static_b_) {
            const auto ynn_bt = b_dtype_ == DType::BF16 ? ynn_type_bf16 : ynn_type_fp32;
            check(::ynn::make_dense_linear_plan(ynn_bt, ynn_bt, ynn_type_fp32, n_, k_, &plan_));
        }

        const auto a_batch_dims = a.dimensions() - 2;
        const auto b_batch_dims = b.dimensions() - 2;
        const auto batch_dims = std::max(a_batch_dims, b_batch_dims);
        out_shape_.reserve(batch_dims + 2);
        a_batch_shape_.assign(batch_dims, 1);
        b_batch_shape_.assign(batch_dims, 1);
        for (std::size_t i = 0; i < batch_dims; ++i) {
            const auto ad = i >= batch_dims - a_batch_dims ? a.shape()[i - (batch_dims - a_batch_dims)] : 1;
            const auto bd = i >= batch_dims - b_batch_dims ? b.shape()[i - (batch_dims - b_batch_dims)] : 1;
            a_batch_shape_[i] = static_cast<std::size_t>(ad);
            b_batch_shape_[i] = static_cast<std::size_t>(bd);
            out_shape_.push_back(std::max(ad, bd));
            total_batches_ *= static_cast<std::size_t>(std::max(ad, bd));
        }
        out_shape_.push_back(a.size(-2));
        out_shape_.push_back(static_cast<std::int64_t>(n_));

        if (b_batch_dims == 0 || b.numel() == static_cast<std::int64_t>(k_ * n_)) {
            b_is_single_matrix_ = true;
            eff_m_ = total_batches_ * m_;
            total_batches_ = 1;
        } else {
            eff_m_ = m_;
            a_batch_strides_.assign(batch_dims, 0);
            b_batch_strides_.assign(batch_dims, 0);
            std::size_t a_stride = m_ * k_ * tensor::element_size(a.dtype());
            std::size_t b_stride = k_ * n_ * tensor::element_size(b.dtype());
            for (std::size_t i = batch_dims; i-- > 0;) {
                a_batch_strides_[i] = a_batch_shape_[i] > 1 ? a_stride : 0;
                b_batch_strides_[i] = b_batch_shape_[i] > 1 ? b_stride : 0;
                a_stride *= a_batch_shape_[i];
                b_stride *= b_batch_shape_[i];
            }
        }

        if (inputs.size() == 3 && !spec.dynamic_parameters) {
            prepare_f32_bias(inputs[2], n_, bias_f32_);
        }
    }

    auto run(TensorInputs inputs) -> Tensor override {
        auto output = pool_.acquire(out_shape_, out_dtype_, tensor::Device::cpu());
        apply(inputs, output);
        return output;
    }
    auto run_(TensorInputs, Tensor&) -> Tensor override {
        throw ops::Failure({ErrorCode::UNSUPPORTED, "dense linear cannot run in place"});
    }
    auto run_into(TensorInputs inputs, std::span<Tensor> outputs) -> void override { apply(inputs, outputs.back()); }
    auto allocations() const -> AllocationStats override { return pool_.allocations(); }

private:
    auto run_one_matrix(const void* a_ptr, DType a_dtype, const void* packed_b, const float* bias_ptr, float* out_f32)
        -> void {
        const std::size_t a_elem_bytes = tensor::element_size(b_dtype_);
        const void* eff_a = a_ptr;
        if (a_dtype != b_dtype_) {
            if (b_dtype_ == DType::BF16) {
                if (a_conv_.size() < eff_m_ * k_ * sizeof(std::uint16_t))
                    a_conv_.resize(eff_m_ * k_ * sizeof(std::uint16_t));
                ::ynn::convert_f32_to_bf16_rows(eff_m_, k_, static_cast<const float*>(a_ptr), k_,
                                                reinterpret_cast<std::uint16_t*>(a_conv_.data()), k_, 0, eff_m_);
            } else {
                if (a_conv_.size() < eff_m_ * k_ * sizeof(float)) a_conv_.resize(eff_m_ * k_ * sizeof(float));
                ::ynn::convert_bf16_to_f32_rows(eff_m_, k_, static_cast<const std::uint16_t*>(a_ptr), k_,
                                                reinterpret_cast<float*>(a_conv_.data()), k_, 0, eff_m_);
            }
            eff_a = a_conv_.data();
        }

        bool transposed_a = false;
        std::size_t tile_m = 1;
        std::size_t a_packed_bytes = 0;
        if (!::ynn::dense_linear_select_kernel(plan_, eff_m_, &transposed_a, &tile_m, &a_packed_bytes))
            throw ops::Failure({ErrorCode::UNSUPPORTED, "no matching YNNPACK dense dot kernel"});

        if (transposed_a) {
            if (a_packed_.size() < a_packed_bytes) a_packed_.resize(a_packed_bytes);
            ::ynn::dense_linear_pack_a(plan_, eff_m_, tile_m, eff_a, k_ * a_elem_bytes, a_packed_.data());
            eff_a = a_packed_.data();
        }

        const auto threads = ynn::thread_count();
        if (threads <= 1 || plan_.blocks_n <= 1 || (eff_m_ == 1 && k_ * n_ <= 64 * 1024)) {
            ::ynn::dense_linear_run_tiles(plan_, eff_m_, transposed_a, tile_m, eff_a, k_ * a_elem_bytes, packed_b,
                                          bias_ptr, out_f32, n_, 0, plan_.blocks_n);
            return;
        }
        require(ynn::parallel_ranges(plan_.blocks_n, 1, [&](std::size_t t_begin, std::size_t t_end) {
            ::ynn::dense_linear_run_tiles(plan_, eff_m_, transposed_a, tile_m, eff_a, k_ * a_elem_bytes, packed_b,
                                          bias_ptr, out_f32, n_, t_begin, t_end);
        }));
    }

    auto apply(TensorInputs inputs, Tensor& output) -> void {
        if (eff_m_ == 0 || n_ == 0 || k_ == 0) return;
        const auto* a_base = require(inputs[0].host_bytes()).data();
        auto* out_raw = require(output.host_bytes()).data();
        const float* bias_ptr = nullptr;
        if (inputs.size() == 3) {
            if (bias_f32_.empty()) prepare_f32_bias(inputs[2], n_, dynamic_bias_f32_);
            bias_ptr = bias_f32_.empty() ? dynamic_bias_f32_.data() : bias_f32_.data();
        }

        if (b_is_single_matrix_) {
            const void* packed_ptr = nullptr;
            if (static_b_) {
                packed_ptr = require(std::as_const(packed_b_).host_bytes()).data();
            } else {
                if (dynamic_b_.size() < plan_.total_bytes) dynamic_b_.resize(plan_.total_bytes);
                const auto* b_bytes = require(inputs[1].host_bytes()).data();
                const std::size_t b_row_stride = (transpose_b_ ? k_ : n_) * tensor::element_size(b_dtype_);
                if (ynn::thread_count() > 1 && plan_.blocks_n >= 4) {
                    require(ynn::parallel_ranges(plan_.blocks_n, 1, [&](std::size_t t_begin, std::size_t t_end) {
                        ::ynn::pack_dense_linear_tiles(plan_, transpose_b_, b_bytes, b_row_stride, t_begin, t_end,
                                                       dynamic_b_.data());
                    }));
                } else {
                    ::ynn::pack_dense_linear_tiles(plan_, transpose_b_, b_bytes, b_row_stride, 0, plan_.blocks_n,
                                                   dynamic_b_.data());
                }
                packed_ptr = dynamic_b_.data();
            }

            float* out_f32 = nullptr;
            if (out_dtype_ == DType::F32) {
                out_f32 = reinterpret_cast<float*>(out_raw);
            } else {
                if (out_f32_.size() < eff_m_ * n_) out_f32_.resize(eff_m_ * n_);
                out_f32 = out_f32_.data();
            }
            run_one_matrix(a_base, inputs[0].dtype(), packed_ptr, bias_ptr, out_f32);
            if (out_dtype_ == DType::BF16) {
                ::ynn::convert_f32_to_bf16_rows(eff_m_, n_, out_f32, n_, reinterpret_cast<std::uint16_t*>(out_raw), n_,
                                                0, eff_m_);
            }
            return;
        }

        const auto* b_base = require(inputs[1].host_bytes()).data();
        const std::size_t b_row_stride = (transpose_b_ ? k_ : n_) * tensor::element_size(b_dtype_);
        if (dynamic_b_.size() < plan_.total_bytes) dynamic_b_.resize(plan_.total_bytes);
        if (out_dtype_ == DType::BF16 && out_f32_.size() < m_ * n_) out_f32_.resize(m_ * n_);
        const std::size_t c_matrix_bytes = m_ * n_ * tensor::element_size(out_dtype_);

        for (std::size_t batch = 0; batch < total_batches_; ++batch) {
            std::size_t rem = batch;
            std::size_t a_off = 0;
            std::size_t b_off = 0;
            for (std::size_t d = a_batch_strides_.size(); d-- > 0;) {
                const std::size_t dim = std::max(a_batch_shape_[d], b_batch_shape_[d]);
                const std::size_t idx = rem % dim;
                rem /= dim;
                a_off += idx * a_batch_strides_[d];
                b_off += idx * b_batch_strides_[d];
            }
            ::ynn::pack_dense_linear_tiles(plan_, transpose_b_, b_base + b_off, b_row_stride, 0, plan_.blocks_n,
                                           dynamic_b_.data());
            float* out_f32 =
                out_dtype_ == DType::F32 ? reinterpret_cast<float*>(out_raw + batch * c_matrix_bytes) : out_f32_.data();
            run_one_matrix(a_base + a_off, inputs[0].dtype(), dynamic_b_.data(), bias_ptr, out_f32);
            if (out_dtype_ == DType::BF16) {
                ::ynn::convert_f32_to_bf16_rows(
                    m_, n_, out_f32, n_, reinterpret_cast<std::uint16_t*>(out_raw + batch * c_matrix_bytes), n_, 0, m_);
            }
        }
    }

    DType out_dtype_, b_dtype_;
    bool transpose_b_;
    bool static_b_;
    bool b_is_single_matrix_ = false;
    std::size_t m_ = 0, eff_m_ = 0, n_ = 0, k_ = 0, total_batches_ = 1;
    ::ynn::dense_linear_plan plan_{};
    Tensor packed_b_;
    std::vector<std::int64_t> out_shape_;
    std::vector<std::size_t> a_batch_shape_, b_batch_shape_, a_batch_strides_, b_batch_strides_;
    std::vector<float> bias_f32_, dynamic_bias_f32_, out_f32_;
    std::vector<std::byte> a_conv_;
    ::ynn::aligned_byte_buffer a_packed_, dynamic_b_;
    OutputPool pool_;
};

class QuantizedLinearOp final : public Operator {
public:
    static auto supports(const OperatorSpec& spec, TensorInputs inputs) noexcept -> bool {
        if (spec.operation != Operation::QUANTIZED_LINEAR || (inputs.size() != 3 && inputs.size() != 4) ||
            spec.attributes.empty() || spec.dtype != DType::F32)
            return false;
        const auto& a = inputs[0];
        const auto& b = inputs[1];
        const auto& scale = inputs[2];
        if (a.dtype() != DType::F32 || b.dtype() != DType::I8 || scale.dtype() != DType::F32 || !a.is_contiguous() ||
            !b.is_contiguous() || !scale.is_contiguous() || a.dimensions() < 2 || b.dimensions() != 2)
            return false;
        const bool transpose_b = spec.attributes[0] != 0;
        const auto k = static_cast<std::size_t>(a.size(-1));
        const auto b_k = static_cast<std::size_t>(transpose_b ? b.size(1) : b.size(0));
        const auto n = static_cast<std::size_t>(transpose_b ? b.size(0) : b.size(1));
        if (k == 0 || n == 0 || k != b_k || (scale.numel() != 1 && scale.numel() != n)) return false;
        if (inputs.size() == 4) {
            const auto& bias = inputs[3];
            if ((bias.dtype() != DType::F32 && bias.dtype() != DType::BF16) || !bias.is_contiguous() ||
                (bias.numel() != n && bias.numel() != 1))
                return false;
        }
        return true;
    }

    static auto pack_static_b(const Tensor& b, bool transpose_b, const ::ynn::dense_linear_plan& plan)
        -> std::pair<Tensor, Tensor> {
        auto packed = require(Tensor::empty({static_cast<std::int64_t>(plan.total_bytes)}, DType::U8));
        auto col_sums = require(Tensor::empty({static_cast<std::int64_t>(plan.n)}, DType::I32));
        const auto* b_bytes = require(std::as_const(b).host_bytes()).data();
        auto* dst = require(packed.host_bytes()).data();
        auto* sums = require(col_sums.data<std::int32_t>()).data();
        const std::size_t b_row_stride = transpose_b ? plan.k : plan.n;
        const auto workers = std::min(ynn::thread_count(), plan.blocks_n);
        if (workers > 1 && plan.blocks_n >= 4) {
            const auto per_worker = (plan.blocks_n + workers - 1) / workers;
            require(ynn::parallel_tasks(workers, [&](std::size_t task) {
                const auto t_begin = task * per_worker;
                const auto t_end = std::min(plan.blocks_n, t_begin + per_worker);
                if (t_begin < t_end)
                    ::ynn::pack_dense_linear_tiles(plan, transpose_b, b_bytes, b_row_stride, t_begin, t_end, dst, sums);
            }));
        } else {
            ::ynn::pack_dense_linear_tiles(plan, transpose_b, b_bytes, b_row_stride, 0, plan.blocks_n, dst, sums);
        }
        return {std::move(packed), std::move(col_sums)};
    }

    QuantizedLinearOp(const OperatorSpec& spec, TensorInputs inputs, PackedDenseEntry static_b, tensor::Arena& arena)
        : transpose_b_(spec.attributes[0] != 0),
          static_b_(static_b.packed.defined()),
          plan_(static_b.plan),
          packed_b_(std::move(static_b.packed)),
          col_sums_(std::move(static_b.col_sums)),
          pool_(&arena) {
        const auto& a = inputs[0];
        const auto& b = inputs[1];
        k_ = static_cast<std::size_t>(a.size(-1));
        n_ = static_cast<std::size_t>(transpose_b_ ? b.size(0) : b.size(1));
        rows_ = static_cast<std::size_t>(a.numel() / k_);
        out_shape_.assign(a.shape().begin(), a.shape().end());
        out_shape_.back() = static_cast<std::int64_t>(n_);
        if (!static_b_)
            check(::ynn::make_dense_linear_plan(ynn_type_int8, ynn_type_int8, ynn_type_int32, n_, k_, &plan_));
        if (!spec.dynamic_parameters) {
            prepare_scales(inputs[2], w_scales_);
            if (inputs.size() == 4) prepare_f32_bias(inputs[3], n_, bias_f32_);
        }
    }

    auto run(TensorInputs inputs) -> Tensor override {
        auto output = pool_.acquire(out_shape_, DType::F32, tensor::Device::cpu());
        apply(inputs, output);
        return output;
    }
    auto run_(TensorInputs, Tensor&) -> Tensor override {
        throw ops::Failure({ErrorCode::UNSUPPORTED, "quantized linear cannot run in place"});
    }
    auto run_into(TensorInputs inputs, std::span<Tensor> outputs) -> void override { apply(inputs, outputs.back()); }
    auto allocations() const -> AllocationStats override { return pool_.allocations(); }

private:
    auto prepare_scales(const Tensor& scale, std::vector<float>& dst) const -> void {
        dst.resize(n_);
        const auto src = require(std::as_const(scale).data<float>());
        if (src.size() == 1)
            std::ranges::fill(dst, src[0]);
        else
            std::ranges::copy(src, dst.begin());
    }

    auto apply(TensorInputs inputs, Tensor& output) -> void {
        if (rows_ == 0) return;
        const auto* a_f32 = require(inputs[0].data<float>()).data();
        auto* y = require(output.data<float>()).data();

        const void* packed_ptr = nullptr;
        const std::int32_t* sum_ptr = nullptr;
        if (static_b_) {
            packed_ptr = require(std::as_const(packed_b_).host_bytes()).data();
            sum_ptr = require(std::as_const(col_sums_).data<std::int32_t>()).data();
        } else {
            if (dynamic_b_.size() < plan_.total_bytes) dynamic_b_.resize(plan_.total_bytes);
            if (dynamic_sums_.size() < n_) dynamic_sums_.resize(n_);
            const auto* b_bytes = require(inputs[1].host_bytes()).data();
            ::ynn::pack_dense_linear_tiles(plan_, transpose_b_, b_bytes, transpose_b_ ? k_ : n_, 0, plan_.blocks_n,
                                           dynamic_b_.data(), dynamic_sums_.data());
            packed_ptr = dynamic_b_.data();
            sum_ptr = dynamic_sums_.data();
        }

        const float* scale_ptr = nullptr;
        if (w_scales_.empty()) {
            prepare_scales(inputs[2], dynamic_scales_);
            scale_ptr = dynamic_scales_.data();
        } else {
            scale_ptr = w_scales_.data();
        }

        const float* bias_ptr = nullptr;
        if (inputs.size() == 4) {
            if (bias_f32_.empty()) prepare_f32_bias(inputs[3], n_, dynamic_bias_f32_);
            bias_ptr = bias_f32_.empty() ? dynamic_bias_f32_.data() : bias_f32_.data();
        }

        if (q_a_.size() < rows_ * k_) q_a_.resize(rows_ * k_);
        if (a_scales_.size() < rows_) a_scales_.resize(rows_);
        if (a_zps_.size() < rows_) a_zps_.resize(rows_);

        const auto threads = ynn::thread_count();
        if (threads <= 1 || rows_ * k_ < 8192) {
            ::ynn::quantize_dynamic_int8_rows_f32(rows_, k_, a_f32, q_a_.data(), a_scales_.data(), a_zps_.data(), 0,
                                                  rows_);
        } else {
            require(ynn::parallel_ranges(rows_, 1, [&](std::size_t r_begin, std::size_t r_end) {
                ::ynn::quantize_dynamic_int8_rows_f32(rows_, k_, a_f32, q_a_.data(), a_scales_.data(), a_zps_.data(),
                                                      r_begin, r_end);
            }));
        }

        bool transposed_a = false;
        std::size_t tile_m = 1;
        std::size_t a_packed_bytes = 0;
        if (!::ynn::dense_linear_select_kernel(plan_, rows_, &transposed_a, &tile_m, &a_packed_bytes))
            throw ops::Failure({ErrorCode::UNSUPPORTED, "no matching YNNPACK quantized dot kernel"});

        const void* eff_a = q_a_.data();
        if (transposed_a) {
            if (a_packed_.size() < a_packed_bytes) a_packed_.resize(a_packed_bytes);
            ::ynn::dense_linear_pack_a(plan_, rows_, tile_m, q_a_.data(), k_, a_packed_.data());
            eff_a = a_packed_.data();
        }

        const std::size_t scratch_bytes = ::ynn::quantized_linear_scratch_bytes(plan_, rows_, tile_m);
        if (threads <= 1 || plan_.blocks_n <= 1 || (rows_ == 1 && k_ * n_ <= 64 * 1024)) {
            if (fallback_scratch_.size() < scratch_bytes / sizeof(std::int32_t))
                fallback_scratch_.resize(scratch_bytes / sizeof(std::int32_t));
            ::ynn::quantized_linear_run_tiles(plan_, rows_, transposed_a, tile_m, eff_a, k_, packed_ptr, sum_ptr,
                                              a_scales_.data(), a_zps_.data(), scale_ptr, bias_ptr, y, n_,
                                              fallback_scratch_.data(), 0, plan_.blocks_n);
            return;
        }
        require(ynn::parallel_ranges_with_worker(
            plan_.blocks_n, 1, scratch_bytes,
            [&](const ynn::WorkerContext& worker, std::size_t t_begin, std::size_t t_end) {
                ::ynn::quantized_linear_run_tiles(plan_, rows_, transposed_a, tile_m, eff_a, k_, packed_ptr, sum_ptr,
                                                  a_scales_.data(), a_zps_.data(), scale_ptr, bias_ptr, y, n_,
                                                  reinterpret_cast<std::int32_t*>(worker.scratch), t_begin, t_end);
            }));
    }

    bool transpose_b_;
    bool static_b_;
    std::size_t rows_ = 0, n_ = 0, k_ = 0;
    ::ynn::dense_linear_plan plan_{};
    Tensor packed_b_, col_sums_;
    std::vector<std::int64_t> out_shape_;
    std::vector<float> w_scales_, dynamic_scales_, bias_f32_, dynamic_bias_f32_, a_scales_;
    std::vector<std::int32_t> a_zps_, dynamic_sums_, fallback_scratch_;
    std::vector<std::int8_t> q_a_;
    ::ynn::aligned_byte_buffer a_packed_, dynamic_b_;
    OutputPool pool_;
};

auto cast_ynn_type(DType dtype) -> ynn_type {
    switch (dtype) {
        case DType::F32:
            return ynn_type_fp32;
        case DType::BF16:
            return ynn_type_bf16;
        case DType::F16:
            return ynn_type_fp16;
        case DType::E4M3:
            return ynn_type_fp8_e4m3;
        case DType::E5M2:
            return ynn_type_fp8_e5m2;
        case DType::I32:
            return ynn_type_int32;
        case DType::I8:
            return ynn_type_int8;
        case DType::U8:
            return ynn_type_uint8;
        default:
            throw ops::Failure({ErrorCode::UNSUPPORTED, "unsupported CPU cast dtype"});
    }
}

class BroadcastBinaryOp final : public Operator {
public:
    static auto supports(const OperatorSpec& spec, TensorInputs inputs) noexcept -> bool {
        if ((spec.operation != Operation::ADD && spec.operation != Operation::MULTIPLY) || inputs.size() != 2 ||
            !inputs[0].is_contiguous() || !inputs[1].is_contiguous())
            return false;
        const auto a_dt = inputs[0].dtype(), b_dt = inputs[1].dtype();
        const bool is_float =
            (a_dt == DType::F32 || a_dt == DType::BF16) && (b_dt == DType::F32 || b_dt == DType::BF16);
        const bool is_int = (a_dt == DType::I32 || a_dt == DType::I64) && a_dt == b_dt && spec.dtype == a_dt;
        if (!is_float && !is_int) return false;
        const auto ra = inputs[0].dimensions(), rb = inputs[1].dimensions();
        const auto rank = std::max(ra, rb);
        for (std::size_t i = 0; i < rank; ++i) {
            const auto ad = i >= rank - ra ? inputs[0].shape()[i - (rank - ra)] : 1;
            const auto bd = i >= rank - rb ? inputs[1].shape()[i - (rank - rb)] : 1;
            if (ad != bd && ad != 1 && bd != 1) return false;
        }
        return true;
    }

    BroadcastBinaryOp(const OperatorSpec& spec, TensorInputs inputs, tensor::Arena& arena)
        : is_multiply_(spec.operation == Operation::MULTIPLY), out_dtype_(spec.dtype), pool_(&arena) {
        const auto ra = inputs[0].dimensions(), rb = inputs[1].dimensions();
        const auto rank = std::max(ra, rb);
        shape_.resize(rank);
        out_shape_.resize(rank);
        a_strides_.assign(rank, 0);
        b_strides_.assign(rank, 0);
        std::size_t a_stride = 1, b_stride = 1;
        out_elements_ = 1;
        for (std::size_t i = rank; i-- > 0;) {
            const auto ad = i >= rank - ra ? static_cast<std::size_t>(inputs[0].shape()[i - (rank - ra)]) : 1;
            const auto bd = i >= rank - rb ? static_cast<std::size_t>(inputs[1].shape()[i - (rank - rb)]) : 1;
            const auto od = std::max(ad, bd);
            shape_[i] = static_cast<std::int64_t>(od);
            out_shape_[i] = od;
            out_elements_ *= od;
            a_strides_[i] = ad > 1 ? a_stride : 0;
            b_strides_[i] = bd > 1 ? b_stride : 0;
            a_stride *= ad;
            b_stride *= bd;
        }
        if (rank == 0) {
            out_shape_.push_back(1);
            a_strides_.push_back(0);
            b_strides_.push_back(0);
        }
    }

    auto run(TensorInputs inputs) -> Tensor override {
        auto output = pool_.acquire(shape_, out_dtype_, tensor::Device::cpu());
        apply(inputs, output);
        return output;
    }
    auto run_(TensorInputs inputs, Tensor& destination) -> Tensor override {
        if (destination.dtype() != out_dtype_ || !std::ranges::equal(destination.shape(), shape_))
            throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "in-place operation cannot change shape or dtype"});
        apply(inputs, destination);
        return destination;
    }
    auto run_into(TensorInputs inputs, std::span<Tensor> outputs) -> void override { apply(inputs, outputs[0]); }
    auto allocations() const -> AllocationStats override { return pool_.allocations(); }

private:
    static auto ensure_f32(const Tensor& tensor, std::vector<float>& scratch) -> const float* {
        if (tensor.dtype() == DType::F32) return require(tensor.data<float>()).data();
        const auto count = tensor.numel();
        if (scratch.size() < count) scratch.resize(count);
        const auto raw = require(tensor.host_bytes());
        ::ynn::convert_bf16_to_f32_rows(1, count, reinterpret_cast<const std::uint16_t*>(raw.data()), count,
                                        scratch.data(), count, 0, 1);
        return scratch.data();
    }

    auto apply(TensorInputs inputs, Tensor& output) -> void {
        if (out_elements_ == 0) return;
        const auto rank = out_shape_.size();
        if (out_dtype_ == DType::I32) {
            ::ynn::broadcast_binary_i32(is_multiply_, rank, out_shape_.data(), a_strides_.data(), b_strides_.data(),
                                        out_elements_, require(inputs[0].data<std::int32_t>()).data(),
                                        require(inputs[1].data<std::int32_t>()).data(),
                                        require(output.data<std::int32_t>()).data());
            return;
        }
        if (out_dtype_ == DType::I64) {
            ::ynn::broadcast_binary_i64(is_multiply_, rank, out_shape_.data(), a_strides_.data(), b_strides_.data(),
                                        out_elements_, require(inputs[0].data<std::int64_t>()).data(),
                                        require(inputs[1].data<std::int64_t>()).data(),
                                        require(output.data<std::int64_t>()).data());
            return;
        }
        const auto* a = ensure_f32(inputs[0], a_f32_);
        const auto* b = ensure_f32(inputs[1], b_f32_);
        float* dst = nullptr;
        if (out_dtype_ == DType::F32) {
            dst = require(output.data<float>()).data();
        } else {
            if (out_f32_.size() < out_elements_) out_f32_.resize(out_elements_);
            dst = out_f32_.data();
        }
        if (out_elements_ < 32768 || ynn::thread_count() <= 1) {
            ::ynn::broadcast_binary_f32(is_multiply_, rank, out_shape_.data(), a_strides_.data(), b_strides_.data(),
                                        out_elements_, a, b, dst, 0, out_elements_);
        } else {
            require(ynn::parallel_ranges(out_elements_, 8192, [&](std::size_t begin, std::size_t end) {
                ::ynn::broadcast_binary_f32(is_multiply_, rank, out_shape_.data(), a_strides_.data(), b_strides_.data(),
                                            out_elements_, a, b, dst, begin, end);
            }));
        }
        if (out_dtype_ == DType::BF16) {
            auto raw = require(output.host_bytes());
            ::ynn::convert_f32_to_bf16_rows(1, out_elements_, dst, out_elements_,
                                            reinterpret_cast<std::uint16_t*>(raw.data()), out_elements_, 0, 1);
        }
    }

    bool is_multiply_;
    DType out_dtype_;
    std::size_t out_elements_ = 0;
    std::vector<std::int64_t> shape_;
    std::vector<std::size_t> out_shape_, a_strides_, b_strides_;
    std::vector<float> a_f32_, b_f32_, out_f32_;
    OutputPool pool_;
};

class CastOp final : public Operator {
public:
    CastOp(const OperatorSpec& spec, const Tensor& input, tensor::Arena& arena)
        : src_type_(cast_ynn_type(input.dtype())),
          dst_type_(cast_ynn_type(spec.dtype)),
          out_dtype_(spec.dtype),
          count_(input.numel()),
          shape_(input.shape().begin(), input.shape().end()),
          pool_(&arena) {}

    auto run(TensorInputs inputs) -> Tensor override {
        auto output = pool_.acquire(shape_, out_dtype_, tensor::Device::cpu());
        apply(inputs[0], output);
        return output;
    }
    auto run_(TensorInputs, Tensor&) -> Tensor override {
        throw ops::Failure({ErrorCode::UNSUPPORTED, "cast cannot run in place"});
    }
    auto run_into(TensorInputs inputs, std::span<Tensor> outputs) -> void override { apply(inputs[0], outputs[0]); }
    auto allocations() const -> AllocationStats override { return pool_.allocations(); }

private:
    auto apply(const Tensor& input, Tensor& output) const -> void {
        if (count_ == 0) return;
        const auto* src = require(input.host_bytes()).data();
        auto* dst = require(output.host_bytes()).data();
        if (count_ < 32768 || ynn::thread_count() <= 1) {
            ::ynn::cast_elements(src_type_, dst_type_, count_, src, dst, 0, count_);
            return;
        }
        require(ynn::parallel_ranges(count_, 8192, [&](std::size_t begin, std::size_t end) {
            ::ynn::cast_elements(src_type_, dst_type_, count_, src, dst, begin, end);
        }));
    }

    ynn_type src_type_, dst_type_;
    DType out_dtype_;
    std::size_t count_;
    std::vector<std::int64_t> shape_;
    OutputPool pool_;
};

class LayerNormOp final : public Operator {
public:
    static auto supports(const OperatorSpec& spec, TensorInputs inputs) noexcept -> bool {
        const bool paired = spec.operation == Operation::RESIDUAL_NORM;
        if ((spec.operation != Operation::LAYER_NORM && !paired) || inputs.size() != (paired ? 4U : 3U)) return false;
        const auto& x = inputs[0];
        if ((x.dtype() != DType::F32 && x.dtype() != DType::BF16) || !x.is_contiguous() || x.dimensions() == 0 ||
            x.size(-1) == 0)
            return false;
        if (paired && (inputs[1].dtype() != x.dtype() || !inputs[1].is_contiguous() ||
                       !std::ranges::equal(inputs[1].shape(), x.shape())))
            return false;
        const auto w_idx = paired ? 2U : 1U;
        for (std::size_t i = w_idx; i < inputs.size(); ++i) {
            if ((inputs[i].dtype() != DType::F32 && inputs[i].dtype() != DType::BF16) || !inputs[i].is_contiguous() ||
                inputs[i].numel() != x.size(-1))
                return false;
        }
        return true;
    }

    LayerNormOp(const OperatorSpec& spec, TensorInputs inputs, tensor::Arena& arena)
        : paired_(spec.operation == Operation::RESIDUAL_NORM),
          epsilon_(spec.epsilon),
          width_(static_cast<std::size_t>(inputs[0].size(-1))),
          rows_(inputs[0].numel() / width_),
          shape_(inputs[0].shape().begin(), inputs[0].shape().end()),
          pool_(&arena) {
        const auto w_idx = paired_ ? 2U : 1U;
        if (!spec.dynamic_parameters) {
            load_f32(inputs[w_idx], weight_f32_);
            load_f32(inputs[w_idx + 1], bias_f32_);
        }
    }

    auto run(TensorInputs inputs) -> Tensor override {
        auto norm = pool_.acquire(shape_, DType::F32, tensor::Device::cpu());
        apply(inputs, norm, nullptr);
        return norm;
    }
    auto run_(TensorInputs inputs, Tensor& destination) -> Tensor override {
        apply(inputs, destination, nullptr);
        return destination;
    }
    auto run_pair(TensorInputs inputs) -> std::array<Tensor, 2> override {
        auto residual = pool_.acquire(shape_, DType::F32, tensor::Device::cpu());
        auto norm = pool_.acquire(shape_, DType::F32, tensor::Device::cpu());
        apply(inputs, norm, &residual);
        return {std::move(residual), std::move(norm)};
    }
    auto run_into(TensorInputs inputs, std::span<Tensor> outputs) -> void override {
        apply(inputs, outputs.back(), outputs.size() == 2 ? &outputs.front() : nullptr);
    }
    auto allocations() const -> AllocationStats override { return pool_.allocations(); }

private:
    static auto load_f32(const Tensor& tensor, std::vector<float>& dst) -> const float* {
        if (tensor.dtype() == DType::F32) {
            const auto src = require(std::as_const(tensor).data<float>());
            dst.assign(src.begin(), src.end());
            return dst.data();
        }
        const auto count = tensor.numel();
        dst.resize(count);
        const auto raw = require(std::as_const(tensor).host_bytes());
        ::ynn::convert_bf16_to_f32_rows(1, count, reinterpret_cast<const std::uint16_t*>(raw.data()), count, dst.data(),
                                        count, 0, 1);
        return dst.data();
    }

    auto apply(TensorInputs inputs, Tensor& out_norm, Tensor* out_residual) -> void {
        if (rows_ == 0) return;
        const float* x =
            inputs[0].dtype() == DType::F32 ? require(inputs[0].data<float>()).data() : load_f32(inputs[0], x_f32_);
        const float* res = nullptr;
        if (paired_) {
            res = inputs[1].dtype() == DType::F32 ? require(inputs[1].data<float>()).data()
                                                  : load_f32(inputs[1], res_f32_);
        }
        const auto w_idx = paired_ ? 2U : 1U;
        const float* w = weight_f32_.empty()
                             ? (inputs[w_idx].dtype() == DType::F32 ? require(inputs[w_idx].data<float>()).data()
                                                                    : load_f32(inputs[w_idx], dyn_w_))
                             : weight_f32_.data();
        const float* b = bias_f32_.empty() ? (inputs[w_idx + 1].dtype() == DType::F32
                                                  ? require(inputs[w_idx + 1].data<float>()).data()
                                                  : load_f32(inputs[w_idx + 1], dyn_b_))
                                           : bias_f32_.data();
        auto* yn = require(out_norm.data<float>()).data();
        auto* yr = out_residual ? require(out_residual->data<float>()).data() : nullptr;
        for_rows(rows_, width_, [&](std::size_t begin, std::size_t end) {
            ::ynn::layer_norm_f32(rows_, width_, epsilon_, x, res, w, b, yn, yr, begin, end);
        });
    }

    bool paired_;
    float epsilon_;
    std::size_t width_, rows_;
    std::vector<std::int64_t> shape_;
    std::vector<float> weight_f32_, bias_f32_, dyn_w_, dyn_b_, x_f32_, res_f32_;
    OutputPool pool_;
};

class SoftmaxOp final : public Operator {
public:
    SoftmaxOp(const OperatorSpec& spec, const Tensor& input, tensor::Arena& arena)
        : logarithmic_(spec.operation == Operation::LOG_SOFTMAX),
          width_(static_cast<std::size_t>(input.size(-1))),
          rows_(input.numel() / width_),
          out_dtype_(spec.dtype),
          shape_(input.shape().begin(), input.shape().end()),
          pool_(&arena) {}

    auto run(TensorInputs inputs) -> Tensor override {
        auto output = pool_.acquire(shape_, out_dtype_, tensor::Device::cpu());
        apply(inputs[0], output);
        return output;
    }
    auto run_(TensorInputs inputs, Tensor& destination) -> Tensor override {
        apply(inputs[0], destination);
        return destination;
    }
    auto run_into(TensorInputs inputs, std::span<Tensor> outputs) -> void override { apply(inputs[0], outputs[0]); }
    auto allocations() const -> AllocationStats override { return pool_.allocations(); }

private:
    auto apply(const Tensor& input, Tensor& output) -> void {
        if (rows_ == 0) return;
        const float* src = nullptr;
        if (input.dtype() == DType::F32) {
            src = require(input.data<float>()).data();
        } else {
            if (x_f32_.size() < rows_ * width_) x_f32_.resize(rows_ * width_);
            const auto raw = require(input.host_bytes());
            ::ynn::convert_bf16_to_f32_rows(rows_, width_, reinterpret_cast<const std::uint16_t*>(raw.data()), width_,
                                            x_f32_.data(), width_, 0, rows_);
            src = x_f32_.data();
        }
        float* dst = nullptr;
        if (out_dtype_ == DType::F32) {
            dst = require(output.data<float>()).data();
        } else {
            if (out_f32_.size() < rows_ * width_) out_f32_.resize(rows_ * width_);
            dst = out_f32_.data();
        }
        for_rows(rows_, width_, [&](std::size_t begin, std::size_t end) {
            ::ynn::softmax_f32(rows_, width_, logarithmic_, src, dst, begin, end);
        });
        if (out_dtype_ == DType::BF16) {
            auto raw = require(output.host_bytes());
            ::ynn::convert_f32_to_bf16_rows(rows_, width_, dst, width_, reinterpret_cast<std::uint16_t*>(raw.data()),
                                            width_, 0, rows_);
        }
    }

    bool logarithmic_;
    std::size_t width_, rows_;
    DType out_dtype_;
    std::vector<std::int64_t> shape_;
    std::vector<float> x_f32_, out_f32_;
    OutputPool pool_;
};

class UnaryElementwiseOp final : public Operator {
public:
    UnaryElementwiseOp(const OperatorSpec& spec, const Tensor& input, tensor::Arena& arena)
        : op_(spec.operation),
          approximate_(!spec.attributes.empty() && spec.attributes[0] != 0),
          scale_(spec.epsilon),
          count_(input.numel()),
          out_dtype_(spec.dtype),
          shape_(input.shape().begin(), input.shape().end()),
          pool_(&arena) {}

    auto run(TensorInputs inputs) -> Tensor override {
        auto output = pool_.acquire(shape_, out_dtype_, tensor::Device::cpu());
        apply(inputs[0], output);
        return output;
    }
    auto run_(TensorInputs inputs, Tensor& destination) -> Tensor override {
        apply(inputs[0], destination);
        return destination;
    }
    auto run_into(TensorInputs inputs, std::span<Tensor> outputs) -> void override { apply(inputs[0], outputs[0]); }
    auto allocations() const -> AllocationStats override { return pool_.allocations(); }

private:
    auto apply(const Tensor& input, Tensor& output) -> void {
        if (count_ == 0) return;
        const float* src = nullptr;
        if (input.dtype() == DType::F32) {
            src = require(input.data<float>()).data();
        } else {
            if (x_f32_.size() < count_) x_f32_.resize(count_);
            const auto raw = require(input.host_bytes());
            ::ynn::convert_bf16_to_f32_rows(1, count_, reinterpret_cast<const std::uint16_t*>(raw.data()), count_,
                                            x_f32_.data(), count_, 0, 1);
            src = x_f32_.data();
        }
        float* dst = nullptr;
        if (out_dtype_ == DType::F32) {
            dst = require(output.data<float>()).data();
        } else {
            if (out_f32_.size() < count_) out_f32_.resize(count_);
            dst = out_f32_.data();
        }
        const auto run_chunk = [&](std::size_t begin, std::size_t end) {
            if (op_ == Operation::GELU)
                ::ynn::gelu_f32(count_, approximate_, src, dst, begin, end);
            else if (op_ == Operation::TANH)
                ::ynn::tanh_f32(count_, src, dst, begin, end);
            else
                ::ynn::static_round_f32(count_, scale_, src, dst, begin, end);
        };
        if (count_ < 16384 || ynn::thread_count() <= 1) {
            run_chunk(0, count_);
        } else {
            require(ynn::parallel_ranges(count_, 4096, run_chunk));
        }
        if (out_dtype_ == DType::BF16) {
            auto raw = require(output.host_bytes());
            ::ynn::convert_f32_to_bf16_rows(1, count_, dst, count_, reinterpret_cast<std::uint16_t*>(raw.data()),
                                            count_, 0, 1);
        }
    }

    Operation op_;
    bool approximate_;
    float scale_;
    std::size_t count_;
    DType out_dtype_;
    std::vector<std::int64_t> shape_;
    std::vector<float> x_f32_, out_f32_;
    OutputPool pool_;
};

class TransposeOp final : public Operator {
public:
    TransposeOp(const OperatorSpec& spec, const Tensor& input, tensor::Arena& arena)
        : dtype_(input.dtype()), elem_bytes_(tensor::element_size(dtype_)), pool_(&arena) {
        const auto rank = input.dimensions();
        in_shape_.resize(rank);
        out_shape_.resize(rank);
        std::vector<std::size_t> out_axis_stride(rank, 1);
        for (std::size_t d = 0; d < rank; ++d) {
            in_shape_[d] = static_cast<std::size_t>(input.shape()[d]);
            const auto src_axis = static_cast<std::size_t>(spec.attributes[d]);
            out_shape_[d] = input.shape()[src_axis];
        }
        std::size_t stride = 1;
        for (std::size_t d = rank; d-- > 0;) {
            out_axis_stride[d] = stride;
            stride *= static_cast<std::size_t>(out_shape_[d]);
        }
        out_strides_.resize(rank, 0);
        for (std::size_t d = 0; d < rank; ++d)
            out_strides_[static_cast<std::size_t>(spec.attributes[d])] = out_axis_stride[d];
        outer_ = 1;
        for (std::size_t d = 0; d + 1 < rank; ++d) outer_ *= in_shape_[d];
    }

    auto run(TensorInputs inputs) -> Tensor override {
        auto output = pool_.acquire(out_shape_, dtype_, tensor::Device::cpu());
        apply(inputs[0], output);
        return output;
    }
    auto run_(TensorInputs, Tensor&) -> Tensor override {
        throw ops::Failure({ErrorCode::UNSUPPORTED, "transpose cannot run in place"});
    }
    auto run_into(TensorInputs inputs, std::span<Tensor> outputs) -> void override { apply(inputs[0], outputs[0]); }
    auto allocations() const -> AllocationStats override { return pool_.allocations(); }

private:
    auto apply(const Tensor& input, Tensor& output) const -> void {
        if (input.numel() == 0) return;
        const auto* src = require(input.host_bytes()).data();
        auto* dst = require(output.host_bytes()).data();
        const auto* s_bytes = reinterpret_cast<const std::uint8_t*>(src);
        auto* d_bytes = reinterpret_cast<std::uint8_t*>(dst);
        if (outer_ <= 1 || input.numel() < 16384 || ynn::thread_count() <= 1) {
            ::ynn::transpose_elements(elem_bytes_, in_shape_.size(), in_shape_.data(), out_strides_.data(), s_bytes,
                                      d_bytes, 0, outer_);
            return;
        }
        require(ynn::parallel_ranges(outer_, 1, [&](std::size_t begin, std::size_t end) {
            ::ynn::transpose_elements(elem_bytes_, in_shape_.size(), in_shape_.data(), out_strides_.data(), s_bytes,
                                      d_bytes, begin, end);
        }));
    }

    DType dtype_;
    std::size_t elem_bytes_, outer_ = 1;
    std::vector<std::int64_t> out_shape_;
    std::vector<std::size_t> in_shape_, out_strides_;
    OutputPool pool_;
};

class GatherOp final : public Operator {
public:
    GatherOp(const OperatorSpec& spec, TensorInputs inputs, tensor::Arena& arena)
        : dtype_(inputs[0].dtype()),
          idx_dtype_(inputs[1].dtype()),
          elem_bytes_(tensor::element_size(dtype_)),
          pool_(&arena) {
        const auto rank = static_cast<std::int64_t>(inputs[0].dimensions());
        auto axis = spec.attributes[0];
        if (axis < 0) axis += rank;
        if (axis < 0 || axis >= rank)
            throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "gather axis outside tensor rank"});
        const auto u_axis = static_cast<std::size_t>(axis);
        axis_len_ = static_cast<std::size_t>(inputs[0].size(u_axis));
        index_count_ = inputs[1].numel();
        outer_ = 1;
        for (std::size_t d = 0; d < u_axis; ++d) outer_ *= static_cast<std::size_t>(inputs[0].shape()[d]);
        inner_ = 1;
        for (std::size_t d = u_axis + 1; d < static_cast<std::size_t>(rank); ++d)
            inner_ *= static_cast<std::size_t>(inputs[0].shape()[d]);

        out_shape_.assign(inputs[0].shape().begin(), inputs[0].shape().end());
        if (inputs[1].dimensions() == 1) {
            out_shape_[u_axis] = static_cast<std::int64_t>(index_count_);
        } else {
            out_shape_.erase(out_shape_.begin() + static_cast<std::ptrdiff_t>(u_axis));
            out_shape_.insert(out_shape_.begin() + static_cast<std::ptrdiff_t>(u_axis), inputs[1].shape().begin(),
                              inputs[1].shape().end());
        }
    }

    auto run(TensorInputs inputs) -> Tensor override {
        auto output = pool_.acquire(out_shape_, dtype_, tensor::Device::cpu());
        apply(inputs, output);
        return output;
    }
    auto run_(TensorInputs, Tensor&) -> Tensor override {
        throw ops::Failure({ErrorCode::UNSUPPORTED, "gather cannot run in place"});
    }
    auto run_into(TensorInputs inputs, std::span<Tensor> outputs) -> void override { apply(inputs, outputs[0]); }
    auto allocations() const -> AllocationStats override { return pool_.allocations(); }

private:
    auto apply(TensorInputs inputs, Tensor& output) const -> void {
        if (outer_ == 0 || index_count_ == 0 || inner_ == 0) return;
        const auto* src = require(inputs[0].host_bytes()).data();
        auto* dst = require(output.host_bytes()).data();
        const auto* idx_bytes = require(inputs[1].host_bytes()).data();
        const bool idx_is_i64 = idx_dtype_ == DType::I64;
        for (std::size_t j = 0; j < index_count_; ++j) {
            const auto raw = idx_is_i64
                                 ? reinterpret_cast<const std::int64_t*>(idx_bytes)[j]
                                 : static_cast<std::int64_t>(reinterpret_cast<const std::int32_t*>(idx_bytes)[j]);
            if (raw < 0 || static_cast<std::size_t>(raw) >= axis_len_)
                throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "gather index out of range"});
        }
        const auto total_copies = outer_ * index_count_;
        if (total_copies * inner_ * elem_bytes_ < 65536 || ynn::thread_count() <= 1) {
            ::ynn::gather_elements(elem_bytes_, axis_len_, inner_, index_count_, idx_is_i64, src, idx_bytes, dst, 0,
                                   total_copies);
            return;
        }
        require(ynn::parallel_ranges(total_copies, 1, [&](std::size_t begin, std::size_t end) {
            ::ynn::gather_elements(elem_bytes_, axis_len_, inner_, index_count_, idx_is_i64, src, idx_bytes, dst, begin,
                                   end);
        }));
    }

    DType dtype_, idx_dtype_;
    std::size_t elem_bytes_, outer_ = 1, axis_len_ = 0, inner_ = 1, index_count_ = 0;
    std::vector<std::int64_t> out_shape_;
    OutputPool pool_;
};

class ConcatOp final : public Operator {
public:
    ConcatOp(const OperatorSpec& spec, TensorInputs inputs, tensor::Arena& arena)
        : dtype_(inputs[0].dtype()), pool_(&arena) {
        const auto rank = static_cast<std::int64_t>(inputs[0].dimensions());
        auto axis = spec.attributes[0];
        if (axis < 0) axis += rank;
        if (axis < 0 || axis >= rank)
            throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "concat axis outside tensor rank"});
        const auto u_axis = static_cast<std::size_t>(axis);
        out_shape_.assign(inputs[0].shape().begin(), inputs[0].shape().end());
        out_shape_[u_axis] = 0;
        for (std::size_t idx = 0; idx < inputs.size(); ++idx) out_shape_[u_axis] += inputs[idx].size(u_axis);
        outer_ = 1;
        for (std::size_t d = 0; d < u_axis; ++d) outer_ *= static_cast<std::size_t>(inputs[0].shape()[d]);
        std::size_t inner = 1;
        for (std::size_t d = u_axis + 1; d < static_cast<std::size_t>(rank); ++d)
            inner *= static_cast<std::size_t>(inputs[0].shape()[d]);
        inner_bytes_ = inner * tensor::element_size(dtype_);
        dst_stride_bytes_ = static_cast<std::size_t>(out_shape_[u_axis]) * inner_bytes_;
        axis_ = u_axis;
    }

    auto run(TensorInputs inputs) -> Tensor override {
        auto output = pool_.acquire(out_shape_, dtype_, tensor::Device::cpu());
        apply(inputs, output);
        return output;
    }
    auto run_(TensorInputs, Tensor&) -> Tensor override {
        throw ops::Failure({ErrorCode::UNSUPPORTED, "concat cannot run in place"});
    }
    auto run_into(TensorInputs inputs, std::span<Tensor> outputs) -> void override { apply(inputs, outputs[0]); }
    auto allocations() const -> AllocationStats override { return pool_.allocations(); }

private:
    auto apply(TensorInputs inputs, Tensor& output) const -> void {
        if (outer_ == 0 || dst_stride_bytes_ == 0) return;
        auto* dst = require(output.host_bytes()).data();
        std::size_t dst_off = 0;
        for (std::size_t idx = 0; idx < inputs.size(); ++idx) {
            const auto& t = inputs[idx];
            const auto src_stride = static_cast<std::size_t>(t.size(axis_)) * inner_bytes_;
            if (src_stride == 0) continue;
            const auto* src = require(t.host_bytes()).data();
            for (std::size_t o = 0; o < outer_; ++o) {
                std::memcpy(dst + o * dst_stride_bytes_ + dst_off, src + o * src_stride, src_stride);
            }
            dst_off += src_stride;
        }
    }

    DType dtype_;
    std::size_t axis_ = 0, outer_ = 1, inner_bytes_ = 0, dst_stride_bytes_ = 0;
    std::vector<std::int64_t> out_shape_;
    OutputPool pool_;
};

class PortablePackedLinearOp final : public Operator {
public:
    PortablePackedLinearOp(const OperatorSpec& spec, TensorInputs inputs, tensor::Arena& arena)
        : bits_(static_cast<std::int32_t>(spec.attributes[0])),
          block_size_(static_cast<std::size_t>(spec.attributes[1])),
          input_scale_(spec.epsilon),
          output_scale_(std::bit_cast<float>(static_cast<std::int32_t>(spec.attributes[2]))),
          n_(static_cast<std::size_t>(inputs[1].size(0))),
          k_(static_cast<std::size_t>(inputs[0].size(-1))),
          blocks_(k_ / block_size_),
          has_zero_(inputs.size() == 4),
          pool_(&arena) {
        if (!spec.dynamic_parameters) {
            unpack_params(inputs);
            cached_ = true;
        }
    }

    auto run(TensorInputs inputs) -> Tensor override {
        std::vector<std::int64_t> shape(inputs[0].shape().begin(), inputs[0].shape().end());
        shape.back() = static_cast<std::int64_t>(n_);
        auto output = pool_.acquire(shape, DType::F32, tensor::Device::cpu());
        apply(inputs, output);
        return output;
    }
    auto run_(TensorInputs, Tensor&) -> Tensor override {
        throw ops::Failure({ErrorCode::UNSUPPORTED, "packed linear cannot run in place"});
    }
    auto run_into(TensorInputs inputs, std::span<Tensor> outputs) -> void override { apply(inputs, outputs.back()); }
    auto allocations() const -> AllocationStats override { return pool_.allocations(); }

private:
    auto unpack_params(TensorInputs inputs) -> void {
        w_.resize(n_ * k_);
        ::ynn::portable_packed_linear_unpack_weights(
            bits_, reinterpret_cast<const std::uint8_t*>(require(std::as_const(inputs[1]).host_bytes()).data()), n_, k_,
            static_cast<std::size_t>(inputs[1].size(1)), w_.data());
        scales_.resize(inputs[2].numel());
        ::ynn::portable_packed_linear_decode_meta(require(std::as_const(inputs[2]).host_bytes()).data(),
                                                  inputs[2].dtype() == DType::F16, scales_.size(), scales_.data());
        if (has_zero_) {
            zeros_.resize(inputs[3].numel());
            ::ynn::portable_packed_linear_decode_meta(require(std::as_const(inputs[3]).host_bytes()).data(),
                                                      inputs[3].dtype() == DType::F16, zeros_.size(), zeros_.data());
        }
    }

    auto apply(TensorInputs inputs, Tensor& output) -> void {
        if (!cached_) unpack_params(inputs);
        const auto rows = inputs[0].numel() / k_;
        if (rows == 0) return;
        const auto* x = require(inputs[0].data<float>()).data();
        auto* y = require(output.data<float>()).data();
        if (rows <= 1 || ynn::thread_count() <= 1 || rows * n_ * k_ < 32768) {
            ::ynn::portable_packed_linear_f32(rows, n_, k_, block_size_, input_scale_, output_scale_, has_zero_, x,
                                              w_.data(), scales_.data(), has_zero_ ? zeros_.data() : nullptr, y, 0,
                                              rows);
            return;
        }
        require(ynn::parallel_ranges(rows, 1, [&](std::size_t begin, std::size_t end) {
            ::ynn::portable_packed_linear_f32(rows, n_, k_, block_size_, input_scale_, output_scale_, has_zero_, x,
                                              w_.data(), scales_.data(), has_zero_ ? zeros_.data() : nullptr, y, begin,
                                              end);
        }));
    }

    std::int32_t bits_;
    std::size_t block_size_;
    float input_scale_, output_scale_;
    std::size_t n_, k_, blocks_;
    bool has_zero_, cached_ = false;
    std::vector<std::int8_t> w_;
    std::vector<float> scales_, zeros_;
    OutputPool pool_;
};

class VqExpertOp final : public Operator {
public:
    VqExpertOp(const OperatorSpec& spec, TensorInputs inputs, tensor::Arena& arena)
        : gated_(spec.attributes[0] != 0),
          gate_cap_(std::bit_cast<float>(static_cast<std::int32_t>(spec.attributes[1]))),
          up_cap_(std::bit_cast<float>(static_cast<std::int32_t>(spec.attributes[2]))),
          format_(static_cast<ops::BlockScaledFormat>(spec.attributes[3])),
          dim_(format_ == ops::BlockScaledFormat::VQ3D ? 3 : 4),
          out_dtype_(spec.dtype),
          pool_(&arena) {
        const auto& routes = inputs[4];
        const auto tokens = static_cast<std::int64_t>(routes.size(0));
        const auto top = static_cast<std::int64_t>(routes.size(1));
        const auto rows = static_cast<std::int64_t>(inputs[1].size(1));
        if (gated_)
            shape_ = {tokens, top, rows / 2};
        else
            shape_ = {tokens, rows};
    }
    auto run(TensorInputs inputs) -> Tensor override {
        auto output = pool_.acquire(shape_, out_dtype_, tensor::Device::cpu());
        apply(inputs, output);
        return output;
    }
    auto run_(TensorInputs, Tensor&) -> Tensor override {
        throw ops::Failure({ErrorCode::UNSUPPORTED, "VQ expert projection is not an in-place operation"});
    }
    auto run_into(TensorInputs inputs, std::span<Tensor> outputs) -> void override { apply(inputs, outputs[0]); }
    auto allocations() const -> AllocationStats override { return pool_.allocations(); }

private:
    auto apply(TensorInputs inputs, Tensor& output) const -> void {
        const auto& routes = inputs[4];
        const auto tokens = static_cast<std::size_t>(routes.size(0));
        const auto top = static_cast<std::size_t>(routes.size(1));
        const auto rows = static_cast<std::size_t>(inputs[1].size(1));
        const auto cols = static_cast<std::size_t>(inputs[0].size(-1));
        const auto k_pad = ::ynn::vq_padded_cols(dim_, cols);
        const auto had_col = ::ynn::vq_col_hadamard_block(cols);
        const auto had_row = ::ynn::vq_row_hadamard_block(rows);
        const auto row_codes = static_cast<std::size_t>(inputs[1].size(2));
        const auto row_groups = static_cast<std::size_t>(inputs[2].size(2));

        const auto in_f32 = require(inputs[0].data<float>());
        const auto values = require(inputs[1].data<std::uint8_t>());
        const auto scales = require(inputs[2].data<std::uint8_t>());
        const auto global = require(inputs[3].data<float>());
        const auto route_vals = require(routes.data<float>());
        auto out_f32 = require(output.data<float>());

        const auto in_rows = gated_ ? tokens : (tokens * top);
        std::vector<float> x_rot(in_rows * k_pad, 0.0F);
        for (std::size_t r = 0; r < in_rows; ++r) {
            std::memcpy(x_rot.data() + r * k_pad, in_f32.data() + r * cols, cols * sizeof(float));
            if (had_col) ::ynn::hadamard_transform_rows_f32(x_rot.data() + r * k_pad, 1, cols, had_col);
        }

        const auto run_expert_matvec = [&](std::size_t expert, const float* xr, float* dst) {
            const auto* v = values.data() + expert * rows * row_codes;
            const auto* s = scales.data() + expert * rows * row_groups;
            const auto* g = global.data() + expert * rows;
            ::ynn::vq_linear_rotated_f32(dim_, 1, rows, cols, xr, v, s, g, dst, 0, rows);
        };

        if (gated_) {
            const auto hidden = rows / 2;
            const auto pairs = tokens * top;
            require(ynn::parallel_ranges(pairs, 1, [&](std::size_t p0, std::size_t p1) {
                std::vector<float> y_tmp(rows);
                for (std::size_t p = p0; p < p1; ++p) {
                    const auto t = p / top;
                    const auto expert = static_cast<std::size_t>(route_vals[p * 2]);
                    const float weight = route_vals[p * 2 + 1];
                    run_expert_matvec(expert, x_rot.data() + t * k_pad, y_tmp.data());
                    if (had_row) ::ynn::hadamard_transform_rows_f32(y_tmp.data(), 1, rows, had_row);
                    float* dst = out_f32.data() + p * hidden;
                    for (std::size_t i = 0; i < hidden; ++i) {
                        const float g = y_tmp[i];
                        const float u = y_tmp[hidden + i];
                        const float gc = std::tanh(g / gate_cap_) * gate_cap_;
                        const float uc = std::tanh(u / up_cap_) * up_cap_;
                        dst[i] = (gc / (1.0F + std::exp(-g))) * uc * weight;
                    }
                }
            }));
        } else {
            require(ynn::parallel_ranges(tokens, 1, [&](std::size_t t0, std::size_t t1) {
                std::vector<float> y_tmp(rows);
                for (std::size_t t = t0; t < t1; ++t) {
                    float* acc = out_f32.data() + t * rows;
                    std::fill(acc, acc + rows, 0.0F);
                    for (std::size_t slot = 0; slot < top; ++slot) {
                        const auto p = t * top + slot;
                        const auto expert = static_cast<std::size_t>(route_vals[p * 2]);
                        run_expert_matvec(expert, x_rot.data() + p * k_pad, y_tmp.data());
                        // Sum in rotated space (linear!) then un-rotate once after the top-k loop.
                        for (std::size_t r = 0; r < rows; ++r) acc[r] += y_tmp[r];
                    }
                    if (had_row) ::ynn::hadamard_transform_rows_f32(acc, 1, rows, had_row);
                }
            }));
        }
    }

    bool gated_;
    float gate_cap_, up_cap_;
    ops::BlockScaledFormat format_;
    std::int32_t dim_;
    DType out_dtype_;
    std::vector<std::int64_t> shape_;
    OutputPool pool_;
};

class BlockScaledLinear final : public Operator {
public:
    BlockScaledLinear(const OperatorSpec& spec, TensorInputs inputs, tensor::Arena& arena)
        : gated_(spec.attributes[0] != 0),
          gate_cap_(std::bit_cast<float>(static_cast<std::int32_t>(spec.attributes[1]))),
          up_cap_(std::bit_cast<float>(static_cast<std::int32_t>(spec.attributes[2]))),
          format_(static_cast<ops::BlockScaledFormat>(spec.attributes[3])),
          group_(spec.attributes[4]),
          group_shift_(spec.attributes[6]),
          has_global_(spec.attributes[5] != 0),
          pool_(&arena) {
        shape_.assign(inputs[0].shape().begin(), inputs[0].shape().end());
        shape_.back() = static_cast<std::int64_t>(inputs[1].size(0) / (gated_ ? 2 : 1));
    }
    auto run(TensorInputs inputs) -> Tensor override {
        auto output = pool_.acquire(shape_, inputs[0].dtype(), tensor::Device::cpu());
        apply(inputs, output);
        return output;
    }
    auto run_(TensorInputs, Tensor&) -> Tensor override {
        throw ops::Failure({ErrorCode::UNSUPPORTED, "block-scaled projection is not an in-place operation"});
    }
    auto run_into(TensorInputs inputs, std::span<Tensor> outputs) -> void override { apply(inputs, outputs[0]); }
    auto allocations() const -> AllocationStats override { return pool_.allocations(); }

private:
    static auto bfloat(float value) -> std::uint16_t {
        auto bits = std::bit_cast<std::uint32_t>(value);
        return static_cast<std::uint16_t>((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16);
    }
    auto apply(TensorInputs inputs, Tensor& output) const -> void {
        static const auto e4m3 = [] {
            std::array<float, 256> table{};
            for (std::size_t byte = 0; byte < table.size(); ++byte) {
                const int exponent = (byte >> 3) & 15, mantissa = byte & 7;
                const float magnitude = exponent ? std::ldexp(1.F + mantissa / 8.F, exponent - 7) : mantissa / 512.F;
                table[byte] = exponent == 15 && mantissa == 7 ? NAN : (byte & 128 ? -magnitude : magnitude);
            }
            return table;
        }();
        constexpr std::array<float, 16> e2m1{0, .5F, 1, 1.5F, 2, 3, 4, 6, -0.F, -.5F, -1, -1.5F, -2, -3, -4, -6};
        const bool fp8 = format_ == ops::BlockScaledFormat::MXFP8, bf16 = inputs[0].dtype() == DType::BF16;
        const auto width = inputs[0].size(-1), columns = output.size(-1);
        const auto values = require(inputs[1].host_bytes()), scales = require(inputs[2].host_bytes());
        const auto source = require(inputs[0].host_bytes());
        auto destination = require(output.host_bytes());
        const float global = has_global_ ? require(inputs[3].data<float>()).front() : 1.F;
        const auto routing = gated_ ? require(inputs.back().data<float>()) : std::span<const float>{};
        const auto decode = [&](std::size_t row, std::size_t column) {
            const auto index = row * width + column;
            const auto scale_index = group_shift_ >= 0 ? index >> group_shift_ : index / group_;
            float scale;
            const auto scale_type = inputs[2].dtype();
            if (scale_type == DType::U8 || scale_type == DType::E4M3) {
                const auto byte = std::to_integer<std::uint8_t>(scales[scale_index]);
                scale = fp8 && scale_type == DType::U8
                            ? (byte == 255 ? NAN : std::ldexp(1.F, static_cast<int>(byte) - 127))
                            : e4m3[byte];
            } else if (scale_type == DType::F32)
                std::memcpy(&scale, scales.data() + scale_index * 4, 4);
            else {
                std::uint16_t bits;
                std::memcpy(&bits, scales.data() + scale_index * 2, 2);
                scale = scale_type == DType::F16 ? checkpoint::ggml::detail::fp16_to_fp32(bits)
                                                 : std::bit_cast<float>(static_cast<std::uint32_t>(bits) << 16);
            }
            float value;
            if (!fp8) {
                const auto byte = std::to_integer<std::uint8_t>(values[index / 2]);
                const auto code = column % 2 ? byte >> 4 : byte & 15;
                value = format_ == ops::BlockScaledFormat::INT4 ? static_cast<float>(code < 8 ? code : code - 16)
                                                                : e2m1[code];
            } else {
                value = e4m3[std::to_integer<std::uint8_t>(values[index])];
            }
            value *= scale;
            value *= global;
            return bf16 ? std::bit_cast<float>(static_cast<std::uint32_t>(bfloat(value)) << 16) : value;
        };
        require(ynn::parallel_ranges(output.numel(), 8, [&](std::size_t begin, std::size_t end) {
            for (auto index = begin; index < end; ++index) {
                const auto token = index / columns, row = index % columns * (gated_ ? 2 : 1);
                float gate = 0, up = 0;
                for (std::size_t column = 0; column < width; ++column) {
                    const auto position = token * width + column;
                    float activation;
                    if (bf16) {
                        std::uint16_t bits;
                        std::memcpy(&bits, source.data() + position * 2, 2);
                        activation = std::bit_cast<float>(static_cast<std::uint32_t>(bits) << 16);
                    } else
                        std::memcpy(&activation, source.data() + position * 4, 4);
                    gate += activation * decode(row, column);
                    if (gated_) up += activation * decode(row + 1, column);
                }
                if (gated_) {
                    if (bf16) {
                        gate = std::bit_cast<float>(static_cast<std::uint32_t>(bfloat(gate)) << 16);
                        up = std::bit_cast<float>(static_cast<std::uint32_t>(bfloat(up)) << 16);
                    }
                    gate = std::tanh(gate / gate_cap_) * gate_cap_ / (1 + std::exp(-gate)) * std::tanh(up / up_cap_) *
                           up_cap_ * routing[token];
                }
                if (bf16) {
                    const auto bits = bfloat(gate);
                    std::memcpy(destination.data() + index * 2, &bits, 2);
                } else
                    std::memcpy(destination.data() + index * 4, &gate, 4);
            }
        }));
    }
    bool gated_;
    float gate_cap_, up_cap_;
    ops::BlockScaledFormat format_;
    std::int64_t group_, group_shift_;
    bool has_global_;
    std::vector<std::int64_t> shape_;
    OutputPool pool_;
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
        if (SliceOp::supports(spec, inputs)) return std::make_unique<SliceOp>(spec, inputs[0], arena_);
        if ((spec.operation == Operation::ADD || spec.operation == Operation::MULTIPLY) &&
            ElementwiseBinaryF32::supports(inputs))
            return std::make_unique<ElementwiseBinaryF32>(inputs, spec.operation == Operation::MULTIPLY, arena_);
        if ((spec.operation == Operation::SILU_GATE || spec.operation == Operation::GELU_GATE) &&
            GatedActivationOp::supports(inputs))
            return std::make_unique<GatedActivationOp>(inputs[0], spec.operation == Operation::SILU_GATE, arena_);
        if ((spec.operation == Operation::SILU_MULTIPLY || spec.operation == Operation::GELU_MULTIPLY) &&
            GatedMultiplyOp::supports(inputs))
            return std::make_unique<GatedMultiplyOp>(inputs[0], spec.operation == Operation::SILU_MULTIPLY, arena_);
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
                                    bits,
                                    false};
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
        if (SymmetricPackedLinear::supports(spec, inputs)) {
            const auto bits = static_cast<std::int32_t>(spec.attributes[0]);
            const auto block_size = static_cast<std::size_t>(spec.attributes[1]);
            const PackedTileKey key{inputs[1].storage_identity(),
                                    inputs[1].storage_offset(),
                                    inputs[2].storage_identity(),
                                    inputs[2].storage_offset(),
                                    nullptr,
                                    0,
                                    block_size,
                                    bits,
                                    true};
            auto& cached_tiles = packed_tile_cache_[key];
            if (!cached_tiles.defined()) {
                cached_tiles = SymmetricPackedLinear::pack_tiles(bits, inputs[1], inputs[2], block_size);
                if (inputs[1].size(0) < 65536) {
                    inputs[1].release_pages();
                    inputs[2].release_pages();
                }
            }
            return std::make_unique<SymmetricPackedLinear>(spec, inputs[0], inputs[1], cached_tiles, arena_);
        }
        if (spec.operation == Operation::BLOCK_SCALED_LINEAR)
            return std::make_unique<BlockScaledLinear>(spec, inputs, arena_);
        if (spec.operation == Operation::EXPERT_LINEAR && spec.attributes.size() >= 4 &&
            (spec.attributes[3] == static_cast<std::int64_t>(ops::BlockScaledFormat::VQ3D) ||
             spec.attributes[3] == static_cast<std::int64_t>(ops::BlockScaledFormat::VQ4D)))
            return std::make_unique<VqExpertOp>(spec, inputs, arena_);
        if (PackedFeedForward::supports(spec, inputs))
            return std::make_unique<PackedFeedForward>(spec, inputs, feed_forward_cache_, arena_);
        if (DenseLinearOp::supports(spec, inputs)) {
            PackedDenseEntry entry;
            if (!spec.dynamic_parameters && spec.operation == Operation::LINEAR && inputs[1].dimensions() == 2) {
                const bool transpose_b = spec.attributes[0] != 0;
                const auto k = static_cast<std::size_t>(inputs[0].size(-1));
                const auto n = static_cast<std::size_t>(transpose_b ? inputs[1].size(0) : inputs[1].size(1));
                const DenseTileKey key{inputs[1].storage_identity(),
                                       inputs[1].storage_offset(),
                                       n,
                                       k,
                                       transpose_b,
                                       inputs[1].dtype(),
                                       false};
                auto found = dense_tile_cache_.find(key);
                if (found == dense_tile_cache_.end()) {
                    const auto ynn_bt = inputs[1].dtype() == DType::BF16 ? ynn_type_bf16 : ynn_type_fp32;
                    ::ynn::dense_linear_plan plan{};
                    check(::ynn::make_dense_linear_plan(ynn_bt, ynn_bt, ynn_type_fp32, n, k, &plan));
                    auto packed = DenseLinearOp::pack_static_b(inputs[1], transpose_b, plan);
                    found = dense_tile_cache_.emplace(key, PackedDenseEntry{plan, std::move(packed), {}}).first;
                    if (inputs[1].size(0) < 65536) inputs[1].release_pages();
#if defined(__APPLE__)
                    if (inputs[1].nbytes() >= (std::size_t{8} << 20)) ::malloc_zone_pressure_relief(nullptr, 0);
#endif
                }
                entry = found->second;
            }
            return std::make_unique<DenseLinearOp>(spec, inputs, std::move(entry), arena_);
        }
        if (QuantizedLinearOp::supports(spec, inputs)) {
            PackedDenseEntry entry;
            if (!spec.dynamic_parameters) {
                const bool transpose_b = spec.attributes[0] != 0;
                const auto k = static_cast<std::size_t>(inputs[0].size(-1));
                const auto n = static_cast<std::size_t>(transpose_b ? inputs[1].size(0) : inputs[1].size(1));
                const DenseTileKey key{inputs[1].storage_identity(),
                                       inputs[1].storage_offset(),
                                       n,
                                       k,
                                       transpose_b,
                                       inputs[1].dtype(),
                                       true};
                auto found = dense_tile_cache_.find(key);
                if (found == dense_tile_cache_.end()) {
                    ::ynn::dense_linear_plan plan{};
                    check(::ynn::make_dense_linear_plan(ynn_type_int8, ynn_type_int8, ynn_type_int32, n, k, &plan));
                    auto [packed, sums] = QuantizedLinearOp::pack_static_b(inputs[1], transpose_b, plan);
                    found = dense_tile_cache_.emplace(key, PackedDenseEntry{plan, std::move(packed), std::move(sums)})
                                .first;
                    if (inputs[1].size(0) < 65536) inputs[1].release_pages();
#if defined(__APPLE__)
                    if (inputs[1].nbytes() >= (std::size_t{8} << 20)) ::malloc_zone_pressure_relief(nullptr, 0);
#endif
                }
                entry = found->second;
            }
            return std::make_unique<QuantizedLinearOp>(spec, inputs, std::move(entry), arena_);
        }
        if (spec.operation == Operation::ATTENTION) {
#if defined(__APPLE__)
            if (PytorchParityAttention::supports(spec, inputs))
                return std::make_unique<PytorchParityAttention>(spec, arena_);
#endif
            if (DirectAttentionOp::supports(spec, inputs)) return std::make_unique<DirectAttentionOp>(spec, arena_);
            const bool byte_cache = inputs[1].dtype() == DType::I8 && !spec.quantization[0].scales.empty() &&
                                    !spec.quantization[1].scales.empty();
            if (byte_cache && inputs[0].dtype() == DType::F32 && inputs[2].dtype() == inputs[1].dtype())
                return std::make_unique<QuantizedAttention>(spec, inputs, arena_);
        }
        if (BroadcastBinaryOp::supports(spec, inputs)) return std::make_unique<BroadcastBinaryOp>(spec, inputs, arena_);
        if (spec.operation == Operation::CAST && inputs.size() == 1)
            return std::make_unique<CastOp>(spec, inputs[0], arena_);
        if (LayerNormOp::supports(spec, inputs)) return std::make_unique<LayerNormOp>(spec, inputs, arena_);
        if ((spec.operation == Operation::SOFTMAX || spec.operation == Operation::LOG_SOFTMAX) && inputs.size() == 1 &&
            inputs[0].dimensions() > 0)
            return std::make_unique<SoftmaxOp>(spec, inputs[0], arena_);
        if ((spec.operation == Operation::GELU || spec.operation == Operation::TANH ||
             spec.operation == Operation::STATIC_ROUND) &&
            inputs.size() == 1)
            return std::make_unique<UnaryElementwiseOp>(spec, inputs[0], arena_);
        if (spec.operation == Operation::TRANSPOSE && inputs.size() == 1 &&
            spec.attributes.size() == inputs[0].dimensions())
            return std::make_unique<TransposeOp>(spec, inputs[0], arena_);
        if (spec.operation == Operation::GATHER && inputs.size() == 2 && !spec.attributes.empty() &&
            (inputs[1].dtype() == DType::I32 || inputs[1].dtype() == DType::I64))
            return std::make_unique<GatherOp>(spec, inputs, arena_);
        if (spec.operation == Operation::CONCAT && inputs.size() > 0 && !spec.attributes.empty())
            return std::make_unique<ConcatOp>(spec, inputs, arena_);
        if (spec.operation == Operation::PACKED_LINEAR && (inputs.size() == 3 || inputs.size() == 4) &&
            spec.attributes.size() >= 3)
            return std::make_unique<PortablePackedLinearOp>(spec, inputs, arena_);
        throw ops::Failure({ErrorCode::UNSUPPORTED,
                            "CPU backend does not support operation: " + std::string(operation_name(spec.operation))});
    }

private:
    tensor::Arena arena_;
    ynn::FeedForwardCache feed_forward_cache_;
    std::map<PackedTileKey, Tensor> packed_tile_cache_;
    std::map<DenseTileKey, PackedDenseEntry> dense_tile_cache_;
};
} // namespace
auto cpu_operators() -> std::unique_ptr<OperatorBackend> { return std::make_unique<CpuBackend>(); }
} // namespace kidi::runtime