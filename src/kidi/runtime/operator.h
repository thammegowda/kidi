#pragma once
#include <array>
#include <memory>
#include <vector>
#include <initializer_list>
#include "kidi/tensor/tensor.h"
#include "kidi/tensor/arena.h"

namespace kidi::runtime {
/// Borrowed operands, either a contiguous tensor span or a list of tensor addresses.
class TensorInputs {
public:
    TensorInputs(std::span<const tensor::Tensor> inputs) : contiguous_(inputs.data()), size_(inputs.size()) {}
    TensorInputs(std::initializer_list<const tensor::Tensor*> inputs)
        : indirect_(inputs.begin()), size_(inputs.size()) {}
    template <std::size_t Size>
    TensorInputs(const std::array<tensor::Tensor, Size>& inputs)
        : TensorInputs(std::span<const tensor::Tensor>(inputs)) {}
    TensorInputs(const std::vector<tensor::Tensor>& inputs) : TensorInputs(std::span<const tensor::Tensor>(inputs)) {}
    auto operator[](std::size_t index) const -> const tensor::Tensor& {
        return indirect_ ? *indirect_[index] : contiguous_[index];
    }
    auto front() const -> const tensor::Tensor& { return (*this)[0]; }
    auto back() const -> const tensor::Tensor& { return (*this)[size_ - 1]; }
    auto size() const noexcept -> std::size_t { return size_; }
    auto first(std::size_t count) const -> TensorInputs {
        auto result = *this;
        result.size_ = count;
        return result;
    }

private:
    const tensor::Tensor* contiguous_ = nullptr;
    const tensor::Tensor* const* indirect_ = nullptr;
    std::size_t size_;
};

enum class Operation {
    ADD,
    MULTIPLY,
    CAST,
    MATMUL,
    LINEAR,
    QUANTIZED_LINEAR,
    GELU,
    LAYER_NORM,
    SOFTMAX,
    LOG_SOFTMAX,
    TRANSPOSE,
    SLICE,
    GATHER,
    CONCAT,
    SCATTER,
    ATTENTION,
    RESIDUAL_NORM,
    RMS_NORM,
    TANH,
    ROTARY,
    PACKED_LINEAR,
    RMS_NORM_RESIDUAL,
    STATIC_ROUND,
    GREEDY_TOKEN,
    RMS_ROTARY,
    GELU_MULTIPLY
};
struct OperatorSpec {
    Operation operation;
    std::span<const std::int64_t> attributes;
    tensor::DType dtype = tensor::DType::F32;
    float epsilon = 0;
    bool dynamic_parameters = false;
    bool packed_prefill = false;
    bool vector_projection = false;
};

struct AllocationStats {
    std::uint64_t count = 0;
    std::uint64_t bytes = 0;
};

class Operator {
public:
    virtual ~Operator() = default;
    virtual auto run(TensorInputs inputs) -> tensor::Tensor = 0;
    virtual auto run_(TensorInputs inputs, tensor::Tensor& destination) -> tensor::Tensor = 0;
    virtual auto run_pair(TensorInputs inputs) -> std::array<tensor::Tensor, 2> {
        return {tensor::Tensor{}, run(inputs)};
    }
    virtual auto allocations() const -> AllocationStats { return {}; }
};
class OperatorBackend {
public:
    virtual ~OperatorBackend() = default;
    virtual auto prepare(const OperatorSpec&, TensorInputs) -> std::unique_ptr<Operator> = 0;
    virtual auto synchronize() -> void = 0;
    virtual auto release_cached_buffers() -> void {}
    virtual auto copy_slice_(tensor::Tensor& destination, const tensor::Tensor& source, std::size_t outer,
                             std::size_t source_bytes, std::size_t destination_bytes, std::size_t offset_bytes)
        -> void = 0;
};
class OutputPool {
public:
    explicit OutputPool(tensor::Arena* arena = nullptr) : arena_(arena) { buffers_.reserve(32); }
    auto acquire(std::span<const std::int64_t> shape, tensor::DType dtype, tensor::Device device) -> tensor::Tensor;
    auto allocations() const -> AllocationStats { return allocations_; }

private:
    tensor::Arena* arena_;
    std::vector<tensor::Tensor> buffers_;
    AllocationStats allocations_;
};
auto cpu_operators() -> std::unique_ptr<OperatorBackend>;
auto metal_operators() -> std::unique_ptr<OperatorBackend>;
} // namespace kidi::runtime