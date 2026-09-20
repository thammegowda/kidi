#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

#include "kidi/core/error.h"
#include "kidi/tensor/dtype.h"
#include "kidi/tensor/tensor.h"
#include "kidi/runtime/operator.h"

namespace kidi::runtime::mps {

class CommandBatch;

class Value {
public:
    Value() = default;

    auto valid() const noexcept -> bool;

private:
    explicit Value(std::uint32_t id) noexcept : id_(id) {}

    static constexpr std::uint32_t INVALID = UINT32_MAX;
    std::uint32_t id_ = INVALID;

    friend class Graph;
};

class Executable {
public:
    Executable(Executable&&) noexcept;
    auto operator=(Executable&&) noexcept -> Executable&;
    ~Executable();

    Executable(const Executable&) = delete;
    auto operator=(const Executable&) -> Executable& = delete;

    auto encode(CommandBatch& batch, TensorInputs inputs, std::span<tensor::Tensor> outputs) const -> Result<void>;
    auto output_shapes() const -> Result<std::vector<std::vector<std::int64_t>>>;

private:
    struct Impl;
    explicit Executable(std::unique_ptr<Impl> impl) noexcept;

    std::unique_ptr<Impl> impl_;

    friend class Graph;
};

class Graph {
public:
    Graph(Graph&&) noexcept;
    auto operator=(Graph&&) noexcept -> Graph&;
    ~Graph();

    Graph(const Graph&) = delete;
    auto operator=(const Graph&) -> Graph& = delete;

    static auto create() -> Result<Graph>;

    auto placeholder(std::span<const std::int64_t> shape, tensor::DType dtype, std::string_view name = {}) -> Value;
    auto placeholder(std::initializer_list<std::int64_t> shape, tensor::DType dtype, std::string_view name = {})
        -> Value {
        return placeholder(std::span<const std::int64_t>(shape.begin(), shape.size()), dtype, name);
    }
    auto constant(const tensor::Tensor& value, std::string_view name = {}) -> Value;
    auto scalar(double value, tensor::DType dtype) -> Value;

    auto cast(Value value, tensor::DType dtype, std::string_view name = {}) -> Value;
    auto add(Value left, Value right, std::string_view name = {}) -> Value;
    auto subtract(Value left, Value right, std::string_view name = {}) -> Value;
    auto multiply(Value left, Value right, std::string_view name = {}) -> Value;
    auto exp(Value value, std::string_view name = {}) -> Value;
    auto log(Value value, std::string_view name = {}) -> Value;
    auto erf(Value value, std::string_view name = {}) -> Value;
    auto tanh(Value value, std::string_view name = {}) -> Value;

    auto matmul(Value left, Value right, bool transpose_left = false, bool transpose_right = false,
                std::string_view name = {}) -> Value;
    auto reshape(Value value, std::span<const std::int64_t> shape, std::string_view name = {}) -> Value;
    auto reshape(Value value, std::initializer_list<std::int64_t> shape, std::string_view name = {}) -> Value {
        return reshape(value, std::span<const std::int64_t>(shape.begin(), shape.size()), name);
    }
    auto transpose(Value value, std::span<const std::int64_t> permutation, std::string_view name = {}) -> Value;
    auto slice(Value value, std::int64_t axis, std::int64_t start, std::int64_t length, std::string_view name = {})
        -> Value;
    auto concat(std::span<const Value> values, std::int64_t axis, std::string_view name = {}) -> Value;
    auto gather(Value values, Value indices, std::int64_t axis, std::string_view name = {}) -> Value;
    auto scatter(Value data, Value updates, Value indices, std::int64_t axis, std::string_view name = {}) -> Value;

    auto reduce_sum(Value value, std::span<const std::int64_t> axes, std::string_view name = {}) -> Value;
    auto reduce_max(Value value, std::span<const std::int64_t> axes, std::string_view name = {}) -> Value;
    auto softmax(Value value, std::int64_t axis, std::string_view name = {}) -> Value;
    auto normalize(Value value, Value mean, Value variance, Value scale, Value bias, float epsilon,
                   std::string_view name = {}) -> Value;

    auto compile(std::span<const Value> inputs, std::span<const Value> outputs) const -> Result<Executable>;

private:
    struct Impl;
    explicit Graph(std::shared_ptr<Impl> impl) noexcept;

    std::shared_ptr<Impl> impl_;
};

} // namespace kidi::runtime::mps