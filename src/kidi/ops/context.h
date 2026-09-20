#pragma once

#include <memory>
#include <array>
#include <stdexcept>
#include "kidi/tensor/tensor.h"

namespace kidi::ops {
using tensor::Tensor;

/// Execution mode for the current call; public operations scope and restore it.
extern thread_local bool is_inplace;

class Failure : public std::runtime_error {
public:
    explicit Failure(Error error) : std::runtime_error(error.message), error_(std::move(error)) {}
    auto error() const noexcept -> const Error& { return error_; }

private:
    Error error_;
};

template <typename Type>
auto require(Result<Type> result) -> Type {
    if (!result) throw Failure(std::move(result.error()));
    return std::move(*result);
}
inline auto require(Result<void> result) -> void {
    if (!result) throw Failure(std::move(result.error()));
}

/// Eager tensor operations: func preserves inputs; func_ mutates and returns the first tensor.
/// In-place calls preserve shape, dtype, and storage. Synchronize before host reads on Metal.
class Context {
public:
    explicit Context(tensor::Device device = tensor::Device::cpu());
    ~Context();
    Context(Context&&) noexcept;
    auto operator=(Context&&) noexcept -> Context&;
    Context(const Context&) = delete;
    auto operator=(const Context&) -> Context& = delete;
    auto device() const noexcept -> tensor::Device;
    auto synchronize() -> void;
    auto preparation_ns() const noexcept -> std::uint64_t;
    auto profile_phase(std::string_view phase) -> void;
    auto add(const Tensor& left, const Tensor& right) -> Tensor;
    auto add_(Tensor& left, const Tensor& right) -> Tensor&;
    auto multiply(const Tensor& left, const Tensor& right) -> Tensor;
    auto multiply_(Tensor& left, const Tensor& right) -> Tensor&;
    auto cast(const Tensor& input, tensor::DType dtype) -> Tensor;
    auto matmul(const Tensor& left, const Tensor& right, bool transpose_right = false) -> Tensor;
    auto scaled_dot_product_attention(const Tensor& query, const Tensor& key, const Tensor& value, std::int32_t heads,
                                      const Tensor& mask = {}) -> Tensor;
    auto grouped_query_attention(const Tensor& query, const Tensor& key, const Tensor& value, std::int32_t heads,
                                 std::int32_t key_value_heads, const Tensor& mask, float scale = 1.F) -> Tensor;
    auto rotary(const Tensor& input, const Tensor& cosine, const Tensor& sine) -> Tensor;
    auto linear(const Tensor& input, const Tensor& weight, const Tensor& bias, bool transpose_weight = false) -> Tensor;
    auto quantized_linear(const Tensor& input, const Tensor& weight, const Tensor& scale, const Tensor& bias) -> Tensor;
    auto gelu(const Tensor& input, bool approximate = false) -> Tensor;
    auto gelu_(Tensor& input, bool approximate = false) -> Tensor&;
    auto tanh(const Tensor& input) -> Tensor;
    auto rms_norm(const Tensor& input, const Tensor& scale, float epsilon) -> Tensor;
    auto layer_norm(const Tensor& input, const Tensor& scale, const Tensor& bias, float epsilon) -> Tensor;
    auto layer_norm_(Tensor& input, const Tensor& scale, const Tensor& bias, float epsilon) -> Tensor&;
    auto residual_layer_norm(const Tensor& input, const Tensor& residual, const Tensor& scale, const Tensor& bias,
                             float epsilon) -> std::array<Tensor, 2>;
    auto softmax(const Tensor& input, bool logarithmic = false) -> Tensor;
    auto softmax_(Tensor& input, bool logarithmic = false) -> Tensor&;
    auto transpose(const Tensor& input, std::span<const std::int64_t> axes) -> Tensor;
    auto slice(const Tensor& input, std::int64_t axis, std::int64_t start, std::int64_t length) -> Tensor;
    auto gather(const Tensor& input, const Tensor& indices, std::int64_t axis = 0) -> Tensor;
    auto concat(std::span<const Tensor> inputs, std::int64_t axis) -> Tensor;
    auto scatter(const Tensor& input, const Tensor& updates, const Tensor& indices) -> Tensor;
    auto scatter_(Tensor& input, const Tensor& updates, const Tensor& indices) -> Tensor&;
    auto reshape(const Tensor& input, std::vector<std::int64_t> shape) -> Tensor;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace kidi::ops