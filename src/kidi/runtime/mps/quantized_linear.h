#pragma once

#include <memory>
#include <functional>
#include <string>
#include "kidi/tensor/tensor.h"
#include "kidi/runtime/mps/command_batch.h"

namespace kidi::runtime::mps {

class QuantizedLinear {
public:
    QuantizedLinear(QuantizedLinear&&) noexcept;
    auto operator=(QuantizedLinear&&) noexcept -> QuantizedLinear&;
    ~QuantizedLinear();
    static auto create(const tensor::Tensor& weights, const tensor::Tensor& scales, const tensor::Tensor& bias)
        -> Result<QuantizedLinear>;
    auto prepare(const tensor::Tensor& input) -> Result<void>;
    auto encode(CommandBatch& batch, const tensor::Tensor& input, tensor::Tensor& output) -> Result<void>;
    auto run(const tensor::Tensor& input, tensor::Tensor& output) -> Result<void>;

private:
    struct Impl;
    explicit QuantizedLinear(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};
} // namespace kidi::runtime::mps