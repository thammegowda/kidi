#pragma once

#include "kidi/runtime/operator.h"
#include "kidi/runtime/mps/command_batch.h"

namespace kidi::runtime::mps {
auto encode_greedy_token(CommandBatch& batch, const tensor::Tensor& input, tensor::Tensor& scratch,
                         tensor::Tensor& output) -> Result<void>;
auto encode_eager(CommandBatch& batch, Operation operation, float epsilon, TensorInputs inputs, tensor::Tensor& output)
    -> Result<void>;
} // namespace kidi::runtime::mps