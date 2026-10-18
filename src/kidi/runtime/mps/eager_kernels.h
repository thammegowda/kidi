#pragma once

#include "kidi/runtime/operator.h"
#include "kidi/runtime/mps/command_batch.h"

namespace kidi::runtime::mps {
auto encode_greedy_token(CommandBatch& batch, const tensor::Tensor& input, tensor::Tensor& scratch,
                         tensor::Tensor& output) -> Result<void>;
auto encode_quantize_int8(CommandBatch& batch, float scale, const tensor::Tensor& input, tensor::Tensor& output)
    -> Result<void>;
auto encode_eager(CommandBatch& batch, Operation operation, float epsilon, TensorInputs inputs, tensor::Tensor& output)
    -> Result<void>;
auto encode_audio_unfold(CommandBatch& batch, const tensor::Tensor& input, const tensor::Tensor* alpha,
                         tensor::Tensor& output, std::int32_t kernel, std::int32_t stride, std::int32_t padding,
                         std::int32_t dilation) -> Result<void>;
auto encode_snake(CommandBatch& batch, const tensor::Tensor& input, const tensor::Tensor& alpha,
                  tensor::Tensor& output) -> Result<void>;
auto encode_conv_transpose_fold(CommandBatch& batch, const tensor::Tensor& projected, const tensor::Tensor& bias,
                                tensor::Tensor& output, std::int32_t kernel, std::int32_t stride,
                                std::int32_t padding, std::int32_t dilation) -> Result<void>;
} // namespace kidi::runtime::mps