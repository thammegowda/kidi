#include "kidi/runtime/operator.h"

#include <algorithm>
#include <array>

#include "kidi/ops/context.h"

namespace kidi::runtime {

auto Operator::run_into(TensorInputs, std::span<tensor::Tensor>) -> void {
    throw ops::Failure({ErrorCode::UNSUPPORTED, "this backend cannot replay captured steps"});
}

#if !defined(KIDI_HAS_QNN)
auto npu_step_compiler() -> Result<std::shared_ptr<StepCompiler>> {
    return std::unexpected(Error{ErrorCode::UNSUPPORTED, "the Qualcomm NPU backend is not built"});
}
#endif

auto operation_name(Operation operation) -> std::string_view {
    constexpr std::array names{"add",           "multiply",
                               "cast",          "matmul",
                               "linear",        "quantized_linear",
                               "gelu",          "layer_norm",
                               "softmax",       "log_softmax",
                               "transpose",     "slice",
                               "gather",        "concat",
                               "scatter",       "attention",
                               "residual_norm", "rms_norm",
                               "tanh",          "rotary",
                               "packed_linear", "rms_norm_residual",
                               "static_round",  "greedy_token",
                               "rms_rotary",    "gelu_multiply",
                               "embedding",     "gated_feed_forward"};
    return names.at(static_cast<std::size_t>(operation));
}

auto OutputPool::acquire(std::span<const std::int64_t> shape, tensor::DType dtype,
                         tensor::Device device) -> tensor::Tensor {
    for (const auto& buffer : buffers_)
        if (buffer.owns_unique_storage() && buffer.dtype() == dtype && buffer.device() == device &&
            std::ranges::equal(buffer.shape(), shape))
            return buffer;
    auto output = ops::require(arena_ ? arena_->allocate({shape.begin(), shape.end()}, dtype)
                                      : tensor::Tensor::empty({shape.begin(), shape.end()}, dtype, device));
    ++allocations_.count;
    allocations_.bytes += output.nbytes();
    buffers_.push_back(output);
    return output;
}

} // namespace kidi::runtime