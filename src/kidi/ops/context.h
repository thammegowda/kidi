#pragma once

#include <memory>
#include <array>
#include <functional>
#include <stdexcept>
#include <string_view>
#include "kidi/tensor/tensor.h"

namespace kidi::runtime {
class StepCompiler;
}

namespace kidi::ops {
class DecodeScope {
public:
    explicit DecodeScope(bool enabled);
    ~DecodeScope();
    DecodeScope(const DecodeScope&) = delete;
    auto operator=(const DecodeScope&) -> DecodeScope& = delete;

private:
    bool previous_;
};
/// Attaches `compiler` to CPU contexts constructed on this thread while the scope is alive, so their captured
/// steps run on an accelerator. Scopes nest and restore the previous compiler.
class StepCompilerScope {
public:
    explicit StepCompilerScope(std::shared_ptr<runtime::StepCompiler> compiler);
    ~StepCompilerScope();
    StepCompilerScope(const StepCompilerScope&) = delete;
    auto operator=(const StepCompilerScope&) -> StepCompilerScope& = delete;

private:
    std::shared_ptr<runtime::StepCompiler> previous_;
};
using tensor::Tensor;

struct BlockwiseQuantization {
    std::vector<float> scales;
    std::vector<std::int32_t> zero_points;
    std::size_t block_size = 0;
};

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
    /// One fixed-shape step body: ordinary eager operations over `inputs`, returning the step outputs.
    using Step = std::function<std::vector<Tensor>(std::span<const Tensor> inputs)>;

    explicit Context(tensor::Device device = tensor::Device::cpu(), bool packed_prefill = false);
    ~Context();
    Context(Context&&) noexcept;
    auto operator=(Context&&) noexcept -> Context&;
    Context(const Context&) = delete;
    auto operator=(const Context&) -> Context& = delete;
    auto device() const noexcept -> tensor::Device;
    auto synchronize() -> void;
    auto preparation_ns() const noexcept -> std::uint64_t;
    auto profile_phase(std::string_view phase) -> void;

    /// Runs `step`, capturing its operators the first two times `key` is used and replaying them afterwards.
    /// A step reads only `inputs`, parameters, and tensors whose storage stays fixed; values that change per call
    /// are written into `inputs` before calling. The two captures must match, which rejects host-computed
    /// temporaries and per-call operator arguments. Replay may rebind inputs of the same shape and dtype. Outputs
    /// belong to the captured step and are overwritten by the next call with the same key. Backends without replay,
    /// or `KIDI_REPLAY=0`, run `step` eagerly on every call.
    auto replay(std::string_view key, std::span<const Tensor> inputs, const Step& step) -> std::span<const Tensor>;
    auto replay_enabled() const noexcept -> bool;
    auto clear_replays() -> void;
    auto release_workspaces() -> void;
    /// Evicts captured steps whose key begins with `prefix`, for example completed prefill shapes before decode.
    auto clear_replays(std::string_view prefix) -> void;
    /// Name of the accelerator compiling captured steps, or empty when steps replay on this context's device.
    auto accelerator() const noexcept -> std::string_view;
    /// The attached step compiler, e.g. to keep it when a model re-creates its context.
    auto step_compiler() const noexcept -> std::shared_ptr<runtime::StepCompiler>;
    /// Attention key extent for a captured step needing `required` keys from a cache of `capacity`.
    auto step_extent(std::size_t required, std::size_t capacity) const -> std::size_t;
    /// Whether captured steps may crop local-attention keys (each crop offset is a separate step shape).
    auto crop_local_attention() const noexcept -> bool;
    auto prefill_chunk_size(std::size_t requested) const -> std::size_t;
    auto prepare_linear_weights(const Tensor& weight, std::int32_t bits, std::int32_t group_size = 128,
                                bool packed_prefill = true) -> void;
    auto add(const Tensor& left, const Tensor& right) -> Tensor;
    auto add_(Tensor& left, const Tensor& right) -> Tensor&;
    auto multiply(const Tensor& left, const Tensor& right) -> Tensor;
    auto multiply_(Tensor& left, const Tensor& right) -> Tensor&;
    auto cast(const Tensor& input, tensor::DType dtype, float scale = 0) -> Tensor;
    auto matmul(const Tensor& left, const Tensor& right, bool transpose_right = false) -> Tensor;
    auto scaled_dot_product_attention(const Tensor& query, const Tensor& key, const Tensor& value, std::int32_t heads,
                                      const Tensor& mask = {}) -> Tensor;
    auto grouped_query_attention(const Tensor& query, const Tensor& key, const Tensor& value, std::int32_t heads,
                                 std::int32_t key_value_heads, const Tensor& mask, float scale = 1.F,
                                 std::int64_t key_start = 0, const BlockwiseQuantization& key_quantization = {},
                                 const BlockwiseQuantization& value_quantization = {}) -> Tensor;
    auto rotary(const Tensor& input, const Tensor& cosine, const Tensor& sine) -> Tensor;
    auto linear(const Tensor& input, const Tensor& weight, const Tensor& bias, bool transpose_weight = false) -> Tensor;
    auto quantized_linear(const Tensor& input, const Tensor& weight, const Tensor& scale, const Tensor& bias,
                          bool transpose_weight = false) -> Tensor;
    auto packed_linear(const Tensor& input, const Tensor& weight, const Tensor& scales, std::int32_t bits,
                       std::int32_t group_size, float input_scale = 0.F, float output_scale = 0.F) -> Tensor;
    auto gelu(const Tensor& input, bool approximate = false) -> Tensor;
    auto gelu_multiply(const Tensor& gate, const Tensor& value) -> Tensor;
    auto gated_feed_forward(const Tensor& input, const Tensor& gate_up_weight, const Tensor& gate_up_scales,
                            const Tensor& down_weight, const Tensor& down_scales, std::int32_t bits,
                            std::int32_t input_size, std::int32_t intermediate_size, float gate_up_input_scale,
                            float gate_up_output_scale, float down_input_scale, float down_output_scale) -> Tensor;
    auto gelu_(Tensor& input, bool approximate = false) -> Tensor&;
    auto tanh(const Tensor& input) -> Tensor;
    auto static_round(const Tensor& input, float scale) -> Tensor;
    auto greedy_token(const Tensor& logits) -> Tensor;
    auto embedding(const Tensor& indices, const Tensor& weight, const Tensor& scales, std::int32_t width,
                   std::int32_t bits, float multiplier) -> Tensor;
    auto rms_norm(const Tensor& input, const Tensor& scale, float epsilon) -> Tensor;
    auto rms_rotary(const Tensor& input, const Tensor& scale, const Tensor& cosine, const Tensor& sine,
                    float epsilon) -> Tensor;
    auto rms_norm_residual(const Tensor& input, const Tensor& scale, const Tensor& residual, float epsilon,
                           const Tensor& output_scale = {}) -> Tensor;
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
    auto copy_slice_(Tensor& destination, const Tensor& source, std::int64_t axis, std::int64_t start) -> Tensor&;
    auto reshape(const Tensor& input, std::vector<std::int64_t> shape) -> Tensor;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace kidi::ops