#pragma once

#include <filesystem>
#include <memory>
#include <array>
#include <functional>
#include <stdexcept>
#include <string_view>
#include "kidi/core/precision.h"
#include "kidi/ops/quantization.h"
#include "kidi/tensor/tensor.h"

namespace kidi::graph {
class Compiler;
}

namespace kidi::ops {
enum class TokenOutput { HOST, DEVICE };
class DecodeScope {
public:
    explicit DecodeScope(bool enabled);
    DecodeScope(bool attention, bool projections);
    ~DecodeScope();
    DecodeScope(const DecodeScope&) = delete;
    auto operator=(const DecodeScope&) -> DecodeScope& = delete;

private:
    bool previous_attention_, previous_projections_;
};
/// Attaches `compiler` to CPU contexts constructed on this thread while the scope is alive, so their captured
/// steps run on an accelerator. Scopes nest and restore the previous compiler.
class StepCompilerScope {
public:
    explicit StepCompilerScope(std::shared_ptr<graph::Compiler> compiler);
    ~StepCompilerScope();
    StepCompilerScope(const StepCompilerScope&) = delete;
    auto operator=(const StepCompilerScope&) -> StepCompilerScope& = delete;

private:
    std::shared_ptr<graph::Compiler> previous_;
};
/// Contexts constructed on this thread while the scope is alive run captured steps eagerly, even with an attached
/// step compiler; tests use them as the reference for replayed steps.
class EagerScope {
public:
    EagerScope();
    ~EagerScope();
    EagerScope(const EagerScope&) = delete;
    auto operator=(const EagerScope&) -> EagerScope& = delete;

private:
    bool previous_;
};
using tensor::Tensor;

struct BlockwiseQuantization {
    std::vector<float> scales;
    std::vector<std::int32_t> zero_points;
    std::size_t block_size = 0;
};

/// Work applied to a projection's input before it is quantized. CUDA prefill fuses it into the projection's
/// activation quantization; other paths run it as a separate operation first.
struct ProjectionPrologue {
    enum class Kind { NONE, RMS_NORM, GELU_GATE };
    Kind kind = Kind::NONE;
    const Tensor* scale = nullptr; ///< RMS-norm weight.
    float epsilon = 0;             ///< RMS-norm epsilon.
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

/// Operator profiling for developer tools; profiles print to stderr. HOST times operators as dispatched, SYNC also
/// waits for each operator, and CUDA adds GRAPH (captured-step phases) and EVENTS (per-node captured-step events).
enum class ProfileMode : std::uint8_t { OFF, HOST, SYNC, GRAPH, EVENTS };
struct ProfileOptions {
    ProfileMode mode = ProfileMode::OFF;
    std::size_t skip_requests = 0;   ///< Leading requests, such as warm-ups, excluded from HOST and SYNC profiles.
    bool log_graph_captures = false; ///< Reports each CUDA graph capture on stderr.
};
/// Process-wide; set before creating contexts.
auto set_profile_options(ProfileOptions options) -> void;
auto profile_options() -> ProfileOptions;
auto parse_profile_mode(std::string_view name) -> std::optional<ProfileMode>;
/// Upper bound on expanded FP16 weights Metal keeps for faster prefill (default: unlimited). Process-wide; set
/// before creating contexts. Projections beyond the bound use the slower packed path.
auto set_metal_prefill_cache_bytes(std::size_t bytes) -> void;
auto metal_prefill_cache_bytes() -> std::size_t;

/// Eager tensor operations: func preserves inputs; func_ mutates and returns the first tensor.
/// In-place calls preserve shape, dtype, and storage. Synchronize before host reads on Metal.
class Context {
public:
    /// One fixed-shape step body: ordinary eager operations over `inputs`, returning the step outputs.
    using Step = std::function<std::vector<Tensor>(std::span<const Tensor> inputs)>;

    explicit Context(tensor::Device device = tensor::Device::cpu(), bool packed_prefill = false,
                     std::optional<bool> replay_enabled = std::nullopt,
                     tensor::DType parameter_dtype = tensor::DType::F32);
    ~Context();
    Context(Context&&) noexcept;
    auto operator=(Context&&) noexcept -> Context&;
    Context(const Context&) = delete;
    auto operator=(const Context&) -> Context& = delete;
    auto device() const noexcept -> tensor::Device;
    auto activation_dtype() const noexcept -> tensor::DType;
    auto stage_inputs(std::span<const Tensor> inputs) -> std::vector<Tensor>;
    auto execution() const -> std::string;
    auto synchronize() -> void;
    auto preparation_ns() const noexcept -> std::uint64_t;
    auto configure_cache(const std::filesystem::path& directory, const std::filesystem::path& source) -> void;
    auto set_precision(core::InferencePrecision precision) noexcept -> void;
    auto precision() const noexcept -> core::InferencePrecision;
    auto profile_phase(std::string_view phase) -> void;

    /// Runs `step`, normally capturing its operators twice to validate structure before replay.
    /// A step reads only `inputs`, parameters, and tensors whose storage stays fixed; values that change per call
    /// are written into `inputs` before calling. The two captures must match, which rejects host-computed
    /// temporaries and per-call operator arguments. Replay may rebind inputs of the same shape and dtype. Outputs
    /// belong to the captured step and are overwritten by the next call with the same key. Backends without replay,
    /// or `KIDI_REPLAY=0`, run `step` eagerly on every call. Trusted operator-only steps may disable validation.
    auto replay(std::string_view key, std::span<const Tensor> inputs, const Step& step, bool validate_structure = true)
        -> std::span<const Tensor>;
    auto replay_enabled() const noexcept -> bool;
    auto set_replay_capacity(std::size_t capacity) -> void;
    auto clear_replays() -> void;
    auto release_workspaces() -> void;
    /// Drops unreferenced output caches while retaining bounded prepared operators and captured graphs.
    auto trim_workspaces() -> void;
    /// Evicts captured steps whose key begins with `prefix`, for example completed prefill shapes before decode.
    auto clear_replays(std::string_view prefix) -> void;
    /// Name of the accelerator compiling captured steps, or empty when steps replay on this context's device.
    auto accelerator() const noexcept -> std::string_view;
    /// The attached step compiler, e.g. to keep it when a model re-creates its context.
    auto step_compiler() const noexcept -> std::shared_ptr<graph::Compiler>;
    /// Attention key extent for a captured step needing `required` keys from a cache of `capacity`.
    auto step_extent(std::size_t required, std::size_t capacity) const -> std::size_t;
    /// Whether captured steps may crop local-attention keys (each crop offset is a separate step shape).
    auto crop_local_attention() const noexcept -> bool;
    auto prefill_chunk_size(std::size_t requested) const -> std::size_t;
    auto prepare_linear_weights(const Tensor& weight, std::int32_t bits, std::int32_t group_size = 128,
                                bool packed_prefill = true) -> void;
    auto bind_packed_zero(const Tensor& scales, Tensor zeros) -> void;
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
    /// CUDA causal attention uses I32 query positions shaped [1, 1, queries, 1]; window 0 means unbounded history.
    auto grouped_query_attention_causal(const Tensor& query, const Tensor& key, const Tensor& value,
                                        std::int32_t heads, std::int32_t key_value_heads,
                                        const Tensor& query_positions, std::int64_t window, float scale = 1.F,
                                        std::int64_t key_start = 0,
                                        const BlockwiseQuantization& key_quantization = {},
                                        const BlockwiseQuantization& value_quantization = {}) -> Tensor;
    auto rotary(const Tensor& input, const Tensor& cosine, const Tensor& sine) -> Tensor;
    auto linear(const Tensor& input, const Tensor& weight, const Tensor& bias, bool transpose_weight = false) -> Tensor;
    auto quantized_linear(const Tensor& input, const Tensor& weight, const Tensor& scale, const Tensor& bias,
                          bool transpose_weight = false) -> Tensor;
    auto packed_linear(const Tensor& input, const Tensor& weight, const Tensor& scales, std::int32_t bits,
                       std::int32_t group_size, float input_scale = 0.F, float output_scale = 0.F,
                       const Tensor& zeros = {}, const ProjectionPrologue& prologue = {}) -> Tensor;
    /// True when packed projections of `input` can fuse `prologue`; otherwise each one runs it separately.
    auto fuses(const Tensor& input, const ProjectionPrologue& prologue) const -> bool;
    /// Runs a projection prologue as its own operation.
    auto apply(const Tensor& input, const ProjectionPrologue& prologue) -> Tensor;
    auto gelu(const Tensor& input, bool approximate = false) -> Tensor;
    auto gelu_multiply(const Tensor& gate, const Tensor& value) -> Tensor;
    auto silu_multiply(const Tensor& gate, const Tensor& value) -> Tensor;
    auto gelu_gate(const Tensor& gate_up) -> Tensor;
    auto silu_gate(const Tensor& gate_up, float gate_cap = 0.F, float up_cap = 0.F, const Tensor& row_weights = {})
        -> Tensor;
    /// Expands NVFP4 (U8, E4M3 block scales and FP32 expert scales) or MXFP8 (E4M3, U8 exponents) to
    /// BF16 [experts, rows, columns] for reference comparisons. Native inference does not expand or rearrange weights.
    auto dequantize_experts(const Tensor& values, const Tensor& scales, const Tensor& global = {},
                            bool interleaved = false) -> Tensor;
    /// Packs decoded expert weights directly to signed INT8 with FP32 per-row scales, without BF16 expansion.
    /// Uses pack_weight's rounding convention and rejects non-finite values. CPU or CUDA.
    auto pack_experts(const Tensor& values, const Tensor& scales, const Tensor& global = {}, bool interleaved = false)
        -> PackedWeight;
    /// CPU/CUDA projection from unchanged FP4 E2M1, FP8 E4M3 or symmetric signed INT4 block-scaled weights.
    /// Values have shape [rows, columns / 2] for 4-bit formats, [rows, columns] for FP8.
    /// Group size is configurable; a scalar FP32 global scale is optional. No persistent floating expansion.
    auto block_scaled_linear(const Tensor& input, const BlockScaledWeight& weight) -> Tensor;
    /// Consumes adjacent gate/up row pairs directly and fuses softcapped SiLU and per-token routing weights.
    auto block_scaled_silu(const Tensor& input, const BlockScaledWeight& weight, const Tensor& row_weights,
                           float gate_cap, float up_cap) -> Tensor;
    /// CUDA top-k routing without a host round trip: ranks sigmoid(logits) + bias and normalizes the selected
    /// sigmoid scores to sum to `scale`. Returns [tokens, top, 2] FP32 rows of (expert index, routing weight).
    auto route_experts(const Tensor& logits, const Tensor& bias, std::int32_t top, float scale) -> Tensor;
    /// Routed gate/up projection over stacked experts (values [experts, rows, packed]): CUDA block-scaled formats or
    /// CPU/Metal VQ3D/VQ4D. Reads each selected expert's interleaved gate/up row pairs in place. Returns
    /// [tokens, slots, rows / 2] activations that include the softcapped SiLU gate and routing weight. The global
    /// scale, when present, holds one FP32 per expert (one per expert row for VQ formats).
    auto expert_silu(const Tensor& input, const BlockScaledWeight& weight, const Tensor& routes, float gate_cap,
                     float up_cap) -> Tensor;
    /// Routed down projection (CUDA block-scaled formats or CPU/Metal VQ3D/VQ4D) of [tokens, slots, width]
    /// activations; returns [tokens, rows] summed over slots (FP32 by default, or BF16 when requested).
    auto expert_linear(const Tensor& input, const BlockScaledWeight& weight, const Tensor& routes,
                       std::optional<tensor::DType> output_dtype = {}) -> Tensor;
    auto gated_feed_forward(const Tensor& input, const Tensor& gate_up_weight, const Tensor& gate_up_scales,
                            const Tensor& down_weight, const Tensor& down_scales, std::int32_t bits,
                            std::int32_t input_size, std::int32_t intermediate_size, float gate_up_input_scale,
                            float gate_up_output_scale, float down_input_scale, float down_output_scale,
                            bool silu = false) -> Tensor;
    auto gelu_(Tensor& input, bool approximate = false) -> Tensor&;
    auto tanh(const Tensor& input) -> Tensor;
    auto static_round(const Tensor& input, float scale) -> Tensor;
    auto greedy_token(const Tensor& logits, TokenOutput output = TokenOutput::HOST) -> Tensor;
    auto embedding(const Tensor& indices, const Tensor& weight, const Tensor& scales, std::int32_t width,
                   std::int32_t bits, float multiplier, bool unsigned_packed = false) -> Tensor;
    auto embedding_sum(const Tensor& tokens, const Tensor& weight, const Tensor& scales, std::int32_t width,
                       std::int32_t bits, float multiplier) -> Tensor;
    auto diffusion_candidates(const Tensor& logits, std::size_t target, std::int32_t codebooks, std::int32_t vocabulary,
                              std::int32_t mask, float guidance) -> Tensor;
    auto diffusion_update(const Tensor& tokens, const Tensor& candidates, const Tensor& noise, std::size_t count,
                          std::int32_t mask, float penalty, float temperature) -> Tensor;
    auto conv1d(const Tensor& input, const Tensor& weight, const Tensor& scales, const Tensor& bias,
                std::int32_t kernel, std::int32_t stride, std::int32_t padding, std::int32_t dilation) -> Tensor;
    auto snake_conv1d(const Tensor& input, const Tensor& alpha, const Tensor& weight, const Tensor& scales,
                      const Tensor& bias, std::int32_t kernel, std::int32_t stride, std::int32_t padding,
                      std::int32_t dilation) -> Tensor;
    auto snake_quantized_linear(const Tensor& input, const Tensor& alpha, const Tensor& weight, const Tensor& scale,
                                const Tensor& bias, bool transpose_weight = false) -> Tensor;
    auto conv_transpose_fold(const Tensor& projected, const Tensor& bias, std::int32_t kernel, std::int32_t stride,
                             std::int32_t padding, std::int32_t output_padding, std::int32_t dilation) -> Tensor;
    auto snake(const Tensor& input, const Tensor& alpha) -> Tensor;

    /// Specialized operations with CPU and CUDA implementations.
    /// Positions are `start + row` unless a device I32 [tokens] `positions` tensor supplies them. A positive
    /// `log_floor` multiplies each query's logits by 1 + log_alpha * log(max(1, (position + 1) / log_floor)).
    struct RelativeAttention {
        std::int64_t heads, key_heads, start, window, extent;
        float log_alpha = 0, log_floor = 0;
        bool ring_mode = false;
    };
    /// RMS-normalizes each [tokens, width] source and mixes them per token with softmax weights from the
    /// [queries, width] pseudo-queries; returns FP32 [queries, tokens, width].
    auto residual_mix(std::span<const Tensor> sources, const Tensor& queries, float epsilon) -> Tensor;
    /// Causal depthwise convolution of `padded` [W - 1 + tokens, channels] with lag-0-first `weight` [W, channels]
    /// plus optional per-token `taps` [tokens, groups * W]; returns (convolution + current input) * scale.
    auto short_convolution(const Tensor& padded, const Tensor& weight, const Tensor& taps, float scale) -> Tensor;
    /// Ring-buffered causal depthwise convolution of `input` [tokens, channels] reading prior rows from `state`
    /// [ring, channels] starting before `slot` (I32 [1], index of row 0 mod `ring`), optionally adding `residual`
    /// and updating `state[slot]` in place when `tokens == 1 && update_state`.
    auto short_convolution(const Tensor& input, const Tensor& state, const Tensor& weight, const Tensor& taps,
                           const Tensor& slot, float scale, const Tensor& residual = {},
                           std::optional<tensor::DType> output_dtype = {}, bool update_state = false) -> Tensor;
    /// Multiplies [..., heads, width] rows by the optional [heads, width] scale and caps each row's L2 norm.
    auto l2_clip(const Tensor& input, const Tensor& scale, float threshold) -> Tensor;
    /// Causal attention of [tokens, heads * width] queries over [capacity, key_heads * width] caches, with relative
    /// logits [tokens, heads, extent], sink logits [tokens, heads] and sigmoid `gate`. CPU I8 caches require
    /// F32 scales [capacity, key_heads * groups_per_head]; scale groups must divide each head's width.
    auto relative_attention(const Tensor& query, const Tensor& key, const Tensor& value, const Tensor& relative,
                            const Tensor& sink, const Tensor& gate, RelativeAttention shape,
                            const Tensor& positions = {}, const Tensor& key_scales = {},
                            const Tensor& value_scales = {}) -> Tensor;
    /// CPU: quantizes [tokens, heads, width] into I8 cache [1, capacity, heads * width] at absolute `positions`,
    /// with F32 scales [capacity, heads * groups_per_head]. Existing rows outside `positions` are unchanged.
    auto quantize_cache_(Tensor& cache, Tensor& scales, const Tensor& input, const Tensor& positions) -> void;
    /// Unit-RMS-normalizes `destination` and `message` [tokens, width] and writes concatenated BF16 [tokens, 2 * width].
    auto blend_prologue(const Tensor& destination, const Tensor& message, float epsilon) -> Tensor;
    /// Computes mixed = destination * (1 + tanh(forget)) + message * tanh(gain) from [tokens, 2 * width]
    /// `gain_forget`, RMS-normalizes by `norm` [width], and writes BF16 [tokens, width].
    auto blend_epilogue(const Tensor& destination, const Tensor& message, const Tensor& gain_forget,
                        const Tensor& norm, float epsilon) -> Tensor;
    auto rms_norm(const Tensor& input, const Tensor& scale, float epsilon,
                  std::optional<tensor::DType> output_dtype = std::nullopt) -> Tensor;
    auto rms_rotary(const Tensor& input, const Tensor& scale, const Tensor& cosine, const Tensor& sine, float epsilon)
        -> Tensor;
    auto rms_rotary_slice(const Tensor& input, const Tensor& scale, const Tensor& cosine, const Tensor& sine,
                          float epsilon, std::int64_t heads, std::int64_t width, std::int64_t channel_offset,
                          std::int64_t sequence_offset, std::int64_t sequence_length) -> Tensor;
    auto rms_axial_rotary(const Tensor& input, const Tensor& scale, const Tensor& cosine, const Tensor& sine,
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
    /// In-place scatter for index tensors known to contain unique positions.
    auto scatter_unique_(Tensor& input, const Tensor& updates, const Tensor& indices) -> Tensor&;
    auto copy_slice_(Tensor& destination, const Tensor& source, std::int64_t axis, std::int64_t start) -> Tensor&;
    auto reshape(const Tensor& input, std::vector<std::int64_t> shape) -> Tensor;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace kidi::ops