#pragma once

#include <cstdint>
#include <span>

#include "kidi/core/error.h"

/// Host kernels behind the CPU paths of Context's custom operations. Inputs are read in place, math runs in F32,
/// and work is split across the CPU worker pool.
namespace kidi::ops::cpu {

/// Row-major matrix stored as F32, BF16 or INT8. INT8 rows carry one F32 scale per `scale_group` columns; attention
/// reads use groups that divide the head width.
struct Rows {
    enum class Storage { F32, BF16, I8 };
    const void* data = nullptr;
    Storage storage = Storage::F32;
    std::int64_t stride = 0;            // elements between consecutive rows
    const float* scales = nullptr;      // I8 only: [rows, stride / scale_group]
    std::int64_t scale_group = 0;
};

struct RelativeAttentionShape {
    std::int64_t tokens, heads, key_heads, width, extent, window;
    float log_alpha, log_floor;
};

/// Gated attention with sink logits and relative-position offsets; each query attends to cache rows
/// [position - window + 1, position] (all earlier rows when window is 0). Query-side operands are F32:
/// query/gate [tokens, heads * width], relative [tokens, heads, extent], sink [tokens, heads].
auto relative_attention(const float* query, const Rows& key, const Rows& value, const float* relative,
                        const float* sink, const float* gate, std::span<const std::int32_t> positions,
                        const RelativeAttentionShape& shape, float* output) -> Result<void>;

/// Softcapped SiLU gate over rows laid out as [gate | up] (2 * width columns):
/// out = gate_cap * tanh(g / gate_cap) * sigmoid(g) * up_cap * tanh(u / up_cap) * row_weights[row].
/// `row_weights` may be null.
auto silu_gate(const float* input, std::int64_t rows, std::int64_t width, float gate_cap, float up_cap,
               const float* row_weights, float* output) -> Result<void>;

/// Blend epilogue: mixed = destination * (1 + tanh(forget)) + message * tanh(gain), RMS-normalized and scaled by
/// `norm`. `gain_forget` rows are laid out as [gain | forget].
auto blend_epilogue(const float* destination, const float* message, const float* gain_forget, const float* norm,
                    std::int64_t rows, std::int64_t width, float epsilon, float* output) -> Result<void>;

struct ShortConvolutionShape {
    std::int64_t rows, channels, width, groups;
    float scale;
};

/// Depthwise causal convolution: out[t] = scale * (current[t] + sum_lag (weight[lag] + taps[t, group, lag]) *
/// sources[t * width + lag]) + residual[t]. `sources` holds one channel row per (token, lag), so callers can
/// resolve padding or ring history; `taps` ([rows, groups * width]) and `residual` ([rows, channels]) may be null.
auto short_convolution(const float* const* sources, const float* current, const float* weight, const float* taps,
                       const float* residual, const ShortConvolutionShape& shape, float* output) -> Result<void>;

/// One routed expert in row-scaled INT8: gate_up is [2 * intermediate, hidden] laid out as [gate | up] rows and
/// down is [hidden, intermediate], each with one F32 scale per row.
struct ExpertRows {
    const std::int8_t* gate_up;
    const float* gate_up_scales;
    const std::int8_t* down;
    const float* down_scales;
    float route;
};

/// Single-token MoE in W8A8: output[hidden] = sum_e route_e * down_e(silu_gate(gate_up_e(input))). The softcapped
/// SiLU matches `silu_gate`; caps of 0 disable capping. Two pool dispatches cover all experts.
auto expert_decode(const float* input, std::span<const ExpertRows> experts, std::int64_t hidden,
                   std::int64_t intermediate, float gate_cap, float up_cap, float* output) -> Result<void>;

struct RoutedExpert {
    ExpertRows weight;
    std::span<const std::int32_t> tokens;
    std::span<const float> routes;
};

/// Batched W8A8 MoE: input/output [tokens, hidden]. Groups work by expert to reuse weight slices, quantizes
/// inputs once, and reduces routed outputs without per-expert tensor gathers/scatters.
auto expert_prefill(const float* input, std::int64_t tokens, std::span<const RoutedExpert> experts,
                    std::int64_t hidden, std::int64_t intermediate, float gate_cap, float up_cap,
                    float* output) -> Result<void>;

/// W8A8 projection: output[t, n] = dot(input[t, :], weight[n, :]) * scales[n], with `input` [tokens, k]
/// quantized per row and BF16 `output` [tokens, n]. Work is split so each pool item reads a similar weight slice.
auto int8_linear(const float* input, std::int64_t tokens, std::int64_t k, const std::int8_t* weight,
                 const float* scales, std::int64_t n, std::uint16_t* output) -> Result<void>;

// Width-normalized softmax mix of `sources` (each [tokens, width]) for each query row; output is
// [query_count, tokens, width]. At most eight sources.
auto residual_mix(std::span<const float* const> sources, const float* queries, std::int64_t query_count,
                  std::int64_t tokens, std::int64_t width, float epsilon, float* output) -> Result<void>;

// Optionally scales rows by `scale` ([scale_rows, width], cycling over rows), then clips each row to L2 norm
// `threshold`. `output` may alias `input`.
auto l2_clip(const float* input, const float* scale, std::int64_t scale_rows, std::int64_t rows, std::int64_t width,
             float threshold, float* output) -> Result<void>;

/// Element-wise BF16 <-> F32 conversion; F32 -> BF16 rounds to nearest even and keeps NaN.
auto bf16_to_f32(const std::uint16_t* input, std::int64_t count, float* output) -> void;
auto f32_to_bf16(const float* input, std::int64_t count, std::uint16_t* output) -> void;

/// Quantizes `rows` F32 rows of `width` into symmetric INT8 with one scale per `group` columns.
auto quantize_rows(const float* input, std::int64_t rows, std::int64_t width, std::int64_t group,
                   std::int8_t* values, float* scales) -> void;

} // namespace kidi::ops::cpu
