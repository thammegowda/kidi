#include "kidi/ops/cpu_kernels.h"

#include "kidi/runtime/ynn/graph.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <limits>
#include <vector>

#if defined(__x86_64__) && defined(__GNUC__)
#include <immintrin.h>
#define KIDI_CPU_VNNI 1
#endif

// Hot loops are compiled for AVX-512 and AVX2 as well; the loader picks the best clone for the running CPU.
#if defined(__x86_64__) && defined(__GNUC__) && !defined(__clang__) && defined(__ELF__)
#define KIDI_CPU_CLONES __attribute__((target_clones("avx512f", "avx2,fma", "default")))
#else
#define KIDI_CPU_CLONES
#endif
// Helpers called from cloned kernels must inline, or they are emitted once for the baseline target (no AVX).
#define KIDI_CPU_INLINE [[gnu::always_inline]] inline

namespace kidi::ops::cpu {
namespace {

KIDI_CPU_INLINE auto bf16_to_float(std::uint16_t value) -> float {
    return std::bit_cast<float>(std::uint32_t{value} << 16);
}

// Branch-free exp (relative error ~2e-7) so element-wise loops vectorize; std::exp/std::tanh are scalar libm calls.
KIDI_CPU_INLINE auto fast_exp(float x) -> float {
    x = std::min(88.F, std::max(-87.F, x));
    const auto t = x * 1.44269504088896341F;
    const auto whole = (t + 12582912.F) - 12582912.F; // round to nearest
    const auto f = t - whole;
    auto p = 1.535336188319500e-4F;
    p = p * f + 1.339887440266574e-3F;
    p = p * f + 9.618437357674640e-3F;
    p = p * f + 5.550332471162809e-2F;
    p = p * f + 2.402264791363012e-1F;
    p = p * f + 6.931472028550421e-1F;
    p = p * f + 1.F;
    return std::bit_cast<float>(std::bit_cast<std::int32_t>(p) + (static_cast<std::int32_t>(whole) << 23));
}
KIDI_CPU_INLINE auto fast_tanh(float x) -> float {
    x = std::min(9.F, std::max(-9.F, x));
    return 1.F - 2.F / (fast_exp(2.F * x) + 1.F);
}
KIDI_CPU_INLINE auto fast_sigmoid(float x) -> float { return 1.F / (1.F + fast_exp(-x)); }

// Sums term(0..n) with 16 independent partial sums; a single scalar accumulator would block vectorization.
template <typename Term>
KIDI_CPU_INLINE auto sum_lanes(std::int64_t n, Term&& term) -> float {
    constexpr std::int64_t LANES = 16;
    std::array<float, LANES> lanes{};
    std::int64_t i = 0;
    for (; i + LANES <= n; i += LANES)
        for (std::int64_t j = 0; j < LANES; ++j) lanes[static_cast<std::size_t>(j)] += term(i + j);
    float sum = 0;
    for (; i < n; ++i) sum += term(i);
    for (const auto lane : lanes) sum += lane;
    return sum;
}

// Splits rows over the worker pool once there is enough element-wise work to pay for the dispatch.
template <typename Body>
auto for_rows(std::int64_t rows, std::int64_t width, Body&& body) -> Result<void> {
    constexpr std::int64_t SERIAL_ELEMENTS = 32 * 1024;
    if (rows <= 1 || rows * width <= SERIAL_ELEMENTS) {
        body(std::int64_t{0}, rows);
        return {};
    }
    const auto grain = static_cast<std::size_t>(std::max<std::int64_t>(1, SERIAL_ELEMENTS / 4 / width));
    return runtime::ynn::parallel_ranges(static_cast<std::size_t>(rows), grain,
                                         [&](std::size_t begin, std::size_t end) {
                                             body(static_cast<std::int64_t>(begin), static_cast<std::int64_t>(end));
                                         });
}

// Copies one cache row into F32 (I8 rows folded with their scale) so it can be shared by every query head of a group.
template <Rows::Storage S>
KIDI_CPU_INLINE auto load_row(const Rows& rows, std::int64_t row, std::int64_t column, std::int64_t width,
                              float* target) -> const float* {
    const auto base = row * rows.stride + column;
    if constexpr (S == Rows::Storage::F32) {
        return static_cast<const float*>(rows.data) + base;
    } else if constexpr (S == Rows::Storage::BF16) {
        const auto* source = static_cast<const std::uint16_t*>(rows.data) + base;
        for (std::int64_t i = 0; i < width; ++i) target[i] = bf16_to_float(source[i]);
        return target;
    } else {
        const auto* source = static_cast<const std::int8_t*>(rows.data) + base;
        for (std::int64_t block = 0; block < width; block += rows.scale_group) {
            const auto scale = rows.scales[(base + block) / rows.scale_group];
            for (std::int64_t i = block; i < std::min(block + rows.scale_group, width); ++i)
                target[i] = scale * static_cast<float>(source[i]);
        }
        return target;
    }
}

struct AttentionArgs {
    const float* query;
    const Rows* key;
    const Rows* value;
    const float *relative, *sink, *gate;
    std::span<const std::int32_t> positions;
    const RelativeAttentionShape* shape;
    std::int64_t splits;
    float* output;
    float* partials; // splits > 1: per (row, split) [maximum, denominator, unnormalized output...]
    std::int64_t token_tile = 1;
};

constexpr float NO_LOGIT = -1e30F;

// One item is (token tile, key head, key segment): query tokens/heads share each K/V row load. With a single
// segment the sink and gate are applied in place; otherwise softmax statistics are left in partials for merge_split.
template <Rows::Storage K, Rows::Storage V>
KIDI_CPU_CLONES auto attend_range(std::size_t begin, std::size_t end, const AttentionArgs& a) -> void {
    const auto& shape = *a.shape;
    const auto width = shape.width;
    const auto group = shape.heads / shape.key_heads;
    const auto scale = 1.F / std::sqrt(static_cast<float>(width));
    struct QueryRow {
        std::int64_t row, position, first;
        float tau;
    };
    thread_local std::vector<QueryRow> queries;
    thread_local std::vector<float> logits, row_buffer, maxima, sums;
    row_buffer.resize(static_cast<std::size_t>(width));
    for (auto item = static_cast<std::int64_t>(begin); item < static_cast<std::int64_t>(end); ++item) {
        const auto split = item % a.splits;
        const auto key_head = item / a.splits % shape.key_heads;
        const auto token_begin = item / a.splits / shape.key_heads * a.token_tile;
        queries.clear();
        auto first = std::numeric_limits<std::int64_t>::max(), position = std::int64_t{0};
        for (auto token = token_begin; token < std::min(token_begin + a.token_tile, shape.tokens); ++token) {
            const auto pos = static_cast<std::int64_t>(a.positions[static_cast<std::size_t>(token)]);
            const auto start = shape.window ? std::max<std::int64_t>(0, pos - shape.window + 1) : 0;
            first = std::min(first, start);
            position = std::max(position, pos);
            const auto tau = shape.log_floor > 0
                                 ? 1.F + shape.log_alpha *
                                             std::log(std::max(1.F, static_cast<float>(pos + 1) / shape.log_floor))
                                 : 1.F;
            for (std::int64_t head = 0; head < group; ++head)
                queries.push_back({token * shape.heads + key_head * group + head, pos, start, tau});
        }
        const auto query_count = static_cast<std::int64_t>(queries.size());
        maxima.resize(queries.size());
        sums.resize(queries.size());
        const auto segment = (position - first + a.splits) / a.splits;
        const auto lo = first + split * segment;
        const auto hi = std::min(position + 1, lo + segment);
        const auto count = std::max<std::int64_t>(0, hi - lo);
        const auto column = key_head * width;
        logits.assign(static_cast<std::size_t>(query_count * count), NO_LOGIT);
        for (std::int64_t h = 0; h < query_count; ++h)
            maxima[static_cast<std::size_t>(h)] = a.splits == 1 ? a.sink[queries[h].row] : NO_LOGIT;
        for (auto index = lo; index < hi; ++index) {
            const auto* key = load_row<K>(*a.key, index, column, width, row_buffer.data());
            for (std::int64_t h = 0; h < query_count; ++h) {
                const auto& work = queries[h];
                if (index < work.first || index > work.position) continue;
                const auto distance = work.position - index;
                const auto* query = a.query + work.row * width;
                auto logit = sum_lanes(width, [&](std::int64_t i) { return query[i] * key[i]; }) * scale;
                if (distance < shape.extent) logit += a.relative[work.row * shape.extent + distance];
                logit *= work.tau;
                logits[static_cast<std::size_t>(h * count + index - lo)] = logit;
                auto& maximum = maxima[static_cast<std::size_t>(h)];
                maximum = std::max(maximum, logit);
            }
        }
        for (std::int64_t h = 0; h < query_count; ++h) {
            const auto maximum = maxima[static_cast<std::size_t>(h)];
            float sum = a.splits == 1 ? fast_exp(a.sink[queries[h].row] - maximum) : 0.F;
            auto* weights = logits.data() + h * count;
            for (std::int64_t i = 0; i < count; ++i) {
                weights[i] = weights[i] == NO_LOGIT ? 0.F : fast_exp(weights[i] - maximum);
                sum += weights[i];
            }
            sums[static_cast<std::size_t>(h)] = sum;
        }
        const auto stride = width + 2;
        auto target = [&](std::int64_t h) {
            return a.splits == 1 ? a.output + queries[h].row * width
                                 : a.partials + (queries[h].row * a.splits + split) * stride + 2;
        };
        for (std::int64_t h = 0; h < query_count; ++h) std::fill_n(target(h), width, 0.F);
        for (auto index = lo; index < hi; ++index) {
            const auto* value = load_row<V>(*a.value, index, column, width, row_buffer.data());
            for (std::int64_t h = 0; h < query_count; ++h) {
                const auto weight = logits[static_cast<std::size_t>(h * count + index - lo)];
                auto* out = target(h);
                for (std::int64_t i = 0; i < width; ++i) out[i] += weight * value[i];
            }
        }
        for (std::int64_t h = 0; h < query_count; ++h) {
            auto* out = target(h);
            if (a.splits == 1) {
                const auto inverse = 1.F / sums[static_cast<std::size_t>(h)];
                const auto* gate = a.gate + queries[h].row * width;
                for (std::int64_t i = 0; i < width; ++i) out[i] *= inverse * fast_sigmoid(gate[i]);
            } else {
                out[-2] = maxima[static_cast<std::size_t>(h)];
                out[-1] = sums[static_cast<std::size_t>(h)];
            }
        }
    }
}

// Combines per-segment softmax statistics (flash-decoding style), then applies the sink and output gate.
KIDI_CPU_CLONES auto merge_split(const AttentionArgs& a) -> void {
    const auto& shape = *a.shape;
    const auto width = shape.width, stride = width + 2;
    for (std::int64_t row = 0; row < shape.tokens * shape.heads; ++row) {
        const auto* parts = a.partials + row * a.splits * stride;
        auto maximum = a.sink[row];
        for (std::int64_t s = 0; s < a.splits; ++s) maximum = std::max(maximum, parts[s * stride]);
        auto denominator = fast_exp(a.sink[row] - maximum);
        auto* out = a.output + row * width;
        std::fill_n(out, width, 0.F);
        for (std::int64_t s = 0; s < a.splits; ++s) {
            const auto* part = parts + s * stride;
            if (part[1] == 0.F) continue;
            const auto factor = fast_exp(part[0] - maximum);
            denominator += factor * part[1];
            for (std::int64_t i = 0; i < width; ++i) out[i] += factor * part[2 + i];
        }
        const auto inverse = 1.F / denominator;
        const auto* gate = a.gate + row * width;
        for (std::int64_t i = 0; i < width; ++i) out[i] *= inverse * fast_sigmoid(gate[i]);
    }
}

using AttendRange = void (*)(std::size_t, std::size_t, const AttentionArgs&);

template <Rows::Storage K>
auto pick_value(Rows::Storage value) -> AttendRange {
    switch (value) {
    case Rows::Storage::F32: return &attend_range<K, Rows::Storage::F32>;
    case Rows::Storage::BF16: return &attend_range<K, Rows::Storage::BF16>;
    case Rows::Storage::I8: return &attend_range<K, Rows::Storage::I8>;
    }
    return nullptr;
}

auto pick(Rows::Storage key, Rows::Storage value) -> AttendRange {
    switch (key) {
    case Rows::Storage::F32: return pick_value<Rows::Storage::F32>(value);
    case Rows::Storage::BF16: return pick_value<Rows::Storage::BF16>(value);
    case Rows::Storage::I8: return pick_value<Rows::Storage::I8>(value);
    }
    return nullptr;
}

// Splits long key ranges so decode (few tokens x key heads) still yields several items per worker.
auto attention_splits(const RelativeAttentionShape& shape, std::span<const std::int32_t> positions) -> std::int64_t {
    constexpr std::int64_t MIN_SEGMENT = 256, ITEMS_PER_THREAD = 4;
    auto longest = std::int64_t{0};
    for (const auto position : positions) longest = std::max<std::int64_t>(longest, position + 1);
    if (shape.window) longest = std::min(longest, shape.window);
    const auto items = shape.tokens * shape.key_heads;
    const auto wanted = static_cast<std::int64_t>(runtime::ynn::thread_count()) * ITEMS_PER_THREAD;
    if (items >= wanted) return 1;
    return std::clamp<std::int64_t>(std::min(longest / MIN_SEGMENT, (wanted + items - 1) / items), 1, 64);
}

} // namespace

auto relative_attention(const float* query, const Rows& key, const Rows& value, const float* relative,
                        const float* sink, const float* gate, std::span<const std::int32_t> positions,
                        const RelativeAttentionShape& shape, float* output) -> Result<void> {
    const auto run = pick(key.storage, value.storage);
    AttentionArgs args{query, &key,  &value, relative, sink, gate, positions, &shape,
                       attention_splits(shape, positions), output, nullptr};
    if (args.splits == 1 && shape.tokens >= 4) args.token_tile = 4;
    thread_local std::vector<float> partials;
    if (args.splits > 1) {
        partials.resize(static_cast<std::size_t>(shape.tokens * shape.heads * args.splits * (shape.width + 2)));
        args.partials = partials.data();
    }
    const auto tiles = (shape.tokens + args.token_tile - 1) / args.token_tile;
    const auto items = static_cast<std::size_t>(tiles * shape.key_heads * args.splits);
    auto status = runtime::ynn::parallel_ranges(items, 1, [&](std::size_t begin, std::size_t end) {
        run(begin, end, args);
    });
    if (!status || args.splits == 1) return status;
    merge_split(args);
    return {};
}

namespace {

KIDI_CPU_CLONES auto silu_gate_rows(const float* input, std::int64_t begin, std::int64_t end, std::int64_t width,
                                    float gate_cap, float up_cap, const float* row_weights, float* output) -> void {
    const auto inverse_gate = 1.F / gate_cap, inverse_up = 1.F / up_cap;
    for (auto row = begin; row < end; ++row) {
        const auto* gate = input + row * 2 * width;
        const auto* up = gate + width;
        const auto weight = (row_weights ? row_weights[row] : 1.F) * gate_cap * up_cap;
        auto* target = output + row * width;
        for (std::int64_t column = 0; column < width; ++column)
            target[column] = fast_tanh(gate[column] * inverse_gate) * fast_sigmoid(gate[column]) *
                             fast_tanh(up[column] * inverse_up) * weight;
    }
}

KIDI_CPU_CLONES auto blend_rows(const float* destination, const float* message, const float* gain_forget,
                                const float* norm, std::int64_t begin, std::int64_t end, std::int64_t width,
                                float epsilon, float* output) -> void {
    for (auto row = begin; row < end; ++row) {
        const auto* d = destination + row * width;
        const auto* m = message + row * width;
        const auto* gain = gain_forget + row * 2 * width;
        const auto* forget = gain + width;
        auto* target = output + row * width;
        for (std::int64_t column = 0; column < width; ++column)
            target[column] = d[column] * (1.F + fast_tanh(forget[column])) + m[column] * fast_tanh(gain[column]);
        const auto squares = sum_lanes(width, [&](std::int64_t i) { return target[i] * target[i]; });
        const auto inverse = 1.F / std::sqrt(squares / static_cast<float>(width) + epsilon);
        for (std::int64_t column = 0; column < width; ++column) target[column] *= inverse * norm[column];
    }
}

KIDI_CPU_CLONES auto convolve_rows(const float* const* sources, const float* current, const float* weight,
                                   const float* taps, const float* residual, const ShortConvolutionShape& shape,
                                   std::int64_t begin, std::int64_t end, float* output) -> void {
    const auto channels = shape.channels, width = shape.width;
    const auto span = channels / shape.groups;
    for (auto row = begin; row < end; ++row) {
        auto* target = output + row * channels;
        std::copy_n(current + row * channels, channels, target);
        for (std::int64_t lag = 0; lag < width; ++lag) {
            const auto* source = sources[row * width + lag];
            const auto* tap = weight + lag * channels;
            for (std::int64_t group = 0; group < shape.groups; ++group) {
                const auto dynamic = taps ? taps[(row * shape.groups + group) * width + lag] : 0.F;
                for (auto channel = group * span; channel < (group + 1) * span; ++channel)
                    target[channel] += (tap[channel] + dynamic) * source[channel];
            }
        }
        const auto* extra = residual ? residual + row * channels : nullptr;
        for (std::int64_t channel = 0; channel < channels; ++channel)
            target[channel] = target[channel] * shape.scale + (extra ? extra[channel] : 0.F);
    }
}

} // namespace

auto silu_gate(const float* input, std::int64_t rows, std::int64_t width, float gate_cap, float up_cap,
               const float* row_weights, float* output) -> Result<void> {
    return for_rows(rows, width, [&](std::int64_t begin, std::int64_t end) {
        silu_gate_rows(input, begin, end, width, gate_cap, up_cap, row_weights, output);
    });
}

auto blend_epilogue(const float* destination, const float* message, const float* gain_forget, const float* norm,
                    std::int64_t rows, std::int64_t width, float epsilon, float* output) -> Result<void> {
    return for_rows(rows, width, [&](std::int64_t begin, std::int64_t end) {
        blend_rows(destination, message, gain_forget, norm, begin, end, width, epsilon, output);
    });
}

auto short_convolution(const float* const* sources, const float* current, const float* weight, const float* taps,
                       const float* residual, const ShortConvolutionShape& shape, float* output) -> Result<void> {
    return for_rows(shape.rows, shape.channels * shape.width, [&](std::int64_t begin, std::int64_t end) {
        convolve_rows(sources, current, weight, taps, residual, shape, begin, end, output);
    });
}

namespace {

// Symmetric per-vector INT8 quantization; returns the scale.
KIDI_CPU_CLONES auto quantize_vector(const float* input, std::int64_t count, std::int8_t* output) -> float {
    float maximum = 0;
    for (std::int64_t index = 0; index < count; ++index) maximum = std::max(maximum, std::abs(input[index]));
    if (!(maximum > 0)) {
        std::fill_n(output, count, std::int8_t{0});
        return 0.F;
    }
    const auto inverse = 127.F / maximum;
    for (std::int64_t index = 0; index < count; ++index)
        output[index] = static_cast<std::int8_t>(std::nearbyint(input[index] * inverse));
    return maximum / 127.F;
}

// Integer dot products of four weight rows (stride `stride`) against one INT8 vector.
KIDI_CPU_CLONES auto dot4_portable(const std::int8_t* x, const std::int8_t* w, std::int64_t stride, std::int64_t n,
                                   std::int32_t* out) -> void {
    std::int32_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    for (std::int64_t i = 0; i < n; ++i) {
        const std::int32_t v = x[i];
        s0 += v * w[i];
        s1 += v * w[stride + i];
        s2 += v * w[2 * stride + i];
        s3 += v * w[3 * stride + i];
    }
    out[0] = s0, out[1] = s1, out[2] = s2, out[3] = s3;
}

#ifdef KIDI_CPU_VNNI
// VPDPBUSD multiplies unsigned by signed bytes, so feed |x| and move x's sign onto the weights (weights never hold
// -128, so negation is exact).
__attribute__((target("avx512f,avx512bw,avx512vnni"))) auto dot4_vnni(const std::int8_t* x, const std::int8_t* w,
                                                                      std::int64_t stride, std::int64_t n,
                                                                      std::int32_t* out) -> void {
    const auto zero = _mm512_setzero_si512();
    __m512i a0 = zero, a1 = zero, a2 = zero, a3 = zero;
    for (std::int64_t i = 0; i < n; i += 64) {
        const auto v = _mm512_loadu_si512(x + i);
        const auto negative = _mm512_movepi8_mask(v);
        const auto magnitude = _mm512_abs_epi8(v);
        const auto w0 = _mm512_loadu_si512(w + i);
        const auto w1 = _mm512_loadu_si512(w + stride + i);
        const auto w2 = _mm512_loadu_si512(w + 2 * stride + i);
        const auto w3 = _mm512_loadu_si512(w + 3 * stride + i);
        a0 = _mm512_dpbusd_epi32(a0, magnitude, _mm512_mask_sub_epi8(w0, negative, zero, w0));
        a1 = _mm512_dpbusd_epi32(a1, magnitude, _mm512_mask_sub_epi8(w1, negative, zero, w1));
        a2 = _mm512_dpbusd_epi32(a2, magnitude, _mm512_mask_sub_epi8(w2, negative, zero, w2));
        a3 = _mm512_dpbusd_epi32(a3, magnitude, _mm512_mask_sub_epi8(w3, negative, zero, w3));
    }
    out[0] = _mm512_reduce_add_epi32(a0), out[1] = _mm512_reduce_add_epi32(a1);
    out[2] = _mm512_reduce_add_epi32(a2), out[3] = _mm512_reduce_add_epi32(a3);
}
#endif

using Dot4 = void (*)(const std::int8_t*, const std::int8_t*, std::int64_t, std::int64_t, std::int32_t*);

auto pick_dot4(std::int64_t n) -> Dot4 {
#ifdef KIDI_CPU_VNNI
    static const bool vnni = __builtin_cpu_supports("avx512vnni") && __builtin_cpu_supports("avx512bw");
    if (vnni && n % 64 == 0) return &dot4_vnni;
#endif
    (void)n;
    return &dot4_portable;
}

// Rows [begin, end) of an INT8 matrix times an INT8 vector, scaled per row by `scales[row] * scale`; writes
// out[row - begin].
auto matvec_rows(Dot4 dot4, const std::int8_t* x, float scale, const std::int8_t* w, const float* scales,
                 std::int64_t n, std::int64_t begin, std::int64_t end, float* out) -> void {
    std::array<std::int32_t, 4> sums{};
    auto row = begin;
    for (; row + 4 <= end; row += 4) {
        dot4(x, w + row * n, n, n, sums.data());
        for (int k = 0; k < 4; ++k) out[row - begin + k] = static_cast<float>(sums[k]) * scales[row + k] * scale;
    }
    for (; row < end; ++row) {
        std::int32_t sum = 0;
        for (std::int64_t i = 0; i < n; ++i) sum += std::int32_t{x[i]} * w[row * n + i];
        out[row - begin] = static_cast<float>(sum) * scales[row] * scale;
    }
}

// In place: gate[i] becomes the (optionally soft-capped) SiLU(gate) * up.
KIDI_CPU_CLONES auto activate(float* gate, const float* up, std::int64_t width, float gate_cap, float up_cap) -> void {
    if (gate_cap > 0 && up_cap > 0) {
        const auto inverse_gate = 1.F / gate_cap, inverse_up = 1.F / up_cap, caps = gate_cap * up_cap;
        for (std::int64_t i = 0; i < width; ++i)
            gate[i] = fast_tanh(gate[i] * inverse_gate) * fast_sigmoid(gate[i]) * fast_tanh(up[i] * inverse_up) * caps;
        return;
    }
    for (std::int64_t i = 0; i < width; ++i) gate[i] = gate[i] * fast_sigmoid(gate[i]) * up[i];
}

} // namespace

auto expert_decode(const float* input, std::span<const ExpertRows> experts, std::int64_t hidden,
                   std::int64_t intermediate, float gate_cap, float up_cap, float* output) -> Result<void> {
    constexpr std::int64_t CHUNK = 32;
    constexpr std::int64_t DOWN_CHUNK = 16;
    const auto count = static_cast<std::int64_t>(experts.size());
    const auto gate_rows = 2 * intermediate;
    thread_local std::vector<std::int8_t> quantized;
    thread_local std::vector<float> projected, scales;
    quantized.resize(static_cast<std::size_t>(hidden + count * intermediate));
    projected.resize(static_cast<std::size_t>(count * gate_rows));
    scales.resize(static_cast<std::size_t>(count));
    // Worker threads have their own (empty) thread_local instances, so lambdas must use these pointers.
    auto* x = quantized.data();
    auto* hidden_codes = x + hidden;
    auto* gate_up_out = projected.data();
    auto* expert_scales = scales.data();
    const auto x_scale = quantize_vector(input, hidden, x);

    const auto up_dot = pick_dot4(hidden), down_dot = pick_dot4(intermediate);
    // Each item projects matching gate and up rows, so the activation runs inside the parallel phase.
    const auto gate_chunks = (intermediate + CHUNK - 1) / CHUNK;
    if (auto done = runtime::ynn::parallel_ranges(
            static_cast<std::size_t>(count * gate_chunks), 1,
            [&](std::size_t begin, std::size_t end) {
                for (auto item = static_cast<std::int64_t>(begin); item < static_cast<std::int64_t>(end); ++item) {
                    const auto expert = item / gate_chunks, first = item % gate_chunks * CHUNK;
                    const auto last = std::min(first + CHUNK, intermediate);
                    const auto& rows = experts[static_cast<std::size_t>(expert)];
                    auto* gate = gate_up_out + expert * gate_rows + first;
                    auto* up = gate + intermediate;
                    matvec_rows(up_dot, x, x_scale, rows.gate_up, rows.gate_up_scales, hidden, first, last, gate);
                    matvec_rows(up_dot, x, x_scale, rows.gate_up, rows.gate_up_scales, hidden, intermediate + first,
                                intermediate + last, up);
                    activate(gate, up, last - first, gate_cap, up_cap);
                }
            });
        !done)
        return done;
    for (std::int64_t expert = 0; expert < count; ++expert)
        expert_scales[expert] =
            quantize_vector(gate_up_out + expert * gate_rows, intermediate, hidden_codes + expert * intermediate) *
            experts[static_cast<std::size_t>(expert)].route;

    const auto down_chunks = (hidden + DOWN_CHUNK - 1) / DOWN_CHUNK;
    return runtime::ynn::parallel_ranges(static_cast<std::size_t>(down_chunks), 1, [&](std::size_t begin,
                                                                                        std::size_t end) {
        std::array<float, DOWN_CHUNK> part{};
        for (auto chunk = static_cast<std::int64_t>(begin); chunk < static_cast<std::int64_t>(end); ++chunk) {
            const auto first = chunk * DOWN_CHUNK, last = std::min(first + DOWN_CHUNK, hidden);
            std::fill(output + first, output + last, 0.F);
            for (std::int64_t expert = 0; expert < count; ++expert) {
                const auto& rows = experts[static_cast<std::size_t>(expert)];
                matvec_rows(down_dot, hidden_codes + expert * intermediate, expert_scales[expert],
                            rows.down, rows.down_scales, intermediate, first, last, part.data());
                for (auto row = first; row < last; ++row) output[row] += part[static_cast<std::size_t>(row - first)];
            }
        }
    });
}

auto expert_prefill(const float* input, std::int64_t tokens, std::span<const RoutedExpert> experts,
                    std::int64_t hidden, std::int64_t intermediate, float gate_cap, float up_cap,
                    float* output) -> Result<void> {
    constexpr std::int64_t CHUNK = 32;
    struct Job {
        std::size_t expert;
        std::int64_t first;
    };
    std::vector<std::size_t> offsets{0};
    std::vector<Job> up_jobs, down_jobs;
    std::vector<std::vector<std::size_t>> contributions(static_cast<std::size_t>(tokens));
    for (std::size_t e = 0; e < experts.size(); ++e) {
        const auto& expert = experts[e];
        if (expert.tokens.size() != expert.routes.size())
            return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "MoE tokens/routes size mismatch"});
        for (std::size_t r = 0; r < expert.tokens.size(); ++r) {
            const auto t = expert.tokens[r];
            if (t < 0 || t >= tokens)
                return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "MoE token outside input"});
            contributions[static_cast<std::size_t>(t)].push_back(offsets.back() + r);
        }
        offsets.push_back(offsets.back() + expert.tokens.size());
        for (std::int64_t first = 0; first < intermediate; first += CHUNK) up_jobs.push_back({e, first});
        for (std::int64_t first = 0; first < hidden; first += CHUNK) down_jobs.push_back({e, first});
    }
    const auto count = offsets.back();
    thread_local std::vector<std::int8_t> input_codes, hidden_codes;
    thread_local std::vector<float> input_scales, hidden_scales, gate_up, projected;
    input_codes.resize(static_cast<std::size_t>(tokens * hidden));
    input_scales.resize(static_cast<std::size_t>(tokens));
    hidden_codes.resize(count * intermediate);
    hidden_scales.resize(count);
    gate_up.resize(count * 2 * intermediate);
    projected.resize(count * hidden);
    auto* x = input_codes.data();
    auto* xs = input_scales.data();
    auto* h = hidden_codes.data();
    auto* hs = hidden_scales.data();
    auto* gu = gate_up.data();
    auto* parts = projected.data();
    for (std::int64_t t = 0; t < tokens; ++t) xs[t] = quantize_vector(input + t * hidden, hidden, x + t * hidden);
    const auto up_dot = pick_dot4(hidden), down_dot = pick_dot4(intermediate);
    if (auto status = runtime::ynn::parallel_ranges(up_jobs.size(), 1, [&](std::size_t begin, std::size_t end) {
            for (auto job = begin; job < end; ++job) {
                const auto [e, first] = up_jobs[job];
                const auto last = std::min(first + CHUNK, intermediate);
                const auto& expert = experts[e];
                for (std::size_t r = 0; r < expert.tokens.size(); ++r) {
                    const auto t = expert.tokens[r];
                    auto* gate = gu + (offsets[e] + r) * 2 * intermediate + first;
                    auto* up = gate + intermediate;
                    matvec_rows(up_dot, x + t * hidden, xs[t], expert.weight.gate_up,
                                expert.weight.gate_up_scales, hidden, first, last, gate);
                    matvec_rows(up_dot, x + t * hidden, xs[t], expert.weight.gate_up,
                                expert.weight.gate_up_scales, hidden, intermediate + first, intermediate + last, up);
                    activate(gate, up, last - first, gate_cap, up_cap);
                }
            }
        });
        !status)
        return status;
    if (auto status = runtime::ynn::parallel_ranges(experts.size(), 1, [&](std::size_t begin, std::size_t end) {
            for (auto e = begin; e < end; ++e)
                for (std::size_t r = 0; r < experts[e].tokens.size(); ++r) {
                    const auto row = offsets[e] + r;
                    hs[row] = quantize_vector(gu + row * 2 * intermediate, intermediate, h + row * intermediate) *
                              experts[e].routes[r];
                }
        });
        !status)
        return status;
    if (auto status = runtime::ynn::parallel_ranges(down_jobs.size(), 1, [&](std::size_t begin, std::size_t end) {
            for (auto job = begin; job < end; ++job) {
                const auto [e, first] = down_jobs[job];
                const auto last = std::min(first + CHUNK, hidden);
                const auto& expert = experts[e];
                for (auto row = offsets[e]; row < offsets[e + 1]; ++row)
                    matvec_rows(down_dot, h + row * intermediate, hs[row], expert.weight.down,
                                expert.weight.down_scales, intermediate, first, last, parts + row * hidden + first);
            }
        });
        !status)
        return status;
    return for_rows(tokens, hidden, [&](std::int64_t begin, std::int64_t end) {
        for (auto t = begin; t < end; ++t) {
            auto* out = output + t * hidden;
            std::fill_n(out, hidden, 0.F);
            for (const auto row : contributions[static_cast<std::size_t>(t)])
                for (std::int64_t i = 0; i < hidden; ++i) out[i] += parts[row * hidden + i];
        }
    });
}

auto int8_linear(const float* input, std::int64_t tokens, std::int64_t k, const std::int8_t* weight,
                 const float* scales, std::int64_t n, std::uint16_t* output) -> Result<void> {
    // ~16KB of weights per item spreads small matrices over the pool; tiny ones run inline.
    constexpr std::int64_t ITEM_BYTES = 16 << 10;
    constexpr std::int64_t MAX_CHUNK = 256;
    thread_local std::vector<std::int8_t> quantized;
    thread_local std::vector<float> row_scales;
    quantized.resize(static_cast<std::size_t>(tokens * k));
    row_scales.resize(static_cast<std::size_t>(tokens));
    auto* codes = quantized.data();
    auto* code_scales = row_scales.data();
    for (std::int64_t t = 0; t < tokens; ++t) code_scales[t] = quantize_vector(input + t * k, k, codes + t * k);
    const auto dot = pick_dot4(k);
    const auto chunk = std::clamp<std::int64_t>((ITEM_BYTES / std::max<std::int64_t>(k, 1) + 3) / 4 * 4, 4, MAX_CHUNK);
    const auto chunks = (n + chunk - 1) / chunk;
    constexpr std::int64_t TOKEN_TILE = 8;
    const auto token_tiles = (tokens + TOKEN_TILE - 1) / TOKEN_TILE;
    const auto items = chunks * token_tiles;
    const auto body = [&](std::size_t begin, std::size_t end) {
        std::array<float, MAX_CHUNK> part{};
        for (auto item = static_cast<std::int64_t>(begin); item < static_cast<std::int64_t>(end); ++item) {
            const auto first = item % chunks * chunk, last = std::min(first + chunk, n);
            const auto token_begin = item / chunks * TOKEN_TILE;
            for (auto t = token_begin; t < std::min(token_begin + TOKEN_TILE, tokens); ++t) {
                matvec_rows(dot, codes + t * k, code_scales[t], weight, scales, k, first, last, part.data());
                f32_to_bf16(part.data(), last - first, output + t * n + first);
            }
        }
    };
    if (tokens * n * k <= 2 * ITEM_BYTES) {
        body(0, static_cast<std::size_t>(items));
        return {};
    }
    return runtime::ynn::parallel_ranges(static_cast<std::size_t>(items), 1, body);
}

KIDI_CPU_CLONES auto residual_mix(std::span<const float* const> sources, const float* queries,
                                  std::int64_t query_count, std::int64_t tokens, std::int64_t width, float epsilon,
                                  float* output) -> Result<void> {
    constexpr std::size_t MAX_SOURCES = 8;
    const auto source_count = std::min(sources.size(), MAX_SOURCES);
    const auto scale = 1.F / std::sqrt(static_cast<float>(width));
    for (std::int64_t token = 0; token < tokens; ++token) {
        std::array<const float*, MAX_SOURCES> rows{};
        std::array<float, MAX_SOURCES> inverse{};
        for (std::size_t source = 0; source < source_count; ++source) {
            const auto* row = rows[source] = sources[source] + token * width;
            inverse[source] =
                1.F / std::sqrt(sum_lanes(width, [&](std::int64_t i) { return row[i] * row[i]; }) /
                                    static_cast<float>(width) +
                                epsilon);
        }
        for (std::int64_t query = 0; query < query_count; ++query) {
            const auto* q = queries + query * width;
            std::array<float, MAX_SOURCES> weights{};
            auto maximum = -std::numeric_limits<float>::infinity();
            for (std::size_t source = 0; source < source_count; ++source) {
                const auto* row = rows[source];
                weights[source] = sum_lanes(width, [&](std::int64_t i) { return q[i] * row[i]; }) *
                                  inverse[source] * scale;
                maximum = std::max(maximum, weights[source]);
            }
            float denominator = 0;
            for (std::size_t source = 0; source < source_count; ++source)
                denominator += weights[source] = std::exp(weights[source] - maximum);
            for (std::size_t source = 0; source < source_count; ++source)
                weights[source] *= inverse[source] / denominator;
            auto* target = output + (query * tokens + token) * width;
            std::fill_n(target, width, 0.F);
            for (std::size_t source = 0; source < source_count; ++source) {
                const auto* row = rows[source];
                const auto weight = weights[source];
                for (std::int64_t i = 0; i < width; ++i) target[i] += weight * row[i];
            }
        }
    }
    return {};
}

KIDI_CPU_CLONES auto l2_clip(const float* input, const float* scale, std::int64_t scale_rows, std::int64_t rows,
                             std::int64_t width, float threshold, float* output) -> Result<void> {
    for (std::int64_t row = 0; row < rows; ++row) {
        const auto* source = input + row * width;
        auto* target = output + row * width;
        if (scale) {
            const auto* factors = scale + row % scale_rows * width;
            for (std::int64_t i = 0; i < width; ++i) target[i] = source[i] * factors[i];
        } else if (target != source) {
            std::copy_n(source, width, target);
        }
        const auto norm = std::sqrt(sum_lanes(width, [&](std::int64_t i) { return target[i] * target[i]; }));
        const auto factor = std::min(1.F, threshold / (norm + 1e-6F));
        for (std::int64_t i = 0; i < width; ++i) target[i] *= factor;
    }
    return {};
}

KIDI_CPU_CLONES auto bf16_to_f32(const std::uint16_t* input, std::int64_t count, float* output) -> void {
    for (std::int64_t index = 0; index < count; ++index) output[index] = bf16_to_float(input[index]);
}

KIDI_CPU_CLONES auto f32_to_bf16(const float* input, std::int64_t count, std::uint16_t* output) -> void {
    for (std::int64_t index = 0; index < count; ++index) {
        const auto bits = std::bit_cast<std::uint32_t>(input[index]);
        const auto rounded = (bits + 0x7FFFU + ((bits >> 16) & 1U)) >> 16;
        const auto nan = (bits & 0x7FFFFFFFU) > 0x7F800000U;
        output[index] = static_cast<std::uint16_t>(nan ? (bits >> 16) | 0x40U : rounded);
    }
}

KIDI_CPU_CLONES auto quantize_rows(const float* input, std::int64_t rows, std::int64_t width, std::int64_t group,
                                   std::int8_t* values, float* scales) -> void {
    const auto groups = width / group;
    for (std::int64_t row = 0; row < rows; ++row)
        for (std::int64_t block = 0; block < groups; ++block) {
            const auto* source = input + row * width + block * group;
            float maximum = 0;
            for (std::int64_t i = 0; i < group; ++i) maximum = std::max(maximum, std::abs(source[i]));
            const auto scale = maximum > 0 ? maximum / 127.F : 1.F;
            const auto inverse = 1.F / scale;
            auto* target = values + row * width + block * group;
            for (std::int64_t i = 0; i < group; ++i)
                target[i] = static_cast<std::int8_t>(std::clamp(std::nearbyint(source[i] * inverse), -127.F, 127.F));
            scales[row * groups + block] = scale;
        }
}

} // namespace kidi::ops::cpu
