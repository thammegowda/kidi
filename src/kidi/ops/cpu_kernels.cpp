#include "kidi/ops/cpu_kernels.h"

#include "kidi/runtime/ynn/graph.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <type_traits>
#include <vector>

#if defined(__x86_64__) && defined(__GNUC__)
#include <immintrin.h>
#define KIDI_CPU_VNNI 1
#if defined(__linux__)
#include <cpuid.h>
#include <sys/syscall.h>
#include <unistd.h>
#define KIDI_CPU_AMX 1
#endif
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

KIDI_CPU_INLINE auto f16_to_float(std::uint16_t value) -> float {
    const auto wide = static_cast<std::uint32_t>(value) << 16;
    const auto sign = wide & 0x80000000U;
    const auto two_w = wide + wide;
    const auto normalized = std::bit_cast<float>((two_w >> 4) + (0xE0U << 23)) * 0x1.0p-112F;
    const auto denormalized = std::bit_cast<float>((two_w >> 17) | (126U << 23)) - 0.5F;
    const auto bits = sign | (two_w < (1U << 27) ? std::bit_cast<std::uint32_t>(denormalized)
                                                 : std::bit_cast<std::uint32_t>(normalized));
    return std::bit_cast<float>(bits);
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

// Include short speculative verification blocks when there is enough work to pay for dispatch.
template <typename Body>
auto for_rows(std::int64_t rows, std::int64_t width, Body&& body) -> Result<void> {
    constexpr std::int64_t SERIAL_ELEMENTS = 4 * 1024;
    if (rows <= 1 || rows * width <= SERIAL_ELEMENTS) {
        body(std::int64_t{0}, rows);
        return {};
    }
    const auto grain = static_cast<std::size_t>(std::max<std::int64_t>(1, 8 * 1024 / width));
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
    case Rows::Storage::F16: return nullptr;
    }
    return nullptr;
}

auto pick(Rows::Storage key, Rows::Storage value) -> AttendRange {
    switch (key) {
    case Rows::Storage::F32: return pick_value<Rows::Storage::F32>(value);
    case Rows::Storage::BF16: return pick_value<Rows::Storage::BF16>(value);
    case Rows::Storage::I8: return pick_value<Rows::Storage::I8>(value);
    case Rows::Storage::F16: return nullptr;
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

auto flash_supported(const AttentionArgs& args) -> bool;
auto flash_attention(const AttentionArgs& args) -> Result<void>;

} // namespace

auto relative_attention(const float* query, const Rows& key, const Rows& value, const float* relative,
                        const float* sink, const float* gate, std::span<const std::int32_t> positions,
                        const RelativeAttentionShape& shape, float* output) -> Result<void> {
    const auto run = pick(key.storage, value.storage);
    AttentionArgs args{query, &key,  &value, relative, sink, gate, positions, &shape,
                       attention_splits(shape, positions), output, nullptr};
    if (flash_supported(args)) return flash_attention(args);
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

// AMX INT8 multiplies a 16-row x 64-byte A tile by a (16 x 4)-row x 16-column B tile into a 16x16 INT32 tile.
// Row-major weights are A tiles as stored; only the (small) token block is repacked into B order, so prefill
// can use AMX without changing the weight layout that decode kernels read.
constexpr std::int64_t AMX_TOKENS = 16, AMX_ROWS = 16, AMX_DEPTH = 64;

auto amx_ready() -> bool {
#ifdef KIDI_CPU_AMX
    static const bool ready = [] {
        unsigned a = 0, b = 0, c = 0, d = 0;
        if (!__get_cpuid_count(7, 0, &a, &b, &c, &d) || !(d & (1U << 24)) || !(d & (1U << 25))) return false;
        // Linux keeps AMX tile state disabled until the process requests it (ARCH_REQ_XCOMP_PERM, XTILEDATA).
        return syscall(SYS_arch_prctl, 0x1023, 18) == 0;
    }();
    return ready;
#else
    return false;
#endif
}

// B tile order: for each 4-byte depth quad, the quads of up to 16 token rows side by side (missing rows are 0).
auto pack_tokens(std::span<const std::int8_t* const> rows, std::int64_t depth, std::uint32_t* packed) -> void {
    const auto quads = depth / 4;
    for (std::size_t t = 0; t < static_cast<std::size_t>(AMX_TOKENS); ++t)
        for (std::int64_t q = 0; q < quads; ++q) {
            std::uint32_t value = 0;
            if (t < rows.size()) std::memcpy(&value, rows[t] + q * 4, 4);
            packed[q * AMX_TOKENS + static_cast<std::int64_t>(t)] = value;
        }
}

#ifdef KIDI_CPU_AMX
// out[b][r][t] = dot(blocks[b] row r, token t) for BLOCKS 16-row weight blocks against one packed token tile.
template <int BLOCKS>
__attribute__((target("amx-tile,amx-int8"))) auto amx_dot(const std::array<const std::int8_t*, 4>& blocks,
                                                           std::int64_t depth, const std::uint32_t* packed,
                                                           std::int32_t* out) -> void {
    static_assert(BLOCKS >= 1 && BLOCKS <= 4);
    struct alignas(64) {
        std::uint8_t palette = 1, start_row = 0, reserved[14]{};
        std::uint16_t columns[16]{};
        std::uint8_t rows[16]{};
    } config;
    for (int tile = 0; tile < 7; ++tile) config.columns[tile] = 64, config.rows[tile] = 16;
    _tile_loadconfig(&config);
    _tile_zero(0);
    if constexpr (BLOCKS > 1) _tile_zero(1);
    if constexpr (BLOCKS > 2) _tile_zero(2);
    if constexpr (BLOCKS > 3) _tile_zero(3);
    for (std::int64_t k = 0; k < depth; k += AMX_DEPTH) {
        _tile_loadd(6, packed + k / 4 * AMX_TOKENS, AMX_TOKENS * 4);
        _tile_loadd(4, blocks[0] + k, depth);
        _tile_dpbssd(0, 4, 6);
        if constexpr (BLOCKS > 1) {
            _tile_loadd(5, blocks[1] + k, depth);
            _tile_dpbssd(1, 5, 6);
        }
        if constexpr (BLOCKS > 2) {
            _tile_loadd(4, blocks[2] + k, depth);
            _tile_dpbssd(2, 4, 6);
        }
        if constexpr (BLOCKS > 3) {
            _tile_loadd(5, blocks[3] + k, depth);
            _tile_dpbssd(3, 5, 6);
        }
    }
    constexpr auto STRIDE = AMX_TOKENS * 4, TILE = AMX_ROWS * AMX_TOKENS;
    _tile_stored(0, out, STRIDE);
    if constexpr (BLOCKS > 1) _tile_stored(1, out + TILE, STRIDE);
    if constexpr (BLOCKS > 2) _tile_stored(2, out + 2 * TILE, STRIDE);
    if constexpr (BLOCKS > 3) _tile_stored(3, out + 3 * TILE, STRIDE);
}
#else
template <int BLOCKS>
auto amx_dot(const std::array<const std::int8_t*, 4>&, std::int64_t, const std::uint32_t*, std::int32_t*) -> void {}
#endif

// INT8 flash attention for prefill over an INT8 KV cache. One item is (key head, up to 64 query rows from
// consecutive tokens); keys stream in blocks of 16. Q.K runs as exact INT8 dots: each query scale group is stored
// as two INT8 codes, the value and its quantization residual (q ~= s*c + s_r*c_r), so the logits keep the accuracy
// of the float-query decode kernel. Softmax and P.V accumulate in F32.
constexpr std::int64_t FLASH_ROWS = 64, FLASH_KEYS = 16, FLASH_TILE = AMX_ROWS * FLASH_KEYS;

auto flash_tokens(const RelativeAttentionShape& shape) -> std::int64_t {
    return std::min<std::int64_t>(16, FLASH_ROWS / (shape.heads / shape.key_heads));
}

// dots[block][group][row][key] for 16-row query blocks against up to 16 key rows.
#ifdef KIDI_CPU_AMX
// Tiles 0-3 hold per-group dots, tile 4 a 16-row query group slice, tile 5 the matching packed key slice.
// Loaded once per thread range; amx_group_dots relies on it.
__attribute__((target("amx-tile,amx-int8"))) auto amx_flash_config(std::int64_t group_width) -> void {
    struct alignas(64) {
        std::uint8_t palette = 1, start_row = 0, reserved[14]{};
        std::uint16_t columns[16]{};
        std::uint8_t rows[16]{};
    } config;
    for (int tile = 0; tile < 4; ++tile) config.columns[tile] = 64, config.rows[tile] = 16;
    config.columns[4] = static_cast<std::uint16_t>(group_width), config.rows[4] = 16;
    config.columns[5] = 64, config.rows[5] = static_cast<std::uint8_t>(group_width / 4);
    _tile_loadconfig(&config);
}

template <int GROUPS>
__attribute__((target("amx-tile,amx-int8"))) auto amx_group_dots(const std::int8_t* queries, std::int64_t blocks,
                                                                  std::int64_t width, const std::uint32_t* packed,
                                                                  std::int32_t* dots) -> void {
    const auto group_width = width / GROUPS;
    constexpr auto STRIDE = FLASH_KEYS * 4;
    for (std::int64_t block = 0; block < blocks; ++block) {
        const auto* rows = queries + block * AMX_ROWS * width;
        auto* out = dots + block * GROUPS * FLASH_TILE;
        const auto* keys = packed;
        const auto quads = group_width / 4 * FLASH_KEYS;
        _tile_zero(0);
        _tile_loadd(4, rows, width);
        _tile_loadd(5, keys, STRIDE);
        _tile_dpbssd(0, 4, 5);
        _tile_stored(0, out, STRIDE);
        if constexpr (GROUPS > 1) {
            _tile_zero(1);
            _tile_loadd(4, rows + group_width, width);
            _tile_loadd(5, keys + quads, STRIDE);
            _tile_dpbssd(1, 4, 5);
            _tile_stored(1, out + FLASH_TILE, STRIDE);
        }
        if constexpr (GROUPS > 2) {
            _tile_zero(2);
            _tile_loadd(4, rows + 2 * group_width, width);
            _tile_loadd(5, keys + 2 * quads, STRIDE);
            _tile_dpbssd(2, 4, 5);
            _tile_stored(2, out + 2 * FLASH_TILE, STRIDE);
        }
        if constexpr (GROUPS > 3) {
            _tile_zero(3);
            _tile_loadd(4, rows + 3 * group_width, width);
            _tile_loadd(5, keys + 3 * quads, STRIDE);
            _tile_dpbssd(3, 4, 5);
            _tile_stored(3, out + 3 * FLASH_TILE, STRIDE);
        }
    }
}
#endif

KIDI_CPU_CLONES auto portable_group_dots(const std::int8_t* queries, std::int64_t blocks, std::int64_t width,
                                         std::int64_t group_width, std::span<const std::int8_t* const> keys,
                                         std::int32_t* dots) -> void {
    const auto groups = width / group_width;
    std::fill_n(dots, blocks * groups * FLASH_TILE, 0);
    for (std::int64_t block = 0; block < blocks; ++block)
        for (std::int64_t g = 0; g < groups; ++g)
            for (std::int64_t r = 0; r < AMX_ROWS; ++r) {
                const auto* query = queries + (block * AMX_ROWS + r) * width + g * group_width;
                auto* out = dots + (block * groups + g) * FLASH_TILE + r * FLASH_KEYS;
                for (std::size_t k = 0; k < keys.size(); ++k) {
                    const auto* key = keys[k] + g * group_width;
                    std::int32_t sum = 0;
                    for (std::int64_t i = 0; i < group_width; ++i)
                        sum += std::int32_t{query[i]} * std::int32_t{key[i]};
                    out[k] = sum;
                }
            }
}

// accumulators[r] = accumulators[r] * corrections[r] + sum_k probabilities[r][k] * values[k] for one key block.
KIDI_CPU_CLONES auto portable_flash_pv(std::int64_t rows, std::int64_t width, std::int64_t count,
                                       const float* probabilities, const float* corrections, const float* values,
                                       float* accumulators) -> void {
    for (std::int64_t r = 0; r < rows; ++r) {
        auto* out = accumulators + r * width;
        const auto* weights = probabilities + r * FLASH_KEYS;
        for (std::int64_t i = 0; i < width; ++i) out[i] *= corrections[r];
        for (std::int64_t k = 0; k < count; ++k)
            for (std::int64_t i = 0; i < width; ++i) out[i] += weights[k] * values[k * width + i];
    }
}

#ifdef KIDI_CPU_VNNI
// Keeps a 64-column accumulator slice in registers across the key block; needs width % 64 == 0.
__attribute__((target("avx512f"))) auto avx512_flash_pv(std::int64_t rows, std::int64_t width, std::int64_t count,
                                                        const float* probabilities, const float* corrections,
                                                        const float* values, float* accumulators) -> void {
    for (std::int64_t r = 0; r < rows; ++r) {
        auto* out = accumulators + r * width;
        const auto* weights = probabilities + r * FLASH_KEYS;
        const auto correction = _mm512_set1_ps(corrections[r]);
        for (std::int64_t c = 0; c < width; c += 64) {
            auto a0 = _mm512_mul_ps(_mm512_loadu_ps(out + c), correction);
            auto a1 = _mm512_mul_ps(_mm512_loadu_ps(out + c + 16), correction);
            auto a2 = _mm512_mul_ps(_mm512_loadu_ps(out + c + 32), correction);
            auto a3 = _mm512_mul_ps(_mm512_loadu_ps(out + c + 48), correction);
            for (std::int64_t k = 0; k < count; ++k) {
                const auto weight = _mm512_set1_ps(weights[k]);
                const auto* value = values + k * width + c;
                a0 = _mm512_fmadd_ps(weight, _mm512_loadu_ps(value), a0);
                a1 = _mm512_fmadd_ps(weight, _mm512_loadu_ps(value + 16), a1);
                a2 = _mm512_fmadd_ps(weight, _mm512_loadu_ps(value + 32), a2);
                a3 = _mm512_fmadd_ps(weight, _mm512_loadu_ps(value + 48), a3);
            }
            _mm512_storeu_ps(out + c, a0), _mm512_storeu_ps(out + c + 16, a1);
            _mm512_storeu_ps(out + c + 32, a2), _mm512_storeu_ps(out + c + 48, a3);
        }
    }
}
#endif

struct FlashQuery {
    std::int64_t row, position, first;
    float tau;
};

// One key block of online softmax for every query row: turns integer group dots into logits (scales, relative
// bias, log scaling, causal/window mask), updates the running maximum and sum, and leaves exp weights and the
// accumulator correction for flash_pv.
struct FlashSoftmax {
    const FlashQuery* queries;
    std::int64_t rows, groups, extent, key_begin, count;
    const std::int32_t* dots;      // [row block][group][16 rows][16 keys]
    const std::int32_t* residual_dots;
    const float* query_scales;     // [row][group]
    const float* residual_scales;
    const float* key_scales;       // [group][16 keys], premultiplied by 1/sqrt(width)
    const float* relative;         // [row][extent]
    float *maxima, *sums, *probabilities, *corrections;
};

KIDI_CPU_INLINE auto flash_row_dots(const std::int32_t* dots, std::int64_t groups, std::int64_t r)
    -> const std::int32_t* {
    return dots + (r / AMX_ROWS) * groups * FLASH_TILE + (r % AMX_ROWS) * FLASH_KEYS;
}

KIDI_CPU_CLONES auto portable_flash_softmax(const FlashSoftmax& s) -> void {
    for (std::int64_t r = 0; r < s.rows; ++r) {
        const auto& work = s.queries[r];
        auto* weights = s.probabilities + r * FLASH_KEYS;
        const auto first_k = std::max(s.key_begin, work.first) - s.key_begin;
        const auto last_k = std::min(s.key_begin + s.count, work.position + 1) - s.key_begin;
        if (first_k >= last_k) {
            std::fill_n(weights, FLASH_KEYS, 0.F);
            s.corrections[r] = 1.F;
            continue;
        }
        const auto* row_dots = flash_row_dots(s.dots, s.groups, r);
        const auto* row_residual = flash_row_dots(s.residual_dots, s.groups, r);
        const auto* relative = s.relative + work.row * s.extent;
        std::array<float, FLASH_KEYS> logits{};
        for (std::int64_t g = 0; g < s.groups; ++g) {
            const auto query_scale = s.query_scales[r * s.groups + g];
            const auto residual_scale = s.residual_scales[r * s.groups + g];
            for (std::int64_t k = 0; k < FLASH_KEYS; ++k) {
                const auto dot = static_cast<float>(row_dots[g * FLASH_TILE + k]) * query_scale +
                                 static_cast<float>(row_residual[g * FLASH_TILE + k]) * residual_scale;
                logits[static_cast<std::size_t>(k)] += dot * s.key_scales[g * FLASH_KEYS + k];
            }
        }
        auto block_max = NO_LOGIT;
        for (auto k = first_k; k < last_k; ++k) {
            const auto distance = work.position - s.key_begin - k;
            auto& logit = logits[static_cast<std::size_t>(k)];
            logit = (logit + (distance < s.extent ? relative[distance] : 0.F)) * work.tau;
            block_max = std::max(block_max, logit);
        }
        const auto next = std::max(s.maxima[r], block_max);
        const auto correction = fast_exp(s.maxima[r] - next);
        s.maxima[r] = next;
        float total = 0;
        for (std::int64_t k = 0; k < FLASH_KEYS; ++k) {
            const auto weight = k >= first_k && k < last_k ? fast_exp(logits[static_cast<std::size_t>(k)] - next) : 0.F;
            weights[k] = weight;
            total += weight;
        }
        s.sums[r] = s.sums[r] * correction + total;
        s.corrections[r] = correction;
    }
}

#ifdef KIDI_CPU_VNNI
__attribute__((target("avx512f"))) inline auto exp_avx512(__m512 x) -> __m512 {
    x = _mm512_min_ps(_mm512_set1_ps(88.F), _mm512_max_ps(_mm512_set1_ps(-87.F), x));
    const auto t = _mm512_mul_ps(x, _mm512_set1_ps(1.44269504088896341F));
    const auto whole = _mm512_roundscale_ps(t, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    const auto f = _mm512_sub_ps(t, whole);
    auto p = _mm512_set1_ps(1.535336188319500e-4F);
    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(1.339887440266574e-3F));
    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(9.618437357674640e-3F));
    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(5.550332471162809e-2F));
    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(2.402264791363012e-1F));
    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(6.931472028550421e-1F));
    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(1.F));
    return _mm512_scalef_ps(p, whole);
}

__attribute__((target("avx512f"))) auto avx512_flash_softmax(const FlashSoftmax& s) -> void {
    const auto reverse = _mm512_set_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    for (std::int64_t r = 0; r < s.rows; ++r) {
        const auto& work = s.queries[r];
        auto* weights = s.probabilities + r * FLASH_KEYS;
        const auto first_k = std::max(s.key_begin, work.first) - s.key_begin;
        const auto last_k = std::min(s.key_begin + s.count, work.position + 1) - s.key_begin;
        if (first_k >= last_k) {
            _mm512_storeu_ps(weights, _mm512_setzero_ps());
            s.corrections[r] = 1.F;
            continue;
        }
        const auto* row_dots = flash_row_dots(s.dots, s.groups, r);
        const auto* row_residual = flash_row_dots(s.residual_dots, s.groups, r);
        auto logits = _mm512_setzero_ps();
        for (std::int64_t g = 0; g < s.groups; ++g) {
            auto dot = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_loadu_si512(row_dots + g * FLASH_TILE)),
                                     _mm512_set1_ps(s.query_scales[r * s.groups + g]));
            dot = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_loadu_si512(row_residual + g * FLASH_TILE)),
                                  _mm512_set1_ps(s.residual_scales[r * s.groups + g]), dot);
            logits = _mm512_fmadd_ps(dot, _mm512_loadu_ps(s.key_scales + g * FLASH_KEYS), logits);
        }
        // Key k sits at distance d0 - k, so the 16 biases are relative[d0 - 15 .. d0] reversed.
        const auto d0 = work.position - s.key_begin;
        const auto* relative = s.relative + work.row * s.extent;
        __m512 bias;
        if (d0 >= FLASH_KEYS - 1 && d0 < s.extent) {
            bias = _mm512_permutexvar_ps(reverse, _mm512_loadu_ps(relative + d0 - (FLASH_KEYS - 1)));
        } else {
            alignas(64) std::array<float, FLASH_KEYS> lanes{};
            for (std::int64_t k = 0; k < FLASH_KEYS; ++k) {
                const auto distance = d0 - k;
                lanes[static_cast<std::size_t>(k)] = distance >= 0 && distance < s.extent ? relative[distance] : 0.F;
            }
            bias = _mm512_load_ps(lanes.data());
        }
        const auto valid = static_cast<__mmask16>(((1U << last_k) - 1U) & ~((1U << first_k) - 1U));
        logits = _mm512_mul_ps(_mm512_add_ps(logits, bias), _mm512_set1_ps(work.tau));
        logits = _mm512_mask_mov_ps(_mm512_set1_ps(NO_LOGIT), valid, logits);
        const auto next = std::max(s.maxima[r], _mm512_reduce_max_ps(logits));
        const auto correction = fast_exp(s.maxima[r] - next);
        s.maxima[r] = next;
        const auto p = _mm512_maskz_mov_ps(valid, exp_avx512(_mm512_sub_ps(logits, _mm512_set1_ps(next))));
        _mm512_storeu_ps(weights, p);
        s.sums[r] = s.sums[r] * correction + _mm512_reduce_add_ps(p);
        s.corrections[r] = correction;
    }
}
#endif

auto flash_softmax() -> void (*)(const FlashSoftmax&) {
#ifdef KIDI_CPU_VNNI
    static const bool avx512 = __builtin_cpu_supports("avx512f");
    if (avx512) return &avx512_flash_softmax;
#endif
    return &portable_flash_softmax;
}

using FlashPv = void (*)(std::int64_t, std::int64_t, std::int64_t, const float*, const float*, const float*, float*);

auto flash_pv(std::int64_t width) -> FlashPv {
#ifdef KIDI_CPU_VNNI
    static const bool avx512 = __builtin_cpu_supports("avx512f");
    if (avx512 && width % 64 == 0) return &avx512_flash_pv;
#endif
    return &portable_flash_pv;
}

auto flash_supported(const AttentionArgs& a) -> bool {
    const auto& shape = *a.shape;
    const auto& key = *a.key;
    const auto group = shape.heads / shape.key_heads;
    if (a.splits != 1 || shape.tokens < 16 || shape.extent <= 0 || group > FLASH_ROWS ||
        key.storage != Rows::Storage::I8 || a.value->storage != Rows::Storage::I8 || key.scale_group <= 0 ||
        key.scale_group % 4 || shape.width % key.scale_group || key.stride % key.scale_group)
        return false;
    const auto groups = shape.width / key.scale_group;
    return groups >= 1 && groups <= 4 && (key.scale_group <= 64 || !amx_ready());
}

KIDI_CPU_CLONES auto flash_range(std::size_t begin, std::size_t end, const AttentionArgs& a) -> void {
    const auto& shape = *a.shape;
    const auto& keys = *a.key;
    const auto width = shape.width, group = shape.heads / shape.key_heads;
    const auto group_width = keys.scale_group, groups = width / group_width;
    const auto scale_stride = keys.stride / group_width;
    const auto scale = 1.F / std::sqrt(static_cast<float>(width));
    const auto tile_tokens = flash_tokens(shape);
#ifdef KIDI_CPU_AMX
    const bool amx = amx_ready();
#else
    const bool amx = false;
#endif
    thread_local std::vector<FlashQuery> queries;
    thread_local std::vector<std::int8_t> codes, residual_codes;
    thread_local std::vector<float> query_scales, residual_scales, residual, accumulators, maxima, sums, values, probabilities, corrections;
    thread_local std::vector<std::uint32_t> packed;
    thread_local std::vector<std::int32_t> dots, residual_dots;
    codes.resize(static_cast<std::size_t>(FLASH_ROWS * width));
    residual_codes.resize(codes.size());
    query_scales.resize(static_cast<std::size_t>(FLASH_ROWS * groups));
    residual_scales.resize(query_scales.size());
    residual.resize(static_cast<std::size_t>(group_width));
    accumulators.resize(static_cast<std::size_t>(FLASH_ROWS * width));
    maxima.resize(FLASH_ROWS);
    sums.resize(FLASH_ROWS);
    values.resize(static_cast<std::size_t>(FLASH_KEYS * width));
    probabilities.resize(static_cast<std::size_t>(FLASH_ROWS * FLASH_KEYS));
    corrections.resize(FLASH_ROWS);
    const auto pv = flash_pv(width);
    const auto softmax = flash_softmax();
    packed.resize(static_cast<std::size_t>(width / 4 * FLASH_KEYS));
    dots.resize(static_cast<std::size_t>(FLASH_ROWS / AMX_ROWS * groups * FLASH_TILE));
    residual_dots.resize(dots.size());
    const auto* cache = static_cast<const std::int8_t*>(keys.data);
#ifdef KIDI_CPU_AMX
    if (amx) amx_flash_config(group_width);
#endif
    for (auto item = static_cast<std::int64_t>(begin); item < static_cast<std::int64_t>(end); ++item) {
        const auto key_head = item % shape.key_heads;
        const auto token_begin = item / shape.key_heads * tile_tokens;
        const auto token_end = std::min(token_begin + tile_tokens, shape.tokens);
        const auto column = key_head * width;
        queries.clear();
        auto lo = std::numeric_limits<std::int64_t>::max(), hi = std::int64_t{0};
        for (auto token = token_begin; token < token_end; ++token) {
            const auto pos = static_cast<std::int64_t>(a.positions[static_cast<std::size_t>(token)]);
            const auto first = shape.window ? std::max<std::int64_t>(0, pos - shape.window + 1) : 0;
            const auto tau = shape.log_floor > 0
                                 ? 1.F + shape.log_alpha *
                                             std::log(std::max(1.F, static_cast<float>(pos + 1) / shape.log_floor))
                                 : 1.F;
            lo = std::min(lo, first);
            hi = std::max(hi, pos + 1);
            for (std::int64_t head = 0; head < group; ++head)
                queries.push_back({token * shape.heads + key_head * group + head, pos, first, tau});
        }
        const auto rows = static_cast<std::int64_t>(queries.size());
        const auto blocks = (rows + AMX_ROWS - 1) / AMX_ROWS;
        for (std::int64_t r = 0; r < rows; ++r) {
            const auto* query = a.query + queries[r].row * width;
            for (std::int64_t g = 0; g < groups; ++g) {
                const auto* values_in = query + g * group_width;
                const auto offset = r * width + g * group_width;
                const auto at = static_cast<std::size_t>(r * groups + g);
                query_scales[at] = quantize_vector(values_in, group_width, codes.data() + offset);
                for (std::int64_t i = 0; i < group_width; ++i)
                    residual[static_cast<std::size_t>(i)] =
                        values_in[i] - query_scales[at] * static_cast<float>(codes[static_cast<std::size_t>(offset + i)]);
                residual_scales[at] = quantize_vector(residual.data(), group_width, residual_codes.data() + offset);
            }
            maxima[static_cast<std::size_t>(r)] = a.sink[queries[r].row];
            sums[static_cast<std::size_t>(r)] = 1.F; // exp(sink - running maximum)
            std::fill_n(accumulators.data() + r * width, width, 0.F);
        }
        std::fill(codes.begin() + rows * width, codes.begin() + blocks * AMX_ROWS * width, std::int8_t{0});
        std::fill(residual_codes.begin() + rows * width, residual_codes.begin() + blocks * AMX_ROWS * width,
                  std::int8_t{0});
        for (auto key_begin = lo; key_begin < hi; key_begin += FLASH_KEYS) {
            const auto count = std::min(FLASH_KEYS, hi - key_begin);
            std::array<const std::int8_t*, FLASH_KEYS> key_rows{};
            std::array<float, 4 * FLASH_KEYS> key_scales{};
            for (std::int64_t k = 0; k < count; ++k) {
                const auto index = key_begin + k;
                key_rows[static_cast<std::size_t>(k)] = cache + index * keys.stride + column;
                for (std::int64_t g = 0; g < groups; ++g)
                    key_scales[static_cast<std::size_t>(g * FLASH_KEYS + k)] =
                        keys.scales[index * scale_stride + key_head * groups + g] * scale;
                load_row<Rows::Storage::I8>(*a.value, index, column, width, values.data() + k * width);
            }
            const auto present = std::span<const std::int8_t* const>(key_rows.data(), static_cast<std::size_t>(count));
            if (amx) {
#ifdef KIDI_CPU_AMX
                pack_tokens(present, width, packed.data());
                for (const auto& [input, output] : {std::pair{codes.data(), dots.data()},
                                                    std::pair{residual_codes.data(), residual_dots.data()}}) {
                    switch (groups) {
                    case 1: amx_group_dots<1>(input, blocks, width, packed.data(), output); break;
                    case 2: amx_group_dots<2>(input, blocks, width, packed.data(), output); break;
                    case 3: amx_group_dots<3>(input, blocks, width, packed.data(), output); break;
                    default: amx_group_dots<4>(input, blocks, width, packed.data(), output); break;
                    }
                }
#endif
            } else {
                portable_group_dots(codes.data(), blocks, width, group_width, present, dots.data());
                portable_group_dots(residual_codes.data(), blocks, width, group_width, present, residual_dots.data());
            }
            softmax({queries.data(), rows, groups, shape.extent, key_begin, count, dots.data(), residual_dots.data(),
                     query_scales.data(), residual_scales.data(), key_scales.data(), a.relative, maxima.data(), sums.data(), probabilities.data(),
                     corrections.data()});
            pv(rows, width, count, probabilities.data(), corrections.data(), values.data(), accumulators.data());
        }
        for (std::int64_t r = 0; r < rows; ++r) {
            const auto row = queries[static_cast<std::size_t>(r)].row;
            const auto inverse = 1.F / sums[static_cast<std::size_t>(r)];
            const auto* gate = a.gate + row * width;
            const auto* in = accumulators.data() + r * width;
            auto* out = a.output + row * width;
            for (std::int64_t i = 0; i < width; ++i) out[i] = in[i] * inverse * fast_sigmoid(gate[i]);
        }
    }
}

auto flash_attention(const AttentionArgs& args) -> Result<void> {
    const auto& shape = *args.shape;
    const auto tiles = (shape.tokens + flash_tokens(shape) - 1) / flash_tokens(shape);
    return runtime::ynn::parallel_ranges(static_cast<std::size_t>(tiles * shape.key_heads), 1,
                                         [&](std::size_t begin, std::size_t end) { flash_range(begin, end, args); });
}

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

namespace {

// AMX variant of expert_prefill after input quantization: same integer sums and scaling order as the VNNI path.
auto expert_prefill_amx(const std::int8_t* x, const float* xs, std::int64_t tokens,
                        std::span<const RoutedExpert> experts, std::span<const std::size_t> offsets,
                        std::span<const std::vector<std::size_t>> contributions, std::int64_t hidden,
                        std::int64_t intermediate, float gate_cap, float up_cap, float* output) -> Result<void> {
    constexpr std::int64_t CHUNK = 2 * AMX_ROWS;
    constexpr auto TILE = AMX_ROWS * AMX_TOKENS;
    std::vector<std::size_t> groups{0};
    for (const auto& expert : experts)
        groups.push_back(groups.back() + (expert.tokens.size() + AMX_TOKENS - 1) / AMX_TOKENS);
    const auto count = offsets.back();
    thread_local std::vector<std::uint32_t> packed_input, packed_hidden;
    thread_local std::vector<std::int8_t> hidden_codes;
    thread_local std::vector<float> hidden_scales, gate_up, projected;
    packed_input.resize(groups.back() * static_cast<std::size_t>(hidden / 4 * AMX_TOKENS));
    packed_hidden.resize(groups.back() * static_cast<std::size_t>(intermediate / 4 * AMX_TOKENS));
    hidden_codes.resize(count * intermediate);
    hidden_scales.resize(count);
    gate_up.resize(count * 2 * intermediate);
    projected.resize(count * hidden);
    auto* px = packed_input.data();
    auto* ph = packed_hidden.data();
    auto* h = hidden_codes.data();
    auto* hs = hidden_scales.data();
    auto* gu = gate_up.data();
    auto* parts = projected.data();
    const auto pack = [&](std::size_t e, std::int64_t depth, std::uint32_t* packed, auto&& row) {
        const auto& expert = experts[e];
        std::array<const std::int8_t*, AMX_TOKENS> rows{};
        for (std::size_t first = 0, group = groups[e]; first < expert.tokens.size(); first += AMX_TOKENS, ++group) {
            const auto n = std::min<std::size_t>(AMX_TOKENS, expert.tokens.size() - first);
            for (std::size_t r = 0; r < n; ++r) rows[r] = row(first + r);
            pack_tokens(std::span(rows).first(n), depth, packed + group * (depth / 4 * AMX_TOKENS));
        }
    };
    if (auto status = runtime::ynn::parallel_ranges(experts.size(), 1, [&](std::size_t begin, std::size_t end) {
            for (auto e = begin; e < end; ++e)
                pack(e, hidden, px, [&](std::size_t r) { return x + experts[e].tokens[r] * hidden; });
        });
        !status)
        return status;
    const auto up_chunks = static_cast<std::size_t>(intermediate / CHUNK);
    if (auto status = runtime::ynn::parallel_ranges(experts.size() * up_chunks, 1, [&](std::size_t begin,
                                                                                        std::size_t end) {
            alignas(64) std::array<std::int32_t, 4 * TILE> sums;
            for (auto job = begin; job < end; ++job) {
                const auto e = job / up_chunks;
                const auto first = static_cast<std::int64_t>(job % up_chunks) * CHUNK;
                const auto& expert = experts[e];
                const auto* w = expert.weight.gate_up;
                const auto* scales = expert.weight.gate_up_scales;
                const std::array blocks{w + first * hidden, w + (first + AMX_ROWS) * hidden,
                                        w + (intermediate + first) * hidden,
                                        w + (intermediate + first + AMX_ROWS) * hidden};
                for (std::size_t lead = 0, group = groups[e]; lead < expert.tokens.size();
                     lead += AMX_TOKENS, ++group) {
                    amx_dot<4>(blocks, hidden, px + group * (hidden / 4 * AMX_TOKENS), sums.data());
                    const auto n = std::min<std::size_t>(AMX_TOKENS, expert.tokens.size() - lead);
                    for (std::size_t t = 0; t < n; ++t) {
                        const auto scale = xs[expert.tokens[lead + t]];
                        auto* gate = gu + (offsets[e] + lead + t) * 2 * intermediate + first;
                        auto* up = gate + intermediate;
                        for (std::int64_t i = 0; i < CHUNK; ++i) {
                            const auto block = i / AMX_ROWS, row = i % AMX_ROWS;
                            gate[i] = static_cast<float>(sums[block * TILE + row * AMX_TOKENS + t]) *
                                      scales[first + i] * scale;
                            up[i] = static_cast<float>(sums[(2 + block) * TILE + row * AMX_TOKENS + t]) *
                                    scales[intermediate + first + i] * scale;
                        }
                        activate(gate, up, CHUNK, gate_cap, up_cap);
                    }
                }
            }
        });
        !status)
        return status;
    if (auto status = runtime::ynn::parallel_ranges(experts.size(), 1, [&](std::size_t begin, std::size_t end) {
            for (auto e = begin; e < end; ++e) {
                for (std::size_t r = 0; r < experts[e].tokens.size(); ++r) {
                    const auto row = offsets[e] + r;
                    hs[row] = quantize_vector(gu + row * 2 * intermediate, intermediate, h + row * intermediate) *
                              experts[e].routes[r];
                }
                pack(e, intermediate, ph, [&](std::size_t r) { return h + (offsets[e] + r) * intermediate; });
            }
        });
        !status)
        return status;
    const auto down_chunks = static_cast<std::size_t>(hidden / CHUNK);
    if (auto status = runtime::ynn::parallel_ranges(experts.size() * down_chunks, 1, [&](std::size_t begin,
                                                                                          std::size_t end) {
            alignas(64) std::array<std::int32_t, 2 * TILE> sums;
            for (auto job = begin; job < end; ++job) {
                const auto e = job / down_chunks;
                const auto first = static_cast<std::int64_t>(job % down_chunks) * CHUNK;
                const auto& expert = experts[e];
                const auto* w = expert.weight.down;
                const auto* scales = expert.weight.down_scales;
                const std::array blocks{w + first * intermediate, w + (first + AMX_ROWS) * intermediate,
                                        static_cast<const std::int8_t*>(nullptr),
                                        static_cast<const std::int8_t*>(nullptr)};
                for (std::size_t lead = 0, group = groups[e]; lead < expert.tokens.size();
                     lead += AMX_TOKENS, ++group) {
                    amx_dot<2>(blocks, intermediate, ph + group * (intermediate / 4 * AMX_TOKENS), sums.data());
                    const auto n = std::min<std::size_t>(AMX_TOKENS, expert.tokens.size() - lead);
                    for (std::size_t t = 0; t < n; ++t) {
                        const auto row = offsets[e] + lead + t;
                        auto* out = parts + row * hidden + first;
                        for (std::int64_t i = 0; i < CHUNK; ++i)
                            out[i] = static_cast<float>(sums[i / AMX_ROWS * TILE + i % AMX_ROWS * AMX_TOKENS + t]) *
                                     scales[first + i] * hs[row];
                    }
                }
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

} // namespace

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
    if (amx_ready() && hidden % AMX_DEPTH == 0 && intermediate % AMX_DEPTH == 0 && CHUNK == 2 * AMX_ROWS)
        return expert_prefill_amx(x, xs, tokens, experts, offsets, contributions, hidden, intermediate, gate_cap,
                                  up_cap, output);
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

namespace {
constexpr std::size_t MAX_MIX_SOURCES = 8;

KIDI_CPU_CLONES auto residual_mix_rows(std::span<const float* const> sources, const float* queries,
                                       std::int64_t query_count, std::int64_t tokens, std::int64_t width,
                                       float epsilon, std::int64_t first, std::int64_t last, float* output) -> void {
    const auto source_count = std::min(sources.size(), MAX_MIX_SOURCES);
    const auto scale = 1.F / std::sqrt(static_cast<float>(width));
    for (auto token = first; token < last; ++token) {
        std::array<const float*, MAX_MIX_SOURCES> rows{};
        std::array<float, MAX_MIX_SOURCES> inverse{};
        for (std::size_t source = 0; source < source_count; ++source) {
            const auto* row = rows[source] = sources[source] + token * width;
            inverse[source] =
                1.F / std::sqrt(sum_lanes(width, [&](std::int64_t i) { return row[i] * row[i]; }) /
                                    static_cast<float>(width) +
                                epsilon);
        }
        for (std::int64_t query = 0; query < query_count; ++query) {
            const auto* q = queries + query * width;
            std::array<float, MAX_MIX_SOURCES> weights{};
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
}

KIDI_CPU_CLONES auto l2_clip_rows(const float* input, const float* scale, std::int64_t scale_rows,
                                  std::int64_t width, float threshold, std::int64_t first, std::int64_t last,
                                  float* output) -> void {
    for (auto row = first; row < last; ++row) {
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
}
} // namespace

auto residual_mix(std::span<const float* const> sources, const float* queries, std::int64_t query_count,
                  std::int64_t tokens, std::int64_t width, float epsilon, float* output) -> Result<void> {
    const auto work = width * static_cast<std::int64_t>(std::min(sources.size(), MAX_MIX_SOURCES)) * query_count;
    return for_rows(tokens, work, [&](std::int64_t first, std::int64_t last) {
        residual_mix_rows(sources, queries, query_count, tokens, width, epsilon, first, last, output);
    });
}

auto l2_clip(const float* input, const float* scale, std::int64_t scale_rows, std::int64_t rows, std::int64_t width,
             float threshold, float* output) -> Result<void> {
    return for_rows(rows, width, [&](std::int64_t first, std::int64_t last) {
        l2_clip_rows(input, scale, scale_rows, width, threshold, first, last, output);
    });
}

namespace {
template <Rows::Storage Source, typename T>
KIDI_CPU_INLINE auto pack_int8_rows_impl(const T* input, std::int64_t rows, std::int64_t width,
                                         std::uint8_t* output, float* scales) -> bool {
    const auto read = [](T value) {
        if constexpr (Source == Rows::Storage::F32) return value;
        else if constexpr (Source == Rows::Storage::BF16) return bf16_to_float(value);
        else return f16_to_float(value);
    };
    for (std::int64_t row = 0; row < rows; ++row) {
        const auto* source = input + row * width;
        auto* target = output + row * width;
        float maximum = 0;
        bool finite = true;
        for (std::int64_t column = 0; column < width; ++column) {
            const auto value = read(source[column]);
            finite &= std::isfinite(value);
            maximum = std::max(maximum, std::abs(value));
        }
        const auto scale = maximum == 0 ? 1.F : maximum / 127.F;
        if (!finite || scale == 0) return false;
        scales[row] = scale;
        for (std::int64_t column = 0; column < width; ++column) {
            const auto quotient = read(source[column]) / scale;
            auto rounded = std::trunc(quotient);
            if (std::abs(quotient - rounded) >= 0.5F) rounded += std::copysign(1.F, quotient);
            target[column] = static_cast<std::uint8_t>(
                static_cast<int>(std::clamp(rounded, -127.F, 127.F)));
        }
    }
    return true;
}
} // namespace

KIDI_CPU_CLONES auto pack_int8_rows(const Rows& input, std::int64_t rows, std::uint8_t* output, float* scales)
    -> bool {
    if (input.storage == Rows::Storage::BF16)
        return pack_int8_rows_impl<Rows::Storage::BF16>(static_cast<const std::uint16_t*>(input.data), rows,
                                                        input.stride, output, scales);
    if (input.storage == Rows::Storage::F16)
        return pack_int8_rows_impl<Rows::Storage::F16>(static_cast<const std::uint16_t*>(input.data), rows,
                                                       input.stride, output, scales);
    if (input.storage != Rows::Storage::F32) return false;
    return pack_int8_rows_impl<Rows::Storage::F32>(static_cast<const float*>(input.data), rows, input.stride, output,
                                                   scales);
}

auto parallel_rows(std::int64_t rows, std::int64_t width, const std::function<void(std::int64_t, std::int64_t)>& body)
    -> Result<void> {
    return for_rows(rows, width, body);
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
