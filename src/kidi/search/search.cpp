#include "kidi/search/search.h"
#include <array>
#include <bit>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <zlib.h>

#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define KIDI_HNSW_AVX512 1
#endif
#if defined(__ARM_NEON) && (defined(__aarch64__) || defined(__arm64__))
#include "ynnpack/base/arch.h"
#include <arm_neon.h>
#define KIDI_HNSW_NEON 1
#endif

namespace kidi::search {
namespace {

class StagingFile {
public:
    std::filesystem::path directory;
    auto file() const -> std::filesystem::path { return directory / "index"; }
    auto cleanup() -> Result<void> {
        if (directory.empty()) return {};
        std::error_code error;
        std::filesystem::remove(file(), error);
        if (!error) std::filesystem::remove(directory, error);
        if (error)
            return std::unexpected(Error{ErrorCode::IO, "cannot clean HNSW staging directory: " + error.message()});
        directory.clear();
        return {};
    }
    ~StagingFile() {
        const auto result = cleanup();
        if (!result) std::cerr << "kidi_hnsw_save|cleanup_error=" << result.error().message << '\n';
    }
};

template <bool bfloat>
auto weight(const std::byte* row, std::size_t column) noexcept -> float {
    if constexpr (bfloat)
        return std::bit_cast<float>(static_cast<std::uint32_t>(reinterpret_cast<const std::uint16_t*>(row)[column])
                                    << 16);
    else
        return reinterpret_cast<const float*>(row)[column];
}

template <bool bfloat>
auto dot_portable(const float* query, const std::byte* row, std::size_t width) -> float {
    std::array<float, 16> lanes{};
    std::size_t column = 0;
    for (; column + 16 <= width; column += 16)
        for (std::size_t lane = 0; lane < 16; ++lane)
            lanes[lane] = std::fma(query[column + lane], weight<bfloat>(row, column + lane), lanes[lane]);
    for (std::size_t offset = 8; offset; offset /= 2)
        for (std::size_t lane = 0; lane < offset; ++lane) lanes[lane] += lanes[lane + offset];
    float tail = 0;
    for (; column < width; ++column) tail = std::fma(query[column], weight<bfloat>(row, column), tail);
    return lanes[0] + tail;
}

template <bool bfloat>
auto dot_rows_portable(const std::byte* left, const std::byte* right, std::size_t width) -> float {
    std::array<float, 16> lanes{};
    std::size_t column = 0;
    for (; column + 16 <= width; column += 16)
        for (std::size_t lane = 0; lane < 16; ++lane)
            lanes[lane] =
                std::fma(weight<bfloat>(left, column + lane), weight<bfloat>(right, column + lane), lanes[lane]);
    for (std::size_t offset = 8; offset; offset /= 2)
        for (std::size_t lane = 0; lane < offset; ++lane) lanes[lane] += lanes[lane + offset];
    float tail = 0;
    for (; column < width; ++column) tail = std::fma(weight<bfloat>(left, column), weight<bfloat>(right, column), tail);
    return lanes[0] + tail;
}

#if defined(KIDI_HNSW_AVX512)
template <bool bfloat>
__attribute__((target("avx512f,avx512bw,fma"))) auto dot_avx512(const float* query, const std::byte* row,
                                                                std::size_t width) -> float {
    auto sum = _mm512_setzero_ps();
    std::size_t column = 0;
    for (; column + 16 <= width; column += 16) {
        __m512 values;
        if constexpr (bfloat) {
            const auto packed =
                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(row + column * sizeof(std::uint16_t)));
            values = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(packed), 16));
        } else {
            values = _mm512_loadu_ps(reinterpret_cast<const float*>(row) + column);
        }
        sum = _mm512_fmadd_ps(_mm512_loadu_ps(query + column), values, sum);
    }
    float tail = 0;
    for (; column < width; ++column) tail = std::fma(query[column], weight<bfloat>(row, column), tail);
    return _mm512_reduce_add_ps(sum) + tail;
}

template <bool bfloat>
__attribute__((target("avx512f,avx512bw,fma"))) auto dot_rows_avx512(const std::byte* left, const std::byte* right,
                                                                     std::size_t width) -> float {
    auto sum = _mm512_setzero_ps();
    std::size_t column = 0;
    for (; column + 16 <= width; column += 16) {
        __m512 first, second;
        if constexpr (bfloat) {
            const auto left_values =
                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(left + column * sizeof(std::uint16_t)));
            const auto right_values =
                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(right + column * sizeof(std::uint16_t)));
            first = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(left_values), 16));
            second = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(right_values), 16));
        } else {
            first = _mm512_loadu_ps(reinterpret_cast<const float*>(left) + column);
            second = _mm512_loadu_ps(reinterpret_cast<const float*>(right) + column);
        }
        sum = _mm512_fmadd_ps(first, second, sum);
    }
    float tail = 0;
    for (; column < width; ++column) tail = std::fma(weight<bfloat>(left, column), weight<bfloat>(right, column), tail);
    return _mm512_reduce_add_ps(sum) + tail;
}
#endif

#if defined(KIDI_HNSW_NEON)
template <bool bfloat>
auto dot_neon(const float* query, const std::byte* row, std::size_t width) -> float {
    float32x4_t s0 = vdupq_n_f32(0.F), s1 = vdupq_n_f32(0.F);
    float32x4_t s2 = vdupq_n_f32(0.F), s3 = vdupq_n_f32(0.F);
    std::size_t column = 0;
    for (; column + 16 <= width; column += 16) {
        const float32x4_t q0 = vld1q_f32(query + column);
        const float32x4_t q1 = vld1q_f32(query + column + 4);
        const float32x4_t q2 = vld1q_f32(query + column + 8);
        const float32x4_t q3 = vld1q_f32(query + column + 12);
        if constexpr (bfloat) {
            const auto* w_u16 = reinterpret_cast<const std::uint16_t*>(row) + column;
            const uint16x8_t lo = vld1q_u16(w_u16);
            const uint16x8_t hi = vld1q_u16(w_u16 + 8);
            const float32x4_t w0 = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(lo), 16));
            const float32x4_t w1 = vreinterpretq_f32_u32(vshll_high_n_u16(lo, 16));
            const float32x4_t w2 = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(hi), 16));
            const float32x4_t w3 = vreinterpretq_f32_u32(vshll_high_n_u16(hi, 16));
            s0 = vfmaq_f32(s0, q0, w0);
            s1 = vfmaq_f32(s1, q1, w1);
            s2 = vfmaq_f32(s2, q2, w2);
            s3 = vfmaq_f32(s3, q3, w3);
        } else {
            const auto* w_f32 = reinterpret_cast<const float*>(row) + column;
            s0 = vfmaq_f32(s0, q0, vld1q_f32(w_f32));
            s1 = vfmaq_f32(s1, q1, vld1q_f32(w_f32 + 4));
            s2 = vfmaq_f32(s2, q2, vld1q_f32(w_f32 + 8));
            s3 = vfmaq_f32(s3, q3, vld1q_f32(w_f32 + 12));
        }
    }
    float sum = vaddvq_f32(vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3)));
    for (; column < width; ++column) sum = std::fma(query[column], weight<bfloat>(row, column), sum);
    return sum;
}

template <bool bfloat>
auto dot_rows_neon(const std::byte* left, const std::byte* right, std::size_t width) -> float {
    float32x4_t s0 = vdupq_n_f32(0.F), s1 = vdupq_n_f32(0.F);
    float32x4_t s2 = vdupq_n_f32(0.F), s3 = vdupq_n_f32(0.F);
    std::size_t column = 0;
    for (; column + 16 <= width; column += 16) {
        if constexpr (bfloat) {
            const auto* l_u16 = reinterpret_cast<const std::uint16_t*>(left) + column;
            const auto* r_u16 = reinterpret_cast<const std::uint16_t*>(right) + column;
            const uint16x8_t l_lo = vld1q_u16(l_u16), l_hi = vld1q_u16(l_u16 + 8);
            const uint16x8_t r_lo = vld1q_u16(r_u16), r_hi = vld1q_u16(r_u16 + 8);
            s0 = vfmaq_f32(s0, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(l_lo), 16)),
                           vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(r_lo), 16)));
            s1 = vfmaq_f32(s1, vreinterpretq_f32_u32(vshll_high_n_u16(l_lo, 16)),
                           vreinterpretq_f32_u32(vshll_high_n_u16(r_lo, 16)));
            s2 = vfmaq_f32(s2, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(l_hi), 16)),
                           vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(r_hi), 16)));
            s3 = vfmaq_f32(s3, vreinterpretq_f32_u32(vshll_high_n_u16(l_hi, 16)),
                           vreinterpretq_f32_u32(vshll_high_n_u16(r_hi, 16)));
        } else {
            const auto* l_f32 = reinterpret_cast<const float*>(left) + column;
            const auto* r_f32 = reinterpret_cast<const float*>(right) + column;
            s0 = vfmaq_f32(s0, vld1q_f32(l_f32), vld1q_f32(r_f32));
            s1 = vfmaq_f32(s1, vld1q_f32(l_f32 + 4), vld1q_f32(r_f32 + 4));
            s2 = vfmaq_f32(s2, vld1q_f32(l_f32 + 8), vld1q_f32(r_f32 + 8));
            s3 = vfmaq_f32(s3, vld1q_f32(l_f32 + 12), vld1q_f32(r_f32 + 12));
        }
    }
    float sum = vaddvq_f32(vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3)));
    for (; column < width; ++column) sum = std::fma(weight<bfloat>(left, column), weight<bfloat>(right, column), sum);
    return sum;
}

__attribute__((target("bf16"))) auto dot_rows_bf16_neon(const std::byte* left, const std::byte* right,
                                                        std::size_t width) -> float {
    float32x4_t s0 = vdupq_n_f32(0.F), s1 = vdupq_n_f32(0.F);
    float32x4_t s2 = vdupq_n_f32(0.F), s3 = vdupq_n_f32(0.F);
    const auto* l_u16 = reinterpret_cast<const std::uint16_t*>(left);
    const auto* r_u16 = reinterpret_cast<const std::uint16_t*>(right);
    std::size_t column = 0;
    for (; column + 32 <= width; column += 32) {
        s0 = vbfdotq_f32(s0, vreinterpretq_bf16_u16(vld1q_u16(l_u16 + column)),
                         vreinterpretq_bf16_u16(vld1q_u16(r_u16 + column)));
        s1 = vbfdotq_f32(s1, vreinterpretq_bf16_u16(vld1q_u16(l_u16 + column + 8)),
                         vreinterpretq_bf16_u16(vld1q_u16(r_u16 + column + 8)));
        s2 = vbfdotq_f32(s2, vreinterpretq_bf16_u16(vld1q_u16(l_u16 + column + 16)),
                         vreinterpretq_bf16_u16(vld1q_u16(r_u16 + column + 16)));
        s3 = vbfdotq_f32(s3, vreinterpretq_bf16_u16(vld1q_u16(l_u16 + column + 24)),
                         vreinterpretq_bf16_u16(vld1q_u16(r_u16 + column + 24)));
    }
    for (; column + 8 <= width; column += 8) {
        s0 = vbfdotq_f32(s0, vreinterpretq_bf16_u16(vld1q_u16(l_u16 + column)),
                         vreinterpretq_bf16_u16(vld1q_u16(r_u16 + column)));
    }
    float sum = vaddvq_f32(vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3)));
    for (; column < width; ++column) sum = std::fma(weight<true>(left, column), weight<true>(right, column), sum);
    return sum;
}
#endif

auto read_meta(const std::byte* base, bool f16, std::size_t idx) noexcept -> float {
#if defined(KIDI_HNSW_NEON)
    if (f16) return static_cast<float>(reinterpret_cast<const __fp16*>(base)[idx]);
#else
    if (f16) {
        const auto h = reinterpret_cast<const std::uint16_t*>(base)[idx];
        const std::uint32_t sign = (h & 0x8000U) << 16;
        const std::uint32_t exp = (h >> 10) & 0x1FU;
        const std::uint32_t mant = h & 0x03FFU;
        if (exp == 0) return std::copysign(static_cast<float>(mant) * (1.F / 16777216.F), std::bit_cast<float>(sign));
        if (exp == 31) return std::bit_cast<float>(sign | 0x7F800000U | (mant << 13));
        return std::bit_cast<float>(sign | ((exp + 112U) << 23) | (mant << 13));
    }
#endif
    return reinterpret_cast<const float*>(base)[idx];
}

} // namespace

auto hnsw_crc32(std::span<const std::byte> bytes) -> std::uint32_t {
    return static_cast<std::uint32_t>(crc32_z(0, reinterpret_cast<const Bytef*>(bytes.data()), bytes.size()));
}

auto read_file(const std::filesystem::path& path) -> Result<std::vector<std::byte>> {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) return std::unexpected(Error{ErrorCode::IO, "cannot open HNSW index: " + path.string()});
    const auto size = stream.tellg();
    if (size < 0 || static_cast<std::uint64_t>(size) > std::numeric_limits<std::size_t>::max() ||
        static_cast<std::uint64_t>(size) > static_cast<std::uint64_t>(std::numeric_limits<std::streamsize>::max()))
        return std::unexpected(Error{ErrorCode::IO, "cannot determine HNSW index size"});
    std::vector<std::byte> bytes(static_cast<std::size_t>(size));
    stream.seekg(0);
    if (!stream.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size())))
        return std::unexpected(Error{ErrorCode::IO, "failed reading HNSW index"});
    return bytes;
}

auto save_atomic_file(const std::filesystem::path& path, std::span<const std::byte> bytes) -> Result<void> {
    if (path.empty()) return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "HNSW output path must not be empty"});
    const auto destination = std::filesystem::absolute(path);
    std::error_code error;
    const auto status = std::filesystem::symlink_status(destination, error);
    if (error && error != std::errc::no_such_file_or_directory)
        return std::unexpected(Error{ErrorCode::IO, "cannot inspect HNSW output: " + error.message()});
    if (status.type() != std::filesystem::file_type::not_found)
        return std::unexpected(Error{ErrorCode::IO, "HNSW output already exists: " + destination.string()});
    if (bytes.size() > static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max()))
        return std::unexpected(Error{ErrorCode::IO, "HNSW output exceeds stream size limits"});
    StagingFile staging;
    std::random_device random;
    for (int attempt = 0; attempt < 16 && staging.directory.empty(); ++attempt) {
        const auto directory = destination.parent_path() / (".kidi-hnsw-" + std::to_string(random()));
        error.clear();
        if (std::filesystem::create_directory(directory, error)) {
            staging.directory = directory;
        } else if (error && error != std::errc::file_exists) {
            return std::unexpected(Error{ErrorCode::IO, "cannot create HNSW staging directory: " + error.message()});
        }
    }
    if (staging.directory.empty())
        return std::unexpected(Error{ErrorCode::IO, "cannot allocate a unique HNSW staging directory"});
    std::ofstream output(staging.file(), std::ios::binary | std::ios::trunc);
    if (!output) return std::unexpected(Error{ErrorCode::IO, "cannot create staged HNSW index"});
    output.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    output.flush();
    if (!output) return std::unexpected(Error{ErrorCode::IO, "failed writing HNSW index"});
    output.close();
    if (!output) return std::unexpected(Error{ErrorCode::IO, "failed closing HNSW index"});
    std::filesystem::create_hard_link(staging.file(), destination, error);
    if (error) return std::unexpected(Error{ErrorCode::IO, "cannot publish HNSW index: " + error.message()});
    return staging.cleanup();
}

auto dot_kernels(tensor::DType dtype) -> std::pair<Dot, RowDot> {
    Dot dot = dtype == tensor::DType::BF16 ? dot_portable<true> : dot_portable<false>;
    RowDot row_dot = dtype == tensor::DType::BF16 ? dot_rows_portable<true> : dot_rows_portable<false>;
#if defined(KIDI_HNSW_AVX512)
    if (__builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") && __builtin_cpu_supports("fma")) {
        dot = dtype == tensor::DType::BF16 ? dot_avx512<true> : dot_avx512<false>;
        row_dot = dtype == tensor::DType::BF16 ? dot_rows_avx512<true> : dot_rows_avx512<false>;
    }
#elif defined(KIDI_HNSW_NEON)
    dot = dtype == tensor::DType::BF16 ? dot_neon<true> : dot_neon<false>;
    row_dot = dtype == tensor::DType::BF16
                  ? (ynn::is_arch_supported(ynn::arch_flag::neonbf16) ? dot_rows_bf16_neon : dot_rows_neon<true>)
                  : dot_rows_neon<false>;
#endif
    return {dot, row_dot};
}

auto weight_element(const std::byte* row, std::size_t column, bool bfloat) noexcept -> float {
    return bfloat ? weight<true>(row, column) : weight<false>(row, column);
}

auto prefetch_row(const std::byte* row_ptr, std::size_t row_bytes, std::int32_t bits, std::size_t label,
                  std::size_t groups, std::span<const std::byte> scale_bytes, bool scale_f16,
                  std::span<const std::byte> zero_bytes, bool zero_f16, bool has_zero) noexcept -> void {
    const std::size_t limit = std::min<std::size_t>(row_bytes, 1024);
#if defined(__GNUC__) || defined(__clang__)
    for (std::size_t off = 0; off < limit; off += 128) __builtin_prefetch(row_ptr + off, 0, 1);
    if (bits) {
        const auto s_elem = scale_f16 ? 2U : 4U;
        __builtin_prefetch(scale_bytes.data() + label * groups * s_elem, 0, 1);
        if (has_zero) {
            const auto z_elem = zero_f16 ? 2U : 4U;
            __builtin_prefetch(zero_bytes.data() + label * groups * z_elem, 0, 1);
        }
    }
#else
    (void)row_ptr;
    (void)limit;
    (void)bits;
    (void)label;
    (void)groups;
    (void)scale_bytes;
    (void)scale_f16;
    (void)zero_bytes;
    (void)zero_f16;
    (void)has_zero;
#endif
}

auto prepare_quantized_query(std::span<const float> query, std::size_t width, std::int32_t bits,
                             std::int32_t group_size, float input_scale, std::vector<std::int8_t>& q_i8,
                             std::vector<float>& q_scale, std::vector<float>& q_bsum) -> void {
    if (!bits) return;
    const auto G = static_cast<std::size_t>(group_size);
    const auto B = width / G;
    q_i8.resize(width);
    q_scale.resize(B);
    q_bsum.resize(B);
    for (std::size_t b = 0; b < B; ++b) {
        const float* x = query.data() + b * G;
        float scale = input_scale;
        float inv_scale = 0.F;
        if (scale > 0.F) {
            inv_scale = 1.F / scale;
        } else {
            float max_abs = 0.F;
            for (std::size_t i = 0; i < G; ++i) max_abs = std::max(max_abs, std::abs(x[i]));
            scale = max_abs > 0.F ? max_abs / 127.F : 0.F;
            inv_scale = max_abs > 0.F ? 127.F / max_abs : 0.F;
        }
        q_scale[b] = scale;
        const float min_q = input_scale > 0.F ? -128.F : -127.F;
        if (bits == 8) {
            auto* dst = q_i8.data() + b * G;
            for (std::size_t i = 0; i < G; ++i)
                dst[i] = static_cast<std::int8_t>(std::clamp(std::nearbyint(x[i] * inv_scale), min_q, 127.F));
        } else if (bits == 4) {
            const auto chunks = G / 32;
            std::int32_t bsum = 0;
            for (std::size_t c = 0; c < chunks; ++c) {
                const float* xc = x + c * 32;
                auto* dst = q_i8.data() + (b * chunks + c) * 32;
                for (std::size_t i = 0; i < 16; ++i) {
                    const auto qe =
                        static_cast<std::int8_t>(std::clamp(std::nearbyint(xc[2 * i] * inv_scale), min_q, 127.F));
                    const auto qo =
                        static_cast<std::int8_t>(std::clamp(std::nearbyint(xc[2 * i + 1] * inv_scale), min_q, 127.F));
                    dst[i] = qe;
                    dst[16 + i] = qo;
                    bsum += static_cast<std::int32_t>(qe) + static_cast<std::int32_t>(qo);
                }
            }
            q_bsum[b] = static_cast<float>(bsum) * scale;
        } else if (bits == 2) {
            const auto quarter = G / 4;
            auto* dst = q_i8.data() + b * G;
            for (std::size_t i = 0; i < quarter; ++i) {
                dst[0 * quarter + i] =
                    static_cast<std::int8_t>(std::clamp(std::nearbyint(x[4 * i + 0] * inv_scale), min_q, 127.F));
                dst[1 * quarter + i] =
                    static_cast<std::int8_t>(std::clamp(std::nearbyint(x[4 * i + 1] * inv_scale), min_q, 127.F));
                dst[2 * quarter + i] =
                    static_cast<std::int8_t>(std::clamp(std::nearbyint(x[4 * i + 2] * inv_scale), min_q, 127.F));
                dst[3 * quarter + i] =
                    static_cast<std::int8_t>(std::clamp(std::nearbyint(x[4 * i + 3] * inv_scale), min_q, 127.F));
            }
        }
    }
}

auto quantized_row_dot(const std::uint8_t* w_row, std::size_t label, std::size_t width, std::int32_t bits,
                       std::int32_t group_size, std::span<const std::byte> scale_bytes, bool scale_f16,
                       std::span<const std::byte> zero_bytes, bool zero_f16, bool has_zero,
                       const std::vector<std::int8_t>& q_i8, const std::vector<float>& q_scale,
                       const std::vector<float>& q_bsum) -> float {
    const auto G = static_cast<std::size_t>(group_size);
    const auto B = width / G;
    float score = 0.F;
    if (bits == 8) {
        for (std::size_t b = 0; b < B; ++b) {
            const auto* q_ptr = q_i8.data() + b * G;
            const auto* w_ptr = reinterpret_cast<const std::int8_t*>(w_row + b * G);
            std::int32_t dot_i32 = 0;
#if defined(KIDI_HNSW_NEON) && defined(__ARM_FEATURE_DOTPROD)
            int32x4_t acc0 = vdupq_n_s32(0), acc1 = vdupq_n_s32(0);
            int32x4_t acc2 = vdupq_n_s32(0), acc3 = vdupq_n_s32(0);
            std::size_t c = 0;
            for (; c + 64 <= G; c += 64) {
                acc0 = vdotq_s32(acc0, vld1q_s8(q_ptr + c), vld1q_s8(w_ptr + c));
                acc1 = vdotq_s32(acc1, vld1q_s8(q_ptr + c + 16), vld1q_s8(w_ptr + c + 16));
                acc2 = vdotq_s32(acc2, vld1q_s8(q_ptr + c + 32), vld1q_s8(w_ptr + c + 32));
                acc3 = vdotq_s32(acc3, vld1q_s8(q_ptr + c + 48), vld1q_s8(w_ptr + c + 48));
            }
            dot_i32 = vaddvq_s32(vaddq_s32(vaddq_s32(acc0, acc1), vaddq_s32(acc2, acc3)));
            for (; c < G; ++c) dot_i32 += static_cast<std::int32_t>(q_ptr[c]) * static_cast<std::int32_t>(w_ptr[c]);
#else
            for (std::size_t c = 0; c < G; ++c)
                dot_i32 += static_cast<std::int32_t>(q_ptr[c]) * static_cast<std::int32_t>(w_ptr[c]);
#endif
            const float w_s = read_meta(scale_bytes.data(), scale_f16, label * B + b);
            score += static_cast<float>(dot_i32) * (q_scale[b] * w_s);
        }
    } else if (bits == 4) {
        const auto chunks = G / 32;
#if defined(KIDI_HNSW_NEON) && defined(__ARM_FEATURE_DOTPROD)
        if (B % 4 == 0) {
            float32x4_t f_acc = vdupq_n_f32(0.F);
            const auto* s_f16 = scale_f16 ? reinterpret_cast<const __fp16*>(scale_bytes.data()) + label * B : nullptr;
            const auto* s_f32 = scale_f16 ? nullptr : reinterpret_cast<const float*>(scale_bytes.data()) + label * B;
            const auto* z_f16 =
                (has_zero && zero_f16) ? reinterpret_cast<const __fp16*>(zero_bytes.data()) + label * B : nullptr;
            const auto* z_f32 =
                (has_zero && !zero_f16) ? reinterpret_cast<const float*>(zero_bytes.data()) + label * B : nullptr;
            for (std::size_t b = 0; b < B; b += 4) {
                int32x4_t d[4]{vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0)};
                for (std::size_t r = 0; r < 4; ++r) {
                    const auto blk = b + r;
                    for (std::size_t c = 0; c < chunks; ++c) {
                        const uint8x16_t raw = vld1q_u8(w_row + (blk * chunks + c) * 16);
                        const int8x16_t we = vshrq_n_s8(vreinterpretq_s8_u8(vshlq_n_u8(raw, 4)), 4);
                        const int8x16_t wo = vshrq_n_s8(vreinterpretq_s8_u8(raw), 4);
                        const auto* qp = q_i8.data() + (blk * chunks + c) * 32;
                        d[r] = vdotq_s32(vdotq_s32(d[r], we, vld1q_s8(qp)), wo, vld1q_s8(qp + 16));
                    }
                }
                const int32x4_t dots = vpaddq_s32(vpaddq_s32(d[0], d[1]), vpaddq_s32(d[2], d[3]));
                const float32x4_t ws = s_f16 ? vcvt_f32_f16(vld1_f16(s_f16 + b)) : vld1q_f32(s_f32 + b);
                const float32x4_t qs = vld1q_f32(q_scale.data() + b);
                f_acc = vfmaq_f32(f_acc, vcvtq_f32_s32(dots), vmulq_f32(ws, qs));
                if (has_zero) {
                    const float32x4_t wz = z_f16 ? vcvt_f32_f16(vld1_f16(z_f16 + b)) : vld1q_f32(z_f32 + b);
                    const float32x4_t qz = vld1q_f32(q_bsum.data() + b);
                    f_acc = vfmaq_f32(f_acc, wz, qz);
                }
            }
            score = vaddvq_f32(f_acc);
        } else
#endif
        {
            for (std::size_t b = 0; b < B; ++b) {
                std::int32_t dot_i32 = 0;
#if defined(KIDI_HNSW_NEON) && defined(__ARM_FEATURE_DOTPROD)
                int32x4_t acc = vdupq_n_s32(0);
                for (std::size_t c = 0; c < chunks; ++c) {
                    const uint8x16_t raw = vld1q_u8(w_row + (b * chunks + c) * 16);
                    const int8x16_t we = vshrq_n_s8(vreinterpretq_s8_u8(vshlq_n_u8(raw, 4)), 4);
                    const int8x16_t wo = vshrq_n_s8(vreinterpretq_s8_u8(raw), 4);
                    const auto* qp = q_i8.data() + (b * chunks + c) * 32;
                    acc = vdotq_s32(vdotq_s32(acc, we, vld1q_s8(qp)), wo, vld1q_s8(qp + 16));
                }
                dot_i32 = vaddvq_s32(acc);
#else
                for (std::size_t c = 0; c < chunks; ++c) {
                    const auto* wp = w_row + (b * chunks + c) * 16;
                    const auto* qp = q_i8.data() + (b * chunks + c) * 32;
                    for (std::size_t i = 0; i < 16; ++i) {
                        const auto byte = wp[i];
                        const auto we = static_cast<std::int8_t>(static_cast<std::uint8_t>(byte << 4)) >> 4;
                        const auto wo = static_cast<std::int8_t>(byte) >> 4;
                        dot_i32 += static_cast<std::int32_t>(we) * qp[i] + static_cast<std::int32_t>(wo) * qp[16 + i];
                    }
                }
#endif
                const float w_s = read_meta(scale_bytes.data(), scale_f16, label * B + b);
                score += static_cast<float>(dot_i32) * (q_scale[b] * w_s);
                if (has_zero) {
                    const float w_z = read_meta(zero_bytes.data(), zero_f16, label * B + b);
                    score += q_bsum[b] * w_z;
                }
            }
        }
    } else if (bits == 2) {
        const auto quarter = G / 4;
        for (std::size_t b = 0; b < B; ++b) {
            const auto* wp = w_row + b * quarter;
            const auto* qp = q_i8.data() + b * G;
            std::int32_t dot_i32 = 0;
#if defined(KIDI_HNSW_NEON) && defined(__ARM_FEATURE_DOTPROD)
            int32x4_t acc = vdupq_n_s32(0);
            std::size_t c = 0;
            for (; c + 16 <= quarter; c += 16) {
                const int8x16_t raw = vreinterpretq_s8_u8(vld1q_u8(wp + c));
                const int8x16_t w0 = vshrq_n_s8(vshlq_n_s8(raw, 6), 6);
                const int8x16_t w1 = vshrq_n_s8(vshlq_n_s8(raw, 4), 6);
                const int8x16_t w2 = vshrq_n_s8(vshlq_n_s8(raw, 2), 6);
                const int8x16_t w3 = vshrq_n_s8(raw, 6);
                acc = vdotq_s32(acc, w0, vld1q_s8(qp + 0 * quarter + c));
                acc = vdotq_s32(acc, w1, vld1q_s8(qp + 1 * quarter + c));
                acc = vdotq_s32(acc, w2, vld1q_s8(qp + 2 * quarter + c));
                acc = vdotq_s32(acc, w3, vld1q_s8(qp + 3 * quarter + c));
            }
            dot_i32 = vaddvq_s32(acc);
            for (; c < quarter; ++c) {
                const auto raw = static_cast<std::int8_t>(wp[c]);
                dot_i32 += static_cast<std::int32_t>(static_cast<std::int8_t>(raw << 6) >> 6) * qp[0 * quarter + c] +
                           static_cast<std::int32_t>(static_cast<std::int8_t>(raw << 4) >> 6) * qp[1 * quarter + c] +
                           static_cast<std::int32_t>(static_cast<std::int8_t>(raw << 2) >> 6) * qp[2 * quarter + c] +
                           static_cast<std::int32_t>(raw >> 6) * qp[3 * quarter + c];
            }
#else
            for (std::size_t c = 0; c < quarter; ++c) {
                const auto raw = static_cast<std::int8_t>(wp[c]);
                dot_i32 += static_cast<std::int32_t>(static_cast<std::int8_t>(raw << 6) >> 6) * qp[0 * quarter + c] +
                           static_cast<std::int32_t>(static_cast<std::int8_t>(raw << 4) >> 6) * qp[1 * quarter + c] +
                           static_cast<std::int32_t>(static_cast<std::int8_t>(raw << 2) >> 6) * qp[2 * quarter + c] +
                           static_cast<std::int32_t>(raw >> 6) * qp[3 * quarter + c];
            }
#endif
            const float w_s = read_meta(scale_bytes.data(), scale_f16, label * B + b);
            score += static_cast<float>(dot_i32) * (q_scale[b] * w_s);
        }
    }
    return score;
}

} // namespace kidi::search
