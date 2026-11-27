#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <span>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <ynnpack.h>
#include "ynnpack/composites/composites.h"

#include "kidi/core/precision.h"
#include "kidi/ops/context.h"
#include "kidi/runtime/ynn/feed_forward_cache.h"
#include "kidi/runtime/ynn/graph.h"
#include "kidi/tensor/tensor.h"

namespace {

#if !defined(_WIN32) && !defined(__EMSCRIPTEN__)
constexpr std::int32_t ROWS = 32, INPUT_SIZE = 128, INTERMEDIATE = 2056, BITS = 4;

class YnnGraphTest : public ::testing::Test {
protected:
    void SetUp() override {
        const auto* info = ::testing::UnitTest::GetInstance()->current_test_info();
        cache_root = std::filesystem::temp_directory_path() /
                     (std::string("kidi-ynn-feed-forward-cache-test-") + info->name());
        std::filesystem::remove_all(cache_root);
        std::filesystem::create_directories(cache_root);
        source = cache_root / "weights.bin";
        std::ofstream(source, std::ios::binary).write("source", 6);
    }

    void TearDown() override {
        std::error_code ignored;
        std::filesystem::permissions(cache_root, std::filesystem::perms::owner_all, ignored);
        std::filesystem::remove_all(cache_root, ignored);
    }

    std::filesystem::path cache_root;
    std::filesystem::path source;
};

auto gate_up_scale_tensor() -> kidi::tensor::Tensor {
    std::vector<float> gate_up_scales(2 * INTERMEDIATE);
    for (std::size_t index = 0; index < gate_up_scales.size(); ++index)
        gate_up_scales[index] = 0.00390625F * static_cast<float>(1 + index % 5);
    return kidi::ops::require(
        kidi::tensor::Tensor::from_host({2 * INTERMEDIATE, 1}, std::span<const float>(gate_up_scales)));
}

auto down_scale_tensor() -> kidi::tensor::Tensor {
    std::vector<float> down_scales(INPUT_SIZE);
    for (std::size_t index = 0; index < down_scales.size(); ++index)
        down_scales[index] = 0.00390625F * static_cast<float>(1 + index % 7);
    return kidi::ops::require(kidi::tensor::Tensor::from_host({INPUT_SIZE, 1}, std::span<const float>(down_scales)));
}

TEST_F(YnnGraphTest, FeedForwardCacheValidation) {
    kidi::runtime::ynn::FeedForwardCache cache;
    cache.configure(cache_root / "cache", source);
    kidi::runtime::ynn::FeedForwardCacheSpec cache_spec{
        .bits = 4, .input_size = 8, .intermediate_size = 16, .source_hash = 17};
    const std::array<std::uint8_t, 6> cached_bytes{1, 2, 3, 5, 8, 13};
    cache.store(cache_spec, cached_bytes);
    const auto cached = cache.load(cache_spec, cached_bytes.size());
    ASSERT_TRUE(cached) << "stored entry loads";
    EXPECT_TRUE(cached.owner) << "loaded entry has an owner";
    EXPECT_TRUE(std::ranges::equal(cached.bytes, cached_bytes)) << "loaded bytes match stored bytes";
    auto incompatible = cache_spec;
    incompatible.bits = 8;
    EXPECT_FALSE(cache.load(incompatible, cached_bytes.size())) << "incompatible spec misses";
    const auto source_time = std::filesystem::last_write_time(source);
    std::ofstream(source, std::ios::binary | std::ios::trunc).write("change", 6);
    std::filesystem::last_write_time(source, source_time);
    auto replaced = cache_spec;
    replaced.source_hash = 18;
    kidi::runtime::ynn::FeedForwardCache replacement;
    replacement.configure(cache_root / "cache", source);
    EXPECT_FALSE(replacement.load(replaced, cached_bytes.size())) << "replaced source hash misses";
    std::ofstream(source, std::ios::binary | std::ios::app).write("changed", 7);
    kidi::runtime::ynn::FeedForwardCache stale;
    stale.configure(cache_root / "cache", source);
    EXPECT_FALSE(stale.load(cache_spec, cached_bytes.size())) << "stale source misses";
    const auto read_only_parent = cache_root / "read-only";
    std::filesystem::create_directories(read_only_parent);
    std::filesystem::permissions(read_only_parent,
                                 std::filesystem::perms::owner_read | std::filesystem::perms::owner_exec);
    kidi::runtime::ynn::FeedForwardCache uncreatable;
    uncreatable.configure(read_only_parent / "nested", source);
    std::filesystem::permissions(read_only_parent, std::filesystem::perms::owner_all);
    EXPECT_FALSE(uncreatable.enabled()) << "uncreatable cache dir disables the cache";
    EXPECT_FALSE(uncreatable.writable()) << "uncreatable cache dir is not writable";
}

TEST_F(YnnGraphTest, CachedFeedForward) {
    if (!ynn::is_symmetric_linear_supported(BITS))
        GTEST_SKIP() << "symmetric linear unsupported for bits=" << BITS;
    std::vector<float> activations(ROWS * INPUT_SIZE);
    for (std::size_t index = 0; index < activations.size(); ++index)
        activations[index] = static_cast<float>(static_cast<int>(index % 17) - 8) * 0.0625F;
    std::vector<std::uint8_t> gate_up_bytes(2 * INTERMEDIATE * (INPUT_SIZE / 2));
    for (std::size_t index = 0; index < gate_up_bytes.size(); ++index)
        gate_up_bytes[index] = static_cast<std::uint8_t>((index * 13 + 5) & 0xFFU);
    std::vector<std::uint8_t> down_bytes(INPUT_SIZE * (INTERMEDIATE / 2));
    for (std::size_t index = 0; index < down_bytes.size(); ++index)
        down_bytes[index] = static_cast<std::uint8_t>((index * 7 + 3) & 0xFFU);
    const auto input_tensor =
        kidi::ops::require(kidi::tensor::Tensor::from_host({1, ROWS, INPUT_SIZE}, std::span<const float>(activations)));
    const auto gate_up_tensor = kidi::ops::require(kidi::tensor::Tensor::from_host(
        {2 * INTERMEDIATE, INPUT_SIZE / 2}, std::span<const std::uint8_t>(gate_up_bytes)));
    const auto down_tensor = kidi::ops::require(
        kidi::tensor::Tensor::from_host({INPUT_SIZE, INTERMEDIATE / 2}, std::span<const std::uint8_t>(down_bytes)));
    const auto gate_up_scales = gate_up_scale_tensor();
    const auto down_scales = down_scale_tensor();
    const auto run_ffn = [&](kidi::ops::Context& context) {
        return context.gated_feed_forward(input_tensor, gate_up_tensor, gate_up_scales, down_tensor, down_scales, BITS,
                                          INPUT_SIZE, INTERMEDIATE, 0.125F, 0.0625F, 0.03125F, 0.125F);
    };
    kidi::ops::Context reference_context(kidi::tensor::Device::cpu());
    const auto reference_ffn = run_ffn(reference_context);
    const auto ffn_cache_dir = cache_root / "ffn-multi-tile";
    kidi::ops::Context populate_context(kidi::tensor::Device::cpu());
    populate_context.configure_cache(ffn_cache_dir, source);
    const auto populated_ffn = run_ffn(populate_context);
    ASSERT_FALSE(std::filesystem::is_empty(ffn_cache_dir)) << "populating run writes the cache";
    std::filesystem::permissions(ffn_cache_dir,
                                 std::filesystem::perms::owner_read | std::filesystem::perms::owner_exec);
    kidi::ops::Context cached_context(kidi::tensor::Device::cpu());
    cached_context.configure_cache(ffn_cache_dir, source);
    const auto cached_ffn = run_ffn(cached_context);
    std::filesystem::permissions(ffn_cache_dir, std::filesystem::perms::owner_all);
    const auto expected_values = kidi::ops::require(reference_ffn.data<float>());
    const auto populated_values = kidi::ops::require(populated_ffn.data<float>());
    const auto cached_values = kidi::ops::require(cached_ffn.data<float>());
    EXPECT_TRUE(std::ranges::equal(expected_values, populated_values)) << "populating run matches reference";
    EXPECT_TRUE(std::ranges::equal(expected_values, cached_values)) << "cached run matches reference";
}

// Verify fused GATED_FEED_FORWARD matches composed PACKED_LINEAR + GELU/SILU_MULTIPLY + PACKED_LINEAR across
// bits = 2, 4, 8 and both M = 1 (decode) and M = 17 (odd multi-row prefill) with 4 worker threads.
TEST(YnnFusedTest, FeedForward) {
    const auto gate_up_scales = gate_up_scale_tensor();
    const auto down_scales = down_scale_tensor();
    kidi::runtime::ynn::set_thread_count(4);
    for (const std::int32_t bits : {2, 4, 8}) {
        const std::size_t vals_per_byte = static_cast<std::size_t>(8 / bits);
        for (const std::int32_t m_rows : {1, 17}) {
            std::vector<float> x_vals(m_rows * INPUT_SIZE);
            for (std::size_t i = 0; i < x_vals.size(); ++i)
                x_vals[i] = static_cast<float>(static_cast<int>((i * 5) % 19) - 9) * 0.0625F;
            std::vector<std::uint8_t> gu_b(2 * INTERMEDIATE * (INPUT_SIZE / vals_per_byte));
            for (std::size_t i = 0; i < gu_b.size(); ++i) gu_b[i] = static_cast<std::uint8_t>((i * 11 + 7) & 0xFFU);
            std::vector<std::uint8_t> dn_b(INPUT_SIZE * (INTERMEDIATE / vals_per_byte));
            for (std::size_t i = 0; i < dn_b.size(); ++i) dn_b[i] = static_cast<std::uint8_t>((i * 17 + 3) & 0xFFU);
            const auto x_t = kidi::ops::require(
                kidi::tensor::Tensor::from_host({1, m_rows, INPUT_SIZE}, std::span<const float>(x_vals)));
            const auto gu_t = kidi::ops::require(kidi::tensor::Tensor::from_host(
                {2 * INTERMEDIATE, static_cast<std::int64_t>(INPUT_SIZE / vals_per_byte)},
                std::span<const std::uint8_t>(gu_b)));
            const auto dn_t = kidi::ops::require(
                kidi::tensor::Tensor::from_host({INPUT_SIZE, static_cast<std::int64_t>(INTERMEDIATE / vals_per_byte)},
                                                std::span<const std::uint8_t>(dn_b)));

            for (const bool silu : {false, true}) {
                kidi::ops::Context ctx(kidi::tensor::Device::cpu());
                const auto fused =
                    ctx.gated_feed_forward(x_t, gu_t, gate_up_scales, dn_t, down_scales, bits, INPUT_SIZE,
                                           INTERMEDIATE, 0.125F, 0.0625F, 0.03125F, 0.125F, silu);
                const auto proj = ctx.packed_linear(x_t, gu_t, gate_up_scales, bits, INPUT_SIZE, 0.125F, 0.0625F);
                const auto g_slice = ctx.slice(proj, -1, 0, INTERMEDIATE);
                const auto u_slice = ctx.slice(proj, -1, INTERMEDIATE, INTERMEDIATE);
                const auto hid = silu ? ctx.silu_multiply(g_slice, u_slice) : ctx.gelu_multiply(g_slice, u_slice);
                const auto composed = ctx.packed_linear(hid, dn_t, down_scales, bits, INTERMEDIATE, 0.03125F, 0.125F);
                const auto f_span = kidi::ops::require(fused.data<float>());
                const auto c_span = kidi::ops::require(composed.data<float>());
                for (std::size_t i = 0; i < f_span.size(); ++i) {
                    ASSERT_NEAR(f_span[i], c_span[i], 1e-5F)
                        << "FFN mismatch bits=" << bits << " rows=" << m_rows << " silu=" << silu << " idx=" << i;
                }
            }
            // Cross-check calibrated SymmetricPackedLinear (bits = 2, 4, 8) against the QAT-FP32 dequantized reference.
            kidi::ops::Context fast_ctx(kidi::tensor::Device::cpu());
            kidi::ops::Context parity_ctx(kidi::tensor::Device::cpu());
            parity_ctx.set_precision(kidi::core::InferencePrecision::QAT_FP32);
            const auto fast_out = fast_ctx.packed_linear(x_t, gu_t, gate_up_scales, bits, INPUT_SIZE, 0.125F, 0.0625F);
            const auto ref_out = parity_ctx.packed_linear(x_t, gu_t, gate_up_scales, bits, INPUT_SIZE, 0.125F, 0.0625F);
            const auto fast_span = kidi::ops::require(fast_out.data<float>());
            const auto ref_span = kidi::ops::require(ref_out.data<float>());
            for (std::size_t i = 0; i < fast_span.size(); ++i) {
                ASSERT_NEAR(fast_span[i], ref_span[i], 1e-5F)
                    << "Calibrated packed_linear parity mismatch bits=" << bits << " rows=" << m_rows << " idx=" << i;
            }
        }
    }
}
#endif

#if defined(__APPLE__) && (defined(__aarch64__) || defined(__arm64__))
TEST(YnnGraphArchTest, AppleArm64Kernels) {
    EXPECT_NE(kidi::runtime::ynn::supported_arch_flags(), 0)
        << "Apple arm64 YNNPACK feature detection returned no supported kernels";
}
#endif

TEST(YnnThreadTest, ParallelFor) {
    kidi::runtime::ynn::set_thread_count(2);
    ASSERT_EQ(kidi::runtime::ynn::thread_count(), 2);
    std::array<int, 16> parallel_hits{};
    ASSERT_TRUE(
        kidi::runtime::ynn::parallel_for(parallel_hits.size(), [&](std::size_t idx) { parallel_hits[idx] = 1; }));
    for (int v : parallel_hits) EXPECT_EQ(v, 1);
}

} // namespace
