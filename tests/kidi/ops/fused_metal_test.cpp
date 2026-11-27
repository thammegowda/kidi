#include "kidi/model/inkling2.h"
#include "kidi/model/registry.h"
#include "kidi/ops/context.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <random>
#include <vector>

#include <gtest/gtest.h>

namespace {
using namespace kidi;
using ops::require;
using tensor::Device;
using tensor::DType;
using tensor::Tensor;

auto random_values(std::size_t count, std::mt19937& random, float scale = 1.F) -> std::vector<float> {
    std::normal_distribution<float> normal(0.F, scale);
    std::vector<float> values(count);
    for (auto& value : values) value = normal(random);
    return values;
}

auto host(std::vector<std::int64_t> shape, std::span<const float> values) -> Tensor {
    return require(Tensor::from_host(std::move(shape), values));
}

class FusedMetalTest : public ::testing::Test {
protected:
    ops::Context cpu{Device::cpu(), false, false, DType::F32};
    ops::Context gpu{Device::apple_gpu(), true, false, DType::F32};
    std::mt19937 random{7};
    auto floats(const Tensor& tensor) -> std::vector<float> {
        gpu.synchronize();
        auto values = require(tensor.to(Device::cpu()));
        if (values.dtype() != DType::F32) values = cpu.cast(values, DType::F32);
        const auto data = require(values.data<float>());
        return {data.begin(), data.end()};
    }
    void expect_close(const Tensor& actual, const Tensor& expected, float tolerance) {
        ASSERT_TRUE(std::ranges::equal(actual.shape(), expected.shape()));
        const auto a = floats(actual), e = floats(expected);
        for (std::size_t index = 0; index < a.size(); ++index)
            ASSERT_NEAR(a[index], e[index], tolerance * (1.F + std::abs(e[index]))) << "at " << index;
    }
    auto pair(std::vector<std::int64_t> shape, float scale = 1.F) -> std::pair<Tensor, Tensor> {
        std::size_t count = 1;
        for (const auto size : shape) count *= static_cast<std::size_t>(size);
        const auto values = random_values(count, random, scale);
        auto tensor = host(std::move(shape), values);
        return {tensor, require(tensor.to(Device::apple_gpu()))};
    }
};

TEST_F(FusedMetalTest, RejectsMismatchedOperandGeometry) {
    const auto [source, source_gpu] = pair({2, 32});
    const auto [other, other_gpu] = pair({1, 32});
    const auto [queries, queries_gpu] = pair({1, 32});
    const std::array sources{source_gpu, other_gpu};
    EXPECT_THROW(gpu.residual_mix(sources, queries_gpu, 1e-6F), ops::Failure);
}

TEST_F(FusedMetalTest, RoutingMixClipAndBlendMatchCpu) {
    constexpr std::int64_t TOKENS = 5, EXPERTS = 40, TOP = 8;
    const auto [logits, logits_gpu] = pair({TOKENS, EXPERTS});
    const auto [bias, bias_gpu] = pair({EXPERTS}, 0.1F);
    const auto routes = floats(gpu.route_experts(logits_gpu, bias_gpu, TOP, 2.5F));
    const auto logit_values = floats(logits), bias_values = floats(bias);
    for (std::int64_t token = 0; token < TOKENS; ++token) {
        std::vector<float> scores(EXPERTS), biased(EXPERTS);
        for (std::int64_t expert = 0; expert < EXPERTS; ++expert) {
            scores[expert] = 1.F / (1.F + std::exp(-logit_values[token * EXPERTS + expert]));
            biased[expert] = scores[expert] + bias_values[expert];
        }
        std::vector<std::int64_t> chosen;
        float total = 0;
        for (std::int64_t slot = 0; slot < TOP; ++slot) {
            const auto best = std::ranges::max_element(biased) - biased.begin();
            chosen.push_back(best);
            total += scores[best];
            biased[best] = -INFINITY;
        }
        for (std::int64_t slot = 0; slot < TOP; ++slot) {
            const auto index = (token * TOP + slot) * 2;
            EXPECT_EQ(routes[index], static_cast<float>(chosen[slot]));
            EXPECT_NEAR(routes[index + 1], scores[chosen[slot]] / total * 2.5F, 1e-5F);
        }
    }

    std::vector<Tensor> sources, sources_gpu;
    for (int source = 0; source < 5; ++source) {
        auto [values, values_gpu] = pair({3, 96});
        sources.push_back(values);
        sources_gpu.push_back(values_gpu);
    }
    const auto [queries, queries_gpu] = pair({3, 96});
    expect_close(gpu.residual_mix(sources_gpu, queries_gpu, 1e-6F), cpu.residual_mix(sources, queries, 1e-6F), 1e-5F);

    const auto [rows, rows_gpu] = pair({3, 2, 64}, 3.F);
    const auto [scale, scale_gpu] = pair({2, 64});
    expect_close(gpu.l2_clip(rows_gpu, {}, 4.F), cpu.l2_clip(rows, {}, 4.F), 1e-5F);
    expect_close(gpu.l2_clip(rows_gpu, scale_gpu, 4.F), cpu.l2_clip(rows, scale, 4.F), 1e-5F);

    const auto [gate_up, gate_up_gpu] = pair({4, 48}, 4.F);
    const auto [row_weights, row_weights_gpu] = pair({4});
    expect_close(gpu.silu_gate(gate_up_gpu, 4.F, 25.F, row_weights_gpu), cpu.silu_gate(gate_up, 4.F, 25.F, row_weights),
                 1e-5F);
    expect_close(gpu.silu_gate(gate_up_gpu, 4.F, 25.F), cpu.silu_gate(gate_up, 4.F, 25.F), 1e-5F);

    // The CPU blend rounds to BF16; Metal keeps F32.
    const auto [destination, destination_gpu] = pair({3, 80});
    const auto [message, message_gpu] = pair({3, 80});
    const auto [gain_forget, gain_forget_gpu] = pair({3, 160});
    const auto [norm, norm_gpu] = pair({80});
    expect_close(gpu.blend_prologue(destination_gpu, message_gpu, 1e-6F),
                 cpu.blend_prologue(destination, message, 1e-6F), 1e-2F);
    expect_close(gpu.blend_epilogue(destination_gpu, message_gpu, gain_forget_gpu, norm_gpu, 1e-6F),
                 cpu.blend_epilogue(destination, message, gain_forget, norm, 1e-6F), 1e-2F);
}

TEST_F(FusedMetalTest, ShortConvolutionMatchesCpu) {
    constexpr std::int64_t WIDTH = 4, CHANNELS = 32, TOKENS = 5, RING = 6;
    const auto [weight, weight_gpu] = pair({WIDTH, CHANNELS});
    const auto [padded, padded_gpu] = pair({TOKENS + WIDTH - 1, CHANNELS});
    const auto [taps, taps_gpu] = pair({TOKENS, 2 * WIDTH});
    expect_close(gpu.short_convolution(padded_gpu, weight_gpu, {}, 0.5F),
                 cpu.short_convolution(padded, weight, {}, 0.5F), 1e-5F);
    expect_close(gpu.short_convolution(padded_gpu, weight_gpu, taps_gpu, 0.5F),
                 cpu.short_convolution(padded, weight, taps, 0.5F), 1e-5F);

    const auto [input, input_gpu] = pair({TOKENS, CHANNELS});
    const auto [state, state_gpu] = pair({RING, CHANNELS});
    const std::array<std::int32_t, 1> slot_value{4};
    const auto slot = require(Tensor::from_host({1}, std::span<const std::int32_t>(slot_value)));
    const auto slot_gpu = require(slot.to(Device::apple_gpu()));
    expect_close(gpu.short_convolution(input_gpu, state_gpu, weight_gpu, taps_gpu, slot_gpu, 0.5F),
                 cpu.short_convolution(input, state, weight, taps, slot, 0.5F), 1e-5F);
    expect_close(gpu.short_convolution(input_gpu, state_gpu, weight_gpu, taps_gpu, slot_gpu, 0.5F, input_gpu),
                 cpu.short_convolution(input, state, weight, taps, slot, 0.5F, input), 1e-5F);

    const auto row = require(input.narrow(0, 0, 1));
    const auto row_gpu = require(input_gpu.narrow(0, 0, 1));
    auto cpu_state = host({RING, CHANNELS}, floats(state));
    expect_close(gpu.short_convolution(row_gpu, state_gpu, weight_gpu, {}, slot_gpu, 0.5F, row_gpu, {}, true),
                 cpu.short_convolution(row, cpu_state, weight, {}, slot, 0.5F, row, {}, true), 1e-5F);
    gpu.synchronize();
    expect_close(state_gpu, cpu_state, 0.F);
}

TEST_F(FusedMetalTest, RelativeAttentionMatchesCpu) {
    constexpr std::int64_t TOKENS = 3, HEADS = 4, KEY_HEADS = 2, WIDTH = 64, EXTENT = 5, CAPACITY = 40;
    const auto [query, query_gpu] = pair({TOKENS, HEADS * WIDTH});
    const auto [key, key_gpu] = pair({CAPACITY, KEY_HEADS * WIDTH});
    const auto [value, value_gpu] = pair({CAPACITY, KEY_HEADS * WIDTH});
    const auto [relative, relative_gpu] = pair({TOKENS, HEADS, EXTENT});
    const auto [sink, sink_gpu] = pair({TOKENS, HEADS});
    const auto [gate, gate_gpu] = pair({TOKENS, HEADS * WIDTH});
    const std::array<std::int32_t, TOKENS> position_values{3, 21, 39};
    const auto positions = require(Tensor::from_host({TOKENS}, std::span<const std::int32_t>(position_values)));
    const auto positions_gpu = require(positions.to(Device::apple_gpu()));
    for (const auto& shape : {ops::Context::RelativeAttention{HEADS, KEY_HEADS, 30, 0, EXTENT},
                              ops::Context::RelativeAttention{HEADS, KEY_HEADS, 30, 8, EXTENT, 0.3F, 4.F}}) {
        expect_close(gpu.relative_attention(query_gpu, key_gpu, value_gpu, relative_gpu, sink_gpu, gate_gpu, shape),
                     cpu.relative_attention(query, key, value, relative, sink, gate, shape), 1e-4F);
        expect_close(gpu.relative_attention(query_gpu, key_gpu, value_gpu, relative_gpu, sink_gpu, gate_gpu, shape,
                                            positions_gpu),
                     cpu.relative_attention(query, key, value, relative, sink, gate, shape, positions), 1e-4F);
        const auto [query128, query128_gpu] = pair({37, HEADS * 128});
        const auto [key128, key128_gpu] = pair({64, KEY_HEADS * 128});
        const auto [value128, value128_gpu] = pair({64, KEY_HEADS * 128});
        const auto [rel128, rel128_gpu] = pair({37, HEADS, EXTENT});
        const auto [sink128, sink128_gpu] = pair({37, HEADS});
        const auto [gate128, gate128_gpu] = pair({37, HEADS * 128});
        const ops::Context::RelativeAttention flash_shape{HEADS, KEY_HEADS, 12, 16, EXTENT, 0.3F, 4.F};
        expect_close(gpu.relative_attention(query128_gpu, key128_gpu, value128_gpu, rel128_gpu, sink128_gpu,
                                            gate128_gpu, flash_shape),
                     cpu.relative_attention(query128, key128, value128, rel128, sink128, gate128, flash_shape), 2e-3F);
    }
}

TEST_F(FusedMetalTest, NativeSliceConcatAndCastMatchCpu) {
    const auto [a, a_gpu] = pair({7, 13});
    const auto [b, b_gpu] = pair({7, 19});
    const auto [c, c_gpu] = pair({7, 5});
    const std::array cpu_parts{a, b, c};
    const std::array gpu_parts{a_gpu, b_gpu, c_gpu};
    const auto joined_cpu = cpu.concat(cpu_parts, 1), joined_gpu = gpu.concat(gpu_parts, 1);
    expect_close(joined_gpu, joined_cpu, 0.F);
    expect_close(gpu.slice(joined_gpu, 1, 13, 19), b, 0.F);
    expect_close(gpu.cast(gpu.cast(a_gpu, DType::BF16), DType::F32), cpu.cast(cpu.cast(a, DType::BF16), DType::F32),
                 0.F);
    expect_close(gpu.cast(gpu.cast(a_gpu, DType::F16), DType::F32), cpu.cast(cpu.cast(a, DType::F16), DType::F32), 0.F);
}

TEST_F(FusedMetalTest, VqExpertsMatchCpu) {
    constexpr std::int64_t EXPERTS = 4, TOP = 2, WIDTH = 192, HIDDEN = 128;
    constexpr float GATE_CAP = 4.F, UP_CAP = 25.F;
    for (const auto [gate_fmt, down_fmt] : {std::pair{ops::BlockScaledFormat::VQ3D, ops::BlockScaledFormat::VQ3D},
                                            std::pair{ops::BlockScaledFormat::VQ4D, ops::BlockScaledFormat::VQ3D}}) {
        const auto gate_w =
            host({EXPERTS, 2 * HIDDEN, WIDTH}, random_values(EXPERTS * 2 * HIDDEN * WIDTH, random, 0.3F));
        const auto down_w = host({EXPERTS, WIDTH, HIDDEN}, random_values(EXPERTS * WIDTH * HIDDEN, random, 0.3F));
        const auto gate_cpu = require(ops::pack_vq_weight(gate_w, gate_fmt));
        const auto down_cpu = require(ops::pack_vq_weight(down_w, down_fmt));
        const ops::BlockScaledWeight gate_gpu{gate_fmt, require(gate_cpu.values.to(Device::apple_gpu())),
                                              require(gate_cpu.scales.to(Device::apple_gpu())),
                                              require(gate_cpu.global.to(Device::apple_gpu())), gate_cpu.group_size};
        const ops::BlockScaledWeight down_gpu{down_fmt, require(down_cpu.values.to(Device::apple_gpu())),
                                              require(down_cpu.scales.to(Device::apple_gpu())),
                                              require(down_cpu.global.to(Device::apple_gpu())), down_cpu.group_size};
        for (const std::int64_t tokens : {1, 5, 37}) {
            SCOPED_TRACE(tokens);
            std::vector<float> routes;
            for (std::int64_t token = 0; token < tokens; ++token) {
                const float other = 1.F + static_cast<float>(token % 2);
                const float weight = 0.25F + 0.05F * static_cast<float>(token % 5);
                if (token % 2 == 0)
                    routes.insert(routes.end(), {other, weight, 0.F, 1.F - weight});
                else
                    routes.insert(routes.end(), {0.F, 1.F - weight, other, weight});
            }
            const auto routes_cpu = host({tokens, TOP, 2}, routes);
            const auto routes_gpu = require(routes_cpu.to(Device::apple_gpu()));
            const auto [x_cpu, x_gpu] = pair({tokens, WIDTH});
            const auto h_cpu = cpu.expert_silu(x_cpu, gate_cpu, routes_cpu, GATE_CAP, UP_CAP);
            const auto h_gpu = gpu.expert_silu(x_gpu, gate_gpu, routes_gpu, GATE_CAP, UP_CAP);
            const auto y_cpu = cpu.expert_linear(h_cpu, down_cpu, routes_cpu);
            const auto y_gpu = gpu.expert_linear(h_gpu, down_gpu, routes_gpu);
            expect_close(h_gpu, h_cpu, 5e-4F);
            expect_close(y_gpu, y_cpu, 5e-4F);
        }
    }
}

TEST_F(FusedMetalTest, VqExpertPrefillMatchesSingleTokenArithmetic) {
    constexpr std::int64_t TOKENS = 17, TOP = 2, EXPERTS = 3, WIDTH = 1088;
    const auto device = Device::apple_gpu();
    ops::Context exact(device, false, false, DType::F32);
    std::vector<float> routes;
    for (std::int64_t t = 0; t < TOKENS; ++t) {
        const float second = t == TOKENS - 1 ? 2.F : 1.F;
        routes.insert(routes.end(), {0.F, 0.25F, second, 0.75F});
    }
    const auto routed = require(host({TOKENS, TOP, 2}, routes).to(device));
    const auto gate_input = pair({TOKENS, WIDTH}).second;
    const auto down_input = pair({TOKENS, TOP, WIDTH}).second;
    for (const auto format : {ops::BlockScaledFormat::VQ3D, ops::BlockScaledFormat::VQ4D}) {
        const auto bank = [&](std::int64_t rows) {
            auto weights = require(ops::pack_vq_weight(pair({EXPERTS, rows, WIDTH}, 0.3F).first, format));
            weights.values = require(weights.values.to(device));
            weights.scales = require(weights.scales.to(device));
            weights.global = require(weights.global.to(device));
            return weights;
        };
        const auto gate = bank(14), down = bank(9);
        const auto gate_batch = exact.expert_silu(gate_input, gate, routed, 4.F, 25.F);
        const auto down_batch = exact.expert_linear(down_input, down, routed);
        for (const auto token : {std::int64_t{0}, TOKENS - 1}) {
            const auto route = require(routed.narrow(0, token, 1));
            const auto gate_one = exact.expert_silu(require(gate_input.narrow(0, token, 1)), gate, route, 4.F, 25.F);
            const auto down_one = exact.expert_linear(require(down_input.narrow(0, token, 1)), down, route);
            exact.synchronize();
            expect_close(require(gate_batch.narrow(0, token, 1)), gate_one, 0.F);
            expect_close(require(down_batch.narrow(0, token, 1)), down_one, 0.F);
        }
    }
}

TEST(Inkling2MetalTest, CacheGrowthKeepsPrefix) {
    const auto config = YAML::Load(R"(
type: inkling2_text
model_type: inkling2_loop_text
hidden_size: 64
num_hidden_layers: 2
num_attention_heads: 1
num_key_value_heads: 1
head_dim: 64
vocab_size: 64
n_routed_experts: 4
num_experts_per_tok: 2
intermediate_size: 64
n_layers_per_attn_res_block: 1
sliding_window_size: 64
local_layer_ids: [0]
rel_extent: 64
model_max_length: 2048
sconv_kernel_size: 4
use_value_norm_no_gamma: true
attn_res_normalize_values: true
q_bias: false
o_bias: false
route_scale: 1
mlp_gate_softcap: 4
mlp_up_softcap: 25
final_logit_softcapping: 30
qk_l2_clip_threshold: 1
rel_l2_clip_threshold: 1
log_scaling_alpha: 0
log_scaling_n_floor: 0
)");
    auto network = require(model::create_model_as<model::CausalLanguageModel>(config, {.device = Device::apple_gpu()}));
    constexpr std::byte MARK{0x35};
    for (const auto precision : {core::KVCachePrecision::BF16, core::KVCachePrecision::INT8}) {
        SCOPED_TRACE(core::to_string(precision));
        network->set_kv_cache_precision(precision);
        auto state = require(network->create_state(192));
        auto& cache = dynamic_cast<model::Inkling2State&>(*state);
        cache.position = 3;
        for (auto& visit : cache.visits)
            for (auto* plane : {&visit.key, &visit.value, &visit.key_scales, &visit.value_scales})
                if (plane->defined()) std::ranges::fill(require(plane->host_bytes()), MARK);
        require(network->mark_rewind_point(cache));
        auto key_conv = cache.visits.front().key_conv;

        require(network->grow_state(cache, 768));
        ASSERT_EQ(cache.sequence_position(), 3U);
        ASSERT_EQ(cache.sequence_capacity(), 768U);
        ASSERT_LT(cache.visits.front().key.size(1), cache.visits.back().key.size(1));
        EXPECT_EQ(cache.visits.front().key_conv.storage_identity(), key_conv.storage_identity());
        for (auto& visit : cache.visits)
            for (auto* plane : {&visit.key, &visit.value, &visit.key_scales, &visit.value_scales}) {
                if (!plane->defined()) continue;
                const auto bytes = require(plane->host_bytes());
                const auto used = 3 * plane->size(-1) * tensor::element_size(plane->dtype());
                EXPECT_TRUE(std::ranges::all_of(bytes.first(used), [](auto b) { return b == MARK; }));
                EXPECT_EQ(bytes[used], std::byte{0});
            }

        auto ring = cache.visits.front().key;
        require(ring.host_bytes()).back() = MARK;
        cache.position = cache.local_capacity + 64;
        EXPECT_EQ(network->rewind_floor(cache, cache.position - 1), 0U);
        const auto overwritten = network->rewind(cache, 3);
        ASSERT_FALSE(overwritten);
        EXPECT_EQ(overwritten.error().code, ErrorCode::UNSUPPORTED);
        require(network->mark_rewind_point(cache));
        const auto rewind = cache.position++;
        require(network->grow_state(cache, 1536));
        EXPECT_EQ(cache.visits.front().key.storage_identity(), ring.storage_identity());
        EXPECT_EQ(require(cache.visits.front().key.host_bytes()).back(), MARK);
        EXPECT_GT(cache.visits.back().key.size(1), 768U);
        EXPECT_EQ(network->rewind_floor(cache, rewind), rewind);
        require(network->rewind(cache, rewind));
        EXPECT_EQ(cache.sequence_position(), rewind);
        network->release_workspaces();
    }
}

TEST_F(FusedMetalTest, VqPrefillReportsInvalidExpertId) {
    constexpr std::int64_t TOKENS = 16, EXPERTS = 2, ROWS = 8, WIDTH = 1024;
    const auto device = Device::apple_gpu();
    const ops::BlockScaledWeight bank{ops::BlockScaledFormat::VQ4D,
                                      require(Tensor::zeros({EXPERTS, ROWS, WIDTH / 4}, DType::U8, device)),
                                      require(Tensor::zeros({EXPERTS, ROWS, WIDTH / 64}, DType::U8, device)),
                                      require(Tensor::zeros({EXPERTS, ROWS}, DType::F32, device)), 64};
    std::array<float, 2 * TOKENS> routes{};
    routes[routes.size() - 2] = static_cast<float>(EXPERTS);
    const auto routed = require(host({TOKENS, 1, 2}, routes).to(device));
    const auto input = require(Tensor::zeros({TOKENS, 1, WIDTH}, DType::F32, device));
    gpu.expert_linear(input, bank, routed);
    try {
        gpu.synchronize();
        FAIL() << "out-of-range expert ID was accepted";
    } catch (const ops::Failure& failure) {
        EXPECT_EQ(failure.error().code, ErrorCode::INVALID_ARGUMENT);
        EXPECT_EQ(failure.error().message, "Metal expert ID out of bounds");
    }
}

TEST_F(FusedMetalTest, RelativeAttentionBf16AndInt8CacheMatchCpu) {
    constexpr std::int64_t HEADS = 8, KV_HEADS = 2, DIM = 128, EXTENT = 32, CAP = 96;
    const auto [k_all_cpu, k_all_gpu] = pair({CAP, KV_HEADS, DIM}, 0.4F);
    const auto [v_all_cpu, v_all_gpu] = pair({CAP, KV_HEADS, DIM}, 0.4F);
    std::vector<std::int32_t> all_pos(CAP);
    std::iota(all_pos.begin(), all_pos.end(), 0);
    const auto all_pos_cpu = require(Tensor::from_host({CAP}, std::span<const std::int32_t>(all_pos), Device::cpu()));
    const auto all_pos_gpu = require(all_pos_cpu.to(Device::apple_gpu()));

    // 1. BF16 KV cache (prefill T=37 + decode T=1)
    {
        const auto k_bf16_cpu = cpu.cast(require(k_all_cpu.reshape({CAP, KV_HEADS * DIM})), DType::BF16);
        const auto v_bf16_cpu = cpu.cast(require(v_all_cpu.reshape({CAP, KV_HEADS * DIM})), DType::BF16);
        const auto k_bf16_gpu = require(k_bf16_cpu.to(Device::apple_gpu()));
        const auto v_bf16_gpu = require(v_bf16_cpu.to(Device::apple_gpu()));
        for (const auto [start, tokens] : {std::pair{12, 37}, std::pair{75, 1}}) {
            const auto [q_cpu, q_gpu] = pair({tokens, HEADS * DIM}, 0.3F);
            const auto [rel_cpu, rel_gpu] = pair({tokens, HEADS, EXTENT}, 0.2F);
            const auto [sink_cpu, sink_gpu] = pair({tokens, HEADS}, 0.2F);
            const auto [gate_cpu, gate_gpu] = pair({tokens, HEADS * DIM}, 0.3F);
            std::vector<std::int32_t> pos(static_cast<std::size_t>(tokens));
            std::iota(pos.begin(), pos.end(), start);
            const auto pos_cpu =
                require(Tensor::from_host({tokens}, std::span<const std::int32_t>(pos), Device::cpu()));
            const auto pos_gpu = require(pos_cpu.to(Device::apple_gpu()));
            const ops::Context::RelativeAttention shape{HEADS, KV_HEADS, 0, 0, EXTENT, 0.1F, 1.0F, false};
            const auto out_cpu =
                cpu.relative_attention(q_cpu, k_bf16_cpu, v_bf16_cpu, rel_cpu, sink_cpu, gate_cpu, shape, pos_cpu);
            const auto out_gpu =
                gpu.relative_attention(q_gpu, k_bf16_gpu, v_bf16_gpu, rel_gpu, sink_gpu, gate_gpu, shape, pos_gpu);
            expect_close(out_gpu, out_cpu, tokens >= 8 ? 3e-3F : 5e-4F);
        }
    }

    // 2. INT8 online-quantized KV cache (quantize_cache_ + relative_attention, T=37 + T=1)
    {
        constexpr std::int64_t GROUPS = (KV_HEADS * DIM) / 32;
        auto k_i8_cpu = require(Tensor::zeros({1, CAP, KV_HEADS * DIM}, DType::I8, Device::cpu()));
        auto v_i8_cpu = require(Tensor::zeros({1, CAP, KV_HEADS * DIM}, DType::I8, Device::cpu()));
        auto ks_cpu = require(Tensor::zeros({CAP, GROUPS}, DType::F32, Device::cpu()));
        auto vs_cpu = require(Tensor::zeros({CAP, GROUPS}, DType::F32, Device::cpu()));
        cpu.quantize_cache_(k_i8_cpu, ks_cpu, k_all_cpu, all_pos_cpu);
        cpu.quantize_cache_(v_i8_cpu, vs_cpu, v_all_cpu, all_pos_cpu);

        auto k_i8_gpu = require(Tensor::zeros({1, CAP, KV_HEADS * DIM}, DType::I8, Device::apple_gpu()));
        auto v_i8_gpu = require(Tensor::zeros({1, CAP, KV_HEADS * DIM}, DType::I8, Device::apple_gpu()));
        auto ks_gpu = require(Tensor::zeros({CAP, GROUPS}, DType::F32, Device::apple_gpu()));
        auto vs_gpu = require(Tensor::zeros({CAP, GROUPS}, DType::F32, Device::apple_gpu()));
        gpu.quantize_cache_(k_i8_gpu, ks_gpu, k_all_gpu, all_pos_gpu);
        gpu.quantize_cache_(v_i8_gpu, vs_gpu, v_all_gpu, all_pos_gpu);
        expect_close(ks_gpu, ks_cpu, 1e-5F);
        expect_close(vs_gpu, vs_cpu, 1e-5F);

        for (const auto [start, tokens] : {std::pair{12, 37}, std::pair{75, 1}}) {
            const auto [q_cpu, q_gpu] = pair({tokens, HEADS * DIM}, 0.3F);
            const auto [rel_cpu, rel_gpu] = pair({tokens, HEADS, EXTENT}, 0.2F);
            const auto [sink_cpu, sink_gpu] = pair({tokens, HEADS}, 0.2F);
            const auto [gate_cpu, gate_gpu] = pair({tokens, HEADS * DIM}, 0.3F);
            std::vector<std::int32_t> pos(static_cast<std::size_t>(tokens));
            std::iota(pos.begin(), pos.end(), start);
            const auto pos_cpu =
                require(Tensor::from_host({tokens}, std::span<const std::int32_t>(pos), Device::cpu()));
            const auto pos_gpu = require(pos_cpu.to(Device::apple_gpu()));
            const ops::Context::RelativeAttention shape{HEADS, KV_HEADS, 0, 0, EXTENT, 0.1F, 1.0F, false};
            const auto out_cpu = cpu.relative_attention(q_cpu, require(k_i8_cpu.reshape({CAP, KV_HEADS * DIM})),
                                                        require(v_i8_cpu.reshape({CAP, KV_HEADS * DIM})), rel_cpu,
                                                        sink_cpu, gate_cpu, shape, pos_cpu, ks_cpu, vs_cpu);
            const auto out_gpu = gpu.relative_attention(q_gpu, require(k_i8_gpu.reshape({CAP, KV_HEADS * DIM})),
                                                        require(v_i8_gpu.reshape({CAP, KV_HEADS * DIM})), rel_gpu,
                                                        sink_gpu, gate_gpu, shape, pos_gpu, ks_gpu, vs_gpu);
            expect_close(out_gpu, out_cpu, tokens >= 8 ? 5e-3F : 1e-3F);
        }
    }
}

TEST_F(FusedMetalTest, RelativeAttentionSwaRingBufferMatchesLinearCpu) {
    constexpr std::int64_t HEADS = 8, KV_HEADS = 2, DIM = 128, EXTENT = 16;
    constexpr std::int64_t WINDOW = 16, RING_CAP = 32, TOTAL = 48;
    constexpr std::int64_t KV = KV_HEADS * DIM, GROUPS = KV / 32;
    const auto [k_lin_cpu, k_lin_gpu] = pair({TOTAL, KV_HEADS, DIM}, 0.4F);
    const auto [v_lin_cpu, v_lin_gpu] = pair({TOTAL, KV_HEADS, DIM}, 0.4F);

    for (const auto cache_dtype : {DType::BF16, DType::I8}) {
        SCOPED_TRACE(cache_dtype == DType::BF16 ? "BF16-ring" : "I8-ring");
        auto k_lin_store = require(Tensor::zeros({1, TOTAL, KV}, cache_dtype, Device::cpu()));
        auto v_lin_store = require(Tensor::zeros({1, TOTAL, KV}, cache_dtype, Device::cpu()));
        auto ks_lin =
            cache_dtype == DType::I8 ? require(Tensor::zeros({TOTAL, GROUPS}, DType::F32, Device::cpu())) : Tensor{};
        auto vs_lin =
            cache_dtype == DType::I8 ? require(Tensor::zeros({TOTAL, GROUPS}, DType::F32, Device::cpu())) : Tensor{};

        auto k_ring_gpu = require(Tensor::zeros({1, RING_CAP, KV}, cache_dtype, Device::apple_gpu()));
        auto v_ring_gpu = require(Tensor::zeros({1, RING_CAP, KV}, cache_dtype, Device::apple_gpu()));
        auto ks_ring = cache_dtype == DType::I8
                           ? require(Tensor::zeros({RING_CAP, GROUPS}, DType::F32, Device::apple_gpu()))
                           : Tensor{};
        auto vs_ring = cache_dtype == DType::I8
                           ? require(Tensor::zeros({RING_CAP, GROUPS}, DType::F32, Device::apple_gpu()))
                           : Tensor{};

        // Populate 48 positions into the 32-slot ring in 16-token ring-safe chunks (0..15, 16..31, 32..47).
        for (std::int64_t start = 0; start < TOTAL; start += 16) {
            constexpr std::int64_t CHUNK = 16;
            std::vector<std::int32_t> pos(CHUNK), lpos(CHUNK);
            for (std::int64_t i = 0; i < CHUNK; ++i) {
                pos[static_cast<std::size_t>(i)] = static_cast<std::int32_t>(start + i);
                lpos[static_cast<std::size_t>(i)] = static_cast<std::int32_t>((start + i) % RING_CAP);
            }
            const auto pos_cpu = require(Tensor::from_host({CHUNK}, std::span<const std::int32_t>(pos), Device::cpu()));
            const auto pos_gpu = require(pos_cpu.to(Device::apple_gpu()));
            const auto lpos_gpu =
                require(Tensor::from_host({CHUNK}, std::span<const std::int32_t>(lpos), Device::apple_gpu()));
            const auto k_chunk_cpu = cpu.slice(k_lin_cpu, 0, start, CHUNK);
            const auto v_chunk_cpu = cpu.slice(v_lin_cpu, 0, start, CHUNK);
            const auto k_chunk_gpu = gpu.slice(k_lin_gpu, 0, start, CHUNK);
            const auto v_chunk_gpu = gpu.slice(v_lin_gpu, 0, start, CHUNK);
            if (cache_dtype == DType::I8) {
                cpu.quantize_cache_(k_lin_store, ks_lin, k_chunk_cpu, pos_cpu);
                cpu.quantize_cache_(v_lin_store, vs_lin, v_chunk_cpu, pos_cpu);
                gpu.quantize_cache_(k_ring_gpu, ks_ring, k_chunk_gpu, lpos_gpu);
                gpu.quantize_cache_(v_ring_gpu, vs_ring, v_chunk_gpu, lpos_gpu);
            } else {
                cpu.scatter_unique_(k_lin_store, cpu.cast(require(k_chunk_cpu.reshape({1, CHUNK, KV})), DType::BF16),
                                    pos_cpu);
                cpu.scatter_unique_(v_lin_store, cpu.cast(require(v_chunk_cpu.reshape({1, CHUNK, KV})), DType::BF16),
                                    pos_cpu);
                gpu.scatter_unique_(k_ring_gpu, gpu.cast(require(k_chunk_gpu.reshape({1, CHUNK, KV})), DType::BF16),
                                    lpos_gpu);
                gpu.scatter_unique_(v_ring_gpu, gpu.cast(require(v_chunk_gpu.reshape({1, CHUNK, KV})), DType::BF16),
                                    lpos_gpu);
            }
            // Test attention after wrap-around on the final chunk (positions 32..47, querying across ring boundary
            // 31->0!).
            if (start == 32) {
                for (const auto [q_start, tokens] : {std::pair{32, 16}, std::pair{47, 1}}) {
                    const auto [q_cpu, q_gpu] = pair({tokens, HEADS * DIM}, 0.3F);
                    const auto [rel_cpu, rel_gpu] = pair({tokens, HEADS, EXTENT}, 0.2F);
                    const auto [sink_cpu, sink_gpu] = pair({tokens, HEADS}, 0.2F);
                    const auto [gate_cpu, gate_gpu] = pair({tokens, HEADS * DIM}, 0.3F);
                    std::vector<std::int32_t> qpos(static_cast<std::size_t>(tokens));
                    std::iota(qpos.begin(), qpos.end(), q_start);
                    const auto qpos_cpu =
                        require(Tensor::from_host({tokens}, std::span<const std::int32_t>(qpos), Device::cpu()));
                    const auto qpos_gpu = require(qpos_cpu.to(Device::apple_gpu()));
                    const ops::Context::RelativeAttention lin_shape{HEADS,  KV_HEADS, 0,   WINDOW,
                                                                    EXTENT, 0.F,      0.F, false};
                    const ops::Context::RelativeAttention ring_shape{HEADS,  KV_HEADS, 0,   WINDOW,
                                                                     EXTENT, 0.F,      0.F, true};
                    const auto out_cpu = cpu.relative_attention(
                        q_cpu, require(k_lin_store.reshape({TOTAL, KV})), require(v_lin_store.reshape({TOTAL, KV})),
                        rel_cpu, sink_cpu, gate_cpu, lin_shape, qpos_cpu, ks_lin, vs_lin);
                    const auto out_gpu = gpu.relative_attention(
                        q_gpu, require(k_ring_gpu.reshape({RING_CAP, KV})), require(v_ring_gpu.reshape({RING_CAP, KV})),
                        rel_gpu, sink_gpu, gate_gpu, ring_shape, qpos_gpu, ks_ring, vs_ring);
                    expect_close(out_gpu, out_cpu, tokens >= 8 ? 5e-3F : 1e-3F);
                }
            }
        }
    }
}
} // namespace
