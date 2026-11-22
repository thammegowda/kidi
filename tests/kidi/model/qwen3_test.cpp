#include "kidi/model/qwen3.h"

#include <cmath>
#include <filesystem>

#include <gtest/gtest.h>

namespace {
auto make_config() -> YAML::Node {
    YAML::Node config;
    config["model_type"] = "qwen3";
    config["hidden_act"] = "silu";
    config["vocab_size"] = 8;
    config["hidden_size"] = 4;
    config["intermediate_size"] = 6;
    config["num_hidden_layers"] = 1;
    config["num_attention_heads"] = 2;
    config["num_key_value_heads"] = 1;
    config["head_dim"] = 2;
    config["max_position_embeddings"] = 16;
    config["rms_norm_eps"] = 1e-6F;
    config["rope_parameters"]["rope_theta"] = 10000.F;
    config["layer_types"].push_back("full_attention");
    return config;
}

class Qwen3Test : public ::testing::Test {
protected:
    void SetUp() override {
        state = model->state_dict();
        for (auto& [name, tensor] : state) {
            if (tensor.dtype() == kidi::tensor::DType::I8) {
                auto values = tensor.data<std::int8_t>();
                ASSERT_TRUE(values.has_value()) << "cannot initialize Qwen3 INT8 state";
                std::ranges::fill(*values, name.ends_with("norm.weight") ? 127 : 1);
            } else {
                auto values = tensor.data<float>();
                ASSERT_TRUE(values.has_value()) << "cannot initialize Qwen3 scale state";
                std::ranges::fill(*values, name.ends_with("norm.scale") ? 1.F / 127.F : 0.01F);
            }
        }
        embeddings = model->embed(context, tokens);
        output = model->forward(context, embeddings);
        context.synchronize();
        const auto values = output.data<float>();
        ASSERT_TRUE(values.has_value()) << "Qwen3 output contains a non-finite value";
        expected.assign(values->begin(), values->end());
    }

    YAML::Node config = make_config();
    const kidi::ModuleScope scope{kidi::tensor::DType::I8, true, kidi::tensor::Device::cpu()};
    kidi::model::Qwen3 model{config};
    kidi::StateDict state;
    kidi::ops::Context context{kidi::tensor::Device::cpu()};
    const std::array<std::int32_t, 2> tokens{1, 2};
    kidi::tensor::Tensor embeddings, output;
    std::vector<float> expected;
};

TEST_F(Qwen3Test, FusesProjections) {
    for (const auto* name : {"layers.0.self_attn.qkv_proj.weight", "layers.0.self_attn.qkv_proj.scale",
                             "layers.0.mlp.gate_up_proj.weight", "layers.0.mlp.gate_up_proj.scale"})
        EXPECT_TRUE(state.contains(name)) << "Qwen3 fused projection state is missing: " << name;
    for (const auto* name :
         {"layers.0.self_attn.q_proj.weight", "layers.0.self_attn.k_proj.weight", "layers.0.self_attn.v_proj.weight",
          "layers.0.mlp.gate_proj.weight", "layers.0.mlp.up_proj.weight"})
        EXPECT_FALSE(state.contains(name)) << "Qwen3 retained an unfused projection: " << name;
    ASSERT_TRUE(state.contains("layers.0.self_attn.qkv_proj.weight"));
    ASSERT_TRUE(state.contains("layers.0.mlp.gate_up_proj.weight"));
    EXPECT_TRUE(
        std::ranges::equal(state.at("layers.0.self_attn.qkv_proj.weight").shape(), std::array<std::int64_t, 2>{8, 4}))
        << "Qwen3 fused projection shape is invalid";
    EXPECT_TRUE(
        std::ranges::equal(state.at("layers.0.mlp.gate_up_proj.weight").shape(), std::array<std::int64_t, 2>{12, 4}))
        << "Qwen3 fused projection shape is invalid";
}

TEST_F(Qwen3Test, FiniteAndMaskInvariant) {
    ASSERT_TRUE(output.dimensions() == 3 && output.size(0) == 1 && output.size(1) == 2 && output.size(2) == 4)
        << "Qwen3 output shape is invalid";
    EXPECT_TRUE(std::ranges::all_of(expected, [](float value) { return std::isfinite(value); }))
        << "Qwen3 output contains a non-finite value";
    const auto mask = kidi::ops::require(kidi::tensor::Tensor::zeros({1, 1, 2, 2}, kidi::tensor::DType::F32));
    const auto explicit_mask = model->forward(context, embeddings, mask);
    const auto replayed = model->forward(context, embeddings);
    context.synchronize();
    const auto explicit_values = kidi::ops::require(explicit_mask.data<float>());
    const auto replayed_values = kidi::ops::require(replayed.data<float>());
    EXPECT_TRUE(std::ranges::equal(expected, explicit_values)) << "Qwen3 masked replay changed the output";
    EXPECT_TRUE(std::ranges::equal(expected, replayed_values)) << "Qwen3 masked replay changed the output";
}

TEST_F(Qwen3Test, PackedMatchesSeparate) {
    const std::array<std::int32_t, 1> one_token{2};
    const auto single_embedding = model->embed(context, one_token);
    const auto single_output = model->forward(context, single_embedding);
    const std::array packed_inputs{embeddings, single_embedding};
    const auto packed_embedding = context.concat(packed_inputs, 1);
    constexpr std::array<float, 9> MASK{0.F, 0.F, -1.0e9F, 0.F, 0.F, -1.0e9F, -1.0e9F, -1.0e9F, 0.F};
    const auto packed_mask =
        kidi::ops::require(kidi::tensor::Tensor::from_host({1, 1, 3, 3}, std::span<const float>(MASK)));
    const auto packed_output = model->forward(context, packed_embedding, packed_mask);
    const std::array<std::size_t, 2> segment_lengths{2, 1};
    const auto segmented_output = model->forward_segmented(context, packed_embedding, segment_lengths);
    context.synchronize();
    const auto single_values = kidi::ops::require(single_output.data<float>());
    const auto packed_values = kidi::ops::require(packed_output.data<float>());
    const auto segmented_values = kidi::ops::require(segmented_output.data<float>());
    for (std::size_t index = 0; index < expected.size(); ++index)
        ASSERT_NEAR(expected[index], packed_values[index], 1e-5F) << "Qwen3 packed first segment differs";
    for (std::size_t index = 0; index < single_values.size(); ++index)
        ASSERT_NEAR(single_values[index], packed_values[expected.size() + index], 1e-5F)
            << "Qwen3 packed second segment differs";
    for (std::size_t index = 0; index < expected.size(); ++index)
        ASSERT_NEAR(expected[index], segmented_values[index], 1e-5F) << "Qwen3 segmented first segment differs";
    for (std::size_t index = 0; index < single_values.size(); ++index)
        ASSERT_NEAR(single_values[index], segmented_values[expected.size() + index], 1e-5F)
            << "Qwen3 segmented second segment differs";
}

TEST_F(Qwen3Test, BatchedGuidanceMatchesSingle) {
    const std::array batch_inputs{embeddings, embeddings};
    const auto batch_embedding = context.concat(batch_inputs, 0);
    const auto batch_mask = kidi::ops::require(kidi::tensor::Tensor::zeros({2, 1, 2, 2}, kidi::tensor::DType::F32));
    const auto batch_output = model->forward(context, batch_embedding, batch_mask);
    context.synchronize();
    const auto batch_values = kidi::ops::require(batch_output.data<float>());
    for (std::size_t index = 0; index < batch_values.size(); ++index)
        ASSERT_NEAR(expected[index % expected.size()], batch_values[index], 1e-5F) << "Qwen3 batched guidance differs";
}

TEST_F(Qwen3Test, FusesUnfusedCheckpoint) {
    const auto path = std::filesystem::temp_directory_path() / "kidi-qwen3-mapping-test.safetensors";
    std::filesystem::remove(path);
    kidi::StateDict unfused;
    for (const auto& [name, tensor] : state) {
        const auto key = "model." + name;
        if (const auto pos = key.find(".self_attn.qkv_proj."); pos != std::string::npos) {
            const auto prefix = key.substr(0, pos) + ".self_attn.";
            const auto suffix = key.substr(pos + std::string_view(".self_attn.qkv_proj.").size());
            unfused.emplace(prefix + "q_proj." + suffix, kidi::ops::require(tensor.narrow(0, 0, 4)));
            unfused.emplace(prefix + "k_proj." + suffix, kidi::ops::require(tensor.narrow(0, 4, 2)));
            unfused.emplace(prefix + "v_proj." + suffix, kidi::ops::require(tensor.narrow(0, 6, 2)));
        } else if (const auto pos = key.find(".mlp.gate_up_proj."); pos != std::string::npos) {
            const auto prefix = key.substr(0, pos) + ".mlp.";
            const auto suffix = key.substr(pos + std::string_view(".mlp.gate_up_proj.").size());
            unfused.emplace(prefix + "gate_proj." + suffix, kidi::ops::require(tensor.narrow(0, 0, 6)));
            unfused.emplace(prefix + "up_proj." + suffix, kidi::ops::require(tensor.narrow(0, 6, 6)));
        } else {
            unfused.emplace(key, tensor);
        }
    }
    kidi::ops::require(kidi::checkpoint::Weights::save(path, unfused));
    const auto loaded = kidi::ops::require(kidi::checkpoint::Weights::load(path));
    std::filesystem::remove(path);
    kidi::model::Qwen3 from_unfused(config);
    kidi::ops::require(from_unfused->set_checkpoint(loaded));
    const auto fused_output = from_unfused->forward(context, from_unfused->embed(context, tokens));
    context.synchronize();
    EXPECT_TRUE(std::ranges::equal(expected, kidi::ops::require(fused_output.data<float>())))
        << "Qwen3 unfused checkpoint fusion changed output";
}

#if defined(__APPLE__)
TEST_F(Qwen3Test, MetalMatchesCpu) {
    const std::array<std::int32_t, 4> metal_tokens{1, 2, 1, 2};
    const auto cpu_metal_reference = model->forward(context, model->embed(context, metal_tokens));
    context.synchronize();
    const auto reference_bytes = kidi::ops::require(cpu_metal_reference.copy_to_host());
    const kidi::ModuleScope metal_scope(kidi::tensor::DType::I8, false, kidi::tensor::Device::apple_gpu());
    kidi::model::Qwen3 metal(config);
    kidi::ops::require(metal->set_state(state));
    kidi::ops::Context metal_context(kidi::tensor::Device::apple_gpu());
    const auto metal_output = metal->forward(metal_context, metal->embed(metal_context, metal_tokens));
    metal_context.synchronize();
    const auto metal_bytes = kidi::ops::require(metal_output.copy_to_host());
    const auto reference =
        std::span(reinterpret_cast<const float*>(reference_bytes.data()), reference_bytes.size() / sizeof(float));
    const auto actual =
        std::span(reinterpret_cast<const float*>(metal_bytes.data()), metal_bytes.size() / sizeof(float));
    ASSERT_EQ(reference.size(), actual.size()) << "Qwen3 Metal output shape differs";
    for (std::size_t index = 0; index < actual.size(); ++index) {
        ASSERT_TRUE(std::isfinite(actual[index])) << "Qwen3 Metal numerical mismatch";
        ASSERT_NEAR(reference[index], actual[index], 0.02F) << "Qwen3 Metal numerical mismatch";
    }
}
#endif
} // namespace
