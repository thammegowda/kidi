#include <cmath>
#include <filesystem>

#include <gtest/gtest.h>

#include "kidi/model/whisper.h"
#include "kidi/ops/quantization.h"

namespace {
using kidi::ops::require;
using kidi::tensor::Tensor;

auto tiny_config() -> YAML::Node {
    return YAML::Load(R"({
        "type":"whisper", "model_type":"whisper", "architectures":["WhisperForConditionalGeneration"],
        "activation_function":"gelu", "activation_dropout":0, "attention_dropout":0,
        "decoder_layerdrop":0, "dropout":0, "encoder_layerdrop":0, "scale_embedding":false,
        "is_encoder_decoder":true, "use_cache":true, "d_model":384, "encoder_attention_heads":6,
        "encoder_ffn_dim":1536, "encoder_layers":4, "decoder_attention_heads":6, "decoder_ffn_dim":1536,
        "decoder_layers":4, "max_source_positions":1500, "max_target_positions":448,
        "num_mel_bins":80, "vocab_size":51865
    })");
}

class WhisperTest : public ::testing::Test {
protected:
    void SetUp() override {
        auto created = kidi::model::WhisperImpl::create(config);
        ASSERT_TRUE(created.has_value()) << created.error().message;
        model.emplace(std::move(*created));
    }

    YAML::Node config = tiny_config();
    const kidi::ModuleScope scope{kidi::tensor::DType::F32, false, kidi::tensor::Device::cpu()};
    std::optional<kidi::model::Whisper> model;
};

TEST_F(WhisperTest, FusesProjectionState) {
    const auto state = (*model)->state_dict();
    for (const auto* name :
         {"encoder.conv1.weight", "encoder.layers.3.self_attn.qkv_proj.weight", "decoder.embed_tokens.weight",
          "decoder.layers.0.self_attn.qkv_proj.bias", "decoder.layers.0.encoder_attn.q_proj.bias",
          "decoder.layers.0.encoder_attn.kv_proj.weight", "proj_out.weight"})
        EXPECT_TRUE(state.contains(name)) << name;
    for (const auto* name : {"encoder.layers.0.self_attn.k_proj.weight", "decoder.layers.0.self_attn.q_proj.weight",
                             "decoder.layers.0.encoder_attn.k_proj.weight"})
        EXPECT_FALSE(state.contains(name)) << name;
}

TEST_F(WhisperTest, MapsUnfusedCheckpoint) {
    const auto path = std::filesystem::temp_directory_path() / "kidi-whisper-mapping-test.safetensors";
    std::filesystem::remove(path);
    const auto matrix = [](float fill) {
        const std::array<float, 16> values{fill, fill, fill, fill, fill, fill, fill, fill,
                                           fill, fill, fill, fill, fill, fill, fill, fill};
        return require(Tensor::from_host({4, 4}, std::span<const float>(values)));
    };
    const auto vector = [](float fill) {
        const std::array<float, 4> values{fill, fill, fill, fill};
        return require(Tensor::from_host({4}, std::span<const float>(values)));
    };
    kidi::StateDict unfused{
        {"model.decoder.embed_tokens.weight", matrix(1.F)},
        {"model.proj_out.weight", matrix(1.F)},
        {"model.encoder.layers.0.self_attn.q_proj.weight", matrix(0.25F)},
        {"model.encoder.layers.0.self_attn.k_proj.weight", matrix(0.5F)},
        {"model.encoder.layers.0.self_attn.v_proj.weight", matrix(0.75F)},
        {"model.encoder.layers.0.self_attn.q_proj.bias", vector(0.125F)},
        {"model.encoder.layers.0.self_attn.v_proj.bias", vector(-0.25F)},
        {"model.decoder.layers.0.encoder_attn.k_proj.weight", matrix(1.5F)},
        {"model.decoder.layers.0.encoder_attn.v_proj.weight", matrix(2.5F)},
        {"model.decoder.layers.0.encoder_attn.v_proj.bias", vector(-0.5F)},
    };
    require(kidi::checkpoint::Weights::save(path, unfused));
    const auto loaded_weights = require(kidi::checkpoint::Weights::load(path));
    std::filesystem::remove(path);
    const auto mapped = require((*model)->map_checkpoint(loaded_weights));
    EXPECT_FALSE(mapped.contains("proj_out.weight"));
    EXPECT_FALSE(mapped.contains("encoder.layers.0.self_attn.q_proj.weight"));
    EXPECT_FALSE(mapped.contains("decoder.layers.0.encoder_attn.v_proj.bias"));
    const auto qkv_weight_tensor = require(mapped.tensor("encoder.layers.0.self_attn.qkv_proj.weight"));
    const auto qkv_bias_tensor = require(mapped.tensor("encoder.layers.0.self_attn.qkv_proj.bias"));
    const auto kv_weight_tensor = require(mapped.tensor("decoder.layers.0.encoder_attn.kv_proj.weight"));
    const auto kv_bias_tensor = require(mapped.tensor("decoder.layers.0.encoder_attn.kv_proj.bias"));
    const auto qkv_weight = require(qkv_weight_tensor.data<float>());
    const auto qkv_bias = require(qkv_bias_tensor.data<float>());
    const auto kv_weight = require(kv_weight_tensor.data<float>());
    const auto kv_bias = require(kv_bias_tensor.data<float>());
    ASSERT_EQ(qkv_weight.size(), 48U);
    EXPECT_EQ(qkv_weight[0], 0.25F);
    EXPECT_EQ(qkv_weight[16], 0.5F);
    EXPECT_EQ(qkv_weight[32], 0.75F);
    ASSERT_EQ(qkv_bias.size(), 12U);
    EXPECT_EQ(qkv_bias[0], 0.125F);
    EXPECT_EQ(qkv_bias[4], 0.F);
    EXPECT_EQ(qkv_bias[8], -0.25F);
    ASSERT_EQ(kv_weight.size(), 32U);
    EXPECT_EQ(kv_weight[0], 1.5F);
    EXPECT_EQ(kv_weight[16], 2.5F);
    ASSERT_EQ(kv_bias.size(), 8U);
    EXPECT_EQ(kv_bias[0], 0.F);
    EXPECT_EQ(kv_bias[4], -0.5F);
}

TEST_F(WhisperTest, ValidatesConfigs) {
    config["d_model"] = 512;
    config["encoder_layers"] = 6;
    config["decoder_layers"] = 6;
    config["encoder_attention_heads"] = 8;
    config["decoder_attention_heads"] = 8;
    config["encoder_ffn_dim"] = 2048;
    config["decoder_ffn_dim"] = 2048;
    EXPECT_TRUE(kidi::model::WhisperImpl::validate_config(config)) << "base";
    config["d_model"] = 1024;
    EXPECT_FALSE(kidi::model::WhisperImpl::validate_config(config)) << "base with d_model 1024";
    config["d_model"] = 1280;
    config["encoder_layers"] = 32;
    config["decoder_layers"] = 32;
    config["encoder_attention_heads"] = 20;
    config["decoder_attention_heads"] = 20;
    config["encoder_ffn_dim"] = 5120;
    config["decoder_ffn_dim"] = 5120;
    config["num_mel_bins"] = 128;
    config["vocab_size"] = 51866;
    EXPECT_TRUE(kidi::model::WhisperImpl::validate_config(config)) << "large-v3";
    for (const auto* name : {"num_mel_bins", "vocab_size", "decoder_layers", "decoder_attention_heads"}) {
        const auto original = config[name].as<int>();
        config[name] = original - 1;
        EXPECT_FALSE(kidi::model::WhisperImpl::validate_config(config)) << name;
        config[name] = original;
    }
    config["decoder_layers"] = 4;
    EXPECT_TRUE(kidi::model::WhisperImpl::validate_config(config)) << "turbo";
    config["decoder_layers"] = 3;
    EXPECT_FALSE(kidi::model::WhisperImpl::validate_config(config)) << "three decoder layers";
}

TEST_F(WhisperTest, RejectsMismatchedMelBins) {
    EXPECT_FALSE((*model)->encode_audio({std::vector<float>(128 * 3000), 128, 3000}));
}

TEST_F(WhisperTest, QuantizedConv1d) {
    const std::array<std::int8_t, 18> bytes{0, 8, 0, 0, 0, 0, 0, 0, 0, 0, -8, 0, 4, 0, -4, 0, 4, 0};
    const std::array<float, 3> scales{0.125F, 0.25F, 0.5F};
    std::array<float, 18> floating;
    for (std::size_t index = 0; index < bytes.size(); ++index) floating[index] = bytes[index] * scales[index / 6];
    const auto input = require(
        Tensor::from_host({1, 4, 2}, std::span<const float>(std::array{-1.F, 0.F, 0.5F, 1.F, -0.5F, 0.5F, 1.F, -1.F})));
    const auto bias = require(Tensor::from_host({3}, std::span<const float>(std::array{0.5F, -0.25F, 0.F})));
    kidi::layers::WhisperConv1d reference(2, 3, 1);
    require(reference->set_state(
        {{"weight", require(Tensor::from_host({3, 2, 3}, std::span<const float>(floating)))}, {"bias", bias}}));
    const kidi::ModuleScope int8(kidi::tensor::DType::I8, false);
    kidi::layers::WhisperConv1d quantized(2, 3, 1);
    require(
        quantized->set_state({{"weight", require(Tensor::from_host({3, 2, 3}, std::span<const std::int8_t>(bytes)))},
                              {"bias", bias},
                              {"scale", require(Tensor::from_host({3, 1}, std::span<const float>(scales)))}}));
    kidi::ops::Context context;
    const auto expected = reference->forward(context, input);
    const auto actual = quantized->forward(context, input);
    const auto expected_values = require(expected.data<float>()), actual_values = require(actual.data<float>());
    for (std::size_t index = 0; index < actual_values.size(); ++index) {
        ASSERT_TRUE(std::isfinite(actual_values[index])) << "at " << index;
        ASSERT_LE(std::abs(actual_values[index] - expected_values[index]), 0.02F) << "at " << index;
    }
    const auto quiet = require(Tensor::zeros({1, 4, 2}, kidi::tensor::DType::F32));
    const auto repeated = quantized->forward(context, quiet);
    const auto repeated_values = require(repeated.data<float>()), bias_values = require(bias.data<float>());
    for (std::size_t index = 0; index < repeated_values.size(); ++index)
        ASSERT_LE(std::abs(repeated_values[index] - bias_values[index % 3]), 1e-6F) << "at " << index;
    for (const std::int64_t length : {17, 33, 3}) {
        SCOPED_TRACE(length);
        const auto varying = require(Tensor::zeros({1, length, 2}, kidi::tensor::DType::F32));
        const auto resized = quantized->forward(context, varying);
        ASSERT_EQ(resized.size(1), static_cast<std::size_t>(length));
        const auto values = require(resized.data<float>());
        for (std::size_t index = 0; index < values.size(); ++index)
            ASSERT_LE(std::abs(values[index] - bias_values[index % 3]), 1e-6F) << "at " << index;
    }
}

TEST(WhisperPackTest, PacksFp16LikeFp32) {
    const std::array<std::uint16_t, 4> fp16_bits{0x3c00, 0xc000, 0x3800, 0x0000};
    const std::array<float, 4> fp32_values{1.F, -2.F, 0.5F, 0.F};
    const auto fp16_weight =
        require(Tensor::from_bytes({2, 2}, kidi::tensor::DType::F16,
                                   std::span(reinterpret_cast<const std::byte*>(fp16_bits.data()), sizeof(fp16_bits))));
    const auto fp32_weight = require(Tensor::from_host({2, 2}, std::span<const float>(fp32_values)));
    const auto packed_fp16 = require(kidi::ops::pack_weight(fp16_weight, 8, 2));
    const auto packed_fp32 = require(kidi::ops::pack_weight(fp32_weight, 8, 2));
    EXPECT_TRUE(std::ranges::equal(require(packed_fp16.values.host_bytes()), require(packed_fp32.values.host_bytes())));
    EXPECT_TRUE(
        std::ranges::equal(require(packed_fp16.scales.data<float>()), require(packed_fp32.scales.data<float>())));
}

TEST(WhisperAttentionTest, ChunkedSelfAttn) {
    const kidi::ModuleScope allocated(kidi::tensor::DType::F32, true, kidi::tensor::Device::cpu());
    kidi::ops::Context context;
    kidi::layers::WhisperSelfAttention chunked_attention(4, 2);
    const auto chunked_input = require(Tensor::zeros({1, 129, 4}, kidi::tensor::DType::F32));
    const auto chunked_mask = require(Tensor::zeros({1, 1, 129, 129}, kidi::tensor::DType::F32));
    const auto chunked_output = chunked_attention->forward(context, chunked_input, chunked_mask);
    EXPECT_EQ(chunked_output.dimensions(), 3U);
    EXPECT_EQ(chunked_output.size(1), 129U);
    EXPECT_EQ(chunked_output.size(2), 4U);
}
} // namespace
