#include <bit>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <string_view>

#include <gtest/gtest.h>

#include "kidi/checkpoint/package.h"
#include "kidi/model/registry.h"
#include "kidi/model/transformer.h"
#include "kidi/inference/translator.h"

namespace {

constexpr std::string_view TOKENIZER_JSON = R"({
  "version": "1.0",
  "decoder": {"type": "Fuse"},
  "model": {
    "type": "WordPiece",
    "unk_token": "<unk>",
    "continuing_subword_prefix": "",
    "max_input_chars_per_word": 100,
    "fuse_unk": false,
    "vocab": {"<pad>": 0, "<unk>": 1, "<s>": 2, "</s>": 3, "hi": 4}
  }
})";

constexpr std::string_view MANIFEST_JSON = R"({
  "model_type": "rtg_transformer_nmt",
  "weights_file": "model.safetensors",
  "tokenizers": {"source": "tokenizer.src.json", "target": "tokenizer.tgt.json"},
  "io": {"input_format": "moses_tokenized", "output_format": "moses_tokenized"},
  "source_tokens": 8,
  "encoder_layers": 1,
  "decoder_layers": 1,
  "hidden_size": 4,
  "feed_forward_size": 8,
  "attention_heads": 2,
  "source_vocabulary_size": 5,
  "target_vocabulary_size": 5,
  "activation": "gelu",
  "attention_bias": true,
  "tied_embeddings": "one-way",
  "layer_norm_epsilon": 1.0e-5,
  "position_encoding": "sinusoidal",
  "maximum_position": 32,
  "decode": {
    "beam_size": 2,
    "maximum_extra_tokens": 4,
    "length_penalty": 0.6
  }
})";

auto write(const std::filesystem::path& path, std::string_view contents = {}) -> void {
    std::ofstream(path, std::ios::binary) << contents;
}

auto write_empty_safetensors(const std::filesystem::path& path) -> void {
    constexpr std::string_view HEADER = "{}      ";
    std::uint64_t header_size = HEADER.size();
    if constexpr (std::endian::native == std::endian::big) {
        header_size = std::byteswap(header_size);
    }
    std::ofstream output(path, std::ios::binary);
    output.write(reinterpret_cast<const char*>(&header_size), sizeof(header_size));
    output.write(HEADER.data(), static_cast<std::streamsize>(HEADER.size()));
}

class PackageTest : public ::testing::Test {
protected:
    void SetUp() override {
        directory = std::filesystem::temp_directory_path() /
                    (std::string("kidi-rtg-package-test-") +
                     ::testing::UnitTest::GetInstance()->current_test_info()->name());
        std::filesystem::remove_all(directory);
        std::filesystem::create_directories(directory);
        write_empty_safetensors(directory / "model.safetensors");
        write(directory / "tokenizer.src.json", TOKENIZER_JSON);
        write(directory / "tokenizer.tgt.json", TOKENIZER_JSON);
        write(directory / "config.json", MANIFEST_JSON);
    }

    void TearDown() override {
        std::error_code error;
        std::filesystem::remove_all(directory, error);
    }

    std::filesystem::path directory;
};

TEST_F(PackageTest, LoadsRtgPackage) {
    auto package = kidi::checkpoint::Package::load(directory);
    ASSERT_TRUE(package.has_value()) << "valid RTG package was rejected: " << package.error().message;
    EXPECT_EQ(package->source_tokenizer().token_id("<s>"), 2) << "valid RTG package was rejected";
    EXPECT_EQ(package->target_tokenizer().vocabulary_size(), 5U) << "valid RTG package was rejected";
    EXPECT_EQ(package->config()["model"]["source_pad_id"].as<int>(), 0) << "valid RTG package was rejected";
    EXPECT_EQ(package->config()["decode"]["begin_id"].as<int>(), 2) << "valid RTG package was rejected";
    EXPECT_EQ(package->config()["decode"]["end_id"].as<int>(), 3) << "valid RTG package was rejected";
    EXPECT_FALSE(kidi::checkpoint::Package::load(directory / "config.json"))
        << "manifest path was accepted as a model directory";
}

TEST_F(PackageTest, ValidatesTransformerConfig) {
    auto package = kidi::checkpoint::Package::load(directory);
    ASSERT_TRUE(package.has_value()) << package.error().message;
    auto model_config = YAML::Clone(package->config()["model"]);
    ASSERT_TRUE(kidi::model::TransformerImpl::validate_config(model_config));
    auto scratch = kidi::model::TransformerImpl::create(model_config);
    ASSERT_TRUE(scratch.has_value()) << scratch.error().message;
    const auto initialized = (*scratch)->state_dict();
    ASSERT_TRUE(initialized.contains("encoder.0.qkv.weight"));
    EXPECT_TRUE(initialized.at("encoder.0.qkv.weight").defined());
    const auto target = initialized.at("target_embedding.weight").host_bytes();
    const auto projection = initialized.at("generator.weight").host_bytes();
    ASSERT_TRUE(target);
    ASSERT_TRUE(projection);
    EXPECT_EQ(target->data(), projection->data());
    for (const auto* key : {"type", "hidden_size", "source_pad_id"}) {
        auto invalid = YAML::Clone(model_config);
        invalid.remove(key);
        auto result = kidi::model::TransformerImpl::create(invalid);
        ASSERT_FALSE(result) << key;
        EXPECT_EQ(result.error().code, kidi::ErrorCode::INVALID_MANIFEST) << key;
    }
    auto invalid = YAML::Clone(model_config);
    invalid["attention_heads"] = 3;
    EXPECT_FALSE(kidi::model::TransformerImpl::validate_config(invalid));
    invalid = YAML::Clone(model_config);
    invalid["hidden_size"] = 0;
    EXPECT_FALSE(kidi::model::TransformerImpl::validate_config(invalid));
    invalid = YAML::Clone(model_config);
    invalid["source_pad_id"] = -1;
    EXPECT_FALSE(kidi::model::TransformerImpl::validate_config(invalid));
    invalid = YAML::Clone(model_config);
    invalid["layer_norm_epsilon"] = std::numeric_limits<float>::quiet_NaN();
    EXPECT_FALSE(kidi::model::TransformerImpl::validate_config(invalid));
    const auto malformed_type = kidi::model::validate_model_config(YAML::Load("{type: {nested: value}}"));
    ASSERT_FALSE(malformed_type);
    EXPECT_EQ(malformed_type.error().code, kidi::ErrorCode::INVALID_MANIFEST);
    auto encoder_memory = (*scratch)->encode_text(std::array{std::vector<std::int32_t>{1, 2}});
    ASSERT_TRUE(encoder_memory.has_value()) << encoder_memory.error().message;
    auto batched_state = (*scratch)->create_decoder_state(*encoder_memory, 2, 8);
    ASSERT_TRUE(batched_state.has_value()) << batched_state.error().message;
    const std::array<std::int32_t, 2> batched_tokens{1, 2};
    const auto single_forward = (*scratch)->forward(batched_tokens, **batched_state);
    ASSERT_FALSE(single_forward);
    EXPECT_EQ(single_forward.error().code, kidi::ErrorCode::INVALID_ARGUMENT);
}

TEST_F(PackageTest, RejectsDecodeSettings) {
    auto invalid_decode = YAML::Load(std::string(MANIFEST_JSON));
    invalid_decode["io"]["input_format"] = "plain";
    write(directory / "config.json", YAML::Dump(invalid_decode));
    auto translator = kidi::inference::Translator::load(directory, kidi::module_device);
    ASSERT_FALSE(translator);
    EXPECT_EQ(translator.error().code, kidi::ErrorCode::INVALID_MANIFEST);
    invalid_decode["io"]["input_format"] = "moses_tokenized";
    invalid_decode["decode"]["beam_size"] = 0;
    write(directory / "config.json", YAML::Dump(invalid_decode));
    translator = kidi::inference::Translator::load(directory, kidi::module_device);
    ASSERT_FALSE(translator);
    EXPECT_EQ(translator.error().code, kidi::ErrorCode::INVALID_ARGUMENT);
    invalid_decode["decode"]["beam_size"] = 1;
    invalid_decode["decode"]["length_penalty"] = std::numeric_limits<float>::quiet_NaN();
    write(directory / "config.json", YAML::Dump(invalid_decode));
    translator = kidi::inference::Translator::load(directory, kidi::module_device);
    ASSERT_FALSE(translator);
    EXPECT_EQ(translator.error().code, kidi::ErrorCode::INVALID_ARGUMENT);
}

TEST_F(PackageTest, DerivesSpecialIds) {
    const std::string original(TOKENIZER_JSON);
    auto reordered = original;
    const std::string vocabulary = R"("<pad>": 0, "<unk>": 1, "<s>": 2, "</s>": 3, "hi": 4)";
    reordered.replace(reordered.find(vocabulary), vocabulary.size(),
                      R"("<pad>": 4, "<unk>": 0, "<s>": 1, "</s>": 2, "hi": 3)");
    write(directory / "tokenizer.tgt.json", reordered);
    auto different_ids = kidi::checkpoint::Package::load(directory);
    ASSERT_TRUE(different_ids.has_value()) << different_ids.error().message;
    EXPECT_EQ(different_ids->config()["model"]["source_pad_id"].as<int>(), 0);
    EXPECT_EQ(different_ids->config()["decode"]["source_end_id"].as<int>(), 3);
    EXPECT_EQ(different_ids->config()["decode"]["begin_id"].as<int>(), 1);
    EXPECT_EQ(different_ids->config()["decode"]["end_id"].as<int>(), 2);
    EXPECT_EQ(different_ids->config()["decode"]["pad_id"].as<int>(), 4);
    auto missing = original;
    missing.replace(missing.find("<s>"), 3, "<missing>");
    write(directory / "tokenizer.tgt.json", missing);
    EXPECT_FALSE(kidi::checkpoint::Package::load(directory));
}

} // namespace
