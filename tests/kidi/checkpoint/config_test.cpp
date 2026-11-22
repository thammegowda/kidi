#include <chrono>
#include <filesystem>
#include <fstream>

#include <gtest/gtest.h>

#include "kidi/checkpoint/config.h"
#include "kidi/checkpoint/prepare.h"
#include "kidi/model/whisper.h"
#include "kidi/model/gemma4.h"

namespace {
auto write(const std::filesystem::path& path, std::string_view content = {}) -> void { std::ofstream(path) << content; }

struct FixtureModel {
    inline static int conversions = 0;

    static auto preparation(const YAML::Node& config) -> kidi::Result<kidi::checkpoint::Preparation> {
        return kidi::checkpoint::Preparation{
            .source = config["weights_file"].as<std::string>(),
            .cache_directory = config["model"]["cache"].as<std::string>(),
            .format = "fixture-v1",
            .files = {"config.json", "tokenizer.json", "preprocessor_config.json", "generation_config.json"},
            .metadata = {{"kind", "fixture"}},
            .convert = [](const YAML::Node& model,
                          const kidi::checkpoint::Weights& weights) -> kidi::Result<kidi::StateDict> {
                ++conversions;
                if (model["fail"].as<bool>(false)) return kidi::StateDict{};
                return weights.state_dict();
            },
        };
    }
};
const auto RTG_MANIFEST = R"({
      "model_type": "rtg_transformer_nmt",
      "weights_file": "model.safetensors",
      "tokenizers": {"source": "src.json", "target": "tgt.json.gz"},
      "io": {"input_format": "moses_tokenized", "output_format": "moses_tokenized"},
      "hidden_size": 768,
      "decode": {"beam_size": 4}
    })";

class ConfigTest : public ::testing::Test {
protected:
    void SetUp() override {
        directory = std::filesystem::temp_directory_path() /
                    (std::string("kidi-config-test-") + ::testing::UnitTest::GetInstance()->current_test_info()->name());
        std::filesystem::remove_all(directory);
        std::filesystem::create_directories(directory);
        write(directory / "model.safetensors");
        write(directory / "src.json");
        write(directory / "tgt.json.gz");
    }

    void TearDown() override {
        std::error_code error;
        std::filesystem::remove_all(directory, error);
    }

    // Files a Hugging Face checkpoint directory needs besides config.json and the weights.
    auto write_hub_files(bool sharded) const -> void {
        write(directory / "tokenizer.json");
        write(directory / "preprocessor_config.json");
        write(directory / "generation_config.json", "{}");
        if (sharded) write(directory / "model-00001-of-00001.safetensors");
    }

    std::filesystem::path directory;
};

TEST_F(ConfigTest, LoadsManifest) {
    write(directory / "config.json", YAML::Dump(YAML::Load(RTG_MANIFEST)));
    auto loaded = kidi::checkpoint::load_config(directory);
    ASSERT_TRUE(loaded.has_value()) << loaded.error().message;
    EXPECT_EQ((*loaded)["model"]["hidden_size"].as<int>(), 768);
    EXPECT_EQ((*loaded)["model"]["type"].as<std::string>(), "rtg_transformer_nmt");
    EXPECT_EQ((*loaded)["weights_file"].as<std::string>(), (directory / "model.safetensors").string());
    EXPECT_EQ((*loaded)["tokenizers"]["target"].as<std::string>(), (directory / "tgt.json.gz").string());
}

TEST_F(ConfigTest, RejectsInvalidManifestPaths) {
    const auto config = YAML::Load(RTG_MANIFEST);
    for (const auto* field : {"weights_file", "source", "target"}) {
        for (const auto* bad_path : {"../outside", "/outside", "missing"}) {
            SCOPED_TRACE(::testing::Message() << field << "=" << bad_path);
            auto invalid = YAML::Clone(config);
            if (std::string_view(field) == "weights_file")
                invalid[field] = bad_path;
            else
                invalid["tokenizers"][field] = bad_path;
            write(directory / "invalid.json", YAML::Dump(invalid));
            auto result = kidi::checkpoint::load_config(directory / "invalid.json");
            ASSERT_FALSE(result);
            EXPECT_EQ(result.error().code, kidi::ErrorCode::INVALID_MANIFEST);
        }
    }
}

TEST_F(ConfigTest, LoadsGemmaManifest) {
    YAML::Node gemma;
    gemma["model_type"] = "gemma4_text";
    gemma["weights_file"] = "model.safetensors";
    gemma["tokenizer_file"] = "src.json";
    gemma["hidden_size"] = 768;
    gemma["decode"]["beam_size"] = 1;
    write(directory / "gemma.json", YAML::Dump(gemma));
    auto gemma_config = kidi::checkpoint::load_config(directory / "gemma.json");
    ASSERT_TRUE(gemma_config.has_value()) << gemma_config.error().message;
    EXPECT_EQ((*gemma_config)["tokenizer_file"].as<std::string>(), (directory / "src.json").string());
    gemma["tokenizer_file"] = "../outside";
    write(directory / "invalid-gemma.json", YAML::Dump(gemma));
    EXPECT_FALSE(kidi::checkpoint::load_config(directory / "invalid-gemma.json"));
}

TEST_F(ConfigTest, LoadsOmniVoiceManifest) {
    YAML::Node omnivoice;
    omnivoice["model_type"] = "omnivoice";
    omnivoice["weights_file"] = "model.safetensors";
    omnivoice["tokenizer_file"] = "src.json";
    omnivoice["synthesis"]["steps"] = 32;
    omnivoice["license_files"].push_back("src.json");
    omnivoice["template_file"] = "src.json";
    omnivoice["sentence_prefixes_file"] = "src.json";
    write(directory / "omnivoice.json", YAML::Dump(omnivoice));
    auto omnivoice_config = kidi::checkpoint::load_config(directory / "omnivoice.json");
    ASSERT_TRUE(omnivoice_config.has_value()) << omnivoice_config.error().message;
    EXPECT_EQ((*omnivoice_config)["tokenizer_file"].as<std::string>(), (directory / "src.json").string());
    auto unlicensed_omnivoice = YAML::Clone(omnivoice);
    unlicensed_omnivoice.remove("license_files");
    write(directory / "unlicensed-omnivoice.json", YAML::Dump(unlicensed_omnivoice));
    EXPECT_FALSE(kidi::checkpoint::load_config(directory / "unlicensed-omnivoice.json"));
    omnivoice.remove("synthesis");
    write(directory / "invalid-omnivoice.json", YAML::Dump(omnivoice));
    auto missing_synthesis = kidi::checkpoint::load_config(directory / "invalid-omnivoice.json");
    ASSERT_FALSE(missing_synthesis);
    EXPECT_EQ(missing_synthesis.error().code, kidi::ErrorCode::INVALID_MANIFEST);
    EXPECT_EQ(missing_synthesis.error().message.find("invalid node"), std::string::npos)
        << missing_synthesis.error().message;
}

TEST_F(ConfigTest, LegacyOmniVoice) {
    const auto legacy_dir = directory / "legacy-omnivoice";
    std::filesystem::create_directories(legacy_dir / "licenses");
    write(legacy_dir / "model.safetensors");
    write(legacy_dir / "tokenizer.json.gz");
    write(legacy_dir / "sentence-prefixes.tsv");
    write(legacy_dir / "synthesis_template.jinja");
    write(legacy_dir / "NOTICE");
    write(legacy_dir / "licenses/OMNIVOICE-CC-BY-NC-4.0.txt");
    write(legacy_dir / "licenses/META-LLAMA-3-LICENSE.txt");
    write(legacy_dir / "licenses/META-LLAMA-3-USE-POLICY.md");
    write(legacy_dir / "licenses/HIGGS-AUDIO-2-LICENSE.txt");
    write(legacy_dir / "config.json", R"({
      "name": "OmniVoice-q8",
      "model_type": "kidi_omnivoice",
      "architectures": ["KidiOmniVoiceForTextToSpeech"]
    })");
    YAML::Node legacy_manifest;
    legacy_manifest["weights_file"] = "model.safetensors";
    legacy_manifest["tokenizer_file"] = "tokenizer.json.gz";
    legacy_manifest["template_file"] = "synthesis_template.jinja";
    legacy_manifest["sentence_prefixes_file"] = "sentence-prefixes.tsv";
    legacy_manifest["license_files"].push_back("NOTICE");
    legacy_manifest["model"]["type"] = "omnivoice";
    legacy_manifest["synthesis"]["steps"] = 32;
    write(legacy_dir / "model.yaml", YAML::Dump(legacy_manifest));
    const auto legacy_hub = kidi::checkpoint::hub_files(YAML::LoadFile((legacy_dir / "config.json").string()));
    ASSERT_TRUE(legacy_hub.has_value()) << legacy_hub.error().message;
    EXPECT_TRUE(std::ranges::contains(*legacy_hub, "model.yaml"));
    const auto legacy_config = kidi::checkpoint::load_config(legacy_dir);
    ASSERT_TRUE(legacy_config.has_value()) << legacy_config.error().message;
    EXPECT_EQ((*legacy_config)["model"]["type"].as<std::string>(), "omnivoice");
    EXPECT_EQ((*legacy_config)["tokenizer_file"].as<std::string>(), (legacy_dir / "tokenizer.json.gz").string());
    EXPECT_EQ((*legacy_config)["synthesis"]["steps"].as<int>(), 32);
}

TEST_F(ConfigTest, LoadsKokoroManifest) {
    YAML::Node kokoro;
    kokoro["model_type"] = "kokoro";
    kokoro["weights_file"] = "model.safetensors";
    kokoro["synthesis"]["default_voice"] = "af_heart";
    kokoro["lexicon_file"] = "src.json";
    kokoro["sentence_prefixes_file"] = "src.json";
    kokoro["license_files"].push_back("src.json");
    write(directory / "kokoro.json", YAML::Dump(kokoro));
    auto kokoro_config = kidi::checkpoint::load_config(directory / "kokoro.json");
    ASSERT_TRUE(kokoro_config.has_value()) << kokoro_config.error().message;
    EXPECT_EQ((*kokoro_config)["lexicon_file"].as<std::string>(), (directory / "src.json").string());
    kokoro["lexicon_file"] = "../outside";
    write(directory / "invalid-kokoro.json", YAML::Dump(kokoro));
    EXPECT_FALSE(kidi::checkpoint::load_config(directory / "invalid-kokoro.json"));
}

TEST_F(ConfigTest, ImportsGemmaHubConfig) {
    write_hub_files(false);
    for (const int hidden : {1536, 2560})
        for (const bool quantized : {false, true}) {
            YAML::Node document;
            document["model_type"] = "gemma4";
            document["text_config"]["model_type"] = "gemma4_text";
            document["text_config"]["hidden_size"] = hidden;
            document["text_config"]["max_position_embeddings"] = 131072;
            document["tie_word_embeddings"] = true;
            if (quantized) {
                document["quantization_config"]["quant_method"] = "gemma";
                document["quantization_config"]["num_bits"] = 4;
            } else
                document["quantization_config"] = YAML::Null;
            write(directory / "config.json", YAML::Dump(document));
            const auto imported = kidi::checkpoint::load_config(directory);
            SCOPED_TRACE(::testing::Message() << "Gemma runtime import failed for hidden=" << hidden
                                              << " quantized=" << quantized << ": "
                                              << (imported ? YAML::Dump(*imported) : imported.error().message));
            ASSERT_TRUE(imported.has_value());
            EXPECT_EQ((*imported)["model"]["type"].as<std::string>(), "gemma4_text");
            EXPECT_EQ((*imported)["model"]["hidden_size"].as<int>(), hidden);
            EXPECT_TRUE((*imported)["model"]["tie_word_embeddings"].as<bool>());
            EXPECT_EQ(static_cast<bool>((*imported)["model"]["quantization_config"]), quantized);
            EXPECT_EQ((*imported)["weights_file"].as<std::string>(), (directory / "model.safetensors").string());
        }
    write(directory / "config.json",
          R"({"model_type":"gemma4_text","hidden_size":2048,"tie_word_embeddings":true,"quantization_config":null})");
    const auto text_only = kidi::checkpoint::load_config(directory);
    ASSERT_TRUE(text_only.has_value()) << "Gemma text config import failed: " << text_only.error().message;
    EXPECT_EQ((*text_only)["model"]["hidden_size"].as<int>(), 2048)
        << "Gemma text config import failed: " << YAML::Dump(*text_only);
    EXPECT_FALSE((*text_only)["model"]["quantization_config"])
        << "Gemma text config import failed: " << YAML::Dump(*text_only);
}

TEST_F(ConfigTest, ImportsInklingHubConfig) {
    write_hub_files(false);
    write(directory / "config.json", R"({
      "model_type":"inkling2_loop", "dtype":"bfloat16", "eos_token_id":200006,
      "quantization_config":{"quant_method":"modelopt","quant_algo":"NVFP4"},
      "text_config":{"model_max_length":1048576,"eos_token_id":1,"torch_dtype":"bfloat16"}
    })");
    write(directory / "model-00001-of-00001.safetensors");
    write(directory / "generation_config.json", R"({"eos_token_id":[200006,200007],"pad_token_id":199999})");
    auto inkling = kidi::checkpoint::load_config(directory);
    ASSERT_TRUE(inkling.has_value()) << inkling.error().message;
    EXPECT_EQ((*inkling)["model"]["dtype"].as<std::string>(), "bfloat16");
    EXPECT_EQ((*inkling)["model"]["max_position_embeddings"].as<int>(), 1048576);
    EXPECT_EQ((*inkling)["model"]["quantization_config"]["quant_algo"].as<std::string>(), "NVFP4");
    EXPECT_FALSE(kidi::checkpoint::native_qat((*inkling)["model"]));
    EXPECT_EQ((*inkling)["generation"]["eos_token_id"].as<std::vector<int>>(), (std::vector<int>{200006, 200007}));
    EXPECT_EQ((*inkling)["generation"]["pad_token_id"].as<int>(), 199999);
    EXPECT_FALSE((*inkling)["special_tokens"]);
    EXPECT_FALSE((*inkling)["parameter_tensor"]);
    std::filesystem::remove(directory / "generation_config.json");
    inkling = kidi::checkpoint::load_config(directory);
    ASSERT_TRUE(inkling.has_value()) << inkling.error().message;
    EXPECT_EQ((*inkling)["generation"]["eos_token_id"].as<int>(), 200006);
    write(directory / "generation_config.json", "[]");
    const auto malformed_generation = kidi::checkpoint::load_config(directory);
    ASSERT_FALSE(malformed_generation);
    EXPECT_EQ(malformed_generation.error().code, kidi::ErrorCode::INVALID_MANIFEST);
}

TEST_F(ConfigTest, ImportsGemmaConfig) {
    write_hub_files(true);
    write(directory / "config.json", R"({
      "model_type":"gemma4", "dtype":"bfloat16", "eos_token_id":[1,106],
      "quantization_config":{"quant_method":"gemma","num_bits":4},
      "text_config":{"eos_token_id":1,"dtype":null,"quantization_config":null}
    })");
    const auto gemma_generation = kidi::checkpoint::load_config(directory);
    ASSERT_TRUE(gemma_generation.has_value()) << gemma_generation.error().message;
    EXPECT_TRUE(kidi::checkpoint::native_qat((*gemma_generation)["model"]));
    EXPECT_EQ((*gemma_generation)["model"]["dtype"].as<std::string>(), "bfloat16");
    EXPECT_EQ((*gemma_generation)["generation"]["eos_token_id"].as<std::vector<int>>(), (std::vector<int>{1, 106}));
    const auto chat_files = kidi::checkpoint::hub_files(YAML::LoadFile((directory / "config.json").string()));
    ASSERT_TRUE(chat_files.has_value()) << chat_files.error().message;
    EXPECT_TRUE(std::ranges::contains(*chat_files, "generation_config.json"));
    auto relocated_tokenizer = YAML::LoadFile((directory / "config.json").string());
    relocated_tokenizer["tokenizer_file"] = "text/tokenizer.json";
    const auto relocated_files = kidi::checkpoint::hub_files(relocated_tokenizer);
    ASSERT_TRUE(relocated_files.has_value()) << relocated_files.error().message;
    EXPECT_TRUE(std::ranges::contains(*relocated_files, "generation_config.json"));
    EXPECT_FALSE(std::ranges::contains(*relocated_files, "text/generation_config.json"));
    EXPECT_TRUE(std::ranges::contains(*relocated_files, "text/tokenizer_config.json"));
}

TEST_F(ConfigTest, ImportsWhisperHubConfig) {
    write_hub_files(true);
    write(directory / "config.json", R"({"model_type":"whisper","d_model":384})");
    auto whisper = kidi::checkpoint::load_config(directory);
    ASSERT_TRUE(whisper.has_value()) << whisper.error().message;
    EXPECT_EQ((*whisper)["model"]["type"].as<std::string>(), "whisper");
    EXPECT_EQ((*whisper)["model"]["d_model"].as<int>(), 384);
    EXPECT_EQ((*whisper)["weights_file"].as<std::string>(), (directory / "model.safetensors").string());
    EXPECT_EQ((*whisper)["tokenizer_file"].as<std::string>(), (directory / "tokenizer.json").string());
    std::filesystem::remove(directory / "generation_config.json");
    EXPECT_FALSE(kidi::checkpoint::load_config(directory));
}

TEST_F(ConfigTest, RejectsMalformedConfigs) {
    write(directory / "missing-type.json", R"({"hidden_size": 768})");
    EXPECT_FALSE(kidi::checkpoint::load_config(directory / "missing-type.json"));
    write(directory / "malformed.json", "{\"model_type\": [");
    EXPECT_FALSE(kidi::checkpoint::load_config(directory / "malformed.json"));
}

TEST_F(ConfigTest, PreparesCachedCheckpoint) {
    write_hub_files(true);
    FixtureModel::conversions = 0;
    write(directory / "generation_config.json");
    auto tensor = kidi::tensor::Tensor::zeros({1}, kidi::tensor::DType::F32, kidi::tensor::Device::cpu());
    const auto source = directory / "model.safetensors";
    std::filesystem::remove(source);
    ASSERT_TRUE(tensor.has_value()) << tensor.error().message;
    ASSERT_TRUE(kidi::checkpoint::Weights::save(source, {{"weight", *tensor}}));
    write(directory / "config.json", R"({"model_type":"whisper","cache":"prepared-fixture"})");
    const auto prepare_fixture = [&] { return kidi::checkpoint::prepare(directory, FixtureModel::preparation); };
    auto prepared = prepare_fixture();
    ASSERT_TRUE(prepared.has_value()) << prepared.error().message;
    EXPECT_EQ(*prepared, directory / "prepared-fixture");
    EXPECT_EQ(FixtureModel::conversions, 1);
    EXPECT_TRUE(kidi::checkpoint::Weights::load(*prepared / "model.safetensors"));
    EXPECT_TRUE(std::filesystem::is_regular_file(*prepared / "config.json"));
    EXPECT_TRUE(prepare_fixture());
    EXPECT_EQ(FixtureModel::conversions, 1);
    std::filesystem::last_write_time(source, std::filesystem::last_write_time(source) + std::chrono::seconds(1));
    EXPECT_TRUE(prepare_fixture());
    EXPECT_EQ(FixtureModel::conversions, 2);
    write(directory / "config.json", R"({"model_type":"whisper","cache":"failed-fixture","fail":true})");
    EXPECT_FALSE(prepare_fixture());
    EXPECT_FALSE(std::filesystem::exists(directory / "failed-fixture"));
    EXPECT_EQ(FixtureModel::conversions, 3);
    for (const auto& entry : std::filesystem::directory_iterator(directory))
        EXPECT_FALSE(entry.path().filename().string().starts_with(".kidi-checkpoint-")) << entry.path();
}

} // namespace
