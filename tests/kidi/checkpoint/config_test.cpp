#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>

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
} // namespace

auto main() -> int {
    const auto directory = std::filesystem::temp_directory_path() / "kidi-config-test";
    std::filesystem::remove_all(directory);
    std::filesystem::create_directories(directory);
    write(directory / "model.safetensors");
    write(directory / "src.json");
    write(directory / "tgt.json.gz");
    const auto config = YAML::Load(R"({
      "model_type": "rtg_transformer_nmt",
      "weights_file": "model.safetensors",
      "tokenizers": {"source": "src.json", "target": "tgt.json.gz"},
      "io": {"input_format": "moses_tokenized", "output_format": "moses_tokenized"},
      "hidden_size": 768,
      "decode": {"beam_size": 4}
    })");
    write(directory / "config.json", YAML::Dump(config));
    auto loaded = kidi::checkpoint::load_config(directory);
    if (!loaded || (*loaded)["model"]["hidden_size"].as<int>() != 768 ||
        (*loaded)["model"]["type"].as<std::string>() != "rtg_transformer_nmt" ||
        (*loaded)["weights_file"].as<std::string>() != (directory / "model.safetensors").string() ||
        (*loaded)["tokenizers"]["target"].as<std::string>() != (directory / "tgt.json.gz").string())
        return 1;
    for (const auto* field : {"weights_file", "source", "target"}) {
        for (const auto* bad_path : {"../outside", "/outside", "missing"}) {
            auto invalid = YAML::Clone(config);
            if (std::string_view(field) == "weights_file")
                invalid[field] = bad_path;
            else
                invalid["tokenizers"][field] = bad_path;
            write(directory / "invalid.json", YAML::Dump(invalid));
            auto result = kidi::checkpoint::load_config(directory / "invalid.json");
            if (result || result.error().code != kidi::ErrorCode::INVALID_MANIFEST) return 1;
        }
    }
    YAML::Node gemma;
    gemma["model_type"] = "gemma4_text";
    gemma["weights_file"] = "model.safetensors";
    gemma["tokenizer_file"] = "src.json";
    gemma["hidden_size"] = 768;
    gemma["decode"]["beam_size"] = 1;
    write(directory / "gemma.json", YAML::Dump(gemma));
    auto gemma_config = kidi::checkpoint::load_config(directory / "gemma.json");
    if (!gemma_config || (*gemma_config)["tokenizer_file"].as<std::string>() != (directory / "src.json").string())
        return 1;
    gemma["tokenizer_file"] = "../outside";
    write(directory / "invalid-gemma.json", YAML::Dump(gemma));
    if (kidi::checkpoint::load_config(directory / "invalid-gemma.json")) return 1;
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
    if (!omnivoice_config ||
        (*omnivoice_config)["tokenizer_file"].as<std::string>() != (directory / "src.json").string())
        return 1;
    auto unlicensed_omnivoice = YAML::Clone(omnivoice);
    unlicensed_omnivoice.remove("license_files");
    write(directory / "unlicensed-omnivoice.json", YAML::Dump(unlicensed_omnivoice));
    if (kidi::checkpoint::load_config(directory / "unlicensed-omnivoice.json")) return 1;
    omnivoice.remove("synthesis");
    write(directory / "invalid-omnivoice.json", YAML::Dump(omnivoice));
    auto missing_synthesis = kidi::checkpoint::load_config(directory / "invalid-omnivoice.json");
    if (missing_synthesis || missing_synthesis.error().code != kidi::ErrorCode::INVALID_MANIFEST ||
        missing_synthesis.error().message.find("invalid node") != std::string::npos)
        return 1;
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
    if (!legacy_hub || !std::ranges::contains(*legacy_hub, "model.yaml")) return 1;
    const auto legacy_config = kidi::checkpoint::load_config(legacy_dir);
    if (!legacy_config || (*legacy_config)["model"]["type"].as<std::string>() != "omnivoice" ||
        (*legacy_config)["tokenizer_file"].as<std::string>() != (legacy_dir / "tokenizer.json.gz").string() ||
        (*legacy_config)["synthesis"]["steps"].as<int>() != 32)
        return 1;
    YAML::Node kokoro;
    kokoro["model_type"] = "kokoro";
    kokoro["weights_file"] = "model.safetensors";
    kokoro["synthesis"]["default_voice"] = "af_heart";
    kokoro["lexicon_file"] = "src.json";
    kokoro["sentence_prefixes_file"] = "src.json";
    kokoro["license_files"].push_back("src.json");
    write(directory / "kokoro.json", YAML::Dump(kokoro));
    auto kokoro_config = kidi::checkpoint::load_config(directory / "kokoro.json");
    if (!kokoro_config || (*kokoro_config)["lexicon_file"].as<std::string>() != (directory / "src.json").string())
        return 1;
    kokoro["lexicon_file"] = "../outside";
    write(directory / "invalid-kokoro.json", YAML::Dump(kokoro));
    if (kidi::checkpoint::load_config(directory / "invalid-kokoro.json")) return 1;
    write(directory / "tokenizer.json");
    write(directory / "preprocessor_config.json");
    write(directory / "generation_config.json");
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
            if (!imported || (*imported)["model"]["type"].as<std::string>() != "gemma4_text" ||
                (*imported)["model"]["hidden_size"].as<int>() != hidden ||
                !(*imported)["model"]["tie_word_embeddings"].as<bool>() ||
                static_cast<bool>((*imported)["model"]["quantization_config"]) != quantized ||
                (*imported)["weights_file"].as<std::string>() != (directory / "model.safetensors").string()) {
                std::cerr << "Gemma runtime import failed for hidden=" << hidden << " quantized=" << quantized << ": "
                          << (imported ? YAML::Dump(*imported) : imported.error().message) << '\n';
                return 1;
            }
        }
    write(directory / "config.json",
          R"({"model_type":"gemma4_text","hidden_size":2048,"tie_word_embeddings":true,"quantization_config":null})");
    const auto text_only = kidi::checkpoint::load_config(directory);
    if (!text_only || (*text_only)["model"]["hidden_size"].as<int>() != 2048 ||
        (*text_only)["model"]["quantization_config"]) {
        std::cerr << "Gemma text config import failed: "
                  << (text_only ? YAML::Dump(*text_only) : text_only.error().message) << '\n';
        return 1;
    }
    write(directory / "config.json", R"({"model_type":"whisper","d_model":384})");
    auto whisper = kidi::checkpoint::load_config(directory);
    if (!whisper || (*whisper)["model"]["type"].as<std::string>() != "whisper" ||
        (*whisper)["model"]["d_model"].as<int>() != 384 ||
        (*whisper)["weights_file"].as<std::string>() != (directory / "model.safetensors").string() ||
        (*whisper)["tokenizer_file"].as<std::string>() != (directory / "tokenizer.json").string())
        return 1;
    std::filesystem::remove(directory / "generation_config.json");
    if (kidi::checkpoint::load_config(directory)) return 1;
    write(directory / "missing-type.json", R"({"hidden_size": 768})");
    if (kidi::checkpoint::load_config(directory / "missing-type.json")) return 1;
    write(directory / "malformed.json", "{\"model_type\": [");
    if (kidi::checkpoint::load_config(directory / "malformed.json")) return 1;

    write(directory / "generation_config.json");
    auto tensor = kidi::tensor::Tensor::zeros({1}, kidi::tensor::DType::F32, kidi::tensor::Device::cpu());
    const auto source = directory / "model.safetensors";
    std::filesystem::remove(source);
    if (!tensor || !kidi::checkpoint::Weights::save(source, {{"weight", *tensor}})) return 1;
    write(directory / "config.json", R"({"model_type":"whisper","cache":"prepared-fixture"})");
    const auto prepare_fixture = [&] { return kidi::checkpoint::prepare(directory, FixtureModel::preparation); };
    auto prepared = prepare_fixture();
    if (!prepared || *prepared != directory / "prepared-fixture" || FixtureModel::conversions != 1 ||
        !kidi::checkpoint::Weights::load(*prepared / "model.safetensors") ||
        !std::filesystem::is_regular_file(*prepared / "config.json"))
        return 1;
    if (!prepare_fixture() || FixtureModel::conversions != 1) return 1;
    std::filesystem::last_write_time(source, std::filesystem::last_write_time(source) + std::chrono::seconds(1));
    if (!prepare_fixture() || FixtureModel::conversions != 2) return 1;
    write(directory / "config.json", R"({"model_type":"whisper","cache":"failed-fixture","fail":true})");
    if (prepare_fixture() || std::filesystem::exists(directory / "failed-fixture") || FixtureModel::conversions != 3)
        return 1;
    for (const auto& entry : std::filesystem::directory_iterator(directory))
        if (entry.path().filename().string().starts_with(".kidi-checkpoint-")) return 1;
    std::filesystem::remove_all(directory);
    return 0;
}