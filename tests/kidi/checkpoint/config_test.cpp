#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>

#include "kidi/checkpoint/config.h"
#include "kidi/checkpoint/prepare.h"
#include "kidi/model/whisper.h"

namespace {
auto write(const std::filesystem::path& path, std::string_view content = {}) -> void { std::ofstream(path) << content; }

struct FixtureModel {
    inline static int conversions = 0;

    static auto checkpoint_config() -> kidi::checkpoint::ConfigAdapter {
        return {"fixture.json", [](const kidi::checkpoint::ConfigSource& source) -> kidi::Result<YAML::Node> {
                    constexpr std::array<std::string_view, 1> candidates{"source.safetensors"};
                    auto weights = source.weights(candidates);
                    if (!weights) return std::unexpected(weights.error());
                    YAML::Node config;
                    config["model"] = source.document;
                    config["source"] = (*weights)["path"];
                    return config;
                }};
    }

    static auto preparation(const YAML::Node& config) -> kidi::Result<kidi::checkpoint::Preparation> {
        return kidi::checkpoint::Preparation{
            .source = config["source"].as<std::string>(),
            .cache_directory = config["model"]["cache"].as<std::string>(),
            .format = "fixture-v1",
            .files = {"fixture.json"},
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
    const auto config = YAML::Load(R"(
format_version: 1
weights_file: model.safetensors
tokenizers: {source: src.json, target: tgt.json.gz}
io: {input_format: moses_tokenized, output_format: moses_tokenized}
model: {type: rtg_transformer_nmt, hidden_size: 768}
decode: {beam_size: 4}
)");
    write(directory / "model.yaml", YAML::Dump(config));
    auto loaded = kidi::checkpoint::load_config(directory / "model.yaml");
    if (!loaded || (*loaded)["model"]["hidden_size"].as<int>() != 768 ||
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
            write(directory / "invalid.yaml", YAML::Dump(invalid));
            auto result = kidi::checkpoint::load_config(directory / "invalid.yaml");
            if (result || result.error().code != kidi::ErrorCode::INVALID_MANIFEST) return 1;
        }
    }
    auto old_format = YAML::Clone(config);
    auto gemma = YAML::Clone(config);
    gemma["model"]["type"] = "gemma4_text";
    gemma.remove("io");
    gemma.remove("tokenizers");
    gemma["tokenizer_file"] = "src.json";
    write(directory / "gemma.yaml", YAML::Dump(gemma));
    auto gemma_config = kidi::checkpoint::load_config(directory / "gemma.yaml");
    if (!gemma_config || (*gemma_config)["tokenizer_file"].as<std::string>() != (directory / "src.json").string())
        return 1;
    gemma["tokenizer_file"] = "../outside";
    write(directory / "invalid-gemma.yaml", YAML::Dump(gemma));
    if (kidi::checkpoint::load_config(directory / "invalid-gemma.yaml")) return 1;
    write(directory / "config.json", R"({"model_type":"whisper","d_model":384})");
    write(directory / "tokenizer.json");
    write(directory / "preprocessor_config.json");
    write(directory / "generation_config.json");
    const auto adapter = kidi::model::WhisperImpl::checkpoint_config();
    auto whisper = kidi::checkpoint::load_config(directory, adapter);
    if (!whisper || (*whisper)["model"]["type"].as<std::string>() != "whisper" ||
        (*whisper)["model"]["d_model"].as<int>() != 384 ||
        (*whisper)["model_file"].as<std::string>() != (directory / "model.safetensors").string() ||
        (*whisper)["tokenizer_file"].as<std::string>() != (directory / "tokenizer.json").string())
        return 1;
    std::filesystem::remove(directory / "generation_config.json");
    if (kidi::checkpoint::load_config(directory, adapter)) return 1;
    old_format.remove("model");
    old_format["architecture"]["hidden_size"] = 768;
    write(directory / "old.yaml", YAML::Dump(old_format));
    if (kidi::checkpoint::load_config(directory / "old.yaml")) return 1;
    write(directory / "malformed.yaml", "model: [");
    if (kidi::checkpoint::load_config(directory / "malformed.yaml")) return 1;

    auto tensor = kidi::tensor::Tensor::zeros({1}, kidi::tensor::DType::F32, kidi::tensor::Device::cpu());
    const auto source = directory / "source.safetensors";
    if (!tensor || !kidi::checkpoint::Weights::save(source, {{"weight", *tensor}})) return 1;
    write(directory / "fixture.json", R"({"cache":"prepared-fixture"})");
    const auto prepare_fixture = [&] {
        return kidi::checkpoint::prepare(directory, FixtureModel::checkpoint_config(), FixtureModel::preparation);
    };
    auto prepared = prepare_fixture();
    if (!prepared || *prepared != directory / "prepared-fixture" || FixtureModel::conversions != 1 ||
        !kidi::checkpoint::Weights::load(*prepared / "model.safetensors") ||
        !std::filesystem::is_regular_file(*prepared / "fixture.json"))
        return 1;
    if (!prepare_fixture() || FixtureModel::conversions != 1) return 1;
    std::filesystem::last_write_time(source, std::filesystem::last_write_time(source) + std::chrono::seconds(1));
    if (!prepare_fixture() || FixtureModel::conversions != 2) return 1;
    write(directory / "fixture.json", R"({"cache":"failed-fixture","fail":true})");
    if (prepare_fixture() || std::filesystem::exists(directory / "failed-fixture") || FixtureModel::conversions != 3)
        return 1;
    for (const auto& entry : std::filesystem::directory_iterator(directory))
        if (entry.path().filename().string().starts_with(".kidi-checkpoint-")) return 1;
    std::filesystem::remove_all(directory);
    return 0;
}