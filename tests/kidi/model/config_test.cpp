#include <filesystem>
#include <fstream>
#include <iostream>

#include "kidi/model/config.h"

namespace {
auto write(const std::filesystem::path& path, std::string_view content = {}) -> void { std::ofstream(path) << content; }
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
    auto loaded = kidi::model::load_config(directory / "model.yaml");
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
            auto result = kidi::model::load_config(directory / "invalid.yaml");
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
    auto gemma_config = kidi::model::load_config(directory / "gemma.yaml");
    if (!gemma_config || (*gemma_config)["tokenizer_file"].as<std::string>() != (directory / "src.json").string())
        return 1;
    gemma["tokenizer_file"] = "../outside";
    write(directory / "invalid-gemma.yaml", YAML::Dump(gemma));
    if (kidi::model::load_config(directory / "invalid-gemma.yaml")) return 1;
    old_format.remove("model");
    old_format["architecture"]["hidden_size"] = 768;
    write(directory / "old.yaml", YAML::Dump(old_format));
    if (kidi::model::load_config(directory / "old.yaml")) return 1;
    write(directory / "malformed.yaml", "model: [");
    if (kidi::model::load_config(directory / "malformed.yaml")) return 1;
    std::filesystem::remove_all(directory);
    return 0;
}