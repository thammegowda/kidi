#include "kidi/checkpoint/config.h"
// Imports model and generation metadata from checkpoint files, independently of the inference backend.
#include "kidi/checkpoint/ggml/reader.h"
#include "packages_jsonl.h"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <span>
#include <string>
#include <string_view>

namespace kidi::checkpoint {
namespace {

struct PackageSpec {
    std::string model_type;
    std::string type;
    std::string task;
    std::string model_key;
    std::string fallback;
    std::vector<std::string> requires_keys;
    std::vector<std::string> copy_keys;
    std::vector<std::string> files;
    std::vector<std::string> sidecars;
};

auto string_list(const YAML::Node& node) -> std::vector<std::string> {
    std::vector<std::string> result;
    if (!node || !node.IsSequence()) return result;
    for (const auto& entry : node)
        if (entry.IsScalar()) result.push_back(entry.as<std::string>());
    return result;
}

auto package_specs() -> const std::vector<PackageSpec>& {
    static const std::vector<PackageSpec> specs = [] {
        std::vector<PackageSpec> parsed;
        std::istringstream stream{std::string(KIDI_PACKAGE_SPECS_JSONL)};
        std::string line;
        while (std::getline(stream, line)) {
            if (line.empty() || line.find_first_not_of(" \t\r") == std::string::npos) continue;
            const auto node = YAML::Load(line);
            PackageSpec spec;
            spec.model_type = node["model_type"].as<std::string>();
            spec.type = node["type"] ? node["type"].as<std::string>() : spec.model_type;
            if (node["task"]) spec.task = node["task"].as<std::string>();
            if (node["model_key"]) spec.model_key = node["model_key"].as<std::string>();
            if (node["fallback"]) spec.fallback = node["fallback"].as<std::string>();
            spec.requires_keys = string_list(node["requires"]);
            spec.copy_keys = string_list(node["copy"]);
            spec.files = string_list(node["files"]);
            spec.sidecars = string_list(node["sidecars"]);
            parsed.push_back(std::move(spec));
        }
        return parsed;
    }();
    return specs;
}

auto find_package_spec(std::string_view type) -> const PackageSpec* {
    for (const auto& spec : package_specs())
        if (spec.model_type == type) return &spec;
    return nullptr;
}

auto is_weight_file(std::string_view name) -> bool { return name.ends_with(".safetensors") || name.ends_with(".bin"); }

auto config_key_for(std::string_view filename) -> std::string {
    const auto base = std::filesystem::path(filename).filename().string();
    if (base == "tokenizer.json" || base == "tokenizer.json.gz") return "tokenizer_file";
    if (base.ends_with(".jinja")) return "template_file";
    if (base == "sentence-prefixes.tsv") return "sentence_prefixes_file";
    if (base.ends_with(".tsv")) return "lexicon_file";
    return std::filesystem::path(base).stem().string() + "_file";
}

auto valid_relative_path(std::string_view value) -> bool {
    constexpr std::string_view invalid_chars{"\\:*?[]\0", 7};
    if (value.empty() || value.find_first_of(invalid_chars) != std::string_view::npos) return false;
    const auto path = std::filesystem::path(value);
    if (path.is_absolute()) return false;
    return std::ranges::none_of(path, [](const auto& part) { return part.empty() || part == ".."; });
}

auto resolve_file(const std::filesystem::path& directory, std::string_view name, bool required = true)
    -> Result<std::filesystem::path> {
    if (!valid_relative_path(name))
        return std::unexpected(Error{ErrorCode::INVALID_MANIFEST, "package files must use safe relative paths"});
    const auto resolved = (directory / std::filesystem::path(name)).lexically_normal();
    const auto relative = resolved.lexically_relative(directory);
    if (relative.empty() || *relative.begin() == "..")
        return std::unexpected(Error{ErrorCode::INVALID_MANIFEST, "file path escapes the model package"});
    if (required && !std::filesystem::is_regular_file(resolved))
        return std::unexpected(Error{ErrorCode::INVALID_MANIFEST, "package file does not exist: " + resolved.string()});
    return resolved;
}

auto string_or(const YAML::Node& node, std::string_view key, std::string_view fallback) -> Result<std::string> {
    const auto value = node[std::string(key)];
    if (!value || value.IsNull()) return std::string(fallback);
    if (!value.IsScalar())
        return std::unexpected(Error{ErrorCode::INVALID_MANIFEST, std::string(key) + " must be a string"});
    return value.as<std::string>();
}

auto required_string(const YAML::Node& node, std::string_view key) -> Result<std::string> {
    const auto value = node[std::string(key)];
    if (!value || !value.IsScalar())
        return std::unexpected(Error{ErrorCode::INVALID_MANIFEST, std::string(key) + " must be a string"});
    return value.as<std::string>();
}

auto resolve_weights(const std::filesystem::path& directory, const std::filesystem::path& explicit_weights,
                     std::span<const std::string> candidates) -> Result<YAML::Node> {
    auto source = explicit_weights;
    if (source.empty()) {
        for (const auto& candidate : candidates) {
            auto resolved = resolve_file(directory, candidate, false);
            if (!resolved) return std::unexpected(resolved.error());
            if (std::filesystem::is_regular_file(*resolved)) {
                source = *resolved;
                break;
            }
        }
    }
    if (source.empty() || !std::filesystem::is_regular_file(source))
        return std::unexpected(
            Error{ErrorCode::INVALID_MANIFEST, "checkpoint weights not found in " + directory.string()});
    YAML::Node result;
    result["path"] = source.string();
    std::ifstream stream(source, std::ios::binary);
    std::uint32_t magic = 0;
    stream.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    result["format"] = magic == ggml::GGML_MAGIC ? "ggml" : magic == ggml::GGUF_MAGIC ? "gguf" : "safetensors";
    if (magic == ggml::GGML_MAGIC) {
        auto imported = ggml::GgmlFile::open(source);
        if (!imported) return std::unexpected(imported.error());
        for (const auto value : *imported->whisper_header()) result["ggml_header"].push_back(value);
    }
    return result;
}

auto push_checked(std::vector<std::string>& out, std::string value) -> Result<void> {
    if (!valid_relative_path(value))
        return std::unexpected(Error{ErrorCode::INVALID_MANIFEST, "invalid package file path: " + value});
    if (!std::ranges::contains(out, value)) out.push_back(std::move(value));
    return {};
}

auto missing_required(const PackageSpec& spec, const YAML::Node& manifest) -> bool {
    return std::ranges::any_of(spec.requires_keys, [&](const auto& key) { return !manifest[key]; });
}

auto check_required(const PackageSpec& spec, const YAML::Node& manifest) -> Result<void> {
    for (const auto& key : spec.requires_keys) {
        const auto node = manifest[key];
        if (key == "license_files") {
            if (!node || !node.IsSequence() || !node.size())
                return std::unexpected(Error{ErrorCode::INVALID_MANIFEST, spec.type + " requires " + key});
        } else if (!node || !node.IsMap()) {
            return std::unexpected(Error{ErrorCode::INVALID_MANIFEST, spec.type + " requires " + key});
        }
    }
    return {};
}

} // namespace

auto native_qat(const YAML::Node& model_config) -> bool {
    const auto quantization = model_config["quantization_config"];
    return quantization && quantization.IsMap() && quantization["quant_method"].as<std::string>("") == "gemma";
}

auto hub_files(const YAML::Node& document) -> Result<std::vector<std::string>> {
    try {
        if (!document.IsMap())
            return std::unexpected(Error{ErrorCode::INVALID_MANIFEST, "config.json must contain a JSON object"});
        const auto type = document["model_type"].as<std::string>("");
        const auto* spec = find_package_spec(type);
        if (!spec)
            return std::unexpected(
                Error{ErrorCode::UNSUPPORTED, "Automatic setup supports packages listed in packages.jsonl only"});
        const bool fallback = !spec->fallback.empty() && missing_required(*spec, document);
        if (!fallback) {
            if (auto ok = check_required(*spec, document); !ok) return std::unexpected(ok.error());
        }

        std::vector<std::string> files{"config.json"};
        bool added_weights = false;
        if (document["weights_file"] && !document["weights_file"].IsNull()) {
            auto value = required_string(document, "weights_file");
            if (!value) return std::unexpected(value.error());
            if (auto ok = push_checked(files, *value); !ok) return std::unexpected(ok.error());
            added_weights = true;
        }
        std::filesystem::path tokenizer_parent;
        for (const auto& entry : spec->files) {
            if (is_weight_file(entry)) {
                if (added_weights) continue;
                added_weights = true;
            }
            const auto key = config_key_for(entry);
            auto selected = entry;
            if (!is_weight_file(entry) && document[key] && !document[key].IsNull()) {
                auto override = required_string(document, key);
                if (!override) return std::unexpected(override.error());
                selected = *override;
            }
            if (key == "tokenizer_file") tokenizer_parent = std::filesystem::path(selected).parent_path();
            if (auto ok = push_checked(files, std::move(selected)); !ok) return std::unexpected(ok.error());
        }
        for (const auto& sidecar : spec->sidecars) {
            const auto rel = (tokenizer_parent / sidecar).generic_string();
            if (auto ok = push_checked(files, rel); !ok) return std::unexpected(ok.error());
        }
        if (spec->task == "chat") {
            if (auto ok = push_checked(files, "generation_config.json"); !ok) return std::unexpected(ok.error());
        }
        if (document["tokenizers"] && document["tokenizers"].IsMap()) {
            for (const auto& entry : document["tokenizers"]) {
                if (!entry.second.IsScalar())
                    return std::unexpected(Error{ErrorCode::INVALID_MANIFEST, "tokenizer path must be a string"});
                if (auto ok = push_checked(files, entry.second.as<std::string>()); !ok)
                    return std::unexpected(ok.error());
            }
        }
        if (document["license_files"]) {
            if (!document["license_files"].IsSequence())
                return std::unexpected(Error{ErrorCode::INVALID_MANIFEST, "license_files must be a list"});
            for (const auto& license : document["license_files"]) {
                if (!license.IsScalar())
                    return std::unexpected(Error{ErrorCode::INVALID_MANIFEST, "license_files entries must be strings"});
                if (auto ok = push_checked(files, license.as<std::string>()); !ok) return std::unexpected(ok.error());
            }
        }
        return files;
    } catch (const YAML::Exception& error) {
        return std::unexpected(Error{ErrorCode::INVALID_MANIFEST, "invalid config.json: " + std::string(error.what())});
    }
}

auto load_config(const std::filesystem::path& path) -> Result<YAML::Node> {
    try {
        const auto input = std::filesystem::absolute(path).lexically_normal();
        std::filesystem::path directory;
        std::filesystem::path config_path;
        std::filesystem::path explicit_weights;
        if (std::filesystem::is_regular_file(input)) {
            directory = input.parent_path();
            if (input.extension() == ".json") {
                config_path = input;
            } else {
                config_path = directory / "config.json";
                explicit_weights = input;
            }
        } else if (std::filesystem::is_directory(input)) {
            directory = input;
            config_path = directory / "config.json";
        } else {
            return std::unexpected(Error{ErrorCode::IO, "not a model directory: " + input.string()});
        }
        if (!std::filesystem::is_regular_file(config_path))
            return std::unexpected(Error{ErrorCode::IO, "config does not exist: " + config_path.string()});

        const auto document = YAML::LoadFile(config_path.string());
        if (!document.IsMap())
            return std::unexpected(Error{ErrorCode::INVALID_MANIFEST, "config.json must contain a mapping"});
        const auto type = document["model_type"].as<std::string>("");
        const auto* spec = find_package_spec(type);
        if (!spec)
            return std::unexpected(Error{ErrorCode::UNSUPPORTED, type.empty() ? "config.json requires model_type"
                                                                              : "unsupported model_type: " + type});

        const bool use_fallback = !spec->fallback.empty() && missing_required(*spec, document) &&
                                  std::filesystem::is_regular_file(directory / spec->fallback);
        const auto manifest = use_fallback ? YAML::LoadFile((directory / spec->fallback).string()) : document;
        if (!manifest.IsMap())
            return std::unexpected(Error{ErrorCode::INVALID_MANIFEST, spec->fallback + " must contain a mapping"});
        if (auto ok = check_required(*spec, manifest); !ok) return std::unexpected(ok.error());

        YAML::Node config;
        config["format_version"] = 1;
        if (!spec->task.empty()) config["task"] = spec->task;
        if (document["external_tensors"]) config["external_tensors"] = YAML::Clone(document["external_tensors"]);
        if (manifest["vision_config"]) {
            config["vision"] = YAML::Clone(manifest["vision_config"]);
            if (manifest["image_token_id"]) config["image_token_id"] = manifest["image_token_id"];
        }

        const auto model_node = !spec->model_key.empty() ? manifest[spec->model_key] : manifest["model"];
        config["model"] = YAML::Clone(model_node && model_node.IsMap() ? model_node : manifest);
        if (!spec->model_key.empty()) {
            for (const auto* inherited : {"packed_weights_signed", "tie_word_embeddings", "dtype", "torch_dtype"})
                if ((!config["model"][inherited] || config["model"][inherited].IsNull()) && manifest[inherited])
                    config["model"][inherited] = YAML::Clone(manifest[inherited]);
            if (const auto quantization = manifest["quantization_config"];
                quantization && quantization.IsMap() &&
                (!config["model"]["quantization_config"] || config["model"]["quantization_config"].IsNull()))
                config["model"]["quantization_config"] = YAML::Clone(quantization);
        }
        if (const auto model_quant = config["model"]["quantization_config"];
            model_quant && model_quant.IsNull())
            config["model"].remove("quantization_config");
        if (spec->task == "chat") {
            config["generation"] = YAML::Node(YAML::NodeType::Map);
            for (const auto* key : {"bos_token_id", "eos_token_id", "pad_token_id"}) {
                const auto value = manifest[key] && !manifest[key].IsNull() ? manifest[key] : config["model"][key];
                if (value && !value.IsNull()) config["generation"][key] = YAML::Clone(value);
            }
            const auto generation_path = directory / "generation_config.json";
            if (std::filesystem::is_regular_file(generation_path)) {
                const auto generation = YAML::LoadFile(generation_path.string());
                if (!generation.IsMap())
                    return std::unexpected(
                        Error{ErrorCode::INVALID_MANIFEST, "generation_config.json must contain a mapping"});
                for (const auto& entry : generation)
                    if (!entry.second.IsNull())
                        config["generation"][entry.first.as<std::string>()] = YAML::Clone(entry.second);
            }
        }
        if (!config["model"]["max_position_embeddings"] && config["model"]["model_max_length"])
            config["model"]["max_position_embeddings"] = YAML::Clone(config["model"]["model_max_length"]);
        config["model"]["type"] = spec->type;

        for (const auto& key : spec->copy_keys)
            if (manifest[key] && !manifest[key].IsNull()) config[key] = YAML::Clone(manifest[key]);

        std::vector<std::string> weight_candidates;
        if (manifest["weights_file"] && !manifest["weights_file"].IsNull()) {
            auto weights_name = required_string(manifest, "weights_file");
            if (!weights_name) return std::unexpected(weights_name.error());
            weight_candidates.push_back(*weights_name);
        } else {
            for (const auto& entry : spec->files)
                if (is_weight_file(entry)) weight_candidates.push_back(entry);
            if (weight_candidates.empty()) weight_candidates.emplace_back("model.safetensors");
        }
        auto weights = resolve_weights(directory, explicit_weights, weight_candidates);
        if (!weights) return std::unexpected(weights.error());
        config["weights_file"] = (*weights)["path"];
        if (const auto format = (*weights)["format"].as<std::string>(); format != "safetensors") {
            config["weights_format"] = format;
            if ((*weights)["ggml_header"]) config["ggml_header"] = YAML::Clone((*weights)["ggml_header"]);
        }

        for (const auto& entry : spec->files) {
            if (is_weight_file(entry) || entry == spec->fallback) continue;
            const auto key = config_key_for(entry);
            auto name = string_or(manifest, key, entry);
            if (!name) return std::unexpected(name.error());
            auto resolved = resolve_file(directory, *name);
            if (!resolved) return std::unexpected(resolved.error());
            config[key] = resolved->string();
        }
        if (manifest["tokenizers"] && manifest["tokenizers"].IsMap()) {
            for (const auto& entry : manifest["tokenizers"]) {
                if (!entry.first.IsScalar() || !entry.second.IsScalar())
                    return std::unexpected(Error{ErrorCode::INVALID_MANIFEST, "tokenizers entries must be strings"});
                auto resolved = resolve_file(directory, entry.second.as<std::string>());
                if (!resolved) return std::unexpected(resolved.error());
                config["tokenizers"][entry.first.as<std::string>()] = resolved->string();
            }
        }
        if (manifest["license_files"]) {
            if (!manifest["license_files"].IsSequence())
                return std::unexpected(Error{ErrorCode::INVALID_MANIFEST, "license_files must be a sequence"});
            for (std::size_t index = 0; index < manifest["license_files"].size(); ++index) {
                if (!manifest["license_files"][index].IsScalar())
                    return std::unexpected(Error{ErrorCode::INVALID_MANIFEST, "license_files entries must be strings"});
                auto license = resolve_file(directory, manifest["license_files"][index].as<std::string>(), false);
                if (!license) return std::unexpected(license.error());
                config["license_files"].push_back(license->string());
            }
        }
        return config;
    } catch (const YAML::Exception& error) {
        return std::unexpected(Error{ErrorCode::INVALID_MANIFEST, "invalid config.json: " + std::string(error.what())});
    } catch (const std::filesystem::filesystem_error& error) {
        return std::unexpected(Error{ErrorCode::IO, error.what()});
    }
}

} // namespace kidi::checkpoint