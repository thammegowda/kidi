#include "kidi/checkpoint/config.h"
#include "kidi/checkpoint/ggml/reader.h"
#include "kidi/ops/context.h"

#include <fstream>
#include <string>

namespace kidi::checkpoint {
namespace {

auto resolve_file(const std::filesystem::path& directory, const YAML::Node& value, bool required = true)
    -> Result<std::filesystem::path> {
    const auto declared = std::filesystem::path(value.as<std::string>());
    if (declared.empty() || declared.is_absolute())
        return std::unexpected(Error{ErrorCode::INVALID_MANIFEST, "package files must use relative paths"});
    const auto resolved = (directory / declared).lexically_normal();
    const auto relative = resolved.lexically_relative(directory);
    if (relative.empty() || *relative.begin() == "..")
        return std::unexpected(Error{ErrorCode::INVALID_MANIFEST, "file path escapes the model package"});
    if (required && !std::filesystem::is_regular_file(resolved))
        return std::unexpected(Error{ErrorCode::INVALID_MANIFEST, "package file does not exist: " + resolved.string()});
    return resolved;
}

} // namespace

auto load_config(const std::filesystem::path& path) -> Result<YAML::Node> {
    try {
        if (!std::filesystem::is_regular_file(path))
            return std::unexpected(Error{ErrorCode::IO, "config does not exist: " + path.string()});
        const auto directory = std::filesystem::absolute(path).lexically_normal().parent_path();
        auto config = YAML::LoadFile(path.string());
        if (!config.IsMap() || config["format_version"].as<int>() != 1)
            return std::unexpected(Error{ErrorCode::INVALID_MANIFEST, "unsupported config; expected format_version 1"});
        if (!config["model"].IsMap() || !config["decode"].IsMap())
            return std::unexpected(Error{ErrorCode::INVALID_MANIFEST, "model and decode must be YAML mappings"});
        auto weights = resolve_file(directory, config["weights_file"]);
        if (!weights) return std::unexpected(std::move(weights.error()));
        config["weights_file"] = weights->string();
        if (config["model"]["type"].as<std::string>() == "gemma4_text") {
            auto tokenizer = resolve_file(directory, config["tokenizer_file"]);
            if (!tokenizer) return std::unexpected(std::move(tokenizer.error()));
            config["tokenizer_file"] = tokenizer->string();
            return config;
        }
        for (const auto* key : {"input_format", "output_format"})
            if (config["io"][key].as<std::string>() != "moses_tokenized")
                return std::unexpected(Error{ErrorCode::INVALID_MANIFEST, "input and output must be moses_tokenized"});
        for (const auto* key : {"source", "target"}) {
            auto tokenizer = resolve_file(directory, config["tokenizers"][key]);
            if (!tokenizer) return std::unexpected(std::move(tokenizer.error()));
            config["tokenizers"][key] = tokenizer->string();
        }
        return config;
    } catch (const YAML::Exception& error) {
        return std::unexpected(Error{ErrorCode::INVALID_MANIFEST, "invalid YAML config: " + std::string(error.what())});
    } catch (const std::filesystem::filesystem_error& error) {
        return std::unexpected(Error{ErrorCode::IO, error.what()});
    }
}

auto ConfigSource::file(std::string_view name) const -> Result<std::filesystem::path> {
    return resolve_file(root, YAML::Node(std::string(name)));
}

auto ConfigSource::weights(std::span<const std::string_view> candidates) const -> Result<YAML::Node> {
    auto source = explicit_weights;
    if (source.empty()) {
        for (const auto candidate : candidates) {
            auto resolved = resolve_file(root, YAML::Node(std::string(candidate)), false);
            if (!resolved) return std::unexpected(resolved.error());
            if (std::filesystem::is_regular_file(*resolved)) {
                source = *resolved;
                break;
            }
        }
    }
    if (source.empty() || !std::filesystem::is_regular_file(source))
        return std::unexpected(Error{ErrorCode::IO, "checkpoint weights not found in " + root.string()});
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

auto load_config(const std::filesystem::path& path, const ConfigAdapter& adapter) -> Result<YAML::Node> {
    try {
        if (adapter.filename.empty() || !adapter.configure)
            return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "invalid checkpoint config adapter"});
        const auto input = std::filesystem::absolute(path).lexically_normal();
        const bool explicit_file = std::filesystem::is_regular_file(input);
        const auto root = explicit_file ? input.parent_path() : input;
        if (!std::filesystem::is_directory(root))
            return std::unexpected(Error{ErrorCode::IO, "not a model directory: " + root.string()});
        ConfigSource source{root, explicit_file ? input : std::filesystem::path{}, {}};
        auto metadata = source.file(adapter.filename);
        if (!metadata) return std::unexpected(metadata.error());
        source.document = YAML::LoadFile(metadata->string());
        return adapter.configure(source);
    } catch (const ops::Failure& error) {
        return std::unexpected(error.error());
    } catch (const YAML::Exception& error) {
        return std::unexpected(
            Error{ErrorCode::INVALID_MANIFEST, "invalid checkpoint config: " + std::string(error.what())});
    } catch (const std::filesystem::filesystem_error& error) {
        return std::unexpected(Error{ErrorCode::IO, error.what()});
    }
}

} // namespace kidi::checkpoint