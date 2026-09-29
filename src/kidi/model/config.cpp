#include "kidi/model/config.h"
#include "kidi/model/ggml.h"

#include <fstream>
#include <string>

namespace kidi::model {
namespace {

auto resolve_file(const std::filesystem::path& directory, const YAML::Node& value) -> Result<std::filesystem::path> {
    const auto declared = std::filesystem::path(value.as<std::string>());
    if (declared.empty() || declared.is_absolute())
        return std::unexpected(Error{ErrorCode::INVALID_MANIFEST, "package files must use relative paths"});
    const auto resolved = (directory / declared).lexically_normal();
    const auto relative = resolved.lexically_relative(directory);
    if (relative.empty() || *relative.begin() == "..")
        return std::unexpected(Error{ErrorCode::INVALID_MANIFEST, "file path escapes the model package"});
    if (!std::filesystem::is_regular_file(resolved))
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

auto load_whisper_config(const std::filesystem::path& directory) -> Result<YAML::Node> {
    try {
        const auto input = std::filesystem::absolute(directory).lexically_normal();
        const bool explicit_file = std::filesystem::is_regular_file(input);
        const auto root = explicit_file ? input.parent_path() : input;
        if (!std::filesystem::is_directory(root))
            return std::unexpected(Error{ErrorCode::IO, "not a model directory: " + directory.string()});
        const auto config_path = root / "config.json";
        if (!std::filesystem::is_regular_file(config_path))
            return std::unexpected(Error{ErrorCode::IO, "config does not exist: " + config_path.string()});
        auto model = YAML::LoadFile(config_path.string());
        if (!model.IsMap() || model["model_type"].as<std::string>() != "whisper")
            return std::unexpected(Error{ErrorCode::UNSUPPORTED, "checkpoint is not a Hugging Face Whisper model"});

        YAML::Node config;
        config["model"] = model;
        config["model"]["type"] = "whisper";
        const auto source = explicit_file                                                  ? input
                            : std::filesystem::is_regular_file(root / "model.safetensors") ? root / "model.safetensors"
                                                                                           : root / "ggml-model.bin";
        if (!std::filesystem::is_regular_file(source))
            return std::unexpected(Error{ErrorCode::IO, "Whisper weights do not exist: " + source.string()});
        config["model_file"] = source.string();
        std::ifstream stream(source, std::ios::binary);
        std::uint32_t magic = 0;
        stream.read(reinterpret_cast<char*>(&magic), sizeof(magic));
        if (magic == GGML_MAGIC) {
            auto imported = GgmlFile::open(source);
            if (!imported) return std::unexpected(imported.error());
            const auto& header = *imported->whisper_header();
            constexpr std::array keys{"vocab_size",     "max_source_positions", "d_model", "encoder_attention_heads",
                                      "encoder_layers", "max_target_positions", "d_model", "decoder_attention_heads",
                                      "decoder_layers", "num_mel_bins"};
            for (std::size_t index = 0; index < keys.size(); ++index)
                if (model[keys[index]].as<std::int32_t>() != header[index])
                    return std::unexpected(
                        Error{ErrorCode::INVALID_MANIFEST,
                              "Whisper GGML header disagrees with config: " + std::string(keys[index])});
            config["weights_format"] = "whisper_ggml";
        } else if (magic == GGUF_MAGIC) {
            return std::unexpected(
                Error{ErrorCode::UNSUPPORTED, "Whisper GGUF model mapping is not supported; use legacy Whisper GGML"});
        }
        for (const auto* name : {"tokenizer.json", "preprocessor_config.json", "generation_config.json"}) {
            auto resolved = resolve_file(root, YAML::Node(name));
            if (!resolved) return std::unexpected(std::move(resolved.error()));
            config[std::filesystem::path(name).stem().string() + "_file"] = resolved->string();
        }
        return config;
    } catch (const YAML::Exception& error) {
        return std::unexpected(
            Error{ErrorCode::INVALID_MANIFEST, "invalid Whisper config: " + std::string(error.what())});
    } catch (const std::filesystem::filesystem_error& error) {
        return std::unexpected(Error{ErrorCode::IO, error.what()});
    }
}

} // namespace kidi::model