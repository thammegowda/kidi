#include "kidi/checkpoint/config.h"
#include "kidi/checkpoint/ggml/reader.h"
#include "kidi/model/whisper.h"

#include <algorithm>
#include <array>
#include <fstream>
#include <span>
#include <string>
#include <string_view>

namespace kidi::checkpoint {
namespace {

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
                     std::span<const std::string_view> candidates) -> Result<YAML::Node> {
    auto source = explicit_weights;
    if (source.empty()) {
        for (const auto candidate : candidates) {
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

} // namespace

auto hub_files(const YAML::Node& document) -> Result<std::vector<std::string>> {
    try {
        if (!document.IsMap())
            return std::unexpected(Error{ErrorCode::INVALID_MANIFEST, "config.json must contain a JSON object"});
        const auto type = document["model_type"].as<std::string>("");
        std::vector<std::string> files{"config.json"};
        if (type == "gemma4" || type == "gemma4_text") {
            const auto text = type == "gemma4" ? document["text_config"] : document;
            if (!text || !text.IsMap() || text["enable_moe_block"].as<bool>(false))
                return std::unexpected(Error{ErrorCode::UNSUPPORTED, "only dense Gemma 4 checkpoints are supported"});
            const auto quantization = document["quantization_config"];
            if (quantization && !quantization.IsNull() &&
                (!quantization.IsMap() || quantization["quant_method"].as<std::string>("") != "gemma"))
                return std::unexpected(
                    Error{ErrorCode::UNSUPPORTED,
                          "only native Gemma mobile QAT or original floating-point checkpoints are supported"});
            auto weights = string_or(document, "weights_file", "model.safetensors");
            if (!weights) return std::unexpected(weights.error());
            auto tokenizer = string_or(document, "tokenizer_file", "tokenizer.json");
            if (!tokenizer) return std::unexpected(tokenizer.error());
            if (auto ok = push_checked(files, *weights); !ok) return std::unexpected(ok.error());
            if (auto ok = push_checked(files, *tokenizer); !ok) return std::unexpected(ok.error());
            const auto parent = std::filesystem::path(*tokenizer).parent_path();
            for (const auto* sidecar : {"tokenizer_config.json", "chat_template.jinja"}) {
                const auto rel = (parent / sidecar).generic_string();
                if (auto ok = push_checked(files, rel); !ok) return std::unexpected(ok.error());
            }
            return files;
        }
        if (type == "gemma4_assistant") {
            const auto text = document["text_config"];
            if (!text || !text.IsMap() || text["enable_moe_block"].as<bool>(false) ||
                (document["quantization_config"] && !document["quantization_config"].IsNull()) ||
                (text["quantization_config"] && !text["quantization_config"].IsNull()))
                return std::unexpected(Error{ErrorCode::UNSUPPORTED, "unsupported Gemma 4 MTP assistant checkpoint"});
            auto weights = string_or(document, "weights_file", "model.safetensors");
            if (!weights) return std::unexpected(weights.error());
            if (auto ok = push_checked(files, *weights); !ok) return std::unexpected(ok.error());
            return files;
        }
        if (type == "whisper") {
            YAML::Node candidate = YAML::Clone(document);
            candidate["type"] = "whisper";
            if (auto valid = model::WhisperImpl::validate_config(candidate); !valid)
                return std::unexpected(valid.error());
            for (const auto* name :
                 {"model.safetensors", "tokenizer.json", "preprocessor_config.json", "generation_config.json"})
                files.emplace_back(name);
            return files;
        }
        if (type == "rtg_transformer_nmt") {
            if (!document["decode"].IsMap() || !document["tokenizers"].IsMap())
                return std::unexpected(
                    Error{ErrorCode::INVALID_MANIFEST, "RTG config.json requires decode and tokenizers"});
            auto weights = string_or(document, "weights_file", "model.safetensors");
            if (!weights) return std::unexpected(weights.error());
            if (auto ok = push_checked(files, *weights); !ok) return std::unexpected(ok.error());
            for (const auto* key : {"source", "target"}) {
                auto tok = required_string(document["tokenizers"], key);
                if (!tok) return std::unexpected(tok.error());
                if (auto ok = push_checked(files, *tok); !ok) return std::unexpected(ok.error());
            }
            return files;
        }
        if (type == "omnivoice" || type == "kidi_omnivoice" || type == "kokoro") {
            if (!document["synthesis"].IsMap() || !document["license_files"].IsSequence() ||
                !document["license_files"].size())
                return std::unexpected(
                    Error{ErrorCode::INVALID_MANIFEST, type + " config.json requires synthesis and license_files"});
            auto weights = string_or(document, "weights_file", "model.safetensors");
            if (!weights) return std::unexpected(weights.error());
            auto prefixes = string_or(document, "sentence_prefixes_file", "sentence-prefixes.tsv");
            if (!prefixes) return std::unexpected(prefixes.error());
            if (auto ok = push_checked(files, *weights); !ok) return std::unexpected(ok.error());
            if (auto ok = push_checked(files, *prefixes); !ok) return std::unexpected(ok.error());
            if (type == "kokoro") {
                auto lexicon = string_or(document, "lexicon_file", "lexicon-us.tsv");
                if (!lexicon) return std::unexpected(lexicon.error());
                if (auto ok = push_checked(files, *lexicon); !ok) return std::unexpected(ok.error());
            } else {
                auto tokenizer = string_or(document, "tokenizer_file", "tokenizer.json");
                auto tmpl = string_or(document, "template_file", "synthesis_template.jinja");
                if (!tokenizer) return std::unexpected(tokenizer.error());
                if (!tmpl) return std::unexpected(tmpl.error());
                if (auto ok = push_checked(files, *tokenizer); !ok) return std::unexpected(ok.error());
                if (auto ok = push_checked(files, *tmpl); !ok) return std::unexpected(ok.error());
            }
            for (const auto& license : document["license_files"]) {
                if (!license.IsScalar())
                    return std::unexpected(Error{ErrorCode::INVALID_MANIFEST, "license_files entries must be strings"});
                if (auto ok = push_checked(files, license.as<std::string>()); !ok) return std::unexpected(ok.error());
            }
            return files;
        }
        return std::unexpected(
            Error{ErrorCode::UNSUPPORTED,
                  "Automatic setup supports dense Gemma 4, Gemma 4 MTP assistants, Whisper, RTG, OmniVoice, or Kokoro "
                  "config.json packages only"});
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
        YAML::Node config;
        config["format_version"] = 1;
        if (document["external_tensors"]) config["external_tensors"] = YAML::Clone(document["external_tensors"]);

        if (type == "gemma4" || type == "gemma4_text") {
            const auto text = document["text_config"];
            config["model"] = YAML::Clone(text && text.IsMap() ? text : document);
            config["model"]["type"] = "gemma4_text";
            if (document["packed_weights_signed"])
                config["model"]["packed_weights_signed"] = document["packed_weights_signed"];
            const auto quantization = document["quantization_config"];
            if (quantization && quantization.IsMap())
                config["model"]["quantization_config"] = YAML::Clone(quantization);
            if (config["model"]["quantization_config"].IsNull()) config["model"].remove("quantization_config");
            if (!config["model"]["tie_word_embeddings"] && document["tie_word_embeddings"])
                config["model"]["tie_word_embeddings"] = document["tie_word_embeddings"];
            auto weights_name = string_or(document, "weights_file", "model.safetensors");
            if (!weights_name) return std::unexpected(weights_name.error());
            const std::array<std::string_view, 1> candidates{*weights_name};
            auto weights = resolve_weights(directory, explicit_weights, candidates);
            if (!weights) return std::unexpected(weights.error());
            if ((*weights)["format"].as<std::string>() != "safetensors")
                return std::unexpected(Error{ErrorCode::UNSUPPORTED, "Gemma 4 requires safetensors weights"});
            config["weights_file"] = (*weights)["path"];
            auto tokenizer_name = string_or(document, "tokenizer_file", "tokenizer.json");
            if (!tokenizer_name) return std::unexpected(tokenizer_name.error());
            auto tokenizer = resolve_file(directory, *tokenizer_name);
            if (!tokenizer) return std::unexpected(tokenizer.error());
            config["tokenizer_file"] = tokenizer->string();
            if (document["decode"] && document["decode"].IsMap()) {
                config["decode"] = YAML::Clone(document["decode"]);
            } else {
                config["decode"]["context_size"] =
                    std::min<std::size_t>(config["model"]["max_position_embeddings"].as<std::size_t>(16384), 16384);
                config["decode"]["maximum_new_tokens"] = 8192;
            }
            if (document["vision_config"]) {
                config["vision"] = YAML::Clone(document["vision_config"]);
                config["image_token_id"] = document["image_token_id"];
            }
            return config;
        }

        if (type == "gemma4_assistant") {
            config["model"] = YAML::Clone(document);
            config["model"]["type"] = "gemma4_assistant";
            auto weights_name = string_or(document, "weights_file", "model.safetensors");
            if (!weights_name) return std::unexpected(weights_name.error());
            const std::array<std::string_view, 1> candidates{*weights_name};
            auto weights = resolve_weights(directory, explicit_weights, candidates);
            if (!weights) return std::unexpected(weights.error());
            if ((*weights)["format"].as<std::string>() != "safetensors")
                return std::unexpected(Error{ErrorCode::UNSUPPORTED, "Gemma MTP requires safetensors weights"});
            config["weights_file"] = (*weights)["path"];
            return config;
        }

        if (type == "whisper") {
            config["model"] = YAML::Clone(document);
            config["model"]["type"] = "whisper";
            constexpr std::array<std::string_view, 2> candidates{"model.safetensors", "ggml-model.bin"};
            auto weights = resolve_weights(directory, explicit_weights, candidates);
            if (!weights) return std::unexpected(weights.error());
            config["weights_file"] = (*weights)["path"];
            if ((*weights)["format"].as<std::string>() == "ggml") {
                constexpr std::array keys{
                    "vocab_size",     "max_source_positions", "d_model", "encoder_attention_heads",
                    "encoder_layers", "max_target_positions", "d_model", "decoder_attention_heads",
                    "decoder_layers", "num_mel_bins"};
                for (std::size_t index = 0; index < keys.size(); ++index)
                    if (document[keys[index]].as<std::int32_t>() != (*weights)["ggml_header"][index].as<std::int32_t>())
                        return std::unexpected(
                            Error{ErrorCode::INVALID_MANIFEST,
                                  "Whisper GGML header disagrees with config: " + std::string(keys[index])});
                config["weights_format"] = "whisper_ggml";
            } else if ((*weights)["format"].as<std::string>() == "gguf") {
                return std::unexpected(Error{ErrorCode::UNSUPPORTED,
                                             "Whisper GGUF model mapping is not supported; use legacy Whisper GGML"});
            }
            for (const auto* name : {"tokenizer.json", "preprocessor_config.json", "generation_config.json"}) {
                auto file = resolve_file(directory, name);
                if (!file) return std::unexpected(file.error());
                config[std::filesystem::path(name).stem().string() + "_file"] = file->string();
            }
            return config;
        }

        if (type == "omnivoice" || type == "kidi_omnivoice" || type == "kokoro") {
            const auto normalized_type = type == "kokoro" ? std::string("kokoro") : std::string("omnivoice");
            if (!document["synthesis"].IsMap())
                return std::unexpected(
                    Error{ErrorCode::INVALID_MANIFEST, normalized_type + " requires synthesis settings"});
            if (!document["license_files"].IsSequence() || !document["license_files"].size())
                return std::unexpected(
                    Error{ErrorCode::INVALID_MANIFEST, normalized_type + " requires redistributed license files"});
            auto weights_name = string_or(document, "weights_file", "model.safetensors");
            if (!weights_name) return std::unexpected(weights_name.error());
            auto weights = resolve_file(directory, *weights_name);
            if (!weights) return std::unexpected(weights.error());
            config["task"] = "tts";
            config["weights_file"] = weights->string();
            config["synthesis"] = YAML::Clone(document["synthesis"]);
            for (std::size_t index = 0; index < document["license_files"].size(); ++index) {
                if (!document["license_files"][index].IsScalar())
                    return std::unexpected(Error{ErrorCode::INVALID_MANIFEST, "license_files entries must be strings"});
                auto license = resolve_file(directory, document["license_files"][index].as<std::string>());
                if (!license) return std::unexpected(license.error());
                config["license_files"].push_back(license->string());
            }
            auto prefixes_name = string_or(document, "sentence_prefixes_file", "sentence-prefixes.tsv");
            if (!prefixes_name) return std::unexpected(prefixes_name.error());
            auto prefixes = resolve_file(directory, *prefixes_name);
            if (!prefixes) return std::unexpected(prefixes.error());
            config["sentence_prefixes_file"] = prefixes->string();
            if (normalized_type == "omnivoice") {
                auto tokenizer_name = string_or(document, "tokenizer_file", "tokenizer.json");
                auto template_name = string_or(document, "template_file", "synthesis_template.jinja");
                if (!tokenizer_name) return std::unexpected(tokenizer_name.error());
                if (!template_name) return std::unexpected(template_name.error());
                auto tokenizer = resolve_file(directory, *tokenizer_name);
                if (!tokenizer) return std::unexpected(tokenizer.error());
                auto synthesis_template = resolve_file(directory, *template_name);
                if (!synthesis_template) return std::unexpected(synthesis_template.error());
                config["tokenizer_file"] = tokenizer->string();
                config["template_file"] = synthesis_template->string();
            } else {
                auto lexicon_name = string_or(document, "lexicon_file", "lexicon-us.tsv");
                if (!lexicon_name) return std::unexpected(lexicon_name.error());
                auto lexicon = resolve_file(directory, *lexicon_name);
                if (!lexicon) return std::unexpected(lexicon.error());
                config["lexicon_file"] = lexicon->string();
            }
            config["model"] =
                YAML::Clone(document["model"] && document["model"].IsMap() ? document["model"] : document);
            config["model"]["type"] = normalized_type;
            return config;
        }

        if (type == "rtg_transformer_nmt") {
            auto weights_name = string_or(document, "weights_file", "model.safetensors");
            if (!weights_name) return std::unexpected(weights_name.error());
            auto weights = resolve_file(directory, *weights_name);
            if (!weights) return std::unexpected(weights.error());
            if (!document["decode"].IsMap())
                return std::unexpected(Error{ErrorCode::INVALID_MANIFEST, "decode must be a mapping"});
            for (const auto* key : {"input_format", "output_format"})
                if (!document["io"].IsMap() || document["io"][key].as<std::string>("") != "moses_tokenized")
                    return std::unexpected(
                        Error{ErrorCode::INVALID_MANIFEST, "input and output must be moses_tokenized"});
            if (!document["tokenizers"].IsMap())
                return std::unexpected(Error{ErrorCode::INVALID_MANIFEST, "tokenizers must be a mapping"});
            config["weights_file"] = weights->string();
            config["io"] = YAML::Clone(document["io"]);
            config["decode"] = YAML::Clone(document["decode"]);
            for (const auto* key : {"source", "target"}) {
                auto tok_name = required_string(document["tokenizers"], key);
                if (!tok_name) return std::unexpected(tok_name.error());
                auto tokenizer = resolve_file(directory, *tok_name);
                if (!tokenizer) return std::unexpected(tokenizer.error());
                config["tokenizers"][key] = tokenizer->string();
            }
            config["model"] =
                YAML::Clone(document["model"] && document["model"].IsMap() ? document["model"] : document);
            config["model"]["type"] = "rtg_transformer_nmt";
            return config;
        }

        return std::unexpected(Error{ErrorCode::UNSUPPORTED, type.empty() ? "config.json requires model_type"
                                                                          : "unsupported model_type: " + type});
    } catch (const YAML::Exception& error) {
        return std::unexpected(Error{ErrorCode::INVALID_MANIFEST, "invalid config.json: " + std::string(error.what())});
    } catch (const std::filesystem::filesystem_error& error) {
        return std::unexpected(Error{ErrorCode::IO, error.what()});
    }
}

} // namespace kidi::checkpoint