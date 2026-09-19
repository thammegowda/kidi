#include "kidi/model/manifest.h"

#include <algorithm>
#include <array>
#include <optional>
#include <set>
#include <string_view>

#include <yaml-cpp/yaml.h>

namespace kidi::model {
namespace {

template <typename T>
T required(const YAML::Node& node, std::string_view key) {
    const auto value = node[std::string(key)];
    if (!value) {
        throw std::runtime_error("missing required field '" + std::string(key) + "'");
    }
    return value.as<T>();
}

std::expected<std::filesystem::path, std::string> resolve_package_file(const std::filesystem::path& package_directory,
                                                                       const YAML::Node& node, std::string_view key) {
    const auto declared = std::filesystem::path(required<std::string>(node, key));
    if (declared.empty() || declared.is_absolute()) {
        return std::unexpected("field '" + std::string(key) + "' must be a relative file path");
    }

    const auto resolved = (package_directory / declared).lexically_normal();
    const auto relative = resolved.lexically_relative(package_directory);
    if (relative.empty() || *relative.begin() == "..") {
        return std::unexpected("field '" + std::string(key) + "' escapes the model package");
    }
    if (!std::filesystem::is_regular_file(resolved)) {
        return std::unexpected("file for field '" + std::string(key) + "' does not exist: " + resolved.string());
    }
    return resolved;
}

std::optional<std::string> require_positive(std::int32_t value, std::string_view field) {
    if (value <= 0) {
        return "field '" + std::string(field) + "' must be positive";
    }
    return std::nullopt;
}

WeightEncoding parse_weight_encoding(const YAML::Node& weights) {
    if (!weights) return WeightEncoding::F32;
    const auto encoding =
        weights["encoding"] ? weights["encoding"].as<std::string>() : weights["data_type"].as<std::string>("F32");
    if (encoding == "F32") return WeightEncoding::F32;
    if (encoding == "BF16") return WeightEncoding::BF16;
    if (encoding == "INT8_PER_CHANNEL") return WeightEncoding::INT8_PER_CHANNEL;
    throw std::runtime_error("unsupported weight encoding '" + encoding + "'");
}

LinearWeightLayout parse_linear_weight_layout(const YAML::Node& weights) {
    if (!weights) return LinearWeightLayout::OUTPUT_INPUT;
    const auto layout = weights["linear_layout"].as<std::string>("OUTPUT_INPUT");
    if (layout == "OUTPUT_INPUT") return LinearWeightLayout::OUTPUT_INPUT;
    if (layout == "INPUT_OUTPUT") return LinearWeightLayout::INPUT_OUTPUT;
    throw std::runtime_error("unsupported linear weight layout '" + layout + "'");
}

std::optional<std::string> validate(const ModelManifest& manifest) {
    if (manifest.format_version != 1) {
        return "unsupported format_version; expected 1";
    }
    if (manifest.model_type != "rtg_transformer_nmt") {
        return "unsupported model_type '" + manifest.model_type + "'";
    }
    if (manifest.input_format != "moses_tokenized" || manifest.output_format != "moses_tokenized") {
        return "the RTG MVP requires moses_tokenized input and output";
    }
    if (manifest.weights.format != "safetensors") {
        return "the RTG MVP requires Safetensors weights";
    }
    if (manifest.weights.encoding != WeightEncoding::F32 &&
        manifest.weights.linear_layout != LinearWeightLayout::INPUT_OUTPUT) {
        return "reduced-precision weights require INPUT_OUTPUT linear layout";
    }

    const std::array positive_fields = {
        std::pair{manifest.architecture.encoder_layers, std::string_view{"architecture.encoder_layers"}},
        std::pair{manifest.architecture.decoder_layers, std::string_view{"architecture.decoder_layers"}},
        std::pair{manifest.architecture.hidden_size, std::string_view{"architecture.hidden_size"}},
        std::pair{manifest.architecture.feed_forward_size, std::string_view{"architecture.feed_forward_size"}},
        std::pair{manifest.architecture.attention_heads, std::string_view{"architecture.attention_heads"}},
        std::pair{manifest.architecture.source_vocabulary_size,
                  std::string_view{"architecture.source_vocabulary_size"}},
        std::pair{manifest.architecture.target_vocabulary_size,
                  std::string_view{"architecture.target_vocabulary_size"}},
        std::pair{manifest.architecture.maximum_position, std::string_view{"architecture.maximum_position"}},
        std::pair{manifest.limits.source_tokens, std::string_view{"limits.source_tokens"}},
        std::pair{manifest.limits.maximum_extra_tokens, std::string_view{"limits.maximum_extra_tokens"}},
        std::pair{manifest.limits.maximum_beam_size, std::string_view{"limits.maximum_beam_size"}},
    };
    for (const auto& [value, field] : positive_fields) {
        if (auto error = require_positive(value, field)) return error;
    }
    if (manifest.architecture.hidden_size % manifest.architecture.attention_heads != 0) {
        return "hidden_size must be divisible by attention_heads";
    }
    if (manifest.architecture.activation != "gelu") {
        return "the RTG MVP supports only gelu activation";
    }
    if (!manifest.architecture.attention_bias) {
        return "the RTG MVP requires attention bias";
    }
    if (manifest.architecture.tied_embeddings != "one-way") {
        return "the RTG MVP requires one-way tied embeddings";
    }
    if (manifest.architecture.layer_norm_epsilon <= 0.0F) {
        return "architecture.layer_norm_epsilon must be positive";
    }
    if (manifest.architecture.position_encoding != "sinusoidal") {
        return "the RTG MVP supports only sinusoidal position encoding";
    }

    if (manifest.decode_defaults.beam_size <= 0 ||
        manifest.decode_defaults.beam_size > manifest.limits.maximum_beam_size) {
        return "decode.beam_size exceeds the package beam limit";
    }
    if (manifest.decode_defaults.maximum_extra_tokens <= 0 ||
        manifest.decode_defaults.maximum_extra_tokens > manifest.limits.maximum_extra_tokens) {
        return "decode.maximum_extra_tokens exceeds the package limit";
    }
    if (manifest.decode_defaults.length_penalty < 0.0F) {
        return "decode.length_penalty cannot be negative";
    }

    const std::array ids = {
        manifest.special_tokens.pad,
        manifest.special_tokens.unknown,
        manifest.special_tokens.begin,
        manifest.special_tokens.end,
    };
    if (std::ranges::any_of(ids, [](std::int32_t id) { return id < 0; }) ||
        std::set(ids.begin(), ids.end()).size() != ids.size()) {
        return "special token IDs must be distinct non-negative integers";
    }
    if (*std::ranges::max_element(ids) >= manifest.architecture.target_vocabulary_size) {
        return "special token ID exceeds target vocabulary size";
    }
    return std::nullopt;
}

} // namespace

Result<ModelManifest> ModelManifest::load(const std::filesystem::path& path) {
    try {
        if (!std::filesystem::is_regular_file(path)) {
            return std::unexpected(Error{ErrorCode::IO, "manifest does not exist: " + path.string()});
        }
        const auto absolute_path = std::filesystem::absolute(path).lexically_normal();
        const auto package_directory = absolute_path.parent_path();
        const auto root = YAML::LoadFile(absolute_path.string());

        ModelManifest manifest;
        manifest.format_version = required<std::int32_t>(root, "format_version");
        manifest.model_type = required<std::string>(root, "model_type");
        manifest.package_directory = package_directory;
        auto weights_file = resolve_package_file(package_directory, root, "weights_file");
        if (!weights_file) {
            return std::unexpected(Error{ErrorCode::INVALID_MANIFEST, std::move(weights_file.error())});
        }
        manifest.weights_file = std::move(*weights_file);

        const auto tokenizers = root["tokenizers"];
        auto source_tokenizer = resolve_package_file(package_directory, tokenizers, "source");
        if (!source_tokenizer) {
            return std::unexpected(Error{ErrorCode::INVALID_MANIFEST, std::move(source_tokenizer.error())});
        }
        auto target_tokenizer = resolve_package_file(package_directory, tokenizers, "target");
        if (!target_tokenizer) {
            return std::unexpected(Error{ErrorCode::INVALID_MANIFEST, std::move(target_tokenizer.error())});
        }
        manifest.tokenizers = {
            .source = std::move(*source_tokenizer),
            .target = std::move(*target_tokenizer),
        };

        const auto io = root["io"];
        manifest.input_format = required<std::string>(io, "input_format");
        manifest.output_format = required<std::string>(io, "output_format");

        const auto special = root["special_tokens"];
        manifest.special_tokens = {
            .pad = required<std::int32_t>(special, "pad"),
            .unknown = required<std::int32_t>(special, "unknown"),
            .begin = required<std::int32_t>(special, "begin"),
            .end = required<std::int32_t>(special, "end"),
        };

        const auto architecture = root["architecture"];
        manifest.architecture = {
            .encoder_layers = required<std::int32_t>(architecture, "encoder_layers"),
            .decoder_layers = required<std::int32_t>(architecture, "decoder_layers"),
            .hidden_size = required<std::int32_t>(architecture, "hidden_size"),
            .feed_forward_size = required<std::int32_t>(architecture, "feed_forward_size"),
            .attention_heads = required<std::int32_t>(architecture, "attention_heads"),
            .source_vocabulary_size = required<std::int32_t>(architecture, "source_vocabulary_size"),
            .target_vocabulary_size = required<std::int32_t>(architecture, "target_vocabulary_size"),
            .activation = required<std::string>(architecture, "activation"),
            .attention_bias = required<bool>(architecture, "attention_bias"),
            .tied_embeddings = required<std::string>(architecture, "tied_embeddings"),
            .layer_norm_epsilon = required<float>(architecture, "layer_norm_epsilon"),
            .position_encoding = required<std::string>(architecture, "position_encoding"),
            .maximum_position = required<std::int32_t>(architecture, "maximum_position"),
        };

        const auto limits = root["limits"];
        manifest.limits = {
            .source_tokens = required<std::int32_t>(limits, "source_tokens"),
            .maximum_extra_tokens = required<std::int32_t>(limits, "maximum_extra_tokens"),
            .maximum_beam_size = required<std::int32_t>(limits, "maximum_beam_size"),
        };

        const auto decode = root["decode"];
        manifest.decode_defaults = {
            .beam_size = required<std::int32_t>(decode, "beam_size"),
            .maximum_extra_tokens = required<std::int32_t>(decode, "maximum_extra_tokens"),
            .length_penalty = required<float>(decode, "length_penalty"),
        };

        const auto weights = root["weights"];
        manifest.weights = {
            .format = weights ? weights["format"].as<std::string>("safetensors") : "safetensors",
            .encoding = parse_weight_encoding(weights),
            .linear_layout = parse_linear_weight_layout(weights),
        };

        if (auto error = validate(manifest)) {
            return std::unexpected(Error{ErrorCode::INVALID_MANIFEST, std::move(*error)});
        }
        return manifest;
    } catch (const YAML::Exception& error) {
        return std::unexpected(
            Error{ErrorCode::INVALID_MANIFEST, "invalid YAML manifest: " + std::string(error.what())});
    } catch (const std::exception& error) {
        return std::unexpected(Error{ErrorCode::INVALID_MANIFEST, error.what()});
    }
}

} // namespace kidi::model