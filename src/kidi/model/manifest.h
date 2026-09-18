#pragma once

#include <cstdint>
#include <expected>
#include <filesystem>
#include <string>

#include "kidi/core/error.h"

namespace kidi::model {

struct TokenizerFiles {
    std::filesystem::path source;
    std::filesystem::path target;
};

struct SpecialTokenIds {
    std::int32_t pad;
    std::int32_t unknown;
    std::int32_t begin;
    std::int32_t end;
};

struct RtgArchitecture {
    std::int32_t encoder_layers;
    std::int32_t decoder_layers;
    std::int32_t hidden_size;
    std::int32_t feed_forward_size;
    std::int32_t attention_heads;
    std::int32_t source_vocabulary_size;
    std::int32_t target_vocabulary_size;
    std::string activation;
    bool attention_bias;
    std::string tied_embeddings;
    float layer_norm_epsilon;
    std::string position_encoding;
    std::int32_t maximum_position;
};

struct RtgLimits {
    std::int32_t source_tokens;
    std::int32_t maximum_extra_tokens;
    std::int32_t maximum_beam_size;
};

struct DecodeDefaults {
    std::int32_t beam_size;
    std::int32_t maximum_extra_tokens;
    float length_penalty;
};

struct ModelManifest {
    std::int32_t format_version;
    std::string model_type;
    std::filesystem::path package_directory;
    std::filesystem::path weights_file;
    TokenizerFiles tokenizers;
    SpecialTokenIds special_tokens;
    RtgArchitecture architecture;
    RtgLimits limits;
    DecodeDefaults decode_defaults;
    std::string input_format;
    std::string output_format;

    [[nodiscard]] static std::expected<ModelManifest, core::Error> load(const std::filesystem::path& path);
};

} // namespace kidi::model