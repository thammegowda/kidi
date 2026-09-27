#pragma once

#include <filesystem>
#include <map>
#include <string>

#include <yaml-cpp/yaml.h>

#include "kidi/audio/whisper.h"
#include "kidi/model/whisper.h"
#include "kidi/text/tokenizer.h"

namespace kidi::inference {

struct TranscriptionOptions {
    std::string language = "auto";
    std::string task = "transcribe";
    std::size_t maximum_tokens = 128;
};

struct TranscriptionStats {
    std::uint64_t feature_ns = 0, encode_ns = 0, decode_ns = 0, preparation_ns = 0;
};

struct Transcription {
    std::string text, language;
    std::vector<std::int32_t> token_ids;
    TranscriptionStats stats;
};

class Transcriber {
public:
    static auto load(const std::filesystem::path& directory, tensor::Device device = tensor::Device::cpu())
        -> Result<Transcriber>;
    auto transcribe(std::span<const float> waveform, std::uint32_t sample_rate,
                    const TranscriptionOptions& options = {}) -> Result<Transcription>;

private:
    Transcriber(YAML::Node config, YAML::Node generation, audio::WhisperFeatureExtractor extractor,
                text::Tokenizer tokenizer, model::Whisper model, std::map<std::string, std::int32_t> languages,
                std::vector<std::int32_t> suppress, std::vector<std::int32_t> begin_suppress);

    YAML::Node config_, generation_;
    audio::WhisperFeatureExtractor extractor_;
    text::Tokenizer tokenizer_;
    model::Whisper model_;
    std::map<std::string, std::int32_t> languages_;
    std::vector<std::int32_t> suppress_, begin_suppress_;
};

} // namespace kidi::inference