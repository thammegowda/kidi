#pragma once

#include <filesystem>
#include <functional>
#include <map>
#include <string>
#include <string_view>

#include <yaml-cpp/yaml.h>

#include "kidi/audio/whisper.h"
#include "kidi/model/whisper.h"
#include "kidi/text/tokenizer.h"
#include "kidi/inference/coroutine.h"

namespace kidi::inference {

struct TranscriptionOptions {
    std::string language = "auto";
    std::string task = "transcribe";
    std::size_t maximum_tokens = 128;
    /// Encodes the audio plus at least one second of silence, rounded up to whole encoder slices, instead of the full
    /// 30-second window. Short dictation encodes several times faster; the margin keeps Whisper from repeating text.
    bool fit_audio = false;
    std::function<void(std::string_view, std::string_view)> on_partial;
    /// Polled between stages and decoder steps; returning true abandons the request with an error.
    std::function<bool()> cancelled;
};

struct TranscriptionStats {
    std::uint64_t feature_ns = 0, encode_ns = 0, decode_ns = 0, preparation_ns = 0;
    std::size_t attempts = 1;
};

struct Transcription {
    std::string text, language;
    std::vector<std::int32_t> token_ids;
    TranscriptionStats stats;
};

class TranscriptionSession {
public:
    virtual ~TranscriptionSession() = default;
    virtual auto step() -> Result<std::optional<Transcription>> = 0;
};

class Transcriber {
public:
    static auto load(const std::filesystem::path& directory,
                     tensor::Device device = tensor::Device::cpu()) -> Result<Transcriber>;
    auto transcribe(std::span<const float> waveform, std::uint32_t sample_rate,
                    const TranscriptionOptions& options = {}) -> Result<Transcription>;
    auto start(std::span<const float> waveform, std::uint32_t sample_rate, const TranscriptionOptions& options = {})
        -> Result<std::unique_ptr<TranscriptionSession>>;
    /// Transcribes a second of silence so weights are packed and operators prepared before the first request.
    auto warm_up() -> Result<void>;

    /// Where transcription runs, e.g. "cpu".
    auto execution() const -> std::string { return std::string(tensor::to_string(model_->device().kind)); }

private:
    using Session = CoroutineSession<Transcription, TranscriptionSession>;
    auto transcription_session_(std::vector<float> waveform, std::uint32_t sample_rate,
                                 TranscriptionOptions options) -> Session;
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