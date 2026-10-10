#pragma once

#include <cstdint>
#include <filesystem>
#include <span>
#include <utility>
#include <vector>

#include "kidi/audio/wav.h"

namespace kidi::audio {

struct WhisperFeatures {
    std::vector<float> values;
    std::size_t bins;
    std::size_t frames;
};

class WhisperFeatureExtractor {
public:
    static auto load(const std::filesystem::path& path) -> Result<WhisperFeatureExtractor>;
    auto extract(std::span<const float> waveform, std::uint32_t sample_rate) const -> Result<WhisperFeatures>;

private:
    explicit WhisperFeatureExtractor(std::vector<float> filters) : filters_(std::move(filters)) {}

    std::vector<float> filters_;
};

} // namespace kidi::audio