#pragma once

#include <cstdint>
#include <filesystem>
#include <span>
#include <utility>
#include <vector>

#include "kidi/core/error.h"

namespace kidi::audio {

struct Waveform {
    std::vector<float> samples;
    std::uint32_t sample_rate;
};

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

auto load_wav(const std::filesystem::path& path) -> Result<Waveform>;

} // namespace kidi::audio