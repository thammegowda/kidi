#pragma once

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <span>
#include <vector>

#include "kidi/core/error.h"

namespace kidi::audio {

struct Waveform {
    std::vector<float> samples;
    std::uint32_t sample_rate;
};

class WavReader {
public:
    static auto open(const std::filesystem::path& path) -> Result<WavReader>;
    /// Reads at most maximum_samples mono frames; an empty vector indicates end of audio.
    auto read(std::size_t maximum_samples) -> Result<std::vector<float>>;
    auto sample_rate() const noexcept -> std::uint32_t { return sample_rate_; }
    auto sample_count() const noexcept -> std::uint64_t { return sample_count_; }

private:
    WavReader(std::ifstream input, std::uint32_t sample_rate, std::uint16_t format, std::uint16_t channels,
              std::uint16_t bits, std::uint64_t data_bytes);

    std::ifstream input_;
    std::uint32_t sample_rate_;
    std::uint16_t format_, channels_, bits_;
    std::uint64_t remaining_bytes_, sample_count_;
};

auto load_wav(const std::filesystem::path& path) -> Result<Waveform>;
auto save_wav(const std::filesystem::path& path, std::span<const float> samples, std::uint32_t sample_rate)
    -> Result<void>;

} // namespace kidi::audio
