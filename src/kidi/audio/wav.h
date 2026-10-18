#pragma once

#include <cstdint>
#include <filesystem>
#include <span>
#include <vector>

#include "kidi/core/error.h"

namespace kidi::audio {

struct Waveform {
    std::vector<float> samples;
    std::uint32_t sample_rate;
};

auto load_wav(const std::filesystem::path& path) -> Result<Waveform>;
auto save_wav(const std::filesystem::path& path, std::span<const float> samples, std::uint32_t sample_rate)
    -> Result<void>;

} // namespace kidi::audio
