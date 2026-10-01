#pragma once

#include "kidi/core/error.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace kidi::image {
inline constexpr std::size_t MAX_IMAGE_PIXELS = 24'000'000;
inline constexpr std::size_t MAX_RESIZED_PIXELS = 3'000'000;
inline constexpr std::size_t MIN_RESIZED_PIXELS = 70 * 48 * 48;

struct Gemma4Image {
    std::int32_t patch_rows = 0, patch_columns = 0;
    std::vector<float> patches;
};

auto prepare_gemma4(std::span<const std::uint8_t> encoded, std::int32_t soft_tokens = 280,
                    std::size_t max_resized_pixels = MAX_RESIZED_PIXELS) -> Result<Gemma4Image>;
} // namespace kidi::image