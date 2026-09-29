#pragma once

#include "kidi/core/error.h"

#include <cstdint>
#include <span>
#include <vector>

namespace kidi::image {
struct Gemma4Image {
    std::int32_t patch_rows = 0, patch_columns = 0;
    std::vector<float> patches;
};

auto prepare_gemma4(std::span<const std::uint8_t> encoded, std::int32_t soft_tokens = 280) -> Result<Gemma4Image>;
} // namespace kidi::image