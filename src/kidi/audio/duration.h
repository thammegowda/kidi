#pragma once

#include <optional>
#include <string_view>

#include "kidi/core/error.h"

namespace kidi::audio {

class RuleDurationEstimator {
public:
    auto weight(std::string_view text) const -> Result<float>;
    auto estimate(std::string_view target, std::string_view reference, float reference_duration,
                  std::optional<float> low_threshold = 50.F, float boost_strength = 3.F) const -> Result<float>;
};

} // namespace kidi::audio
