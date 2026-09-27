#pragma once

#include <filesystem>
#include <yaml-cpp/yaml.h>

#include "kidi/core/error.h"

namespace kidi::model {

auto load_config(const std::filesystem::path& path) -> Result<YAML::Node>;
auto load_whisper_config(const std::filesystem::path& directory) -> Result<YAML::Node>;

} // namespace kidi::model