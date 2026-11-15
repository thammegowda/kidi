#pragma once

#include <filesystem>
#include <string>
#include <vector>
#include <yaml-cpp/yaml.h>

#include "kidi/core/error.h"

namespace kidi::checkpoint {

auto load_config(const std::filesystem::path& path) -> Result<YAML::Node>;
auto hub_files(const YAML::Node& document) -> Result<std::vector<std::string>>;

} // namespace kidi::checkpoint