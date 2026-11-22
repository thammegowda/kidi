#pragma once
// Normalizes checkpoint manifests and lists the files needed to download a model package.

#include <filesystem>
#include <string>
#include <vector>
#include <yaml-cpp/yaml.h>

#include "kidi/core/error.h"

namespace kidi::checkpoint {

auto load_config(const std::filesystem::path& path) -> Result<YAML::Node>;
auto hub_files(const YAML::Node& document) -> Result<std::vector<std::string>>;
auto native_qat(const YAML::Node& model_config) -> bool;

} // namespace kidi::checkpoint