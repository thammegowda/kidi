#pragma once

#include <filesystem>
#include <map>
#include <string>
#include <vector>

#include "kidi/checkpoint/config.h"
#include "kidi/checkpoint/weights.h"

namespace kidi::checkpoint {

struct Preparation {
    std::filesystem::path source;
    std::string cache_directory;
    std::string format;
    std::vector<std::string> files;
    std::map<std::string, std::string> metadata;
    auto (*convert)(const YAML::Node&, const Weights&) -> Result<StateDict> = nullptr;
};

using PreparationHook = auto (*)(const YAML::Node&) -> Result<Preparation>;

auto prepare(const std::filesystem::path& path, const ConfigAdapter& config, PreparationHook customize)
    -> Result<std::filesystem::path>;

} // namespace kidi::checkpoint