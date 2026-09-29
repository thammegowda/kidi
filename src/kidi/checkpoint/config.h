#pragma once

#include <filesystem>
#include <span>
#include <string_view>
#include <yaml-cpp/yaml.h>

#include "kidi/core/error.h"

namespace kidi::checkpoint {

struct ConfigSource {
    std::filesystem::path root;
    std::filesystem::path explicit_weights;
    YAML::Node document;

    auto file(std::string_view name) const -> Result<std::filesystem::path>;
    auto weights(std::span<const std::string_view> candidates) const -> Result<YAML::Node>;
};

struct ConfigAdapter {
    std::string_view filename;
    auto (*configure)(const ConfigSource&) -> Result<YAML::Node> = nullptr;
};

auto load_config(const std::filesystem::path& path) -> Result<YAML::Node>;
auto load_config(const std::filesystem::path& path, const ConfigAdapter& adapter) -> Result<YAML::Node>;

} // namespace kidi::checkpoint