#pragma once

#include <map>
#include <string>
#include <string_view>

#include "kidi/core/error.h"

namespace YAML {
class Node;
}

namespace kidi::app {

using Prompts = std::map<std::string, std::string, std::less<>>;

auto application_prompts(const YAML::Node& config, std::string_view runtime = {}) -> Result<Prompts>;

} // namespace kidi::app
