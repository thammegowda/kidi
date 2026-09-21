#pragma once

#include "kidi/core/error.h"
#include <filesystem>
#include <functional>
#include <string_view>

namespace kidi::cli {
using ModelResolver = std::function<Result<std::filesystem::path>(std::string_view, const std::filesystem::path&)>;
auto main(int argc, const char* const argv[], const ModelResolver& resolve_model = {}) -> int;
} // namespace kidi::cli