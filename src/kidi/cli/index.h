#pragma once

#include "kidi/cli/argparse.h"
#include <filesystem>

namespace kidi::cli {
auto configure_index_parser(ArgumentParser& parser) -> void;
auto index_command(const Namespace& arguments, const std::filesystem::path& directory) -> int;
} // namespace kidi::cli
