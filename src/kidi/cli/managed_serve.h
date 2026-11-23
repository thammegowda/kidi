#pragma once

#include "kidi/code/command.h"
#include "kidi/cli/serve.h"

namespace kidi::cli {

auto configure_managed_code_parser(ArgumentParser& parser) -> void;
auto configure_managed_serve_parser(ArgumentParser& parser) -> void;
auto managed_code_command(const Namespace& arguments) -> int;
auto protect_managed_lock(const Namespace& arguments) -> void;
auto serve_managed(inference::Generator& generator, const ServeOptions& options, const Namespace& arguments) -> int;

} // namespace kidi::cli
