#pragma once

#include "kidi/cli/argparse.h"
#include "kidi/code/openai.h"

namespace kidi::code {

auto configure_parser(cli::ArgumentParser& parser, bool managed = false) -> void;
using ProviderFactory = std::function<Result<Provider>(OpenAIOptions)>;
auto code_command(const cli::Namespace& arguments, ProviderFactory create_provider) -> int;

} // namespace kidi::code
