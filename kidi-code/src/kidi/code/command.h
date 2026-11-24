#pragma once

#include "kidi/cli/argparse.h"
#include "kidi/code/chat_client.h"

namespace kidi::code {

auto configure_parser(cli::ArgumentParser& parser, bool managed = false) -> void;
using ProviderFactory = std::function<Result<Provider>(ChatClientOptions)>;
auto code_command(const cli::Namespace& arguments, ProviderFactory create_provider) -> int;

} // namespace kidi::code
