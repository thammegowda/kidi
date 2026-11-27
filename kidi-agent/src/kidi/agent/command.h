#pragma once

#include "kidi/cli/argparse.h"
#include "kidi/agent/chat_client.h"

namespace kidi::agent {

auto configure_parser(cli::ArgumentParser& parser, bool managed = false) -> void;
using ProviderFactory = std::function<Result<Provider>(ChatClientOptions)>;
auto agent_command(const cli::Namespace& arguments, ProviderFactory create_provider) -> int;

} // namespace kidi::agent
