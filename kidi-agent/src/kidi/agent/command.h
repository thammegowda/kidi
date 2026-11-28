#pragma once

#include "kidi/cli/argparse.h"
#include "kidi/agent/chat_client.h"
#include "kidi/agent/session.h"
#include "kidi/agent/speech.h"

namespace kidi::agent {

auto configure_parser(cli::ArgumentParser& parser, bool managed = false) -> void;
// create_speech: optional dictation and read-aloud for --web, built from the agent.yaml `speech` section.
auto agent_command(const cli::Namespace& arguments, ProviderFactory create_provider,
                   SpeechFactory create_speech = {}) -> int;

} // namespace kidi::agent
