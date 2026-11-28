#pragma once

#include "kidi/agent/session.h"
#include "kidi/agent/speech.h"

#include <cstdint>
#include <filesystem>

namespace kidi::agent {

// Serves the agent's web UI on a loopback port until interrupted. Returns the process exit status.
// A non-empty assets_directory serves the UI live from disk instead of the embedded copy.
auto web_agent(SessionOptions options, Speech speech, std::int32_t port,
               const std::filesystem::path& assets_directory = {}) -> int;

} // namespace kidi::agent
