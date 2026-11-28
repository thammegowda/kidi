#pragma once

#include "kidi/agent/speech.h"

#include <string>

namespace kidi::cli {

// Whisper dictation and OmniVoice read-aloud for `kidi agent --web`, configured by agent.yaml and downloaded and
// loaded on first use on the device that `backend` (the agent's --backend) selects.
auto agent_speech(std::string backend, const agent::SpeechConfig& config) -> agent::Speech;

} // namespace kidi::cli
