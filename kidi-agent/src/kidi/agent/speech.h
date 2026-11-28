#pragma once

#include "kidi/core/error.h"

#include <cstdint>
#include <functional>
#include <map>
#include <span>
#include <string>
#include <string_view>

namespace kidi::agent {

// The agent.yaml `speech` section: Hub models for --web dictation and read-aloud. An empty model disables that
// feature.
struct SpeechConfig {
    std::string transcribe_model;
    std::string speak_model;
    std::string reference_text; // synthesized once so that every reply uses the same voice
    std::size_t steps = 16;
    float pace = 1; // duration scale; below 1 speaks faster
    std::map<std::string, std::string, std::less<>> voice;
};

// Local speech models supplied by the host (the kidi CLI), loaded on first use. Either function may be empty.
// Both may be called concurrently and must poll `cancelled`.
struct Speech {
    using Cancelled = std::function<bool()>;
    using Play = std::function<bool(std::span<const float> samples, std::uint32_t sample_rate)>;
    // Transcribes 16 kHz mono samples.
    std::function<Result<std::string>(std::span<const float> samples, const Cancelled& cancelled)> transcribe;
    // Speaks text, passing mono audio chunks to `play` in order; stops when `play` returns false.
    std::function<Result<void>(std::string_view text, const Play& play, const Cancelled& cancelled)> speak;
};

using SpeechFactory = std::function<Speech(const SpeechConfig&)>;

} // namespace kidi::agent
