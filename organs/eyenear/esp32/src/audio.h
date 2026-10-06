#pragma once

#include "esp32.h"

#include <cstddef>
#include <cstdint>

namespace kidi::esp32 {

constexpr std::uint32_t AUDIO_SAMPLE_RATE = 16000;

struct AudioStats {
    unsigned overruns = 0;
    unsigned dma_errors = 0;
};

struct AudioBackend {
    void* context = nullptr;
    Error (*start)(void*) = nullptr;
    Error (*read)(void*, std::uint8_t*, std::size_t) = nullptr;
    Error (*stats)(void*, AudioStats&) = nullptr;
    Error (*stop)(void*) = nullptr;
};

auto install_audio_backend(const AudioBackend& backend) -> Error;
auto audio_configured() -> bool;
auto start_audio() -> Error;
auto read_audio(std::uint8_t* destination, std::size_t length) -> Error;
auto read_audio_stats(AudioStats& stats) -> Error;
auto stop_audio() -> Error;

} // namespace kidi::esp32
