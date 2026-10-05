#pragma once

#include "esp32.h"

#include <cstddef>
#include <cstdint>

namespace kidi::esp32 {

constexpr std::uint32_t AUDIO_SAMPLE_RATE = 16000;

auto start_audio(QueueHandle* events = nullptr) -> Error;
auto read_audio(std::uint8_t* destination, std::size_t length) -> Error;
auto stop_audio() -> Error;

} // namespace kidi::esp32
