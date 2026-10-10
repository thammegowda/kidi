#pragma once

#include "clip.h"

namespace kidi::esp32 {

auto initialize_usb(const MediaExecutionConfig& media = {}) -> void;
auto poll_usb() -> void;

} // namespace kidi::esp32
