#pragma once

#include <Arduino.h>

namespace kidi::esp32 {

auto initialize_wireless() -> void;
auto poll_wireless() -> void;
auto handle_wireless_command(const String& command) -> bool;

} // namespace kidi::esp32
