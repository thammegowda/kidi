#pragma once

#include "esp32.h"

#include <cstddef>
#include <cstdint>

namespace kidi::esp32 {

enum class ClipKind { AUDIO, VIDEO, AV };

using ClipWriter = Error (*)(const std::uint8_t*, std::size_t, void*);

auto capture_clip(ClipKind kind, unsigned seconds, ClipWriter writer, void* context) -> Error;
auto clip_busy() -> bool;
auto acquire_capture() -> Error;
auto release_capture() -> void;

} // namespace kidi::esp32
