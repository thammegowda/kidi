#pragma once

#include "esp32.h"

#include <cstddef>
#include <cstdint>

namespace kidi::esp32 {

enum class ClipKind { AUDIO, VIDEO, AV };

using ClipWriter = Error (*)(const std::uint8_t*, std::size_t, void*);

struct MediaExecutionConfig {
    BaseType_t camera_core = tskNO_AFFINITY;
    BaseType_t audio_core = tskNO_AFFINITY;
    UBaseType_t camera_priority = 2;
    UBaseType_t audio_priority = 2;
    std::uint32_t camera_stack = 4096;
    std::uint32_t audio_stack = 8192;
};

auto initialize_media(const MediaExecutionConfig& configuration = {}) -> Error;
auto media_execution_config() -> const MediaExecutionConfig&;
auto capture_clip(ClipKind kind, unsigned seconds, ClipWriter writer, void* context) -> Error;
auto clip_busy() -> bool;
auto acquire_capture() -> Error;
auto release_capture() -> void;

} // namespace kidi::esp32
