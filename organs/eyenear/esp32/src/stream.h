#pragma once

#include "clip.h"
#include "camera.h"

namespace kidi::esp32 {

constexpr Error STREAM_CANCELLED = ESP_ERR_NOT_FINISHED;
constexpr unsigned STREAM_TARGET_FPS = 10;
constexpr unsigned STREAM_JPEG_QUALITY = 16;
constexpr std::size_t STREAM_SEND_BYTES = 4096;

auto stream_media(bool with_audio, unsigned seconds, ClipWriter writer, void* context,
                  CameraProfile profile = CameraProfile::VIDEO_HD, std::size_t write_quantum = STREAM_SEND_BYTES)
    -> Error;

} // namespace kidi::esp32
