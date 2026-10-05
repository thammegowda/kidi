#pragma once

#include "clip.h"

namespace kidi::esp32 {

constexpr Error STREAM_CANCELLED = ESP_ERR_NOT_FINISHED;

auto stream_media(bool with_audio, unsigned seconds, ClipWriter writer, void* context,
                  CameraFrameSize frame_size = FRAMESIZE_HD) -> Error;

} // namespace kidi::esp32
