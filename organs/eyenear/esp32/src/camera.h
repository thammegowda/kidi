#pragma once

#include "esp32.h"

#include <cstddef>
#include <cstdint>

namespace kidi::esp32 {

struct PhotoFrame {
    const std::uint8_t* data;
    std::size_t size;
    std::size_t width;
    std::size_t height;
};

using PhotoWriter = Error (*)(const PhotoFrame&, void*);

auto initialize_camera() -> Error;
auto prepare_camera(CameraFrameSize frame_size = FRAMESIZE_HD, unsigned frame_buffers = 1) -> Error;
auto release_camera() -> Error;
auto camera_busy() -> bool;
auto capture_photo_to(CameraFrameSize frame_size, PhotoWriter writer, void* context) -> Error;

} // namespace kidi::esp32
