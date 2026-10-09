#pragma once

#include "esp32.h"

#include <cstddef>
#include <cstdint>

namespace kidi::esp32 {

enum class CameraProfile {
    PHOTO_FULL,
    PHOTO_VGA,
    VIDEO_HD,
    VIDEO_LOW,
};

enum class ImageEncoding {
    UNKNOWN,
    JPEG,
};

struct CameraOptions {
    CameraProfile profile = CameraProfile::VIDEO_HD;
    unsigned frame_buffers = 1;
    unsigned jpeg_quality = 12;
};

struct CameraFrame {
    const std::uint8_t* data = nullptr;
    std::size_t size = 0;
    std::size_t width = 0;
    std::size_t height = 0;
    std::int64_t timestamp_us = 0;
    ImageEncoding encoding = ImageEncoding::UNKNOWN;
    void* native = nullptr;
};

struct CameraBackend {
    void* context = nullptr;
    Error (*prepare)(void*, const CameraOptions&) = nullptr;
    Error (*release)(void*) = nullptr;
    Error (*acquire)(void*, CameraFrame&) = nullptr;
    void (*return_frame)(void*, CameraFrame&) = nullptr;
};

struct PhotoFrame {
    const std::uint8_t* data;
    std::size_t size;
    std::size_t width;
    std::size_t height;
};

using PhotoWriter = Error (*)(const PhotoFrame&, void*);

auto install_camera_backend(const CameraBackend& backend) -> Error;
auto camera_configured() -> bool;
auto camera_profile_width(CameraProfile profile) -> unsigned;
auto camera_profile_height(CameraProfile profile) -> unsigned;
auto acquire_camera_frame(CameraFrame& frame) -> Error;
auto return_camera_frame(CameraFrame& frame) -> void;
auto prepare_camera(const CameraOptions& options) -> Error;
auto release_camera() -> Error;
auto camera_busy() -> bool;
auto capture_photo_to(CameraProfile profile, PhotoWriter writer, void* context) -> Error;

} // namespace kidi::esp32
