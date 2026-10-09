#include "camera.h"
#include "clip.h"

namespace kidi::esp32 {
namespace {

SemaphoreHandle CAMERA_MUTEX = nullptr;
CameraBackend camera_backend;

auto valid_profile(CameraProfile profile) -> bool {
    switch (profile) {
        case CameraProfile::PHOTO_FULL:
        case CameraProfile::PHOTO_VGA:
        case CameraProfile::VIDEO_HD:
        case CameraProfile::VIDEO_LOW:
            return true;
    }
    return false;
}

} // namespace

auto install_camera_backend(const CameraBackend& backend) -> Error {
    if (camera_configured()) return ESP_ERR_INVALID_STATE;
    if (backend.prepare == nullptr || backend.release == nullptr || backend.acquire == nullptr ||
        backend.return_frame == nullptr)
        return ESP_ERR_INVALID_ARG;
    auto* mutex = xSemaphoreCreateMutex();
    if (mutex == nullptr) return ESP_ERR_NO_MEM;
    camera_backend = backend;
    CAMERA_MUTEX = mutex;
    return ESP_OK;
}

auto camera_configured() -> bool { return CAMERA_MUTEX != nullptr; }

auto camera_profile_width(CameraProfile profile) -> unsigned {
    switch (profile) {
        case CameraProfile::PHOTO_FULL:
            return 2048;
        case CameraProfile::PHOTO_VGA:
            return 640;
        case CameraProfile::VIDEO_HD:
            return 1280;
        case CameraProfile::VIDEO_LOW:
            return 96;
    }
    return 0;
}

auto camera_profile_height(CameraProfile profile) -> unsigned {
    switch (profile) {
        case CameraProfile::PHOTO_FULL:
            return 1536;
        case CameraProfile::PHOTO_VGA:
            return 480;
        case CameraProfile::VIDEO_HD:
            return 720;
        case CameraProfile::VIDEO_LOW:
            return 96;
    }
    return 0;
}

auto camera_busy() -> bool { return camera_configured() && uxSemaphoreGetCount(CAMERA_MUTEX) == 0; }

auto acquire_camera_frame(CameraFrame& frame) -> Error {
    if (!camera_busy()) return ESP_ERR_INVALID_STATE;
    frame = {};
    const auto result = camera_backend.acquire(camera_backend.context, frame);
    if (result != ESP_OK) return result;
    if (frame.data == nullptr || frame.size == 0 || frame.width == 0 || frame.height == 0 ||
        frame.encoding != ImageEncoding::JPEG || frame.native == nullptr) {
        if (frame.native != nullptr) camera_backend.return_frame(camera_backend.context, frame);
        frame = {};
        return ESP_ERR_INVALID_RESPONSE;
    }
    return ESP_OK;
}

auto return_camera_frame(CameraFrame& frame) -> void {
    if (frame.native == nullptr) {
        report_error("returning an invalid camera frame", ESP_ERR_INVALID_ARG);
        return;
    }
    camera_backend.return_frame(camera_backend.context, frame);
    frame = {};
}

auto prepare_camera(const CameraOptions& options) -> Error {
    if (!camera_configured()) return ESP_ERR_INVALID_STATE;
    if (!valid_profile(options.profile) || options.frame_buffers == 0 || options.jpeg_quality > 63)
        return ESP_ERR_INVALID_ARG;
    if (xSemaphoreTake(CAMERA_MUTEX, 0) != pdTRUE) return ESP_ERR_TIMEOUT;
    const auto result = camera_backend.prepare(camera_backend.context, options);
    if (result != ESP_OK) xSemaphoreGive(CAMERA_MUTEX);
    return result;
}

auto release_camera() -> Error {
    if (!camera_busy()) return ESP_ERR_INVALID_STATE;
    const auto result = camera_backend.release(camera_backend.context);
    xSemaphoreGive(CAMERA_MUTEX);
    return result;
}

auto capture_photo_to(CameraProfile profile, PhotoWriter writer, void* context) -> Error {
    if (writer == nullptr || (profile != CameraProfile::PHOTO_FULL && profile != CameraProfile::PHOTO_VGA))
        return ESP_ERR_INVALID_ARG;
    const auto acquired = acquire_capture();
    if (acquired != ESP_OK) return acquired;
    auto result = prepare_camera({profile, 1, 12});
    if (result != ESP_OK) {
        release_capture();
        return result;
    }
    CameraFrame frame;
    result = acquire_camera_frame(frame);
    if (result == ESP_OK) {
        if (frame.width != camera_profile_width(profile) || frame.height != camera_profile_height(profile))
            result = ESP_ERR_INVALID_RESPONSE;
        else
            result = writer({frame.data, frame.size, frame.width, frame.height}, context);
        return_camera_frame(frame);
    }
    const auto cleanup = release_camera();
    release_capture();
    if (result != ESP_OK && cleanup != ESP_OK) report_error("camera shutdown", cleanup);
    return result != ESP_OK ? result : cleanup;
}

} // namespace kidi::esp32
