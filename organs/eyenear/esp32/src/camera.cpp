#include "camera.h"
#include "clip.h"
#include "w11.h"

#include <Arduino.h>

namespace kidi::esp32 {
namespace {

SemaphoreHandle CAMERA_MUTEX = nullptr;

} // namespace

auto initialize_camera() -> Error {
    CAMERA_MUTEX = xSemaphoreCreateMutex();
    return CAMERA_MUTEX == nullptr ? ESP_ERR_NO_MEM : ESP_OK;
}

auto camera_busy() -> bool { return CAMERA_MUTEX == nullptr || uxSemaphoreGetCount(CAMERA_MUTEX) == 0; }

auto release_camera() -> Error {
    const auto result = esp_camera_deinit();
    xSemaphoreGive(CAMERA_MUTEX);
    return result;
}

auto prepare_camera(CameraFrameSize frame_size, unsigned frame_buffers) -> Error {
    if (CAMERA_MUTEX == nullptr) return ESP_ERR_INVALID_STATE;
    if (xSemaphoreTake(CAMERA_MUTEX, 0) != pdTRUE) return ESP_ERR_TIMEOUT;
    CameraConfig config{};
    config.pin_pwdn = W11Pins::CAMERA_POWER_DOWN;
    config.pin_reset = W11Pins::CAMERA_RESET;
    config.pin_xclk = W11Pins::CAMERA_CLOCK;
    config.pin_sccb_sda = W11Pins::CAMERA_SDA;
    config.pin_sccb_scl = W11Pins::CAMERA_SCL;
    config.pin_d0 = W11Pins::CAMERA_DATA[0];
    config.pin_d1 = W11Pins::CAMERA_DATA[1];
    config.pin_d2 = W11Pins::CAMERA_DATA[2];
    config.pin_d3 = W11Pins::CAMERA_DATA[3];
    config.pin_d4 = W11Pins::CAMERA_DATA[4];
    config.pin_d5 = W11Pins::CAMERA_DATA[5];
    config.pin_d6 = W11Pins::CAMERA_DATA[6];
    config.pin_d7 = W11Pins::CAMERA_DATA[7];
    config.pin_vsync = W11Pins::CAMERA_VSYNC;
    config.pin_href = W11Pins::CAMERA_HREF;
    config.pin_pclk = W11Pins::CAMERA_PIXEL_CLOCK;
    config.xclk_freq_hz = 20000000;
    config.ledc_timer = LEDC_TIMER_0;
    config.ledc_channel = LEDC_CHANNEL_0;
    config.pixel_format = PIXFORMAT_JPEG;
    config.frame_size = frame_size;
    config.jpeg_quality = 12;
    config.fb_count = frame_buffers;
    config.fb_location = CAMERA_FB_IN_PSRAM;
    config.grab_mode = frame_buffers > 1 ? CAMERA_GRAB_LATEST : CAMERA_GRAB_WHEN_EMPTY;
    auto result = esp_camera_init(&config);
    if (result != ESP_OK) {
        xSemaphoreGive(CAMERA_MUTEX);
        return result;
    }
    const auto* sensor = esp_camera_sensor_get();
    if (sensor == nullptr) {
        const auto cleanup = release_camera();
        if (cleanup != ESP_OK) report_error("camera shutdown", cleanup);
        return ESP_FAIL;
    }
    Serial.printf("KIDI_SENSOR pid=0x%04x\n", sensor->id.PID);
    delay(500);
    for (int index = 0; index < 3; ++index) {
        auto* frame = esp_camera_fb_get();
        if (frame == nullptr) {
            const auto cleanup = release_camera();
            if (cleanup != ESP_OK) report_error("camera shutdown", cleanup);
            return ESP_FAIL;
        }
        esp_camera_fb_return(frame);
        delay(100);
    }
    return ESP_OK;
}

auto capture_photo_to(CameraFrameSize frame_size, PhotoWriter writer, void* context) -> Error {
    const auto acquired = acquire_capture();
    if (acquired != ESP_OK) return acquired;
    auto result = prepare_camera(frame_size);
    if (result != ESP_OK) {
        release_capture();
        return result;
    }
    auto* frame = esp_camera_fb_get();
    if (frame == nullptr) {
        result = ESP_FAIL;
    } else {
        result = writer({frame->buf, frame->len, frame->width, frame->height}, context);
        esp_camera_fb_return(frame);
    }
    const auto cleanup = release_camera();
    release_capture();
    if (result != ESP_OK && cleanup != ESP_OK) report_error("camera shutdown", cleanup);
    return result != ESP_OK ? result : cleanup;
}

} // namespace kidi::esp32
