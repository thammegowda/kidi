#include "parallel_camera.h"

#include <Arduino.h>
#include <driver/gpio.h>
#include <esp_camera.h>

namespace kidi::esp32 {
namespace {

ParallelCameraConfig configuration;

auto frame_size(CameraProfile profile, framesize_t& size) -> Error {
    switch (profile) {
        case CameraProfile::PHOTO_FULL:
            size = FRAMESIZE_QXGA;
            return ESP_OK;
        case CameraProfile::PHOTO_VGA:
            size = FRAMESIZE_VGA;
            return ESP_OK;
        case CameraProfile::VIDEO_HD:
            size = FRAMESIZE_HD;
            return ESP_OK;
        case CameraProfile::VIDEO_LOW:
            size = FRAMESIZE_96X96;
            return ESP_OK;
    }
    return ESP_ERR_INVALID_ARG;
}

auto prepare(void*, const CameraOptions& options) -> Error {
    framesize_t size;
    auto result = frame_size(options.profile, size);
    if (result != ESP_OK) return result;
    camera_config_t sdk{};
    sdk.pin_pwdn = configuration.power_down;
    sdk.pin_reset = configuration.reset;
    sdk.pin_xclk = configuration.clock;
    sdk.pin_sccb_sda = configuration.sda;
    sdk.pin_sccb_scl = configuration.scl;
    sdk.pin_d0 = configuration.data[0];
    sdk.pin_d1 = configuration.data[1];
    sdk.pin_d2 = configuration.data[2];
    sdk.pin_d3 = configuration.data[3];
    sdk.pin_d4 = configuration.data[4];
    sdk.pin_d5 = configuration.data[5];
    sdk.pin_d6 = configuration.data[6];
    sdk.pin_d7 = configuration.data[7];
    sdk.pin_vsync = configuration.vsync;
    sdk.pin_href = configuration.href;
    sdk.pin_pclk = configuration.pixel_clock;
    sdk.xclk_freq_hz = configuration.clock_hz;
    sdk.ledc_timer = LEDC_TIMER_0;
    sdk.ledc_channel = LEDC_CHANNEL_0;
    sdk.pixel_format = PIXFORMAT_JPEG;
    sdk.frame_size = size;
    sdk.jpeg_quality = options.jpeg_quality;
    sdk.fb_count = options.frame_buffers;
    sdk.fb_location = CAMERA_FB_IN_PSRAM;
    sdk.grab_mode = options.frame_buffers > 1 ? CAMERA_GRAB_LATEST : CAMERA_GRAB_WHEN_EMPTY;
    result = esp_camera_init(&sdk);
    if (result != ESP_OK) return result;
    const auto* sensor = esp_camera_sensor_get();
    if (sensor == nullptr) {
        esp_camera_deinit();
        return ESP_FAIL;
    }
    Serial.printf("KIDI_SENSOR pid=0x%04x\n", sensor->id.PID);
    delay(500);
    for (int index = 0; index < 3; ++index) {
        auto* frame = esp_camera_fb_get();
        if (frame == nullptr) {
            esp_camera_deinit();
            return ESP_FAIL;
        }
        esp_camera_fb_return(frame);
        delay(100);
    }
    return ESP_OK;
}

auto release(void*) -> Error { return esp_camera_deinit(); }

auto acquire(void*, CameraFrame& frame) -> Error {
    auto* native = esp_camera_fb_get();
    if (native == nullptr) return ESP_FAIL;
    frame = {
        native->buf,
        native->len,
        native->width,
        native->height,
        native->timestamp.tv_sec * 1000000LL + native->timestamp.tv_usec,
        ImageEncoding::JPEG,
        native,
    };
    return ESP_OK;
}

auto return_frame(void*, CameraFrame& frame) -> void { esp_camera_fb_return(static_cast<camera_fb_t*>(frame.native)); }

auto valid_configuration(const ParallelCameraConfig& candidate) -> bool {
    const std::array<int, 16> pins = {
        candidate.clock,   candidate.sda,         candidate.scl,        candidate.data[0],
        candidate.data[1], candidate.data[2],     candidate.data[3],    candidate.data[4],
        candidate.data[5], candidate.data[6],     candidate.data[7],    candidate.vsync,
        candidate.href,    candidate.pixel_clock, candidate.power_down, candidate.reset,
    };
    if (candidate.clock_hz == 0) return false;
    for (std::size_t index = 0; index < pins.size(); ++index) {
        if (index >= pins.size() - 2 && pins[index] == -1) continue;
        if (pins[index] < 0 || !GPIO_IS_VALID_GPIO(pins[index])) return false;
        for (std::size_t previous = 0; previous < index; ++previous)
            if (pins[previous] == pins[index]) return false;
    }
    for (const auto pin : {candidate.clock, candidate.sda, candidate.scl, candidate.power_down, candidate.reset})
        if (pin != -1 && !GPIO_IS_VALID_OUTPUT_GPIO(pin)) return false;
    return true;
}

} // namespace

auto install_parallel_camera(const ParallelCameraConfig& candidate) -> Error {
    if (!valid_configuration(candidate)) return ESP_ERR_INVALID_ARG;
    const auto result = install_camera_backend({nullptr, prepare, release, acquire, return_frame});
    if (result == ESP_OK) configuration = candidate;
    return result;
}

} // namespace kidi::esp32
