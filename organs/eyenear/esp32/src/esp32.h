#pragma once

#include <driver/i2s.h>
#include <driver/temp_sensor.h>
#include <esp_camera.h>
#include <esp_err.h>
#include <freertos/queue.h>
#include <freertos/semphr.h>

namespace kidi::esp32 {

using Error = esp_err_t;
using CameraConfig = camera_config_t;
using CameraFrame = camera_fb_t;
using CameraFrameSize = framesize_t;
using I2sConfig = i2s_config_t;
using I2sPins = i2s_pin_config_t;
using I2sPort = i2s_port_t;
using I2sMode = i2s_mode_t;
using I2sEvent = i2s_event_t;
using QueueHandle = QueueHandle_t;
using SemaphoreHandle = SemaphoreHandle_t;
using TemperatureConfig = temp_sensor_config_t;

auto report_error(const char* operation, Error error) -> void;
auto read_chip_temperature(float& celsius) -> Error;

} // namespace kidi::esp32
