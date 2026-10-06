#pragma once

#include <esp_err.h>
#include <esp_log.h>
#include <esp_wifi.h>
#include <freertos/queue.h>
#include <freertos/semphr.h>

namespace kidi::esp32 {

using Error = esp_err_t;
using QueueHandle = QueueHandle_t;
using SemaphoreHandle = SemaphoreHandle_t;
using WifiPowerSaveMode = wifi_ps_type_t;
using LogPrinter = vprintf_like_t;
using CriticalSection = portMUX_TYPE;

auto report_error(const char* operation, Error error) -> void;
auto read_chip_temperature(float& celsius) -> Error;

} // namespace kidi::esp32
