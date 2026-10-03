#include "esp32.h"

#include <cmath>

#include <Arduino.h>

namespace kidi::esp32 {

auto report_error(const char* operation, Error error) -> void {
    Serial.printf("KIDI_ERROR %s: %s (0x%x)\n", operation, esp_err_to_name(error), static_cast<unsigned>(error));
}

auto read_chip_temperature(float& celsius) -> Error {
    TemperatureConfig config = TSENS_CONFIG_DEFAULT();
    config.dac_offset = TSENS_DAC_L1;
    auto result = temp_sensor_set_config(config);
    if (result != ESP_OK) return result;
    result = temp_sensor_start();
    if (result != ESP_OK) return result;
    result = temp_sensor_read_celsius(&celsius);
    const auto stop_result = temp_sensor_stop();
    if (result != ESP_OK) return result;
    if (stop_result != ESP_OK) return stop_result;
    return std::isfinite(celsius) ? ESP_OK : ESP_FAIL;
}

} // namespace kidi::esp32
