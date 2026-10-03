#include "sensors.h"
#include "esp32.h"
#include "w11.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>

#include <Arduino.h>
#include <SensorQMI8658.hpp>
#include <Wire.h>

namespace kidi::esp32 {
namespace {

constexpr std::uint32_t I2C_FREQUENCY = 400000;
constexpr std::array<std::uint8_t, 2> IMU_ADDRESSES = {0x6a, 0x6b};
constexpr std::array<std::uint8_t, 2> TEMPERATURE_ADDRESSES = {0x4a, 0x4b};

struct ImuReading {
    std::array<float, 3> acceleration{};
    std::array<float, 3> gyroscope{};
    float temperature = 0;
    const char* error = nullptr;
};

struct TemperatureReading {
    float celsius = 0;
    const char* error = nullptr;
};

auto find_address(TwoWire& bus, const std::array<std::uint8_t, 2>& addresses) -> std::uint8_t {
    for (const auto address : addresses) {
        bus.beginTransmission(address);
        if (bus.endTransmission() == 0) return address;
    }
    return 0;
}

auto read_imu(TwoWire& bus) -> ImuReading {
    ImuReading reading;
    const auto address = find_address(bus, IMU_ADDRESSES);
    if (address == 0) {
        reading.error = "QMI8658 not detected";
        return reading;
    }
    SensorQMI8658 imu;
    if (!imu.begin(bus, address)) {
        reading.error = "QMI8658 initialization failed";
        return reading;
    }
    if (imu.configAccelerometer(SensorQMI8658::ACC_RANGE_4G, SensorQMI8658::ACC_ODR_1000Hz) != 0 ||
        imu.configGyroscope(SensorQMI8658::GYR_RANGE_256DPS, SensorQMI8658::GYR_ODR_896_8Hz) != 0 ||
        !imu.enableAccelerometer() || !imu.enableGyroscope()) {
        reading.error = "QMI8658 configuration failed";
    } else {
        const auto start = millis();
        auto ready = imu.getDataReady();
        while (!ready && millis() - start < 200) {
            delay(1);
            ready = imu.getDataReady();
        }
        if (!ready ||
            !imu.getAccelerometer(reading.acceleration[0], reading.acceleration[1], reading.acceleration[2]) ||
            !imu.getGyroscope(reading.gyroscope[0], reading.gyroscope[1], reading.gyroscope[2])) {
            reading.error = "QMI8658 data read failed";
        } else {
            reading.temperature = imu.getTemperature_C();
            const auto finite = [](float value) { return std::isfinite(value); };
            if (!std::isfinite(reading.temperature) ||
                !std::all_of(reading.acceleration.begin(), reading.acceleration.end(), finite) ||
                !std::all_of(reading.gyroscope.begin(), reading.gyroscope.end(), finite)) {
                reading.error = "QMI8658 returned non-finite data";
            }
        }
    }
    const auto acceleration_stopped = imu.disableAccelerometer();
    const auto gyroscope_stopped = imu.disableGyroscope();
    if (!acceleration_stopped || !gyroscope_stopped) reading.error = "QMI8658 shutdown failed";
    return reading;
}

auto temperature_crc(std::uint8_t high, std::uint8_t low) -> std::uint8_t {
    std::uint8_t crc = 0xff;
    for (const auto byte : {high, low}) {
        crc ^= byte;
        for (int bit = 0; bit < 8; ++bit) {
            crc = static_cast<std::uint8_t>((crc & 0x80) ? (crc << 1) ^ 0x31 : crc << 1);
        }
    }
    return crc;
}

auto read_board_temperature(TwoWire& bus) -> TemperatureReading {
    TemperatureReading reading;
    const auto address = find_address(bus, TEMPERATURE_ADDRESSES);
    if (address == 0) {
        reading.error = "STS35 not detected";
        return reading;
    }
    bus.beginTransmission(address);
    bus.write(0x24);
    bus.write(0x00);
    if (bus.endTransmission() != 0) {
        reading.error = "STS35 measurement command failed";
        return reading;
    }
    delay(20);
    if (bus.requestFrom(address, static_cast<std::size_t>(3), true) != 3) {
        reading.error = "STS35 incomplete measurement";
        return reading;
    }
    const auto high = static_cast<std::uint8_t>(bus.read());
    const auto low = static_cast<std::uint8_t>(bus.read());
    const auto crc = static_cast<std::uint8_t>(bus.read());
    if (temperature_crc(high, low) != crc) {
        reading.error = "STS35 measurement CRC mismatch";
        return reading;
    }
    const auto raw = static_cast<std::uint16_t>((static_cast<std::uint16_t>(high) << 8) | low);
    reading.celsius = -45.0F + 175.0F * raw / 65535.0F;
    return reading;
}

auto print_temperature_reading(const char* name, const TemperatureReading& reading) -> void {
    Serial.printf("\"%s\":{", name);
    if (reading.error != nullptr) {
        Serial.printf("\"available\":false,\"error\":\"%s\"}", reading.error);
    } else {
        Serial.printf("\"available\":true,\"celsius\":%.3f}", reading.celsius);
    }
}

} // namespace

auto print_temperature() -> void {
    float celsius = 0;
    const auto result = read_chip_temperature(celsius);
    if (result != ESP_OK) {
        report_error("chip temperature", result);
        return;
    }
    Serial.printf("KIDI_TEMPERATURE celsius=%.2f cpu_mhz=%u\n", celsius, getCpuFrequencyMhz());
}

auto print_sensors() -> void {
    TemperatureReading die;
    const auto temperature_result = read_chip_temperature(die.celsius);
    if (temperature_result != ESP_OK) die.error = esp_err_to_name(temperature_result);

    // Keep the board's sensor bus separate from the camera's SCCB controller.
    TwoWire bus(1);
    const auto bus_started = bus.begin(W11Pins::SENSOR_SDA, W11Pins::SENSOR_SCL, I2C_FREQUENCY);
    ImuReading imu;
    TemperatureReading board;
    if (bus_started) {
        imu = read_imu(bus);
        board = read_board_temperature(bus);
    } else {
        imu.error = "sensor I2C initialization failed";
        board.error = imu.error;
    }
    const auto bus_stopped = !bus_started || bus.end();

    analogSetPinAttenuation(W11Pins::BATTERY_ADC, ADC_11db);
    std::uint32_t millivolts = 0;
    for (int sample = 0; sample < 16; ++sample) millivolts += analogReadMilliVolts(W11Pins::BATTERY_ADC);
    const auto rail_voltage = millivolts / 16.0F / 1000.0F * W11Pins::BATTERY_DIVIDER;

    Serial.printf("KIDI_SENSORS {\"schema_version\":1,\"timestamp_us\":%llu,\"cpu_mhz\":%u,",
                  static_cast<unsigned long long>(esp_timer_get_time()), getCpuFrequencyMhz());
    print_temperature_reading("chip_temperature", die);
    Serial.print(",\"imu\":{");
    if (imu.error != nullptr) {
        Serial.printf("\"available\":false,\"error\":\"%s\"}", imu.error);
    } else {
        Serial.printf(
            "\"available\":true,\"acceleration_g\":[%.6f,%.6f,%.6f],"
            "\"gyroscope_dps\":[%.6f,%.6f,%.6f],\"temperature_c\":%.3f}",
            imu.acceleration[0], imu.acceleration[1], imu.acceleration[2], imu.gyroscope[0], imu.gyroscope[1],
            imu.gyroscope[2], imu.temperature);
    }
    Serial.print(",");
    print_temperature_reading("board_temperature", board);
    Serial.printf(",\"battery_rail\":{\"available\":true,\"voltage_v\":%.3f,\"battery_presence\":\"unknown\"}",
                  rail_voltage);
    if (!bus_stopped) Serial.print(",\"i2c_error\":\"sensor I2C shutdown failed\"");
    Serial.println("}");
}

} // namespace kidi::esp32
