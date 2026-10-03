#pragma once

#include <array>

namespace kidi::esp32 {

struct W11Pins {
    static constexpr int CAMERA_POWER_DOWN = -1;
    static constexpr int CAMERA_RESET = -1;
    static constexpr int CAMERA_CLOCK = 10;
    static constexpr int CAMERA_SDA = 40;
    static constexpr int CAMERA_SCL = 39;
    static constexpr std::array<int, 8> CAMERA_DATA = {15, 17, 18, 16, 14, 12, 11, 48};
    static constexpr int CAMERA_VSYNC = 38;
    static constexpr int CAMERA_HREF = 47;
    static constexpr int CAMERA_PIXEL_CLOCK = 13;
    static constexpr int MICROPHONE_CLOCK = 42;
    static constexpr int MICROPHONE_DATA = 41;
    static constexpr int SENSOR_SDA = 5;
    static constexpr int SENSOR_SCL = 4;
    static constexpr int BATTERY_ADC = 2;
    static constexpr float BATTERY_DIVIDER = 2.0F;
};

} // namespace kidi::esp32
