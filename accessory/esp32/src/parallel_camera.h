#pragma once

#include "camera.h"

#include <array>

namespace kidi::esp32 {

struct ParallelCameraConfig {
    int power_down = -1;
    int reset = -1;
    int clock = -1;
    int sda = -1;
    int scl = -1;
    std::array<int, 8> data = {-1, -1, -1, -1, -1, -1, -1, -1};
    int vsync = -1;
    int href = -1;
    int pixel_clock = -1;
    unsigned clock_hz = 20000000;
};

auto install_parallel_camera(const ParallelCameraConfig& configuration) -> Error;

} // namespace kidi::esp32
