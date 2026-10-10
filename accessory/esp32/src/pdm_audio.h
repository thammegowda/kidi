#pragma once

#include "audio.h"

#include <driver/i2s.h>

namespace kidi::esp32 {

struct PdmMicrophoneConfig {
    i2s_port_t port = I2S_NUM_0;
    int clock = -1;
    int data = -1;
};

auto install_pdm_microphone(const PdmMicrophoneConfig& configuration) -> Error;

} // namespace kidi::esp32
