#include "audio.h"
#include "camera.h"
#include "clip.h"
#include "esp32.h"
#include "parallel_camera.h"
#include "pdm_audio.h"
#include "usb.h"

#include <Arduino.h>
#include <cassert>

auto setup() -> void {
    assert(!kidi::esp32::camera_configured());
    assert(kidi::esp32::install_parallel_camera({}) == ESP_ERR_INVALID_ARG);
    assert(!kidi::esp32::audio_configured());
    assert(kidi::esp32::install_pdm_microphone({}) == ESP_ERR_INVALID_ARG);
    kidi::esp32::initialize_usb();
    Serial.println("KIDI_COMPILE_CHECK_ONLY no board pin map; do not use as device firmware");
}

auto loop() -> void { kidi::esp32::poll_usb(); }
