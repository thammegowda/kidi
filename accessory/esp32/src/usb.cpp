#include "usb.h"
#include "audio.h"
#include "camera.h"
#include "clip.h"
#include "esp32.h"
#include "stream.h"
#include "wireless.h"

#include <Arduino.h>

namespace kidi::esp32 {
namespace {

constexpr std::size_t USB_RX_BUFFER_SIZE = 8192;
constexpr std::size_t USB_TX_BUFFER_SIZE = 4096;
constexpr std::uint32_t USB_TX_TIMEOUT_MS = 1000;

auto write_usb_clip(const std::uint8_t* data, std::size_t length, void*) -> Error {
    if (!Serial) return ESP_ERR_INVALID_STATE;
    if (Serial.available()) {
        auto command = Serial.readStringUntil('\n');
        command.trim();
        if (command == "stream-stop") return STREAM_CANCELLED;
        Serial.println("KIDI_ERROR unexpected command during media transfer");
        return ESP_ERR_INVALID_ARG;
    }
    return Serial.write(data, length) == length ? ESP_OK : ESP_FAIL;
}

auto write_usb_photo(const PhotoFrame& frame, void*) -> Error {
    Serial.printf("KIDI_PHOTO width=%u height=%u\n", static_cast<unsigned>(frame.width),
                  static_cast<unsigned>(frame.height));
    Serial.printf("KIDI_BEGIN jpeg %u\n", static_cast<unsigned>(frame.size));
    const auto written = Serial.write(frame.data, frame.size);
    Serial.println();
    Serial.println("KIDI_END");
    return written == frame.size ? ESP_OK : ESP_FAIL;
}

auto capture_photo(CameraProfile profile = CameraProfile::PHOTO_FULL) -> void {
    const auto result = capture_photo_to(profile, write_usb_photo, nullptr);
    if (result != ESP_OK) report_error("USB photo", result);
}

auto print_status() -> void {
    Serial.printf(
        "KIDI_STATUS flash=%u psram=%u free_psram=%u sample_rate=%u video=1280x720 "
        "camera_configured=%u microphone_configured=%u\n",
        ESP.getFlashChipSize(), ESP.getPsramSize(), ESP.getFreePsram(), AUDIO_SAMPLE_RATE,
        static_cast<unsigned>(camera_configured()), static_cast<unsigned>(audio_configured()));
    Serial.println(
        "KIDI_READY commands: status, temperature, photo [640x480|2048x1536], "
        "audio <1-10>, video <1-10>, av <1-10>, stream <video|av> <1-300> [1280x720|96x96]");
}

auto print_temperature() -> void {
    float celsius = 0;
    const auto result = read_chip_temperature(celsius);
    if (result != ESP_OK) {
        report_error("chip temperature", result);
        return;
    }
    Serial.printf("KIDI_TEMPERATURE celsius=%.2f cpu_mhz=%u\n", celsius, getCpuFrequencyMhz());
}

} // namespace

auto initialize_usb(const MediaExecutionConfig& media) -> void {
    const auto rx_size = Serial.setRxBufferSize(USB_RX_BUFFER_SIZE);
    const auto tx_size = Serial.setTxBufferSize(USB_TX_BUFFER_SIZE);
    Serial.setTxTimeoutMs(USB_TX_TIMEOUT_MS);
    Serial.begin(115200);
    if (rx_size != USB_RX_BUFFER_SIZE) report_error("USB receive buffer allocation", ESP_ERR_NO_MEM);
    if (tx_size != USB_TX_BUFFER_SIZE) report_error("USB transmit buffer allocation", ESP_ERR_NO_MEM);
    Serial.setTimeout(1000);
    const auto start = millis();
    while (!Serial && millis() - start < 3000) delay(10);
    const auto result = initialize_media(media);
    if (result != ESP_OK) report_error("capture mutex initialization", result);
    Serial.println("KIDI_ESP32_MEDIA protocol=1");
    print_status();
    initialize_wireless();
}

auto poll_usb() -> void {
    poll_wireless();
    if (!Serial.available()) {
        delay(10);
        return;
    }
    auto command = Serial.readStringUntil('\n');
    command.trim();
    if (handle_wireless_command(command)) return;
    if (command == "status") {
        print_status();
    } else if (command == "temperature") {
        print_temperature();
    } else if (command == "stream-stop") {
        Serial.println("KIDI_STREAM_STOPPED");
    } else if (command == "photo") {
        capture_photo();
    } else if (command == "photo 640x480") {
        capture_photo(CameraProfile::PHOTO_VGA);
    } else if (command == "photo 2048x1536") {
        capture_photo(CameraProfile::PHOTO_FULL);
    } else if (command.startsWith("stream ")) {
        const auto separator = command.indexOf(' ', 7);
        const auto kind = command.substring(7, separator);
        const auto resolution_separator = command.indexOf(' ', separator + 1);
        const auto duration = resolution_separator < 0 ? command.substring(separator + 1)
                                                       : command.substring(separator + 1, resolution_separator);
        const auto stream_resolution =
            resolution_separator < 0 ? String("1280x720") : command.substring(resolution_separator + 1);
        auto valid = separator > 7 && duration.length() > 0 && duration.length() <= 3;
        for (unsigned index = 0; index < duration.length(); ++index)
            valid = valid && duration[index] >= '0' && duration[index] <= '9';
        if (!valid || (kind != "video" && kind != "av") ||
            (stream_resolution != "1280x720" && stream_resolution != "96x96")) {
            Serial.println("KIDI_ERROR invalid stream request");
        } else {
            const auto result =
                stream_media(kind == "av", static_cast<unsigned>(duration.toInt()), write_usb_clip, nullptr,
                             stream_resolution == "96x96" ? CameraProfile::VIDEO_LOW : CameraProfile::VIDEO_HD);
            if (result != ESP_OK) report_error("USB stream", result);
            Serial.println("KIDI_STREAM_STOPPED");
        }
    } else if (command.startsWith("audio ") || command.startsWith("video ") || command.startsWith("av ")) {
        const auto duration = command.substring(command.startsWith("av ") ? 3 : 6);
        auto valid = duration.length() > 0;
        for (unsigned index = 0; index < duration.length(); ++index)
            valid = valid && duration[index] >= '0' && duration[index] <= '9';
        if (!valid || duration.length() > 2) {
            Serial.println("KIDI_ERROR invalid capture duration");
        } else {
            const auto kind = command.startsWith("video ") ? ClipKind::VIDEO
                              : command.startsWith("av ")  ? ClipKind::AV
                                                           : ClipKind::AUDIO;
            const auto result = capture_clip(kind, static_cast<unsigned>(duration.toInt()), write_usb_clip, nullptr);
            if (result != ESP_OK) report_error("USB clip", result);
        }
    } else {
        Serial.println("KIDI_ERROR unknown command");
    }
}

} // namespace kidi::esp32
