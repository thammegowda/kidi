#include "esp32.h"
#include "camera.h"
#include "clip.h"
#include "audio.h"
#include "stream.h"
#include "sensors.h"
#include "w11.h"
#include "wireless.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>

#include <Arduino.h>
#include <esp_heap_caps.h>

namespace kidi::esp32 {
namespace {

constexpr std::uint32_t SAMPLE_RATE = AUDIO_SAMPLE_RATE;
constexpr I2sPort MICROPHONE_PORT = I2S_NUM_0;
constexpr CameraFrameSize VIDEO_FRAME_SIZE = FRAMESIZE_HD;
constexpr std::size_t USB_RX_BUFFER_SIZE = 8192;
constexpr std::size_t USB_TX_BUFFER_SIZE = 4096;
constexpr std::uint32_t USB_TX_TIMEOUT_MS = 1000;
using BoardPins = W11Pins;

class ClipOutput : public Print {
public:
    ClipWriter writer = nullptr;
    void* context = nullptr;
    bool failed = false;
    Error error = ESP_OK;

    auto write(std::uint8_t byte) -> std::size_t override { return write(&byte, 1); }
    auto write(const std::uint8_t* bytes, std::size_t length) -> std::size_t override {
        if (failed || writer == nullptr) return 0;
        error = writer(bytes, length, context);
        if (error != ESP_OK) {
            failed = true;
            return 0;
        }
        return length;
    }
};

SemaphoreHandle CLIP_MUTEX = nullptr;
ClipOutput clip_output;
bool clip_failed = false;

auto report_clip_error(const char* message) -> void {
    clip_failed = true;
    clip_output.printf("KIDI_ERROR %s\n", message);
}

auto report_clip_error(const char* operation, Error error) -> void {
    clip_failed = true;
    clip_output.printf("KIDI_ERROR %s: %s (0x%x)\n", operation, esp_err_to_name(error), static_cast<unsigned>(error));
}

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

auto capture_photo(CameraFrameSize frame_size = FRAMESIZE_QXGA) -> void {
    const auto result = capture_photo_to(frame_size, write_usb_photo, nullptr);
    if (result != ESP_OK) report_error("USB photo", result);
}

auto stream_video_frames(std::int64_t start, std::int64_t duration, unsigned& frame_count) -> bool {
    constexpr std::int64_t FRAME_INTERVAL_US = 100000;
    auto next_frame = start;
    std::int64_t acquisition_us = 0;
    std::int64_t transfer_us = 0;
    while (esp_timer_get_time() - start < duration) {
        if (esp_timer_get_time() < next_frame) {
            delay(1);
            continue;
        }
        const auto acquisition_start = esp_timer_get_time();
        CameraFrame* frame = esp_camera_fb_get();
        acquisition_us += esp_timer_get_time() - acquisition_start;
        if (frame == nullptr) {
            report_clip_error("video frame capture failed");
            return false;
        }
        // The camera timestamp marks acquisition, not completion of USB transfer.
        const auto timestamp = frame->timestamp.tv_sec * 1000000LL + frame->timestamp.tv_usec - start;
        if (timestamp < 0 || timestamp >= duration) {
            esp_camera_fb_return(frame);
            continue;
        }
        const auto length = frame->len;
        const auto transfer_start = esp_timer_get_time();
        clip_output.printf("KIDI_FRAME %llu %u\n", static_cast<unsigned long long>(timestamp),
                           static_cast<unsigned>(length));
        const auto written = clip_output.write(frame->buf, length);
        clip_output.println();
        clip_output.println("KIDI_END");
        transfer_us += esp_timer_get_time() - transfer_start;
        esp_camera_fb_return(frame);
        if (written != length) {
            report_clip_error("incomplete video frame transfer");
            return false;
        }
        ++frame_count;
        next_frame += FRAME_INTERVAL_US;
        if (next_frame < esp_timer_get_time()) next_frame = esp_timer_get_time();
    }
    clip_output.printf("KIDI_VIDEO_METRICS acquisition_us=%llu transfer_us=%llu frames=%u\n",
                       static_cast<unsigned long long>(acquisition_us), static_cast<unsigned long long>(transfer_us),
                       frame_count);
    return true;
}

auto capture_video(unsigned seconds) -> void {
    if (seconds < 1 || seconds > 10) {
        report_clip_error("video duration must be 1 to 10 seconds");
        return;
    }
    const auto prepared = prepare_camera(VIDEO_FRAME_SIZE);
    if (prepared != ESP_OK) {
        report_clip_error("video camera preparation", prepared);
        return;
    }
    const auto start = esp_timer_get_time();
    unsigned frame_count = 0;
    clip_output.printf("KIDI_VIDEO_START seconds=%u target_fps=10\n", seconds);
    const auto success = stream_video_frames(start, seconds * 1000000LL, frame_count);
    const auto elapsed = esp_timer_get_time() - start;
    if (success) {
        clip_output.printf("KIDI_VIDEO_END duration_us=%llu frames=%u\n", static_cast<unsigned long long>(elapsed),
                           frame_count);
    }
    const auto result = release_camera();
    if (result != ESP_OK) report_clip_error("camera shutdown", result);
}

auto read_microphone_bytes(std::uint8_t* destination, std::size_t length) -> Error {
    return read_audio(destination, length);
}

auto read_microphone(std::uint8_t* destination, std::size_t length) -> bool {
    const auto result = read_microphone_bytes(destination, length);
    if (result != ESP_OK) report_clip_error("microphone read", result);
    return result == ESP_OK;
}

auto start_microphone(QueueHandle* events = nullptr) -> bool {
    const auto result = start_audio(events);
    if (result != ESP_OK) {
        report_clip_error("microphone initialization", result);
        return false;
    }
    return true;
}

auto capture_audio(unsigned seconds) -> void {
    if (seconds < 1 || seconds > 10) {
        report_clip_error("audio duration must be 1 to 10 seconds");
        return;
    }
    const auto length = SAMPLE_RATE * seconds * sizeof(std::int16_t);
    auto* samples = static_cast<std::uint8_t*>(heap_caps_malloc(length, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
    if (samples == nullptr) {
        report_clip_error("audio buffer allocation failed");
        return;
    }
    if (!start_microphone()) {
        std::free(samples);
        return;
    }
    std::uint8_t warmup[4096];
    if (!read_microphone(warmup, sizeof(warmup))) {
        const auto cleanup = i2s_driver_uninstall(MICROPHONE_PORT);
        if (cleanup != ESP_OK) report_clip_error("microphone shutdown", cleanup);
        std::free(samples);
        return;
    }
    clip_output.printf("KIDI_RECORDING %u seconds\n", seconds);
    auto success = read_microphone(samples, length);
    const auto result = i2s_driver_uninstall(MICROPHONE_PORT);
    if (result != ESP_OK) {
        report_clip_error("microphone shutdown", result);
        success = false;
    }
    if (success) {
        clip_output.printf("KIDI_AUDIO_FORMAT sample_rate=%u channels=1 bits=16\n", SAMPLE_RATE);
        clip_output.printf("KIDI_BEGIN pcm16 %u\n", static_cast<unsigned>(length));
        const auto written = clip_output.write(samples, length);
        clip_output.println();
        clip_output.println("KIDI_END");
        if (written != length) report_clip_error("incomplete PCM transfer");
    }
    std::free(samples);
}

struct ConcurrentAudio {
    std::uint8_t* samples = nullptr;
    std::size_t length = 0;
    std::size_t received = 0;
    QueueHandle events = nullptr;
    SemaphoreHandle ready = nullptr;
    SemaphoreHandle done = nullptr;
    std::atomic<bool> cancel{false};
    std::atomic<Error> result{ESP_FAIL};
    std::int64_t started = 0;
    std::int64_t finished = 0;
    std::int64_t max_read_time = 0;
    unsigned overruns = 0;
};

auto drain_microphone_events(ConcurrentAudio& audio) -> void {
    I2sEvent event;
    while (xQueueReceive(audio.events, &event, 0) == pdTRUE) {
        if (event.type == I2S_EVENT_RX_Q_OVF || event.type == I2S_EVENT_DMA_ERROR) ++audio.overruns;
    }
}

auto record_concurrent_audio(void* argument) -> void {
    auto& audio = *static_cast<ConcurrentAudio*>(argument);
    std::uint8_t warmup[4096];
    audio.result = read_microphone_bytes(warmup, sizeof(warmup));
    drain_microphone_events(audio);
    audio.overruns = 0;
    audio.started = esp_timer_get_time();
    xSemaphoreGive(audio.ready);
    while (audio.result.load() == ESP_OK && audio.received < audio.length && !audio.cancel.load()) {
        const auto remaining = audio.length - audio.received;
        const auto block = remaining < 2048 ? remaining : 2048;
        const auto before = esp_timer_get_time();
        audio.result = read_microphone_bytes(audio.samples + audio.received, block);
        const auto elapsed = esp_timer_get_time() - before;
        if (elapsed > audio.max_read_time) audio.max_read_time = elapsed;
        drain_microphone_events(audio);
        if (audio.result.load() == ESP_OK) audio.received += block;
    }
    audio.finished = esp_timer_get_time();
    xSemaphoreGive(audio.done);
    vTaskDelete(nullptr);
}

auto capture_av(unsigned seconds) -> void {
    if (seconds < 1 || seconds > 10) {
        report_clip_error("AV duration must be 1 to 10 seconds");
        return;
    }
    ConcurrentAudio audio;
    audio.length = SAMPLE_RATE * seconds * sizeof(std::int16_t);
    audio.samples = static_cast<std::uint8_t*>(heap_caps_malloc(audio.length, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
    audio.ready = xSemaphoreCreateBinary();
    audio.done = xSemaphoreCreateBinary();
    if (audio.samples == nullptr || audio.ready == nullptr || audio.done == nullptr) {
        report_clip_error("concurrent audio resource allocation failed");
        if (audio.ready != nullptr) vSemaphoreDelete(audio.ready);
        if (audio.done != nullptr) vSemaphoreDelete(audio.done);
        std::free(audio.samples);
        return;
    }
    const auto camera_result = prepare_camera(VIDEO_FRAME_SIZE);
    const auto camera_ready = camera_result == ESP_OK;
    if (!camera_ready) report_clip_error("AV camera preparation", camera_result);
    const auto microphone_ready = camera_ready && start_microphone(&audio.events);
    auto video_success = false;
    unsigned frame_count = 0;
    if (microphone_ready) {
        const auto task_created =
            xTaskCreatePinnedToCore(record_concurrent_audio, "kidi-mic", 8192, &audio, 2, nullptr, 0) == pdPASS;
        if (!task_created) {
            report_clip_error("concurrent microphone task creation failed");
        } else {
            xSemaphoreTake(audio.ready, portMAX_DELAY);
            if (audio.result.load() == ESP_OK) {
                clip_output.printf("KIDI_AV_START seconds=%u target_fps=10 sample_rate=%u\n", seconds, SAMPLE_RATE);
                video_success = stream_video_frames(audio.started, seconds * 1000000LL, frame_count);
            } else {
                report_clip_error("concurrent microphone warmup", audio.result.load());
            }
            if (!video_success) audio.cancel.store(true);
            const auto video_elapsed = esp_timer_get_time() - audio.started;
            xSemaphoreTake(audio.done, portMAX_DELAY);
            if (video_success && audio.result.load() == ESP_OK && audio.received == audio.length &&
                audio.overruns == 0) {
                clip_output.printf("KIDI_VIDEO_END duration_us=%llu frames=%u\n",
                                   static_cast<unsigned long long>(video_elapsed), frame_count);
                clip_output.printf(
                    "KIDI_AUDIO_META start_us=0 duration_us=%llu samples=%u sample_rate=%u overruns=%u "
                    "max_read_us=%llu\n",
                    static_cast<unsigned long long>(audio.finished - audio.started),
                    static_cast<unsigned>(audio.received / sizeof(std::int16_t)), SAMPLE_RATE, audio.overruns,
                    static_cast<unsigned long long>(audio.max_read_time));
                clip_output.printf("KIDI_BEGIN pcm16 %u\n", static_cast<unsigned>(audio.received));
                const auto written = clip_output.write(audio.samples, audio.received);
                clip_output.println();
                clip_output.println("KIDI_END");
                if (written == audio.received) {
                    clip_output.println("KIDI_AV_END");
                } else {
                    report_clip_error("incomplete concurrent audio transfer");
                }
            } else if (video_success) {
                clip_failed = true;
                clip_output.printf("KIDI_ERROR concurrent audio result=0x%x bytes=%u overruns=%u\n",
                                   static_cast<unsigned>(audio.result.load()), static_cast<unsigned>(audio.received),
                                   audio.overruns);
            }
        }
    }
    if (microphone_ready) {
        const auto result = i2s_driver_uninstall(MICROPHONE_PORT);
        if (result != ESP_OK) report_clip_error("microphone shutdown", result);
    }
    if (camera_ready) {
        const auto result = release_camera();
        if (result != ESP_OK) report_clip_error("camera shutdown", result);
    }
    vSemaphoreDelete(audio.ready);
    vSemaphoreDelete(audio.done);
    std::free(audio.samples);
}

auto print_status() -> void {
    Serial.printf("KIDI_STATUS flash=%u psram=%u free_psram=%u sample_rate=%u video=1280x720\n", ESP.getFlashChipSize(),
                  ESP.getPsramSize(), ESP.getFreePsram(), SAMPLE_RATE);
    Serial.println(
        "KIDI_READY commands: status, temperature, sensors, photo [640x480|2048x1536], audio <1-10>, video <1-10>, av "
        "<1-10>");
}

} // namespace

auto clip_busy() -> bool { return CLIP_MUTEX == nullptr || uxSemaphoreGetCount(CLIP_MUTEX) == 0; }

auto acquire_capture() -> Error {
    if (CLIP_MUTEX == nullptr) return ESP_ERR_INVALID_STATE;
    return xSemaphoreTake(CLIP_MUTEX, 0) == pdTRUE ? ESP_OK : ESP_ERR_TIMEOUT;
}

auto release_capture() -> void { xSemaphoreGive(CLIP_MUTEX); }

auto capture_clip(ClipKind kind, unsigned seconds, ClipWriter writer, void* context) -> Error {
    if (seconds < 1 || seconds > 10 || writer == nullptr) return ESP_ERR_INVALID_ARG;
    const auto acquired = acquire_capture();
    if (acquired != ESP_OK) return acquired;
    clip_output.writer = writer;
    clip_output.context = context;
    clip_output.failed = false;
    clip_output.error = ESP_OK;
    clip_failed = false;
    switch (kind) {
        case ClipKind::AUDIO:
            capture_audio(seconds);
            break;
        case ClipKind::VIDEO:
            capture_video(seconds);
            break;
        case ClipKind::AV:
            capture_av(seconds);
            break;
    }
    if (!clip_failed && !clip_output.failed) clip_output.println("KIDI_CLIP_END");
    const auto result = clip_output.error != ESP_OK ? clip_output.error : (clip_failed ? ESP_FAIL : ESP_OK);
    clip_output.writer = nullptr;
    clip_output.context = nullptr;
    release_capture();
    return result;
}

auto initialize() -> void {
    const auto rx_size = Serial.setRxBufferSize(USB_RX_BUFFER_SIZE);
    const auto tx_size = Serial.setTxBufferSize(USB_TX_BUFFER_SIZE);
    Serial.setTxTimeoutMs(USB_TX_TIMEOUT_MS);
    Serial.begin(115200);
    if (rx_size != USB_RX_BUFFER_SIZE) report_error("USB receive buffer allocation", ESP_ERR_NO_MEM);
    if (tx_size != USB_TX_BUFFER_SIZE) report_error("USB transmit buffer allocation", ESP_ERR_NO_MEM);
    Serial.setTimeout(1000);
    const auto start = millis();
    while (!Serial && millis() - start < 3000) delay(10);
    const auto camera_result = initialize_camera();
    if (camera_result != ESP_OK) report_error("camera mutex initialization", camera_result);
    CLIP_MUTEX = xSemaphoreCreateMutex();
    if (CLIP_MUTEX == nullptr) report_error("capture mutex initialization", ESP_ERR_NO_MEM);
    Serial.println("KIDI_W11_USB_WIFI_DIAGNOSTIC v8");
    print_status();
    initialize_wireless();
}

auto poll() -> void {
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
    } else if (command == "sensors") {
        print_sensors();
    } else if (command == "stream-stop") {
        Serial.println("KIDI_STREAM_STOPPED");
    } else if (command == "photo") {
        capture_photo();
    } else if (command == "photo 640x480") {
        capture_photo(FRAMESIZE_VGA);
    } else if (command == "photo 2048x1536") {
        capture_photo(FRAMESIZE_QXGA);
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
            const auto result = stream_media(kind == "av", static_cast<unsigned>(duration.toInt()), write_usb_clip,
                                             nullptr, stream_resolution == "96x96" ? FRAMESIZE_96X96 : FRAMESIZE_HD);
            if (result != ESP_OK) report_error("USB stream", result);
            Serial.println("KIDI_STREAM_STOPPED");
        }
    } else if (command.startsWith("audio ") || command.startsWith("video ") || command.startsWith("av ")) {
        const auto duration = command.substring(command.startsWith("av ") ? 3 : 6);
        auto valid = duration.length() > 0;
        for (unsigned index = 0; index < duration.length(); ++index) {
            valid = valid && duration[index] >= '0' && duration[index] <= '9';
        }
        if (!valid || duration.length() > 2) {
            Serial.println("KIDI_ERROR invalid capture duration");
        } else if (command.startsWith("video ")) {
            const auto result =
                capture_clip(ClipKind::VIDEO, static_cast<unsigned>(duration.toInt()), write_usb_clip, nullptr);
            if (result != ESP_OK) report_error("USB video", result);
        } else if (command.startsWith("av ")) {
            const auto result =
                capture_clip(ClipKind::AV, static_cast<unsigned>(duration.toInt()), write_usb_clip, nullptr);
            if (result != ESP_OK) report_error("USB AV", result);
        } else {
            const auto result =
                capture_clip(ClipKind::AUDIO, static_cast<unsigned>(duration.toInt()), write_usb_clip, nullptr);
            if (result != ESP_OK) report_error("USB audio", result);
        }
    } else {
        Serial.println("KIDI_ERROR unknown command");
    }
}

} // namespace kidi::esp32

auto setup() -> void { kidi::esp32::initialize(); }
auto loop() -> void { kidi::esp32::poll(); }
