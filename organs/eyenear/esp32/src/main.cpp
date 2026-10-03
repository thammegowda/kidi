#include "esp32.h"
#include "sensors.h"
#include "w11.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>

#include <Arduino.h>
#include <esp_heap_caps.h>

namespace kidi::esp32 {
namespace {

constexpr std::uint32_t SAMPLE_RATE = 48000;
constexpr I2sPort MICROPHONE_PORT = I2S_NUM_0;
constexpr CameraFrameSize VIDEO_FRAME_SIZE = FRAMESIZE_HD;
using BoardPins = W11Pins;

auto start_camera(CameraFrameSize frame_size) -> bool {
    CameraConfig config{};
    config.pin_pwdn = BoardPins::CAMERA_POWER_DOWN;
    config.pin_reset = BoardPins::CAMERA_RESET;
    config.pin_xclk = BoardPins::CAMERA_CLOCK;
    config.pin_sccb_sda = BoardPins::CAMERA_SDA;
    config.pin_sccb_scl = BoardPins::CAMERA_SCL;
    config.pin_d0 = BoardPins::CAMERA_DATA[0];
    config.pin_d1 = BoardPins::CAMERA_DATA[1];
    config.pin_d2 = BoardPins::CAMERA_DATA[2];
    config.pin_d3 = BoardPins::CAMERA_DATA[3];
    config.pin_d4 = BoardPins::CAMERA_DATA[4];
    config.pin_d5 = BoardPins::CAMERA_DATA[5];
    config.pin_d6 = BoardPins::CAMERA_DATA[6];
    config.pin_d7 = BoardPins::CAMERA_DATA[7];
    config.pin_vsync = BoardPins::CAMERA_VSYNC;
    config.pin_href = BoardPins::CAMERA_HREF;
    config.pin_pclk = BoardPins::CAMERA_PIXEL_CLOCK;
    config.xclk_freq_hz = 20000000;
    config.ledc_timer = LEDC_TIMER_0;
    config.ledc_channel = LEDC_CHANNEL_0;
    config.pixel_format = PIXFORMAT_JPEG;
    config.frame_size = frame_size;
    config.jpeg_quality = 12;
    config.fb_count = 1;
    config.fb_location = CAMERA_FB_IN_PSRAM;
    config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
    const auto result = esp_camera_init(&config);
    if (result != ESP_OK) {
        report_error("camera initialization", result);
        return false;
    }
    const auto* sensor = esp_camera_sensor_get();
    if (sensor == nullptr) {
        Serial.println("KIDI_ERROR camera sensor unavailable");
        esp_camera_deinit();
        return false;
    }
    Serial.printf("KIDI_SENSOR pid=0x%04x\n", sensor->id.PID);
    return true;
}

auto prepare_camera(CameraFrameSize frame_size = VIDEO_FRAME_SIZE) -> bool {
    if (!start_camera(frame_size)) return false;
    delay(500);
    for (int index = 0; index < 3; ++index) {
        CameraFrame* warmup = esp_camera_fb_get();
        if (warmup == nullptr) {
            Serial.println("KIDI_ERROR camera warmup capture failed");
            esp_camera_deinit();
            return false;
        }
        esp_camera_fb_return(warmup);
        delay(100);
    }
    return true;
}

auto capture_photo(CameraFrameSize frame_size = FRAMESIZE_QXGA) -> void {
    if (!prepare_camera(frame_size)) return;
    CameraFrame* frame = esp_camera_fb_get();
    if (frame == nullptr) {
        Serial.println("KIDI_ERROR camera capture failed");
        esp_camera_deinit();
        return;
    }
    Serial.printf("KIDI_PHOTO width=%u height=%u\n", static_cast<unsigned>(frame->width),
                  static_cast<unsigned>(frame->height));
    Serial.printf("KIDI_BEGIN jpeg %u\n", static_cast<unsigned>(frame->len));
    const auto written = Serial.write(frame->buf, frame->len);
    Serial.println();
    Serial.println("KIDI_END");
    if (written != frame->len) Serial.println("KIDI_ERROR incomplete JPEG USB transfer");
    esp_camera_fb_return(frame);
    const auto result = esp_camera_deinit();
    if (result != ESP_OK) report_error("camera shutdown", result);
}

auto stream_video_frames(std::int64_t start, std::int64_t duration, unsigned& frame_count) -> bool {
    constexpr std::int64_t FRAME_INTERVAL_US = 100000;
    auto next_frame = start;
    while (esp_timer_get_time() - start < duration) {
        if (esp_timer_get_time() < next_frame) {
            delay(1);
            continue;
        }
        CameraFrame* frame = esp_camera_fb_get();
        if (frame == nullptr) {
            Serial.println("KIDI_ERROR video frame capture failed");
            return false;
        }
        // The camera timestamp marks acquisition, not completion of USB transfer.
        const auto timestamp = frame->timestamp.tv_sec * 1000000LL + frame->timestamp.tv_usec - start;
        if (timestamp < 0 || timestamp >= duration) {
            esp_camera_fb_return(frame);
            continue;
        }
        const auto length = frame->len;
        Serial.printf("KIDI_FRAME %llu %u\n", static_cast<unsigned long long>(timestamp),
                      static_cast<unsigned>(length));
        const auto written = Serial.write(frame->buf, length);
        Serial.println();
        Serial.println("KIDI_END");
        esp_camera_fb_return(frame);
        if (written != length) {
            Serial.println("KIDI_ERROR incomplete video frame USB transfer");
            return false;
        }
        ++frame_count;
        next_frame += FRAME_INTERVAL_US;
        if (next_frame < esp_timer_get_time()) next_frame = esp_timer_get_time();
    }
    return true;
}

auto capture_video(unsigned seconds) -> void {
    if (seconds < 1 || seconds > 10) {
        Serial.println("KIDI_ERROR video duration must be 1 to 10 seconds");
        return;
    }
    if (!prepare_camera()) return;
    const auto start = esp_timer_get_time();
    unsigned frame_count = 0;
    Serial.printf("KIDI_VIDEO_START seconds=%u target_fps=10\n", seconds);
    const auto success = stream_video_frames(start, seconds * 1000000LL, frame_count);
    const auto elapsed = esp_timer_get_time() - start;
    if (success) {
        Serial.printf("KIDI_VIDEO_END duration_us=%llu frames=%u\n", static_cast<unsigned long long>(elapsed),
                      frame_count);
    }
    const auto result = esp_camera_deinit();
    if (result != ESP_OK) report_error("camera shutdown", result);
}

auto read_microphone_bytes(std::uint8_t* destination, std::size_t length) -> Error {
    std::size_t offset = 0;
    while (offset < length) {
        std::size_t received = 0;
        const auto result =
            i2s_read(MICROPHONE_PORT, destination + offset, length - offset, &received, pdMS_TO_TICKS(2000));
        if (result != ESP_OK) return result;
        if (received == 0 || received % sizeof(std::int16_t) != 0) return ESP_ERR_TIMEOUT;
        offset += received;
    }
    return ESP_OK;
}

auto read_microphone(std::uint8_t* destination, std::size_t length) -> bool {
    const auto result = read_microphone_bytes(destination, length);
    if (result != ESP_OK) report_error("microphone read", result);
    return result == ESP_OK;
}

auto start_microphone(QueueHandle* events = nullptr) -> bool {
    I2sConfig config{};
    config.mode = static_cast<I2sMode>(I2S_MODE_MASTER | I2S_MODE_RX | I2S_MODE_PDM);
    config.sample_rate = SAMPLE_RATE;
    config.bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT;
    config.channel_format = I2S_CHANNEL_FMT_ONLY_LEFT;
    config.communication_format = I2S_COMM_FORMAT_STAND_I2S;
    config.intr_alloc_flags = ESP_INTR_FLAG_LEVEL1;
    config.dma_buf_count = 8;
    config.dma_buf_len = 256;
    auto result = i2s_driver_install(MICROPHONE_PORT, &config, events == nullptr ? 0 : 32, events);
    if (result != ESP_OK) {
        report_error("microphone initialization", result);
        return false;
    }
    I2sPins pins{};
    pins.mck_io_num = I2S_PIN_NO_CHANGE;
    pins.bck_io_num = I2S_PIN_NO_CHANGE;
    pins.ws_io_num = BoardPins::MICROPHONE_CLOCK;
    pins.data_out_num = I2S_PIN_NO_CHANGE;
    pins.data_in_num = BoardPins::MICROPHONE_DATA;
    result = i2s_set_pin(MICROPHONE_PORT, &pins);
    if (result != ESP_OK) {
        report_error("microphone pins", result);
        i2s_driver_uninstall(MICROPHONE_PORT);
        return false;
    }
    return true;
}

auto capture_audio(unsigned seconds) -> void {
    if (seconds < 1 || seconds > 10) {
        Serial.println("KIDI_ERROR audio duration must be 1 to 10 seconds");
        return;
    }
    const auto length = SAMPLE_RATE * seconds * sizeof(std::int16_t);
    auto* samples = static_cast<std::uint8_t*>(heap_caps_malloc(length, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
    if (samples == nullptr) {
        Serial.println("KIDI_ERROR audio buffer allocation failed");
        return;
    }
    if (!start_microphone()) {
        std::free(samples);
        return;
    }
    std::uint8_t warmup[4096];
    if (!read_microphone(warmup, sizeof(warmup))) {
        i2s_driver_uninstall(MICROPHONE_PORT);
        std::free(samples);
        return;
    }
    Serial.printf("KIDI_RECORDING %u seconds\n", seconds);
    auto success = read_microphone(samples, length);
    const auto result = i2s_driver_uninstall(MICROPHONE_PORT);
    if (result != ESP_OK) {
        report_error("microphone shutdown", result);
        success = false;
    }
    if (success) {
        Serial.printf("KIDI_AUDIO_FORMAT sample_rate=%u channels=1 bits=16\n", SAMPLE_RATE);
        Serial.printf("KIDI_BEGIN pcm16 %u\n", static_cast<unsigned>(length));
        const auto written = Serial.write(samples, length);
        Serial.println();
        Serial.println("KIDI_END");
        if (written != length) Serial.println("KIDI_ERROR incomplete PCM USB transfer");
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
        Serial.println("KIDI_ERROR AV duration must be 1 to 10 seconds");
        return;
    }
    ConcurrentAudio audio;
    audio.length = SAMPLE_RATE * seconds * sizeof(std::int16_t);
    audio.samples = static_cast<std::uint8_t*>(heap_caps_malloc(audio.length, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
    audio.ready = xSemaphoreCreateBinary();
    audio.done = xSemaphoreCreateBinary();
    if (audio.samples == nullptr || audio.ready == nullptr || audio.done == nullptr) {
        Serial.println("KIDI_ERROR concurrent audio resource allocation failed");
        if (audio.ready != nullptr) vSemaphoreDelete(audio.ready);
        if (audio.done != nullptr) vSemaphoreDelete(audio.done);
        std::free(audio.samples);
        return;
    }
    const auto camera_ready = prepare_camera();
    const auto microphone_ready = camera_ready && start_microphone(&audio.events);
    auto video_success = false;
    unsigned frame_count = 0;
    if (microphone_ready) {
        const auto task_created =
            xTaskCreatePinnedToCore(record_concurrent_audio, "kidi-mic", 8192, &audio, 2, nullptr, 0) == pdPASS;
        if (!task_created) {
            Serial.println("KIDI_ERROR concurrent microphone task creation failed");
        } else {
            xSemaphoreTake(audio.ready, portMAX_DELAY);
            if (audio.result.load() == ESP_OK) {
                Serial.printf("KIDI_AV_START seconds=%u target_fps=10 sample_rate=%u\n", seconds, SAMPLE_RATE);
                video_success = stream_video_frames(audio.started, seconds * 1000000LL, frame_count);
            } else {
                report_error("concurrent microphone warmup", audio.result.load());
            }
            if (!video_success) audio.cancel.store(true);
            const auto video_elapsed = esp_timer_get_time() - audio.started;
            xSemaphoreTake(audio.done, portMAX_DELAY);
            if (video_success && audio.result.load() == ESP_OK && audio.received == audio.length &&
                audio.overruns == 0) {
                Serial.printf("KIDI_VIDEO_END duration_us=%llu frames=%u\n",
                              static_cast<unsigned long long>(video_elapsed), frame_count);
                Serial.printf(
                    "KIDI_AUDIO_META start_us=0 duration_us=%llu samples=%u sample_rate=%u overruns=%u "
                    "max_read_us=%llu\n",
                    static_cast<unsigned long long>(audio.finished - audio.started),
                    static_cast<unsigned>(audio.received / sizeof(std::int16_t)), SAMPLE_RATE, audio.overruns,
                    static_cast<unsigned long long>(audio.max_read_time));
                Serial.printf("KIDI_BEGIN pcm16 %u\n", static_cast<unsigned>(audio.received));
                const auto written = Serial.write(audio.samples, audio.received);
                Serial.println();
                Serial.println("KIDI_END");
                if (written == audio.received) {
                    Serial.println("KIDI_AV_END");
                } else {
                    Serial.println("KIDI_ERROR incomplete concurrent audio USB transfer");
                }
            } else if (video_success) {
                Serial.printf("KIDI_ERROR concurrent audio result=0x%x bytes=%u overruns=%u\n",
                              static_cast<unsigned>(audio.result.load()), static_cast<unsigned>(audio.received),
                              audio.overruns);
            }
        }
    }
    if (microphone_ready) {
        const auto result = i2s_driver_uninstall(MICROPHONE_PORT);
        if (result != ESP_OK) report_error("microphone shutdown", result);
    }
    if (camera_ready) {
        const auto result = esp_camera_deinit();
        if (result != ESP_OK) report_error("camera shutdown", result);
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

auto initialize() -> void {
    Serial.begin(115200);
    Serial.setTimeout(1000);
    const auto start = millis();
    while (!Serial && millis() - start < 3000) delay(10);
    Serial.println("KIDI_W11_USB_DIAGNOSTIC v6");
    print_status();
}

auto poll() -> void {
    if (!Serial.available()) {
        delay(10);
        return;
    }
    auto command = Serial.readStringUntil('\n');
    command.trim();
    if (command == "status") {
        print_status();
    } else if (command == "temperature") {
        print_temperature();
    } else if (command == "sensors") {
        print_sensors();
    } else if (command == "photo") {
        capture_photo();
    } else if (command == "photo 640x480") {
        capture_photo(FRAMESIZE_VGA);
    } else if (command == "photo 2048x1536") {
        capture_photo(FRAMESIZE_QXGA);
    } else if (command.startsWith("audio ") || command.startsWith("video ") || command.startsWith("av ")) {
        const auto duration = command.substring(command.startsWith("av ") ? 3 : 6);
        auto valid = duration.length() > 0;
        for (unsigned index = 0; index < duration.length(); ++index) {
            valid = valid && duration[index] >= '0' && duration[index] <= '9';
        }
        if (!valid || duration.length() > 2) {
            Serial.println("KIDI_ERROR invalid capture duration");
        } else if (command.startsWith("video ")) {
            capture_video(static_cast<unsigned>(duration.toInt()));
        } else if (command.startsWith("av ")) {
            capture_av(static_cast<unsigned>(duration.toInt()));
        } else {
            capture_audio(static_cast<unsigned>(duration.toInt()));
        }
    } else {
        Serial.println("KIDI_ERROR unknown command");
    }
}

} // namespace kidi::esp32

auto setup() -> void { kidi::esp32::initialize(); }
auto loop() -> void { kidi::esp32::poll(); }
