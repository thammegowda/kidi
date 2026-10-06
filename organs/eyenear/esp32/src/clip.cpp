#include "esp32.h"
#include "camera.h"
#include "clip.h"
#include "audio.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>

#include <Arduino.h>
#include <esp_heap_caps.h>

namespace kidi::esp32 {
namespace {

constexpr std::uint32_t SAMPLE_RATE = AUDIO_SAMPLE_RATE;
constexpr CameraProfile VIDEO_PROFILE = CameraProfile::VIDEO_HD;

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
MediaExecutionConfig execution;
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
        CameraFrame frame;
        const auto frame_result = acquire_camera_frame(frame);
        acquisition_us += esp_timer_get_time() - acquisition_start;
        if (frame_result != ESP_OK) {
            report_clip_error("video frame capture failed");
            return false;
        }
        // The camera timestamp marks acquisition, not completion of USB transfer.
        const auto timestamp = frame.timestamp_us - start;
        if (timestamp < 0 || timestamp >= duration) {
            return_camera_frame(frame);
            continue;
        }
        const auto length = frame.size;
        const auto transfer_start = esp_timer_get_time();
        clip_output.printf("KIDI_FRAME %llu %u\n", static_cast<unsigned long long>(timestamp),
                           static_cast<unsigned>(length));
        const auto written = clip_output.write(frame.data, length);
        clip_output.println();
        clip_output.println("KIDI_END");
        transfer_us += esp_timer_get_time() - transfer_start;
        return_camera_frame(frame);
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
    const auto prepared = prepare_camera({VIDEO_PROFILE, 1, 12});
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

auto start_microphone() -> bool {
    const auto result = start_audio();
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
        const auto cleanup = stop_audio();
        if (cleanup != ESP_OK) report_clip_error("microphone shutdown", cleanup);
        std::free(samples);
        return;
    }
    clip_output.printf("KIDI_RECORDING %u seconds\n", seconds);
    auto success = read_microphone(samples, length);
    const auto result = stop_audio();
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
    SemaphoreHandle ready = nullptr;
    SemaphoreHandle done = nullptr;
    std::atomic<bool> cancel{false};
    std::atomic<Error> result{ESP_FAIL};
    std::int64_t started = 0;
    std::int64_t finished = 0;
    std::int64_t max_read_time = 0;
    unsigned baseline_overruns = 0;
    unsigned overruns = 0;
};

auto update_microphone_stats(ConcurrentAudio& audio) -> void {
    AudioStats stats;
    const auto result = read_audio_stats(stats);
    if (result != ESP_OK && audio.result.load() == ESP_OK) audio.result.store(result);
    const auto total = stats.overruns + stats.dma_errors;
    audio.overruns = total >= audio.baseline_overruns ? total - audio.baseline_overruns : total;
}

auto record_concurrent_audio(void* argument) -> void {
    auto& audio = *static_cast<ConcurrentAudio*>(argument);
    std::uint8_t warmup[4096];
    audio.result = read_microphone_bytes(warmup, sizeof(warmup));
    update_microphone_stats(audio);
    audio.baseline_overruns = audio.overruns;
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
        update_microphone_stats(audio);
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
    const auto camera_result = prepare_camera({VIDEO_PROFILE, 1, 12});
    const auto camera_ready = camera_result == ESP_OK;
    if (!camera_ready) report_clip_error("AV camera preparation", camera_result);
    const auto microphone_ready = camera_ready && start_microphone();
    auto video_success = false;
    unsigned frame_count = 0;
    if (microphone_ready) {
        const auto task_created =
            xTaskCreatePinnedToCore(record_concurrent_audio, "kidi-mic", execution.audio_stack, &audio,
                                    execution.audio_priority, nullptr, execution.audio_core) == pdPASS;
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
        const auto result = stop_audio();
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

} // namespace

auto initialize_media(const MediaExecutionConfig& configuration) -> Error {
    if (CLIP_MUTEX != nullptr) return ESP_ERR_INVALID_STATE;
    const auto valid_core = [](BaseType_t core) {
        return core == tskNO_AFFINITY || (core >= 0 && core < portNUM_PROCESSORS);
    };
    if (!valid_core(configuration.camera_core) || !valid_core(configuration.audio_core) ||
        configuration.camera_priority >= configMAX_PRIORITIES || configuration.audio_priority >= configMAX_PRIORITIES ||
        configuration.camera_stack == 0 || configuration.audio_stack == 0)
        return ESP_ERR_INVALID_ARG;
    CLIP_MUTEX = xSemaphoreCreateMutex();
    if (CLIP_MUTEX == nullptr) return ESP_ERR_NO_MEM;
    execution = configuration;
    return ESP_OK;
}

auto media_execution_config() -> const MediaExecutionConfig& { return execution; }

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

} // namespace kidi::esp32
