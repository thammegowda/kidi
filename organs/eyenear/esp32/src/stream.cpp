#include "stream.h"
#include "audio.h"
#include "camera.h"
#include "wireless.h"

#include <array>
#include <algorithm>
#include <atomic>
#include <cstring>
#include <cstdlib>
#include <cstdarg>
#include <cstdio>
#include <memory>

#include <Arduino.h>
#include <cJSON.h>
#include <esp_heap_caps.h>

namespace kidi::esp32 {
namespace {

constexpr unsigned MAX_STREAM_SECONDS = 300;
constexpr std::size_t AUDIO_BYTES = AUDIO_SAMPLE_RATE / 50 * sizeof(std::int16_t);
constexpr std::size_t MAX_PACKET_BYTES = 1024 * 1024;
constexpr std::size_t HEADER_BYTES = 20;
constexpr unsigned AUDIO_QUEUE_PACKETS = 32;
constexpr unsigned AUDIO_BATCH_PACKETS = 8;
constexpr std::size_t CAMERA_LOG_BYTES = 1024;

std::array<char, CAMERA_LOG_BYTES> camera_log{};
CriticalSection camera_log_mutex = portMUX_INITIALIZER_UNLOCKED;
std::size_t camera_log_size = 0;
bool camera_log_active = false;
bool camera_log_truncated = false;
LogPrinter camera_log_previous = std::vprintf;

auto capture_camera_log(const char* format, std::va_list arguments) -> int {
    std::array<char, 512> message{};
    std::va_list copy;
    va_copy(copy, arguments);
    const auto written = std::vsnprintf(message.data(), message.size(), format, copy);
    va_end(copy);
    const auto length = std::strlen(message.data());
    taskENTER_CRITICAL(&camera_log_mutex);
    const auto active = camera_log_active;
    const auto previous = camera_log_previous;
    if (active) {
        const auto bytes = std::min(length, camera_log.size() - 1 - camera_log_size);
        std::memcpy(camera_log.data() + camera_log_size, message.data(), bytes);
        camera_log_size += bytes;
        camera_log[camera_log_size] = '\0';
        camera_log_truncated = camera_log_truncated || bytes < length || written < 0 ||
                               static_cast<std::size_t>(written) >= message.size();
    }
    taskEXIT_CRITICAL(&camera_log_mutex);
    return active ? written : previous(format, arguments);
}

enum class PacketKind : std::uint8_t { JPEG = 1, AUDIO = 2, STATUS = 3, END = 4 };

struct AudioPacket {
    std::uint32_t sequence = 0;
    std::array<std::uint8_t, AUDIO_BYTES> data{};
};

struct StreamSession {
    QueueHandle video = nullptr;
    QueueHandle audio = nullptr;
    SemaphoreHandle done = nullptr;
    SemaphoreHandle frame_slots = nullptr;
    std::atomic<bool> stop{false};
    std::atomic<Error> error{ESP_OK};
    std::atomic<unsigned> video_drops{0};
    std::atomic<unsigned> audio_drops{0};
    std::atomic<unsigned> audio_overruns{0};
    std::atomic<unsigned> video_acquired{0};
    std::atomic<unsigned> capture_us{0};
    unsigned audio_baseline_overruns = 0;
    unsigned frame_width = 0;
    unsigned frame_height = 0;
    std::int64_t start = 0;
    std::int64_t duration = 0;
};

auto return_frame(StreamSession& session, CameraFrame& frame) -> void {
    return_camera_frame(frame);
    xSemaphoreGive(session.frame_slots);
}

auto camera_worker(void* argument) -> void {
    auto& session = *static_cast<StreamSession*>(argument);
    auto next_frame = esp_timer_get_time();
    while (!session.stop.load() && esp_timer_get_time() - session.start < session.duration) {
        const auto now = esp_timer_get_time();
        if (now < next_frame) {
            vTaskDelay(pdMS_TO_TICKS((next_frame - now + 999) / 1000));
            continue;
        }
        if (xSemaphoreTake(session.frame_slots, pdMS_TO_TICKS(10)) != pdTRUE) {
            CameraFrame old;
            if (xQueueReceive(session.video, &old, 0) == pdTRUE) {
                return_frame(session, old);
                ++session.video_drops;
            }
            continue;
        }
        if (session.stop.load()) {
            xSemaphoreGive(session.frame_slots);
            break;
        }
        next_frame = esp_timer_get_time() + 1000000 / STREAM_TARGET_FPS;
        const auto capture_started = esp_timer_get_time();
        CameraFrame frame;
        const auto result = acquire_camera_frame(frame);
        session.capture_us.fetch_add(static_cast<unsigned>(esp_timer_get_time() - capture_started));
        if (result != ESP_OK) {
            xSemaphoreGive(session.frame_slots);
            session.error.store(ESP_FAIL);
            break;
        }
        if (frame.width != session.frame_width || frame.height != session.frame_height) {
            return_frame(session, frame);
            session.error.store(ESP_ERR_INVALID_RESPONSE);
            break;
        }
        ++session.video_acquired;
        CameraFrame old;
        if (xQueueReceive(session.video, &old, 0) == pdTRUE) {
            return_frame(session, old);
            ++session.video_drops;
        }
        if (xQueueSend(session.video, &frame, 0) != pdTRUE) {
            return_frame(session, frame);
            ++session.video_drops;
        }
    }
    xSemaphoreGive(session.done);
    vTaskDelete(nullptr);
}

auto audio_worker(void* argument) -> void {
    auto& session = *static_cast<StreamSession*>(argument);
    AudioPacket packet;
    while (!session.stop.load() && esp_timer_get_time() - session.start < session.duration) {
        const auto result = read_audio(packet.data.data(), packet.data.size());
        if (result != ESP_OK) {
            session.error.store(result);
            break;
        }
        AudioStats stats;
        const auto stats_result = read_audio_stats(stats);
        if (stats_result != ESP_OK) {
            session.error.store(stats_result);
            break;
        }
        const auto total = stats.overruns + stats.dma_errors;
        session.audio_overruns.store(total >= session.audio_baseline_overruns ? total - session.audio_baseline_overruns
                                                                              : total);
        if (xQueueSend(session.audio, &packet, 0) != pdTRUE) ++session.audio_drops;
        ++packet.sequence;
    }
    xSemaphoreGive(session.done);
    vTaskDelete(nullptr);
}

auto put_u32(std::uint8_t* destination, std::uint32_t value) -> void {
    for (int index = 0; index < 4; ++index) destination[index] = static_cast<std::uint8_t>(value >> (24 - index * 8));
}

auto send_packet(std::uint8_t* buffer, PacketKind kind, std::uint32_t sequence, std::uint64_t timestamp,
                 const std::uint8_t* payload, std::size_t length, ClipWriter writer, void* context,
                 std::size_t write_quantum) -> Error {
    if (length == 0 || length > MAX_PACKET_BYTES) return ESP_ERR_INVALID_SIZE;
    buffer[0] = static_cast<std::uint8_t>(kind);
    buffer[1] = buffer[2] = buffer[3] = 0;
    put_u32(buffer + 4, sequence);
    put_u32(buffer + 8, static_cast<std::uint32_t>(timestamp >> 32));
    put_u32(buffer + 12, static_cast<std::uint32_t>(timestamp));
    put_u32(buffer + 16, static_cast<std::uint32_t>(length));
    auto offset = std::min(length, STREAM_SEND_BYTES - HEADER_BYTES);
    std::memcpy(buffer + HEADER_BYTES, payload, offset);
    auto result = writer(buffer, HEADER_BYTES + offset, context);
    while (result == ESP_OK && offset < length) {
        const auto bytes = std::min(write_quantum, length - offset);
        result = writer(payload + offset, bytes, context);
        offset += bytes;
    }
    return result;
}

auto stop_stream_camera(std::uint8_t* buffer, ClipWriter writer, void* context, std::size_t write_quantum,
                        std::uint64_t timestamp) -> Error {
    taskENTER_CRITICAL(&camera_log_mutex);
    camera_log_size = 0;
    camera_log_truncated = false;
    camera_log_active = true;
    camera_log[0] = '\0';
    taskEXIT_CRITICAL(&camera_log_mutex);
    const auto previous = esp_log_set_vprintf(capture_camera_log);
    taskENTER_CRITICAL(&camera_log_mutex);
    camera_log_previous = previous;
    taskEXIT_CRITICAL(&camera_log_mutex);
    const auto result = release_camera();
    esp_log_set_vprintf(previous);
    std::array<char, CAMERA_LOG_BYTES> message{};
    taskENTER_CRITICAL(&camera_log_mutex);
    camera_log_active = false;
    const auto truncated = camera_log_truncated;
    std::memcpy(message.data(), camera_log.data(), camera_log_size + 1);
    taskEXIT_CRITICAL(&camera_log_mutex);
    if (message[0] == '\0' && !truncated) return result;
    using Json = std::unique_ptr<cJSON, decltype(&cJSON_Delete)>;
    Json status(cJSON_CreateObject(), cJSON_Delete);
    if (!status) return result != ESP_OK ? result : ESP_ERR_NO_MEM;
    cJSON_AddStringToObject(status.get(), "sdk_diagnostics", message.data());
    cJSON_AddBoolToObject(status.get(), "sdk_diagnostics_truncated", truncated);
    std::unique_ptr<char, decltype(&std::free)> text(cJSON_PrintUnformatted(status.get()), std::free);
    if (!text) return result != ESP_OK ? result : ESP_ERR_NO_MEM;
    const auto sent =
        send_packet(buffer, PacketKind::STATUS, 0, timestamp, reinterpret_cast<const std::uint8_t*>(text.get()),
                    std::strlen(text.get()), writer, context, write_quantum);
    return result != ESP_OK ? result : sent;
}

} // namespace

auto stream_media(bool with_audio, unsigned seconds, ClipWriter writer, void* context, CameraProfile profile,
                  std::size_t write_quantum) -> Error {
    if (writer == nullptr || seconds < 1 || seconds > MAX_STREAM_SECONDS ||
        (profile != CameraProfile::VIDEO_HD && profile != CameraProfile::VIDEO_LOW) ||
        write_quantum < STREAM_SEND_BYTES || write_quantum > MAX_PACKET_BYTES)
        return ESP_ERR_INVALID_ARG;
    auto result = acquire_capture();
    if (result != ESP_OK) return result;
    StreamSession session;
    session.frame_width = camera_profile_width(profile);
    session.frame_height = camera_profile_height(profile);
    session.video = xQueueCreate(1, sizeof(CameraFrame));
    session.audio = with_audio ? xQueueCreate(AUDIO_QUEUE_PACKETS, sizeof(AudioPacket)) : nullptr;
    session.done = xSemaphoreCreateCounting(2, 0);
    const auto frame_buffers = with_audio ? 3U : 2U;
    const auto jpeg_quality = with_audio ? 12U : STREAM_JPEG_QUALITY;
    session.frame_slots = xSemaphoreCreateCounting(frame_buffers, frame_buffers);
    std::unique_ptr<std::uint8_t, decltype(&std::free)> buffer(
        static_cast<std::uint8_t*>(heap_caps_malloc(STREAM_SEND_BYTES, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT)),
        std::free);
    std::unique_ptr<std::uint8_t, decltype(&std::free)> audio_buffer(
        with_audio ? static_cast<std::uint8_t*>(
                         heap_caps_malloc(AUDIO_BYTES * AUDIO_BATCH_PACKETS, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT))
                   : nullptr,
        std::free);
    auto camera_started = false;
    auto audio_started = false;
    auto protocol_started = false;
    unsigned workers = 0;
    unsigned frames_sent = 0;
    unsigned samples_sent = 0;
    std::uint64_t video_bytes = 0;
    std::uint64_t send_us = 0;
    unsigned send_max_us = 0;
    unsigned frame_age_max_us = 0;
    if (session.video == nullptr || session.done == nullptr || session.frame_slots == nullptr || !buffer ||
        (with_audio && (session.audio == nullptr || !audio_buffer))) {
        result = ESP_ERR_NO_MEM;
    } else {
        result = prepare_camera({profile, frame_buffers, jpeg_quality});
        camera_started = result == ESP_OK;
        if (camera_started && with_audio) {
            result = start_audio();
            audio_started = result == ESP_OK;
            if (audio_started) {
                result = read_audio(audio_buffer.get(), STREAM_SEND_BYTES);
                AudioStats stats;
                if (result == ESP_OK) result = read_audio_stats(stats);
                session.audio_baseline_overruns = stats.overruns + stats.dma_errors;
            }
        }
        if (result == ESP_OK) {
            poll_wireless();
            session.start = esp_timer_get_time();
            session.duration = seconds * 1000000LL;
            constexpr char BEGIN[] = "KIDI_LIVE_BEGIN\n";
            result = writer(reinterpret_cast<const std::uint8_t*>(BEGIN), sizeof(BEGIN) - 1, context);
            protocol_started = result == ESP_OK;
            char status[512]{};
            snprintf(status, sizeof(status),
                     "{\"protocol\":1,\"width\":%u,\"height\":%u,\"sample_rate\":%u,"
                     "\"audio\":%s,\"max_seconds\":%u,\"internal_free\":%u,"
                     "\"target_fps\":%u,\"frame_buffers\":%u,\"jpeg_quality\":%u}",
                     camera_profile_width(profile), camera_profile_height(profile), AUDIO_SAMPLE_RATE,
                     with_audio ? "true" : "false", seconds,
                     static_cast<unsigned>(heap_caps_get_free_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT)),
                     STREAM_TARGET_FPS, frame_buffers, jpeg_quality);
            if (result == ESP_OK)
                result =
                    send_packet(buffer.get(), PacketKind::STATUS, 0, 0, reinterpret_cast<const std::uint8_t*>(status),
                                std::strlen(status), writer, context, write_quantum);
        }
        if (result == ESP_OK) {
            const auto& execution = media_execution_config();
            if (xTaskCreatePinnedToCore(camera_worker, "kidi-camera", execution.camera_stack, &session,
                                        execution.camera_priority, nullptr, execution.camera_core) == pdPASS)
                ++workers;
            else
                result = ESP_ERR_NO_MEM;
        }
        if (result == ESP_OK && with_audio) {
            const auto& execution = media_execution_config();
            if (xTaskCreatePinnedToCore(audio_worker, "kidi-audio", execution.audio_stack, &session,
                                        execution.audio_priority, nullptr, execution.audio_core) == pdPASS)
                ++workers;
            else
                result = ESP_ERR_NO_MEM;
        }
        auto next_status = session.start + 1000000;
        while (result == ESP_OK && session.error.load() == ESP_OK &&
               esp_timer_get_time() - session.start < session.duration) {
            AudioPacket audio;
            if (with_audio && xQueueReceive(session.audio, &audio, 0) == pdTRUE) {
                const auto first = audio.sequence;
                auto expected = first;
                std::size_t length = 0;
                do {
                    if (audio.sequence != expected) {
                        session.error.store(ESP_ERR_INVALID_STATE);
                        break;
                    }
                    std::memcpy(audio_buffer.get() + length, audio.data.data(), audio.data.size());
                    length += audio.data.size();
                    ++expected;
                } while (length < AUDIO_BYTES * AUDIO_BATCH_PACKETS &&
                         xQueueReceive(session.audio, &audio, 0) == pdTRUE);
                result = send_packet(buffer.get(), PacketKind::AUDIO, first, first * 20000ULL, audio_buffer.get(),
                                     length, writer, context, write_quantum);
                if (result == ESP_OK) samples_sent += length / sizeof(std::int16_t);
            } else {
                CameraFrame frame;
                if (xQueueReceive(session.video, &frame, pdMS_TO_TICKS(20)) == pdTRUE) {
                    const auto timestamp = frame.timestamp_us - session.start;
                    if (timestamp >= 0) {
                        const auto sending_started = esp_timer_get_time();
                        const auto age = static_cast<unsigned>(sending_started - session.start - timestamp);
                        frame_age_max_us = std::max(frame_age_max_us, age);
                        result = send_packet(buffer.get(), PacketKind::JPEG, frames_sent,
                                             static_cast<std::uint64_t>(timestamp), frame.data, frame.size, writer,
                                             context, write_quantum);
                        const auto sending_us = static_cast<unsigned>(esp_timer_get_time() - sending_started);
                        send_us += sending_us;
                        send_max_us = std::max(send_max_us, sending_us);
                        if (result == ESP_OK) {
                            ++frames_sent;
                            video_bytes += frame.size;
                        }
                    }
                    return_frame(session, frame);
                }
            }
            if (esp_timer_get_time() >= next_status) {
                char status[768]{};
                snprintf(status, sizeof(status),
                         "{\"video_acquired\":%u,\"video_sent\":%u,\"video_dropped\":%u,"
                         "\"audio_samples_sent\":%u,\"audio_dropped\":%u,\"audio_overruns\":%u,"
                         "\"video_bytes\":%llu,\"send_us\":%llu,\"send_max_us\":%u,"
                         "\"capture_wait_us\":%u,\"frame_age_max_us\":%u}",
                         session.video_acquired.load(), frames_sent, session.video_drops.load(), samples_sent,
                         session.audio_drops.load(), session.audio_overruns.load(),
                         static_cast<unsigned long long>(video_bytes), static_cast<unsigned long long>(send_us),
                         send_max_us, session.capture_us.load(), frame_age_max_us);
                if (result == ESP_OK)
                    result = send_packet(buffer.get(), PacketKind::STATUS, 0, esp_timer_get_time() - session.start,
                                         reinterpret_cast<const std::uint8_t*>(status), std::strlen(status), writer,
                                         context, write_quantum);
                next_status = esp_timer_get_time() + 1000000;
            }
            if (session.audio_drops.load() != 0 || session.audio_overruns.load() != 0)
                session.error.store(ESP_ERR_INVALID_STATE);
        }
    }
    session.stop.store(true);
    for (unsigned worker = 0; worker < workers; ++worker) xSemaphoreTake(session.done, portMAX_DELAY);
    if (session.video != nullptr) {
        CameraFrame frame;
        while (xQueueReceive(session.video, &frame, 0) == pdTRUE) return_frame(session, frame);
    }
    if (audio_started) {
        const auto stopped = stop_audio();
        if (result == ESP_OK && stopped != ESP_OK) result = stopped;
    }
    if (camera_started) {
        const auto stopped = protocol_started ? stop_stream_camera(buffer.get(), writer, context, write_quantum,
                                                                   esp_timer_get_time() - session.start)
                                              : release_camera();
        if (result == ESP_OK && stopped != ESP_OK) result = stopped;
    }
    if (session.video != nullptr) vQueueDelete(session.video);
    if (session.audio != nullptr) vQueueDelete(session.audio);
    if (session.done != nullptr) vSemaphoreDelete(session.done);
    if (session.frame_slots != nullptr) vSemaphoreDelete(session.frame_slots);
    if (result == ESP_OK) result = session.error.load();
    if (result == ESP_OK || result == STREAM_CANCELLED) {
        char complete[768]{};
        snprintf(complete, sizeof(complete),
                 "{\"reason\":\"%s\",\"video_acquired\":%u,\"video_sent\":%u,\"video_dropped\":%u,"
                 "\"audio_samples_sent\":%u,\"audio_dropped\":%u,\"audio_overruns\":%u,"
                 "\"video_bytes\":%llu,\"send_us\":%llu,\"send_max_us\":%u,"
                 "\"capture_wait_us\":%u,\"frame_age_max_us\":%u}",
                 result == STREAM_CANCELLED ? "stopped" : "duration_limit", session.video_acquired.load(), frames_sent,
                 session.video_drops.load(), samples_sent, session.audio_drops.load(), session.audio_overruns.load(),
                 static_cast<unsigned long long>(video_bytes), static_cast<unsigned long long>(send_us), send_max_us,
                 session.capture_us.load(), frame_age_max_us);
        result = send_packet(buffer.get(), PacketKind::END, frames_sent, seconds * 1000000ULL,
                             reinterpret_cast<const std::uint8_t*>(complete), std::strlen(complete), writer, context,
                             write_quantum);
    }
    release_capture();
    return result;
}

} // namespace kidi::esp32
