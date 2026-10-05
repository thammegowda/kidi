#include "stream.h"
#include "audio.h"
#include "camera.h"
#include "wireless.h"

#include <array>
#include <atomic>
#include <cstring>
#include <cstdlib>
#include <memory>

#include <Arduino.h>
#include <esp_heap_caps.h>

namespace kidi::esp32 {
namespace {

constexpr unsigned MAX_STREAM_SECONDS = 300;
constexpr std::size_t AUDIO_BYTES = AUDIO_SAMPLE_RATE / 50 * sizeof(std::int16_t);
constexpr std::size_t MAX_PACKET_BYTES = 1024 * 1024;
constexpr std::size_t HEADER_BYTES = 20;
constexpr unsigned AUDIO_QUEUE_PACKETS = 32;

enum class PacketKind : std::uint8_t { JPEG = 1, AUDIO = 2, STATUS = 3, END = 4 };

struct AudioPacket {
    std::uint32_t sequence = 0;
    std::array<std::uint8_t, AUDIO_BYTES> data{};
};

struct StreamSession {
    QueueHandle video = nullptr;
    QueueHandle audio = nullptr;
    QueueHandle audio_events = nullptr;
    SemaphoreHandle done = nullptr;
    std::atomic<bool> stop{false};
    std::atomic<Error> error{ESP_OK};
    std::atomic<unsigned> video_drops{0};
    std::atomic<unsigned> audio_drops{0};
    std::atomic<unsigned> audio_overruns{0};
    std::atomic<unsigned> video_acquired{0};
    std::int64_t start = 0;
    std::int64_t duration = 0;
};

auto camera_worker(void* argument) -> void {
    auto& session = *static_cast<StreamSession*>(argument);
    while (!session.stop.load() && esp_timer_get_time() - session.start < session.duration) {
        auto* frame = esp_camera_fb_get();
        if (frame == nullptr) {
            session.error.store(ESP_FAIL);
            break;
        }
        ++session.video_acquired;
        CameraFrame* old = nullptr;
        if (xQueueReceive(session.video, &old, 0) == pdTRUE) {
            esp_camera_fb_return(old);
            ++session.video_drops;
        }
        if (xQueueSend(session.video, &frame, 0) != pdTRUE) {
            esp_camera_fb_return(frame);
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
        I2sEvent event;
        while (xQueueReceive(session.audio_events, &event, 0) == pdTRUE) {
            if (event.type == I2S_EVENT_RX_Q_OVF || event.type == I2S_EVENT_DMA_ERROR) ++session.audio_overruns;
        }
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
                 const std::uint8_t* payload, std::size_t length, ClipWriter writer, void* context) -> Error {
    if (length > MAX_PACKET_BYTES) return ESP_ERR_INVALID_SIZE;
    buffer[0] = static_cast<std::uint8_t>(kind);
    buffer[1] = buffer[2] = buffer[3] = 0;
    put_u32(buffer + 4, sequence);
    put_u32(buffer + 8, static_cast<std::uint32_t>(timestamp >> 32));
    put_u32(buffer + 12, static_cast<std::uint32_t>(timestamp));
    put_u32(buffer + 16, static_cast<std::uint32_t>(length));
    if (payload != buffer + HEADER_BYTES) std::memcpy(buffer + HEADER_BYTES, payload, length);
    return writer(buffer, HEADER_BYTES + length, context);
}

} // namespace

auto stream_media(bool with_audio, unsigned seconds, ClipWriter writer, void* context, CameraFrameSize frame_size)
    -> Error {
    if (writer == nullptr || seconds < 1 || seconds > MAX_STREAM_SECONDS ||
        (frame_size != FRAMESIZE_HD && frame_size != FRAMESIZE_96X96))
        return ESP_ERR_INVALID_ARG;
    auto result = acquire_capture();
    if (result != ESP_OK) return result;
    StreamSession session;
    session.video = xQueueCreate(1, sizeof(CameraFrame*));
    session.audio = with_audio ? xQueueCreate(AUDIO_QUEUE_PACKETS, sizeof(AudioPacket)) : nullptr;
    session.done = xSemaphoreCreateCounting(2, 0);
    std::unique_ptr<std::uint8_t, decltype(&std::free)> buffer(
        static_cast<std::uint8_t*>(
            heap_caps_malloc(MAX_PACKET_BYTES + HEADER_BYTES, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT)),
        std::free);
    auto camera_started = false;
    auto audio_started = false;
    unsigned workers = 0;
    unsigned frames_sent = 0;
    unsigned samples_sent = 0;
    if (session.video == nullptr || session.done == nullptr || !buffer || (with_audio && session.audio == nullptr)) {
        result = ESP_ERR_NO_MEM;
    } else {
        result = prepare_camera(frame_size, 3);
        camera_started = result == ESP_OK;
        if (camera_started && with_audio) {
            result = start_audio(&session.audio_events);
            audio_started = result == ESP_OK;
            if (audio_started) {
                std::array<std::uint8_t, 4096> warmup{};
                result = read_audio(warmup.data(), warmup.size());
            }
        }
        if (result == ESP_OK) {
            poll_wireless();
            session.start = esp_timer_get_time();
            session.duration = seconds * 1000000LL;
            constexpr char BEGIN[] = "KIDI_LIVE_BEGIN\n";
            result = writer(reinterpret_cast<const std::uint8_t*>(BEGIN), sizeof(BEGIN) - 1, context);
            char status[256]{};
            snprintf(status, sizeof(status),
                     "{\"protocol\":1,\"width\":%u,\"height\":%u,\"sample_rate\":%u,"
                     "\"audio\":%s,\"max_seconds\":%u,\"internal_free\":%u}",
                     static_cast<unsigned>(resolution[frame_size].width),
                     static_cast<unsigned>(resolution[frame_size].height), AUDIO_SAMPLE_RATE,
                     with_audio ? "true" : "false", seconds,
                     static_cast<unsigned>(heap_caps_get_free_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT)));
            if (result == ESP_OK)
                result =
                    send_packet(buffer.get(), PacketKind::STATUS, 0, 0, reinterpret_cast<const std::uint8_t*>(status),
                                std::strlen(status), writer, context);
        }
        if (result == ESP_OK) {
            if (xTaskCreatePinnedToCore(camera_worker, "kidi-camera", 4096, &session, 2, nullptr, 1) == pdPASS)
                ++workers;
            else
                result = ESP_ERR_NO_MEM;
        }
        if (result == ESP_OK && with_audio) {
            if (xTaskCreatePinnedToCore(audio_worker, "kidi-audio", 8192, &session, 2, nullptr, 0) == pdPASS)
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
                    std::memcpy(buffer.get() + HEADER_BYTES + length, audio.data.data(), audio.data.size());
                    length += audio.data.size();
                    ++expected;
                } while (length < AUDIO_BYTES * 8 && xQueueReceive(session.audio, &audio, 0) == pdTRUE);
                result = send_packet(buffer.get(), PacketKind::AUDIO, first, first * 20000ULL,
                                     buffer.get() + HEADER_BYTES, length, writer, context);
                if (result == ESP_OK) samples_sent += length / sizeof(std::int16_t);
            } else {
                CameraFrame* frame = nullptr;
                if (xQueueReceive(session.video, &frame, pdMS_TO_TICKS(5)) == pdTRUE) {
                    const auto timestamp =
                        frame->timestamp.tv_sec * 1000000LL + frame->timestamp.tv_usec - session.start;
                    if (timestamp >= 0) {
                        result =
                            send_packet(buffer.get(), PacketKind::JPEG, frames_sent,
                                        static_cast<std::uint64_t>(timestamp), frame->buf, frame->len, writer, context);
                        if (result == ESP_OK) ++frames_sent;
                    }
                    esp_camera_fb_return(frame);
                }
            }
            if (esp_timer_get_time() >= next_status) {
                char status[256]{};
                snprintf(status, sizeof(status),
                         "{\"video_acquired\":%u,\"video_sent\":%u,\"video_dropped\":%u,"
                         "\"audio_samples_sent\":%u,\"audio_dropped\":%u,\"audio_overruns\":%u}",
                         session.video_acquired.load(), frames_sent, session.video_drops.load(), samples_sent,
                         session.audio_drops.load(), session.audio_overruns.load());
                if (result == ESP_OK)
                    result = send_packet(buffer.get(), PacketKind::STATUS, 0, esp_timer_get_time() - session.start,
                                         reinterpret_cast<const std::uint8_t*>(status), std::strlen(status), writer,
                                         context);
                next_status = esp_timer_get_time() + 1000000;
            }
            if (session.audio_drops.load() != 0 || session.audio_overruns.load() != 0)
                session.error.store(ESP_ERR_INVALID_STATE);
        }
    }
    session.stop.store(true);
    for (unsigned worker = 0; worker < workers; ++worker) xSemaphoreTake(session.done, portMAX_DELAY);
    if (session.video != nullptr) {
        CameraFrame* frame = nullptr;
        while (xQueueReceive(session.video, &frame, 0) == pdTRUE) esp_camera_fb_return(frame);
    }
    if (audio_started) {
        const auto stopped = stop_audio();
        if (result == ESP_OK && stopped != ESP_OK) result = stopped;
    }
    if (camera_started) {
        const auto stopped = release_camera();
        if (result == ESP_OK && stopped != ESP_OK) result = stopped;
    }
    if (session.video != nullptr) vQueueDelete(session.video);
    if (session.audio != nullptr) vQueueDelete(session.audio);
    if (session.done != nullptr) vSemaphoreDelete(session.done);
    if (result == ESP_OK) result = session.error.load();
    if (result == ESP_OK || result == STREAM_CANCELLED) {
        const auto* complete =
            result == STREAM_CANCELLED ? "{\"reason\":\"stopped\"}" : "{\"reason\":\"duration_limit\"}";
        result = send_packet(buffer.get(), PacketKind::END, frames_sent, seconds * 1000000ULL,
                             reinterpret_cast<const std::uint8_t*>(complete), std::strlen(complete), writer, context);
    }
    release_capture();
    return result;
}

} // namespace kidi::esp32
