#include "audio.h"

namespace kidi::esp32 {
namespace {

AudioBackend audio_backend;
bool configured = false;
bool running = false;

} // namespace

auto install_audio_backend(const AudioBackend& backend) -> Error {
    if (configured || running) return ESP_ERR_INVALID_STATE;
    if (backend.start == nullptr || backend.read == nullptr || backend.stats == nullptr || backend.stop == nullptr)
        return ESP_ERR_INVALID_ARG;
    audio_backend = backend;
    configured = true;
    return ESP_OK;
}

auto audio_configured() -> bool { return configured; }

auto start_audio() -> Error {
    if (!configured || running) return ESP_ERR_INVALID_STATE;
    const auto result = audio_backend.start(audio_backend.context);
    if (result == ESP_OK) running = true;
    return result;
}

auto read_audio(std::uint8_t* destination, std::size_t length) -> Error {
    if (!running) return ESP_ERR_INVALID_STATE;
    if (destination == nullptr || length == 0 || length % sizeof(std::int16_t) != 0) return ESP_ERR_INVALID_ARG;
    return audio_backend.read(audio_backend.context, destination, length);
}

auto read_audio_stats(AudioStats& stats) -> Error {
    if (!running) return ESP_ERR_INVALID_STATE;
    return audio_backend.stats(audio_backend.context, stats);
}

auto stop_audio() -> Error {
    if (!running) return ESP_ERR_INVALID_STATE;
    const auto result = audio_backend.stop(audio_backend.context);
    if (result == ESP_OK) running = false;
    return result;
}

} // namespace kidi::esp32
