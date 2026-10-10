#include "pdm_audio.h"

#include <driver/gpio.h>

namespace kidi::esp32 {
namespace {

struct PdmContext {
    PdmMicrophoneConfig configuration;
    QueueHandle events = nullptr;
    AudioStats stats;
};

PdmContext context;

auto drain_events(PdmContext& state) -> void {
    i2s_event_t event;
    while (state.events != nullptr && xQueueReceive(state.events, &event, 0) == pdTRUE) {
        if (event.type == I2S_EVENT_RX_Q_OVF) ++state.stats.overruns;
        if (event.type == I2S_EVENT_DMA_ERROR) ++state.stats.dma_errors;
    }
}

auto start(void* opaque) -> Error {
    auto& state = *static_cast<PdmContext*>(opaque);
    i2s_config_t sdk{};
    sdk.mode = static_cast<i2s_mode_t>(I2S_MODE_MASTER | I2S_MODE_RX | I2S_MODE_PDM);
    sdk.sample_rate = AUDIO_SAMPLE_RATE;
    sdk.bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT;
    sdk.channel_format = I2S_CHANNEL_FMT_ONLY_LEFT;
    sdk.communication_format = I2S_COMM_FORMAT_STAND_I2S;
    sdk.intr_alloc_flags = ESP_INTR_FLAG_LEVEL1;
    sdk.dma_buf_count = 8;
    sdk.dma_buf_len = 256;
    state.stats = {};
    state.events = nullptr;
    auto result = i2s_driver_install(state.configuration.port, &sdk, 32, &state.events);
    if (result != ESP_OK) return result;
    i2s_pin_config_t pins{};
    pins.mck_io_num = I2S_PIN_NO_CHANGE;
    pins.bck_io_num = I2S_PIN_NO_CHANGE;
    pins.ws_io_num = state.configuration.clock;
    pins.data_out_num = I2S_PIN_NO_CHANGE;
    pins.data_in_num = state.configuration.data;
    result = i2s_set_pin(state.configuration.port, &pins);
    if (result != ESP_OK) {
        i2s_driver_uninstall(state.configuration.port);
        state.events = nullptr;
    }
    return result;
}

auto read(void* opaque, std::uint8_t* destination, std::size_t length) -> Error {
    auto& state = *static_cast<PdmContext*>(opaque);
    std::size_t offset = 0;
    while (offset < length) {
        std::size_t received = 0;
        const auto result =
            i2s_read(state.configuration.port, destination + offset, length - offset, &received, pdMS_TO_TICKS(1000));
        drain_events(state);
        if (result != ESP_OK) return result;
        if (received == 0 || received % sizeof(std::int16_t) != 0) return ESP_ERR_TIMEOUT;
        offset += received;
    }
    return ESP_OK;
}

auto stats(void* opaque, AudioStats& result) -> Error {
    auto& state = *static_cast<PdmContext*>(opaque);
    drain_events(state);
    result = state.stats;
    return ESP_OK;
}

auto stop(void* opaque) -> Error {
    auto& state = *static_cast<PdmContext*>(opaque);
    drain_events(state);
    const auto result = i2s_driver_uninstall(state.configuration.port);
    if (result == ESP_OK) state.events = nullptr;
    return result;
}

} // namespace

auto install_pdm_microphone(const PdmMicrophoneConfig& configuration) -> Error {
    if (configuration.port < I2S_NUM_0 || configuration.port >= I2S_NUM_MAX || configuration.clock < 0 ||
        configuration.data < 0 || configuration.clock == configuration.data ||
        !GPIO_IS_VALID_OUTPUT_GPIO(configuration.clock) || !GPIO_IS_VALID_GPIO(configuration.data))
        return ESP_ERR_INVALID_ARG;
    const auto result = install_audio_backend({&context, start, read, stats, stop});
    if (result == ESP_OK) context.configuration = configuration;
    return result;
}

} // namespace kidi::esp32
