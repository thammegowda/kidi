#include "audio.h"
#include "w11.h"

namespace kidi::esp32 {
namespace {

constexpr I2sPort MICROPHONE_PORT = I2S_NUM_0;

} // namespace

auto start_audio(QueueHandle* events) -> Error {
    I2sConfig config{};
    config.mode = static_cast<I2sMode>(I2S_MODE_MASTER | I2S_MODE_RX | I2S_MODE_PDM);
    config.sample_rate = AUDIO_SAMPLE_RATE;
    config.bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT;
    config.channel_format = I2S_CHANNEL_FMT_ONLY_LEFT;
    config.communication_format = I2S_COMM_FORMAT_STAND_I2S;
    config.intr_alloc_flags = ESP_INTR_FLAG_LEVEL1;
    config.dma_buf_count = 8;
    config.dma_buf_len = 256;
    auto result = i2s_driver_install(MICROPHONE_PORT, &config, events == nullptr ? 0 : 32, events);
    if (result != ESP_OK) return result;
    I2sPins pins{};
    pins.mck_io_num = I2S_PIN_NO_CHANGE;
    pins.bck_io_num = I2S_PIN_NO_CHANGE;
    pins.ws_io_num = W11Pins::MICROPHONE_CLOCK;
    pins.data_out_num = I2S_PIN_NO_CHANGE;
    pins.data_in_num = W11Pins::MICROPHONE_DATA;
    result = i2s_set_pin(MICROPHONE_PORT, &pins);
    if (result != ESP_OK) {
        const auto cleanup = stop_audio();
        if (cleanup != ESP_OK) report_error("microphone shutdown", cleanup);
    }
    return result;
}

auto read_audio(std::uint8_t* destination, std::size_t length) -> Error {
    std::size_t offset = 0;
    while (offset < length) {
        std::size_t received = 0;
        const auto result =
            i2s_read(MICROPHONE_PORT, destination + offset, length - offset, &received, pdMS_TO_TICKS(1000));
        if (result != ESP_OK) return result;
        if (received == 0 || received % sizeof(std::int16_t) != 0) return ESP_ERR_TIMEOUT;
        offset += received;
    }
    return ESP_OK;
}

auto stop_audio() -> Error { return i2s_driver_uninstall(MICROPHONE_PORT); }

} // namespace kidi::esp32
