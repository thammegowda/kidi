#include "kidi_audio.h"

#include <stdbool.h>
#include <string.h>

#include "accessory_protocol.h"
#include "bsp/esp-bsp.h"
#include "esp_check.h"
#include "esp_codec_dev.h"
#include "esp_log.h"

#define KIDI_AUDIO_FRAMES_PER_BLOCK 1024U
#define KIDI_AUDIO_INPUT_CHANNELS 2U
#define KIDI_AUDIO_INPUT_GAIN_DB 42.0F

static const char* TAG = "kidi_audio";

esp_err_t kidi_audio_stream(uint32_t maximum_seconds, kidi_audio_sink_t sink, void* context,
                            kidi_audio_result_t* result) {
    if (maximum_seconds == 0 || maximum_seconds > KIDI_ACCESSORY_MAX_AUDIO_SECONDS || sink == NULL || result == NULL) {
        return ESP_ERR_INVALID_ARG;
    }
    *result = {};
    esp_codec_dev_handle_t speaker = bsp_audio_codec_speaker_init();
    esp_codec_dev_handle_t microphone = bsp_audio_codec_microphone_init();
    ESP_RETURN_ON_FALSE(speaker != NULL && microphone != NULL, ESP_FAIL, TAG, "failed to initialize ES8311 codec");

    esp_codec_dev_sample_info_t format{};
    format.sample_rate = KIDI_ACCESSORY_AUDIO_SAMPLE_RATE;
    format.channel = KIDI_AUDIO_INPUT_CHANNELS;
    format.bits_per_sample = KIDI_ACCESSORY_AUDIO_BITS_PER_SAMPLE;
    ESP_RETURN_ON_ERROR(esp_codec_dev_open(speaker, &format), TAG, "failed to open codec clock path");
    esp_err_t error = esp_codec_dev_set_in_gain(microphone, KIDI_AUDIO_INPUT_GAIN_DB);
    if (error == ESP_OK) {
        error = esp_codec_dev_open(microphone, &format);
    }
    if (error != ESP_OK) {
        esp_codec_dev_close(speaker);
        return error;
    }

    int16_t stereo[KIDI_AUDIO_FRAMES_PER_BLOCK * KIDI_AUDIO_INPUT_CHANNELS];
    int16_t mono[KIDI_AUDIO_FRAMES_PER_BLOCK];
    error = esp_codec_dev_read(microphone, stereo, sizeof(stereo));
    size_t remaining = (size_t)maximum_seconds * (size_t)KIDI_ACCESSORY_AUDIO_SAMPLE_RATE;
    while (error == ESP_OK && remaining > 0) {
        size_t frames = remaining < KIDI_AUDIO_FRAMES_PER_BLOCK ? remaining : KIDI_AUDIO_FRAMES_PER_BLOCK;
        size_t input_bytes = frames * KIDI_AUDIO_INPUT_CHANNELS * sizeof(int16_t);
        error = esp_codec_dev_read(microphone, stereo, input_bytes);
        if (error != ESP_OK) {
            break;
        }
        for (size_t index = 0; index < frames; ++index) {
            mono[index] = stereo[index * KIDI_AUDIO_INPUT_CHANNELS];
        }
        error = sink(mono, frames, context);
        if (error != ESP_OK) {
            break;
        }
        result->sample_count += frames;
        remaining -= frames;
    }

    esp_err_t microphone_close = esp_codec_dev_close(microphone);
    esp_err_t speaker_close = esp_codec_dev_close(speaker);
    if (error == ESP_OK && microphone_close != ESP_OK) {
        error = microphone_close;
    }
    if (error == ESP_OK && speaker_close != ESP_OK) {
        error = speaker_close;
    }
    result->sample_rate = KIDI_ACCESSORY_AUDIO_SAMPLE_RATE;
    result->channels = KIDI_ACCESSORY_AUDIO_CHANNELS;
    result->bits_per_sample = KIDI_ACCESSORY_AUDIO_BITS_PER_SAMPLE;
    return error;
}
