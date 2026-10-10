#pragma once

#include <stddef.h>
#include <stdint.h>

#include "esp_err.h"

typedef esp_err_t (*kidi_audio_sink_t)(const int16_t* samples, size_t sample_count, void* context);

typedef struct {
    size_t sample_count;
    uint32_t sample_rate;
    uint32_t channels;
    uint32_t bits_per_sample;
} kidi_audio_result_t;

esp_err_t kidi_audio_stream(uint32_t maximum_seconds, kidi_audio_sink_t sink, void* context,
                            kidi_audio_result_t* result);
