#pragma once

#include <stddef.h>
#include <stdint.h>

#include "esp_err.h"

typedef struct {
    uint8_t* data;
    size_t size;
    uint32_t width;
    uint32_t height;
} kidi_camera_photo_t;

esp_err_t kidi_camera_capture_photo(kidi_camera_photo_t* photo);
void kidi_camera_release_photo(kidi_camera_photo_t* photo);
