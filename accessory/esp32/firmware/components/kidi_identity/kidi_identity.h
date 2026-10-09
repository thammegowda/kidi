#pragma once

#include <cstddef>
#include <cstdint>

#include "esp_err.h"

#define KIDI_IDENTITY_PRIVATE_KEY_BYTES 32U
#define KIDI_IDENTITY_PUBLIC_KEY_BYTES 65U
#define KIDI_IDENTITY_DEVICE_ID_BYTES 16U
#define KIDI_IDENTITY_MAX_SIGNATURE_BYTES 72U

esp_err_t kidi_identity_init(const char* nvs_namespace, const char* nvs_key);

esp_err_t kidi_identity_get_public_key(uint8_t output[KIDI_IDENTITY_PUBLIC_KEY_BYTES]);

esp_err_t kidi_identity_get_device_id(uint8_t output[KIDI_IDENTITY_DEVICE_ID_BYTES]);

esp_err_t kidi_identity_sign(const uint8_t* message, size_t message_size,
                             uint8_t signature[KIDI_IDENTITY_MAX_SIGNATURE_BYTES], size_t* signature_size);
