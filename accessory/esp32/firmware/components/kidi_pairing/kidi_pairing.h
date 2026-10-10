#pragma once

#include <cstddef>
#include <cstdint>

#include "esp_err.h"
#include "kidi_tls_identity.h"

#define KIDI_PAIRING_INVITATION_SECRET_BYTES 32U

typedef struct {
    const char* device_id_hex;
    const char* device_name;
    const char* host;
    const char* softap_ssid;
    const char* softap_password;
    const uint8_t* control_certificate_sha256;
    const uint8_t* media_certificate_sha256;
    const uint8_t* invitation_secret;
    uint64_t invitation_expires_at;
    uint64_t current_epoch;
} kidi_pairing_config_t;

esp_err_t kidi_pairing_create_response(const uint8_t* request, size_t request_size, const kidi_pairing_config_t* config,
                                       uint8_t* response, size_t response_capacity, size_t* response_size,
                                       bool* created_controller, uint8_t paired_controller_id[16]);
