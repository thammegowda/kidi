#pragma once

#include <cstddef>
#include <cstdint>

#include "esp_err.h"
#include "kidi_pairing.h"

#define KIDI_INVITATION_MAX_LIFETIME_SECONDS 900U
#define KIDI_INVITATION_URI_MAX_BYTES 1024U

esp_err_t kidi_invitation_init(void);

esp_err_t kidi_invitation_create(uint64_t now, uint32_t lifetime_seconds, const char* device_id_hex,
                                 const char* device_name, char output[KIDI_INVITATION_URI_MAX_BYTES],
                                 size_t* output_size);

esp_err_t kidi_invitation_snapshot(uint64_t now, uint8_t secret[KIDI_PAIRING_INVITATION_SECRET_BYTES],
                                   uint64_t* expires_at);

esp_err_t kidi_invitation_consume(void);
