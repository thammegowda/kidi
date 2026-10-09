#pragma once

#include <cstddef>
#include <cstdint>

#include "accessory_protocol.h"
#include "esp_err.h"

#define KIDI_TICKET_KEY_BYTES 32U
#define KIDI_TICKET_CONTROLLER_ID_BYTES 16U

typedef struct {
    kidi_accessory_media_kind_t kind;
    uint64_t issued_at;
    uint64_t expires_at;
    uint8_t nonce[KIDI_ACCESSORY_TICKET_NONCE_BYTES];
    uint8_t controller_id[KIDI_TICKET_CONTROLLER_ID_BYTES];
    uint8_t parameter_hash[KIDI_ACCESSORY_TICKET_PARAMETER_HASH_BYTES];
} kidi_ticket_claims_t;

esp_err_t kidi_ticket_issue(const uint8_t key[KIDI_TICKET_KEY_BYTES], kidi_accessory_media_kind_t kind,
                            uint64_t issued_at, const uint8_t nonce[KIDI_ACCESSORY_TICKET_NONCE_BYTES],
                            const uint8_t controller_id[KIDI_TICKET_CONTROLLER_ID_BYTES],
                            const uint8_t parameter_hash[KIDI_ACCESSORY_TICKET_PARAMETER_HASH_BYTES],
                            uint8_t output[KIDI_ACCESSORY_TICKET_BYTES]);

esp_err_t kidi_ticket_validate(const uint8_t key[KIDI_TICKET_KEY_BYTES], const uint8_t* ticket, size_t ticket_size,
                               kidi_accessory_media_kind_t expected_kind, uint64_t now,
                               const uint8_t expected_parameter_hash[KIDI_ACCESSORY_TICKET_PARAMETER_HASH_BYTES],
                               kidi_ticket_claims_t* claims);

esp_err_t kidi_ticket_encode_base64url(const uint8_t ticket[KIDI_ACCESSORY_TICKET_BYTES],
                                       char output[KIDI_ACCESSORY_TICKET_BASE64URL_BYTES + 1]);

esp_err_t kidi_ticket_decode_base64url(const char* encoded, size_t encoded_size,
                                       uint8_t output[KIDI_ACCESSORY_TICKET_BYTES]);
