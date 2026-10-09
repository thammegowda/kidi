#include "kidi_ticket.h"

#include <stdbool.h>
#include <string.h>

#include "mbedtls/base64.h"
#include "mbedtls/md.h"

enum {
    VERSION_OFFSET = 0,
    KIND_OFFSET = 1,
    ISSUED_AT_OFFSET = 2,
    EXPIRES_AT_OFFSET = 10,
    NONCE_OFFSET = 18,
    CONTROLLER_ID_OFFSET = 34,
    PARAMETER_HASH_OFFSET = 50,
    MAC_OFFSET = KIDI_ACCESSORY_TICKET_BODY_BYTES,
};

static void write_u64_be(uint8_t* destination, uint64_t value) {
    for (unsigned index = 0; index < 8; ++index) {
        destination[7 - index] = (uint8_t)(value >> (index * 8U));
    }
}

static uint64_t read_u64_be(const uint8_t* source) {
    uint64_t value = 0;
    for (unsigned index = 0; index < 8; ++index) {
        value = (value << 8U) | source[index];
    }
    return value;
}

static bool constant_time_equal(const uint8_t* left, const uint8_t* right, size_t size) {
    uint8_t difference = 0;
    for (size_t index = 0; index < size; ++index) {
        difference |= left[index] ^ right[index];
    }
    return difference == 0;
}

static esp_err_t calculate_mac(const uint8_t key[KIDI_TICKET_KEY_BYTES], const uint8_t* body,
                               uint8_t output[KIDI_ACCESSORY_TICKET_MAC_BYTES]) {
    const mbedtls_md_info_t* sha256 = mbedtls_md_info_from_type(MBEDTLS_MD_SHA256);
    if (sha256 == NULL) {
        return ESP_ERR_NOT_SUPPORTED;
    }
    return mbedtls_md_hmac(sha256, key, KIDI_TICKET_KEY_BYTES, body, KIDI_ACCESSORY_TICKET_BODY_BYTES, output) == 0
               ? ESP_OK
               : ESP_FAIL;
}

static bool valid_kind(kidi_accessory_media_kind_t kind) {
    return kind == KIDI_ACCESSORY_MEDIA_PHOTO || kind == KIDI_ACCESSORY_MEDIA_AUDIO;
}

esp_err_t kidi_ticket_issue(const uint8_t key[KIDI_TICKET_KEY_BYTES], kidi_accessory_media_kind_t kind,
                            uint64_t issued_at, const uint8_t nonce[KIDI_ACCESSORY_TICKET_NONCE_BYTES],
                            const uint8_t controller_id[KIDI_TICKET_CONTROLLER_ID_BYTES],
                            const uint8_t parameter_hash[KIDI_ACCESSORY_TICKET_PARAMETER_HASH_BYTES],
                            uint8_t output[KIDI_ACCESSORY_TICKET_BYTES]) {
    if (key == NULL || nonce == NULL || controller_id == NULL || parameter_hash == NULL || output == NULL ||
        !valid_kind(kind) || issued_at > UINT64_MAX - KIDI_ACCESSORY_TICKET_LIFETIME_SECONDS) {
        return ESP_ERR_INVALID_ARG;
    }
    output[VERSION_OFFSET] = KIDI_ACCESSORY_PROTOCOL_VERSION;
    output[KIND_OFFSET] = (uint8_t)kind;
    write_u64_be(output + ISSUED_AT_OFFSET, issued_at);
    write_u64_be(output + EXPIRES_AT_OFFSET, issued_at + KIDI_ACCESSORY_TICKET_LIFETIME_SECONDS);
    memcpy(output + NONCE_OFFSET, nonce, KIDI_ACCESSORY_TICKET_NONCE_BYTES);
    memcpy(output + CONTROLLER_ID_OFFSET, controller_id, KIDI_TICKET_CONTROLLER_ID_BYTES);
    memcpy(output + PARAMETER_HASH_OFFSET, parameter_hash, KIDI_ACCESSORY_TICKET_PARAMETER_HASH_BYTES);
    return calculate_mac(key, output, output + MAC_OFFSET);
}

esp_err_t kidi_ticket_validate(const uint8_t key[KIDI_TICKET_KEY_BYTES], const uint8_t* ticket, size_t ticket_size,
                               kidi_accessory_media_kind_t expected_kind, uint64_t now,
                               const uint8_t expected_parameter_hash[KIDI_ACCESSORY_TICKET_PARAMETER_HASH_BYTES],
                               kidi_ticket_claims_t* claims) {
    if (key == NULL || ticket == NULL || expected_parameter_hash == NULL || claims == NULL ||
        ticket_size != KIDI_ACCESSORY_TICKET_BYTES || !valid_kind(expected_kind)) {
        return ESP_ERR_INVALID_ARG;
    }
    uint8_t expected_mac[KIDI_ACCESSORY_TICKET_MAC_BYTES];
    esp_err_t error = calculate_mac(key, ticket, expected_mac);
    if (error != ESP_OK) {
        return error;
    }
    if (!constant_time_equal(ticket + MAC_OFFSET, expected_mac, sizeof(expected_mac))) {
        return ESP_ERR_INVALID_CRC;
    }
    if (ticket[VERSION_OFFSET] != KIDI_ACCESSORY_PROTOCOL_VERSION || ticket[KIND_OFFSET] != (uint8_t)expected_kind ||
        !constant_time_equal(ticket + PARAMETER_HASH_OFFSET, expected_parameter_hash,
                             KIDI_ACCESSORY_TICKET_PARAMETER_HASH_BYTES)) {
        return ESP_ERR_INVALID_ARG;
    }
    uint64_t issued_at = read_u64_be(ticket + ISSUED_AT_OFFSET);
    uint64_t expires_at = read_u64_be(ticket + EXPIRES_AT_OFFSET);
    if (expires_at < issued_at || expires_at - issued_at > KIDI_ACCESSORY_TICKET_LIFETIME_SECONDS || now < issued_at ||
        now > expires_at) {
        return ESP_ERR_INVALID_STATE;
    }

    claims->kind = expected_kind;
    claims->issued_at = issued_at;
    claims->expires_at = expires_at;
    memcpy(claims->nonce, ticket + NONCE_OFFSET, sizeof(claims->nonce));
    memcpy(claims->controller_id, ticket + CONTROLLER_ID_OFFSET, sizeof(claims->controller_id));
    memcpy(claims->parameter_hash, ticket + PARAMETER_HASH_OFFSET, sizeof(claims->parameter_hash));
    return ESP_OK;
}

esp_err_t kidi_ticket_encode_base64url(const uint8_t ticket[KIDI_ACCESSORY_TICKET_BYTES],
                                       char output[KIDI_ACCESSORY_TICKET_BASE64URL_BYTES + 1]) {
    if (ticket == NULL || output == NULL) {
        return ESP_ERR_INVALID_ARG;
    }
    size_t encoded_size = 0;
    int result = mbedtls_base64_encode((uint8_t*)output, KIDI_ACCESSORY_TICKET_BASE64URL_BYTES + 1, &encoded_size,
                                       ticket, KIDI_ACCESSORY_TICKET_BYTES);
    if (result != 0 || encoded_size != KIDI_ACCESSORY_TICKET_BASE64URL_BYTES) {
        return ESP_FAIL;
    }
    for (size_t index = 0; index < encoded_size; ++index) {
        if (output[index] == '+') {
            output[index] = '-';
        } else if (output[index] == '/') {
            output[index] = '_';
        }
    }
    output[encoded_size] = '\0';
    return ESP_OK;
}

esp_err_t kidi_ticket_decode_base64url(const char* encoded, size_t encoded_size,
                                       uint8_t output[KIDI_ACCESSORY_TICKET_BYTES]) {
    if (encoded == NULL || output == NULL || encoded_size != KIDI_ACCESSORY_TICKET_BASE64URL_BYTES) {
        return ESP_ERR_INVALID_ARG;
    }
    uint8_t standard[KIDI_ACCESSORY_TICKET_BASE64URL_BYTES];
    for (size_t index = 0; index < encoded_size; ++index) {
        char value = encoded[index];
        if (value == '-') {
            value = '+';
        } else if (value == '_') {
            value = '/';
        } else if (!((value >= 'A' && value <= 'Z') || (value >= 'a' && value <= 'z') ||
                     (value >= '0' && value <= '9'))) {
            return ESP_ERR_INVALID_ARG;
        }
        standard[index] = (uint8_t)value;
    }
    size_t decoded_size = 0;
    int result = mbedtls_base64_decode(output, KIDI_ACCESSORY_TICKET_BYTES, &decoded_size, standard, sizeof(standard));
    return result == 0 && decoded_size == KIDI_ACCESSORY_TICKET_BYTES ? ESP_OK : ESP_FAIL;
}
