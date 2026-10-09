#include "kidi_invitation.h"

#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "accessory_protocol.h"
#include "cJSON.h"
#include "esp_random.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "kidi_identity.h"
#include "mbedtls/base64.h"
#include "mbedtls/platform_util.h"

static SemaphoreHandle_t invitation_mutex;
static uint8_t invitation_secret[KIDI_PAIRING_INVITATION_SECRET_BYTES];
static uint64_t invitation_expires_at;
static bool invitation_active;

static esp_err_t encode_base64url(const uint8_t* input, size_t input_size, char* output, size_t output_capacity,
                                  size_t* output_size) {
    size_t required = 4U * ((input_size + 2U) / 3U);
    if (input == NULL || output == NULL || output_size == NULL || output_capacity < required) {
        return ESP_ERR_INVALID_ARG;
    }
    size_t encoded_size = 0;
    int result = mbedtls_base64_encode((uint8_t*)output, output_capacity, &encoded_size, input, input_size);
    if (result != 0) {
        return ESP_FAIL;
    }
    while (encoded_size > 0 && output[encoded_size - 1] == '=') {
        --encoded_size;
    }
    if (encoded_size >= output_capacity) {
        return ESP_ERR_INVALID_SIZE;
    }
    for (size_t index = 0; index < encoded_size; ++index) {
        if (output[index] == '+') {
            output[index] = '-';
        } else if (output[index] == '/') {
            output[index] = '_';
        }
    }
    output[encoded_size] = '\0';
    *output_size = encoded_size;
    return ESP_OK;
}

esp_err_t kidi_invitation_init(void) {
    if (invitation_mutex != NULL) {
        return ESP_OK;
    }
    invitation_mutex = xSemaphoreCreateMutex();
    return invitation_mutex == NULL ? ESP_ERR_NO_MEM : ESP_OK;
}

esp_err_t kidi_invitation_create(uint64_t now, uint32_t lifetime_seconds, const char* device_id_hex,
                                 const char* device_name, char output[KIDI_INVITATION_URI_MAX_BYTES],
                                 size_t* output_size) {
    if (invitation_mutex == NULL || now < 1700000000ULL || lifetime_seconds == 0 ||
        lifetime_seconds > KIDI_INVITATION_MAX_LIFETIME_SECONDS || now > UINT64_MAX - lifetime_seconds ||
        device_id_hex == NULL || strlen(device_id_hex) != 32 || device_name == NULL || device_name[0] == '\0' ||
        strlen(device_name) > 64 || output == NULL || output_size == NULL) {
        return ESP_ERR_INVALID_ARG;
    }
    uint8_t public_key[KIDI_IDENTITY_PUBLIC_KEY_BYTES];
    esp_err_t error = kidi_identity_get_public_key(public_key);
    if (error != ESP_OK) {
        return error;
    }
    uint8_t secret[KIDI_PAIRING_INVITATION_SECRET_BYTES];
    esp_fill_random(secret, sizeof(secret));
    char public_key_encoded[89];
    char secret_encoded[44];
    size_t ignored = 0;
    error = encode_base64url(public_key, sizeof(public_key), public_key_encoded, sizeof(public_key_encoded), &ignored);
    if (error == ESP_OK) {
        error = encode_base64url(secret, sizeof(secret), secret_encoded, sizeof(secret_encoded), &ignored);
    }
    if (error != ESP_OK) {
        mbedtls_platform_zeroize(secret, sizeof(secret));
        return error;
    }

    cJSON* payload = cJSON_CreateObject();
    bool added = payload != NULL && cJSON_AddNumberToObject(payload, "v", KIDI_ACCESSORY_PROTOCOL_VERSION) != NULL &&
                 cJSON_AddStringToObject(payload, "device_id", device_id_hex) != NULL &&
                 cJSON_AddStringToObject(payload, "device_public_key", public_key_encoded) != NULL &&
                 cJSON_AddStringToObject(payload, "invitation", secret_encoded) != NULL &&
                 cJSON_AddNumberToObject(payload, "expires_at", (double)(now + lifetime_seconds)) != NULL &&
                 cJSON_AddStringToObject(payload, "ble_service", KIDI_ACCESSORY_BLE_SERVICE_UUID) != NULL &&
                 cJSON_AddStringToObject(payload, "name", device_name) != NULL;
    if (!added) {
        cJSON_Delete(payload);
        mbedtls_platform_zeroize(secret, sizeof(secret));
        return ESP_ERR_NO_MEM;
    }
    char* json = cJSON_PrintUnformatted(payload);
    cJSON_Delete(payload);
    if (json == NULL) {
        mbedtls_platform_zeroize(secret, sizeof(secret));
        return ESP_ERR_NO_MEM;
    }
    size_t prefix_size = strlen("kidi://pair/v1#");
    memcpy(output, "kidi://pair/v1#", prefix_size);
    size_t encoded_size = 0;
    error = encode_base64url((const uint8_t*)json, strlen(json), output + prefix_size,
                             KIDI_INVITATION_URI_MAX_BYTES - prefix_size, &encoded_size);
    cJSON_free(json);
    if (error != ESP_OK) {
        mbedtls_platform_zeroize(secret, sizeof(secret));
        return error;
    }

    xSemaphoreTake(invitation_mutex, portMAX_DELAY);
    memcpy(invitation_secret, secret, sizeof(invitation_secret));
    invitation_expires_at = now + lifetime_seconds;
    invitation_active = true;
    xSemaphoreGive(invitation_mutex);
    mbedtls_platform_zeroize(secret, sizeof(secret));
    *output_size = prefix_size + encoded_size;
    return ESP_OK;
}

esp_err_t kidi_invitation_snapshot(uint64_t now, uint8_t secret[KIDI_PAIRING_INVITATION_SECRET_BYTES],
                                   uint64_t* expires_at) {
    if (invitation_mutex == NULL || secret == NULL || expires_at == NULL) {
        return ESP_ERR_INVALID_ARG;
    }
    xSemaphoreTake(invitation_mutex, portMAX_DELAY);
    esp_err_t error = ESP_OK;
    if (!invitation_active || now == 0 || now > invitation_expires_at) {
        error = ESP_ERR_INVALID_STATE;
    } else {
        memcpy(secret, invitation_secret, sizeof(invitation_secret));
        *expires_at = invitation_expires_at;
    }
    xSemaphoreGive(invitation_mutex);
    return error;
}

esp_err_t kidi_invitation_consume(void) {
    if (invitation_mutex == NULL) {
        return ESP_ERR_INVALID_STATE;
    }
    xSemaphoreTake(invitation_mutex, portMAX_DELAY);
    mbedtls_platform_zeroize(invitation_secret, sizeof(invitation_secret));
    invitation_expires_at = 0;
    invitation_active = false;
    xSemaphoreGive(invitation_mutex);
    return ESP_OK;
}
