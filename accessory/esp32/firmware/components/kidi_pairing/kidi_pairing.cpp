#include "kidi_pairing.h"

#include <array>
#include <memory>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "accessory_protocol.h"
#include "cJSON.h"
#include "esp_random.h"
#include "kidi_controller_store.h"
#include "kidi_identity.h"
#include "mbedtls/base64.h"
#include "mbedtls/gcm.h"
#include "mbedtls/hkdf.h"
#include "mbedtls/md.h"
#include "mbedtls/platform_util.h"
#include "mbedtls/sha256.h"

#define KIDI_PAIRING_PROFILE_MAX_BYTES 1024U
#define KIDI_PAIRING_CIPHERTEXT_MAX_BYTES (KIDI_PAIRING_PROFILE_MAX_BYTES + 16U)
#define KIDI_PAIRING_IV_BYTES 12U
#define KIDI_PAIRING_NONCE_BYTES 32U
#define KIDI_PAIRING_PROOF_BYTES 32U

static const uint8_t REQUEST_CONTEXT[] = "kidi-pair-request-v1\0";
static const uint8_t RESPONSE_KEY_INFO[] = "kidi-pair-response-v1";

namespace {

struct FreeDeleter {
    void operator()(void* value) const noexcept { free(value); }
};

struct JsonDeleter {
    void operator()(cJSON* value) const noexcept { cJSON_Delete(value); }
};

struct JsonTextDeleter {
    void operator()(char* value) const noexcept { cJSON_free(value); }
};

template <size_t Size>
struct SensitiveBytes {
    std::array<uint8_t, Size> value{};
    ~SensitiveBytes() { mbedtls_platform_zeroize(value.data(), value.size()); }
};

struct SensitiveController {
    kidi_controller_t value{};
    ~SensitiveController() { mbedtls_platform_zeroize(&value, sizeof(value)); }
};

class GcmContext {
public:
    GcmContext() { mbedtls_gcm_init(&value_); }
    ~GcmContext() { mbedtls_gcm_free(&value_); }
    auto get() -> mbedtls_gcm_context* { return &value_; }

    GcmContext(const GcmContext&) = delete;
    auto operator=(const GcmContext&) -> GcmContext& = delete;

private:
    mbedtls_gcm_context value_;
};

using JsonPointer = std::unique_ptr<cJSON, JsonDeleter>;
using JsonTextPointer = std::unique_ptr<char, JsonTextDeleter>;
template <typename Value>
using MallocPointer = std::unique_ptr<Value, FreeDeleter>;

} // namespace

static bool constant_time_equal(const uint8_t* left, const uint8_t* right, size_t size) {
    uint8_t difference = 0;
    for (size_t index = 0; index < size; ++index) {
        difference |= left[index] ^ right[index];
    }
    return difference == 0;
}

static bool object_has_exact_keys(const cJSON* object, const char* const* keys, size_t key_count) {
    size_t found = 0;
    const cJSON* item = NULL;
    cJSON_ArrayForEach(item, object) {
        if (item->string == NULL) {
            return false;
        }
        bool known = false;
        for (size_t index = 0; index < key_count; ++index) {
            if (strcmp(item->string, keys[index]) == 0) {
                known = true;
                break;
            }
        }
        if (!known) {
            return false;
        }
        ++found;
    }
    return found == key_count;
}

static bool valid_name(const char* name) {
    if (name == NULL) {
        return false;
    }
    size_t size = strlen(name);
    if (size == 0 || size > KIDI_CONTROLLER_NAME_BYTES) {
        return false;
    }
    for (size_t index = 0; index < size; ++index) {
        uint8_t value = (uint8_t)name[index];
        if (value < 0x20 || value == 0x7f) {
            return false;
        }
    }
    return true;
}

static esp_err_t decode_hex(const char* value, uint8_t* output, size_t output_size) {
    if (value == NULL || strlen(value) != output_size * 2) {
        return ESP_ERR_INVALID_ARG;
    }
    for (size_t index = 0; index < output_size; ++index) {
        char high = value[index * 2];
        char low = value[index * 2 + 1];
        if (!((high >= '0' && high <= '9') || (high >= 'a' && high <= 'f')) ||
            !((low >= '0' && low <= '9') || (low >= 'a' && low <= 'f'))) {
            return ESP_ERR_INVALID_ARG;
        }
        uint8_t high_value = high <= '9' ? high - '0' : high - 'a' + 10;
        uint8_t low_value = low <= '9' ? low - '0' : low - 'a' + 10;
        output[index] = (high_value << 4U) | low_value;
    }
    return ESP_OK;
}

static esp_err_t decode_base64url(const char* value, uint8_t* output, size_t expected_size) {
    if (value == NULL) {
        return ESP_ERR_INVALID_ARG;
    }
    size_t value_size = strlen(value);
    size_t padded_size = ((value_size + 3U) / 4U) * 4U;
    if (value_size == 0 || padded_size > 2048U) {
        return ESP_ERR_INVALID_SIZE;
    }
    auto* standard = static_cast<uint8_t*>(malloc(padded_size));
    if (standard == NULL) {
        return ESP_ERR_NO_MEM;
    }
    for (size_t index = 0; index < value_size; ++index) {
        char byte = value[index];
        if (byte == '-') {
            byte = '+';
        } else if (byte == '_') {
            byte = '/';
        } else if (!((byte >= 'A' && byte <= 'Z') || (byte >= 'a' && byte <= 'z') || (byte >= '0' && byte <= '9'))) {
            free(standard);
            return ESP_ERR_INVALID_ARG;
        }
        standard[index] = byte;
    }
    for (size_t index = value_size; index < padded_size; ++index) {
        standard[index] = '=';
    }
    size_t decoded_size = 0;
    int result = mbedtls_base64_decode(output, expected_size, &decoded_size, standard, padded_size);
    free(standard);
    return result == 0 && decoded_size == expected_size ? ESP_OK : ESP_ERR_INVALID_ARG;
}

static esp_err_t encode_base64url(const uint8_t* input, size_t input_size, char* output, size_t output_capacity) {
    size_t required = 4U * ((input_size + 2U) / 3U);
    if (input == NULL || output == NULL || output_capacity <= required) {
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

static esp_err_t calculate_request_proof(const uint8_t secret[KIDI_PAIRING_INVITATION_SECRET_BYTES],
                                         const char* device_id, const uint8_t controller_id[KIDI_CONTROLLER_ID_BYTES],
                                         const uint8_t nonce[KIDI_PAIRING_NONCE_BYTES], const char* name,
                                         uint8_t proof[KIDI_PAIRING_PROOF_BYTES]) {
    const mbedtls_md_info_t* sha256 = mbedtls_md_info_from_type(MBEDTLS_MD_SHA256);
    if (sha256 == NULL) {
        return ESP_ERR_NOT_SUPPORTED;
    }
    mbedtls_md_context_t context;
    mbedtls_md_init(&context);
    int result = mbedtls_md_setup(&context, sha256, 1);
    if (result == 0) {
        result = mbedtls_md_hmac_starts(&context, secret, KIDI_PAIRING_INVITATION_SECRET_BYTES);
    }
    if (result == 0) {
        result = mbedtls_md_hmac_update(&context, REQUEST_CONTEXT, sizeof(REQUEST_CONTEXT) - 1);
    }
    if (result == 0) {
        result = mbedtls_md_hmac_update(&context, (const uint8_t*)device_id, strlen(device_id));
    }
    if (result == 0) {
        result = mbedtls_md_hmac_update(&context, controller_id, KIDI_CONTROLLER_ID_BYTES);
    }
    if (result == 0) {
        result = mbedtls_md_hmac_update(&context, nonce, KIDI_PAIRING_NONCE_BYTES);
    }
    if (result == 0) {
        result = mbedtls_md_hmac_update(&context, (const uint8_t*)name, strlen(name));
    }
    if (result == 0) {
        result = mbedtls_md_hmac_finish(&context, proof);
    }
    mbedtls_md_free(&context);
    return result == 0 ? ESP_OK : ESP_FAIL;
}

static esp_err_t sha256(const uint8_t* input, size_t input_size, uint8_t output[32]) {
    mbedtls_sha256_context context;
    mbedtls_sha256_init(&context);
    int result = mbedtls_sha256_starts(&context, 0);
    if (result == 0) {
        result = mbedtls_sha256_update(&context, input, input_size);
    }
    if (result == 0) {
        result = mbedtls_sha256_finish(&context, output);
    }
    mbedtls_sha256_free(&context);
    return result == 0 ? ESP_OK : ESP_FAIL;
}

static char* create_profile(const kidi_pairing_config_t* config, const kidi_controller_t* controller) {
    char control_pin[45];
    char media_pin[45];
    char token[45];
    if (encode_base64url(config->control_certificate_sha256, KIDI_TLS_CERTIFICATE_SHA256_BYTES, control_pin,
                         sizeof(control_pin)) != ESP_OK ||
        encode_base64url(config->media_certificate_sha256, KIDI_TLS_CERTIFICATE_SHA256_BYTES, media_pin,
                         sizeof(media_pin)) != ESP_OK ||
        encode_base64url(controller->token, sizeof(controller->token), token, sizeof(token)) != ESP_OK) {
        return NULL;
    }
    cJSON* profile = cJSON_CreateObject();
    if (profile == NULL) {
        return NULL;
    }
    bool added = cJSON_AddNumberToObject(profile, "v", KIDI_ACCESSORY_PROTOCOL_VERSION) != NULL &&
                 cJSON_AddStringToObject(profile, "device_id", config->device_id_hex) != NULL &&
                 cJSON_AddStringToObject(profile, "name", config->device_name) != NULL &&
                 cJSON_AddStringToObject(profile, "host", config->host) != NULL &&
                 cJSON_AddNumberToObject(profile, "control_port", KIDI_ACCESSORY_CONTROL_PORT) != NULL &&
                 cJSON_AddNumberToObject(profile, "media_port", KIDI_ACCESSORY_MEDIA_PORT) != NULL &&
                 cJSON_AddStringToObject(profile, "control_certificate_sha256", control_pin) != NULL &&
                 cJSON_AddStringToObject(profile, "media_certificate_sha256", media_pin) != NULL &&
                 cJSON_AddStringToObject(profile, "controller_token", token) != NULL &&
                 cJSON_AddStringToObject(profile, "softap_ssid", config->softap_ssid) != NULL &&
                 cJSON_AddStringToObject(profile, "softap_password", config->softap_password) != NULL;
    if (!added) {
        cJSON_Delete(profile);
        return NULL;
    }
    char* text = cJSON_PrintUnformatted(profile);
    cJSON_Delete(profile);
    return text;
}

esp_err_t kidi_pairing_create_response(const uint8_t* request, size_t request_size, const kidi_pairing_config_t* config,
                                       uint8_t* response, size_t response_capacity, size_t* response_size,
                                       bool* created_controller, uint8_t paired_controller_id[16]) {
    static const char* const request_keys[] = {
        "v", "device_id", "controller_id", "name", "nonce", "proof",
    };
    if (request == NULL || request_size == 0 || request_size > KIDI_ACCESSORY_MAX_BLE_MESSAGE_BYTES || config == NULL ||
        response == NULL || response_size == NULL || created_controller == NULL || paired_controller_id == NULL ||
        config->device_id_hex == NULL || strlen(config->device_id_hex) != 32 || config->device_name == NULL ||
        config->host == NULL || config->softap_ssid == NULL || config->softap_password == NULL ||
        config->control_certificate_sha256 == NULL || config->media_certificate_sha256 == NULL ||
        config->invitation_secret == NULL || config->current_epoch == 0 ||
        config->current_epoch > config->invitation_expires_at) {
        return ESP_ERR_INVALID_ARG;
    }
    *response_size = 0;
    *created_controller = false;
    memset(paired_controller_id, 0, KIDI_CONTROLLER_ID_BYTES);
    JsonPointer root{cJSON_ParseWithLength(reinterpret_cast<const char*>(request), request_size)};
    if (root == nullptr || !cJSON_IsObject(root.get()) ||
        !object_has_exact_keys(root.get(), request_keys, std::size(request_keys))) {
        return ESP_ERR_INVALID_ARG;
    }
    const cJSON* version = cJSON_GetObjectItemCaseSensitive(root.get(), "v");
    const cJSON* device_id = cJSON_GetObjectItemCaseSensitive(root.get(), "device_id");
    const cJSON* controller_id_json = cJSON_GetObjectItemCaseSensitive(root.get(), "controller_id");
    const cJSON* name_json = cJSON_GetObjectItemCaseSensitive(root.get(), "name");
    const cJSON* nonce_json = cJSON_GetObjectItemCaseSensitive(root.get(), "nonce");
    const cJSON* proof_json = cJSON_GetObjectItemCaseSensitive(root.get(), "proof");
    if (!cJSON_IsNumber(version) || version->valuedouble != version->valueint ||
        version->valueint != KIDI_ACCESSORY_PROTOCOL_VERSION || !cJSON_IsString(device_id) ||
        strcmp(device_id->valuestring, config->device_id_hex) != 0 || !cJSON_IsString(controller_id_json) ||
        !cJSON_IsString(name_json) || !valid_name(name_json->valuestring) || !cJSON_IsString(nonce_json) ||
        !cJSON_IsString(proof_json)) {
        return ESP_ERR_INVALID_ARG;
    }

    std::array<uint8_t, KIDI_CONTROLLER_ID_BYTES> controller_id{};
    std::array<uint8_t, KIDI_PAIRING_NONCE_BYTES> nonce{};
    SensitiveBytes<KIDI_PAIRING_PROOF_BYTES> received_proof;
    SensitiveBytes<KIDI_PAIRING_PROOF_BYTES> expected_proof;
    esp_err_t error = decode_hex(controller_id_json->valuestring, controller_id.data(), controller_id.size());
    if (error == ESP_OK) {
        error = decode_base64url(nonce_json->valuestring, nonce.data(), nonce.size());
    }
    if (error == ESP_OK) {
        error = decode_base64url(proof_json->valuestring, received_proof.value.data(), received_proof.value.size());
    }
    if (error == ESP_OK) {
        error = calculate_request_proof(config->invitation_secret, config->device_id_hex, controller_id.data(),
                                        nonce.data(), name_json->valuestring, expected_proof.value.data());
    }
    if (error != ESP_OK ||
        !constant_time_equal(received_proof.value.data(), expected_proof.value.data(), received_proof.value.size())) {
        return ESP_ERR_INVALID_CRC;
    }
    memcpy(paired_controller_id, controller_id.data(), controller_id.size());

    SensitiveController controller;
    bool added = false;
    error = kidi_controller_store_find_id(controller_id.data(), &controller.value);
    if (error == ESP_OK) {
        if (strcmp(controller.value.name, name_json->valuestring) != 0) {
            return ESP_ERR_INVALID_STATE;
        }
    } else if (error == ESP_ERR_NOT_FOUND) {
        SensitiveBytes<KIDI_CONTROLLER_TOKEN_BYTES> token;
        esp_fill_random(token.value.data(), token.value.size());
        error = kidi_controller_store_add(controller_id.data(), token.value.data(), name_json->valuestring,
                                          &controller.value);
        if (error != ESP_OK) {
            return error;
        }
        added = true;
    } else {
        return error;
    }

    const auto rollback = [&](esp_err_t cause) {
        if (!added) {
            return cause;
        }
        const esp_err_t rollback_error = kidi_controller_store_remove(controller_id.data());
        return rollback_error == ESP_OK ? cause : rollback_error;
    };

    JsonTextPointer profile{create_profile(config, &controller.value)};
    const size_t profile_size = profile == nullptr ? 0 : strlen(profile.get());
    if (profile == nullptr || profile_size == 0 || profile_size > KIDI_PAIRING_PROFILE_MAX_BYTES) {
        return rollback(ESP_ERR_INVALID_SIZE);
    }
    SensitiveBytes<32> response_key;
    const mbedtls_md_info_t* sha256_info = mbedtls_md_info_from_type(MBEDTLS_MD_SHA256);
    if (sha256_info == NULL ||
        mbedtls_hkdf(sha256_info, nonce.data(), nonce.size(), config->invitation_secret,
                     KIDI_PAIRING_INVITATION_SECRET_BYTES, RESPONSE_KEY_INFO, sizeof(RESPONSE_KEY_INFO) - 1,
                     response_key.value.data(), response_key.value.size()) != 0) {
        return rollback(ESP_FAIL);
    }
    std::array<uint8_t, KIDI_PAIRING_IV_BYTES> iv{};
    esp_fill_random(iv.data(), iv.size());
    std::array<uint8_t, 32 + KIDI_CONTROLLER_ID_BYTES> aad{};
    memcpy(aad.data(), config->device_id_hex, 32);
    memcpy(aad.data() + 32, controller_id.data(), controller_id.size());
    MallocPointer<uint8_t> ciphertext{static_cast<uint8_t*>(malloc(profile_size + 16))};
    if (ciphertext == nullptr) {
        return rollback(ESP_ERR_NO_MEM);
    }
    GcmContext gcm;
    int crypto_result = mbedtls_gcm_setkey(gcm.get(), MBEDTLS_CIPHER_ID_AES, response_key.value.data(), 256);
    if (crypto_result == 0) {
        crypto_result = mbedtls_gcm_crypt_and_tag(
            gcm.get(), MBEDTLS_GCM_ENCRYPT, profile_size, iv.data(), iv.size(), aad.data(), aad.size(),
            reinterpret_cast<const uint8_t*>(profile.get()), ciphertext.get(), 16, ciphertext.get() + profile_size);
    }
    if (crypto_result != 0) {
        return rollback(ESP_FAIL);
    }

    const size_t ciphertext_size = profile_size + 16;
    std::array<uint8_t, 32> request_hash{};
    error = sha256(request, request_size, request_hash.data());
    if (error != ESP_OK) {
        return rollback(error);
    }
    const size_t signed_size = request_hash.size() + iv.size() + ciphertext_size;
    MallocPointer<uint8_t> signed_message{static_cast<uint8_t*>(malloc(signed_size))};
    if (signed_message == nullptr) {
        return rollback(ESP_ERR_NO_MEM);
    }
    memcpy(signed_message.get(), request_hash.data(), request_hash.size());
    memcpy(signed_message.get() + request_hash.size(), iv.data(), iv.size());
    memcpy(signed_message.get() + request_hash.size() + iv.size(), ciphertext.get(), ciphertext_size);
    std::array<uint8_t, KIDI_IDENTITY_MAX_SIGNATURE_BYTES> signature{};
    size_t signature_size = 0;
    error = kidi_identity_sign(signed_message.get(), signed_size, signature.data(), &signature_size);
    if (error != ESP_OK) {
        return rollback(error);
    }

    std::array<char, 17> iv_encoded{};
    const size_t ciphertext_encoded_capacity = 4U * ((ciphertext_size + 2U) / 3U) + 1U;
    MallocPointer<char> ciphertext_encoded{static_cast<char*>(malloc(ciphertext_encoded_capacity))};
    std::array<char, 4U * ((KIDI_IDENTITY_MAX_SIGNATURE_BYTES + 2U) / 3U) + 1U> signature_encoded{};
    if (ciphertext_encoded == nullptr ||
        encode_base64url(iv.data(), iv.size(), iv_encoded.data(), iv_encoded.size()) != ESP_OK ||
        encode_base64url(ciphertext.get(), ciphertext_size, ciphertext_encoded.get(), ciphertext_encoded_capacity) !=
            ESP_OK ||
        encode_base64url(signature.data(), signature_size, signature_encoded.data(), signature_encoded.size()) !=
            ESP_OK) {
        return rollback(ESP_FAIL);
    }
    JsonPointer envelope{cJSON_CreateObject()};
    const bool envelope_added =
        envelope != nullptr &&
        cJSON_AddNumberToObject(envelope.get(), "v", KIDI_ACCESSORY_PROTOCOL_VERSION) != nullptr &&
        cJSON_AddStringToObject(envelope.get(), "iv", iv_encoded.data()) != nullptr &&
        cJSON_AddStringToObject(envelope.get(), "ciphertext", ciphertext_encoded.get()) != nullptr &&
        cJSON_AddStringToObject(envelope.get(), "signature", signature_encoded.data()) != nullptr;
    if (!envelope_added) {
        return rollback(ESP_ERR_NO_MEM);
    }
    JsonTextPointer envelope_text{cJSON_PrintUnformatted(envelope.get())};
    const size_t envelope_size = envelope_text == nullptr ? 0 : strlen(envelope_text.get());
    if (envelope_text == nullptr || envelope_size == 0 || envelope_size > response_capacity ||
        envelope_size > KIDI_ACCESSORY_MAX_BLE_MESSAGE_BYTES) {
        return rollback(ESP_ERR_INVALID_SIZE);
    }
    memcpy(response, envelope_text.get(), envelope_size);
    *response_size = envelope_size;
    *created_controller = added;
    return ESP_OK;
}
