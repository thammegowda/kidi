#include "kidi_identity.h"

#include <stdbool.h>
#include <string.h>

#include "esp_random.h"
#include "mbedtls/ecdsa.h"
#include "mbedtls/ecp.h"
#include "mbedtls/platform_util.h"
#include "mbedtls/sha256.h"
#include "nvs.h"

static uint8_t private_key[KIDI_IDENTITY_PRIVATE_KEY_BYTES];
static uint8_t public_key[KIDI_IDENTITY_PUBLIC_KEY_BYTES];
static uint8_t device_id[KIDI_IDENTITY_DEVICE_ID_BYTES];
static bool initialized;

static int random_bytes(void* context, unsigned char* output, size_t size) {
    (void)context;
    esp_fill_random(output, size);
    return 0;
}

static int populate_keypair(mbedtls_ecp_keypair* keypair, const uint8_t raw_private_key[32]) {
    int result = mbedtls_ecp_group_load(&keypair->MBEDTLS_PRIVATE(grp), MBEDTLS_ECP_DP_SECP256R1);
    if (result == 0) {
        result =
            mbedtls_mpi_read_binary(&keypair->MBEDTLS_PRIVATE(d), raw_private_key, KIDI_IDENTITY_PRIVATE_KEY_BYTES);
    }
    if (result == 0) {
        result = mbedtls_ecp_check_privkey(&keypair->MBEDTLS_PRIVATE(grp), &keypair->MBEDTLS_PRIVATE(d));
    }
    if (result == 0) {
        result = mbedtls_ecp_mul(&keypair->MBEDTLS_PRIVATE(grp), &keypair->MBEDTLS_PRIVATE(Q),
                                 &keypair->MBEDTLS_PRIVATE(d), &keypair->MBEDTLS_PRIVATE(grp).G, random_bytes, NULL);
    }
    if (result == 0) {
        result = mbedtls_ecp_check_pubkey(&keypair->MBEDTLS_PRIVATE(grp), &keypair->MBEDTLS_PRIVATE(Q));
    }
    return result;
}

static esp_err_t derive_public_values(void) {
    mbedtls_ecp_keypair keypair;
    mbedtls_ecp_keypair_init(&keypair);
    int result = populate_keypair(&keypair, private_key);
    size_t public_size = 0;
    if (result == 0) {
        result =
            mbedtls_ecp_point_write_binary(&keypair.MBEDTLS_PRIVATE(grp), &keypair.MBEDTLS_PRIVATE(Q),
                                           MBEDTLS_ECP_PF_UNCOMPRESSED, &public_size, public_key, sizeof(public_key));
    }
    mbedtls_ecp_keypair_free(&keypair);
    if (result != 0 || public_size != sizeof(public_key)) {
        return ESP_ERR_INVALID_STATE;
    }

    uint8_t digest[32];
    mbedtls_sha256_context sha;
    mbedtls_sha256_init(&sha);
    result = mbedtls_sha256_starts(&sha, 0);
    if (result == 0) {
        result = mbedtls_sha256_update(&sha, public_key, sizeof(public_key));
    }
    if (result == 0) {
        result = mbedtls_sha256_finish(&sha, digest);
    }
    mbedtls_sha256_free(&sha);
    if (result != 0) {
        return ESP_FAIL;
    }
    memcpy(device_id, digest, sizeof(device_id));
    mbedtls_platform_zeroize(digest, sizeof(digest));
    return ESP_OK;
}

esp_err_t kidi_identity_init(const char* nvs_namespace, const char* nvs_key) {
    if (initialized) {
        return ESP_OK;
    }
    if (nvs_namespace == NULL || nvs_key == NULL || nvs_namespace[0] == '\0' || nvs_key[0] == '\0') {
        return ESP_ERR_INVALID_ARG;
    }

    nvs_handle_t nvs;
    esp_err_t error = nvs_open(nvs_namespace, NVS_READWRITE, &nvs);
    if (error != ESP_OK) {
        return error;
    }
    size_t size = sizeof(private_key);
    error = nvs_get_blob(nvs, nvs_key, private_key, &size);
    if (error == ESP_ERR_NVS_NOT_FOUND) {
        mbedtls_ecp_keypair keypair;
        mbedtls_ecp_keypair_init(&keypair);
        int result = mbedtls_ecp_gen_key(MBEDTLS_ECP_DP_SECP256R1, &keypair, random_bytes, NULL);
        if (result == 0) {
            result = mbedtls_mpi_write_binary(&keypair.MBEDTLS_PRIVATE(d), private_key, sizeof(private_key));
        }
        mbedtls_ecp_keypair_free(&keypair);
        if (result != 0) {
            error = ESP_FAIL;
        } else {
            error = nvs_set_blob(nvs, nvs_key, private_key, sizeof(private_key));
        }
        if (error == ESP_OK) {
            error = nvs_commit(nvs);
        }
    } else if (error == ESP_OK && size != sizeof(private_key)) {
        error = ESP_ERR_INVALID_SIZE;
    }
    nvs_close(nvs);
    if (error != ESP_OK) {
        mbedtls_platform_zeroize(private_key, sizeof(private_key));
        return error;
    }

    error = derive_public_values();
    if (error != ESP_OK) {
        mbedtls_platform_zeroize(private_key, sizeof(private_key));
        mbedtls_platform_zeroize(public_key, sizeof(public_key));
        mbedtls_platform_zeroize(device_id, sizeof(device_id));
        return error;
    }
    initialized = true;
    return ESP_OK;
}

esp_err_t kidi_identity_get_public_key(uint8_t output[KIDI_IDENTITY_PUBLIC_KEY_BYTES]) {
    if (!initialized || output == NULL) {
        return ESP_ERR_INVALID_STATE;
    }
    memcpy(output, public_key, sizeof(public_key));
    return ESP_OK;
}

esp_err_t kidi_identity_get_device_id(uint8_t output[KIDI_IDENTITY_DEVICE_ID_BYTES]) {
    if (!initialized || output == NULL) {
        return ESP_ERR_INVALID_STATE;
    }
    memcpy(output, device_id, sizeof(device_id));
    return ESP_OK;
}

esp_err_t kidi_identity_sign(const uint8_t* message, size_t message_size,
                             uint8_t signature[KIDI_IDENTITY_MAX_SIGNATURE_BYTES], size_t* signature_size) {
    if (!initialized || message == NULL || message_size == 0 || signature == NULL || signature_size == NULL) {
        return ESP_ERR_INVALID_ARG;
    }
    uint8_t digest[32];
    mbedtls_sha256_context sha;
    mbedtls_sha256_init(&sha);
    int result = mbedtls_sha256_starts(&sha, 0);
    if (result == 0) {
        result = mbedtls_sha256_update(&sha, message, message_size);
    }
    if (result == 0) {
        result = mbedtls_sha256_finish(&sha, digest);
    }
    mbedtls_sha256_free(&sha);

    mbedtls_ecdsa_context ecdsa;
    mbedtls_ecdsa_init(&ecdsa);
    if (result == 0) {
        result = populate_keypair(&ecdsa, private_key);
    }
    if (result == 0) {
        result = mbedtls_ecdsa_write_signature(&ecdsa, MBEDTLS_MD_SHA256, digest, sizeof(digest), signature,
                                               KIDI_IDENTITY_MAX_SIGNATURE_BYTES, signature_size, random_bytes, NULL);
    }
    mbedtls_ecdsa_free(&ecdsa);
    mbedtls_platform_zeroize(digest, sizeof(digest));
    return result == 0 ? ESP_OK : ESP_FAIL;
}
