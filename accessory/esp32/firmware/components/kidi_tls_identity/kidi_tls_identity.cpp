#include "kidi_tls_identity.h"

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "esp_random.h"
#include "mbedtls/pk.h"
#include "mbedtls/platform_util.h"
#include "mbedtls/sha256.h"
#include "mbedtls/x509_crt.h"
#include "nvs.h"

#define KIDI_TLS_PRIVATE_KEY_CAPACITY 1024U
#define KIDI_TLS_CERTIFICATE_CAPACITY 2048U

static int random_bytes(void* context, unsigned char* output, size_t size) {
    (void)context;
    esp_fill_random(output, size);
    return 0;
}

static esp_err_t names(const char* prefix, char certificate[16], char private_key[16]) {
    if (prefix == NULL || prefix[0] == '\0') {
        return ESP_ERR_INVALID_ARG;
    }
    int certificate_size = snprintf(certificate, 16, "%s_cert", prefix);
    int private_key_size = snprintf(private_key, 16, "%s_key", prefix);
    return certificate_size > 0 && certificate_size < 16 && private_key_size > 0 && private_key_size < 16
               ? ESP_OK
               : ESP_ERR_INVALID_SIZE;
}

static esp_err_t load_blob(nvs_handle_t nvs, const char* key, size_t maximum, uint8_t** output, size_t* output_size) {
    size_t size = 0;
    esp_err_t error = nvs_get_blob(nvs, key, NULL, &size);
    if (error != ESP_OK) {
        return error;
    }
    if (size < 2 || size > maximum) {
        return ESP_ERR_INVALID_SIZE;
    }
    auto* value = static_cast<uint8_t*>(malloc(size));
    if (value == NULL) {
        return ESP_ERR_NO_MEM;
    }
    error = nvs_get_blob(nvs, key, value, &size);
    if (error != ESP_OK || value[size - 1] != '\0') {
        mbedtls_platform_zeroize(value, size);
        free(value);
        return error == ESP_OK ? ESP_ERR_INVALID_RESPONSE : error;
    }
    *output = value;
    *output_size = size;
    return ESP_OK;
}

static esp_err_t generate_identity(nvs_handle_t nvs, const char* certificate_key, const char* private_key_key,
                                   const char* common_name, uint8_t** certificate_pem, size_t* certificate_size,
                                   uint8_t** private_key_pem, size_t* private_key_size) {
    if (common_name == NULL || common_name[0] == '\0' || strlen(common_name) > 64) {
        return ESP_ERR_INVALID_ARG;
    }
    auto* certificate = static_cast<uint8_t*>(calloc(1, KIDI_TLS_CERTIFICATE_CAPACITY));
    auto* key_pem = static_cast<uint8_t*>(calloc(1, KIDI_TLS_PRIVATE_KEY_CAPACITY));
    if (certificate == NULL || key_pem == NULL) {
        free(certificate);
        free(key_pem);
        return ESP_ERR_NO_MEM;
    }

    mbedtls_pk_context key;
    mbedtls_pk_init(&key);
    mbedtls_x509write_cert writer;
    mbedtls_x509write_crt_init(&writer);
    int result = mbedtls_pk_setup(&key, mbedtls_pk_info_from_type(MBEDTLS_PK_ECKEY));
    if (result == 0) {
        result = mbedtls_ecp_gen_key(MBEDTLS_ECP_DP_SECP256R1, mbedtls_pk_ec(key), random_bytes, NULL);
    }
    if (result == 0) {
        result = mbedtls_pk_write_key_pem(&key, key_pem, KIDI_TLS_PRIVATE_KEY_CAPACITY);
    }

    char distinguished_name[96];
    int name_size = snprintf(distinguished_name, sizeof(distinguished_name), "CN=%s,O=Kidi", common_name);
    uint8_t serial[16];
    esp_fill_random(serial, sizeof(serial));
    serial[0] &= 0x7f;
    serial[0] |= 1;
    if (result == 0 && (name_size <= 0 || name_size >= sizeof(distinguished_name))) {
        result = MBEDTLS_ERR_X509_BAD_INPUT_DATA;
    }
    if (result == 0) {
        mbedtls_x509write_crt_set_version(&writer, MBEDTLS_X509_CRT_VERSION_3);
        mbedtls_x509write_crt_set_md_alg(&writer, MBEDTLS_MD_SHA256);
        mbedtls_x509write_crt_set_subject_key(&writer, &key);
        mbedtls_x509write_crt_set_issuer_key(&writer, &key);
        result = mbedtls_x509write_crt_set_subject_name(&writer, distinguished_name);
    }
    if (result == 0) {
        result = mbedtls_x509write_crt_set_issuer_name(&writer, distinguished_name);
    }
    if (result == 0) {
        result = mbedtls_x509write_crt_set_serial_raw(&writer, serial, sizeof(serial));
    }
    if (result == 0) {
        result = mbedtls_x509write_crt_set_validity(&writer, "20240101000000", "20491231235959");
    }
    if (result == 0) {
        result = mbedtls_x509write_crt_set_basic_constraints(&writer, 0, -1);
    }
    if (result == 0) {
        result = mbedtls_x509write_crt_set_key_usage(&writer,
                                                     MBEDTLS_X509_KU_DIGITAL_SIGNATURE | MBEDTLS_X509_KU_KEY_AGREEMENT);
    }
    if (result == 0) {
        result = mbedtls_x509write_crt_pem(&writer, certificate, KIDI_TLS_CERTIFICATE_CAPACITY, random_bytes, NULL);
    }
    mbedtls_x509write_crt_free(&writer);
    mbedtls_pk_free(&key);
    if (result != 0) {
        mbedtls_platform_zeroize(key_pem, KIDI_TLS_PRIVATE_KEY_CAPACITY);
        free(key_pem);
        free(certificate);
        return ESP_FAIL;
    }

    *certificate_size = strlen((char*)certificate) + 1;
    *private_key_size = strlen((char*)key_pem) + 1;
    esp_err_t error = nvs_set_blob(nvs, certificate_key, certificate, *certificate_size);
    if (error == ESP_OK) {
        error = nvs_set_blob(nvs, private_key_key, key_pem, *private_key_size);
    }
    if (error == ESP_OK) {
        error = nvs_commit(nvs);
    }
    if (error != ESP_OK) {
        mbedtls_platform_zeroize(key_pem, KIDI_TLS_PRIVATE_KEY_CAPACITY);
        free(key_pem);
        free(certificate);
        return error;
    }
    *certificate_pem = certificate;
    *private_key_pem = key_pem;
    return ESP_OK;
}

static esp_err_t validate_identity(kidi_tls_identity_t* identity) {
    mbedtls_x509_crt certificate;
    mbedtls_x509_crt_init(&certificate);
    mbedtls_pk_context private_key;
    mbedtls_pk_init(&private_key);
    int result = mbedtls_x509_crt_parse(&certificate, identity->certificate_pem, identity->certificate_pem_size);
    if (result == 0) {
        result = mbedtls_pk_parse_key(&private_key, identity->private_key_pem, identity->private_key_pem_size, NULL, 0,
                                      random_bytes, NULL);
    }
    if (result == 0) {
        result = mbedtls_pk_check_pair(&certificate.pk, &private_key, random_bytes, NULL);
    }
    if (result == 0) {
        mbedtls_sha256_context sha;
        mbedtls_sha256_init(&sha);
        result = mbedtls_sha256_starts(&sha, 0);
        if (result == 0) {
            result = mbedtls_sha256_update(&sha, certificate.raw.p, certificate.raw.len);
        }
        if (result == 0) {
            result = mbedtls_sha256_finish(&sha, identity->certificate_sha256);
        }
        mbedtls_sha256_free(&sha);
    }
    mbedtls_pk_free(&private_key);
    mbedtls_x509_crt_free(&certificate);
    return result == 0 ? ESP_OK : ESP_ERR_INVALID_STATE;
}

esp_err_t kidi_tls_identity_init(const char* nvs_namespace, const char* nvs_prefix, const char* common_name,
                                 kidi_tls_identity_t* identity) {
    if (nvs_namespace == NULL || identity == NULL) {
        return ESP_ERR_INVALID_ARG;
    }
    memset(identity, 0, sizeof(*identity));
    char certificate_key[16];
    char private_key_key[16];
    esp_err_t error = names(nvs_prefix, certificate_key, private_key_key);
    if (error != ESP_OK) {
        return error;
    }
    nvs_handle_t nvs;
    error = nvs_open(nvs_namespace, NVS_READWRITE, &nvs);
    if (error != ESP_OK) {
        return error;
    }

    uint8_t* certificate = NULL;
    size_t certificate_size = 0;
    uint8_t* private_key = NULL;
    size_t private_key_size = 0;
    esp_err_t certificate_error =
        load_blob(nvs, certificate_key, KIDI_TLS_CERTIFICATE_CAPACITY, &certificate, &certificate_size);
    esp_err_t private_key_error =
        load_blob(nvs, private_key_key, KIDI_TLS_PRIVATE_KEY_CAPACITY, &private_key, &private_key_size);
    if (certificate_error == ESP_ERR_NVS_NOT_FOUND && private_key_error == ESP_ERR_NVS_NOT_FOUND) {
        error = generate_identity(nvs, certificate_key, private_key_key, common_name, &certificate, &certificate_size,
                                  &private_key, &private_key_size);
    } else if (certificate_error != ESP_OK || private_key_error != ESP_OK) {
        error = certificate_error != ESP_OK ? certificate_error : private_key_error;
    }
    nvs_close(nvs);
    if (error != ESP_OK) {
        if (private_key != NULL) {
            mbedtls_platform_zeroize(private_key, private_key_size);
        }
        free(private_key);
        free(certificate);
        return error;
    }

    identity->certificate_pem = certificate;
    identity->certificate_pem_size = certificate_size;
    identity->private_key_pem = private_key;
    identity->private_key_pem_size = private_key_size;
    error = validate_identity(identity);
    if (error != ESP_OK) {
        kidi_tls_identity_deinit(identity);
    }
    return error;
}

void kidi_tls_identity_deinit(kidi_tls_identity_t* identity) {
    if (identity == NULL) {
        return;
    }
    if (identity->private_key_pem != NULL) {
        mbedtls_platform_zeroize((void*)identity->private_key_pem, identity->private_key_pem_size);
    }
    free((void*)identity->private_key_pem);
    free((void*)identity->certificate_pem);
    memset(identity, 0, sizeof(*identity));
}
