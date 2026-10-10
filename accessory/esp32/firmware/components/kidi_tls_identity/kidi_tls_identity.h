#pragma once

#include <cstddef>
#include <cstdint>

#include "esp_err.h"

#define KIDI_TLS_CERTIFICATE_SHA256_BYTES 32U

typedef struct {
    const uint8_t* certificate_pem;
    size_t certificate_pem_size;
    const uint8_t* private_key_pem;
    size_t private_key_pem_size;
    uint8_t certificate_sha256[KIDI_TLS_CERTIFICATE_SHA256_BYTES];
} kidi_tls_identity_t;

esp_err_t kidi_tls_identity_init(const char* nvs_namespace, const char* nvs_prefix, const char* common_name,
                                 kidi_tls_identity_t* identity);

void kidi_tls_identity_deinit(kidi_tls_identity_t* identity);
