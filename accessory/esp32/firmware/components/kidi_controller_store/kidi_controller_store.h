#pragma once

#include <cstddef>
#include <cstdint>

#include "accessory_protocol.h"
#include "esp_err.h"

#define KIDI_CONTROLLER_ID_BYTES 16U
#define KIDI_CONTROLLER_TOKEN_BYTES 32U
#define KIDI_CONTROLLER_NAME_BYTES 64U

typedef enum {
    KIDI_CONTROLLER_ROLE_OWNER = 1,
    KIDI_CONTROLLER_ROLE_MEMBER = 2,
} kidi_controller_role_t;

typedef struct {
    uint8_t id[KIDI_CONTROLLER_ID_BYTES];
    uint8_t token[KIDI_CONTROLLER_TOKEN_BYTES];
    char name[KIDI_CONTROLLER_NAME_BYTES + 1];
    kidi_controller_role_t role;
} kidi_controller_t;

esp_err_t kidi_controller_store_init(const char* nvs_namespace);
size_t kidi_controller_store_count(void);

esp_err_t kidi_controller_store_find_id(const uint8_t id[KIDI_CONTROLLER_ID_BYTES], kidi_controller_t* controller);

esp_err_t kidi_controller_store_find_token(const uint8_t token[KIDI_CONTROLLER_TOKEN_BYTES],
                                           kidi_controller_t* controller);

esp_err_t kidi_controller_store_add(const uint8_t id[KIDI_CONTROLLER_ID_BYTES],
                                    const uint8_t token[KIDI_CONTROLLER_TOKEN_BYTES], const char* name,
                                    kidi_controller_t* controller);

esp_err_t kidi_controller_store_remove(const uint8_t id[KIDI_CONTROLLER_ID_BYTES]);
