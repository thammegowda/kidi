#include "kidi_controller_store.h"

#include <stdbool.h>
#include <stdio.h>
#include <string.h>

#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "nvs.h"

#define KIDI_CONTROLLER_RECORD_VERSION 1U
#define KIDI_CONTROLLER_RECORD_BYTES 115U

static kidi_controller_t controllers[KIDI_ACCESSORY_MAX_CONTROLLERS];
static bool occupied[KIDI_ACCESSORY_MAX_CONTROLLERS];
static char store_namespace[16];
static SemaphoreHandle_t store_mutex;

static bool constant_time_equal(const uint8_t* left, const uint8_t* right, size_t size) {
    uint8_t difference = 0;
    for (size_t index = 0; index < size; ++index) {
        difference |= left[index] ^ right[index];
    }
    return difference == 0;
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

static void slot_key(size_t slot, char output[8]) { snprintf(output, 8, "ctrl%u", (unsigned)slot); }

static esp_err_t decode_record(const uint8_t record[KIDI_CONTROLLER_RECORD_BYTES], kidi_controller_t* controller) {
    uint8_t name_size = record[50];
    if (record[0] != KIDI_CONTROLLER_RECORD_VERSION ||
        (record[1] != KIDI_CONTROLLER_ROLE_OWNER && record[1] != KIDI_CONTROLLER_ROLE_MEMBER) || name_size == 0 ||
        name_size > KIDI_CONTROLLER_NAME_BYTES) {
        return ESP_ERR_INVALID_RESPONSE;
    }
    memset(controller, 0, sizeof(*controller));
    controller->role = (kidi_controller_role_t)record[1];
    memcpy(controller->id, record + 2, sizeof(controller->id));
    memcpy(controller->token, record + 18, sizeof(controller->token));
    memcpy(controller->name, record + 51, name_size);
    return valid_name(controller->name) ? ESP_OK : ESP_ERR_INVALID_RESPONSE;
}

static void encode_record(const kidi_controller_t* controller, uint8_t record[KIDI_CONTROLLER_RECORD_BYTES]) {
    memset(record, 0, KIDI_CONTROLLER_RECORD_BYTES);
    record[0] = KIDI_CONTROLLER_RECORD_VERSION;
    record[1] = controller->role;
    memcpy(record + 2, controller->id, sizeof(controller->id));
    memcpy(record + 18, controller->token, sizeof(controller->token));
    size_t name_size = strlen(controller->name);
    record[50] = name_size;
    memcpy(record + 51, controller->name, name_size);
}

static esp_err_t save_slot(size_t slot) {
    uint8_t record[KIDI_CONTROLLER_RECORD_BYTES];
    encode_record(&controllers[slot], record);
    char key[8];
    slot_key(slot, key);
    nvs_handle_t nvs;
    esp_err_t error = nvs_open(store_namespace, NVS_READWRITE, &nvs);
    if (error == ESP_OK) {
        error = nvs_set_blob(nvs, key, record, sizeof(record));
        if (error == ESP_OK) {
            error = nvs_commit(nvs);
        }
        nvs_close(nvs);
    }
    return error;
}

esp_err_t kidi_controller_store_init(const char* nvs_namespace) {
    if (store_mutex != NULL) {
        return ESP_OK;
    }
    if (nvs_namespace == NULL || nvs_namespace[0] == '\0' || strlen(nvs_namespace) >= sizeof(store_namespace)) {
        return ESP_ERR_INVALID_ARG;
    }
    memcpy(store_namespace, nvs_namespace, strlen(nvs_namespace) + 1);
    store_mutex = xSemaphoreCreateMutex();
    if (store_mutex == NULL) {
        return ESP_ERR_NO_MEM;
    }

    nvs_handle_t nvs;
    esp_err_t error = nvs_open(store_namespace, NVS_READONLY, &nvs);
    if (error == ESP_ERR_NVS_NOT_FOUND) {
        return ESP_OK;
    }
    if (error != ESP_OK) {
        vSemaphoreDelete(store_mutex);
        store_mutex = NULL;
        return error;
    }
    for (size_t slot = 0; slot < KIDI_ACCESSORY_MAX_CONTROLLERS; ++slot) {
        char key[8];
        slot_key(slot, key);
        uint8_t record[KIDI_CONTROLLER_RECORD_BYTES];
        size_t size = sizeof(record);
        error = nvs_get_blob(nvs, key, record, &size);
        if (error == ESP_ERR_NVS_NOT_FOUND) {
            error = ESP_OK;
            continue;
        }
        if (error != ESP_OK || size != sizeof(record)) {
            break;
        }
        error = decode_record(record, &controllers[slot]);
        if (error != ESP_OK) {
            break;
        }
        occupied[slot] = true;
    }
    nvs_close(nvs);
    if (error == ESP_OK) {
        size_t controller_count = 0;
        size_t owner_count = 0;
        for (size_t slot = 0; slot < KIDI_ACCESSORY_MAX_CONTROLLERS; ++slot) {
            if (occupied[slot]) {
                ++controller_count;
                owner_count += controllers[slot].role == KIDI_CONTROLLER_ROLE_OWNER ? 1 : 0;
            }
        }
        if (controller_count > 0 && owner_count != 1) {
            error = ESP_ERR_INVALID_STATE;
        }
    }
    if (error != ESP_OK) {
        memset(controllers, 0, sizeof(controllers));
        memset(occupied, 0, sizeof(occupied));
        vSemaphoreDelete(store_mutex);
        store_mutex = NULL;
    }
    return error;
}

size_t kidi_controller_store_count(void) {
    if (store_mutex == NULL) {
        return 0;
    }
    xSemaphoreTake(store_mutex, portMAX_DELAY);
    size_t count = 0;
    for (size_t slot = 0; slot < KIDI_ACCESSORY_MAX_CONTROLLERS; ++slot) {
        count += occupied[slot] ? 1 : 0;
    }
    xSemaphoreGive(store_mutex);
    return count;
}

esp_err_t kidi_controller_store_find_id(const uint8_t id[KIDI_CONTROLLER_ID_BYTES], kidi_controller_t* controller) {
    if (store_mutex == NULL || id == NULL || controller == NULL) {
        return ESP_ERR_INVALID_ARG;
    }
    xSemaphoreTake(store_mutex, portMAX_DELAY);
    esp_err_t error = ESP_ERR_NOT_FOUND;
    for (size_t slot = 0; slot < KIDI_ACCESSORY_MAX_CONTROLLERS; ++slot) {
        if (occupied[slot] && memcmp(controllers[slot].id, id, KIDI_CONTROLLER_ID_BYTES) == 0) {
            *controller = controllers[slot];
            error = ESP_OK;
            break;
        }
    }
    xSemaphoreGive(store_mutex);
    return error;
}

esp_err_t kidi_controller_store_find_token(const uint8_t token[KIDI_CONTROLLER_TOKEN_BYTES],
                                           kidi_controller_t* controller) {
    if (store_mutex == NULL || token == NULL || controller == NULL) {
        return ESP_ERR_INVALID_ARG;
    }
    xSemaphoreTake(store_mutex, portMAX_DELAY);
    esp_err_t error = ESP_ERR_NOT_FOUND;
    for (size_t slot = 0; slot < KIDI_ACCESSORY_MAX_CONTROLLERS; ++slot) {
        if (occupied[slot] && constant_time_equal(controllers[slot].token, token, KIDI_CONTROLLER_TOKEN_BYTES)) {
            *controller = controllers[slot];
            error = ESP_OK;
            break;
        }
    }
    xSemaphoreGive(store_mutex);
    return error;
}

esp_err_t kidi_controller_store_add(const uint8_t id[KIDI_CONTROLLER_ID_BYTES],
                                    const uint8_t token[KIDI_CONTROLLER_TOKEN_BYTES], const char* name,
                                    kidi_controller_t* controller) {
    if (store_mutex == NULL || id == NULL || token == NULL || !valid_name(name) || controller == NULL) {
        return ESP_ERR_INVALID_ARG;
    }
    xSemaphoreTake(store_mutex, portMAX_DELAY);
    size_t available = KIDI_ACCESSORY_MAX_CONTROLLERS;
    size_t count = 0;
    for (size_t slot = 0; slot < KIDI_ACCESSORY_MAX_CONTROLLERS; ++slot) {
        if (occupied[slot]) {
            ++count;
            if (memcmp(controllers[slot].id, id, KIDI_CONTROLLER_ID_BYTES) == 0) {
                xSemaphoreGive(store_mutex);
                return ESP_ERR_INVALID_STATE;
            }
            if (constant_time_equal(controllers[slot].token, token, KIDI_CONTROLLER_TOKEN_BYTES)) {
                xSemaphoreGive(store_mutex);
                return ESP_ERR_INVALID_STATE;
            }
        } else if (available == KIDI_ACCESSORY_MAX_CONTROLLERS) {
            available = slot;
        }
    }
    if (available == KIDI_ACCESSORY_MAX_CONTROLLERS) {
        xSemaphoreGive(store_mutex);
        return ESP_ERR_NO_MEM;
    }
    kidi_controller_t value{};
    value.role = count == 0 ? KIDI_CONTROLLER_ROLE_OWNER : KIDI_CONTROLLER_ROLE_MEMBER;
    memcpy(value.id, id, sizeof(value.id));
    memcpy(value.token, token, sizeof(value.token));
    memcpy(value.name, name, strlen(name) + 1);
    controllers[available] = value;
    occupied[available] = true;
    esp_err_t error = save_slot(available);
    if (error != ESP_OK) {
        memset(&controllers[available], 0, sizeof(controllers[available]));
        occupied[available] = false;
    } else {
        *controller = value;
    }
    xSemaphoreGive(store_mutex);
    return error;
}

esp_err_t kidi_controller_store_remove(const uint8_t id[KIDI_CONTROLLER_ID_BYTES]) {
    if (store_mutex == NULL || id == NULL) {
        return ESP_ERR_INVALID_ARG;
    }
    xSemaphoreTake(store_mutex, portMAX_DELAY);
    size_t found = KIDI_ACCESSORY_MAX_CONTROLLERS;
    for (size_t slot = 0; slot < KIDI_ACCESSORY_MAX_CONTROLLERS; ++slot) {
        if (occupied[slot] && memcmp(controllers[slot].id, id, KIDI_CONTROLLER_ID_BYTES) == 0) {
            found = slot;
            break;
        }
    }
    if (found == KIDI_ACCESSORY_MAX_CONTROLLERS) {
        xSemaphoreGive(store_mutex);
        return ESP_ERR_NOT_FOUND;
    }
    size_t promoted = KIDI_ACCESSORY_MAX_CONTROLLERS;
    if (controllers[found].role == KIDI_CONTROLLER_ROLE_OWNER) {
        for (size_t slot = 0; slot < KIDI_ACCESSORY_MAX_CONTROLLERS; ++slot) {
            if (slot != found && occupied[slot]) {
                promoted = slot;
                break;
            }
        }
    }
    char removed_key[8];
    slot_key(found, removed_key);
    nvs_handle_t nvs;
    esp_err_t error = nvs_open(store_namespace, NVS_READWRITE, &nvs);
    if (error == ESP_OK) {
        error = nvs_erase_key(nvs, removed_key);
        if (error == ESP_OK && promoted != KIDI_ACCESSORY_MAX_CONTROLLERS) {
            kidi_controller_t promoted_controller = controllers[promoted];
            promoted_controller.role = KIDI_CONTROLLER_ROLE_OWNER;
            uint8_t record[KIDI_CONTROLLER_RECORD_BYTES];
            encode_record(&promoted_controller, record);
            char promoted_key[8];
            slot_key(promoted, promoted_key);
            error = nvs_set_blob(nvs, promoted_key, record, sizeof(record));
        }
        if (error == ESP_OK) {
            error = nvs_commit(nvs);
        }
        nvs_close(nvs);
    }
    if (error == ESP_OK) {
        memset(&controllers[found], 0, sizeof(controllers[found]));
        occupied[found] = false;
        if (promoted != KIDI_ACCESSORY_MAX_CONTROLLERS) {
            controllers[promoted].role = KIDI_CONTROLLER_ROLE_OWNER;
        }
    }
    xSemaphoreGive(store_mutex);
    return error;
}
