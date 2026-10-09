#include <algorithm>
#include <array>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "accessory_peer_protocol.h"
#include "accessory_protocol.h"
#include "eh_cp.h"
#include "eh_cp_feat_peer_data.h"
#include "esp_check.h"
#include "esp_event.h"
#include "esp_log.h"
#include "esp_mac.h"
#include "esp_netif.h"
#include "esp_random.h"
#include "esp_timer.h"
#include "esp_wifi.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "host/ble_att.h"
#include "host/ble_gap.h"
#include "host/ble_gatt.h"
#include "host/ble_hs.h"
#include "host/ble_uuid.h"
#include "kidi_control_server.h"
#include "kidi_controller_store.h"
#include "kidi_identity.h"
#include "kidi_invitation.h"
#include "kidi_pairing.h"
#include "kidi_ticket.h"
#include "kidi_tls_identity.h"
#include "mbedtls/platform_util.h"
#include "nimble/nimble_port.h"
#include "nimble/nimble_port_freertos.h"
#include "nvs.h"
#include "nvs_flash.h"
#include "os/os_mbuf.h"
#include "services/gap/ble_svc_gap.h"
#include "services/gatt/ble_svc_gatt.h"

#define KIDI_AP_PASSWORD_BYTES 16
#define KIDI_BLE_FRAME_BYTES (KIDI_ACCESSORY_MAX_BLE_MESSAGE_BYTES + 2U)
#define KIDI_EXT_ADV_INSTANCE 0

static const char* TAG = "kidi_c6";
static const char AP_PASSWORD_ALPHABET[] = "ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz23456789";

static auto make_ble_uuid(const std::array<std::uint8_t, 16>& bytes) -> ble_uuid128_t {
    ble_uuid128_t uuid{};
    uuid.u.type = BLE_UUID_TYPE_128;
    std::copy(bytes.begin(), bytes.end(), uuid.value);
    return uuid;
}

static constexpr auto SERVICE_UUID_BYTES =
    kidi::esp32::accessory::ble_uuid_little_endian(kidi::esp32::accessory::BLE_SERVICE_UUID);
static constexpr auto PAIR_REQUEST_UUID_BYTES =
    kidi::esp32::accessory::ble_uuid_little_endian(kidi::esp32::accessory::BLE_PAIR_REQUEST_UUID);
static constexpr auto PAIR_RESPONSE_UUID_BYTES =
    kidi::esp32::accessory::ble_uuid_little_endian(kidi::esp32::accessory::BLE_PAIR_RESPONSE_UUID);
static constexpr auto STATUS_UUID_BYTES =
    kidi::esp32::accessory::ble_uuid_little_endian(kidi::esp32::accessory::BLE_STATUS_UUID);

static const ble_uuid128_t SERVICE_UUID = make_ble_uuid(SERVICE_UUID_BYTES);
static const ble_uuid128_t PAIR_REQUEST_UUID = make_ble_uuid(PAIR_REQUEST_UUID_BYTES);
static const ble_uuid128_t PAIR_RESPONSE_UUID = make_ble_uuid(PAIR_RESPONSE_UUID_BYTES);
static const ble_uuid128_t STATUS_UUID = make_ble_uuid(STATUS_UUID_BYTES);

static uint8_t device_id[16];
static uint8_t request_frame[KIDI_BLE_FRAME_BYTES];
static size_t request_frame_size;
static size_t request_expected_size;
static uint8_t response_frame[KIDI_BLE_FRAME_BYTES];
static size_t response_frame_size;
static size_t response_frame_offset;
static uint16_t response_value_handle;
static uint16_t response_connection = BLE_HS_CONN_HANDLE_NONE;
static uint8_t own_address_type;
static bool advertising_configured;
static kidi_tls_identity_t control_tls_identity;
static uint8_t ticket_key[KIDI_TICKET_KEY_BYTES];
static uint8_t media_certificate_sha256[KIDI_TLS_CERTIFICATE_SHA256_BYTES];
static bool media_identity_ready;
static uint64_t epoch_at_sync;
static int64_t monotonic_at_sync;
static bool clock_ready;
static char device_id_hex[33];
static char softap_ssid[33];
static char softap_password[KIDI_AP_PASSWORD_BYTES + 1];
static bool pending_pairing_response;
static bool pending_created_controller;
static uint8_t pending_controller_id[KIDI_CONTROLLER_ID_BYTES];
static ble_gatt_chr_def gatt_characteristics[4]{};
static ble_gatt_svc_def gatt_services[2]{};

static int gap_event(struct ble_gap_event* event, void* argument);

static void reset_ble_exchange(void) {
    if (pending_pairing_response && pending_created_controller) {
        esp_err_t error = kidi_controller_store_remove(pending_controller_id);
        if (error != ESP_OK) {
            ESP_LOGE(TAG, "failed to roll back unacknowledged controller: %s", esp_err_to_name(error));
        }
    }
    pending_pairing_response = false;
    pending_created_controller = false;
    memset(pending_controller_id, 0, sizeof(pending_controller_id));
    request_frame_size = 0;
    request_expected_size = 0;
    response_frame_size = 0;
    response_frame_offset = 0;
    response_connection = BLE_HS_CONN_HANDLE_NONE;
}

static void complete_ble_exchange(void) {
    if (pending_pairing_response) {
        esp_err_t error = kidi_invitation_consume();
        if (error != ESP_OK) {
            ESP_LOGE(TAG, "failed to consume acknowledged invitation: %s", esp_err_to_name(error));
        }
    }
    pending_pairing_response = false;
    pending_created_controller = false;
    memset(pending_controller_id, 0, sizeof(pending_controller_id));
    request_frame_size = 0;
    request_expected_size = 0;
    response_frame_size = 0;
    response_frame_offset = 0;
    response_connection = BLE_HS_CONN_HANDLE_NONE;
}

static esp_err_t init_nvs(void) {
    esp_err_t error = nvs_flash_init();
    if (error == ESP_ERR_NVS_NO_FREE_PAGES || error == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_RETURN_ON_ERROR(nvs_flash_erase(), TAG, "failed to erase incompatible NVS");
        error = nvs_flash_init();
    }
    return error;
}

static esp_err_t load_or_create_ap_password(char password[KIDI_AP_PASSWORD_BYTES + 1]) {
    nvs_handle_t nvs;
    ESP_RETURN_ON_ERROR(nvs_open("kidi", NVS_READWRITE, &nvs), TAG, "failed to open Kidi NVS");
    size_t size = KIDI_AP_PASSWORD_BYTES + 1;
    esp_err_t error = nvs_get_str(nvs, "ap_password", password, &size);
    if (error == ESP_ERR_NVS_NOT_FOUND) {
        for (size_t index = 0; index < KIDI_AP_PASSWORD_BYTES; ++index) {
            password[index] = AP_PASSWORD_ALPHABET[esp_random() % (sizeof(AP_PASSWORD_ALPHABET) - 1)];
        }
        password[KIDI_AP_PASSWORD_BYTES] = '\0';
        error = nvs_set_str(nvs, "ap_password", password);
        if (error == ESP_OK) {
            error = nvs_commit(nvs);
        }
    }
    nvs_close(nvs);
    if (error != ESP_OK || size != KIDI_AP_PASSWORD_BYTES + 1) {
        memset(password, 0, KIDI_AP_PASSWORD_BYTES + 1);
        return error == ESP_OK ? ESP_ERR_INVALID_SIZE : error;
    }
    return ESP_OK;
}

static esp_err_t load_or_create_ticket_key(void) {
    nvs_handle_t nvs;
    ESP_RETURN_ON_ERROR(nvs_open("kidi", NVS_READWRITE, &nvs), TAG, "failed to open ticket-key NVS");
    size_t size = sizeof(ticket_key);
    esp_err_t error = nvs_get_blob(nvs, "ticket_key", ticket_key, &size);
    if (error == ESP_ERR_NVS_NOT_FOUND) {
        esp_fill_random(ticket_key, sizeof(ticket_key));
        error = nvs_set_blob(nvs, "ticket_key", ticket_key, sizeof(ticket_key));
        if (error == ESP_OK) {
            error = nvs_commit(nvs);
        }
    } else if (error == ESP_OK && size != sizeof(ticket_key)) {
        error = ESP_ERR_INVALID_SIZE;
    }
    nvs_close(nvs);
    return error;
}

static uint64_t current_epoch(void) {
    if (!clock_ready) {
        return 0;
    }
    int64_t elapsed = esp_timer_get_time() - monotonic_at_sync;
    return epoch_at_sync + (elapsed > 0 ? (uint64_t)elapsed / 1000000U : 0);
}

static void sync_request(uint32_t message_id, const uint8_t* data, size_t data_size, void* context) {
    (void)message_id;
    (void)context;
    if (data == NULL || data_size != KIDI_PEER_SYNC_REQUEST_BYTES || data[0] != KIDI_ACCESSORY_PROTOCOL_VERSION) {
        ESP_LOGE(TAG, "rejected malformed P4 security sync");
        return;
    }
    memcpy(media_certificate_sha256, data + 1, sizeof(media_certificate_sha256));
    media_identity_ready = true;
    uint64_t proposed_epoch = kidi_peer_read_u64_be(data + 1 + sizeof(media_certificate_sha256));
    if (proposed_epoch >= 1700000000ULL) {
        epoch_at_sync = proposed_epoch;
        monotonic_at_sync = esp_timer_get_time();
        clock_ready = true;
    }

    uint8_t response[KIDI_PEER_SYNC_RESPONSE_BYTES] = {
        [0] = KIDI_ACCESSORY_PROTOCOL_VERSION,
        [1] = KIDI_PEER_SYNC_STATUS_CLOCK_MISSING,
    };
    uint64_t now = current_epoch();
    if (now != 0) {
        response[1] = KIDI_PEER_SYNC_STATUS_READY;
        kidi_peer_write_u64_be(response + 2, now);
        memcpy(response + 10, ticket_key, sizeof(ticket_key));
    }
    esp_err_t error = eh_cp_feat_peer_data_send(KIDI_PEER_SYNC_RESPONSE_ID, response, sizeof(response));
    if (error != ESP_OK) {
        ESP_LOGE(TAG, "failed to return P4 security sync: %s", esp_err_to_name(error));
    }
}

static void invitation_request(uint32_t message_id, const uint8_t* data, size_t data_size, void* context) {
    (void)message_id;
    (void)context;
    if (data == NULL || data_size != KIDI_PEER_INVITATION_REQUEST_BYTES || data[0] != KIDI_ACCESSORY_PROTOCOL_VERSION) {
        ESP_LOGE(TAG, "rejected malformed invitation request");
        return;
    }
    uint64_t proposed_epoch = kidi_peer_read_u64_be(data + 1);
    uint32_t lifetime_seconds = ((uint32_t)data[9] << 8U) | data[10];
    if (proposed_epoch < 1700000000ULL || lifetime_seconds == 0 ||
        lifetime_seconds > KIDI_INVITATION_MAX_LIFETIME_SECONDS || !media_identity_ready) {
        ESP_LOGE(TAG, "invitation prerequisites are not ready");
        return;
    }
    epoch_at_sync = proposed_epoch;
    monotonic_at_sync = esp_timer_get_time();
    clock_ready = true;
    char uri[KIDI_INVITATION_URI_MAX_BYTES];
    size_t uri_size = 0;
    esp_err_t error =
        kidi_invitation_create(proposed_epoch, lifetime_seconds, device_id_hex, "Kidi accessory", uri, &uri_size);
    if (error == ESP_OK) {
        error = eh_cp_feat_peer_data_send(KIDI_PEER_INVITATION_RESPONSE_ID, (const uint8_t*)uri, uri_size);
    }
    if (error != ESP_OK) {
        ESP_LOGE(TAG, "failed to create pairing invitation: %s", esp_err_to_name(error));
    }
}

static void media_complete(uint32_t message_id, const uint8_t* data, size_t data_size, void* context) {
    (void)message_id;
    (void)context;
    if (data == nullptr || data_size != KIDI_PEER_MEDIA_COMPLETE_BYTES || data[0] != KIDI_ACCESSORY_PROTOCOL_VERSION) {
        ESP_LOGE(TAG, "rejected malformed media completion");
        return;
    }
    kidi::c6::complete_media_session(
        std::span<const uint8_t, KIDI_ACCESSORY_TICKET_NONCE_BYTES>{data + 1, KIDI_ACCESSORY_TICKET_NONCE_BYTES});
}

static esp_err_t start_softap(void) {
    uint8_t mac[6];
    ESP_RETURN_ON_ERROR(esp_read_mac(mac, ESP_MAC_WIFI_SOFTAP), TAG, "failed to read Wi-Fi MAC");

    int written = snprintf(softap_ssid, sizeof(softap_ssid), "Kidi-%02X%02X%02X", mac[3], mac[4], mac[5]);
    ESP_RETURN_ON_FALSE(written > 0 && written < sizeof(softap_ssid), ESP_ERR_INVALID_SIZE, TAG,
                        "failed to format SoftAP SSID");
    ESP_RETURN_ON_ERROR(load_or_create_ap_password(softap_password), TAG, "failed to load SoftAP credentials");

    ESP_RETURN_ON_ERROR(esp_netif_init(), TAG, "failed to initialize esp-netif");
    esp_err_t error = esp_event_loop_create_default();
    ESP_RETURN_ON_FALSE(error == ESP_OK || error == ESP_ERR_INVALID_STATE, error, TAG,
                        "failed to create default event loop");
    ESP_RETURN_ON_FALSE(esp_netif_create_default_wifi_ap() != NULL, ESP_FAIL, TAG, "failed to create SoftAP netif");
    wifi_init_config_t init = WIFI_INIT_CONFIG_DEFAULT();
    ESP_RETURN_ON_ERROR(esp_wifi_init(&init), TAG, "failed to initialize Wi-Fi");
    ESP_RETURN_ON_ERROR(esp_wifi_set_storage(WIFI_STORAGE_RAM), TAG, "failed to select volatile Wi-Fi config");

    wifi_config_t config{};
    memcpy(config.ap.ssid, softap_ssid, strlen(softap_ssid));
    config.ap.ssid_len = strlen(softap_ssid);
    memcpy(config.ap.password, softap_password, strlen(softap_password));
    config.ap.channel = 6;
    config.ap.authmode = WIFI_AUTH_WPA2_PSK;
    config.ap.max_connection = KIDI_ACCESSORY_MAX_CONTROLLERS;
    config.ap.pmf_cfg.capable = true;
    config.ap.pmf_cfg.required = false;
    ESP_RETURN_ON_ERROR(esp_wifi_set_mode(WIFI_MODE_AP), TAG, "failed to select SoftAP mode");
    ESP_RETURN_ON_ERROR(esp_wifi_set_config(WIFI_IF_AP, &config), TAG, "failed to configure SoftAP");
    ESP_RETURN_ON_ERROR(esp_wifi_start(), TAG, "failed to start SoftAP");

    ESP_LOGI(TAG, "SoftAP ready: %s (credential available only through pairing)", softap_ssid);
    return ESP_OK;
}

static int append_request_fragment(struct os_mbuf* mbuf) {
    uint16_t fragment_size = OS_MBUF_PKTLEN(mbuf);
    if (fragment_size == 0 || request_frame_size + fragment_size > sizeof(request_frame)) {
        reset_ble_exchange();
        return BLE_ATT_ERR_INVALID_ATTR_VALUE_LEN;
    }
    uint16_t copied = 0;
    int result = ble_hs_mbuf_to_flat(mbuf, request_frame + request_frame_size,
                                     sizeof(request_frame) - request_frame_size, &copied);
    if (result != 0 || copied != fragment_size) {
        reset_ble_exchange();
        return BLE_ATT_ERR_UNLIKELY;
    }
    request_frame_size += copied;
    if (request_frame_size >= 2 && request_expected_size == 0) {
        request_expected_size = ((size_t)request_frame[0] << 8U) | request_frame[1];
        if (request_expected_size == 0 || request_expected_size > KIDI_ACCESSORY_MAX_BLE_MESSAGE_BYTES) {
            reset_ble_exchange();
            return BLE_ATT_ERR_INVALID_ATTR_VALUE_LEN;
        }
    }
    if (request_expected_size != 0 && request_frame_size > request_expected_size + 2) {
        reset_ble_exchange();
        return BLE_ATT_ERR_INVALID_ATTR_VALUE_LEN;
    }
    return 0;
}

static int queue_error_response(uint16_t connection, const char* reason) {
    char payload[96];
    int payload_size = snprintf(payload, sizeof(payload), "{\"v\":1,\"error\":\"%s\"}", reason);
    if (payload_size <= 0 || payload_size >= sizeof(payload)) {
        return BLE_ATT_ERR_UNLIKELY;
    }
    response_frame[0] = (uint8_t)(payload_size >> 8U);
    response_frame[1] = (uint8_t)payload_size;
    memcpy(response_frame + 2, payload, payload_size);
    response_frame_size = payload_size + 2;
    response_frame_offset = 0;
    response_connection = connection;
    return 0;
}

static int queue_pairing_response(uint16_t connection) {
    uint8_t invitation_secret[KIDI_PAIRING_INVITATION_SECRET_BYTES];
    uint64_t expires_at = 0;
    uint64_t now = current_epoch();
    if (!media_identity_ready || kidi_invitation_snapshot(now, invitation_secret, &expires_at) != ESP_OK) {
        return queue_error_response(connection, "pairing-invitation-not-ready");
    }
    kidi_pairing_config_t config = {
        .device_id_hex = device_id_hex,
        .device_name = "Kidi accessory",
        .host = "192.168.4.1",
        .softap_ssid = softap_ssid,
        .softap_password = softap_password,
        .control_certificate_sha256 = control_tls_identity.certificate_sha256,
        .media_certificate_sha256 = media_certificate_sha256,
        .invitation_secret = invitation_secret,
        .invitation_expires_at = expires_at,
        .current_epoch = now,
    };
    size_t payload_size = 0;
    bool created_controller = false;
    esp_err_t error = kidi_pairing_create_response(request_frame + 2, request_expected_size, &config,
                                                   response_frame + 2, sizeof(response_frame) - 2, &payload_size,
                                                   &created_controller, pending_controller_id);
    mbedtls_platform_zeroize(invitation_secret, sizeof(invitation_secret));
    if (error != ESP_OK) {
        memset(pending_controller_id, 0, sizeof(pending_controller_id));
        return queue_error_response(connection, "pairing-rejected");
    }
    response_frame[0] = (uint8_t)(payload_size >> 8U);
    response_frame[1] = (uint8_t)payload_size;
    response_frame_size = payload_size + 2;
    response_frame_offset = 0;
    response_connection = connection;
    pending_pairing_response = true;
    pending_created_controller = created_controller;
    return 0;
}

static int send_next_response_fragment(void) {
    if (response_connection == BLE_HS_CONN_HANDLE_NONE || response_frame_offset >= response_frame_size) {
        return 0;
    }
    uint16_t mtu = ble_att_mtu(response_connection);
    size_t maximum = mtu > 3 ? mtu - 3 : 1;
    size_t remaining = response_frame_size - response_frame_offset;
    size_t count = remaining < maximum ? remaining : maximum;
    struct os_mbuf* mbuf = ble_hs_mbuf_from_flat(response_frame + response_frame_offset, count);
    if (mbuf == NULL) {
        return BLE_HS_ENOMEM;
    }
    int result = ble_gatts_indicate_custom(response_connection, response_value_handle, mbuf);
    if (result == 0) {
        response_frame_offset += count;
    }
    return result;
}

static int pair_request_access(uint16_t connection, uint16_t attribute, struct ble_gatt_access_ctxt* context,
                               void* argument) {
    (void)attribute;
    (void)argument;
    if (context->op != BLE_GATT_ACCESS_OP_WRITE_CHR) {
        return BLE_ATT_ERR_WRITE_NOT_PERMITTED;
    }
    if (response_connection != BLE_HS_CONN_HANDLE_NONE) {
        return BLE_ATT_ERR_INSUFFICIENT_RES;
    }
    int result = append_request_fragment(context->om);
    if (result != 0) {
        return result;
    }
    if (request_expected_size != 0 && request_frame_size == request_expected_size + 2) {
        result = queue_pairing_response(connection);
        if (result != 0) {
            reset_ble_exchange();
            return result;
        }
        result = send_next_response_fragment();
        request_frame_size = 0;
        request_expected_size = 0;
        if (result != 0) {
            reset_ble_exchange();
            return BLE_ATT_ERR_UNLIKELY;
        }
    }
    return 0;
}

static int status_access(uint16_t connection, uint16_t attribute, struct ble_gatt_access_ctxt* context,
                         void* argument) {
    (void)connection;
    (void)attribute;
    (void)argument;
    if (context->op != BLE_GATT_ACCESS_OP_READ_CHR) {
        return BLE_ATT_ERR_READ_NOT_PERMITTED;
    }
    uint8_t secret[KIDI_PAIRING_INVITATION_SECRET_BYTES];
    uint64_t expires_at = 0;
    bool pairing_ready =
        kidi_invitation_snapshot(current_epoch(), secret, &expires_at) == ESP_OK && media_identity_ready;
    mbedtls_platform_zeroize(secret, sizeof(secret));
    char status[112];
    int size = snprintf(status, sizeof(status), "{\"v\":1,\"pairing\":%s,\"controllers\":%u,\"media_busy\":false}",
                        pairing_ready ? "true" : "false", (unsigned)kidi_controller_store_count());
    if (size <= 0 || size >= sizeof(status)) {
        return BLE_ATT_ERR_UNLIKELY;
    }
    return os_mbuf_append(context->om, status, size) == 0 ? 0 : BLE_ATT_ERR_INSUFFICIENT_RES;
}

static void configure_gatt_services() {
    gatt_characteristics[0].uuid = &PAIR_REQUEST_UUID.u;
    gatt_characteristics[0].access_cb = pair_request_access;
    gatt_characteristics[0].flags = BLE_GATT_CHR_F_WRITE;

    gatt_characteristics[1].uuid = &PAIR_RESPONSE_UUID.u;
    gatt_characteristics[1].flags = BLE_GATT_CHR_F_INDICATE;
    gatt_characteristics[1].val_handle = &response_value_handle;

    gatt_characteristics[2].uuid = &STATUS_UUID.u;
    gatt_characteristics[2].access_cb = status_access;
    gatt_characteristics[2].flags = BLE_GATT_CHR_F_READ;

    gatt_services[0].type = BLE_GATT_SVC_TYPE_PRIMARY;
    gatt_services[0].uuid = &SERVICE_UUID.u;
    gatt_services[0].characteristics = gatt_characteristics;
}

static int append_advertising_field(uint8_t* data, size_t capacity, size_t* offset, uint8_t type, const uint8_t* value,
                                    size_t value_size) {
    if (value_size > UINT8_MAX - 1 || *offset + value_size + 2 > capacity) {
        return BLE_HS_EMSGSIZE;
    }
    data[(*offset)++] = value_size + 1;
    data[(*offset)++] = type;
    memcpy(data + *offset, value, value_size);
    *offset += value_size;
    return 0;
}

static int start_advertising(void) {
    if (!advertising_configured) {
        ble_gap_ext_adv_params parameters{};
        parameters.connectable = 1;
        parameters.own_addr_type = own_address_type;
        parameters.primary_phy = BLE_HCI_LE_PHY_1M;
        parameters.secondary_phy = BLE_HCI_LE_PHY_2M;
        parameters.tx_power = 127;
        parameters.sid = 1;
        parameters.itvl_min = BLE_GAP_ADV_FAST_INTERVAL1_MIN;
        parameters.itvl_max = BLE_GAP_ADV_FAST_INTERVAL1_MAX;
        int result = ble_gap_ext_adv_configure(KIDI_EXT_ADV_INSTANCE, &parameters, NULL, gap_event, NULL);
        if (result != 0) {
            return result;
        }

        uint8_t data[64];
        size_t size = 0;
        static const uint8_t flags = BLE_HS_ADV_F_DISC_GEN | BLE_HS_ADV_F_BREDR_UNSUP;
        result = append_advertising_field(data, sizeof(data), &size, BLE_HS_ADV_TYPE_FLAGS, &flags, sizeof(flags));
        if (result != 0) {
            return result;
        }
        result = append_advertising_field(data, sizeof(data), &size, BLE_HS_ADV_TYPE_COMP_UUIDS128, SERVICE_UUID.value,
                                          sizeof(SERVICE_UUID.value));
        if (result != 0) {
            return result;
        }
        uint8_t service_data[sizeof(SERVICE_UUID.value) + 1 + sizeof(device_id)];
        memcpy(service_data, SERVICE_UUID.value, sizeof(SERVICE_UUID.value));
        service_data[sizeof(SERVICE_UUID.value)] = KIDI_ACCESSORY_PROTOCOL_VERSION;
        memcpy(service_data + sizeof(SERVICE_UUID.value) + 1, device_id, sizeof(device_id));
        result = append_advertising_field(data, sizeof(data), &size, BLE_HS_ADV_TYPE_SVC_DATA_UUID128, service_data,
                                          sizeof(service_data));
        if (result != 0) {
            return result;
        }
        struct os_mbuf* mbuf = ble_hs_mbuf_from_flat(data, size);
        if (mbuf == NULL) {
            return BLE_HS_ENOMEM;
        }
        result = ble_gap_ext_adv_set_data(KIDI_EXT_ADV_INSTANCE, mbuf);
        if (result != 0) {
            return result;
        }
        advertising_configured = true;
    }
    return ble_gap_ext_adv_start(KIDI_EXT_ADV_INSTANCE, 0, 0);
}

static int gap_event(struct ble_gap_event* event, void* argument) {
    (void)argument;
    switch (event->type) {
        case BLE_GAP_EVENT_CONNECT:
            if (event->connect.status != 0) {
                ESP_LOGW(TAG, "BLE connection failed: %d", event->connect.status);
                return start_advertising();
            }
            ESP_LOGI(TAG, "BLE controller connected");
            reset_ble_exchange();
            return 0;
        case BLE_GAP_EVENT_DISCONNECT:
            ESP_LOGI(TAG, "BLE controller disconnected: %d", event->disconnect.reason);
            reset_ble_exchange();
            return start_advertising();
        case BLE_GAP_EVENT_NOTIFY_TX:
            if (event->notify_tx.indication && event->notify_tx.status == 0 &&
                response_frame_offset < response_frame_size) {
                return send_next_response_fragment();
            }
            if (event->notify_tx.indication) {
                if (event->notify_tx.status == 0) {
                    complete_ble_exchange();
                } else {
                    reset_ble_exchange();
                }
            }
            return 0;
        default:
            return 0;
    }
}

static void ble_on_reset(int reason) { ESP_LOGE(TAG, "NimBLE reset: %d", reason); }

static void ble_on_sync(void) {
    int result = ble_hs_id_infer_auto(0, &own_address_type);
    if (result == 0) {
        result = start_advertising();
    }
    if (result != 0) {
        ESP_LOGE(TAG, "failed to start BLE advertising: %d", result);
    } else {
        ESP_LOGI(TAG, "BLE pairing service ready");
    }
}

static void ble_host_task(void* argument) {
    (void)argument;
    nimble_port_run();
    nimble_port_freertos_deinit();
}

static esp_err_t start_ble(void) {
    ESP_RETURN_ON_ERROR(nimble_port_init(), TAG, "failed to initialize NimBLE");
    ble_hs_cfg.reset_cb = ble_on_reset;
    ble_hs_cfg.sync_cb = ble_on_sync;
    ble_svc_gap_init();
    ble_svc_gatt_init();
    int result = ble_svc_gap_device_name_set("Kidi accessory");
    ESP_RETURN_ON_FALSE(result == 0, ESP_FAIL, TAG, "failed to set BLE device name: %d", result);
    configure_gatt_services();
    result = ble_gatts_count_cfg(gatt_services);
    ESP_RETURN_ON_FALSE(result == 0, ESP_FAIL, TAG, "failed to count GATT service: %d", result);
    result = ble_gatts_add_svcs(gatt_services);
    ESP_RETURN_ON_FALSE(result == 0, ESP_FAIL, TAG, "failed to add GATT service: %d", result);
    nimble_port_freertos_init(ble_host_task);
    return ESP_OK;
}

extern "C" void app_main() {
    ESP_ERROR_CHECK(init_nvs());
    ESP_ERROR_CHECK(kidi_identity_init("kidi_identity", "pair_key"));
    ESP_ERROR_CHECK(kidi_identity_get_device_id(device_id));
    for (size_t index = 0; index < sizeof(device_id); ++index) {
        snprintf(device_id_hex + index * 2, 3, "%02x", device_id[index]);
    }
    ESP_ERROR_CHECK(kidi_controller_store_init("kidi_ctrl"));
    ESP_ERROR_CHECK(kidi_invitation_init());
    ESP_ERROR_CHECK(kidi_tls_identity_init("kidi_tls", "control", "Kidi control", &control_tls_identity));
    ESP_ERROR_CHECK(load_or_create_ticket_key());
    ESP_ERROR_CHECK(eh_cp_init());
    ESP_ERROR_CHECK(eh_cp_feat_peer_data_register_callback(KIDI_PEER_SYNC_REQUEST_ID, sync_request, NULL));
    ESP_ERROR_CHECK(eh_cp_feat_peer_data_register_callback(KIDI_PEER_INVITATION_REQUEST_ID, invitation_request, NULL));
    ESP_ERROR_CHECK(eh_cp_feat_peer_data_register_callback(KIDI_PEER_MEDIA_COMPLETE_ID, media_complete, nullptr));
    ESP_ERROR_CHECK(start_softap());
    ESP_ERROR_CHECK(kidi::c6::start_control_server(
        control_tls_identity, std::span<const uint8_t, KIDI_TICKET_KEY_BYTES>{ticket_key}, current_epoch));
    ESP_ERROR_CHECK(start_ble());
    ESP_LOGW(TAG,
             "bring-up firmware only: LP-core supervision remains disabled; do not flash "
             "without a verified C6 recovery image");
    while (true) {
        vTaskDelay(pdMS_TO_TICKS(10000));
    }
}
