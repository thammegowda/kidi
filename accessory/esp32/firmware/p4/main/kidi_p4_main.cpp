#include <algorithm>
#include <array>
#include <stdbool.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <span>
#include <sys/time.h>
#include <time.h>

#include "accessory_peer_protocol.h"
#include "accessory_protocol.h"
#include "eh_host_feat_peer_data.h"
#include "eh_host_power_save.h"
#include "esp_check.h"
#include "esp_console.h"
#include "esp_err.h"
#include "esp_event.h"
#include "esp_hosted.h"
#include "esp_log.h"
#include "esp_netif.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "kidi_tls_identity.h"
#include "kidi_media_server.h"
#include "nvs_flash.h"

#define KIDI_BRINGUP_IDLE_MILLISECONDS 30000U

static const char* TAG = "kidi_p4";
static kidi_tls_identity_t media_tls_identity;
static volatile bool security_synchronized;

static void media_session_complete(std::span<const uint8_t, KIDI_ACCESSORY_TICKET_NONCE_BYTES> nonce) {
    std::array<uint8_t, KIDI_PEER_MEDIA_COMPLETE_BYTES> message{};
    message[0] = KIDI_ACCESSORY_PROTOCOL_VERSION;
    std::copy(nonce.begin(), nonce.end(), message.begin() + 1);
    const esp_err_t error = eh_host_peer_data_send(KIDI_PEER_MEDIA_COMPLETE_ID, message.data(), message.size());
    if (error != ESP_OK) {
        ESP_LOGE(TAG, "failed to report media completion: %s", esp_err_to_name(error));
    }
}

static esp_err_t init_nvs(void) {
    esp_err_t error = nvs_flash_init();
    if (error == ESP_ERR_NVS_NO_FREE_PAGES || error == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        error = nvs_flash_init();
    }
    return error;
}

static esp_err_t init_network_split(void) {
    esp_err_t error = esp_netif_init();
    if (error != ESP_OK && error != ESP_ERR_INVALID_STATE) {
        return error;
    }
    error = esp_event_loop_create_default();
    if (error != ESP_OK && error != ESP_ERR_INVALID_STATE) {
        return error;
    }
    esp_hosted_init();
    esp_hosted_connect_to_slave();
    return ESP_OK;
}

static void sync_response(uint32_t message_id, const uint8_t* data, size_t data_size, void* context) {
    (void)message_id;
    (void)context;
    if (data == NULL || data_size != KIDI_PEER_SYNC_RESPONSE_BYTES || data[0] != KIDI_ACCESSORY_PROTOCOL_VERSION) {
        ESP_LOGE(TAG, "rejected malformed C6 security sync");
        return;
    }
    if (data[1] != KIDI_PEER_SYNC_STATUS_READY) {
        ESP_LOGW(TAG, "C6 security sync is waiting for a trusted epoch");
        return;
    }
    uint64_t epoch = kidi_peer_read_u64_be(data + 2);
    esp_err_t error = kidi::p4::configure_media_security(
        std::span<const uint8_t, KIDI_TICKET_KEY_BYTES>{data + 10, KIDI_TICKET_KEY_BYTES}, epoch);
    if (error != ESP_OK) {
        ESP_LOGE(TAG, "failed to install C6 media security: %s", esp_err_to_name(error));
        return;
    }
    security_synchronized = true;
    ESP_LOGI(TAG, "C6 media ticket key and clock synchronized");
}

static void invitation_response(uint32_t message_id, const uint8_t* data, size_t data_size, void* context) {
    (void)message_id;
    (void)context;
    if (data == NULL || data_size == 0 || data_size >= KIDI_PEER_INVITATION_MAX_RESPONSE_BYTES ||
        data_size < strlen("kidi://pair/v1#") || memcmp(data, "kidi://pair/v1#", strlen("kidi://pair/v1#")) != 0) {
        ESP_LOGE(TAG, "rejected malformed pairing invitation response");
        return;
    }
    printf("\nKIDI_PAIRING_URI %.*s\n", (int)data_size, (const char*)data);
    fflush(stdout);
}

static int pair_invite_command(int argc, char** argv) {
    if (argc < 2 || argc > 3) {
        printf("usage: pair-invite <unix-seconds> [lifetime-seconds]\n");
        return 1;
    }
    errno = 0;
    char* end = NULL;
    unsigned long long epoch = strtoull(argv[1], &end, 10);
    if (errno != 0 || end == argv[1] || *end != '\0' || epoch < 1700000000ULL) {
        printf("pair-invite: invalid Unix time\n");
        return 1;
    }
    unsigned long lifetime = 300;
    if (argc == 3) {
        errno = 0;
        end = NULL;
        lifetime = strtoul(argv[2], &end, 10);
        if (errno != 0 || end == argv[2] || *end != '\0' || lifetime == 0 || lifetime > 900) {
            printf("pair-invite: lifetime must be 1..900 seconds\n");
            return 1;
        }
    }
    timeval time{};
    time.tv_sec = static_cast<time_t>(epoch);
    if (settimeofday(&time, NULL) != 0) {
        printf("pair-invite: failed to set clock\n");
        return 1;
    }
    uint8_t request[KIDI_PEER_INVITATION_REQUEST_BYTES]{};
    request[0] = KIDI_ACCESSORY_PROTOCOL_VERSION;
    kidi_peer_write_u64_be(request + 1, epoch);
    request[9] = (uint8_t)(lifetime >> 8U);
    request[10] = (uint8_t)lifetime;
    esp_err_t error = eh_host_peer_data_send(KIDI_PEER_INVITATION_REQUEST_ID, request, sizeof(request));
    if (error != ESP_OK) {
        printf("pair-invite: C6 request failed: %s\n", esp_err_to_name(error));
        return 1;
    }
    printf("pair-invite: waiting for one-time URI\n");
    return 0;
}

static esp_err_t start_console(void) {
    esp_console_cmd_t command{};
    command.command = "pair-invite";
    command.help = "Create a physical-USB one-time pairing URI";
    command.hint = "<unix-seconds> [lifetime-seconds]";
    command.func = pair_invite_command;
    ESP_RETURN_ON_ERROR(esp_console_cmd_register(&command), TAG, "failed to register pair-invite command");
    esp_console_repl_t* repl = NULL;
    esp_console_repl_config_t repl_config = ESP_CONSOLE_REPL_CONFIG_DEFAULT();
    repl_config.prompt = "kidi> ";
    esp_console_dev_uart_config_t uart_config = ESP_CONSOLE_DEV_UART_CONFIG_DEFAULT();
    ESP_RETURN_ON_ERROR(esp_console_new_repl_uart(&uart_config, &repl_config, &repl), TAG,
                        "failed to create UART console");
    return esp_console_start_repl(repl);
}

static void security_sync_task(void* context) {
    (void)context;
    while (!security_synchronized) {
        uint8_t request[KIDI_PEER_SYNC_REQUEST_BYTES]{};
        request[0] = KIDI_ACCESSORY_PROTOCOL_VERSION;
        memcpy(request + 1, media_tls_identity.certificate_sha256, sizeof(media_tls_identity.certificate_sha256));
        time_t now = time(NULL);
        if (now >= 1700000000) {
            kidi_peer_write_u64_be(request + 1 + sizeof(media_tls_identity.certificate_sha256), (uint64_t)now);
        }
        esp_err_t error = eh_host_peer_data_send(KIDI_PEER_SYNC_REQUEST_ID, request, sizeof(request));
        if (error != ESP_OK) {
            ESP_LOGW(TAG, "P4 security sync request failed: %s", esp_err_to_name(error));
        }
        vTaskDelay(pdMS_TO_TICKS(2000));
    }
    vTaskDelete(NULL);
}

extern "C" void app_main() {
    ESP_ERROR_CHECK(init_nvs());
    ESP_ERROR_CHECK(kidi_tls_identity_init("kidi_tls", "media", "Kidi media", &media_tls_identity));
    bool resumed_from_c6 = eh_host_power_save_enabled() && eh_host_woke_from_power_save();
    ESP_LOGI(TAG, "boot reason: %s", resumed_from_c6 ? "C6 host wake" : "normal reset");
    ESP_ERROR_CHECK(init_network_split());
    ESP_ERROR_CHECK(eh_host_peer_data_register(KIDI_PEER_SYNC_RESPONSE_ID, sync_response, NULL));
    ESP_ERROR_CHECK(eh_host_peer_data_register(KIDI_PEER_INVITATION_RESPONSE_ID, invitation_response, NULL));
    ESP_ERROR_CHECK(kidi::p4::start_media_server(media_tls_identity, media_session_complete));
    BaseType_t task_created = xTaskCreate(security_sync_task, "kidi_security_sync", 4096, NULL, 5, NULL);
    ESP_ERROR_CHECK(task_created == pdPASS ? ESP_OK : ESP_ERR_NO_MEM);
    ESP_ERROR_CHECK(start_console());

    if (!eh_host_power_save_enabled()) {
        ESP_LOGE(TAG, "Hosted deep-sleep support is not enabled");
    } else {
        int result = eh_host_power_save_timer_start(KIDI_BRINGUP_IDLE_MILLISECONDS);
        if (result != 0) {
            ESP_LOGE(TAG, "failed to arm Hosted idle sleep timer: %d", result);
        } else {
            ESP_LOGI(TAG, "idle deep sleep armed for %u ms", KIDI_BRINGUP_IDLE_MILLISECONDS);
        }
    }

    ESP_LOGW(TAG,
             "bring-up firmware only: media requests remain locked until C6 ticket/time sync; "
             "LP-core supervision is not enabled");
    while (true) {
        vTaskDelay(pdMS_TO_TICKS(10000));
    }
}
