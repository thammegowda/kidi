#include "kidi_media_server.h"

#include <algorithm>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>

#include "cJSON.h"
#include "esp_check.h"
#include "esp_http_server.h"
#include "esp_https_server.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "kidi_audio.h"
#include "kidi_camera.h"
#include "mbedtls/platform_util.h"
#include "mbedtls/sha256.h"

#define KIDI_REQUEST_BODY_BYTES 512U
#define KIDI_REPLAY_SLOTS 8U

static const char* TAG = "kidi_media";

typedef struct {
    bool occupied;
    uint64_t expires_at;
    uint8_t nonce[KIDI_ACCESSORY_TICKET_NONCE_BYTES];
} replay_entry_t;

typedef struct {
    httpd_req_t* request;
    size_t sent_samples;
} audio_sink_context_t;

static httpd_handle_t server;
static SemaphoreHandle_t media_mutex;
static SemaphoreHandle_t security_mutex;
static uint8_t active_ticket_key[KIDI_TICKET_KEY_BYTES];
static uint64_t epoch_at_sync;
static int64_t monotonic_at_sync;
static bool security_ready;
static replay_entry_t replay_entries[KIDI_REPLAY_SLOTS];
static kidi::p4::CompletionCallback completion_callback;

static esp_err_t send_json_error(httpd_req_t* request, const char* status, const char* message) {
    char body[160];
    int size = snprintf(body, sizeof(body), "{\"error\":\"%s\"}", message);
    if (size <= 0 || size >= sizeof(body)) {
        return ESP_FAIL;
    }
    httpd_resp_set_status(request, status);
    httpd_resp_set_type(request, "application/json");
    return httpd_resp_send(request, body, size);
}

static esp_err_t receive_body(httpd_req_t* request, char body[KIDI_REQUEST_BODY_BYTES + 1], size_t* body_size) {
    if (request->content_len <= 0 || request->content_len > KIDI_REQUEST_BODY_BYTES) {
        return ESP_ERR_INVALID_SIZE;
    }
    size_t received = 0;
    while (received < request->content_len) {
        int count = httpd_req_recv(request, body + received, request->content_len - received);
        if (count == HTTPD_SOCK_ERR_TIMEOUT) {
            continue;
        }
        if (count <= 0) {
            return ESP_FAIL;
        }
        received += count;
    }
    body[received] = '\0';
    *body_size = received;
    return ESP_OK;
}

static bool object_has_only(const cJSON* object, const char* first, const char* second, const char* third,
                            const char* fourth) {
    const cJSON* item = NULL;
    cJSON_ArrayForEach(item, object) {
        if (item->string == NULL ||
            (strcmp(item->string, first) != 0 && (second == NULL || strcmp(item->string, second) != 0) &&
             (third == NULL || strcmp(item->string, third) != 0) &&
             (fourth == NULL || strcmp(item->string, fourth) != 0))) {
            return false;
        }
    }
    return true;
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

static esp_err_t canonical_photo_parameters(const cJSON* request, uint8_t output[32], size_t* maximum_bytes) {
    if (!cJSON_IsObject(request) || !object_has_only(request, "profile", "maximum_bytes", NULL, NULL)) {
        return ESP_ERR_INVALID_ARG;
    }
    const cJSON* profile = cJSON_GetObjectItemCaseSensitive(request, "profile");
    const cJSON* maximum = cJSON_GetObjectItemCaseSensitive(request, "maximum_bytes");
    if (!cJSON_IsString(profile) || profile->valuestring == NULL || strcmp(profile->valuestring, "chat") != 0 ||
        !cJSON_IsNumber(maximum) || maximum->valuedouble != maximum->valueint || maximum->valueint <= 0 ||
        maximum->valueint > KIDI_ACCESSORY_MAX_PHOTO_BYTES) {
        return ESP_ERR_INVALID_ARG;
    }
    static const char parameters[] = "{\"profile\":\"chat\"}";
    *maximum_bytes = maximum->valueint;
    return sha256((const uint8_t*)parameters, sizeof(parameters) - 1, output);
}

static esp_err_t canonical_audio_parameters(const cJSON* request, uint8_t output[32], uint32_t* maximum_seconds) {
    if (!cJSON_IsObject(request) ||
        !object_has_only(request, "sample_rate", "channels", "bits_per_sample", "maximum_seconds")) {
        return ESP_ERR_INVALID_ARG;
    }
    const cJSON* sample_rate = cJSON_GetObjectItemCaseSensitive(request, "sample_rate");
    const cJSON* channels = cJSON_GetObjectItemCaseSensitive(request, "channels");
    const cJSON* bits = cJSON_GetObjectItemCaseSensitive(request, "bits_per_sample");
    const cJSON* maximum = cJSON_GetObjectItemCaseSensitive(request, "maximum_seconds");
    if (!cJSON_IsNumber(sample_rate) || sample_rate->valuedouble != sample_rate->valueint ||
        sample_rate->valueint != KIDI_ACCESSORY_AUDIO_SAMPLE_RATE || !cJSON_IsNumber(channels) ||
        channels->valuedouble != channels->valueint || channels->valueint != KIDI_ACCESSORY_AUDIO_CHANNELS ||
        !cJSON_IsNumber(bits) || bits->valuedouble != bits->valueint ||
        bits->valueint != KIDI_ACCESSORY_AUDIO_BITS_PER_SAMPLE || !cJSON_IsNumber(maximum) ||
        maximum->valuedouble != maximum->valueint || maximum->valueint <= 0 ||
        maximum->valueint > KIDI_ACCESSORY_MAX_AUDIO_SECONDS) {
        return ESP_ERR_INVALID_ARG;
    }
    char parameters[48];
    int size = snprintf(parameters, sizeof(parameters), "{\"maximum_seconds\":%d}", maximum->valueint);
    if (size <= 0 || size >= sizeof(parameters)) {
        return ESP_FAIL;
    }
    *maximum_seconds = maximum->valueint;
    return sha256((const uint8_t*)parameters, size, output);
}

static bool register_nonce(const kidi_ticket_claims_t* claims, uint64_t now) {
    replay_entry_t* available = NULL;
    for (size_t index = 0; index < KIDI_REPLAY_SLOTS; ++index) {
        replay_entry_t* entry = &replay_entries[index];
        if (entry->occupied && entry->expires_at < now) {
            memset(entry, 0, sizeof(*entry));
        }
        if (entry->occupied && memcmp(entry->nonce, claims->nonce, sizeof(entry->nonce)) == 0) {
            return false;
        }
        if (!entry->occupied && available == NULL) {
            available = entry;
        }
    }
    if (available == NULL) {
        return false;
    }
    available->occupied = true;
    available->expires_at = claims->expires_at;
    memcpy(available->nonce, claims->nonce, sizeof(available->nonce));
    return true;
}

static esp_err_t authorize(httpd_req_t* request, kidi_accessory_media_kind_t kind, const uint8_t parameter_hash[32],
                           kidi_ticket_claims_t& claims) {
    size_t authorization_size = httpd_req_get_hdr_value_len(request, "Authorization");
    if (authorization_size != strlen(KIDI_ACCESSORY_TICKET_AUTH_SCHEME) + 1 + KIDI_ACCESSORY_TICKET_BASE64URL_BYTES) {
        return ESP_ERR_INVALID_ARG;
    }
    char authorization[sizeof(KIDI_ACCESSORY_TICKET_AUTH_SCHEME) + KIDI_ACCESSORY_TICKET_BASE64URL_BYTES + 1];
    if (httpd_req_get_hdr_value_str(request, "Authorization", authorization, sizeof(authorization)) != ESP_OK ||
        strncmp(authorization, KIDI_ACCESSORY_TICKET_AUTH_SCHEME " ", strlen(KIDI_ACCESSORY_TICKET_AUTH_SCHEME) + 1) !=
            0) {
        return ESP_ERR_INVALID_ARG;
    }
    uint8_t ticket[KIDI_ACCESSORY_TICKET_BYTES];
    ESP_RETURN_ON_ERROR(kidi_ticket_decode_base64url(authorization + strlen(KIDI_ACCESSORY_TICKET_AUTH_SCHEME) + 1,
                                                     KIDI_ACCESSORY_TICKET_BASE64URL_BYTES, ticket),
                        TAG, "invalid ticket encoding");

    xSemaphoreTake(security_mutex, portMAX_DELAY);
    if (!security_ready) {
        xSemaphoreGive(security_mutex);
        mbedtls_platform_zeroize(ticket, sizeof(ticket));
        return ESP_ERR_NOT_FINISHED;
    }
    int64_t elapsed = esp_timer_get_time() - monotonic_at_sync;
    uint64_t now = epoch_at_sync + (elapsed > 0 ? (uint64_t)elapsed / 1000000U : 0);
    esp_err_t error =
        kidi_ticket_validate(active_ticket_key, ticket, sizeof(ticket), kind, now, parameter_hash, &claims);
    if (error == ESP_OK && !register_nonce(&claims, now)) {
        error = ESP_ERR_INVALID_STATE;
    }
    xSemaphoreGive(security_mutex);
    mbedtls_platform_zeroize(ticket, sizeof(ticket));
    return error;
}

static bool begin_request(httpd_req_t* request) {
    if (xSemaphoreTake(media_mutex, 0) == pdTRUE) {
        return true;
    }
    send_json_error(request, "409 Conflict", "media-busy");
    return false;
}

static esp_err_t photo_handler(httpd_req_t* request) {
    if (!begin_request(request)) {
        return ESP_OK;
    }
    esp_err_t result = ESP_FAIL;
    char body[KIDI_REQUEST_BODY_BYTES + 1];
    size_t body_size = 0;
    cJSON* json = NULL;
    kidi_camera_photo_t photo{};
    kidi_ticket_claims_t claims{};
    bool ticket_consumed = false;
    do {
        if (receive_body(request, body, &body_size) != ESP_OK) {
            result = send_json_error(request, "400 Bad Request", "invalid-body");
            break;
        }
        json = cJSON_ParseWithLength(body, body_size);
        uint8_t parameter_hash[32];
        size_t maximum_bytes = 0;
        if (json == NULL || canonical_photo_parameters(json, parameter_hash, &maximum_bytes) != ESP_OK) {
            result = send_json_error(request, "400 Bad Request", "invalid-photo-request");
            break;
        }
        esp_err_t authorization = authorize(request, KIDI_ACCESSORY_MEDIA_PHOTO, parameter_hash, claims);
        if (authorization == ESP_ERR_NOT_FINISHED) {
            result = send_json_error(request, "503 Service Unavailable", "security-not-ready");
            break;
        }
        if (authorization != ESP_OK) {
            result = send_json_error(request, "401 Unauthorized", "invalid-ticket");
            break;
        }
        ticket_consumed = true;
        if (kidi_camera_capture_photo(&photo) != ESP_OK) {
            result = send_json_error(request, "500 Internal Server Error", "camera-failed");
            break;
        }
        if (photo.size > maximum_bytes) {
            result = send_json_error(request, "413 Payload Too Large", "photo-too-large");
            break;
        }
        uint8_t digest[32];
        if (sha256(photo.data, photo.size, digest) != ESP_OK) {
            result = send_json_error(request, "500 Internal Server Error", "digest-failed");
            break;
        }
        char digest_hex[65];
        for (size_t index = 0; index < sizeof(digest); ++index) {
            snprintf(digest_hex + index * 2, 3, "%02x", digest[index]);
        }
        char width[12];
        char height[12];
        snprintf(width, sizeof(width), "%" PRIu32, photo.width);
        snprintf(height, sizeof(height), "%" PRIu32, photo.height);
        httpd_resp_set_type(request, KIDI_ACCESSORY_PHOTO_CONTENT_TYPE);
        httpd_resp_set_hdr(request, KIDI_ACCESSORY_PHOTO_DIGEST_HEADER, digest_hex);
        httpd_resp_set_hdr(request, KIDI_ACCESSORY_PHOTO_WIDTH_HEADER, width);
        httpd_resp_set_hdr(request, KIDI_ACCESSORY_PHOTO_HEIGHT_HEADER, height);
        result = httpd_resp_send(request, (const char*)photo.data, photo.size);
    } while (false);
    cJSON_Delete(json);
    kidi_camera_release_photo(&photo);
    if (ticket_consumed && completion_callback != nullptr) {
        completion_callback(std::span<const uint8_t, KIDI_ACCESSORY_TICKET_NONCE_BYTES>{claims.nonce});
    }
    xSemaphoreGive(media_mutex);
    return result;
}

static esp_err_t audio_sink(const int16_t* samples, size_t sample_count, void* context) {
    auto* sink = static_cast<audio_sink_context_t*>(context);
    esp_err_t error = httpd_resp_send_chunk(sink->request, (const char*)samples, sample_count * sizeof(int16_t));
    if (error == ESP_OK) {
        sink->sent_samples += sample_count;
    }
    return error;
}

static esp_err_t audio_handler(httpd_req_t* request) {
    if (!begin_request(request)) {
        return ESP_OK;
    }
    esp_err_t result = ESP_FAIL;
    char body[KIDI_REQUEST_BODY_BYTES + 1];
    size_t body_size = 0;
    cJSON* json = NULL;
    audio_sink_context_t sink{};
    sink.request = request;
    kidi_ticket_claims_t claims{};
    bool ticket_consumed = false;
    do {
        if (receive_body(request, body, &body_size) != ESP_OK) {
            result = send_json_error(request, "400 Bad Request", "invalid-body");
            break;
        }
        json = cJSON_ParseWithLength(body, body_size);
        uint8_t parameter_hash[32];
        uint32_t maximum_seconds = 0;
        if (json == NULL || canonical_audio_parameters(json, parameter_hash, &maximum_seconds) != ESP_OK) {
            result = send_json_error(request, "400 Bad Request", "invalid-audio-request");
            break;
        }
        esp_err_t authorization = authorize(request, KIDI_ACCESSORY_MEDIA_AUDIO, parameter_hash, claims);
        if (authorization == ESP_ERR_NOT_FINISHED) {
            result = send_json_error(request, "503 Service Unavailable", "security-not-ready");
            break;
        }
        if (authorization != ESP_OK) {
            result = send_json_error(request, "401 Unauthorized", "invalid-ticket");
            break;
        }
        ticket_consumed = true;
        char sample_rate[12];
        char channels[4];
        char bits[4];
        snprintf(sample_rate, sizeof(sample_rate), "%u", KIDI_ACCESSORY_AUDIO_SAMPLE_RATE);
        snprintf(channels, sizeof(channels), "%u", KIDI_ACCESSORY_AUDIO_CHANNELS);
        snprintf(bits, sizeof(bits), "%u", KIDI_ACCESSORY_AUDIO_BITS_PER_SAMPLE);
        httpd_resp_set_type(request, KIDI_ACCESSORY_AUDIO_CONTENT_TYPE);
        httpd_resp_set_hdr(request, KIDI_ACCESSORY_AUDIO_SAMPLE_RATE_HEADER, sample_rate);
        httpd_resp_set_hdr(request, KIDI_ACCESSORY_AUDIO_CHANNELS_HEADER, channels);
        httpd_resp_set_hdr(request, KIDI_ACCESSORY_AUDIO_BITS_HEADER, bits);
        kidi_audio_result_t capture;
        result = kidi_audio_stream(maximum_seconds, audio_sink, &sink, &capture);
        if (result == ESP_OK) {
            result = httpd_resp_send_chunk(request, NULL, 0);
        } else if (sink.sent_samples == 0) {
            result = send_json_error(request, "500 Internal Server Error", "microphone-failed");
        }
    } while (false);
    cJSON_Delete(json);
    if (ticket_consumed && completion_callback != nullptr) {
        completion_callback(std::span<const uint8_t, KIDI_ACCESSORY_TICKET_NONCE_BYTES>{claims.nonce});
    }
    xSemaphoreGive(media_mutex);
    return result;
}

namespace kidi::p4 {

auto start_media_server(const kidi_tls_identity_t& identity, CompletionCallback completion) -> esp_err_t {
    if (identity.certificate_pem == nullptr || identity.private_key_pem == nullptr || server != nullptr ||
        completion == nullptr) {
        return ESP_ERR_INVALID_ARG;
    }
    completion_callback = completion;
    media_mutex = xSemaphoreCreateMutex();
    security_mutex = xSemaphoreCreateMutex();
    if (media_mutex == NULL || security_mutex == NULL) {
        if (media_mutex != NULL) {
            vSemaphoreDelete(media_mutex);
            media_mutex = NULL;
        }
        if (security_mutex != NULL) {
            vSemaphoreDelete(security_mutex);
            security_mutex = NULL;
        }
        return ESP_ERR_NO_MEM;
    }
    httpd_ssl_config_t config = HTTPD_SSL_CONFIG_DEFAULT();
    config.port_secure = KIDI_ACCESSORY_MEDIA_PORT;
    config.servercert = identity.certificate_pem;
    config.servercert_len = identity.certificate_pem_size;
    config.prvtkey_pem = identity.private_key_pem;
    config.prvtkey_len = identity.private_key_pem_size;
    config.httpd.max_uri_handlers = 2;
    esp_err_t error = httpd_ssl_start(&server, &config);
    if (error != ESP_OK) {
        vSemaphoreDelete(media_mutex);
        vSemaphoreDelete(security_mutex);
        media_mutex = NULL;
        security_mutex = NULL;
        return error;
    }
    httpd_uri_t photo{};
    photo.uri = KIDI_ACCESSORY_PHOTO_PATH;
    photo.method = HTTP_POST;
    photo.handler = photo_handler;
    httpd_uri_t audio{};
    audio.uri = KIDI_ACCESSORY_AUDIO_PATH;
    audio.method = HTTP_POST;
    audio.handler = audio_handler;
    error = httpd_register_uri_handler(server, &photo);
    if (error == ESP_OK) {
        error = httpd_register_uri_handler(server, &audio);
    }
    if (error != ESP_OK) {
        httpd_ssl_stop(server);
        server = NULL;
        vSemaphoreDelete(media_mutex);
        vSemaphoreDelete(security_mutex);
        media_mutex = NULL;
        security_mutex = NULL;
        return error;
    }
    ESP_LOGI(TAG, "media TLS server listening on %u", KIDI_ACCESSORY_MEDIA_PORT);
    return ESP_OK;
}

auto configure_media_security(std::span<const std::uint8_t, KIDI_TICKET_KEY_BYTES> new_ticket_key,
                              std::uint64_t epoch_seconds) -> esp_err_t {
    if (media_mutex == nullptr || security_mutex == nullptr || epoch_seconds == 0) {
        return ESP_ERR_INVALID_ARG;
    }
    xSemaphoreTake(media_mutex, portMAX_DELAY);
    xSemaphoreTake(security_mutex, portMAX_DELAY);
    std::copy(new_ticket_key.begin(), new_ticket_key.end(), active_ticket_key);
    epoch_at_sync = epoch_seconds;
    monotonic_at_sync = esp_timer_get_time();
    memset(replay_entries, 0, sizeof(replay_entries));
    security_ready = true;
    xSemaphoreGive(security_mutex);
    xSemaphoreGive(media_mutex);
    return ESP_OK;
}

} // namespace kidi::p4
