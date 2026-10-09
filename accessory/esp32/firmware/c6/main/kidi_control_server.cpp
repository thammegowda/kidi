#include "kidi_control_server.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <memory>

#include "accessory_protocol.h"
#include "cJSON.h"
#include "esp_http_server.h"
#include "esp_https_server.h"
#include "esp_random.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "kidi_controller_store.h"
#include "mbedtls/base64.h"
#include "mbedtls/platform_util.h"
#include "mbedtls/sha256.h"

namespace {

constexpr std::size_t request_body_bytes = 512;
constexpr std::size_t controller_token_base64url_bytes = 43;
constexpr std::uint64_t photo_session_seconds = 20;
constexpr std::uint64_t audio_session_grace_seconds = 10;

struct JsonDeleter {
    void operator()(cJSON* value) const noexcept { cJSON_Delete(value); }
};

struct SensitiveController {
    kidi_controller_t value{};
    ~SensitiveController() { mbedtls_platform_zeroize(&value, sizeof(value)); }
};

using JsonPointer = std::unique_ptr<cJSON, JsonDeleter>;

httpd_handle_t server;
SemaphoreHandle_t session_mutex;
std::array<std::uint8_t, KIDI_TICKET_KEY_BYTES> ticket_key;
kidi::c6::Clock current_epoch;
bool media_session_active;
std::uint64_t media_session_deadline;
std::array<std::uint8_t, KIDI_ACCESSORY_TICKET_NONCE_BYTES> active_nonce;

auto send_json_error(httpd_req_t* request, const char* status, const char* reason) -> esp_err_t {
    std::array<char, 160> body{};
    const int size = snprintf(body.data(), body.size(), "{\"error\":\"%s\"}", reason);
    if (size <= 0 || static_cast<std::size_t>(size) >= body.size()) {
        return ESP_FAIL;
    }
    httpd_resp_set_status(request, status);
    httpd_resp_set_type(request, "application/json");
    return httpd_resp_send(request, body.data(), size);
}

auto receive_body(httpd_req_t* request, std::array<char, request_body_bytes + 1>& body, std::size_t& body_size)
    -> esp_err_t {
    if (request->content_len == 0 || request->content_len > request_body_bytes) {
        return ESP_ERR_INVALID_SIZE;
    }
    std::size_t received = 0;
    while (received < request->content_len) {
        const int count = httpd_req_recv(request, body.data() + received, request->content_len - received);
        if (count == HTTPD_SOCK_ERR_TIMEOUT) {
            continue;
        }
        if (count <= 0) {
            return ESP_FAIL;
        }
        received += static_cast<std::size_t>(count);
    }
    body[received] = '\0';
    body_size = received;
    return ESP_OK;
}

auto object_has_exact_keys(const cJSON* object, std::span<const char* const> keys) -> bool {
    std::size_t found = 0;
    const cJSON* item = nullptr;
    cJSON_ArrayForEach(item, object) {
        if (item->string == nullptr) {
            return false;
        }
        bool known = false;
        for (const char* key : keys) {
            if (std::strcmp(item->string, key) == 0) {
                known = true;
                break;
            }
        }
        if (!known) {
            return false;
        }
        ++found;
    }
    return found == keys.size();
}

auto sha256(std::span<const std::uint8_t> input, std::array<std::uint8_t, 32>& output) -> esp_err_t {
    mbedtls_sha256_context context;
    mbedtls_sha256_init(&context);
    int result = mbedtls_sha256_starts(&context, 0);
    if (result == 0) {
        result = mbedtls_sha256_update(&context, input.data(), input.size());
    }
    if (result == 0) {
        result = mbedtls_sha256_finish(&context, output.data());
    }
    mbedtls_sha256_free(&context);
    return result == 0 ? ESP_OK : ESP_FAIL;
}

auto decode_controller_token(const char* value, std::array<std::uint8_t, KIDI_CONTROLLER_TOKEN_BYTES>& output)
    -> esp_err_t {
    if (value == nullptr || std::strlen(value) != controller_token_base64url_bytes) {
        return ESP_ERR_INVALID_ARG;
    }
    std::array<std::uint8_t, 44> standard{};
    for (std::size_t index = 0; index < controller_token_base64url_bytes; ++index) {
        char byte = value[index];
        if (byte == '-') {
            byte = '+';
        } else if (byte == '_') {
            byte = '/';
        } else if (!((byte >= 'A' && byte <= 'Z') || (byte >= 'a' && byte <= 'z') || (byte >= '0' && byte <= '9'))) {
            return ESP_ERR_INVALID_ARG;
        }
        standard[index] = static_cast<std::uint8_t>(byte);
    }
    standard.back() = '=';
    std::size_t decoded_size = 0;
    const int result =
        mbedtls_base64_decode(output.data(), output.size(), &decoded_size, standard.data(), standard.size());
    return result == 0 && decoded_size == output.size() ? ESP_OK : ESP_ERR_INVALID_ARG;
}

auto authenticate(httpd_req_t* request, SensitiveController& controller) -> esp_err_t {
    constexpr std::size_t scheme_size = sizeof(KIDI_ACCESSORY_CONTROLLER_AUTH_SCHEME) - 1;
    constexpr std::size_t expected_size = scheme_size + 1 + controller_token_base64url_bytes;
    if (httpd_req_get_hdr_value_len(request, "Authorization") != expected_size) {
        return ESP_ERR_INVALID_ARG;
    }
    std::array<char, expected_size + 1> authorization{};
    if (httpd_req_get_hdr_value_str(request, "Authorization", authorization.data(), authorization.size()) != ESP_OK ||
        std::memcmp(authorization.data(), KIDI_ACCESSORY_CONTROLLER_AUTH_SCHEME " ", scheme_size + 1) != 0) {
        return ESP_ERR_INVALID_ARG;
    }
    std::array<std::uint8_t, KIDI_CONTROLLER_TOKEN_BYTES> token{};
    const esp_err_t decode = decode_controller_token(authorization.data() + scheme_size + 1, token);
    if (decode != ESP_OK) {
        return decode;
    }
    const esp_err_t result = kidi_controller_store_find_token(token.data(), &controller.value);
    mbedtls_platform_zeroize(token.data(), token.size());
    return result;
}

auto canonical_parameters(const cJSON* root, kidi_accessory_media_kind_t& kind,
                          std::array<std::uint8_t, 32>& parameter_hash, std::uint32_t& maximum_seconds) -> esp_err_t {
    static constexpr std::array root_keys{"kind", "parameters"};
    if (!cJSON_IsObject(root) || !object_has_exact_keys(root, root_keys)) {
        return ESP_ERR_INVALID_ARG;
    }
    const cJSON* kind_json = cJSON_GetObjectItemCaseSensitive(root, "kind");
    const cJSON* parameters = cJSON_GetObjectItemCaseSensitive(root, "parameters");
    if (!cJSON_IsString(kind_json) || !cJSON_IsObject(parameters)) {
        return ESP_ERR_INVALID_ARG;
    }
    if (std::strcmp(kind_json->valuestring, "photo") == 0) {
        static constexpr std::array photo_keys{"profile"};
        const cJSON* profile = cJSON_GetObjectItemCaseSensitive(parameters, "profile");
        if (!object_has_exact_keys(parameters, photo_keys) || !cJSON_IsString(profile) ||
            std::strcmp(profile->valuestring, "chat") != 0) {
            return ESP_ERR_INVALID_ARG;
        }
        static constexpr char canonical[] = "{\"profile\":\"chat\"}";
        kind = KIDI_ACCESSORY_MEDIA_PHOTO;
        maximum_seconds = 0;
        return sha256(std::span{reinterpret_cast<const std::uint8_t*>(canonical), sizeof(canonical) - 1},
                      parameter_hash);
    }
    if (std::strcmp(kind_json->valuestring, "audio") == 0) {
        static constexpr std::array audio_keys{"maximum_seconds"};
        const cJSON* maximum = cJSON_GetObjectItemCaseSensitive(parameters, "maximum_seconds");
        if (!object_has_exact_keys(parameters, audio_keys) || !cJSON_IsNumber(maximum) ||
            maximum->valuedouble != maximum->valueint || maximum->valueint <= 0 ||
            maximum->valueint > KIDI_ACCESSORY_MAX_AUDIO_SECONDS) {
            return ESP_ERR_INVALID_ARG;
        }
        std::array<char, 48> canonical{};
        const int size = snprintf(canonical.data(), canonical.size(), "{\"maximum_seconds\":%d}", maximum->valueint);
        if (size <= 0 || static_cast<std::size_t>(size) >= canonical.size()) {
            return ESP_FAIL;
        }
        kind = KIDI_ACCESSORY_MEDIA_AUDIO;
        maximum_seconds = static_cast<std::uint32_t>(maximum->valueint);
        return sha256(
            std::span{reinterpret_cast<const std::uint8_t*>(canonical.data()), static_cast<std::size_t>(size)},
            parameter_hash);
    }
    return ESP_ERR_INVALID_ARG;
}

auto media_ticket_handler(httpd_req_t* request) -> esp_err_t {
    SensitiveController controller;
    if (authenticate(request, controller) != ESP_OK) {
        return send_json_error(request, "401 Unauthorized", "invalid-controller");
    }
    std::array<char, request_body_bytes + 1> body{};
    std::size_t body_size = 0;
    if (receive_body(request, body, body_size) != ESP_OK) {
        return send_json_error(request, "400 Bad Request", "invalid-body");
    }
    JsonPointer root{cJSON_ParseWithLength(body.data(), body_size)};
    kidi_accessory_media_kind_t kind{};
    std::array<std::uint8_t, 32> parameter_hash{};
    std::uint32_t maximum_seconds = 0;
    if (root == nullptr || canonical_parameters(root.get(), kind, parameter_hash, maximum_seconds) != ESP_OK) {
        return send_json_error(request, "400 Bad Request", "invalid-ticket-request");
    }
    const std::uint64_t now = current_epoch == nullptr ? 0 : current_epoch();
    if (now == 0) {
        return send_json_error(request, "503 Service Unavailable", "clock-not-ready");
    }

    xSemaphoreTake(session_mutex, portMAX_DELAY);
    if (media_session_active && now <= media_session_deadline) {
        xSemaphoreGive(session_mutex);
        return send_json_error(request, "409 Conflict", "media-busy");
    }
    media_session_active = false;
    active_nonce.fill(0);

    std::array<std::uint8_t, KIDI_ACCESSORY_TICKET_NONCE_BYTES> nonce{};
    esp_fill_random(nonce.data(), nonce.size());
    std::array<std::uint8_t, KIDI_ACCESSORY_TICKET_BYTES> ticket{};
    esp_err_t error = kidi_ticket_issue(ticket_key.data(), kind, now, nonce.data(), controller.value.id,
                                        parameter_hash.data(), ticket.data());
    std::array<char, KIDI_ACCESSORY_TICKET_BASE64URL_BYTES + 1> encoded_ticket{};
    if (error == ESP_OK) {
        error = kidi_ticket_encode_base64url(ticket.data(), encoded_ticket.data());
    }
    mbedtls_platform_zeroize(ticket.data(), ticket.size());
    if (error != ESP_OK) {
        xSemaphoreGive(session_mutex);
        return send_json_error(request, "500 Internal Server Error", "ticket-failed");
    }
    media_session_active = true;
    active_nonce = nonce;
    media_session_deadline = now + (kind == KIDI_ACCESSORY_MEDIA_AUDIO ? maximum_seconds + audio_session_grace_seconds
                                                                       : photo_session_seconds);
    xSemaphoreGive(session_mutex);

    std::array<char, 256> response{};
    const int response_size = snprintf(response.data(), response.size(), "{\"protocol\":%u,\"ticket\":\"%s\"}",
                                       KIDI_ACCESSORY_PROTOCOL_VERSION, encoded_ticket.data());
    if (response_size <= 0 || static_cast<std::size_t>(response_size) >= response.size()) {
        kidi::c6::complete_media_session(nonce);
        return send_json_error(request, "500 Internal Server Error", "ticket-response-failed");
    }
    httpd_resp_set_type(request, "application/json");
    error = httpd_resp_send(request, response.data(), response_size);
    if (error != ESP_OK) {
        kidi::c6::complete_media_session(nonce);
    }
    return error;
}

auto status_handler(httpd_req_t* request) -> esp_err_t {
    const std::uint64_t now = current_epoch == nullptr ? 0 : current_epoch();
    xSemaphoreTake(session_mutex, portMAX_DELAY);
    if (media_session_active && now > media_session_deadline) {
        media_session_active = false;
        active_nonce.fill(0);
    }
    const bool busy = media_session_active;
    xSemaphoreGive(session_mutex);
    std::array<char, 160> response{};
    const int size = snprintf(response.data(), response.size(),
                              "{\"protocol\":%u,\"controllers\":%u,\"media_busy\":%s,"
                              "\"clock_ready\":%s}",
                              KIDI_ACCESSORY_PROTOCOL_VERSION, static_cast<unsigned>(kidi_controller_store_count()),
                              busy ? "true" : "false", now != 0 ? "true" : "false");
    if (size <= 0 || static_cast<std::size_t>(size) >= response.size()) {
        return ESP_FAIL;
    }
    httpd_resp_set_type(request, "application/json");
    return httpd_resp_send(request, response.data(), size);
}

auto unpair_self_handler(httpd_req_t* request) -> esp_err_t {
    SensitiveController controller;
    if (authenticate(request, controller) != ESP_OK) {
        return send_json_error(request, "401 Unauthorized", "invalid-controller");
    }
    const esp_err_t error = kidi_controller_store_remove(controller.value.id);
    if (error != ESP_OK) {
        return send_json_error(request, "500 Internal Server Error", "unpair-failed");
    }
    httpd_resp_set_status(request, "204 No Content");
    return httpd_resp_send(request, nullptr, 0);
}

} // namespace

namespace kidi::c6 {

auto start_control_server(const kidi_tls_identity_t& identity,
                          std::span<const std::uint8_t, KIDI_TICKET_KEY_BYTES> new_ticket_key, Clock clock)
    -> esp_err_t {
    if (server != nullptr || identity.certificate_pem == nullptr || identity.private_key_pem == nullptr ||
        clock == nullptr) {
        return ESP_ERR_INVALID_ARG;
    }
    session_mutex = xSemaphoreCreateMutex();
    if (session_mutex == nullptr) {
        return ESP_ERR_NO_MEM;
    }
    ticket_key.fill(0);
    std::copy(new_ticket_key.begin(), new_ticket_key.end(), ticket_key.begin());
    current_epoch = clock;

    httpd_ssl_config_t config = HTTPD_SSL_CONFIG_DEFAULT();
    config.port_secure = KIDI_ACCESSORY_CONTROL_PORT;
    config.servercert = identity.certificate_pem;
    config.servercert_len = identity.certificate_pem_size;
    config.prvtkey_pem = identity.private_key_pem;
    config.prvtkey_len = identity.private_key_pem_size;
    config.httpd.max_uri_handlers = 3;
    esp_err_t error = httpd_ssl_start(&server, &config);
    if (error != ESP_OK) {
        vSemaphoreDelete(session_mutex);
        session_mutex = nullptr;
        return error;
    }
    std::array<httpd_uri_t, 3> handlers{};
    handlers[0].uri = KIDI_ACCESSORY_MEDIA_TICKET_PATH;
    handlers[0].method = HTTP_POST;
    handlers[0].handler = media_ticket_handler;
    handlers[1].uri = KIDI_ACCESSORY_STATUS_PATH;
    handlers[1].method = HTTP_GET;
    handlers[1].handler = status_handler;
    handlers[2].uri = KIDI_ACCESSORY_CONTROLLERS_SELF_PATH;
    handlers[2].method = HTTP_DELETE;
    handlers[2].handler = unpair_self_handler;
    for (const auto& handler : handlers) {
        error = httpd_register_uri_handler(server, &handler);
        if (error != ESP_OK) {
            httpd_ssl_stop(server);
            server = nullptr;
            vSemaphoreDelete(session_mutex);
            session_mutex = nullptr;
            mbedtls_platform_zeroize(ticket_key.data(), ticket_key.size());
            return error;
        }
    }
    return ESP_OK;
}

void complete_media_session(std::span<const std::uint8_t, KIDI_ACCESSORY_TICKET_NONCE_BYTES> nonce) {
    if (session_mutex == nullptr) {
        return;
    }
    xSemaphoreTake(session_mutex, portMAX_DELAY);
    if (media_session_active && std::equal(nonce.begin(), nonce.end(), active_nonce.begin())) {
        media_session_active = false;
        media_session_deadline = 0;
        active_nonce.fill(0);
    }
    xSemaphoreGive(session_mutex);
}

} // namespace kidi::c6
