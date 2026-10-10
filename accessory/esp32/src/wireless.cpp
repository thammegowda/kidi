#include "wireless.h"
#include "camera.h"
#include "clip.h"
#include "stream.h"
#include "audio.h"
#include "esp32.h"

#include <array>
#include <algorithm>
#include <cctype>
#include <cstring>
#include <cmath>
#include <memory>

#include <Preferences.h>
#include <ESPmDNS.h>
#include <WiFi.h>
#include <cJSON.h>
#include <esp_https_server.h>
#include <mbedtls/sha256.h>
#include <cstdlib>
#include <esp_heap_caps.h>
#include <lwip/sockets.h>

extern "C" {
#include <mbedtls/constant_time.h>
}

namespace kidi::esp32 {
namespace {

constexpr std::uint16_t HTTPS_PORT = 8443;
constexpr std::size_t MAX_PHOTO_BYTES = 1024 * 1024;
constexpr std::size_t MAX_REQUEST_BYTES = 128;
constexpr std::uint32_t JOIN_TIMEOUT_MS = 20000;
using Json = std::unique_ptr<cJSON, decltype(&cJSON_Delete)>;

struct Configuration {
    String device_id;
    String certificate;
    String private_key;
    String token_hash;
    String ssid;
    String password;

    auto configured() const -> bool { return !device_id.isEmpty(); }
    auto hostname() const -> String { return "kidi-" + device_id + ".local"; }
};

Configuration configuration;
httpd_handle_t server = nullptr;
bool reconnect_reported = false;

auto make_json() -> Json { return {cJSON_CreateObject(), cJSON_Delete}; }

auto json_string(const cJSON* object, const char* name) -> String {
    const auto* value = cJSON_GetObjectItemCaseSensitive(object, name);
    return cJSON_IsString(value) && value->valuestring != nullptr ? String(value->valuestring) : String();
}

auto print_reply(cJSON* object) -> void {
    auto* text = cJSON_PrintUnformatted(object);
    if (text == nullptr) {
        Serial.println("KIDI_WIFI_REPLY {\"ok\":false,\"error\":\"response allocation failed\"}");
        return;
    }
    Serial.print("KIDI_WIFI_REPLY ");
    Serial.println(text);
    cJSON_free(text);
}

auto reply_error(const char* error, const char* code = "SETUP_FAILED") -> void {
    auto response = make_json();
    if (!response) {
        Serial.println("KIDI_WIFI_REPLY {\"ok\":false,\"error\":\"response allocation failed\"}");
        return;
    }
    cJSON_AddBoolToObject(response.get(), "ok", false);
    cJSON_AddStringToObject(response.get(), "error", error);
    cJSON_AddStringToObject(response.get(), "code", code);
    print_reply(response.get());
}

auto hex_digest(const String& value) -> String {
    std::array<unsigned char, 32> digest{};
    if (mbedtls_sha256_ret(reinterpret_cast<const unsigned char*>(value.c_str()), value.length(), digest.data(), 0) !=
        0)
        return {};
    char output[65]{};
    for (std::size_t index = 0; index < digest.size(); ++index) {
        snprintf(output + index * 2, 3, "%02x", digest[index]);
    }
    return String(output);
}

auto valid_hex(const String& value, std::size_t length) -> bool {
    if (value.length() != length) return false;
    for (unsigned index = 0; index < value.length(); ++index) {
        if (!std::isxdigit(static_cast<unsigned char>(value[index]))) return false;
    }
    return true;
}

auto authorized_token(const String& token) -> bool {
    if (!configuration.configured() || !valid_hex(token, 64)) return false;
    const auto digest = hex_digest(token);
    return digest.length() == 64 && mbedtls_ct_memcmp(digest.c_str(), configuration.token_hash.c_str(), 64) == 0;
}

auto load_configuration(const cJSON* object) -> Configuration {
    return {json_string(object, "device_id"),  json_string(object, "certificate"), json_string(object, "private_key"),
            json_string(object, "token_hash"), json_string(object, "ssid"),        json_string(object, "password")};
}

auto configuration_json(const Configuration& value) -> Json {
    auto object = make_json();
    if (!object) return object;
    cJSON_AddStringToObject(object.get(), "device_id", value.device_id.c_str());
    cJSON_AddStringToObject(object.get(), "certificate", value.certificate.c_str());
    cJSON_AddStringToObject(object.get(), "private_key", value.private_key.c_str());
    cJSON_AddStringToObject(object.get(), "token_hash", value.token_hash.c_str());
    cJSON_AddStringToObject(object.get(), "ssid", value.ssid.c_str());
    cJSON_AddStringToObject(object.get(), "password", value.password.c_str());
    return object;
}

auto save_configuration() -> bool {
    auto object = configuration_json(configuration);
    if (!object) return false;
    auto* text = cJSON_PrintUnformatted(object.get());
    if (text == nullptr) return false;
    Preferences preferences;
    auto saved = false;
    if (preferences.begin("kidi-wifi", false)) {
        saved = preferences.putString("configuration", text) == std::strlen(text);
        preferences.end();
    }
    cJSON_free(text);
    return saved;
}

auto add_network_status(cJSON* object) -> void {
    cJSON_AddNumberToObject(object, "rssi_dbm", WiFi.RSSI());
    cJSON_AddStringToObject(object, "bssid", WiFi.BSSIDstr().c_str());
    cJSON_AddNumberToObject(object, "channel", WiFi.channel());
    WifiPowerSaveMode mode;
    const auto result = esp_wifi_get_ps(&mode);
    if (result == ESP_OK)
        cJSON_AddNumberToObject(object, "power_save_mode", mode);
    else
        cJSON_AddStringToObject(object, "power_save_error", esp_err_to_name(result));
}

auto reply_info() -> void {
    auto response = make_json();
    if (!response) {
        reply_error("response allocation failed");
        return;
    }
    cJSON_AddBoolToObject(response.get(), "ok", true);
    cJSON_AddNumberToObject(response.get(), "protocol", 1);
    cJSON_AddStringToObject(response.get(), "hardware_id", WiFi.macAddress().c_str());
    cJSON_AddBoolToObject(response.get(), "configured", configuration.configured());
    cJSON_AddBoolToObject(response.get(), "camera_configured", camera_configured());
    cJSON_AddBoolToObject(response.get(), "microphone_configured", audio_configured());
    cJSON_AddStringToObject(response.get(), "device_id", configuration.device_id.c_str());
    cJSON_AddStringToObject(response.get(), "hostname", configuration.hostname().c_str());
    cJSON_AddStringToObject(response.get(), "certificate", configuration.certificate.c_str());
    cJSON_AddStringToObject(response.get(), "ssid_hash", hex_digest(configuration.ssid).c_str());
    cJSON_AddBoolToObject(response.get(), "connected", WiFi.status() == WL_CONNECTED);
    cJSON_AddBoolToObject(response.get(), "ready", server != nullptr && WiFi.status() == WL_CONNECTED);
    cJSON_AddStringToObject(response.get(), "address", WiFi.localIP().toString().c_str());
    cJSON_AddNumberToObject(response.get(), "port", HTTPS_PORT);
    add_network_status(response.get());
    print_reply(response.get());
}

auto send_error(httpd_req_t* request, const char* status, const char* message) -> Error {
    httpd_resp_set_status(request, status);
    httpd_resp_set_type(request, "application/json");
    auto object = make_json();
    if (!object) return ESP_ERR_NO_MEM;
    cJSON_AddStringToObject(object.get(), "error", message);
    auto* text = cJSON_PrintUnformatted(object.get());
    if (text == nullptr) return ESP_ERR_NO_MEM;
    const auto result = httpd_resp_send(request, text, HTTPD_RESP_USE_STRLEN);
    cJSON_free(text);
    return result;
}

auto authorize(httpd_req_t* request) -> bool {
    const auto length = httpd_req_get_hdr_value_len(request, "Authorization");
    if (length != 71) return false;
    std::array<char, 72> header{};
    if (httpd_req_get_hdr_value_str(request, "Authorization", header.data(), header.size()) != ESP_OK) return false;
    return std::strncmp(header.data(), "Bearer ", 7) == 0 && authorized_token(String(header.data() + 7));
}

auto handle_status(httpd_req_t* request) -> Error {
    if (!authorize(request)) return send_error(request, "401 Unauthorized", "authentication required");
    auto object = make_json();
    if (!object) return ESP_ERR_NO_MEM;
    cJSON_AddNumberToObject(object.get(), "protocol", 1);
    cJSON_AddStringToObject(object.get(), "device_id", configuration.device_id.c_str());
    cJSON_AddStringToObject(object.get(), "transport", "wifi");
    cJSON_AddStringToObject(object.get(), "firmware", "kidi-esp32-media-reference-1");
    cJSON_AddBoolToObject(object.get(), "camera_configured", camera_configured());
    cJSON_AddBoolToObject(object.get(), "microphone_configured", audio_configured());
    cJSON_AddNumberToObject(object.get(), "max_clip_seconds", 10);
    cJSON_AddNumberToObject(object.get(), "max_stream_seconds", 300);
    cJSON_AddBoolToObject(object.get(), "stream_experimental", true);
    add_network_status(object.get());
    auto* text = cJSON_PrintUnformatted(object.get());
    if (text == nullptr) return ESP_ERR_NO_MEM;
    httpd_resp_set_type(request, "application/json");
    const auto result = httpd_resp_send(request, text, HTTPD_RESP_USE_STRLEN);
    cJSON_free(text);
    return result;
}

struct PhotoResponse {
    httpd_req_t* request;
    bool started = false;
};

auto write_wifi_photo(const PhotoFrame& frame, void* context) -> Error {
    auto& response = *static_cast<PhotoResponse*>(context);
    if (frame.size == 0 || frame.size > MAX_PHOTO_BYTES) return ESP_ERR_INVALID_SIZE;
    char width[16]{};
    char height[16]{};
    snprintf(width, sizeof(width), "%u", static_cast<unsigned>(frame.width));
    snprintf(height, sizeof(height), "%u", static_cast<unsigned>(frame.height));
    httpd_resp_set_type(response.request, "image/jpeg");
    httpd_resp_set_hdr(response.request, "Cache-Control", "no-store");
    httpd_resp_set_hdr(response.request, "X-Kidi-Width", width);
    httpd_resp_set_hdr(response.request, "X-Kidi-Height", height);
    httpd_resp_set_hdr(response.request, "X-Kidi-Transport", "wifi");
    response.started = true;
    return httpd_resp_send(response.request, reinterpret_cast<const char*>(frame.data), frame.size);
}

auto handle_photo(httpd_req_t* request) -> Error {
    if (!authorize(request)) return send_error(request, "401 Unauthorized", "authentication required");
    if (request->content_len <= 0 || request->content_len > MAX_REQUEST_BYTES)
        return send_error(request, "400 Bad Request", "invalid photo request");
    std::array<char, MAX_REQUEST_BYTES + 1> body{};
    std::size_t received = 0;
    while (received < request->content_len) {
        const auto count = httpd_req_recv(request, body.data() + received, request->content_len - received);
        if (count <= 0) return send_error(request, "400 Bad Request", "incomplete photo request");
        received += count;
    }
    Json object(cJSON_ParseWithLength(body.data(), received), cJSON_Delete);
    if (!object) return send_error(request, "400 Bad Request", "invalid JSON");
    const auto resolution = json_string(object.get(), "resolution");
    CameraProfile profile;
    if (resolution == "2048x1536") {
        profile = CameraProfile::PHOTO_FULL;
    } else if (resolution == "640x480") {
        profile = CameraProfile::PHOTO_VGA;
    } else {
        return send_error(request, "400 Bad Request", "unsupported photo resolution");
    }
    if (!camera_configured()) return send_error(request, "501 Not Implemented", "camera backend is not configured");
    PhotoResponse response{request};
    const auto result = capture_photo_to(profile, write_wifi_photo, &response);
    if (result == ESP_OK) return ESP_OK;
    report_error("Wi-Fi photo", result);
    if (response.started) return ESP_FAIL;
    return send_error(request, result == ESP_ERR_TIMEOUT ? "409 Conflict" : "500 Internal Server Error",
                      result == ESP_ERR_TIMEOUT ? "camera busy" : "camera capture failed");
}

struct ClipResponse {
    static constexpr std::size_t BUFFER_BYTES = AUDIO_SAMPLE_RATE * 10 * sizeof(std::int16_t) + 4096;
    std::unique_ptr<std::uint8_t, decltype(&std::free)> buffer{
        static_cast<std::uint8_t*>(heap_caps_malloc(BUFFER_BYTES, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT)), std::free};
    std::size_t buffered = 0;
};

auto write_wifi_clip(const std::uint8_t* data, std::size_t length, void* context) -> Error {
    auto& response = *static_cast<ClipResponse*>(context);
    if (length > ClipResponse::BUFFER_BYTES - response.buffered) return ESP_ERR_INVALID_SIZE;
    std::memcpy(response.buffer.get() + response.buffered, data, length);
    response.buffered += length;
    return ESP_OK;
}

auto handle_clip(httpd_req_t* request) -> Error {
    if (!authorize(request)) return send_error(request, "401 Unauthorized", "authentication required");
    if (request->content_len <= 0 || request->content_len > MAX_REQUEST_BYTES)
        return send_error(request, "400 Bad Request", "invalid clip request");
    std::array<char, MAX_REQUEST_BYTES + 1> body{};
    std::size_t received = 0;
    while (received < request->content_len) {
        const auto count = httpd_req_recv(request, body.data() + received, request->content_len - received);
        if (count <= 0) return send_error(request, "400 Bad Request", "incomplete clip request");
        received += count;
    }
    Json object(cJSON_ParseWithLength(body.data(), received), cJSON_Delete);
    const auto* duration = object ? cJSON_GetObjectItemCaseSensitive(object.get(), "seconds") : nullptr;
    if (!cJSON_IsNumber(duration) || !std::isfinite(duration->valuedouble) || duration->valuedouble < 1 ||
        duration->valuedouble > 10 || duration->valuedouble != duration->valueint)
        return send_error(request, "400 Bad Request", "clip duration must be an integer from 1 to 10");
    if (!audio_configured()) return send_error(request, "501 Not Implemented", "microphone backend is not configured");
    httpd_resp_set_type(request, "application/vnd.kidi.capture");
    httpd_resp_set_hdr(request, "Cache-Control", "no-store");
    httpd_resp_set_hdr(request, "X-Kidi-Transport", "wifi");
    httpd_resp_set_hdr(request, "X-Kidi-Capture-Protocol", "1");
    // Record bounded clips before TLS transfer so network stalls cannot slow acquisition.
    ClipResponse response;
    if (!response.buffer)
        return send_error(request, "500 Internal Server Error", "capture transfer buffer allocation failed");
    const auto result =
        capture_clip(ClipKind::AUDIO, static_cast<unsigned>(duration->valueint), write_wifi_clip, &response);
    if (result == ESP_OK) {
        return httpd_resp_send(request, reinterpret_cast<const char*>(response.buffer.get()), response.buffered);
    }
    report_error("Wi-Fi clip", result);
    if (result == ESP_ERR_INVALID_SIZE)
        return send_error(request, "413 Payload Too Large", "clip exceeded PSRAM capacity; use a shorter duration");
    return send_error(request, result == ESP_ERR_TIMEOUT ? "409 Conflict" : "500 Internal Server Error",
                      result == ESP_ERR_TIMEOUT ? "capture busy" : "clip capture failed");
}

struct StreamResponse {
    httpd_req_t* request;
    bool started = false;
};

auto write_wifi_stream(const std::uint8_t* data, std::size_t length, void* context) -> Error {
    auto& response = *static_cast<StreamResponse*>(context);
    response.started = true;
    return httpd_resp_send_chunk(response.request, reinterpret_cast<const char*>(data), length);
}

auto handle_stream(httpd_req_t* request) -> Error {
    if (!authorize(request)) return send_error(request, "401 Unauthorized", "authentication required");
    if (request->content_len <= 0 || request->content_len > MAX_REQUEST_BYTES)
        return send_error(request, "400 Bad Request", "invalid stream request");
    std::array<char, MAX_REQUEST_BYTES + 1> body{};
    std::size_t received = 0;
    while (received < request->content_len) {
        const auto count = httpd_req_recv(request, body.data() + received, request->content_len - received);
        if (count <= 0) return send_error(request, "400 Bad Request", "incomplete stream request");
        received += count;
    }
    Json object(cJSON_ParseWithLength(body.data(), received), cJSON_Delete);
    const auto mode = object ? json_string(object.get(), "mode") : String();
    const auto* duration = object ? cJSON_GetObjectItemCaseSensitive(object.get(), "seconds") : nullptr;
    const auto* requested_resolution = object ? cJSON_GetObjectItemCaseSensitive(object.get(), "resolution") : nullptr;
    const auto stream_resolution = requested_resolution ? json_string(object.get(), "resolution") : String("1280x720");
    if ((mode != "video" && mode != "av") || !cJSON_IsNumber(duration) || !std::isfinite(duration->valuedouble) ||
        duration->valuedouble < 1 || duration->valuedouble > 300 || duration->valuedouble != duration->valueint)
        return send_error(request, "400 Bad Request", "invalid mode or stream duration (1-300 seconds)");
    if (stream_resolution != "1280x720" && stream_resolution != "96x96")
        return send_error(request, "400 Bad Request", "stream resolution must be 1280x720 or 96x96");
    if (!camera_configured()) return send_error(request, "501 Not Implemented", "camera backend is not configured");
    if (mode == "av" && !audio_configured())
        return send_error(request, "501 Not Implemented", "microphone backend is not configured");
    const int no_delay = 1;
    if (setsockopt(httpd_req_to_sockfd(request), IPPROTO_TCP, TCP_NODELAY, &no_delay, sizeof(no_delay)) != 0) {
        report_error("live socket TCP_NODELAY", ESP_FAIL);
        return send_error(request, "500 Internal Server Error", "stream socket setup failed");
    }
    WifiPowerSaveMode previous_power_save;
    auto power_result = esp_wifi_get_ps(&previous_power_save);
    if (power_result == ESP_OK) power_result = esp_wifi_set_ps(WIFI_PS_NONE);
    if (power_result != ESP_OK) {
        report_error("live Wi-Fi power setup", power_result);
        return send_error(request, "500 Internal Server Error", "stream Wi-Fi setup failed");
    }
    httpd_resp_set_type(request, "application/vnd.kidi.live");
    httpd_resp_set_hdr(request, "Cache-Control", "no-store");
    httpd_resp_set_hdr(request, "X-Kidi-Transport", "wifi");
    httpd_resp_set_hdr(request, "X-Kidi-Stream-Protocol", "1");
    StreamResponse response{request};
    auto result = stream_media(mode == "av", static_cast<unsigned>(duration->valueint), write_wifi_stream, &response,
                               stream_resolution == "96x96" ? CameraProfile::VIDEO_LOW : CameraProfile::VIDEO_HD,
                               MAX_PHOTO_BYTES);
    power_result = esp_wifi_set_ps(previous_power_save);
    if (power_result != ESP_OK) {
        report_error("live Wi-Fi power restore", power_result);
        if (result == ESP_OK) result = power_result;
    }
    if (result == ESP_OK) return httpd_resp_send_chunk(request, nullptr, 0);
    report_error("Wi-Fi live stream", result);
    if (response.started) return ESP_FAIL;
    return send_error(request, result == ESP_ERR_TIMEOUT ? "409 Conflict" : "500 Internal Server Error",
                      result == ESP_ERR_TIMEOUT ? "capture busy" : "stream startup failed");
}

struct SpeedParameters {
    std::size_t bytes = 0;
    std::size_t block_size = 0;
};

auto speed_parameters(httpd_req_t* request, SpeedParameters& parameters) -> bool {
    std::array<char, 96> query{};
    if (httpd_req_get_url_query_str(request, query.data(), query.size()) != ESP_OK) return false;
    const auto read_number = [&](const char* key, std::size_t& output) {
        std::array<char, 12> value{};
        if (httpd_query_key_value(query.data(), key, value.data(), value.size()) != ESP_OK || value[0] == '\0')
            return false;
        output = 0;
        for (const auto character : value) {
            if (character == '\0') break;
            if (character < '0' || character > '9' || output > 1024 * 1024) return false;
            output = output * 10 + character - '0';
        }
        return true;
    };
    return read_number("bytes", parameters.bytes) && read_number("block_size", parameters.block_size) &&
           parameters.bytes >= 4096 && parameters.bytes <= 1024 * 1024 && parameters.block_size >= 256 &&
           parameters.block_size <= 16384;
}

auto handle_speed_download(httpd_req_t* request) -> Error {
    if (!authorize(request)) return send_error(request, "401 Unauthorized", "authentication required");
    SpeedParameters parameters;
    if (!speed_parameters(request, parameters))
        return send_error(request, "400 Bad Request", "invalid speed-test byte/block limits");
    const auto acquired = acquire_capture();
    if (acquired != ESP_OK) return send_error(request, "409 Conflict", "capture or speed test busy");
    std::unique_ptr<std::uint8_t, decltype(&std::free)> buffer(
        static_cast<std::uint8_t*>(heap_caps_malloc(parameters.block_size, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT)),
        std::free);
    if (!buffer) {
        release_capture();
        return send_error(request, "500 Internal Server Error", "speed-test buffer allocation failed");
    }
    httpd_resp_set_type(request, "application/octet-stream");
    httpd_resp_set_hdr(request, "Cache-Control", "no-store");
    httpd_resp_set_hdr(request, "X-Kidi-Speed-Protocol", "1");
    char signal[16]{};
    snprintf(signal, sizeof(signal), "%d", WiFi.RSSI());
    httpd_resp_set_hdr(request, "X-Kidi-RSSI", signal);
    const auto start = esp_timer_get_time();
    auto result = ESP_OK;
    for (std::size_t offset = 0; offset < parameters.bytes && result == ESP_OK;) {
        if (esp_timer_get_time() - start > 60000000) {
            result = ESP_ERR_TIMEOUT;
            break;
        }
        const auto length = std::min(parameters.block_size, parameters.bytes - offset);
        for (std::size_t index = 0; index < length; ++index)
            buffer.get()[index] = static_cast<std::uint8_t>((offset + index) & 0xff);
        result = httpd_resp_send_chunk(request, reinterpret_cast<const char*>(buffer.get()), length);
        offset += length;
    }
    if (result == ESP_OK) result = httpd_resp_send_chunk(request, nullptr, 0);
    release_capture();
    if (result != ESP_OK) report_error("Wi-Fi download speed test", result);
    return result;
}

auto handle_speed_upload(httpd_req_t* request) -> Error {
    if (!authorize(request)) return send_error(request, "401 Unauthorized", "authentication required");
    SpeedParameters parameters;
    if (!speed_parameters(request, parameters) || request->content_len != parameters.bytes)
        return send_error(request, "400 Bad Request", "invalid speed-test payload/limits");
    const auto acquired = acquire_capture();
    if (acquired != ESP_OK) return send_error(request, "409 Conflict", "capture or speed test busy");
    std::unique_ptr<std::uint8_t, decltype(&std::free)> buffer(
        static_cast<std::uint8_t*>(heap_caps_malloc(parameters.block_size, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT)),
        std::free);
    if (!buffer) {
        release_capture();
        return send_error(request, "500 Internal Server Error", "speed-test buffer allocation failed");
    }
    const auto start = esp_timer_get_time();
    std::size_t received = 0;
    auto result = ESP_OK;
    while (received < parameters.bytes) {
        if (esp_timer_get_time() - start > 60000000) {
            result = ESP_ERR_TIMEOUT;
            break;
        }
        const auto count = httpd_req_recv(request, reinterpret_cast<char*>(buffer.get()),
                                          std::min(parameters.block_size, parameters.bytes - received));
        if (count <= 0) {
            result = ESP_FAIL;
            break;
        }
        for (int index = 0; index < count; ++index) {
            if (buffer.get()[index] != static_cast<std::uint8_t>((received + index) & 0xff)) {
                result = ESP_ERR_INVALID_RESPONSE;
                break;
            }
        }
        if (result != ESP_OK) break;
        received += count;
    }
    const auto elapsed = esp_timer_get_time() - start;
    release_capture();
    if (result != ESP_OK) {
        report_error("Wi-Fi upload speed test", result);
        return send_error(request, "400 Bad Request", "incomplete or corrupt speed-test upload");
    }
    char response[256]{};
    snprintf(
        response, sizeof(response), "{\"protocol\":1,\"bytes\":%u,\"elapsed_us\":%llu,\"rssi_dbm\":%d,\"cpu_mhz\":%u}",
        static_cast<unsigned>(received), static_cast<unsigned long long>(elapsed), WiFi.RSSI(), getCpuFrequencyMhz());
    httpd_resp_set_type(request, "application/json");
    return httpd_resp_send(request, response, HTTPD_RESP_USE_STRLEN);
}

auto start_server() -> Error {
    if (server != nullptr) return ESP_OK;
    httpd_ssl_config_t settings = HTTPD_SSL_CONFIG_DEFAULT();
    settings.port_secure = HTTPS_PORT;
    settings.httpd.max_open_sockets = 2;
    settings.httpd.stack_size = 12288;
    settings.httpd.send_wait_timeout = 15;
    settings.cacert_pem = reinterpret_cast<const std::uint8_t*>(configuration.certificate.c_str());
    settings.cacert_len = configuration.certificate.length() + 1;
    settings.prvtkey_pem = reinterpret_cast<const std::uint8_t*>(configuration.private_key.c_str());
    settings.prvtkey_len = configuration.private_key.length() + 1;
    auto result = httpd_ssl_start(&server, &settings);
    if (result != ESP_OK) {
        server = nullptr;
        return result;
    }
    httpd_uri_t status{};
    status.uri = "/v1/status";
    status.method = HTTP_GET;
    status.handler = handle_status;
    httpd_uri_t photo{};
    photo.uri = "/v1/photo";
    photo.method = HTTP_POST;
    photo.handler = handle_photo;
    result = httpd_register_uri_handler(server, &status);
    if (result == ESP_OK) result = httpd_register_uri_handler(server, &photo);
    for (const auto* path : {"/v1/audio"}) {
        httpd_uri_t clip{};
        clip.uri = path;
        clip.method = HTTP_POST;
        clip.handler = handle_clip;
        if (result == ESP_OK) result = httpd_register_uri_handler(server, &clip);
    }
    httpd_uri_t stream{};
    stream.uri = "/v1/stream";
    stream.method = HTTP_POST;
    stream.handler = handle_stream;
    if (result == ESP_OK) result = httpd_register_uri_handler(server, &stream);
    for (const auto method : {HTTP_GET, HTTP_POST}) {
        httpd_uri_t speed{};
        speed.uri = "/v1/speed";
        speed.method = method;
        speed.handler = method == HTTP_GET ? handle_speed_download : handle_speed_upload;
        if (result == ESP_OK) result = httpd_register_uri_handler(server, &speed);
    }
    if (result != ESP_OK) {
        httpd_ssl_stop(server);
        server = nullptr;
    } else if (!MDNS.begin(("kidi-" + configuration.device_id).c_str())) {
        Serial.println("KIDI_WARNING mDNS unavailable; use the USB-reported Wi-Fi address");
    } else {
        MDNS.addService("kidi-photo", "tcp", HTTPS_PORT);
    }
    return result;
}

auto connect_network(const Configuration& next) -> Error {
    if (server != nullptr) {
        httpd_ssl_stop(server);
        server = nullptr;
    }
    MDNS.end();
    configuration = next;
    WiFi.disconnect();
    WiFi.setHostname(("kidi-" + configuration.device_id).c_str());
    WiFi.begin(configuration.ssid.c_str(), configuration.password.c_str());
    const auto start = millis();
    while (WiFi.status() != WL_CONNECTED && millis() - start < JOIN_TIMEOUT_MS) delay(20);
    if (WiFi.status() != WL_CONNECTED) return ESP_ERR_TIMEOUT;
    return start_server();
}

auto configure(const cJSON* request) -> void {
    if (camera_busy() || clip_busy()) {
        reply_error("capture busy; finish capture before network setup");
        return;
    }
    const auto token = json_string(request, "token");
    if (!valid_hex(token, 64) || (configuration.configured() && !authorized_token(token))) {
        reply_error("owner credential required; no automatic ownership replacement", "OWNER_REQUIRED");
        return;
    }
    auto next = configuration;
    next.ssid = json_string(request, "ssid");
    next.password = json_string(request, "password");
    if (next.ssid.isEmpty() || next.ssid.length() > 32 || next.password.length() > 64 ||
        (!next.password.isEmpty() && next.password.length() < 8)) {
        reply_error("invalid Wi-Fi profile");
        return;
    }
    if (!next.configured()) {
        next.device_id = json_string(request, "device_id");
        next.certificate = json_string(request, "certificate");
        next.private_key = json_string(request, "private_key");
        next.token_hash = hex_digest(token);
        if (!valid_hex(next.device_id, 32) || next.certificate.length() > 2048 || next.private_key.length() > 2048 ||
            !next.certificate.startsWith("-----BEGIN CERTIFICATE-----") ||
            !next.private_key.startsWith("-----BEGIN PRIVATE KEY-----") || next.token_hash.length() != 64) {
            reply_error("invalid identity configuration");
            return;
        }
    }
    const auto previous = configuration;
    const auto result = connect_network(next);
    if (result == ESP_OK && save_configuration()) {
        reply_info();
        return;
    }
    if (server != nullptr) {
        httpd_ssl_stop(server);
        server = nullptr;
    }
    configuration = previous;
    if (previous.configured()) {
        const auto rollback = connect_network(previous);
        if (rollback != ESP_OK) report_error("Wi-Fi rollback", rollback);
    } else {
        WiFi.disconnect();
    }
    reply_error(result == ESP_ERR_TIMEOUT ? "Wi-Fi join timed out; check 2.4 GHz support and credentials"
                                          : "secure server or configuration persistence failed",
                result == ESP_ERR_TIMEOUT ? "WIFI_UNAVAILABLE" : "SETUP_FAILED");
}

} // namespace

auto initialize_wireless() -> void {
    WiFi.mode(WIFI_STA);
    WiFi.setScanMethod(WIFI_ALL_CHANNEL_SCAN);
    WiFi.setSortMethod(WIFI_CONNECT_AP_BY_SIGNAL);
    Preferences preferences;
    if (!preferences.begin("kidi-wifi", true)) return;
    const auto saved = preferences.getString("configuration", "");
    preferences.end();
    if (saved.isEmpty()) return;
    Json object(cJSON_Parse(saved.c_str()), cJSON_Delete);
    if (!object) {
        Serial.println("KIDI_ERROR invalid stored Wi-Fi configuration");
        return;
    }
    configuration = load_configuration(object.get());
    if (!configuration.configured() || !valid_hex(configuration.device_id, 32) ||
        !valid_hex(configuration.token_hash, 64)) {
        Serial.println("KIDI_ERROR invalid stored device identity");
        configuration = {};
        return;
    }
    WiFi.setHostname(("kidi-" + configuration.device_id).c_str());
    WiFi.begin(configuration.ssid.c_str(), configuration.password.c_str());
}

auto poll_wireless() -> void {
    if (configuration.configured() && WiFi.status() == WL_CONNECTED && server == nullptr && !reconnect_reported) {
        const auto result = start_server();
        if (result != ESP_OK) {
            report_error("HTTPS startup", result);
            reconnect_reported = true;
        }
    }
    if (WiFi.status() != WL_CONNECTED) reconnect_reported = false;
}

auto handle_wireless_command(const String& command) -> bool {
    if (command == "wifi-info") {
        reply_info();
        return true;
    }
    if (!command.startsWith("wifi-configure ")) return false;
    if (command.length() > 8192) {
        reply_error("setup request too large");
        return true;
    }
    Json request(cJSON_Parse(command.c_str() + 15), cJSON_Delete);
    if (!request)
        reply_error("invalid setup request");
    else
        configure(request.get());
    return true;
}

} // namespace kidi::esp32
