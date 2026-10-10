#pragma once

#ifdef __cplusplus
#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>
#else
#include <stddef.h>
#include <stdint.h>
#endif

#define KIDI_ACCESSORY_PROTOCOL_VERSION 1U
#define KIDI_ACCESSORY_MAX_CONTROLLERS 4U
#define KIDI_ACCESSORY_MAX_MEDIA_SESSIONS 1U
#define KIDI_ACCESSORY_MAX_BLE_MESSAGE_BYTES 2048U

#define KIDI_ACCESSORY_CONTROL_PORT 61443U
#define KIDI_ACCESSORY_MEDIA_PORT 54443U

#define KIDI_ACCESSORY_C6_WAKEUP_GPIO 2
#define KIDI_ACCESSORY_P4_WAKEUP_GPIO 6
#define KIDI_ACCESSORY_P4_C6_RESET_GPIO 54

#define KIDI_ACCESSORY_BLE_UUID_NAMESPACE "DNS"
#define KIDI_ACCESSORY_BLE_UUID_NAMESPACE_UUID "6ba7b810-9dad-11d1-80b4-00c04fd430c8"
#define KIDI_ACCESSORY_BLE_SERVICE_TAG "ai.gowda.kidi.accessory.v1.service"
#define KIDI_ACCESSORY_BLE_PAIR_REQUEST_TAG "ai.gowda.kidi.accessory.v1.pair-request"
#define KIDI_ACCESSORY_BLE_PAIR_RESPONSE_TAG "ai.gowda.kidi.accessory.v1.pair-response"
#define KIDI_ACCESSORY_BLE_STATUS_TAG "ai.gowda.kidi.accessory.v1.status"
#define KIDI_ACCESSORY_BLE_SERVICE_UUID "9a1f0dbc-a6fe-536d-94d8-5ca19ed84493"
#define KIDI_ACCESSORY_BLE_PAIR_REQUEST_UUID "8e68995e-6448-552c-9376-168454089158"
#define KIDI_ACCESSORY_BLE_PAIR_RESPONSE_UUID "e9c06854-b576-5a5a-876d-79fa31eb1e52"
#define KIDI_ACCESSORY_BLE_STATUS_UUID "d7105927-5e98-5292-8fe4-1fb2cab546b9"

#define KIDI_ACCESSORY_STATUS_PATH "/v1/status"
#define KIDI_ACCESSORY_MEDIA_TICKET_PATH "/v1/media-ticket"
#define KIDI_ACCESSORY_CONTROLLERS_SELF_PATH "/v1/controllers/self"
#define KIDI_ACCESSORY_PHOTO_PATH "/v1/photo"
#define KIDI_ACCESSORY_AUDIO_PATH "/v1/audio"

#define KIDI_ACCESSORY_CONTROLLER_AUTH_SCHEME "Bearer"
#define KIDI_ACCESSORY_TICKET_AUTH_SCHEME "Ticket"
#define KIDI_ACCESSORY_PHOTO_CONTENT_TYPE "image/jpeg"
#define KIDI_ACCESSORY_AUDIO_CONTENT_TYPE "application/vnd.kidi.pcm16"
#define KIDI_ACCESSORY_PHOTO_DIGEST_HEADER "X-Kidi-SHA256"
#define KIDI_ACCESSORY_PHOTO_WIDTH_HEADER "X-Kidi-Width"
#define KIDI_ACCESSORY_PHOTO_HEIGHT_HEADER "X-Kidi-Height"
#define KIDI_ACCESSORY_AUDIO_SAMPLE_RATE_HEADER "X-Kidi-Sample-Rate"
#define KIDI_ACCESSORY_AUDIO_CHANNELS_HEADER "X-Kidi-Channels"
#define KIDI_ACCESSORY_AUDIO_BITS_HEADER "X-Kidi-Bits-Per-Sample"

#define KIDI_ACCESSORY_AUDIO_SAMPLE_RATE 16000U
#define KIDI_ACCESSORY_AUDIO_CHANNELS 1U
#define KIDI_ACCESSORY_AUDIO_BITS_PER_SAMPLE 16U
#define KIDI_ACCESSORY_MAX_AUDIO_SECONDS 30U
#define KIDI_ACCESSORY_MAX_PHOTO_BYTES (4U * 1024U * 1024U)
#define KIDI_ACCESSORY_MAX_TICKET_BYTES 512U
#define KIDI_ACCESSORY_TICKET_LIFETIME_SECONDS 10U
#define KIDI_ACCESSORY_TICKET_NONCE_BYTES 16U
#define KIDI_ACCESSORY_TICKET_PARAMETER_HASH_BYTES 32U
#define KIDI_ACCESSORY_TICKET_MAC_BYTES 32U
#define KIDI_ACCESSORY_TICKET_BODY_BYTES 82U
#define KIDI_ACCESSORY_TICKET_BYTES 114U
#define KIDI_ACCESSORY_TICKET_BASE64URL_BYTES 152U

typedef enum {
    KIDI_ACCESSORY_MEDIA_PHOTO = 1,
    KIDI_ACCESSORY_MEDIA_AUDIO = 2,
} kidi_accessory_media_kind_t;

#ifdef __cplusplus
namespace kidi::esp32::accessory {

constexpr unsigned PROTOCOL_VERSION = KIDI_ACCESSORY_PROTOCOL_VERSION;
constexpr unsigned MAX_CONTROLLERS = KIDI_ACCESSORY_MAX_CONTROLLERS;
constexpr unsigned MAX_MEDIA_SESSIONS = KIDI_ACCESSORY_MAX_MEDIA_SESSIONS;
constexpr std::size_t MAX_BLE_MESSAGE_BYTES = KIDI_ACCESSORY_MAX_BLE_MESSAGE_BYTES;

constexpr std::uint16_t CONTROL_PORT = KIDI_ACCESSORY_CONTROL_PORT;
constexpr std::uint16_t MEDIA_PORT = KIDI_ACCESSORY_MEDIA_PORT;

constexpr int C6_WAKEUP_GPIO = KIDI_ACCESSORY_C6_WAKEUP_GPIO;
constexpr int P4_WAKEUP_GPIO = KIDI_ACCESSORY_P4_WAKEUP_GPIO;
constexpr int P4_C6_RESET_GPIO = KIDI_ACCESSORY_P4_C6_RESET_GPIO;

constexpr char BLE_UUID_NAMESPACE[] = KIDI_ACCESSORY_BLE_UUID_NAMESPACE;
constexpr char BLE_UUID_NAMESPACE_UUID[] = KIDI_ACCESSORY_BLE_UUID_NAMESPACE_UUID;
constexpr char BLE_SERVICE_TAG[] = KIDI_ACCESSORY_BLE_SERVICE_TAG;
constexpr char BLE_PAIR_REQUEST_TAG[] = KIDI_ACCESSORY_BLE_PAIR_REQUEST_TAG;
constexpr char BLE_PAIR_RESPONSE_TAG[] = KIDI_ACCESSORY_BLE_PAIR_RESPONSE_TAG;
constexpr char BLE_STATUS_TAG[] = KIDI_ACCESSORY_BLE_STATUS_TAG;
constexpr char BLE_SERVICE_UUID[] = KIDI_ACCESSORY_BLE_SERVICE_UUID;
constexpr char BLE_PAIR_REQUEST_UUID[] = KIDI_ACCESSORY_BLE_PAIR_REQUEST_UUID;
constexpr char BLE_PAIR_RESPONSE_UUID[] = KIDI_ACCESSORY_BLE_PAIR_RESPONSE_UUID;
constexpr char BLE_STATUS_UUID[] = KIDI_ACCESSORY_BLE_STATUS_UUID;

constexpr char STATUS_PATH[] = KIDI_ACCESSORY_STATUS_PATH;
constexpr char MEDIA_TICKET_PATH[] = KIDI_ACCESSORY_MEDIA_TICKET_PATH;
constexpr char CONTROLLERS_SELF_PATH[] = KIDI_ACCESSORY_CONTROLLERS_SELF_PATH;
constexpr char PHOTO_PATH[] = KIDI_ACCESSORY_PHOTO_PATH;
constexpr char AUDIO_PATH[] = KIDI_ACCESSORY_AUDIO_PATH;

constexpr unsigned AUDIO_SAMPLE_RATE = KIDI_ACCESSORY_AUDIO_SAMPLE_RATE;
constexpr unsigned AUDIO_CHANNELS = KIDI_ACCESSORY_AUDIO_CHANNELS;
constexpr unsigned AUDIO_BITS_PER_SAMPLE = KIDI_ACCESSORY_AUDIO_BITS_PER_SAMPLE;
constexpr unsigned MAX_AUDIO_SECONDS = KIDI_ACCESSORY_MAX_AUDIO_SECONDS;
constexpr std::size_t MAX_PHOTO_BYTES = KIDI_ACCESSORY_MAX_PHOTO_BYTES;
constexpr std::size_t MAX_TICKET_BYTES = KIDI_ACCESSORY_MAX_TICKET_BYTES;
constexpr unsigned TICKET_LIFETIME_SECONDS = KIDI_ACCESSORY_TICKET_LIFETIME_SECONDS;
constexpr std::size_t TICKET_NONCE_BYTES = KIDI_ACCESSORY_TICKET_NONCE_BYTES;
constexpr std::size_t TICKET_BODY_BYTES = KIDI_ACCESSORY_TICKET_BODY_BYTES;
constexpr std::size_t TICKET_BYTES = KIDI_ACCESSORY_TICKET_BYTES;
constexpr std::size_t TICKET_BASE64URL_BYTES = KIDI_ACCESSORY_TICKET_BASE64URL_BYTES;

enum class MediaKind : std::uint8_t {
    PHOTO = KIDI_ACCESSORY_MEDIA_PHOTO,
    AUDIO = KIDI_ACCESSORY_MEDIA_AUDIO,
};

#if __cplusplus >= 202002L
consteval
#else
constexpr
#endif
    auto ble_uuid_little_endian(std::string_view text) -> std::array<std::uint8_t, 16> {
    const auto hex = [](char value) -> std::uint8_t {
        if (value >= '0' && value <= '9') {
            return static_cast<std::uint8_t>(value - '0');
        }
        if (value >= 'a' && value <= 'f') {
            return static_cast<std::uint8_t>(value - 'a' + 10);
        }
        return 0;
    };
    std::array<std::uint8_t, 16> canonical{};
    std::size_t output = 0;
    bool high_nibble = true;
    for (const char value : text) {
        if (value == '-') {
            continue;
        }
        if (high_nibble) {
            canonical[output] = static_cast<std::uint8_t>(hex(value) << 4U);
        } else {
            canonical[output++] |= hex(value);
        }
        high_nibble = !high_nibble;
    }
    std::array<std::uint8_t, 16> little_endian{};
    for (std::size_t index = 0; index < canonical.size(); ++index) {
        little_endian[index] = canonical[canonical.size() - index - 1];
    }
    return little_endian;
}

} // namespace kidi::esp32::accessory
#endif
