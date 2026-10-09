#pragma once

#include <stddef.h>
#include <stdint.h>

#include "accessory_protocol.h"

#define KIDI_PEER_SYNC_REQUEST_ID 0x4b490101U
#define KIDI_PEER_SYNC_RESPONSE_ID 0x4b490102U
#define KIDI_PEER_INVITATION_REQUEST_ID 0x4b490103U
#define KIDI_PEER_INVITATION_RESPONSE_ID 0x4b490104U
#define KIDI_PEER_MEDIA_COMPLETE_ID 0x4b490105U

#define KIDI_PEER_SYNC_REQUEST_BYTES 41U
#define KIDI_PEER_SYNC_RESPONSE_BYTES 42U
#define KIDI_PEER_INVITATION_REQUEST_BYTES 11U
#define KIDI_PEER_INVITATION_MAX_RESPONSE_BYTES 1024U
#define KIDI_PEER_MEDIA_COMPLETE_BYTES 17U

#define KIDI_PEER_SYNC_STATUS_CLOCK_MISSING 0U
#define KIDI_PEER_SYNC_STATUS_READY 1U

static inline void kidi_peer_write_u64_be(uint8_t output[8], uint64_t value) {
    for (unsigned index = 0; index < 8; ++index) {
        output[7 - index] = (uint8_t)(value >> (index * 8U));
    }
}

static inline uint64_t kidi_peer_read_u64_be(const uint8_t input[8]) {
    uint64_t value = 0;
    for (unsigned index = 0; index < 8; ++index) {
        value = (value << 8U) | input[index];
    }
    return value;
}
