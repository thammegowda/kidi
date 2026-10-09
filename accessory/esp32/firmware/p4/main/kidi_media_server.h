#pragma once

#include <cstdint>
#include <span>

#include "esp_err.h"
#include "kidi_tls_identity.h"
#include "kidi_ticket.h"

namespace kidi::p4 {

using CompletionCallback = void (*)(std::span<const std::uint8_t, KIDI_ACCESSORY_TICKET_NONCE_BYTES>);

auto start_media_server(const kidi_tls_identity_t& identity, CompletionCallback completion) -> esp_err_t;

auto configure_media_security(std::span<const std::uint8_t, KIDI_TICKET_KEY_BYTES> ticket_key,
                              std::uint64_t epoch_seconds) -> esp_err_t;

} // namespace kidi::p4
