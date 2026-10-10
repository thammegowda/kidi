#pragma once

#include <cstdint>
#include <span>

#include "esp_err.h"
#include "kidi_ticket.h"
#include "kidi_tls_identity.h"

namespace kidi::c6 {

using Clock = std::uint64_t (*)();

auto start_control_server(const kidi_tls_identity_t& identity,
                          std::span<const std::uint8_t, KIDI_TICKET_KEY_BYTES> ticket_key, Clock clock) -> esp_err_t;

void complete_media_session(std::span<const std::uint8_t, KIDI_ACCESSORY_TICKET_NONCE_BYTES> nonce);

} // namespace kidi::c6
