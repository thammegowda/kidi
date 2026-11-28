#pragma once

/// kidi's HTTP API, mounted by `kidi serve` and `kidi agent`'s background server so every client (kidi-agent, other
/// OpenAI-compatible agents) talks to the same `/health`, `/v1/models`, and `/v1/chat/completions`
/// implementation.

#include "kidi/server/engine.h"
#include "kidi/server/chat_api.h"

#include "liteapp/http/server.h"

#include <functional>
#include <string>

namespace kidi::server {

struct RouteOptions {
    std::string model_name;
    GenerationOptions defaults;
    /// Advertised in /v1/models as capabilities.vision: chat accepts image_url parts.
    bool vision = false;
};

auto mount_chat_routes(liteapp::HttpServer& server, ChatEngine& engine, RouteOptions options) -> void;

} // namespace kidi::server
