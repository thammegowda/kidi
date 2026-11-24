#pragma once

/// kidi's HTTP API, mounted by both `kidi serve` and the desktop app so every client (Pi and other OpenAI-compatible
/// agents, the desktop UI) talks to the same `/health`, `/v1/models`, and `/v1/chat/completions` implementation.

#include "kidi/server/engine.h"
#include "kidi/server/chat_api.h"

#include "liteapp/http/server.h"

#include <functional>
#include <string>

namespace kidi::server {

struct RouteOptions {
    std::string model_name;
    GenerationOptions defaults;
    /// Optional host-specific request extension, applied after standard OpenAI validation (e.g. the desktop app's
    /// system prompt). An error becomes a 400 response.
    std::function<Result<void>(const ChatJson& body, ChatRequest& request)> customize;
};

auto mount_chat_routes(liteapp::HttpServer& server, ChatEngine& engine, RouteOptions options) -> void;

} // namespace kidi::server
