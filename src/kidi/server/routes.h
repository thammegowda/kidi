#pragma once

/// kidi's HTTP API, mounted by both `kidi serve` and the desktop app so every client (Pi and other OpenAI-compatible
/// agents, the desktop UI) talks to the same `/health`, `/v1/models`, and `/v1/chat/completions` implementation.

#include "kidi/server/engine.h"
#include "kidi/server/openai.h"

#include "liteapp/http/server.h"

#include <functional>
#include <string>

namespace kidi::server {

struct RouteOptions {
    std::string model_name;
    GenerationOptions defaults;
    /// Optional host-specific request extension, applied after standard OpenAI validation (e.g. the desktop app's
    /// system prompt). An error becomes a 400 response.
    std::function<Result<void>(const OpenAIJson& body, OpenAIChatRequest& request)> customize;
};

auto mount_openai_routes(liteapp::HttpServer& server, ChatEngine& engine, RouteOptions options) -> void;

} // namespace kidi::server
