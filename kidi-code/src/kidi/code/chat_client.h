#pragma once

#include "kidi/code/agent.h"

#include <chrono>

namespace kidi::code {

struct ChatClientOptions {
    std::string base_url = "http://127.0.0.1:18097/v1";
    std::string model = "kidi-local";
    std::size_t max_tokens = 4096;
    // Ignored for summaries, which are always greedy. top_p 1 and top_k 0 are omitted from requests.
    float temperature = 0, top_p = 1;
    std::size_t top_k = 0;
    std::size_t max_response_bytes = 1024 * 1024;
    std::chrono::milliseconds timeout{120000};
    // Called with decoded text deltas on the request thread; true denotes reasoning, not answer text.
    std::function<void(std::string_view, bool)> on_delta;
};

// Kidi server's Chat Completions protocol over plain HTTP. No redirects or retries.
auto http_provider(ChatClientOptions options = {}) -> Result<Provider>;

} // namespace kidi::code
