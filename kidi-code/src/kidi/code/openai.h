#pragma once

#include "kidi/code/agent.h"

#include <chrono>

namespace kidi::code {

struct OpenAIOptions {
    std::string base_url = "http://127.0.0.1:18097/v1";
    std::string model = "kidi-local";
    std::size_t max_tokens = 4096;
    std::size_t max_response_bytes = 1024 * 1024;
    std::chrono::milliseconds timeout{120000};
};

// Kidi server's Chat Completions protocol over plain HTTP. No redirects or retries.
auto openai_provider(OpenAIOptions options = {}) -> Result<Provider>;

} // namespace kidi::code
