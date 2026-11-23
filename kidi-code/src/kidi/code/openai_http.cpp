#include "kidi/code/openai_transport.h"

#include <httplib.h>
#include <algorithm>

namespace kidi::code {

auto http_completion(const OpenAIOptions& options, std::string payload, std::stop_token stop, int expected_pid)
    -> Result<std::string> {
    const auto path = options.base_url.find('/', 7);
    httplib::Client http(options.base_url.substr(0, path));
    http.set_keep_alive(true);
    http.set_connection_timeout(std::min(options.timeout, std::chrono::milliseconds(10000)));
    http.set_read_timeout(options.timeout);
    http.set_write_timeout(options.timeout);
    http.set_max_timeout(options.timeout);
    std::stop_callback cancel(stop, [&] { http.stop(); });
    if (expected_pid) {
        const auto health = http.Get("/kidi-code/ready");
        if (stop.stop_requested()) return std::unexpected(Error{ErrorCode::RUNTIME, "cancelled"});
        if (!health || health->status != 200 || health->body != std::to_string(expected_pid))
            return std::unexpected(Error{ErrorCode::IO, "background inference is no longer ready"});
    }
    std::string body;
    bool too_large = false;
    const auto result = http.Post(options.base_url.substr(path) + "/chat/completions", {}, payload, "application/json",
                                  [&](const char* chunk, std::size_t size) {
                                      if (stop.stop_requested()) return false;
                                      if (size > options.max_response_bytes - body.size()) {
                                          too_large = true;
                                          return false;
                                      }
                                      body.append(chunk, size);
                                      return true;
                                  });
    if (stop.stop_requested()) return std::unexpected(Error{ErrorCode::RUNTIME, "cancelled"});
    if (too_large) return std::unexpected(Error{ErrorCode::IO, "response exceeded max_response_bytes"});
    if (!result)
        return std::unexpected(
            Error{ErrorCode::IO, "Kidi server request failed: " + httplib::to_string(result.error())});
    if (result->status != 200) {
        std::string message = "Kidi server returned HTTP " + std::to_string(result->status);
        const auto error = Json::parse(body, nullptr, false);
        if (error.is_object() && error.contains("error") && error["error"].is_object() &&
            error["error"].contains("message") && error["error"]["message"].is_string())
            message += ": " + error["error"]["message"].get<std::string>();
        return std::unexpected(Error{ErrorCode::RUNTIME, std::move(message)});
    }
    return body;
}

auto openai_provider(OpenAIOptions options) -> Result<Provider> {
    while (options.base_url.ends_with('/')) options.base_url.pop_back();
    const auto path = options.base_url.find('/', 7);
    if (!options.base_url.starts_with("http://") || path == std::string::npos || path == 7 ||
        options.base_url.substr(path) != "/v1" || options.base_url.find_first_of("@?# \t\r\n") != std::string::npos)
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "expected a Kidi server URL: http://host:port/v1"});
    return completion_provider(std::move(options), [](const OpenAIOptions& config, std::string payload,
                                                      std::stop_token stop) {
        return http_completion(config, std::move(payload), stop);
    });
}

} // namespace kidi::code
