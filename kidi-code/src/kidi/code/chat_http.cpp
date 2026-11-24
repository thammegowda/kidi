#include "kidi/code/chat_transport.h"

#include <httplib.h>
#include <algorithm>
#include <map>
#include <utility>

namespace kidi::code {
namespace {

// SSE framing is independent of HTTP chunks, which may split a JSON value or a UTF-8 character.
class CompletionStream {
public:
    Json message{{"role", "assistant"}, {"content", ""}, {"reasoning_content", ""}};
    Json usage, stats, finish;
    std::map<std::size_t, Json> calls;
    bool done = false, started = false;
    std::size_t text_bytes = 0;
    std::string data;

    auto event(const ChatClientOptions& options) -> Result<void> {
        if (data.empty()) return {};
        auto payload = std::exchange(data, {});
        if (done) return invalid("data after [DONE]");
        if (payload == "[DONE]") {
            done = true;
            return {};
        }
        const auto chunk = Json::parse(payload, nullptr, false);
        if (!chunk.is_object()) return invalid("invalid JSON event");
        if (chunk.contains("error")) return invalid("server error: " + chunk["error"].dump());
        if (!chunk.contains("choices") || !chunk["choices"].is_array() || chunk["choices"].size() > 1)
            return invalid("expected at most one choice");
        if (chunk.contains("usage") && !chunk["usage"].is_null()) usage = chunk["usage"];
        if (chunk.contains("kidi_stats")) stats = chunk["kidi_stats"];
        if (chunk["choices"].empty()) return {};
        const auto& choice = chunk["choices"][0];
        if (!choice.is_object() || !choice.contains("index") || choice["index"] != 0 ||
            !choice.contains("delta") || !choice["delta"].is_object() || !finish.is_null())
            return invalid("invalid choice or delta after finish");
        const auto& delta = choice["delta"];
        if (delta.contains("role") && delta["role"] != "assistant") return invalid("invalid assistant role");
        for (const auto* key : {"reasoning_content", "content"}) {
            if (!delta.contains(key) || delta[key].is_null()) continue;
            if (!delta[key].is_string()) return invalid("non-string text delta");
            const auto& text = delta[key].get_ref<const std::string&>();
            if ((text_bytes += text.size()) > options.max_response_bytes) return invalid("response too large");
            message[key].get_ref<std::string&>() += text;
            if (!text.empty()) options.on_delta(text, std::string_view(key) == "reasoning_content");
        }
        if (delta.contains("tool_calls") && !delta["tool_calls"].is_null()) {
            if (!delta["tool_calls"].is_array()) return invalid("invalid tool calls");
            for (const auto& fragment : delta["tool_calls"]) {
                if (!fragment.is_object() || !fragment.contains("index") || !fragment["index"].is_number_unsigned())
                    return invalid("invalid tool call index");
                auto& call = calls[fragment["index"].get<std::size_t>()];
                if (call.is_null())
                    call = {{"id", ""}, {"type", "function"}, {"function", {{"name", ""}, {"arguments", ""}}}};
                if (fragment.contains("type") && fragment["type"] != "function") return invalid("invalid tool type");
                if (fragment.contains("id")) {
                    if (!fragment["id"].is_string()) return invalid("invalid tool id");
                    call["id"].get_ref<std::string&>() += fragment["id"].get_ref<const std::string&>();
                }
                if (fragment.contains("function")) {
                    const auto& function = fragment["function"];
                    if (!function.is_object()) return invalid("invalid function delta");
                    for (const auto* field : {"name", "arguments"}) {
                        if (!function.contains(field)) continue;
                        if (!function[field].is_string()) return invalid("invalid function field");
                        if ((text_bytes += function[field].get_ref<const std::string&>().size()) >
                            options.max_response_bytes)
                            return invalid("response too large");
                        call["function"][field].get_ref<std::string&>() +=
                            function[field].get_ref<const std::string&>();
                    }
                }
            }
        }
        if (choice.contains("finish_reason") && !choice["finish_reason"].is_null()) {
            if (!choice["finish_reason"].is_string()) return invalid("invalid finish reason");
            finish = choice["finish_reason"];
        }
        return {};
    }

    auto line(std::string_view line, const ChatClientOptions& options) -> Result<void> {
        if (line.ends_with('\r')) line.remove_suffix(1);
        if (line.empty()) return event(options);
        if (line.starts_with("data:")) {
            started = true;
            line.remove_prefix(5);
            if (line.starts_with(' ')) line.remove_prefix(1);
            if (!data.empty()) data += '\n';
            data += line;
        }
        return {};
    }

    auto response() -> Result<std::string> {
        if (!done || finish.is_null()) return invalid("incomplete response (missing finish reason or [DONE])");
        if (!calls.empty()) {
            message["tool_calls"] = Json::array();
            for (const auto& [index, call] : calls) {
                if (index != message["tool_calls"].size()) return invalid("non-contiguous tool indices");
                message["tool_calls"].push_back(call);
            }
        }
        Json body{{"choices", Json::array({{{"message", message}, {"finish_reason", finish}}})}};
        if (!usage.is_null()) body["usage"] = usage;
        if (!stats.is_null()) body["kidi_stats"] = stats;
        return body.dump();
    }

private:
    static auto invalid(std::string text) -> std::unexpected<Error> {
        return std::unexpected(Error{ErrorCode::IO, "chat stream: " + std::move(text)});
    }
};

} // namespace

auto http_completion(const ChatClientOptions& options, std::string payload, std::stop_token stop, int expected_pid,
                     std::string_view route) -> Result<std::string> {
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
    const auto request = Json::parse(payload, nullptr, false);
    const bool streaming = options.on_delta && request.is_object() && request.contains("stream") &&
                           request["stream"] == true;
    CompletionStream stream;
    std::size_t consumed = 0;
    std::optional<Error> stream_error;
    bool too_large = false;
    const auto result = http.Post(options.base_url.substr(path) + std::string(route), {}, payload, "application/json",
                                  [&](const char* chunk, std::size_t size) {
                                      if (stop.stop_requested()) return false;
                                      if (size > options.max_response_bytes - body.size()) {
                                          too_large = true;
                                          return false;
                                      }
                                      body.append(chunk, size);
                                      if (streaming) {
                                          // The SSE framing is much larger than the text it carries; bound the
                                          // decoded reply instead and keep only the unparsed tail.
                                          for (;;) {
                                              const auto end = body.find('\n', consumed);
                                              if (end == std::string::npos) break;
                                              const auto parsed = stream.line(
                                                  std::string_view(body).substr(consumed, end - consumed), options);
                                              consumed = end + 1;
                                              if (!parsed) {
                                                  stream_error = parsed.error();
                                                  return false;
                                              }
                                          }
                                          // An HTTP error body is plain JSON with no `data:` lines; keep it intact.
                                          if (stream.started) {
                                              body.erase(0, consumed);
                                              consumed = 0;
                                          }
                                      }
                                      return true;
                                  });
    if (stop.stop_requested()) return std::unexpected(Error{ErrorCode::RUNTIME, "cancelled"});
    if (too_large) return std::unexpected(Error{ErrorCode::IO, "response exceeded max_response_bytes"});
    if (stream_error) return std::unexpected(*stream_error);
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
    if (streaming) return stream.response();
    return body;
}

auto http_provider(ChatClientOptions options) -> Result<Provider> {
    while (options.base_url.ends_with('/')) options.base_url.pop_back();
    const auto path = options.base_url.find('/', 7);
    if (!options.base_url.starts_with("http://") || path == std::string::npos || path == 7 ||
        options.base_url.substr(path) != "/v1" || options.base_url.find_first_of("@?# \t\r\n") != std::string::npos)
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "expected a Kidi server URL: http://host:port/v1"});
    return completion_provider(std::move(options), [](const ChatClientOptions& config, std::string payload,
                                                      std::stop_token stop, std::string_view route) {
        return http_completion(config, std::move(payload), stop, 0, route);
    });
}

} // namespace kidi::code
