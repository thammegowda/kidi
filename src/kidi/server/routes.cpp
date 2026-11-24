#include "kidi/server/routes.h"

#include <atomic>
#include <chrono>
#include <memory>

namespace kidi::server {
namespace {

using Clock = std::chrono::steady_clock;

auto json_response(const ChatJson& body, int status = 200) -> liteapp::Response {
    return liteapp::Response::json(body.dump(), status);
}

auto error_status(const Error& error) -> int { return error.code == ErrorCode::INVALID_ARGUMENT ? 400 : 500; }

auto elapsed_ms(Clock::time_point started) -> double {
    return std::chrono::duration<double, std::milli>(Clock::now() - started).count();
}

auto enqueue(const ChatRequest& request) {
    return [&request](inference::Generator& generator) {
        return generator.enqueue_chat(request.messages, request.generation, request.template_args);
    };
}

/// Emits OpenAI deltas for the part of a parsed assistant response the client has not seen yet.
class DeltaWriter {
public:
    DeltaWriter(const liteapp::Response::ContentWriter& write, std::string id, std::string model,
                std::int64_t created)
        : write_(write), id_(std::move(id)), model_(std::move(model)), created_(created) {}

    auto send(const ChatJson& payload) -> bool { return write_("data: " + payload.dump() + "\n\n"); }
    auto chunk(ChatJson delta, ChatJson finish_reason = nullptr) -> ChatJson {
        return chat_chunk(id_, model_, created_, std::move(delta), std::move(finish_reason));
    }
    auto flush(const text::AssistantResponse& parsed) -> bool {
        ChatJson delta = ChatJson::object();
        if (parsed.reasoning.size() > sent_.reasoning.size())
            delta["reasoning_content"] = parsed.reasoning.substr(sent_.reasoning.size());
        if (parsed.content.size() > sent_.content.size())
            delta["content"] = parsed.content.substr(sent_.content.size());
        if (parsed.tool_calls.size() > sent_.tool_calls.size())
            delta["tool_calls"] = chat_tool_calls(parsed.tool_calls, sent_.tool_calls.size(), "call_" + id_);
        sent_ = parsed;
        return delta.empty() || send(chunk(std::move(delta)));
    }

private:
    const liteapp::Response::ContentWriter& write_;
    std::string id_, model_;
    std::int64_t created_;
    text::AssistantResponse sent_;
};

} // namespace

auto mount_chat_routes(liteapp::HttpServer& server, ChatEngine& engine, RouteOptions options) -> void {
    auto shared = std::make_shared<const RouteOptions>(std::move(options));
    auto next_id = std::make_shared<std::atomic_uint64_t>(1);

    server.get("/health", [](const liteapp::Request&) { return liteapp::Response::json("{}"); });
    server.get("/v1/models", [shared](const liteapp::Request&) {
        return json_response({{"object", "list"},
                              {"data", ChatJson::array({{{"id", shared->model_name},
                                                           {"object", "model"},
                                                           {"owned_by", "kidi"}}})}});
    });
    server.post("/v1/chat/context", [shared, &engine](const liteapp::Request& http) {
        const auto body = ChatJson::parse(http.body, nullptr, false);
        auto prepared = prepare_chat_request(body, shared->defaults);
        if (prepared && shared->customize)
            if (auto customized = shared->customize(body, *prepared); !customized)
                prepared = std::unexpected(customized.error());
        if (!prepared) return json_response(chat_error(prepared.error()), 400);
        auto result = engine.context(prepared->messages, prepared->template_args, prepared->generation);
        if (!result) return json_response(chat_error(result.error()), error_status(result.error()));
        return json_response({{"prompt_tokens", result->prompt_tokens},
                              {"context_tokens", result->context_tokens},
                              {"prompt_limit_tokens", result->prompt_limit_tokens}});
    });
    server.post("/v1/chat/completions", [shared, next_id, &engine](const liteapp::Request& http) {
        const auto started = Clock::now();
        const auto body = ChatJson::parse(http.body, nullptr, false);
        auto prepared = prepare_chat_request(body, shared->defaults);
        if (prepared && shared->customize)
            if (auto customized = shared->customize(body, *prepared); !customized)
                prepared = std::unexpected(customized.error());
        if (!prepared) return json_response(chat_error(prepared.error()), 400);

        const auto id = "chatcmpl-" + std::to_string((*next_id)++);
        const auto created =
            std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch())
                .count();
        const auto& model = shared->model_name;

        if (!prepared->stream) {
            auto result = engine.run({.enqueue = enqueue(*prepared),
                                      .cancelled = http.connection_closed,
                                      .on_text = {},
                                      .release_workspaces = false});
            if (!result) return json_response(chat_error(result.error()), error_status(result.error()));
            auto completion = chat_completion(*result, id, model, created);
            completion["kidi_stats"] = chat_stats(*result, elapsed_ms(started));
            return json_response(completion);
        }

        auto response = liteapp::Response::streamed(
            "text/event-stream",
            [shared, &engine, request = std::move(*prepared), id, created,
             started](const liteapp::Response::ContentWriter& write) {
                DeltaWriter out(write, id, shared->model_name, created);
                if (!out.send(out.chunk({{"role", "assistant"}, {"content", ""}}))) return false;
                std::string raw;
                bool connected = true;
                auto heartbeat = Clock::now();
                auto result = engine.run({.enqueue = enqueue(request),
                                          .cancelled =
                                              [&] {
                                                  const auto now = Clock::now();
                                                  if (now - heartbeat >= std::chrono::seconds(1)) {
                                                      heartbeat = now;
                                                      connected = write(": keep-alive\n\n");
                                                  }
                                                  return !connected;
                                              },
                                          .on_text =
                                              [&](std::string_view delta) {
                                                  raw.append(delta);
                                                  connected = out.flush(text::parse_assistant_response(raw, false));
                                                  return connected;
                                              },
                                          .release_workspaces = false});
                if (!connected) return false;
                if (!result) return out.send(chat_error(result.error())) && write("data: [DONE]\n\n");
                const auto parsed = text::parse_assistant_response(result->text);
                if (!out.flush(parsed) ||
                    !out.send(out.chunk(ChatJson::object(), chat_finish_reason(parsed, *result))))
                    return false;
                if (request.include_usage) {
                    auto tail = out.chunk(ChatJson::object());
                    tail["choices"] = ChatJson::array();
                    tail["usage"] = chat_usage(*result);
                    tail["kidi_stats"] = chat_stats(*result, elapsed_ms(started));
                    if (!out.send(tail)) return false;
                }
                return write("data: [DONE]\n\n");
            });
        return response;
    });
}

} // namespace kidi::server
