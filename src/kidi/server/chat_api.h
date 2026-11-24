#pragma once

/// OpenAI-compatible Chat Completions wire format: request validation and response/chunk/error JSON builders.
/// Used by kidi's single HTTP server (`kidi/server/routes.h`); has no transport dependencies.

#include "kidi/inference/generator.h"
#include "kidi/text/response.h"

#include <nlohmann/json.hpp>

namespace kidi::server {

using ChatJson = nlohmann::ordered_json;
using inference::GenerationOptions;
using inference::TextGeneration;

struct ChatRequest {
    ChatJson messages;
    ChatJson template_args = ChatJson::object();
    GenerationOptions generation;
    bool stream = false;
    bool include_usage = false;
};

auto prepare_chat_request(const ChatJson& body, GenerationOptions defaults)
    -> Result<ChatRequest>;
auto chat_tool_calls(const std::vector<text::ToolCall>& calls, std::size_t first = 0,
                       std::string_view prefix = "call") -> ChatJson;
auto chat_message(const text::AssistantResponse& response, std::string_view prefix = "call") -> ChatJson;
auto chat_finish_reason(const text::AssistantResponse& response, const TextGeneration& generation) -> std::string;
auto chat_usage(const TextGeneration& generation) -> ChatJson;
auto chat_completion(const TextGeneration& generation, std::string_view id, std::string_view model,
                       std::int64_t created) -> ChatJson;
auto chat_chunk(std::string_view id, std::string_view model, std::int64_t created, ChatJson delta,
                  ChatJson finish_reason = nullptr) -> ChatJson;
auto chat_error(const Error& error) -> ChatJson;
/// kidi extension attached as `kidi_stats` to completions and usage chunks; OpenAI clients ignore it.
auto chat_stats(const TextGeneration& generation, double elapsed_ms) -> ChatJson;

} // namespace kidi::server
