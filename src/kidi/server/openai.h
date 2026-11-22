#pragma once

/// OpenAI-compatible Chat Completions wire format: request validation and response/chunk/error JSON builders.
/// Used by kidi's single HTTP server (`kidi/server/routes.h`); has no transport dependencies.

#include "kidi/inference/generator.h"
#include "kidi/text/response.h"

#include <nlohmann/json.hpp>

namespace kidi::server {

using OpenAIJson = nlohmann::ordered_json;
using inference::GenerationOptions;
using inference::TextGeneration;

struct OpenAIChatRequest {
    OpenAIJson messages;
    OpenAIJson template_args = OpenAIJson::object();
    GenerationOptions generation;
    bool stream = false;
    bool include_usage = false;
};

auto prepare_openai_chat_request(const OpenAIJson& body, GenerationOptions defaults)
    -> Result<OpenAIChatRequest>;
auto openai_tool_calls(const std::vector<text::ToolCall>& calls, std::size_t first = 0,
                       std::string_view prefix = "call") -> OpenAIJson;
auto openai_chat_message(const text::AssistantResponse& response, std::string_view prefix = "call") -> OpenAIJson;
auto openai_finish_reason(const text::AssistantResponse& response, const TextGeneration& generation) -> std::string;
auto openai_usage(const TextGeneration& generation) -> OpenAIJson;
auto openai_completion(const TextGeneration& generation, std::string_view id, std::string_view model,
                       std::int64_t created) -> OpenAIJson;
auto openai_chunk(std::string_view id, std::string_view model, std::int64_t created, OpenAIJson delta,
                  OpenAIJson finish_reason = nullptr) -> OpenAIJson;
auto openai_error(const Error& error) -> OpenAIJson;
/// kidi extension attached as `kidi_stats` to completions and usage chunks; OpenAI clients ignore it.
auto openai_stats(const TextGeneration& generation, double elapsed_ms) -> OpenAIJson;

} // namespace kidi::server
