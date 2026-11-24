#include "kidi/server/chat_api.h"

#include <limits>

namespace kidi::server {
namespace {

auto invalid(std::string message) -> Result<ChatRequest> {
    return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, std::move(message)});
}

auto read_flag(const ChatJson& object, const char* key, bool& value) -> bool {
    if (!object.contains(key) || object[key].is_null()) return true;
    if (!object[key].is_boolean()) return false;
    value = object[key].get<bool>();
    return true;
}

auto read_token_limit(const ChatJson& body, const char* field, std::size_t& value) -> bool {
    if (!body.contains(field) || body[field].is_null()) return true;
    const auto& limit = body[field];
    if (limit.is_number_unsigned()) {
        const auto count = limit.get<std::uint64_t>();
        if (!count || count > std::numeric_limits<std::size_t>::max()) return false;
        value = static_cast<std::size_t>(count);
        return true;
    }
    if (!limit.is_number_integer()) return false;
    const auto count = limit.get<std::int64_t>();
    if (count <= 0 || static_cast<std::uint64_t>(count) > std::numeric_limits<std::size_t>::max()) return false;
    value = static_cast<std::size_t>(count);
    return true;
}

auto read_number(const ChatJson& body, const char* field, double minimum, double maximum, float& value) -> bool {
    if (!body.contains(field) || body[field].is_null()) return true;
    if (!body[field].is_number()) return false;
    const auto number = body[field].get<double>();
    if (!(number >= minimum && number <= maximum)) return false;
    value = static_cast<float>(number);
    return true;
}

} // namespace

auto prepare_chat_request(const ChatJson& body, GenerationOptions defaults) -> Result<ChatRequest> {
    if (!body.is_object() || !body.contains("messages") || !body["messages"].is_array() ||
        body["messages"].empty())
        return invalid("messages must be a nonempty array");
    if (body.contains("n") && body["n"] != 1) return invalid("only n=1 is supported");

    ChatRequest request{.messages = body["messages"], .generation = std::move(defaults)};
    for (auto& message : request.messages) {
        if (!message.is_object()) return invalid("messages must be objects");
        if (message.contains("content") && message["content"].is_null()) message["content"] = "";
        if (!message.contains("tool_calls")) continue;
        if (!message["tool_calls"].is_array()) return invalid("tool_calls must be an array");
        for (auto& call : message["tool_calls"]) {
            if (!call.is_object() || !call.contains("function") || !call["function"].is_object() ||
                !call["function"].contains("arguments"))
                continue;
            auto& arguments = call["function"]["arguments"];
            if (!arguments.is_string()) continue;
            const auto text = arguments.get<std::string>();
            arguments = text.empty() ? ChatJson::object() : ChatJson::parse(text, nullptr, false);
            if (arguments.is_discarded()) return invalid("tool call arguments must be JSON");
        }
    }

    if (body.contains("chat_template_kwargs")) {
        if (!body["chat_template_kwargs"].is_object()) return invalid("chat_template_kwargs must be an object");
        request.template_args = body["chat_template_kwargs"];
    }
    if (body.contains("tools")) {
        if (!body["tools"].is_array()) return invalid("tools must be an array");
        if (!body["tools"].empty()) request.template_args["tools"] = body["tools"];
    }
    if (body.contains("reasoning_effort") && !request.template_args.contains("reasoning_effort"))
        request.template_args["reasoning_effort"] = body["reasoning_effort"];

    if (!read_flag(body, "stream", request.stream)) return invalid("stream must be a boolean");
    if (body.contains("stream_options")) {
        if (!body["stream_options"].is_object()) return invalid("stream_options must be an object");
        if (!read_flag(body["stream_options"], "include_usage", request.include_usage))
            return invalid("stream_options.include_usage must be a boolean");
    }
    for (const auto* field : {"max_completion_tokens", "max_tokens"}) {
        auto value = request.generation.maximum_new_tokens;
        if (!read_token_limit(body, field, value)) return invalid(std::string(field) + " must be a positive integer");
        if (body.contains(field) && !body[field].is_null()) {
            request.generation.maximum_new_tokens = value;
            break;
        }
    }
    auto& generation = request.generation;
    if (!read_number(body, "temperature", 0, 2, generation.temperature))
        return invalid("temperature must be a number in [0, 2]");
    if (!read_number(body, "top_p", 0, 1, generation.top_p) || generation.top_p <= 0)
        return invalid("top_p must be a number in (0, 1]");
    if (body.contains("top_k") && !body["top_k"].is_null()) {
        if (!body["top_k"].is_number_integer() || body["top_k"].get<std::int64_t>() < 0)
            return invalid("top_k must be a non-negative integer");
        generation.top_k = body["top_k"].get<std::size_t>();
    }
    if (body.contains("seed") && !body["seed"].is_null()) {
        if (!body["seed"].is_number_integer()) return invalid("seed must be an integer");
        generation.seed = body["seed"].get<std::uint64_t>();
    }
    request.generation.stream_text = true;
    return request;
}

auto chat_tool_calls(const std::vector<text::ToolCall>& calls, std::size_t first, std::string_view prefix)
    -> ChatJson {
    auto result = ChatJson::array();
    for (auto index = first; index < calls.size(); ++index)
        result.push_back({{"index", index},
                          {"id", std::string(prefix) + "_" + std::to_string(index)},
                          {"type", "function"},
                          {"function", {{"name", calls[index].name}, {"arguments", calls[index].arguments.dump()}}}});
    return result;
}

auto chat_message(const text::AssistantResponse& response, std::string_view prefix) -> ChatJson {
    ChatJson message{{"role", "assistant"}, {"content", response.content}};
    if (!response.reasoning.empty()) message["reasoning_content"] = response.reasoning;
    if (!response.tool_calls.empty()) {
        if (response.content.empty()) message["content"] = nullptr;
        message["tool_calls"] = chat_tool_calls(response.tool_calls, 0, prefix);
    }
    return message;
}

auto chat_finish_reason(const text::AssistantResponse& response, const TextGeneration& generation) -> std::string {
    if (!response.tool_calls.empty()) return "tool_calls";
    return generation.generation.truncated ? "length" : "stop";
}

auto chat_usage(const TextGeneration& generation) -> ChatJson {
    const auto completion = generation.generation.token_ids.size();
    return {{"prompt_tokens", generation.stats.prompt_tokens},
            {"completion_tokens", completion},
            {"total_tokens", generation.stats.prompt_tokens + completion}};
}

auto chat_completion(const TextGeneration& generation, std::string_view id, std::string_view model,
                       std::int64_t created) -> ChatJson {
    const auto parsed = text::parse_assistant_response(generation.text);
    return {{"id", id},
            {"object", "chat.completion"},
            {"created", created},
            {"model", model},
            {"choices", ChatJson::array({{{"index", 0},
                                            {"message", chat_message(parsed, "call_" + std::string(id))},
                                            {"finish_reason", chat_finish_reason(parsed, generation)}}})},
            {"usage", chat_usage(generation)}};
}

auto chat_chunk(std::string_view id, std::string_view model, std::int64_t created, ChatJson delta,
                  ChatJson finish_reason) -> ChatJson {
    return {{"id", id},
            {"object", "chat.completion.chunk"},
            {"created", created},
            {"model", model},
            {"choices", ChatJson::array({{{"index", 0},
                                            {"delta", std::move(delta)},
                                            {"finish_reason", std::move(finish_reason)}}})}};
}

auto chat_error(const Error& error) -> ChatJson {
    return {{"error", {{"message", error.message}, {"type", "invalid_request_error"}}}};
}

auto chat_stats(const TextGeneration& generation, double elapsed_ms) -> ChatJson {
    return {{"elapsed_ms", elapsed_ms},
            {"prompt_tokens", generation.stats.prompt_tokens},
            {"reused_prompt_tokens", generation.stats.reused_prompt_tokens},
            {"decode_tokens", generation.stats.decode_tokens},
            {"prefill_ms", static_cast<double>(generation.stats.prefill_ns) / 1e6},
            {"decode_ms", static_cast<double>(generation.stats.decode_ns) / 1e6},
            {"preparation_ms", static_cast<double>(generation.stats.preparation_ns) / 1e6}};
}

} // namespace kidi::server
