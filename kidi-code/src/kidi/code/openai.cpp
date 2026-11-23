#include "kidi/code/openai_transport.h"

#include <limits>

namespace kidi::code {
namespace {

auto invalid(std::string message) -> std::unexpected<Error> {
    return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, std::move(message)});
}

auto role_name(Role role) -> const char* {
    switch (role) {
        case Role::SYSTEM:
            return "system";
        case Role::USER:
            return "user";
        case Role::ASSISTANT:
            return "assistant";
        case Role::TOOL:
            return "tool";
    }
    return "invalid";
}

auto parse_reply(const Json& body) -> Result<Message> {
    if (!body.contains("choices") || !body["choices"].is_array() || body["choices"].size() != 1)
        return invalid("OpenAI response must contain exactly one choice");
    const auto& choice = body["choices"][0];
    if (!choice.is_object() || !choice.contains("message") || !choice["message"].is_object())
        return invalid("OpenAI response is missing an assistant message");
    const auto& message = choice["message"];
    if (!message.contains("role") || message["role"] != "assistant") return invalid("invalid OpenAI message role");
    Message result{.role = Role::ASSISTANT};
    if (message.contains("content") && !message["content"].is_null()) {
        if (!message["content"].is_string()) return invalid("OpenAI message content must be a string or null");
        result.content = message["content"].get<std::string>();
    }
    if (message.contains("reasoning_content") && !message["reasoning_content"].is_null()) {
        if (!message["reasoning_content"].is_string()) return invalid("reasoning_content must be a string or null");
        result.reasoning_content = message["reasoning_content"].get<std::string>();
    }
    if (message.contains("tool_calls") && !message["tool_calls"].is_null()) {
        if (!message["tool_calls"].is_array()) return invalid("OpenAI tool_calls must be an array");
        for (const auto& call : message["tool_calls"]) {
            if (!call.is_object() || !call.contains("id") || !call["id"].is_string() || !call.contains("type") ||
                call["type"] != "function" || !call.contains("function") || !call["function"].is_object())
                return invalid("invalid OpenAI function tool call");
            const auto& function = call["function"];
            if (!function.contains("name") || !function["name"].is_string() || !function.contains("arguments") ||
                !function["arguments"].is_string())
                return invalid("OpenAI function name and arguments must be strings");
            auto arguments = Json::parse(function["arguments"].get_ref<const std::string&>(), nullptr, false);
            if (arguments.is_discarded() || !arguments.is_object())
                return invalid("tool arguments must encode an object");
            result.tool_calls.push_back(
                {call["id"].get<std::string>(), function["name"].get<std::string>(), std::move(arguments)});
        }
    }
    if (!choice.contains("finish_reason") || !choice["finish_reason"].is_string())
        return invalid("OpenAI response is missing finish_reason");
    const auto reason = choice["finish_reason"].get<std::string>();
    if (reason != "stop" && reason != "tool_calls")
        return std::unexpected(Error{ErrorCode::RUNTIME, "OpenAI generation did not complete: " + reason});
    if ((reason == "tool_calls") != !result.tool_calls.empty())
        return invalid("OpenAI finish_reason does not match tool_calls");
    return result;
}

auto encode_request(const OpenAIOptions& options, std::span<const Message> messages, std::span<const ToolSpec> tools)
    -> Result<std::string> {
    std::string payload;
    try {
        Json request{{"model", options.model},
                     {"stream", false},
                     {"max_tokens", options.max_tokens},
                     {"messages", Json::array()}};
        for (const auto& message : messages) {
            Json item{{"role", role_name(message.role)}, {"content", message.content}};
            if (!message.reasoning_content.empty()) item["reasoning_content"] = message.reasoning_content;
            if (message.role == Role::TOOL) item["tool_call_id"] = message.tool_call_id;
            if (!message.tool_calls.empty()) {
                item["tool_calls"] = Json::array();
                for (const auto& call : message.tool_calls)
                    item["tool_calls"].push_back(
                        {{"id", call.id},
                         {"type", "function"},
                         {"function", {{"name", call.name}, {"arguments", call.arguments.dump()}}}});
            }
            request["messages"].push_back(std::move(item));
        }
        if (!tools.empty()) {
            request["tools"] = Json::array();
            request["tool_choice"] = "auto";
            for (const auto& tool : tools)
                request["tools"].push_back(
                    {{"type", "function"},
                     {"function",
                      {{"name", tool.name}, {"description", tool.description}, {"parameters", tool.parameters}}}});
        }
        payload = request.dump();
    } catch (const Json::type_error& error) {
        return invalid("request is not valid UTF-8 JSON: " + std::string(error.what()));
    }
    return payload;
}

} // namespace

auto completion_provider(OpenAIOptions options, CompletionTransport transport) -> Result<Provider> {
    if (options.model.empty() || !options.max_tokens || !options.max_response_bytes || options.timeout.count() <= 0 ||
        options.timeout.count() > std::numeric_limits<long>::max())
        return invalid("model and positive OpenAI limits are required");
    return Provider([options = std::move(options), transport = std::move(transport)](
                        std::span<const Message> messages, std::span<const ToolSpec> tools,
                        std::stop_token stop) -> Result<Message> {
        if (stop.stop_requested()) return std::unexpected(Error{ErrorCode::RUNTIME, "cancelled"});
        auto payload = encode_request(options, messages, tools);
        if (!payload) return std::unexpected(payload.error());
        auto response = transport(options, std::move(*payload), stop);
        if (!response) return std::unexpected(response.error());
        auto body = Json::parse(*response, nullptr, false);
        if (body.is_discarded() || !body.is_object()) return invalid("OpenAI response is not a JSON object");
        return parse_reply(body);
    });
}

} // namespace kidi::code
