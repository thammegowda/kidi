#include "kidi/code/agent.h"

#include <algorithm>
#include <set>
#include <vector>

namespace kidi::code {
namespace {

auto invalid(std::string message) -> std::unexpected<Error> {
    return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, std::move(message)});
}

auto message_bytes(const Message& message) -> std::size_t {
    auto bytes = message.content.size() + message.reasoning_content.size() + message.tool_call_id.size() + 32;
    for (const auto& call : message.tool_calls)
        bytes += call.id.size() + call.name.size() +
                 call.arguments.dump(-1, ' ', false, Json::error_handler_t::replace).size();
    return bytes;
}

auto validate_history(std::span<const Message> history) -> Result<void> {
    std::set<std::string> pending;
    for (const auto& message : history) {
        if (message.role == Role::TOOL) {
            if (!message.tool_calls.empty() || !pending.erase(message.tool_call_id))
                return invalid("history contains an unmatched tool result");
        } else {
            if (!pending.empty() || !message.tool_call_id.empty())
                return invalid("history contains incomplete tool calls");
            if (message.role != Role::ASSISTANT && !message.tool_calls.empty())
                return invalid("only assistant messages can call tools");
            for (const auto& call : message.tool_calls) {
                if (call.id.empty() || call.name.empty() || !call.arguments.is_object() ||
                    !pending.insert(call.id).second)
                    return invalid("history contains invalid tool calls");
            }
        }
    }
    if (!pending.empty()) return invalid("history ends with incomplete tool calls");
    return {};
}

auto execute(const Tool& tool, const ToolCall& call, const RunOptions& options, const Callbacks& callbacks,
             std::stop_token stop) -> Result<std::string> {
    if (stop.stop_requested()) return std::unexpected(Error{ErrorCode::RUNTIME, "cancelled"});
    switch (options.permissions.access(tool.capability)) {
        case Access::DENY:
            return std::unexpected(Error{ErrorCode::RUNTIME, "permission denied: " + call.name});
        case Access::ASK: {
            if (!callbacks.approve)
                return std::unexpected(
                    Error{ErrorCode::RUNTIME, "approval required; no approval callback: " + call.name});
            auto approved = callbacks.approve(tool.spec, call);
            if (!approved) return std::unexpected(approved.error());
            if (!*approved) return std::unexpected(Error{ErrorCode::RUNTIME, "approval denied: " + call.name});
            break;
        }
        case Access::ALLOW:
            break;
    }
    if (stop.stop_requested()) return std::unexpected(Error{ErrorCode::RUNTIME, "cancelled"});
    auto result = tool.invoke(call.arguments, stop);
    if (result && result->size() > options.max_tool_output_bytes)
        return std::unexpected(Error{ErrorCode::RUNTIME, "tool output exceeds max_tool_output_bytes"});
    return result;
}

} // namespace

auto Permissions::access(Capability capability) const -> Access {
    switch (capability) {
        case Capability::READ:
            return read;
        case Capability::WRITE:
            return write;
        case Capability::EXECUTE:
            return execute;
        case Capability::NETWORK:
            return network;
        case Capability::CUSTOM:
            return custom;
    }
    return Access::DENY;
}

auto run_turn(const Provider& provider, std::span<const Tool> tools, std::vector<Message>& history, std::string input,
              const RunOptions& options, const Callbacks& callbacks, std::stop_token stop) -> Result<TurnResult> {
    if (!provider || input.empty() || !options.max_model_calls || !options.max_tool_calls ||
        !options.max_history_bytes || !options.max_tool_output_bytes)
        return invalid("provider, input and positive execution limits are required");
    if (auto valid = validate_history(history); !valid) return std::unexpected(valid.error());
    std::set<std::string> names;
    std::vector<ToolSpec> specs;
    for (const auto& tool : tools) {
        if (tool.spec.name.empty() || !tool.invoke || !tool.spec.parameters.is_object() ||
            !names.insert(tool.spec.name).second)
            return invalid("tools require unique names, object schemas and handlers");
        if (options.permissions.access(tool.capability) != Access::DENY) specs.push_back(tool.spec);
    }
    std::size_t bytes = 0;
    for (const auto& message : history) bytes += message_bytes(message);
    Message user{.role = Role::USER, .content = std::move(input)};
    if (bytes + message_bytes(user) > options.max_history_bytes) return invalid("history exceeds max_history_bytes");
    if (stop.stop_requested()) return std::unexpected(Error{ErrorCode::RUNTIME, "cancelled"});
    bytes += message_bytes(user);
    history.push_back(std::move(user));
    TurnResult turn;
    bool nudged = false;
    std::vector<Message> nudge_request;
    for (; turn.model_calls < options.max_model_calls;) {
        if (stop.stop_requested()) return std::unexpected(Error{ErrorCode::RUNTIME, "cancelled"});
        if (bytes > options.max_history_bytes)
            return std::unexpected(Error{ErrorCode::RUNTIME, "history exceeds max_history_bytes"});
        if (callbacks.observe) callbacks.observe({.kind = EventKind::MODEL_START});
        ++turn.model_calls;
        auto reply = nudge_request.empty() ? provider(history, specs, stop) : provider(nudge_request, specs, stop);
        nudge_request.clear();
        if (!reply) return std::unexpected(reply.error());
        if (stop.stop_requested()) return std::unexpected(Error{ErrorCode::RUNTIME, "cancelled"});
        if (reply->role != Role::ASSISTANT || !reply->tool_call_id.empty())
            return invalid("provider must return an assistant message");
        if (reply->tool_calls.size() > options.max_tool_calls - turn.tool_calls)
            return std::unexpected(Error{ErrorCode::RUNTIME, "tool call limit exceeded"});
        std::set<std::string> ids;
        for (const auto& call : reply->tool_calls) {
            if (call.id.empty() || call.name.empty() || !call.arguments.is_object() || !ids.insert(call.id).second)
                return invalid("provider returned invalid or duplicate tool calls");
        }
        if (bytes + message_bytes(*reply) > options.max_history_bytes)
            return std::unexpected(Error{ErrorCode::RUNTIME, "provider response exceeds max_history_bytes"});
        if (reply->tool_calls.empty()) {
            if (reply->content.empty()) {
                if (nudged || options.continue_prompt.empty() || history.back().role != Role::TOOL)
                    return invalid("provider returned an empty assistant response");
                nudged = true;
                nudge_request = history;
                nudge_request.push_back({.role = Role::USER, .content = options.continue_prompt});
                continue;
            }
            turn.text = reply->content;
            history.push_back(std::move(*reply));
            return turn;
        }
        // Reserve enough room to pair every accepted call, even on denial or cancellation.
        std::size_t reserved = 0;
        for (const auto& call : reply->tool_calls) reserved += call.id.size() + 32 + 128;
        if (reserved > options.max_history_bytes - bytes - message_bytes(*reply))
            return std::unexpected(Error{ErrorCode::RUNTIME, "insufficient history budget for tool results"});
        const auto calls = reply->tool_calls;
        bytes += message_bytes(*reply);
        history.push_back(std::move(*reply));
        for (const auto& call : calls) {
            reserved -= call.id.size() + 32 + 128;
            ++turn.tool_calls;
            if (callbacks.observe)
                callbacks.observe({.kind = EventKind::TOOL_START, .tool_name = call.name, .call_id = call.id});
            const auto tool =
                std::ranges::find_if(tools, [&](const Tool& item) { return item.spec.name == call.name; });
            auto output = tool == tools.end() ? Result<std::string>(std::unexpected(
                                                    Error{ErrorCode::INVALID_ARGUMENT, "unknown tool: " + call.name}))
                                              : execute(*tool, call, options, callbacks, stop);
            auto content =
                output ? std::move(*output)
                       : Json{{"error", output.error().message}}.dump(-1, ' ', false, Json::error_handler_t::replace);
            if (content.size() > options.max_history_bytes - bytes - reserved - call.id.size() - 32) {
                output = std::unexpected(Error{ErrorCode::RUNTIME, "tool result exceeds history budget"});
                content = R"({"error":"tool result exceeds history budget"})";
            }
            if (callbacks.observe) callbacks.observe({EventKind::TOOL_FINISH, call.name, call.id, content, !output});
            Message result{.role = Role::TOOL, .content = content, .tool_call_id = call.id};
            bytes += message_bytes(result);
            history.push_back(std::move(result));
        }
    }
    if (stop.stop_requested()) return std::unexpected(Error{ErrorCode::RUNTIME, "cancelled"});
    return std::unexpected(Error{ErrorCode::RUNTIME, "model call limit exceeded"});
}

} // namespace kidi::code
