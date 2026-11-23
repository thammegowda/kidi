#pragma once

#include "kidi/core/error.h"

#include <nlohmann/json.hpp>

#include <functional>
#include <span>
#include <stop_token>
#include <string>
#include <vector>

namespace kidi::code {

// JSON is the tool-schema and provider boundary, not an inference dependency.
using Json = nlohmann::ordered_json;

enum class Role { SYSTEM, USER, ASSISTANT, TOOL };
enum class Capability { READ, WRITE, EXECUTE, NETWORK, CUSTOM };
enum class Access { DENY, ASK, ALLOW };

struct ToolCall {
    std::string id;
    std::string name;
    Json arguments = Json::object();
};

struct Message {
    Role role = Role::USER;
    std::string content;
    std::string reasoning_content;
    std::vector<ToolCall> tool_calls;
    std::string tool_call_id;
};

struct ToolSpec {
    std::string name;
    std::string description;
    Json parameters = Json::object();
};

struct Tool {
    ToolSpec spec;
    Capability capability = Capability::CUSTOM;
    std::function<Result<std::string>(const Json&, std::stop_token)> invoke;
};

struct Permissions {
    Access read = Access::ALLOW;
    Access write = Access::ASK;
    Access execute = Access::DENY;
    Access network = Access::DENY;
    Access custom = Access::DENY;

    auto access(Capability capability) const -> Access;
};

struct RunOptions {
    Permissions permissions;
    std::size_t max_model_calls = 16;
    std::size_t max_tool_calls = 64;
    std::size_t max_history_bytes = 512 * 1024;
    std::size_t max_tool_output_bytes = 64 * 1024;
    // Sent once, without being kept in history, when a model replies empty right after tool results.
    std::string continue_prompt;
};

enum class EventKind { MODEL_START, TOOL_START, TOOL_FINISH };

struct Event {
    EventKind kind;
    std::string tool_name;
    std::string call_id;
    std::string detail;
    bool is_error = false;
};

struct Callbacks {
    std::function<Result<bool>(const ToolSpec&, const ToolCall&)> approve;
    std::function<void(const Event&)> observe;
};

using Provider = std::function<Result<Message>(std::span<const Message>, std::span<const ToolSpec>, std::stop_token)>;

struct TurnResult {
    std::string text;
    std::size_t model_calls = 0;
    std::size_t tool_calls = 0;
};

// Synchronous, serial execution. Completed messages remain in history even on failure.
// Callbacks and providers must return errors (not throw) and honor cooperative cancellation.
auto run_turn(const Provider& provider, std::span<const Tool> tools, std::vector<Message>& history, std::string input,
              const RunOptions& options = {}, const Callbacks& callbacks = {}, std::stop_token stop = {})
    -> Result<TurnResult>;

} // namespace kidi::code
