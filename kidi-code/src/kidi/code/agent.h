#pragma once

#include "kidi/core/error.h"

#include <nlohmann/json.hpp>

#include <concepts>
#include <functional>
#include <optional>
#include <span>
#include <stop_token>
#include <string>
#include <type_traits>
#include <utility>
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

struct TokenUsage {
    std::size_t prompt_tokens = 0;
    std::size_t completion_tokens = 0;
    /// Server-side timings from Kidi's `kidi_stats`; 0 when the server does not report them.
    double prefill_ms = 0, decode_ms = 0;
};

struct Message {
    Role role = Role::USER;
    std::string content;
    std::string reasoning_content;
    std::vector<ToolCall> tool_calls;
    std::string tool_call_id;
    bool is_summary = false;
    std::optional<TokenUsage> usage;
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
    std::size_t max_history_bytes = 4 * 1024 * 1024; // about 1M tokens; the exact token budget usually binds first
    std::size_t max_tool_output_bytes = 64 * 1024;
    // Sent once, without being kept in history, when a model replies empty right after tool results.
    std::string continue_prompt;
    std::string compact_prompt;
    // Repeated compaction within one turn means the context is too small for the task; stop and say so.
    std::size_t max_compactions = 3;
    std::size_t max_summary_calls = 16;
};

enum class EventKind { MODEL_START, TOOL_START, TOOL_FINISH, COMPACT_START, COMPACT_FINISH };

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

struct ContextUsage {
    std::size_t prompt_tokens = 0;
    std::size_t context_tokens = 0;
    std::size_t completion_tokens = 0;
    std::size_t prompt_limit_tokens = 0;
};

struct Provider {
    using Completion =
        std::function<Result<Message>(std::span<const Message>, std::span<const ToolSpec>, std::stop_token)>;
    Completion complete;
    std::function<Result<ContextUsage>(std::span<const Message>, std::span<const ToolSpec>, std::stop_token)> context;
    std::function<Result<ContextUsage>(std::span<const Message>, std::stop_token)> summary_context;
    std::function<Result<Message>(std::span<const Message>, std::size_t, std::stop_token)> summarize;
    std::string model;

    Provider() = default;
    template <class F>
        requires(!std::same_as<std::remove_cvref_t<F>, Provider> && std::constructible_from<Completion, F>)
    Provider(F&& completion) : complete(std::forward<F>(completion)) {}
    explicit operator bool() const { return static_cast<bool>(complete); }
    auto operator()(std::span<const Message> messages, std::span<const ToolSpec> tools, std::stop_token stop) const
        -> Result<Message> {
        return complete(messages, tools, stop);
    }
};

struct TurnResult {
    std::string text;
    std::size_t model_calls = 0;
    std::size_t tool_calls = 0;
    std::size_t compactions = 0;
};

auto compact_history(const Provider& provider, std::span<const ToolSpec> tools, std::vector<Message>& history,
                     const RunOptions& options, const Callbacks& callbacks = {}, std::stop_token stop = {})
    -> Result<void>;

// Synchronous, serial execution. Completed messages remain in history even on failure.
// Callbacks and providers must return errors (not throw) and honor cooperative cancellation.
auto run_turn(const Provider& provider, std::span<const Tool> tools, std::vector<Message>& history, std::string input,
              const RunOptions& options = {}, const Callbacks& callbacks = {}, std::stop_token stop = {})
    -> Result<TurnResult>;

} // namespace kidi::code
