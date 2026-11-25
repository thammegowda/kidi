#pragma once

#include <cstddef>
#include <nlohmann/json.hpp>
#include <string>
#include <string_view>
#include <vector>

namespace kidi::text {

struct ToolCall {
    std::string name;
    nlohmann::ordered_json arguments = nlohmann::ordered_json::object();
};

/// An assistant turn split into the OpenAI chat fields.
struct AssistantResponse {
    std::string reasoning, content;
    std::vector<ToolCall> tool_calls;
};

struct ResponseDelta {
    std::string reasoning, content;
    std::size_t first_tool_call = 0;
    std::vector<ToolCall> tool_calls;

    auto empty() const noexcept -> bool { return reasoning.empty() && content.empty() && tool_calls.empty(); }
};

/// Splits common model output markup into reasoning, visible text, and completed tool calls.
/// With `complete` false, partial markers and unterminated tool calls are withheld from the result.
auto parse_assistant_response(std::string_view text, bool complete = true) -> AssistantResponse;

/// Accumulates streamed assistant text and emits monotonic deltas as complete markup settles.
class AssistantResponseStream {
public:
    auto append(std::string_view delta) -> ResponseDelta;
    auto finish(std::string_view complete_text) -> ResponseDelta;
    auto response() const noexcept -> const AssistantResponse& { return emitted_; }

private:
    auto advance(std::string_view text, bool complete) -> ResponseDelta;

    std::string raw_;
    AssistantResponse emitted_;
};

} // namespace kidi::text
