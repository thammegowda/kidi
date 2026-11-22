#pragma once

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

/// Splits common model output markup into reasoning, visible text, and completed tool calls.
/// With `complete` false, partial markers and unterminated tool calls are withheld from the result.
auto parse_assistant_response(std::string_view text, bool complete = true) -> AssistantResponse;

} // namespace kidi::text
