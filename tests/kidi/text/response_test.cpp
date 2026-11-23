#include "kidi/text/response.h"

#include <gtest/gtest.h>

namespace {

using kidi::text::parse_assistant_response;

const std::string INKLING_THINKING = "<|message_model|><|content_thinking|>add them<|end_message|>";
const std::string INKLING_CALLS =
    "<|message_model|>read<|content_invoke_tool_json|>{\"name\":\"read\",\"args\":{\"path\":\"a.py\"}}"
    "<|end_message|><|message_model|>bash<|content_invoke_tool_json|>{\"args\":{\"command\":\"ls\"}}"
    "<|end_message|><|content_model_end_sampling|>";

auto grows_monotonically(std::string_view text) -> bool {
    kidi::text::AssistantResponse previous;
    for (std::size_t size = 0; size <= text.size(); ++size) {
        const auto current = parse_assistant_response(text.substr(0, size), false);
        if (!current.reasoning.starts_with(previous.reasoning) || !current.content.starts_with(previous.content) ||
            current.tool_calls.size() < previous.tool_calls.size())
            return false;
        previous = current;
    }
    return true;
}

TEST(ResponseTest, InklingThinkingAndText) {
    const std::string inkling_answer =
        "<|message_model|><|content_text|>391<|end_message|><|content_model_end_sampling|>";
    auto plain = parse_assistant_response(INKLING_THINKING + inkling_answer);
    EXPECT_TRUE(plain.reasoning == "add them" && plain.content == "391" && plain.tool_calls.empty())
        << "Inkling thinking and text";
}

TEST(ResponseTest, InklingToolCalls) {
    auto tools = parse_assistant_response(INKLING_THINKING + INKLING_CALLS);
    EXPECT_TRUE(tools.tool_calls.size() == 2 && tools.content.empty()) << "Inkling multiple tool calls";
    EXPECT_TRUE(tools.tool_calls.size() == 2 && tools.tool_calls[0].name == "read" &&
                tools.tool_calls[0].arguments["path"] == "a.py" && tools.tool_calls[1].name == "bash" &&
                tools.tool_calls[1].arguments["command"] == "ls")
        << "tool names come from the payload or header";
}

TEST(ResponseTest, ChatMLReasoningAndToolCalls) {
    auto thinking = parse_assistant_response("<think>reason</think>answer");
    EXPECT_TRUE(thinking.reasoning == "reason" && thinking.content == "answer") << "ChatML reasoning";
    auto json_call =
        parse_assistant_response("<tool_call>{\"name\":\"read\",\"arguments\":{\"path\":\"a.py\"}}</tool_call>");
    EXPECT_TRUE(json_call.tool_calls.size() == 1 && json_call.tool_calls[0].name == "read" &&
                json_call.tool_calls[0].arguments["path"] == "a.py")
        << "JSON tool call";

    auto xml_call = parse_assistant_response(
        "<tool_call><function=read><parameter=path>a.py</parameter><parameter=line>4</parameter></function></tool_call>");
    EXPECT_TRUE(xml_call.tool_calls.size() == 1 && xml_call.tool_calls[0].name == "read" &&
                xml_call.tool_calls[0].arguments["path"] == "a.py" && xml_call.tool_calls[0].arguments["line"] == 4)
        << "XML function call with typed parameter values";
}

TEST(ResponseTest, GemmaChannelAndToolCall) {
    auto gemma = parse_assistant_response(
        "<|channel>thought\nreason<channel|>answer<|tool_call>call:read{\"path\":\"a.py\"}<tool_call|>");
    EXPECT_TRUE(gemma.reasoning == "reason" && gemma.content == "answer" && gemma.tool_calls.size() == 1 &&
                gemma.tool_calls[0].name == "read" && gemma.tool_calls[0].arguments["path"] == "a.py")
        << "Gemma channel and tool call";
}

TEST(ResponseTest, GemmaQuotedArguments) {
    auto gemma = parse_assistant_response(
        "<|tool_call>call:write{path:<|\"|>a, b.txt<|\"|>,limit:3,tags:[<|\"|>x<|\"|>,true],"
        "meta:{note:<|\"|>}{<|\"|>}}<tool_call|>");
    ASSERT_EQ(gemma.tool_calls.size(), 1u);
    EXPECT_EQ(gemma.tool_calls[0].arguments.dump(),
              R"({"path":"a, b.txt","limit":3,"tags":["x",true],"meta":{"note":"}{"}})");
}

TEST(ResponseTest, MalformedInputStaysVisible) {
    auto broken = parse_assistant_response("<|message_model|>x<|content_invoke_tool_json|>{oops<|end_message|>");
    EXPECT_TRUE(broken.tool_calls.empty() && broken.content == "{oops") << "invalid Inkling tool JSON stays visible";
    auto literal = parse_assistant_response("a <|not a marker", false);
    EXPECT_EQ(literal.content, "a <|not a marker") << "non-marker text is emitted";
}

TEST(ResponseTest, StreamingMonotonic) {
    EXPECT_TRUE(grows_monotonically(INKLING_THINKING + INKLING_CALLS))
        << "Inkling streaming parses grow monotonically";
    EXPECT_TRUE(grows_monotonically("<think>reason</think>answer<tool_call>{\"name\":\"read\",\"args\":{}}</tool_call>"))
        << "tagged streaming parses grow monotonically";
    EXPECT_TRUE(parse_assistant_response("<tool_call>{\"name\":\"read\"", false).tool_calls.empty())
        << "unterminated tool calls are withheld while streaming";
}

} // namespace
