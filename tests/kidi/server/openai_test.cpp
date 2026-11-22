#include "kidi/server/openai.h"

#include <gtest/gtest.h>

namespace {
using kidi::server::GenerationOptions;
using kidi::server::OpenAIJson;
using kidi::server::prepare_openai_chat_request;

TEST(OpenAITest, PreparesChatRequest) {
    auto body = OpenAIJson::object();
    body["messages"] = OpenAIJson::array();
    auto message = OpenAIJson::object();
    message["role"] = "assistant";
    message["tool_calls"] = OpenAIJson::array();
    auto call = OpenAIJson::object();
    call["function"]["arguments"] = "{}";
    message["tool_calls"].push_back(std::move(call));
    body["messages"].push_back(std::move(message));
    body["tools"] = OpenAIJson::array({{{"type", "function"}, {"function", {{"name", "read"}}}}});
    body["reasoning_effort"] = "high";
    body["stream"] = true;
    body["stream_options"] = {{"include_usage", true}};
    body["max_completion_tokens"] = 64;
    auto request = prepare_openai_chat_request(body, GenerationOptions{.maximum_new_tokens = 128});
    ASSERT_TRUE(request.has_value()) << "valid OpenAI chat request: " << request.error().message;
    EXPECT_TRUE(request->stream) << "stream options are parsed";
    EXPECT_TRUE(request->include_usage) << "stream options are parsed";
    EXPECT_EQ(request->generation.maximum_new_tokens, 64) << "generation options are normalized";
    EXPECT_TRUE(request->generation.stream_text) << "generation options are normalized";
    EXPECT_EQ(request->template_args["reasoning_effort"], "high")
        << "template arguments include tools and reasoning effort";
    EXPECT_TRUE(request->template_args.contains("tools")) << "template arguments include tools and reasoning effort";
    EXPECT_TRUE(request->messages[0]["tool_calls"][0]["function"]["arguments"].is_object())
        << "tool-call JSON string arguments are normalized";
}

TEST(OpenAITest, RejectsMalformedToolArguments) {
    auto invalid_body = OpenAIJson::object();
    invalid_body["messages"] = OpenAIJson::array();
    auto invalid_message = OpenAIJson::object();
    invalid_message["tool_calls"] = OpenAIJson::array();
    invalid_message["tool_calls"].push_back({{"function", {{"arguments", "{oops"}}}});
    invalid_body["messages"].push_back(std::move(invalid_message));
    auto invalid = prepare_openai_chat_request(invalid_body, {});
    ASSERT_FALSE(invalid) << "malformed tool arguments are rejected";
    EXPECT_EQ(invalid.error().code, kidi::ErrorCode::INVALID_ARGUMENT) << "malformed tool arguments are rejected";
}

TEST(OpenAITest, BuildsCompletion) {
    kidi::server::TextGeneration generation;
    generation.text = R"(<tool_call>{"name":"read","arguments":{"path":"a.py"}}</tool_call>)";
    generation.generation.token_ids = {1, 2};
    generation.stats.prompt_tokens = 3;
    const auto completion = kidi::server::openai_completion(generation, "chatcmpl-7", "model", 123);
    EXPECT_EQ(completion["choices"][0]["message"]["tool_calls"][0]["id"], "call_chatcmpl-7_0")
        << "tool-call IDs are unique to the completion";
    EXPECT_EQ(completion["usage"]["prompt_tokens"], 3) << "completion usage matches generation statistics";
    EXPECT_EQ(completion["usage"]["completion_tokens"], 2) << "completion usage matches generation statistics";
}
} // namespace
