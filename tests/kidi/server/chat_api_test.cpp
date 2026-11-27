#include "kidi/server/chat_api.h"

#include <gtest/gtest.h>

namespace {
using kidi::server::GenerationOptions;
using kidi::server::ChatJson;
using kidi::server::prepare_chat_request;

TEST(ChatApiTest, PreparesChatRequest) {
    auto body = ChatJson::object();
    body["messages"] = ChatJson::array();
    auto message = ChatJson::object();
    message["role"] = "assistant";
    message["tool_calls"] = ChatJson::array();
    auto call = ChatJson::object();
    call["function"]["arguments"] = "{}";
    message["tool_calls"].push_back(std::move(call));
    body["messages"].push_back(std::move(message));
    body["tools"] = ChatJson::array({{{"type", "function"}, {"function", {{"name", "read"}}}}});
    body["reasoning_effort"] = "high";
    body["stream"] = true;
    body["stream_options"] = {{"include_usage", true}};
    body["max_completion_tokens"] = 64;
    body["temperature"] = 0.7;
    body["top_p"] = 0.9;
    body["top_k"] = 20;
    body["seed"] = 42;
    auto request = prepare_chat_request(body, GenerationOptions{.maximum_new_tokens = 128});
    ASSERT_TRUE(request.has_value()) << "valid chat request: " << request.error().message;
    EXPECT_TRUE(request->stream) << "stream options are parsed";
    EXPECT_TRUE(request->include_usage) << "stream options are parsed";
    EXPECT_EQ(request->generation.maximum_new_tokens, 64) << "generation options are normalized";
    EXPECT_TRUE(request->generation.stream_text) << "generation options are normalized";
    EXPECT_FLOAT_EQ(request->generation.temperature, 0.7F) << "sampling options are parsed";
    EXPECT_FLOAT_EQ(request->generation.top_p, 0.9F) << "sampling options are parsed";
    EXPECT_EQ(request->generation.top_k, 20) << "sampling options are parsed";
    EXPECT_EQ(request->generation.seed, 42) << "sampling options are parsed";
    for (const auto& [field, value] : {std::pair{"temperature", ChatJson(2.5)}, {"top_p", ChatJson(0)},
                                       {"top_k", ChatJson(-1)}, {"seed", ChatJson("x")}}) {
        auto bad = body;
        bad[field] = value;
        EXPECT_FALSE(prepare_chat_request(bad, {})) << field << " is validated";
    }
    EXPECT_EQ(request->template_args["reasoning_effort"], "high")
        << "template arguments include tools and reasoning effort";
    EXPECT_TRUE(request->template_args.contains("tools")) << "template arguments include tools and reasoning effort";
    EXPECT_TRUE(request->messages[0]["tool_calls"][0]["function"]["arguments"].is_object())
        << "tool-call JSON string arguments are normalized";
}

TEST(ChatApiTest, RejectsMalformedToolArguments) {
    auto invalid_body = ChatJson::object();
    invalid_body["messages"] = ChatJson::array();
    auto invalid_message = ChatJson::object();
    invalid_message["tool_calls"] = ChatJson::array();
    invalid_message["tool_calls"].push_back({{"function", {{"arguments", "{oops"}}}});
    invalid_body["messages"].push_back(std::move(invalid_message));
    auto invalid = prepare_chat_request(invalid_body, {});
    ASSERT_FALSE(invalid) << "malformed tool arguments are rejected";
    EXPECT_EQ(invalid.error().code, kidi::ErrorCode::INVALID_ARGUMENT) << "malformed tool arguments are rejected";
}

TEST(ChatApiTest, BuildsCompletion) {
    kidi::server::TextGeneration generation;
    generation.text = R"(<tool_call>{"name":"read","arguments":{"path":"a.py"}}</tool_call>)";
    generation.generation.token_ids = {1, 2};
    generation.stats.prompt_tokens = 3;
    const auto completion = kidi::server::chat_completion(generation, "chatcmpl-7", "model", 123);
    EXPECT_EQ(completion["choices"][0]["message"]["tool_calls"][0]["id"], "call_chatcmpl-7_0")
        << "tool-call IDs are unique to the completion";
    EXPECT_EQ(completion["usage"]["prompt_tokens"], 3) << "completion usage matches generation statistics";
    EXPECT_EQ(completion["usage"]["completion_tokens"], 2) << "completion usage matches generation statistics";
}
} // namespace
