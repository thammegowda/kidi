#include "kidi/code/openai.h"
#include "kidi/code/openai_transport.h"
#include "kidi/code/workspace.h"

#include <gtest/gtest.h>
#include <httplib.h>

#include <atomic>
#include <thread>

namespace kidi::code {
namespace {

TEST(CodeProvider, ContextAndSummaryRequests) {
    int completions = 0, inspections = 0;
    auto provider = completion_provider(
        {.model = "test-model"},
        [&](const OpenAIOptions&, std::string payload, std::stop_token, std::string_view route) -> Result<std::string> {
            const auto request = Json::parse(payload);
            if (route == "/chat/context") {
                ++inspections;
                if (request.contains("reasoning_effort")) {
                    EXPECT_EQ(request["reasoning_effort"], "none");
                    EXPECT_EQ(request["chat_template_kwargs"]["enable_thinking"], false);
                    EXPECT_FALSE(request.contains("tools"));
                    return R"({"prompt_tokens":77,"context_tokens":4096})";
                }
                EXPECT_EQ(request["tools"][0]["function"]["name"], "inspect");
                EXPECT_EQ(request["messages"][1]["tool_calls"][0]["function"]["arguments"], R"({"path":"source"})");
                EXPECT_EQ(request["messages"][2]["tool_call_id"], "read");
                return R"({"prompt_tokens":8000,"context_tokens":4096})";
            }
            ++completions;
            EXPECT_EQ(route, "/chat/completions");
            EXPECT_EQ(request["max_tokens"], 200);
            EXPECT_FALSE(request.contains("tools"));
            EXPECT_EQ(request["reasoning_effort"], "none");
            EXPECT_EQ(request["chat_template_kwargs"]["enable_thinking"], false);
            return R"({"choices":[{"message":{"role":"assistant","content":"handoff"},"finish_reason":"stop"}]})";
        });
    ASSERT_TRUE(provider);
    std::vector<Message> history{{.role = Role::USER, .content = "Read source"},
                                 {.role = Role::ASSISTANT, .tool_calls = {{"read", "inspect", {{"path", "source"}}}}},
                                 {.role = Role::TOOL, .content = "contents", .tool_call_id = "read"}};
    ToolSpec tool{.name = "inspect"};
    auto usage = provider->context(history, std::span(&tool, 1), {});
    ASSERT_TRUE(usage) << usage.error().message;
    EXPECT_EQ(usage->prompt_tokens, 8000);
    EXPECT_EQ(usage->context_tokens, 4096);
    EXPECT_EQ(usage->completion_tokens, 4096);
    auto summary_usage = provider->summary_context(history, {});
    ASSERT_TRUE(summary_usage);
    EXPECT_EQ(summary_usage->prompt_tokens, 77);
    auto summary = provider->summarize(history, 200, {});
    ASSERT_TRUE(summary) << summary.error().message;
    EXPECT_EQ(summary->content, "handoff");
    EXPECT_EQ(inspections, 2);
    EXPECT_EQ(completions, 1);
}

auto response(Json message, std::string reason = "stop") -> Json {
    return {{"choices", Json::array({{{"message", std::move(message)}, {"finish_reason", std::move(reason)}}})}};
}

TEST(CodeProvider, UsageIsMeasuredNotEstimated) {
    auto body = response({{"role", "assistant"}, {"content", "hello"}});
    body["usage"] = {{"prompt_tokens", 120}, {"completion_tokens", 9}};
    auto provider = completion_provider({.model = "actual-model"},
                                        [&](auto&, auto, auto, auto) -> Result<std::string> { return body.dump(); });
    ASSERT_TRUE(provider);
    EXPECT_EQ(provider->model, "actual-model");
    auto reply = (*provider)({}, {}, {});
    ASSERT_TRUE(reply);
    ASSERT_TRUE(reply->usage);
    EXPECT_EQ(reply->usage->prompt_tokens, 120);
    EXPECT_EQ(reply->usage->completion_tokens, 9);
    body.erase("usage");
    reply = (*provider)({}, {}, {});
    ASSERT_TRUE(reply);
    EXPECT_FALSE(reply->usage);
    for (const auto& usage : {Json{{"prompt_tokens", -1}, {"completion_tokens", 2}},
                              Json{{"prompt_tokens", 1}, {"completion_tokens", "2"}}, Json::object()}) {
        body["usage"] = usage;
        EXPECT_FALSE((*provider)({}, {}, {}));
    }
}

TEST(CodeProvider, BadContextAndSummary) {
    for (const auto* body :
         {"{}", R"({"prompt_tokens":-1,"context_tokens":4096})", R"({"prompt_tokens":1,"context_tokens":0})",
          R"({"prompt_tokens":"1","context_tokens":4096})"}) {
        auto provider = completion_provider(
            {.model = "test"}, [body](auto&, auto, auto, auto) -> Result<std::string> { return std::string(body); });
        ASSERT_TRUE(provider);
        EXPECT_FALSE(provider->context({}, {}, {}));
    }
    std::string content = "partial";
    auto provider = completion_provider({.model = "test"}, [&](auto&, auto, auto, auto) -> Result<std::string> {
        return response({{"role", "assistant"}, {"content", content}}, "length").dump();
    });
    ASSERT_TRUE(provider);
    // A truncated handoff is usable; a truncated answer or empty handoff is not.
    auto summary = provider->summarize({}, 200, {});
    ASSERT_TRUE(summary);
    EXPECT_EQ(summary->content, "partial");
    EXPECT_FALSE((*provider)({}, {}, {}));
    content.clear();
    EXPECT_FALSE(provider->summarize({}, 200, {}));
}

class CodeOpenAI : public testing::Test {
protected:
    httplib::Server server_;
    std::jthread thread_;
    std::string base_url_;

    auto SetUp() -> void override {
        const auto port = server_.bind_to_any_port("127.0.0.1");
        ASSERT_GT(port, 0);
        base_url_ = "http://127.0.0.1:" + std::to_string(port) + "/v1";
        // Register routes before starting listen_after_bind in each test.
    }

    auto start() -> void {
        thread_ = std::jthread([this] { server_.listen_after_bind(); });
        server_.wait_until_ready();
    }

    auto TearDown() -> void override {
        server_.stop();
        if (thread_.joinable()) thread_.join();
    }
};

TEST_F(CodeOpenAI, HttpToolRoundTrip) {
    int requests = 0;
    server_.Post("/v1/chat/completions", [&](const httplib::Request& request, httplib::Response& reply) {
        EXPECT_FALSE(request.has_header("Authorization"));
        const auto body = Json::parse(request.body);
        EXPECT_EQ(body["model"], "test-model");
        EXPECT_EQ(body["stream"], false);
        EXPECT_EQ(body["tools"][0]["function"]["name"], "lookup");
        if (++requests == 1) {
            EXPECT_EQ(body["messages"].back()["role"], "user");
            reply.set_content(
                response({{"role", "assistant"},
                          {"content", nullptr},
                          {"reasoning_content", "inspect key"},
                          {"tool_calls",
                           Json::array({{{"id", "id-1"},
                                         {"type", "function"},
                                         {"function", {{"name", "lookup"}, {"arguments", R"({"key":"value"})"}}}}})}},
                         "tool_calls")
                    .dump(),
                "application/json");
        } else {
            EXPECT_EQ(body["messages"].back()["role"], "tool");
            EXPECT_EQ(body["messages"].back()["tool_call_id"], "id-1");
            EXPECT_EQ(body["messages"].back()["content"], "found");
            EXPECT_EQ(body["messages"][1]["tool_calls"][0]["function"]["arguments"], R"({"key":"value"})");
            EXPECT_EQ(body["messages"][1]["reasoning_content"], "inspect key");
            reply.set_content(response({{"role", "assistant"}, {"content", "all done"}}).dump(), "application/json");
        }
    });
    start();
    auto provider = openai_provider({.base_url = base_url_ + "/", .model = "test-model"});
    ASSERT_TRUE(provider);
    int executions = 0;
    Tool tool{{"lookup", "lookup a key", {{"type", "object"}}},
              Capability::READ,
              [&](const Json& args, std::stop_token) -> Result<std::string> {
                  EXPECT_EQ(args["key"], "value");
                  ++executions;
                  return std::string("found");
              }};
    std::vector<Message> history;
    auto result = run_turn(*provider, std::span(&tool, 1), history, "look up key");
    ASSERT_TRUE(result) << result.error().message;
    EXPECT_EQ(result->text, "all done");
    EXPECT_EQ(executions, 1);
    EXPECT_EQ(requests, 2);
}

TEST_F(CodeOpenAI, HttpErrorsNoRetry) {
    int requests = 0;
    server_.Post("/v1/chat/completions", [&](const auto&, auto& reply) {
        ++requests;
        reply.status = 429;
        reply.set_content(R"({"error":{"message":"rate limit"}})", "application/json");
    });
    start();
    auto provider = openai_provider({.base_url = base_url_});
    ASSERT_TRUE(provider);
    auto result = (*provider)({}, {}, {});
    ASSERT_FALSE(result);
    EXPECT_NE(result.error().message.find("429"), std::string::npos);
    EXPECT_EQ(requests, 1);
}

TEST_F(CodeOpenAI, RefusesRedirect) {
    int redirects = 0;
    server_.Post("/v1/chat/completions", [](const auto&, auto& reply) { reply.set_redirect("/stolen", 307); });
    server_.Post("/stolen", [&](const auto&, auto&) { ++redirects; });
    start();
    auto provider = openai_provider({.base_url = base_url_});
    ASSERT_TRUE(provider);
    EXPECT_FALSE((*provider)({}, {}, {}));
    EXPECT_EQ(redirects, 0);
}

TEST_F(CodeOpenAI, RejectsBadAndPartialReplies) {
    std::string body;
    server_.Post("/v1/chat/completions",
                 [&](const auto&, auto& reply) { reply.set_content(body, "application/json"); });
    start();
    auto provider = openai_provider({.base_url = base_url_});
    ASSERT_TRUE(provider);
    const std::vector<std::string> replies{
        "not json",
        "[]",
        R"({"choices":[]})",
        response({{"role", "assistant"}, {"content", "partial"}}, "length").dump(),
        response({{"role", "assistant"}, {"content", "blocked"}}, "content_filter").dump(),
        response({{"role", "assistant"}, {"content", Json::array()}}).dump(),
        response(
            {{"role", "assistant"},
             {"tool_calls",
              Json::array({{{"id", "x"}, {"type", "function"}, {"function", {{"name", "bad"}, {"arguments", "["}}}}})}},
            "tool_calls")
            .dump(),
    };
    for (const auto& text : replies) {
        body = text;
        EXPECT_FALSE((*provider)({}, {}, {})) << text;
    }
}

TEST_F(CodeOpenAI, BoundsResponseAndTimeout) {
    server_.Post("/v1/chat/completions",
                 [](const auto&, auto& reply) { reply.set_content(std::string(8192, 'x'), "application/json"); });
    start();
    auto provider = openai_provider({.base_url = base_url_, .max_response_bytes = 128});
    ASSERT_TRUE(provider);
    auto result = (*provider)({}, {}, {});
    ASSERT_FALSE(result);
    EXPECT_NE(result.error().message.find("max_response_bytes"), std::string::npos);
    std::stop_source source;
    source.request_stop();
    result = (*provider)({}, {}, source.get_token());
    ASSERT_FALSE(result);
    EXPECT_EQ(result.error().message, "cancelled");
}

TEST_F(CodeOpenAI, TimeoutIsAnError) {
    server_.Post("/v1/chat/completions", [](const auto&, auto& reply) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        reply.set_content(response({{"role", "assistant"}, {"content", "late"}}).dump(), "application/json");
    });
    start();
    auto provider = openai_provider({.base_url = base_url_, .timeout = std::chrono::milliseconds(10)});
    ASSERT_TRUE(provider);
    EXPECT_FALSE((*provider)({}, {}, {}));
}

TEST_F(CodeOpenAI, ChecksDaemonIdentity) {
    server_.Get("/kidi-code/ready", [](const auto&, auto& reply) { reply.set_content("123", "text/plain"); });
    int requests = 0;
    server_.Post("/v1/chat/completions", [&](const auto&, auto& reply) {
        ++requests;
        reply.set_content("{}", "application/json");
    });
    start();
    EXPECT_FALSE(http_completion({.base_url = base_url_}, "{}", {}, 456));
    EXPECT_EQ(requests, 0);
    EXPECT_TRUE(http_completion({.base_url = base_url_}, "{}", {}, 123));
    EXPECT_EQ(requests, 1);
}

TEST_F(CodeOpenAI, CancelsPendingRequest) {
    std::stop_source source;
    server_.Post("/v1/chat/completions", [&](const auto&, auto& reply) {
        source.request_stop();
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        reply.set_content("{}", "application/json");
    });
    start();
    const auto result = http_completion({.base_url = base_url_}, "{}", source.get_token());
    ASSERT_FALSE(result);
    EXPECT_EQ(result.error().message, "cancelled");
}

TEST(CodeProvider, RejectsInvalidConfig) {
    EXPECT_FALSE(openai_provider({.base_url = "file:///etc/passwd"}));
    EXPECT_FALSE(openai_provider({.base_url = "https://user:pass@example.com/v1"}));
    EXPECT_FALSE(openai_provider({.base_url = "https://example.com/v1?query"}));
    EXPECT_FALSE(openai_provider({.base_url = "https://example.com/v1"}));
    EXPECT_FALSE(openai_provider({.base_url = "http://user:pass@localhost:18097/v1"}));
    EXPECT_FALSE(openai_provider({.max_tokens = 0}));
    EXPECT_FALSE(openai_provider({.timeout = std::chrono::milliseconds(0)}));
}

TEST(CodeProvider, RejectsInvalidUtf8Input) {
    auto provider = openai_provider();
    ASSERT_TRUE(provider);
    std::vector<Message> history{{
        .role = Role::ASSISTANT,
        .tool_calls = {{"x", "test", {{"bad", std::string("\xff", 1)}}}},
    }};
    EXPECT_NO_THROW({
        const auto result = (*provider)(history, {}, {});
        ASSERT_FALSE(result);
        EXPECT_EQ(result.error().code, ErrorCode::INVALID_ARGUMENT);
    });
}

} // namespace
} // namespace kidi::code
