#include "kidi/agent/chat_client.h"
#include "kidi/agent/chat_transport.h"
#include "kidi/agent/workspace.h"

#include <gtest/gtest.h>
#include <httplib.h>

#include <atomic>
#include <thread>

namespace kidi::agent {
namespace {

TEST(AgentProvider, HeadlessCompletionsStillStream) {
    // Without a UI callback the request must still stream, or slow CPU prefill gets no keep-alives.
    auto provider = completion_provider(
        {.model = "test-model"},
        [](const ChatClientOptions& options, std::string payload, std::stop_token, std::string_view) -> Result<std::string> {
            EXPECT_TRUE(options.on_delta);
            EXPECT_EQ(Json::parse(payload)["stream"], true);
            return R"({"choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]})";
        });
    ASSERT_TRUE(provider);
    const auto reply = (*provider)({}, {}, {});
    ASSERT_TRUE(reply) << reply.error().message;
    EXPECT_EQ(reply->content, "ok");
}

TEST(AgentProvider, ContextAndSummaryRequests) {
    int completions = 0, inspections = 0;
    auto provider = completion_provider(
        {.model = "test-model",
         .temperature = 0.7F,
         .top_k = 20,
         .on_delta = [](auto, auto) { ADD_FAILURE() << "summary/context text must not reach the UI"; }},
        [&](const ChatClientOptions&, std::string payload, std::stop_token, std::string_view route) -> Result<std::string> {
            const auto request = Json::parse(payload);
            // Summaries stream silently so server keep-alives cover long prefills.
            EXPECT_EQ(request["stream"], route == "/chat/completions");
            if (request.contains("reasoning_effort")) {
                EXPECT_EQ(request["temperature"], 0) << "summaries are greedy";
                EXPECT_FALSE(request.contains("top_k"));
            }
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

TEST(AgentProvider, UsageIsMeasuredNotEstimated) {
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
    EXPECT_EQ(reply->usage->decode_ms, 0.0);
    body["kidi_stats"] = {{"prefill_ms", 250.5}, {"decode_ms", 90.0}};
    reply = (*provider)({}, {}, {});
    ASSERT_TRUE(reply && reply->usage);
    EXPECT_EQ(reply->usage->prefill_ms, 250.5);
    EXPECT_EQ(reply->usage->decode_ms, 90.0);
    body.erase("kidi_stats");
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

TEST(AgentProvider, BadContextAndSummary) {
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
    const auto answer = (*provider)({}, {}, {});
    ASSERT_FALSE(answer);
    EXPECT_NE(answer.error().message.find("4096-token limit"), std::string::npos) << answer.error().message;
    EXPECT_NE(answer.error().message.find("--max-tokens"), std::string::npos);
    content.clear();
    const auto empty = provider->summarize({}, 200, {});
    ASSERT_FALSE(empty);
    EXPECT_NE(empty.error().message.find("200-token limit"), std::string::npos) << empty.error().message;
    EXPECT_EQ(empty.error().message.find("--max-tokens"), std::string::npos);
}

TEST(AgentProvider, SendsReasoningEffortExceptForSummaries) {
    std::vector<Json> requests;
    auto provider = completion_provider(
        {.model = "test", .reasoning_effort = "medium"},
        [&](const ChatClientOptions&, std::string payload, std::stop_token,
            std::string_view route) -> Result<std::string> {
            requests.push_back(Json::parse(payload));
            if (route == "/chat/context") return std::string(R"({"prompt_tokens":1,"context_tokens":4096})");
            return response({{"role", "assistant"}, {"content", "ok"}}).dump();
        });
    ASSERT_TRUE(provider);
    ASSERT_TRUE((*provider)({}, {}, {}));
    ASSERT_TRUE(provider->context({}, {}, {}));
    ASSERT_TRUE(provider->summarize({}, 200, {}));
    ASSERT_EQ(requests.size(), 3);
    EXPECT_EQ(requests[0]["reasoning_effort"], "medium");
    EXPECT_EQ(requests[1]["reasoning_effort"], "medium") << "context counts must render the same prompt as replies";
    EXPECT_EQ(requests[2]["reasoning_effort"], "none");
}

class ChatClientTest : public testing::Test {
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

TEST_F(ChatClientTest, StreamsBeforeCompletionAndReassemblesTools) {
    std::atomic<bool> observed = false;
    std::string answer, reasoning;
    server_.Post("/v1/chat/completions", [&](const auto& request, auto& reply) {
        const auto body = Json::parse(request.body);
        EXPECT_EQ(body["stream"], true);
        EXPECT_EQ(body["stream_options"]["include_usage"], true);
        EXPECT_FLOAT_EQ(body["temperature"].template get<float>(), 0.5F);
        EXPECT_FLOAT_EQ(body["top_p"].template get<float>(), 0.9F);
        EXPECT_FALSE(body.contains("top_k")) << "top_k 0 is omitted";
        reply.set_chunked_content_provider("text/event-stream", [&](std::size_t, httplib::DataSink& sink) {
            const auto send = [&](Json delta, Json finish = nullptr) {
                const auto event = "data: " + Json{{"choices", Json::array(
                    {{{"index", 0}, {"delta", std::move(delta)}, {"finish_reason", finish}}})}}.dump() + "\r\n\r\n";
                // Deliberately split every framing, JSON and UTF-8 boundary.
                for (const auto byte : event)
                    if (!sink.write(&byte, 1)) return false;
                return true;
            };
            if (!sink.write(": heartbeat\r\n\r\n", 15)) return false;
            if (!send({{"role", "assistant"}, {"reasoning_content", "inspect"}})) return false;
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
            while (!observed && std::chrono::steady_clock::now() < deadline)
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            EXPECT_TRUE(observed) << "text was buffered until completion";
            if (!send({{"content", "caf\xc3\xa9"}})) return false;
            if (!send({{"tool_calls", Json::array({{{"index", 0}, {"id", "call-1"}, {"type", "function"},
                {"function", {{"name", "look"}, {"arguments", "{\"key\":"}}}}})}})) return false;
            if (!send({{"tool_calls", Json::array({{{"index", 0},
                {"function", {{"name", "up"}, {"arguments", "\"value\"}"}}}}})}}, "tool_calls")) return false;
            const auto tail = "data: " + Json{{"choices", Json::array()},
                {"usage", {{"prompt_tokens", 100}, {"completion_tokens", 10}}},
                {"kidi_stats", {{"decode_ms", 50.0}}}}.dump() + "\n\ndata: [DONE]\n\n";
            if (!sink.write(tail.data(), tail.size())) return false;
            sink.done();
            return true;
        });
    });
    start();
    auto provider = http_provider({.base_url = base_url_, .temperature = 0.5F, .top_p = 0.9F, .on_delta = [&](std::string_view text, bool thinking) {
        (thinking ? reasoning : answer) += text;
        observed = true;
    }});
    ASSERT_TRUE(provider);
    auto reply = (*provider)({}, {}, {});
    ASSERT_TRUE(reply) << reply.error().message;
    EXPECT_EQ(reasoning, "inspect");
    EXPECT_EQ(answer, "caf\xc3\xa9");
    EXPECT_EQ(reply->content, answer);
    EXPECT_EQ(reply->reasoning_content, reasoning);
    ASSERT_EQ(reply->tool_calls.size(), 1);
    EXPECT_EQ(reply->tool_calls[0].name, "lookup");
    EXPECT_EQ(reply->tool_calls[0].arguments["key"], "value");
    ASSERT_TRUE(reply->usage);
    EXPECT_EQ(reply->usage->decode_ms, 50.0);
}

TEST_F(ChatClientTest, KeepAlivesExtendSlowPrefillPastTheIdleTimeout) {
    server_.Post("/v1/chat/completions", [&](const auto&, auto& reply) {
        reply.set_chunked_content_provider("text/event-stream", [&](std::size_t, httplib::DataSink& sink) {
            for (int beat = 0; beat < 8; ++beat) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                if (!sink.write(": keep-alive\n\n", 14)) return false;
            }
            const std::string tail =
                "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"done\"},\"finish_reason\":\"stop\"}]}"
                "\n\ndata: [DONE]\n\n";
            if (!sink.write(tail.data(), tail.size())) return false;
            sink.done();
            return true;
        });
    });
    start();
    auto provider = http_provider(
        {.base_url = base_url_, .timeout = std::chrono::milliseconds(400), .on_delta = [](auto, auto) {}});
    ASSERT_TRUE(provider);
    const auto reply = (*provider)({}, {}, {});
    ASSERT_TRUE(reply) << reply.error().message;
    EXPECT_EQ(reply->content, "done");
}

TEST_F(ChatClientTest, RejectsBrokenStreamsAndCancelsDuringStreaming) {
    std::string wire;
    server_.Post("/v1/chat/completions", [&](const auto&, auto& reply) {
        reply.set_content(wire, "text/event-stream");
    });
    start();
    auto provider = http_provider({.base_url = base_url_, .on_delta = [](auto, auto) {}});
    ASSERT_TRUE(provider);
    for (const auto* broken : {
             "data: not-json\n\n",
             "data: {\"error\":{\"message\":\"failed\"}}\n\n",
             "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"partial\"}}]}\n\n",
             "data: [DONE]\n\n",
             "data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":-1}]}}]}\n\n"}) {
        wire = broken;
        const auto result = (*provider)({}, {}, {});
        EXPECT_FALSE(result) << wire;
    }
    wire = "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"partial\"}}]}\n\n";
    std::stop_source stop;
    provider = http_provider({.base_url = base_url_, .on_delta = [&](auto, auto) { stop.request_stop(); }});
    ASSERT_TRUE(provider);
    const auto cancelled = (*provider)({}, {}, stop.get_token());
    ASSERT_FALSE(cancelled);
    EXPECT_EQ(cancelled.error().message, "cancelled");
}

TEST_F(ChatClientTest, HttpToolRoundTrip) {
    int requests = 0;
    server_.Post("/v1/chat/completions", [&](const httplib::Request& request, httplib::Response& reply) {
        EXPECT_FALSE(request.has_header("Authorization"));
        const auto body = Json::parse(request.body);
        EXPECT_EQ(body["model"], "test-model");
        // Completions always stream; this server answers with plain JSON, which the client must still accept.
        EXPECT_EQ(body["stream"], true);
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
    auto provider = http_provider({.base_url = base_url_ + "/", .model = "test-model"});
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

TEST_F(ChatClientTest, HttpErrorsNoRetry) {
    int requests = 0;
    server_.Post("/v1/chat/completions", [&](const auto&, auto& reply) {
        ++requests;
        reply.status = 429;
        reply.set_content(R"({"error":{"message":"rate limit"}})", "application/json");
    });
    start();
    auto provider = http_provider({.base_url = base_url_});
    ASSERT_TRUE(provider);
    auto result = (*provider)({}, {}, {});
    ASSERT_FALSE(result);
    EXPECT_NE(result.error().message.find("429"), std::string::npos);
    EXPECT_EQ(requests, 1);
}

TEST_F(ChatClientTest, RefusesRedirect) {
    int redirects = 0;
    server_.Post("/v1/chat/completions", [](const auto&, auto& reply) { reply.set_redirect("/stolen", 307); });
    server_.Post("/stolen", [&](const auto&, auto&) { ++redirects; });
    start();
    auto provider = http_provider({.base_url = base_url_});
    ASSERT_TRUE(provider);
    EXPECT_FALSE((*provider)({}, {}, {}));
    EXPECT_EQ(redirects, 0);
}

TEST_F(ChatClientTest, RejectsBadAndPartialReplies) {
    std::string body;
    server_.Post("/v1/chat/completions",
                 [&](const auto&, auto& reply) { reply.set_content(body, "application/json"); });
    start();
    auto provider = http_provider({.base_url = base_url_});
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

TEST_F(ChatClientTest, BoundsResponseAndTimeout) {
    server_.Post("/v1/chat/completions",
                 [](const auto&, auto& reply) { reply.set_content(std::string(8192, 'x'), "application/json"); });
    start();
    auto provider = http_provider({.base_url = base_url_, .max_response_bytes = 128});
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

TEST_F(ChatClientTest, TimeoutIsAnError) {
    server_.Post("/v1/chat/completions", [](const auto&, auto& reply) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        reply.set_content(response({{"role", "assistant"}, {"content", "late"}}).dump(), "application/json");
    });
    start();
    auto provider = http_provider({.base_url = base_url_, .timeout = std::chrono::milliseconds(10)});
    ASSERT_TRUE(provider);
    EXPECT_FALSE((*provider)({}, {}, {}));
}

TEST_F(ChatClientTest, ChecksDaemonIdentity) {
    server_.Get("/kidi-agent/ready", [](const auto&, auto& reply) { reply.set_content("123", "text/plain"); });
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

TEST_F(ChatClientTest, CancelsPendingRequest) {
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

TEST(AgentProvider, RejectsInvalidConfig) {
    EXPECT_FALSE(http_provider({.base_url = "file:///etc/passwd"}));
    EXPECT_FALSE(http_provider({.base_url = "https://user:pass@example.com/v1"}));
    EXPECT_FALSE(http_provider({.base_url = "https://example.com/v1?query"}));
    EXPECT_FALSE(http_provider({.base_url = "https://example.com/v1"}));
    EXPECT_FALSE(http_provider({.base_url = "http://user:pass@localhost:18097/v1"}));
    EXPECT_FALSE(http_provider({.max_tokens = 0}));
    EXPECT_FALSE(http_provider({.timeout = std::chrono::milliseconds(0)}));
}

TEST(AgentProvider, RejectsInvalidUtf8Input) {
    auto provider = http_provider();
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
} // namespace kidi::agent
