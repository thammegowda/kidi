#include "liteapp/http/server.h"
#include "liteapp/core/activity_gate.h"
#include "liteapp/service/router.h"

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <optional>
#include <string>

#include <gtest/gtest.h>

namespace {

TEST(ActivityGateTest, RejectsDuplicates) {
    liteapp::ActivityGate requests;
    ASSERT_TRUE(requests.acquire("chat")) << "chat request acquires";
    ASSERT_TRUE(requests.acquire("tts")) << "TTS overlaps chat";
    ASSERT_FALSE(requests.acquire("chat")) << "duplicate chat is rejected";
    ASSERT_TRUE(requests.acquire("asr")) << "ASR overlaps distinct model work";
    requests.release("asr");
    requests.release("tts");
    requests.release("chat");
    ASSERT_TRUE(requests.acquire("asr")) << "ASR acquires when idle";
    ASSERT_FALSE(requests.acquire("asr")) << "duplicate ASR is rejected";
    requests.release("asr");
}

class WebSocketTest : public ::testing::Test {
protected:
    void SetUp() override {
        auto created = liteapp::HttpServer::create({
            .host = "127.0.0.1",
            .port = 0,
            .max_body_bytes = 1024,
            .local_access = liteapp::LocalAccess{.token = "websocket-test", .cookie_name = "kidi_test_access"},
        });
        ASSERT_TRUE(created) << created.error().message;
        server.emplace(std::move(*created));
        services.register_method("settings.system_prompt.set",
                                 [](liteapp::ServiceContext& context) -> liteapp::Result<nlohmann::json> {
                                     static_cast<void>(context.emit("settings.validated"));
                                     return nlohmann::json{{"prompt", context.payload().value("prompt", "")}};
                                 });
        server->websocket("/services",
                          [this](const liteapp::Request&, liteapp::WebSocket& socket) { services.run(socket); });
        server->websocket("/ws",
                          [](const liteapp::Request&, liteapp::WebSocket& socket) {
                              const auto message = socket.read();
                              if (!message || !*message) return;
                              if ((**message).kind == liteapp::WebSocketMessageKind::text)
                                  static_cast<void>(socket.send_text("echo:" + (**message).data));
                              else {
                                  const auto bytes = std::as_bytes(std::span((**message).data));
                                  static_cast<void>(socket.send_binary(bytes));
                              }
                          },
                          {.max_text_bytes = 64, .max_binary_bytes = 4});
        const auto bound = server->bind();
        ASSERT_TRUE(bound) << bound.error().message;
        port = *bound;
        auto started = server->start();
        ASSERT_TRUE(started) << started.error().message;
        running.emplace(std::move(*started));

        httplib::Client http{"127.0.0.1", port};
        http.set_follow_location(false);
        const auto launch = http.Get("/?token=websocket-test");
        ASSERT_TRUE(launch) << "launch exchanges capability";
        ASSERT_EQ(launch->status, 303) << "launch exchanges capability";
        const auto cookie =
            launch->get_header_value("Set-Cookie").substr(0, std::string{"kidi_test_access=websocket-test"}.size());
        const auto origin = "http://127.0.0.1:" + std::to_string(port);
        headers = {{"Cookie", cookie}, {"Origin", origin}};
    }

    auto url(std::string_view path) const -> std::string {
        return "ws://127.0.0.1:" + std::to_string(port) + std::string(path);
    }

    // Declaration order matters: the running server stops before the router its handlers reference is destroyed.
    liteapp::ServiceRouter services;
    std::optional<liteapp::HttpServer> server;
    std::optional<liteapp::RunningHttpServer> running;
    int port = 0;
    httplib::Headers headers;
};

TEST_F(WebSocketTest, RejectsUnauthenticated) {
    httplib::ws::WebSocketClient rejected{url("/ws")};
    EXPECT_FALSE(rejected.connect()) << "unauthenticated WebSocket is rejected";
}

TEST_F(WebSocketTest, EchoesText) {
    httplib::ws::WebSocketClient accepted{url("/ws"), headers};
    accepted.set_read_timeout(5);
    ASSERT_TRUE(accepted.connect()) << "authenticated WebSocket connects";
    ASSERT_TRUE(accepted.send("hello")) << "authenticated WebSocket sends";
    std::string response;
    ASSERT_EQ(accepted.read(response), httplib::ws::ReadResult::Text) << "WebSocket receives text";
    EXPECT_EQ(response, "echo:hello") << "WebSocket handler echoes payload";
    accepted.close();
}

TEST_F(WebSocketTest, EchoesBinary) {
    httplib::ws::WebSocketClient binary{url("/ws"), headers};
    binary.set_read_timeout(5);
    const std::string payload{"\0pc\1", 4};
    ASSERT_TRUE(binary.connect()) << "second authenticated WebSocket connects";
    ASSERT_TRUE(binary.send(payload.data(), payload.size())) << "WebSocket sends binary";
    std::string response;
    ASSERT_EQ(binary.read(response), httplib::ws::ReadResult::Binary) << "WebSocket receives binary";
    EXPECT_EQ(response, payload) << "WebSocket handler preserves binary payload";
    binary.close();
}

TEST_F(WebSocketTest, RouterProgress) {
    httplib::ws::WebSocketClient service{url("/services"), headers};
    service.set_read_timeout(5);
    const auto request = nlohmann::json{
        {"version", 1},
        {"kind", "request"},
        {"id", "prompt"},
        {"method", "settings.system_prompt.set"},
        {"payload", {{"prompt", "Be concise"}}},
    };
    ASSERT_TRUE(service.connect()) << "service WebSocket connects";
    ASSERT_TRUE(service.send(request.dump())) << "service request sends";
    std::string response;
    ASSERT_EQ(service.read(response), httplib::ws::ReadResult::Text) << "service progress receives";
    auto event = nlohmann::json::parse(response);
    EXPECT_EQ(event["kind"], "event") << "service router emits ordered progress";
    EXPECT_EQ(event["type"], "settings.validated") << "service router emits ordered progress";
    EXPECT_EQ(event["sequence"], 0) << "service router emits ordered progress";
    response.clear();
    ASSERT_EQ(service.read(response), httplib::ws::ReadResult::Text) << "service completion receives";
    event = nlohmann::json::parse(response);
    EXPECT_EQ(event["type"], "completed") << "service router returns completion payload";
    EXPECT_EQ(event["sequence"], 1) << "service router returns completion payload";
    EXPECT_EQ(event["payload"]["prompt"], "Be concise") << "service router returns completion payload";
    service.close();
}

TEST_F(WebSocketTest, OversizedCloses) {
    httplib::ws::WebSocketClient oversized{url("/ws"), headers};
    oversized.set_read_timeout(5);
    ASSERT_TRUE(oversized.connect()) << "oversized-message test connects";
    ASSERT_TRUE(oversized.send(std::string(65, 'x'))) << "oversized-message test sends";
    std::string response;
    EXPECT_EQ(oversized.read(response), httplib::ws::ReadResult::Fail) << "oversized message closes the WebSocket";
}

} // namespace
