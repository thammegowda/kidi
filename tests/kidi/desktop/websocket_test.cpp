#include "liteapp/http/server.h"
#include "liteapp/core/activity_gate.h"
#include "liteapp/service/router.h"

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <iostream>
#include <string>

namespace {
auto expect(bool condition, const char* message) -> bool {
    if (!condition) std::cerr << "FAILED: " << message << '\n';
    return condition;
}
} // namespace

auto main() -> int {
    liteapp::ActivityGate requests;
    if (!expect(requests.acquire("chat"), "chat request acquires") ||
        !expect(requests.acquire("tts"), "TTS overlaps chat") ||
        !expect(!requests.acquire("chat"), "duplicate chat is rejected") ||
        !expect(requests.acquire("asr"), "ASR overlaps distinct model work"))
        return 1;
    requests.release("asr");
    requests.release("tts");
    requests.release("chat");
    if (!expect(requests.acquire("asr"), "ASR acquires when idle") ||
        !expect(!requests.acquire("asr"), "duplicate ASR is rejected"))
        return 1;
    requests.release("asr");

    auto created = liteapp::HttpServer::create({
        .host = "127.0.0.1",
        .port = 0,
        .max_body_bytes = 1024,
        .local_access = liteapp::LocalAccess{.token = "websocket-test", .cookie_name = "kidi_test_access"},
    });
    if (!created) {
        std::cerr << created.error().message << '\n';
        return 1;
    }
    auto server = std::move(*created);
    liteapp::ServiceRouter services;
    services.register_method("settings.system_prompt.set",
                             [](liteapp::ServiceContext& context) -> liteapp::Result<nlohmann::json> {
                                 static_cast<void>(context.emit("settings.validated"));
                                 return nlohmann::json{{"prompt", context.payload().value("prompt", "")}};
                             });
    server.websocket("/services",
                     [&services](const liteapp::Request&, liteapp::WebSocket& socket) { services.run(socket); });
    server.websocket("/ws",
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
    const auto port = server.bind();
    if (!port) {
        std::cerr << port.error().message << '\n';
        return 1;
    }
    auto running = server.start();
    if (!running) {
        std::cerr << running.error().message << '\n';
        return 1;
    }

    httplib::Client http{"127.0.0.1", *port};
    http.set_follow_location(false);
    const auto launch = http.Get("/?token=websocket-test");
    if (!expect(static_cast<bool>(launch) && launch->status == 303, "launch exchanges capability")) return 1;
    const auto cookie =
        launch->get_header_value("Set-Cookie").substr(0, std::string{"kidi_test_access=websocket-test"}.size());
    const auto origin = "http://127.0.0.1:" + std::to_string(*port);

    httplib::ws::WebSocketClient rejected{"ws://127.0.0.1:" + std::to_string(*port) + "/ws"};
    if (!expect(!rejected.connect(), "unauthenticated WebSocket is rejected")) return 1;

    const httplib::Headers headers{{"Cookie", cookie}, {"Origin", origin}};
    httplib::ws::WebSocketClient accepted{"ws://127.0.0.1:" + std::to_string(*port) + "/ws", headers};
    accepted.set_read_timeout(5);
    if (!expect(accepted.connect(), "authenticated WebSocket connects") ||
        !expect(accepted.send("hello"), "authenticated WebSocket sends"))
        return 1;
    std::string response;
    if (!expect(accepted.read(response) == httplib::ws::ReadResult::Text, "WebSocket receives text") ||
        !expect(response == "echo:hello", "WebSocket handler echoes payload"))
        return 1;
    accepted.close();

    httplib::ws::WebSocketClient binary{"ws://127.0.0.1:" + std::to_string(*port) + "/ws", headers};
    binary.set_read_timeout(5);
    const std::string payload{"\0pc\1", 4};
    if (!expect(binary.connect(), "second authenticated WebSocket connects") ||
        !expect(binary.send(payload.data(), payload.size()), "WebSocket sends binary"))
        return 1;
    response.clear();
    if (!expect(binary.read(response) == httplib::ws::ReadResult::Binary, "WebSocket receives binary") ||
        !expect(response == payload, "WebSocket handler preserves binary payload"))
        return 1;
    binary.close();

    httplib::ws::WebSocketClient service{"ws://127.0.0.1:" + std::to_string(*port) + "/services", headers};
    service.set_read_timeout(5);
    const auto request = nlohmann::json{
        {"version", 1},
        {"kind", "request"},
        {"id", "prompt"},
        {"method", "settings.system_prompt.set"},
        {"payload", {{"prompt", "Be concise"}}},
    };
    if (!expect(service.connect(), "service WebSocket connects") ||
        !expect(service.send(request.dump()), "service request sends"))
        return 1;
    response.clear();
    if (!expect(service.read(response) == httplib::ws::ReadResult::Text, "service progress receives")) return 1;
    auto event = nlohmann::json::parse(response);
    if (!expect(event["kind"] == "event" && event["type"] == "settings.validated" && event["sequence"] == 0,
                "service router emits ordered progress"))
        return 1;
    response.clear();
    if (!expect(service.read(response) == httplib::ws::ReadResult::Text, "service completion receives")) return 1;
    event = nlohmann::json::parse(response);
    if (!expect(event["type"] == "completed" && event["sequence"] == 1 && event["payload"]["prompt"] == "Be concise",
                "service router returns completion payload"))
        return 1;
    service.close();

    httplib::ws::WebSocketClient oversized{"ws://127.0.0.1:" + std::to_string(*port) + "/ws", headers};
    oversized.set_read_timeout(5);
    if (!expect(oversized.connect(), "oversized-message test connects") ||
        !expect(oversized.send(std::string(65, 'x')), "oversized-message test sends"))
        return 1;
    response.clear();
    if (!expect(oversized.read(response) == httplib::ws::ReadResult::Fail, "oversized message closes the WebSocket"))
        return 1;
    return 0;
}
