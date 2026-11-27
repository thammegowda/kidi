#include "liteapp/http/server.h"

#include <httplib.h>

#include <thread>

#include <gtest/gtest.h>

namespace {

TEST(HttpTest, StreamsResponseBodyAndHeaders) {
    auto created = liteapp::HttpServer::create();
    ASSERT_TRUE(created) << created.error().message;
    auto server = std::move(*created);
    server.get("/events", [](const liteapp::Request&) {
        auto response = liteapp::Response::streamed("text/event-stream", [](const auto& write) {
            return write("data: first\n\n") && write("data: [DONE]\n\n");
        });
        return response;
    });

    const auto bound = server.bind();
    ASSERT_TRUE(bound) << bound.error().message;
    std::jthread server_thread{[&server] { static_cast<void>(server.listen_after_bind()); }};
    server.wait_until_ready();

    httplib::Client client{"127.0.0.1", *bound};
    const auto response = client.Get("/events");
    server.stop();
    ASSERT_TRUE(response) << "HTTP streaming response did not preserve its body or headers: no response";
    EXPECT_EQ(response->status, 200) << "HTTP streaming response did not preserve its body or headers";
    EXPECT_EQ(response->get_header_value("Content-Type"), "text/event-stream")
        << "HTTP streaming response did not preserve its body or headers";
    EXPECT_EQ(response->get_header_value("Cache-Control"), "no-store")
        << "HTTP streaming response did not preserve its body or headers";
    EXPECT_EQ(response->body, "data: first\n\ndata: [DONE]\n\n")
        << "HTTP streaming response did not preserve its body or headers";
}

} // namespace
