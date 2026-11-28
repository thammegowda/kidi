#include "kidi/cli/serve.h"

#include "kidi/cli/log.h"
#include "kidi/server/routes.h"
#include <atomic>
#include <csignal>

namespace kidi::cli {
namespace {
std::atomic<liteapp::HttpServer*> active_server{nullptr};
} // namespace

auto serve(inference::Generator& generator, const ServeOptions& options) -> int {
    liteapp::ServerOptions http;
    http.host = options.host;
    http.port = options.port;
    // Agent transcripts with tool output grow large; this bounds one request body, not the model context.
    http.max_body_bytes = 64U * 1024U * 1024U;
    auto created = liteapp::HttpServer::create(std::move(http));
    if (!created) {
        log::error("cannot create server: {}", created.error().message);
        return 1;
    }
    auto& server = *created;
    server::ChatEngine engine(generator);
    server::mount_chat_routes(server, engine, {.model_name = options.model_name,
                                                 .defaults = options.generation,
                                                 .vision = generator.vision_supported() && !generator.speculative()});

    active_server = &server;
    std::signal(SIGINT, [](int) {
        if (auto* running = active_server.load()) running->stop();
    });
    std::signal(SIGTERM, [](int) {
        if (auto* running = active_server.load()) running->stop();
    });
    log::info("serving {} on http://{}:{}/v1", options.model_name, options.host, options.port);
    const auto listened = server.listen();
    active_server = nullptr;
    if (!listened) {
        log::error("cannot listen on {}:{}: {}", options.host, options.port, listened.error().message);
        return 1;
    }
    return 0;
}

} // namespace kidi::cli
