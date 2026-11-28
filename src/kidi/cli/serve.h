#pragma once

#include "kidi/inference/generator.h"
#include <string>

namespace kidi::cli {

struct ServeOptions {
    std::string host = "127.0.0.1";
    int port = 8000;
    std::string model_name;
    inference::GenerationOptions generation;
};

/// Hosts kidi's shared HTTP API (`kidi/server/routes.h`) headlessly until the process is stopped.
/// `kidi agent` mounts the same routes in its background server.
auto serve(inference::Generator& generator, const ServeOptions& options) -> int;

} // namespace kidi::cli
