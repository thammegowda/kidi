#pragma once

#include "kidi/agent/command.h"
#include "kidi/cli/main.h"
#include "kidi/cli/serve.h"

namespace kidi::cli {

auto configure_managed_agent_parser(ArgumentParser& parser) -> void;
auto configure_managed_serve_parser(ArgumentParser& parser) -> void;
auto managed_agent_command(const Namespace& arguments, const Launcher& launcher) -> int;
auto protect_managed_lock(const Namespace& arguments) -> void;
/// Serves the background daemon shared by `kidi agent` clients until it is idle. Agent sessions grow long, so
/// KV caches are allocated per request, the last prompt state is kept for prefix reuse, and the context fills
/// free device memory unless `options.generation.context_size` is set.
auto serve_managed(inference::Generator& generator, tensor::Device device, ServeOptions options,
                   inference::ServingOptions serving, const Namespace& arguments) -> int;

} // namespace kidi::cli
