#pragma once

#include "kidi/tensor/backend.h"

#include <filesystem>
#include <string>
#include <vector>

namespace kidi::cli {

/// How to start this CLI in a new process; `kidi agent` launches its background server this way.
/// Hosts such as the Python package pass their interpreter command; empty fields mean this executable.
struct Launcher {
    std::vector<std::string> command;
    std::filesystem::path binary; // file whose identity keys reusable background servers
};

auto auto_inference_device(std::span<const tensor::BackendInfo> backends,
                           tensor::TransformerSupport required = tensor::TransformerSupport::QUANTIZED,
                           bool allow_metal = true) -> tensor::Device;
auto main(int argc, const char* const argv[], const Launcher& launcher = {}) -> int;
} // namespace kidi::cli
