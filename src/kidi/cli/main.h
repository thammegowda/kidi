#pragma once

#include "kidi/tensor/backend.h"

namespace kidi::cli {

auto auto_inference_device(std::span<const tensor::BackendInfo> backends,
                           tensor::TransformerSupport required = tensor::TransformerSupport::QUANTIZED,
                           bool allow_metal = true) -> tensor::Device;
auto main(int argc, const char* const argv[]) -> int;
} // namespace kidi::cli